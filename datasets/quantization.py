import numpy as np
from typing import List
from abc import ABC, abstractmethod
import torch
import MinkowskiEngine as ME


class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc):
        pass


class PolarQuantizer(Quantizer):
    def __init__(self, quant_step: List[float]):
        assert len(
            quant_step) == 3, '3 quantization steps expected: for sector (in degrees), ring and z-coordinate (in meters)'
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)
        self.theta_range = int(360. // self.quant_step[0])
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)

    def __call__(self, pc):
        # Convert to polar coordinates and quantize with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3

        # theta is an angle in degrees in 0..360 range
        theta = 180. + torch.atan2(pc[:, 1], pc[:, 0]) * 180. / np.pi
        # dist is a distance from a coordinate origin
        dist = torch.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2)
        z = pc[:, 2]
        polar_pc = torch.stack([theta, dist, z], dim=1)
        # Scale each coordinate so after quantization with step 1. we got the required quantization step in each dim
        polar_pc = polar_pc / self.quant_step
        quantized_polar_pc, ndx = ME.utils.sparse_quantize(polar_pc, quantization_size=1., return_index=True)
        # Return quantized coordinates and indices of selected elements
        return quantized_polar_pc, ndx


class CartesianQuantizer(Quantizer):
    def __init__(self, quant_step: float):
        self.quant_step = quant_step

    def __call__(self, pc):
        # Converts to polar coordinates and quantizes with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3
        quantized_pc, ndx = ME.utils.sparse_quantize(pc, quantization_size=self.quant_step, return_index=True)
        # Return quantized coordinates and index of selected elements
        return quantized_pc, ndx


# ==============================================================================
# 新增: BEVQuantizer (参考 BEVNet 配置)
# ==============================================================================
class BEVQuantizer(Quantizer):
    def __init__(self,
                 coords_range=[-10., -10, -4, 10, 10, 8],  # Xmin, Ymin, Zmin, Xmax, Ymax, Zmax
                 div_n=[256, 256, 32]):  # Nx, Ny, Nz (Nz即为特征通道数)
        """
        初始化 BEV 量化器
        Args:
            coords_range: 点云裁剪范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            div_n: 网格划分数 [nx, ny, nz]
        """
        super().__init__()
        self.coords_range = torch.tensor(coords_range, dtype=torch.float)
        self.div_n = torch.tensor(div_n, dtype=torch.int32)

        # 计算每个格子的物理尺寸 (Step Size)
        # steps = (max - min) / div
        self.steps = (self.coords_range[3:] - self.coords_range[:3]) / self.div_n

        print(f"BEVQuantizer Initialized:")
        print(f"  Range: {self.coords_range.tolist()}")
        print(f"  Grid: {self.div_n.tolist()}")
        print(f"  Steps: {self.steps.tolist()}")

    def __call__(self, pc):
        """
        Args:
            pc: (N, 3) Tensor, Cartesian coordinates (X, Y, Z)
        Returns:
            unique_xy: (M, 2) Tensor, 唯一的2D坐标 (X_idx, Y_idx)
            features: (M, 32) Tensor, 每个2D坐标对应的Z轴Occupancy特征
        """
        # 确保输入在 CPU 或 GPU
        device = pc.device
        coords_range = self.coords_range.to(device)
        steps = self.steps.to(device)
        div_n = self.div_n.to(device)

        # 1. 过滤掉范围外的点 (Hard Clipping)
        mask = (pc[:, 0] >= coords_range[0]) & (pc[:, 0] < coords_range[3]) & \
               (pc[:, 1] >= coords_range[1]) & (pc[:, 1] < coords_range[4]) & \
               (pc[:, 2] >= coords_range[2]) & (pc[:, 2] < coords_range[5])
        pc = pc[mask]

        if pc.shape[0] == 0:
            # 如果没有点，返回空
            return torch.zeros((0, 2), dtype=torch.int32, device=device), \
                torch.zeros((0, div_n[2]), dtype=torch.float32, device=device)

        # 2. 计算网格索引: (point - min) / step
        # indices shape: (N, 3) -> [x_idx, y_idx, z_idx]
        indices = ((pc - coords_range[:3]) / steps).long()

        # 钳制索引范围，防止精度问题导致的越界
        indices = torch.clamp(indices, min=torch.zeros(3, dtype=torch.long, device=device),
                              max=(div_n - 1))

        # 3. 分离 XY 平面索引和 Z 轴索引
        xy_indices = indices[:, :2]  # (N, 2)
        z_indices = indices[:, 2]  # (N,)

        # 4. 在 XY 平面上进行去重，找到唯一的柱子 (Pillars)
        # unique_xy: (M, 2) - M 个唯一的柱子坐标
        # inverse_indices: (N,) - 原点云中每个点归属哪个柱子 (0 ~ M-1)
        unique_xy, inverse_indices = torch.unique(xy_indices, dim=0, return_inverse=True)

        # 5. 构建特征 (Z轴 Occupancy 编码)
        # features shape: (M, Nz) -> (M, 32)
        # 如果某个柱子 inverse_indices[i] 在高度 z_indices[i] 有点，则对应位置设为 1
        num_unique = unique_xy.shape[0]
        num_channels = div_n[2]

        features = torch.zeros((num_unique, num_channels), dtype=torch.float32, device=device)

        # 使用 index_put_ 进行散点赋值，所有存在的点位置置 1 (Occupancy Grid)
        # inverse_indices 是行索引 (柱子ID), z_indices 是列索引 (高度层ID)
        features.index_put_((inverse_indices, z_indices), torch.tensor(1.0, device=device))

        return unique_xy.int(), features