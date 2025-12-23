import numpy as np
from abc import ABC, abstractmethod
import torch


class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc):
        pass


class BEVQuantizer(Quantizer):
    def __init__(self,
                 coords_range=[-10., -10, -4, 10, 10, 8],
                 div_n=[256, 256, 32]):
        """
        BEV 量化器（Dense版本 - 输出标准2D tensor）
        Args:
            coords_range: 点云裁剪范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            div_n: 网格划分数 [nx, ny, nz]
        """
        super().__init__()
        self.coords_range = torch.tensor(coords_range, dtype=torch.float)
        self.div_n = torch.tensor(div_n, dtype=torch.int32)
        self.steps = (self.coords_range[3:] - self.coords_range[:3]) / self.div_n

        print(f"BEVQuantizer (Dense) Initialized:")
        print(f"  Range: {self.coords_range.tolist()}")
        print(f"  Grid: {self.div_n.tolist()}")
        print(f"  Steps: {self.steps.tolist()}")
        print(f"  Output shape: (batch, {self.div_n[2]}, {self.div_n[1]}, {self.div_n[0]})")

    def __call__(self, pc):
        """
        Args:
            pc: (N, 3) Tensor, Cartesian coordinates (X, Y, Z)
        Returns:
            dense_bev: (C, H, W) Tensor, C=32 channels, H=W=256
        """
        device = pc.device
        coords_range = self.coords_range.to(device)
        steps = self.steps.to(device)
        div_n = self.div_n.to(device)

        # 过滤范围外的点
        mask = (pc[:, 0] >= coords_range[0]) & (pc[:, 0] < coords_range[3]) & \
               (pc[:, 1] >= coords_range[1]) & (pc[:, 1] < coords_range[4]) & \
               (pc[:, 2] >= coords_range[2]) & (pc[:, 2] < coords_range[5])
        pc = pc[mask]

        # 创建dense grid: (C, H, W) = (32, 256, 256)
        dense_bev = torch.zeros((div_n[2].item(), div_n[1].item(), div_n[0].item()),
                                dtype=torch.float32, device=device)

        if pc.shape[0] == 0:
            return dense_bev

        # 计算网格索引
        indices = ((pc - coords_range[:3]) / steps).long()
        indices = torch.clamp(indices, min=torch.zeros(3, dtype=torch.long, device=device),
                              max=(div_n - 1))

        x_indices = indices[:, 0]
        y_indices = indices[:, 1]
        z_indices = indices[:, 2]

        # 填充occupancy: dense_bev[z, y, x] = 1.0
        # 使用index_put_批量填充（比循环快）
        dense_bev.index_put_((z_indices, y_indices, x_indices),
                             torch.ones(pc.shape[0], device=device, dtype=torch.float32),
                             accumulate=False)  # 同一位置多个点只标记一次

        return dense_bev