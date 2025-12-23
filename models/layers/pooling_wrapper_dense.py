import torch.nn as nn
from models.layers.pooling_dense import MAC, SPoC, GeM, NetVLADWrapper


class PoolingWrapper(nn.Module):
    """
    Dense版本的PoolingWrapper (修复版)
    功能：
    1. 支持 MAC, SPoC, GeM, NetVLAD
    2. [新增] 自动维度对齐：如果 backbone 输出维度 (in_dim) 与 目标维度 (output_dim) 不一致，
       且不是 NetVLAD (它自己会处理)，则自动添加一个线性层 (Linear) 进行投影。
    """

    def __init__(self, pool_method, in_dim, output_dim):
        super().__init__()

        self.pool_method = pool_method
        self.in_dim = in_dim
        self.output_dim = output_dim

        # 初始化投影层为 None
        self.projector = None

        # 1. 初始化池化层
        if pool_method == 'MAC':
            self.pooling = MAC(input_dim=in_dim)
        elif pool_method == 'SPoC':
            self.pooling = SPoC(input_dim=in_dim)
        elif pool_method == 'GeM':
            # 删除旧代码中的 assert in_dim == output_dim
            self.pooling = GeM(input_dim=in_dim)
        elif self.pool_method == 'netvlad':
            # NetVLAD 内部通常已经处理了维度或输出特定维度，不需要外部投影
            self.pooling = NetVLADWrapper(feature_size=in_dim, output_dim=output_dim, gating=False)
        elif self.pool_method == 'netvladgc':
            self.pooling = NetVLADWrapper(feature_size=in_dim, output_dim=output_dim, gating=True)
        else:
            raise NotImplementedError('Unknown pooling method: {}'.format(pool_method))

        # 2. [关键修复] 如果不是 NetVLAD 且维度不匹配，添加线性投影层
        # NetVLADWrapper 已经在内部处理了 output_dim，所以跳过
        if 'netvlad' not in self.pool_method and in_dim != output_dim:
            print(f"PoolingWrapper: Adding Linear Projection {in_dim} -> {output_dim}")
            self.projector = nn.Sequential(
                nn.Linear(in_dim, output_dim, bias=True)
            )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) dense tensor
        Returns:
            (B, output_dim) tensor
        """
        # 1. 执行池化 (B, C, H, W) -> (B, in_dim)
        x = self.pooling(x)

        # 2. 如果需要，执行维度投影 (B, in_dim) -> (B, output_dim)
        if self.projector is not None:
            x = self.projector(x)

        return x