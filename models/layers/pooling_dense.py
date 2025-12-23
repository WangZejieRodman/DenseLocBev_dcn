import torch
import torch.nn as nn
import torch.nn.functional as F


class MAC(nn.Module):
    """Global Max Pooling for dense tensors"""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, C) tensor
        """
        return F.adaptive_max_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)


class SPoC(nn.Module):
    """Global Average Pooling for dense tensors"""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, C) tensor
        """
        return F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)


class GeM(nn.Module):
    """Generalized Mean Pooling for dense tensors"""

    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, C) tensor
        """
        # Apply power -> avg pool -> power
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1. / self.p)
        return x.squeeze(-1).squeeze(-1)


class NetVLADWrapper(nn.Module):
    """
    NetVLAD wrapper for dense tensors
    注意：需要将(B, C, H, W)展平为(B, H*W, C)格式
    """

    def __init__(self, feature_size, output_dim, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        # 导入原有的NetVLAD实现
        from models.layers.netvlad import NetVLADLoupe
        self.net_vlad = NetVLADLoupe(feature_size=feature_size, cluster_size=64,
                                     output_dim=output_dim, gating=gating,
                                     add_batch_norm=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, output_dim) tensor
        """
        B, C, H, W = x.shape
        assert C == self.feature_size

        # Reshape: (B, C, H, W) -> (B, H*W, C)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B, H * W, C)  # (B, H*W, C)

        # NetVLAD expects (B, num_points, feature_size)
        x = self.net_vlad(x)
        assert x.shape[0] == B
        assert x.shape[1] == self.output_dim
        return x