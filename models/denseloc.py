import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLoc(torch.nn.Module):
    """Dense版本的场景识别模型（不再依赖MinkowskiEngine）"""

    def __init__(self, backbone: nn.Module, pooling: nn.Module, normalize_embeddings: bool = False):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}

    def forward(self, batch):
        """
        Args:
            batch: dict with 'features' key, shape (B, 32, 256, 256)
        Returns:
            dict with 'global' key, shape (B, output_dim)
        """
        x = batch['features']  # (B, 32, 256, 256)
        x = self.backbone(x)  # (B, 256, 32, 32)

        # x.shape[1] should match pooling.in_dim
        assert x.shape[1] == self.pooling.in_dim, \
            f'Backbone output channels: {x.shape[1]}, Expected: {self.pooling.in_dim}'

        x = self.pooling(x)  # (B, output_dim)

        if hasattr(self.pooling, 'stats'):
            self.stats.update(self.pooling.stats)

        assert x.dim() == 2, f'Expected 2D tensor (B, output_dim), Got {x.dim()}D'
        assert x.shape[1] == self.pooling.output_dim, \
            f'Output channels: {x.shape[1]}, Expected: {self.pooling.output_dim}'

        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)

        return {'global': x}

    def print_info(self):
        print('Model class: DenseLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f'Backbone: {type(self.backbone).__name__} #parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'Pooling method: {self.pooling.pool_method}   #parameters: {n_params}')
        print('# channels from backbone: {}'.format(self.pooling.in_dim))
        print('# output channels: {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')