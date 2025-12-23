import torch.nn as nn
from misc.utils import ModelParams

# 导入dense版本的模块
from models.denseloc import DenseLoc
from models.densebev import DenseBEVBackbone
from models.layers.pooling_wrapper_dense import PoolingWrapper


def model_factory(model_params: ModelParams):
    """
    模型工厂（Dense版本）
    """
    if model_params.model == 'DenseBEV':
        in_channels = getattr(model_params, 'in_channels', 32)

        print(f"Model Factory: Initializing DenseBEV...")
        print(f"  Input Channels (Z-layers): {in_channels}")
        print(f"  Feature Size (Backbone Out): {model_params.feature_size}")

        backbone = DenseBEVBackbone(in_channels=in_channels)

        pooling = PoolingWrapper(pool_method=model_params.pooling,
                                 in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)

        model = DenseLoc(backbone=backbone, pooling=pooling,
                       normalize_embeddings=model_params.normalize_embeddings)

    else:
        raise NotImplementedError(f'Model not implemented: {model_params.model}')

    return model