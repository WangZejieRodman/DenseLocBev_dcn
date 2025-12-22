# Warsaw University of Technology

import torch.nn as nn

from models.minkloc import MinkLoc
from misc.utils import ModelParams
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.layers.eca_block import ECABasicBlock
from models.minkfpn import MinkFPN
from models.layers.pooling_wrapper import PoolingWrapper
# =========================================================
# 新增: 导入我们在第二步写好的 BEV Backbone
# =========================================================
from models.minkbev import MinkBEVBackbone


def model_factory(model_params: ModelParams):
    """
    模型工厂：根据配置参数 model_params.model 来实例化相应的模型结构
    """

    # ---------------------------------------------------------
    # 分支 1: 原版 MinkLoc (3D 稀疏卷积)
    # ---------------------------------------------------------
    if model_params.model == 'MinkLoc':
        in_channels = 1
        block_module = create_resnet_block(model_params.block)
        backbone = MinkFPN(in_channels=in_channels, out_channels=model_params.feature_size,
                           num_top_down=model_params.num_top_down, conv0_kernel_size=model_params.conv0_kernel_size,
                           block=block_module, layers=model_params.layers, planes=model_params.planes)
        pooling = PoolingWrapper(pool_method=model_params.pooling, in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)
        model = MinkLoc(backbone=backbone, pooling=pooling, normalize_embeddings=model_params.normalize_embeddings)

    # ---------------------------------------------------------
    # 分支 2: 新增 MinkLocBEV (BEVNet 风格, 2D 稀疏卷积)
    # ---------------------------------------------------------
    elif model_params.model == 'MinkLocBEV':
        # 获取输入通道数 (对应 BEVQuantizer 的 Z 轴层数)
        # 我们将在配置文件中定义 'in_channels' (例如 32)
        # 如果配置文件没写，默认给 32 (配合 Step 1 的默认设置)
        in_channels = getattr(model_params, 'in_channels', 32)

        print(f"Model Factory: Initializing MinkLocBEV...")
        print(f"  Input Channels (Z-layers): {in_channels}")
        print(f"  Feature Size (Backbone Out): {model_params.feature_size}")

        # 实例化我们新写的 Backbone
        # 注意：这里显式指定 dimension=2，虽然 Backbone 内部默认也是 2
        backbone = MinkBEVBackbone(in_channels=in_channels,
                                   out_channels=model_params.feature_size,
                                   dimension=2)

        # 实例化池化层 (如 GeM)
        # 注意: PoolingWrapper 内部使用的 MinkowskiGeM 或 GlobalAvgPooling
        # 对输入张量的维度是自适应的，所以直接复用即可处理 2D 稀疏张量。
        pooling = PoolingWrapper(pool_method=model_params.pooling,
                                 in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)

        # 组装最终模型
        # MinkLoc 类是通用的，只要 backbone 和 pooling 接口对齐即可
        model = MinkLoc(backbone=backbone, pooling=pooling, normalize_embeddings=model_params.normalize_embeddings)

    else:
        raise NotImplementedError('Model not implemented: {}'.format(model_params.model))

    return model


def create_resnet_block(block_name: str) -> nn.Module:
    if block_name == 'BasicBlock':
        block_module = BasicBlock
    elif block_name == 'Bottleneck':
        block_module = Bottleneck
    elif block_name == 'ECABasicBlock':
        block_module = ECABasicBlock
    else:
        raise NotImplementedError('Unsupported network block: {}'.format(block_name))

    return block_module