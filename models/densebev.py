import torch
import torch.nn as nn
from models.layers.dense_layers import DenseECALayer, DCNv2Block


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class ResBasicBlock(nn.Module):
    """
    改良版 ResNet BasicBlock
    支持切换标准卷积和 DCN (可变形卷积)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_dcn=False):
        super(ResBasicBlock, self).__init__()

        # 第一层卷积：通常负责降采样，保持标准卷积以稳定训练
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二层卷积：如果是深层，使用 DCN 处理几何形变
        if use_dcn:
            self.conv2 = DCNv2Block(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DenseBEVBackbone(nn.Module):
    """
    Next-Gen Dense Backbone for Mine Tunnels
    特点:
    1. ResNet-18 架构: 更深、更强的特征提取
    2. Stem + ECA: 在第一层利用注意力机制筛选高度层 (去噪)
    3. DCN (Stage 3 & 4): 利用可变形卷积适应巷道弯曲

    输入: (B, 32, 256, 256)
    输出: (B, 512, 32, 32)
    """

    def __init__(self, in_channels=32, layers=[2, 2, 2, 2]):
        super(DenseBEVBackbone, self).__init__()
        self.inplanes = 64

        # --- Stem Layers (头部处理) ---
        # 使用 7x7 卷积替代原来的 11x11，减少参数量但保持大感受野
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # [关键改进] 引入 ECA 模块
        # 作用：自动学习 Z 轴通道权重，抑制底板/顶板噪声，增强侧壁特征
        self.eca = DenseECALayer(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Body Layers (骨干网络) ---
        # Layer 1: 64 channels, 64x64 (标准卷积)
        self.layer1 = self._make_layer(ResBasicBlock, 64, layers[0], use_dcn=False)

        # Layer 2: 128 channels, 32x32 (标准卷积)
        self.layer2 = self._make_layer(ResBasicBlock, 128, layers[1], stride=2, use_dcn=False)

        # Layer 3: 256 channels, 16x16 -> 保持 32x32?
        # 原项目最终输出是 32x32。ResNet标准是下采样到 1/32。
        # 为了适配你的 NetVLAD，我们调整 Layer 3 和 4 的 stride 策略，保持 32x32 输出。
        # 这里的 stride=1 意味着不再降采样分辨率，利用 Dilation 或者保持分辨率来维持细节。
        # 但通常我们希望感受野大一点。让我们保持 ResNet 标准 stride，看看输出尺寸。
        # 输入 256 -> Stem(/4) -> 64x64.
        # Layer1 -> 64x64
        # Layer2(/2) -> 32x32
        # Layer3 (原本/2) -> 这里我们设 stride=1 保持 32x32，增强特征
        self.layer3 = self._make_layer(ResBasicBlock, 256, layers[2], stride=1, use_dcn=True)

        # Layer 4: 512 channels, 32x32
        self.layer4 = self._make_layer(ResBasicBlock, 512, layers[3], stride=1, use_dcn=True)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_dcn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_dcn=use_dcn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_dcn=use_dcn))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 32, 256, 256)

        # Stem
        x = self.conv1(x)  # -> (B, 64, 128, 128)
        x = self.bn1(x)
        x = self.relu(x)

        # Apply Attention (Filter noise layers)
        x = self.eca(x)

        x = self.maxpool(x)  # -> (B, 64, 64, 64)

        # Stages
        x = self.layer1(x)  # -> (B, 64, 64, 64)
        x = self.layer2(x)  # -> (B, 128, 32, 32)
        x = self.layer3(x)  # -> (B, 256, 32, 32) [DCN Active]
        x = self.layer4(x)  # -> (B, 512, 32, 32) [DCN Active]

        return x