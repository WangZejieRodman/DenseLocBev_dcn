import math
import torch
import torch.nn as nn
import torchvision.ops as ops


class DenseECALayer(nn.Module):
    """
    ECA Layer for Dense 2D Tensors (B, C, H, W)
    Implementation based on the paper: https://arxiv.org/abs/1910.03151
    """

    def __init__(self, channels, gamma=2, b=1):
        super(DenseECALayer, self).__init__()
        # 计算自适应卷积核大小 k
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D卷积用于处理通道间关系
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()

        # 1. Global Average Pooling -> (B, C, 1, 1)
        y = self.avg_pool(x)

        # 2. Reshape for 1D Conv -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)

        # 3. 1D Conv + Sigmoid
        y = self.conv(y)
        y = self.sigmoid(y)

        # 4. Reshape back -> (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # 5. Channel-wise Multiplication (Broadcasting)
        return x * y.expand_as(x)


class DCNv2Block(nn.Module):
    """
    封装了 Offset/Mask 预测的 Modulated Deformable Convolution (DCNv2)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DCNv2Block, self).__init__()

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Offset (2*k*k) + Mask (k*k)
        # 3*3卷积核需要预测 18个坐标偏移 + 9个mask权重 = 27通道
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # 初始化 offset 为 0, mask 为 0.5 (sigmoid前为0)
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

        # 实际的可变形卷积操作
        self.dcn = ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        # 预测偏移量和掩码
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # 执行卷积
        return self.dcn(x, offset, mask)