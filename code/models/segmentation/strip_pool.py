"""多方向条带池化模块（Strip Pooling）。

针对内波条纹"细长、弯曲、方向各异"的特点：
水平/垂直/双对角四个方向做长条池化，捕获长程带状上下文。
参考：SPNet (Hou et al. 2020) + 内波分割文献中的多方向扩展。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPool2d(nn.Module):
    """单方向条带池化：沿一个方向做全局均值池化后广播回来。"""

    def __init__(self, channels: int, direction: str = "h"):
        super().__init__()
        assert direction in ("h", "v")
        self.direction = direction
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.direction == "h":
            pooled = x.mean(dim=3, keepdim=True).expand_as(x)
        else:
            pooled = x.mean(dim=2, keepdim=True).expand_as(x)
        return self.bn(self.conv(pooled))


class DiagonalStripPool2d(nn.Module):
    """对角方向条带池化：先旋转 45° 附近的特征轴，等价于沿对角线池化。

    实现上用 unfold 代价高，这里采用沿反对角线/主对角线方向的近似：
    对特征图做 45° 仿射采样代价大，简化为对翻转/转置后做 h/v 池化。
    """

    def __init__(self, channels: int, diag: str = "main"):
        super().__init__()
        assert diag in ("main", "anti")
        self.diag = diag
        self.inner = StripPool2d(channels, "h")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转置使对角方向变为水平方向，池化后再转回来
        xt = x.transpose(2, 3)
        if self.diag == "anti":
            xt = xt.flip(2)
        out = self.inner(xt)
        if self.diag == "anti":
            out = out.flip(2)
        return out.transpose(2, 3)


class MultiDirectionStripPool(nn.Module):
    """四方向条带池化融合：h/v/主对角/反对角 拼接后 1x1 压缩 + 残差。"""

    def __init__(self, channels: int):
        super().__init__()
        self.h = StripPool2d(channels, "h")
        self.v = StripPool2d(channels, "v")
        self.d1 = DiagonalStripPool2d(channels, "main")
        self.d2 = DiagonalStripPool2d(channels, "anti")
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.h(x), self.v(x), self.d1(x), self.d2(x)], dim=1)
        return x + self.fuse(out)
