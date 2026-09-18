"""任务头定义（多任务输出的各头部，结构先行）。

- 分割头：主任务，像素级条纹掩膜；
- 传播方向头：辅助监督（IWResNet-MA 思路，多任务正则），
  回归 (sin θ, cos θ) 避免角度周期性问题；
- 波长/振幅头：第三期接 SWOT 强监督与物理约束，本期仅定义接口。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SegHead(nn.Module):
    """1x1 卷积分割头，输出 logits。"""

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DirectionHead(nn.Module):
    """传播方向回归头：全局池化后回归 (sin θ, cos θ)。"""

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(self.pool(x))
        return torch.nn.functional.normalize(out, dim=-1)  # 单位圆上


class AmplitudeHead(nn.Module):
    """振幅反演头（接口占位，第三期实现 KdV/eKdV 物理约束损失）。"""

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1), nn.Softplus(),  # 振幅非负
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(x))
