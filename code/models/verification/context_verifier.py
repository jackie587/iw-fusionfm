"""上下文校验器：对 v7 检出的候选斑块做二分类（真内波波包 vs 虚警）。

两种输入模式（同架构，唯一差别是上下文通道，用于消融）：
- mode="center"：1 通道，仅中心 512 裁块 resize 到 384；
- mode="context"：2 通道，[中心裁块384, 同中心1536上下文降采样384]。
  （mask 通道未用：基线若不含 mask 而上下文模型含，则增益无法归因于上下文。）
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_verifier(mode: str = "context", backbone: str = "resnet18",
                   pretrained: bool = False) -> nn.Module:
    """二分类校验器，输出单个 logit（>0 判真内波）。"""
    in_chans = {"center": 1, "context": 2}[mode]
    model = timm.create_model(backbone, pretrained=pretrained,
                              in_chans=in_chans, num_classes=1)
    return model


class Verifier(nn.Module):
    """带标准化头的封装：vv 通道减均值除标准差。"""

    def __init__(self, mode: str = "context", backbone: str = "resnet18",
                 vv_mean: float = 0.5, vv_std: float = 0.2):
        super().__init__()
        self.mode = mode
        self.net = build_verifier(mode, backbone)
        self.register_buffer("vv_mean", torch.tensor(float(vv_mean)))
        self.register_buffer("vv_std", torch.tensor(float(vv_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.vv_mean) / self.vv_std   # 两通道都是 vv 亮度，同一标准化
        return self.net(x).squeeze(1)
