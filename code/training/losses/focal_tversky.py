"""Focal Tversky 损失：极端不平衡细条纹的主体损失。

Tversky 指数是 Dice 的推广，α 惩罚 FP、β 惩罚 FN；
内波条纹占比 <1% 且漏检代价高 → β>α 保召回。
Focal 项 (1-TI)^γ 聚焦难例。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 4.0 / 3.0, smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1).float()
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1 - ti) ** self.gamma).mean()
