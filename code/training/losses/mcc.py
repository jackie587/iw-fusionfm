"""MCC 损失：Matthews 相关系数，应对极端像素不平衡。

条纹像素占比常 <1%，准确率/IoU 易被大面积负样本掩盖；
MCC 同时考虑 TP/FP/TN/FN，IWResNet-MA 验证对内波分割有效。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MCCLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1).float()
        tp = (p * t).sum(1)
        tn = ((1 - p) * (1 - t)).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        num = tp * tn - fp * fn
        den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + self.smooth)
        return (1 - (num + self.smooth) / den).mean()
