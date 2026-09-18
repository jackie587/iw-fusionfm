"""评估指标：像素级 + 条纹级（容差）+ 虚警率。

设计原则（见《最强技术方案》五：诚实评估）：
- 像素级 Dice/IoU/F1 是基础，但会被大面积负样本掩盖，必须配合条纹级指标；
- 条纹级检出率允许定位容差（默认 3 px，符合专家判读习惯）；
- 虚警率（FAR）= 预测的连通域中与真值无任何重叠的比例。
"""
from __future__ import annotations

import numpy as np
import torch


def _flat(p: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return p.flatten().float(), t.flatten().float()


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    p, t = _flat(pred, target)
    inter = (p * t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    p, t = _flat(pred, target)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + eps) / (union + eps))


def f1_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    return dice_score(pred, target, eps)  # 二值分割下 F1 与 Dice 等价


def false_alarm_rate(pred: np.ndarray, target: np.ndarray,
                     min_area: int = 10) -> float:
    """连通域级虚警率：预测有、真值完全没有重叠的连通域占比。"""
    from scipy import ndimage  # 延迟导入，评估时才用
    pred = (pred > 0).astype(np.uint8)
    lab, n = ndimage.label(pred)
    if n == 0:
        return 0.0
    false = 0
    for i in range(1, n + 1):
        region = lab == i
        if region.sum() < min_area:
            continue
        if not (target[region] > 0).any():
            false += 1
    return false / n


def stripe_recall_with_tolerance(pred: np.ndarray, target: np.ndarray,
                                 tolerance_px: int = 3) -> float:
    """容差条纹级检出率：真值条纹连通域中，附近有预测像素的占比。"""
    from scipy import ndimage
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    lab, n = ndimage.label(target)
    if n == 0:
        return float("nan")
    dil = ndimage.binary_dilation(pred, iterations=tolerance_px)
    hit = sum(1 for i in range(1, n + 1) if dil[lab == i].any())
    return hit / n
