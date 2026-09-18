"""clDice 损失：保条纹连通性（拓扑约束）。

软骨架化（可微迭代腐蚀）后计算预测骨架与真值骨架的重叠度，
让网络不只"描对位置"还要"不断线"。
参考：Shit et al. 2021 clDice (CVPR)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """可微腐蚀：对 1-x 做 3x3 最大池化再取反。"""
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_skeletonize(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """可微软骨架。x: [N,1,H,W]，取值 [0,1]。"""
    skel = torch.zeros_like(x)
    for _ in range(iters):
        eroded = soft_erode(x)
        opened = -soft_erode(-eroded)  # 开运算：腐蚀后膨胀（软近似）
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
        x = eroded
    return skel


class ClDiceLoss(nn.Module):
    def __init__(self, iters: int = 10, smooth: float = 1e-6):
        super().__init__()
        self.iters, self.smooth = iters, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        t = target.float()
        skel_p = soft_skeletonize(p, self.iters)
        skel_t = soft_skeletonize(t, self.iters)
        tprec = ((skel_p * t).sum() + self.smooth) / (skel_p.sum() + self.smooth)
        tsens = ((skel_t * p).sum() + self.smooth) / (skel_t.sum() + self.smooth)
        cldice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1 - cldice
