"""组合损失：L = FT + λ1·clDice + λ2·MCC（Lovász/物理约束后续按需接入）。

权重从 train yaml 的 loss 段读取，weight=0 的项直接跳过（省算力）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.losses.cldice import ClDiceLoss
from training.losses.focal_tversky import FocalTverskyLoss
from training.losses.mcc import MCCLoss


class CompositeLoss(nn.Module):
    def __init__(self, loss_cfg: dict):
        super().__init__()
        self.terms = nn.ModuleList()
        self.weights: list[float] = []
        self.names: list[str] = []

        ft = loss_cfg.get("focal_tversky", {})
        if ft.get("weight", 0) > 0:
            self.terms.append(FocalTverskyLoss(
                alpha=ft.get("alpha", 0.3), beta=ft.get("beta", 0.7),
                gamma=ft.get("gamma", 4.0 / 3.0)))
            self.weights.append(ft["weight"])
            self.names.append("focal_tversky")

        cd = loss_cfg.get("cldice", {})
        if cd.get("weight", 0) > 0:
            self.terms.append(ClDiceLoss(iters=cd.get("iters", 10)))
            self.weights.append(cd["weight"])
            self.names.append("cldice")

        mc = loss_cfg.get("mcc", {})
        if mc.get("weight", 0) > 0:
            self.terms.append(MCCLoss())
            self.weights.append(mc["weight"])
            self.names.append("mcc")

        if not self.terms:
            raise ValueError("组合损失为空：至少给一个损失项配置正权重")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        total = torch.zeros((), device=logits.device)
        detail = {}
        for term, w, name in zip(self.terms, self.weights, self.names):
            v = term(logits, target)
            total = total + w * v
            detail[name] = float(v.detach())
        detail["total"] = float(total.detach())
        return total, detail
