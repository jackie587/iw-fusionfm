"""FiLM 条件调制模块（Feature-wise Linear Modulation）。

用途（第三期融合核心，本期先把结构定好）：
- SWOT 内潮背景 / GDEM 层结 / TPXO 潮相位 / ERA5 风场等环境条件
  编码为条件向量，对 SAR 主干特征做逐通道仿射调制：y = γ(c)·x + β(c)；
- 条件缺失时退化为恒等映射（γ=1, β=0）——单模态输入天然支持；
- 时间差 Δt 作为标量拼进条件向量一并注入。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """对 [N, C, H, W] 特征做条件仿射调制。

    cond_dim=0 时 forward 忽略条件直接返回输入（恒等退化）。
    """

    def __init__(self, channels: int, cond_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, channels * 2),
            )
            # 零初始化输出层 → 初始即为恒等映射，训练稳定
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor | None = None,
                gate: torch.Tensor | None = None) -> torch.Tensor:
        if self.cond_dim == 0 or cond is None:
            return x
        gb = self.mlp(cond)  # [N, 2C]
        if gate is not None:
            # 条件缺失的样本（gate=0）强制恒等调制，与训练初期零初始化一致
            gb = gb * gate[:, None].to(gb.dtype)
        gamma, beta = gb[:, :self.channels], gb[:, self.channels:]
        gamma = 1.0 + gamma  # 以恒等为中心
        return x * gamma[:, :, None, None] + beta[:, :, None, None]
