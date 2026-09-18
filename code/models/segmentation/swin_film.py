"""SwinUNet + SWOT FiLM 弱融合模型（SwinFilmUNet）。

L2 配对实验（2026-09-01，第 8 天）：
- SAR 主干完全复用 SwinUNetStrip（编码器/条带池化/解码器不动），
  继承保证 v7 检查点键名逐层对上，可整模型热启动；
- SWOT 3×20×20（ssha/|grad|/mask，已归一化）经小 CNN 编码为条件向量，
  在 bottleneck（1/32，768 通道）做 FiLM 逐通道仿射调制；
- swot 全零（缺失/dropout）的样本 gate=0 → FiLM 严格恒等退化，
  与 FiLM 零初始化的设计一致，单模态输入天然支持；
- 参数量 = v7 主干（~27.8M）+ swot_encoder/film.mlp（~0.5M），<40M。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segmentation.swin_unet import SwinUNetStrip


class SwotEncoder(nn.Module):
    """3×20×20 → cond_dim 条件向量的小 CNN（stride-2 逐级降采样）。"""

    def __init__(self, in_ch: int = 3, cond_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),   # 20→10
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),      # 10→5
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),     # 5→3
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, cond_dim), nn.ReLU(inplace=True),
        )

    def forward(self, swot: torch.Tensor) -> torch.Tensor:
        return self.net(swot)


class SwinFilmUNet(SwinUNetStrip):
    """SwinUNetStrip 主干 + bottleneck FiLM（条件来自 SWOT 编码器）。"""

    def __init__(self, cfg: dict):
        cond_dim = cfg.get("cond_dim", 128)
        super().__init__({**cfg, "film_cond_dim": cond_dim})
        self.cond_dim = cond_dim
        self.swot_encoder = SwotEncoder(
            in_ch=cfg.get("swot_channels", 3), cond_dim=cond_dim)
        # 微调时冻结主干 BN running stats（2026-09-01 实测：L2 数据与 v7
        # 训练域不同，BN 统计量一个 epoch 即大幅漂移并摧毁热启动校准，
        # val dice 0.80→0.21；回滚 BN 后恢复到 0.49。小 batch 微调惯例）
        self.freeze_bn = cfg.get("freeze_bn", False)
        # 冻结编码器与条带池化（2026-09-01 实测：v7 热启动处于极尖盆地，
        # 编码器权重漂移 ≤0.1% 即致 bottleneck 特征变化 60%、val dice
        # 0.80→0.30；冻结后特征提取固定，只训练解码器/head/FiLM 通路，
        # 两组实验的唯一差别干净地归因于 SWOT-FiLM 通路）
        if cfg.get("freeze_encoder", False):
            for name, p in self.named_parameters():
                if name.startswith(("encoder.", "strip_bot.", "strip_mid.")):
                    p.requires_grad = False

    def train(self, mode: bool = True) -> "SwinFilmUNet":
        super().train(mode)
        if mode and self.freeze_bn:
            for name, m in self.named_modules():
                # swot_encoder 是新模块，其 BN 统计量需正常适配
                if isinstance(m, nn.BatchNorm2d) \
                        and not name.startswith("swot_encoder"):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor,
                swot: torch.Tensor | None = None) -> torch.Tensor:
        input_hw = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)  # 1/4, 1/8, 1/16, 1/32

        if self.use_strip:
            f4 = self.strip_bot(f4)
            f3 = self.strip_mid(f3)

        if swot is not None:
            cond = self.swot_encoder(swot.float())
            # 全零 swot（缺失模态 / dropout）→ gate=0，FiLM 严格恒等
            gate = (swot.abs().sum(dim=(1, 2, 3)) > 0).float()
            f4 = self.film(f4, cond, gate=gate)
        else:
            f4 = self.film(f4, None)

        d = self.dec3(f4, f3)
        d = self.dec2(d, f2)
        d = self.dec1(d, f1)
        d = self.dec0(d, None)
        out = self.head(d)
        return F.interpolate(out, size=input_hw, mode="bilinear",
                             align_corners=False)

    @torch.no_grad()
    def film_modulation_stats(self, swot: torch.Tensor) -> dict:
        """对一批 swot 计算 FiLM γ/β 统计（验证调制是否偏离恒等）。"""
        device = next(self.parameters()).device
        swot = swot.to(device)
        cond = self.swot_encoder(swot.float())
        gate = (swot.abs().sum(dim=(1, 2, 3)) > 0).float()
        gb = self.film.mlp(cond) * gate[:, None]
        gamma = 1.0 + gb[:, :self.film.channels]
        beta = gb[:, self.film.channels:]
        valid = gate > 0
        if not valid.any():
            return {"gamma_dev": 0.0, "beta_abs": 0.0}
        return {
            "gamma_dev": float((gamma[valid] - 1.0).abs().mean()),  # |γ-1| 均值
            "beta_abs": float(beta[valid].abs().mean()),            # |β| 均值
        }


def build_swin_film(cfg: dict) -> SwinFilmUNet:
    return SwinFilmUNet(cfg)
