"""Swin 编码器 + U 型解码 + 多方向 strip pooling 的条纹分割模型。

第二阶段主力架构（IW-FusionFM 的 SAR 单模态核心），本期搭建并冒烟：
- 编码器：timm Swin-T（可加载 SSL4EO-S12 的 S1 自监督权重）；
- 解码器：逐级上采样 + 跳跃连接；
- bottleneck 与 1/8 层插入多方向条带池化，捕获条纹长程结构；
- 可选 FiLM 环境条件注入（cond_dim>0 时）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.swin import SwinEncoder
from models.backbones.ssl4eo_loader import load_ssl4eo_weights
from models.fusion.film import FiLM
from models.segmentation.strip_pool import MultiDirectionStripPool


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(cout + cskip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([skip, x], dim=1)
        else:
            x = F.pad(x, (0, 0, 0, 0))  # 无跳跃连接时直接卷积
        return self.conv(x)


class SwinUNetStrip(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        in_ch = cfg.get("in_channels", 3)
        out_ch = cfg.get("out_channels", 1)
        self.encoder = SwinEncoder(
            model_name=cfg.get("backbone", "swin_tiny_patch4_window7_224"),
            in_channels=in_ch,
            img_size=cfg.get("img_size", 512),
            pretrained=False,  # timm 在线下载本机不可达，走本地权重
            pretrained_path=(cfg.get("imagenet_weights")
                             if cfg.get("pretrained") == "imagenet" else None),
            use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", False),
        )
        ch = self.encoder.out_channels  # 例如 [96, 192, 384, 768]
        dec_ch = 64  # 8GB 显存适配

        self.use_strip = cfg.get("strip_pool", True)
        if self.use_strip:
            self.strip_bot = MultiDirectionStripPool(ch[-1])
            self.strip_mid = MultiDirectionStripPool(ch[-2])

        cond_dim = cfg.get("film_cond_dim", 0)
        self.film = FiLM(ch[-1], cond_dim=cond_dim)

        # 解码：1/32 → 1/16 → 1/8 → 1/4 → 1/1
        self.dec3 = DecoderBlock(ch[3], ch[2], dec_ch * 4)
        self.dec2 = DecoderBlock(dec_ch * 4, ch[1], dec_ch * 2)
        self.dec1 = DecoderBlock(dec_ch * 2, ch[0], dec_ch)
        self.dec0 = DecoderBlock(dec_ch, 0, dec_ch // 2)  # 上采样回 1/2
        self.head = nn.Conv2d(dec_ch // 2, out_ch, 1)

        if cfg.get("pretrained") == "ssl4eo":
            load_ssl4eo_weights(self.encoder,
                                cfg.get("ssl4eo_weights", ""),
                                in_chans=in_ch)

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        input_hw = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)  # 1/4, 1/8, 1/16, 1/32

        if self.use_strip:
            f4 = self.strip_bot(f4)
            f3 = self.strip_mid(f3)
        f4 = self.film(f4, cond)

        d = self.dec3(f4, f3)
        d = self.dec2(d, f2)
        d = self.dec1(d, f1)
        d = self.dec0(d, None)
        out = self.head(d)
        return F.interpolate(out, size=input_hw, mode="bilinear", align_corners=False)


def build_swin_unet(cfg: dict) -> SwinUNetStrip:
    return SwinUNetStrip(cfg)
