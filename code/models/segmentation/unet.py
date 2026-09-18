"""U-Net 基线（条纹二值分割）。

第一期基线：标准 U 型结构，输入 [VV_dB, VH_dB, 入射角] 三通道。
8GB 显存适配：base_channels 默认 32（标准 64 减半）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        ch = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.enc = nn.ModuleList([DoubleConv(in_channels if i == 0 else ch[i - 1], ch[i])
                                  for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2)
                                 for i in range(depth)])
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in range(depth)])
        self.head = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = self.pool(x)
        x = self.enc[self.depth](x)
        for i in range(self.depth - 1, -1, -1):
            x = self.up[i](x)
            s = skips[i]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = self.dec[i](torch.cat([s, x], dim=1))
        return self.head(x)  # logits，训练时用 BCE/组合损失自带 sigmoid


def build_unet(cfg: dict) -> UNet:
    return UNet(in_channels=cfg.get("in_channels", 3),
                out_channels=cfg.get("out_channels", 1),
                base_channels=cfg.get("base_channels", 32),
                depth=cfg.get("depth", 4))
