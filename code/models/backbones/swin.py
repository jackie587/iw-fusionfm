"""Swin-Transformer 编码器封装（timm 实现，输出多级特征供 U 型解码）。

设计要点：
- 用 timm 的 features_only 模式拿 1/4, 1/8, 1/16, 1/32 四级特征；
- 支持 SSL4EO-S12 的 Sentinel-1 自监督权重初始化（见 ssl4eo_loader）。
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

try:
    import timm
except ImportError:  # 允许在未装 timm 的环境下导入本包其它模块
    timm = None


class SwinEncoder(nn.Module):
    """Swin-T 编码器，features_only 输出四级特征（NHWC→NCHW 已转好）。"""

    def __init__(self, model_name: str = "swin_tiny_patch4_window7_224",
                 in_channels: int = 3, img_size: int = 512,
                 pretrained: bool = False,
                 pretrained_path: str | None = None,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        if timm is None:
            raise ImportError("需要 timm：pip install timm")
        self.encoder = timm.create_model(
            model_name, features_only=True, in_chans=in_channels,
            img_size=img_size, pretrained=pretrained,
        )
        if pretrained_path:
            # 本地预训练权重（timm 格式）：timm 在线下载在本机不可达时的替代，
            # 见 models/backbones/convert_tv_swin_to_timm.py
            sd = torch.load(pretrained_path, map_location="cpu",
                            weights_only=True)
            # features_only 包装会把 layers.0 重命名为 layers_0（去点号），对齐之
            sd = {re.sub(r"^layers\.(\d+)\.", r"layers_\1.", k): v
                  for k, v in sd.items()}
            missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
            n = len(self.encoder.state_dict()) - len(missing)
            print(f"[swin] 本地预训练权重 {pretrained_path}："
                  f"匹配 {n}，缺失 {len(missing)}，忽略 {len(unexpected)}")
        self.out_channels = list(self.encoder.feature_info.channels())  # [96,192,384,768]
        if use_gradient_checkpointing:
            self.encoder.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.encoder(x)
        # timm 的 Swin features_only 输出为 NHWC，统一转 NCHW
        out = []
        for f, c in zip(feats, self.out_channels):
            if f.shape[1] != c and f.shape[-1] == c:  # NHWC → NCHW
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out
