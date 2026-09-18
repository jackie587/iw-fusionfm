"""双分支互补融合模型（DualFuse-UNet）。

动机（2026-08-26 对比实验发现）：
- U-Net（CNN 编码器）：高召回（条纹检出率 97.9%）但精度低——局部纹理强、
  全局结构弱，"宁可错杀"；
- Swin-UNet（Transformer 编码器）：dice/IoU 明显占优（0.565/0.507）但召回低
  （83.3%）——全局结构强、"保守精确"。
两者互补 → 双分支编码器 + 逐级特征融合 + 共享解码器。

结构：
- CNN 分支：U-Net 编码器（5 级，stride 1~16），局部纹理/细节；
- Swin 分支：timm Swin-T（4 级，stride 4~32），全局结构先验；
- 融合：在 stride 4/8/16/32 上把 CNN 特征对齐（插值+1×1 卷积）后与
  Swin 特征融合——`fusion: concat`（v1，concat+1×1 压缩）或
  `fusion: gated`（v2，Swin 特征生成门控调制 CNN 注入，抑制虚警）；
- 解码器：标准 U 型上采样回全分辨率；
- 热启动：支持分别从 U-Net / Swin-UNet 检查点加载两分支权重。

8GB 显存适配：CNN 分支 base 32，解码器 dec_ch 64，AMP + 梯度检查点。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.swin import SwinEncoder
from models.segmentation.unet import DoubleConv
from utils.logger import get_logger

logger = get_logger(__name__)


class FuseBlock(nn.Module):
    """CNN 特征对齐到 Swin 特征图后 concat + 1×1 压缩回 Swin 通道数。"""

    def __init__(self, cnn_ch: int, swin_ch: int):
        super().__init__()
        self.align = nn.Conv2d(cnn_ch, swin_ch, 1)
        self.fuse = nn.Conv2d(swin_ch * 2, swin_ch, 1)

    def forward(self, cnn_feat: torch.Tensor, swin_feat: torch.Tensor) -> torch.Tensor:
        if cnn_feat.shape[-2:] != swin_feat.shape[-2:]:
            cnn_feat = F.interpolate(cnn_feat, size=swin_feat.shape[-2:],
                                     mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.align(cnn_feat), swin_feat], dim=1))


class GatedFuseBlock(nn.Module):
    """门控融合：Swin 特征生成门控，调制 CNN 分支注入的信息量。

    动机（dual_fuse_v1 实验结论）：concat 融合里 CNN 分支的虚警会污染融合
    特征（far 23.1%，比两个单模型都高）。改为 Swin 主导：输出 = Swin 特征 +
    门控 × CNN 对齐特征，门控由 Swin 特征生成——全局结构判断"哪里可信"，
    CNN 细节只在可信区域补充，抑制虚警注入。
    """

    def __init__(self, cnn_ch: int, swin_ch: int):
        super().__init__()
        self.align = nn.Conv2d(cnn_ch, swin_ch, 1)
        self.gate = nn.Conv2d(swin_ch, swin_ch, 1)
        # 门控偏置初始化为负：训练初期 CNN 注入弱（≈Swin 单分支），
        # 避免热启动初期 CNN 虚警冲掉 Swin 分支的精度
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, cnn_feat: torch.Tensor, swin_feat: torch.Tensor) -> torch.Tensor:
        if cnn_feat.shape[-2:] != swin_feat.shape[-2:]:
            cnn_feat = F.interpolate(cnn_feat, size=swin_feat.shape[-2:],
                                     mode="bilinear", align_corners=False)
        g = torch.sigmoid(self.gate(swin_feat))
        return swin_feat + g * self.align(cnn_feat)


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = DoubleConv(cin + cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DualFuseUNet(nn.Module):
    """CNN + Swin 双编码器、逐级融合、共享解码器。"""

    # CNN 分支 5 级输出通道（base 32）：stride 1,2,4,8,16
    # Swin 分支 4 级输出通道：stride 4,8,16,32
    # 融合点：stride 4,8,16（对齐后 concat），瓶颈 stride 32 用 Swin 自身
    def __init__(self, cfg: dict):
        super().__init__()
        in_ch = cfg.get("in_channels", 3)
        out_ch = cfg.get("out_channels", 1)
        base = cfg.get("base_channels", 32)
        cnn_ch = [base * (2 ** i) for i in range(5)]  # 32,64,128,256,512

        # CNN 分支（与 U-Net 编码器同构，便于热启动）
        self.cnn_enc = nn.ModuleList(
            [DoubleConv(in_ch if i == 0 else cnn_ch[i - 1], cnn_ch[i])
             for i in range(5)])
        self.pool = nn.MaxPool2d(2)

        # Swin 分支（可加载本地 ImageNet 权重）
        self.swin = SwinEncoder(
            model_name=cfg.get("backbone", "swin_tiny_patch4_window7_224"),
            in_channels=in_ch,
            img_size=cfg.get("img_size", 512),
            pretrained=False,
            pretrained_path=(cfg.get("imagenet_weights")
                             if cfg.get("pretrained") == "imagenet" else None),
            use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", True),
        )
        swin_ch = self.swin.out_channels  # [96, 192, 384, 768]

        # 融合层：cnn stride4(64)→swin s4(96)、stride8(128)→s8(192)、
        # stride16(256)→s16(384)；瓶颈用 Swin stride32(768)
        # fusion: concat（v1）| gated（v2，Swin 门控调制 CNN 注入，抑制虚警）
        fusion = cfg.get("fusion", "concat")
        fuse_cls = {"concat": FuseBlock, "gated": GatedFuseBlock}[fusion]
        self.fuse4 = fuse_cls(cnn_ch[2], swin_ch[0])
        self.fuse8 = fuse_cls(cnn_ch[3], swin_ch[1])
        self.fuse16 = fuse_cls(cnn_ch[4], swin_ch[2])

        # 解码器：1/32 → 1/16 → 1/8 → 1/4 → 1/1
        dec_ch = 64
        self.dec3 = DecoderBlock(swin_ch[3], swin_ch[2], dec_ch * 4)
        self.dec2 = DecoderBlock(dec_ch * 4, swin_ch[1], dec_ch * 2)
        self.dec1 = DecoderBlock(dec_ch * 2, swin_ch[0], dec_ch)
        self.dec0 = DecoderBlock(dec_ch, 0, dec_ch // 2)
        self.head = nn.Conv2d(dec_ch // 2, out_ch, 1)

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        input_hw = x.shape[-2:]

        # CNN 分支：c0..c4 对应 stride 1,2,4,8,16
        c = []
        h = x
        for i in range(4):
            h = self.cnn_enc[i](h)
            c.append(h)
            h = self.pool(h)
        c.append(self.cnn_enc[4](h))

        # Swin 分支：f1..f4 对应 stride 4,8,16,32
        f1, f2, f3, f4 = self.swin(x)

        # 逐级融合
        f1 = self.fuse4(c[2], f1)
        f2 = self.fuse8(c[3], f2)
        f3 = self.fuse16(c[4], f3)

        d = self.dec3(f4, f3)
        d = self.dec2(d, f2)
        d = self.dec1(d, f1)
        d = self.dec0(d, None)
        out = self.head(d)
        return F.interpolate(out, size=input_hw, mode="bilinear",
                             align_corners=False)

    @torch.no_grad()
    def warm_start(self, unet_ckpt: str | Path | None = None,
                   swin_ckpt: str | Path | None = None) -> None:
        """从已训好的 U-Net / Swin-UNet 检查点热启动两个编码器分支。"""
        def load(ckpt_path, key_prefix, target):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            sd = ckpt.get("model", ckpt)
            sub = {k[len(key_prefix):]: v for k, v in sd.items()
                   if k.startswith(key_prefix)
                   and k[len(key_prefix):] in target.state_dict()
                   and target.state_dict()[k[len(key_prefix):]].shape == v.shape}
            missing, unexpected = target.load_state_dict(sub, strict=False)
            logger.info("热启动 %s ← %s：%d 层匹配", key_prefix, ckpt_path, len(sub))

        if unet_ckpt and Path(unet_ckpt).exists():
            load(unet_ckpt, "enc.", self.cnn_enc)
        if swin_ckpt and Path(swin_ckpt).exists():
            # swin_unet 检查点里编码器键形如 encoder.encoder.layers_0....
            ckpt = torch.load(swin_ckpt, map_location="cpu", weights_only=True)
            sd = ckpt.get("model", ckpt)
            sub = {k[len("encoder."):]: v for k, v in sd.items()
                   if k.startswith("encoder.")
                   and k[len("encoder."):] in self.swin.state_dict()
                   and self.swin.state_dict()[k[len("encoder."):]].shape == v.shape}
            self.swin.load_state_dict(sub, strict=False)
            logger.info("热启动 swin ← %s：%d 层匹配", swin_ckpt, len(sub))


def build_dual_fuse(cfg: dict) -> DualFuseUNet:
    model = DualFuseUNet(cfg)
    if cfg.get("warm_start_unet") or cfg.get("warm_start_swin"):
        model.warm_start(cfg.get("warm_start_unet"), cfg.get("warm_start_swin"))
    return model
