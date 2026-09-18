"""SSL4EO-S12 Sentinel-1 自监督权重加载器。

权重获取：https://github.com/DLR-MF-DAS/SSL4EO-S12 （CC-BY-4.0）

勘误（2026-08-26 实测）：SSL4EO-S12 的 S1 分支只发布了 MoCo-ResNet50 和
MAE-ViT 权重，没有 Swin 权重。ViT-S/16 权重（B2_vits16_mae_ep99.pth）
对 timm Swin-T 编码器 0 层匹配（254 层全跳过）。因此：
    - Swin-UNet 用 pretrained: imagenet 初始化（timm 直接支持 3 通道）；
    - SSL4EO 权重留给后续 ViT/ResNet 编码器路线使用。

注意：SSL4EO 的 S1 输入为 2 通道（VV/VH），本模型为 3 通道（+入射角），
对 patch embedding 做通道适配：第 3 通道权重用 VV/VH 均值初始化。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger(__name__)


def adapt_input_channels(weight: torch.Tensor, in_chans: int) -> torch.Tensor:
    """把 patch embedding 的输入通道数适配到 in_chans。

    原权重 shape: [embed_dim, old_in, k, k]。不足通道用已有通道均值填充。
    """
    old_in = weight.shape[1]
    if old_in == in_chans:
        return weight
    if old_in > in_chans:
        return weight[:, :in_chans]
    mean_extra = weight.mean(dim=1, keepdim=True).expand(-1, in_chans - old_in, -1, -1)
    return torch.cat([weight, mean_extra], dim=1)


def load_ssl4eo_weights(model: nn.Module, weights_path: str | Path,
                        in_chans: int = 3) -> list[str]:
    """把 SSL4EO-S12 的 S1 Swin 权重加载进编码器，返回未匹配的键列表。"""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        logger.warning("SSL4EO 权重不存在：%s，跳过预训练初始化", weights_path)
        return ["<missing file>"]

    # weights_only=False：SSL4EO 官方 ckpt 内含 argparse.Namespace 等对象，
    # torch>=2.6 默认 weights_only=True 会拒绝；来源可信（DLR 官方发布）
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    # 去掉常见前缀（MoCo/DINO 保存格式里编码器常带 backbone./encoder. 前缀）
    cleaned = {}
    for k, v in state.items():
        for prefix in ("backbone.", "encoder.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    model_keys = set(model.state_dict().keys())
    loaded, skipped = {}, []
    for k, v in cleaned.items():
        if k not in model_keys:
            skipped.append(k)
            continue
        target = model.state_dict()[k]
        if v.shape != target.shape:
            if "patch_embed" in k and v.dim() == 4:
                v = adapt_input_channels(v, in_chans)
                if v.shape == target.shape:
                    loaded[k] = v
                    continue
            skipped.append(k)
            continue
        loaded[k] = v

    missing = model.load_state_dict(loaded, strict=False)
    logger.info("SSL4EO 权重加载：%d 层匹配，%d 层跳过，%d 层未初始化",
                len(loaded), len(skipped), len(missing.missing_keys))
    return skipped
