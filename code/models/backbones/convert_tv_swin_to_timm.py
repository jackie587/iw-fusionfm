"""torchvision Swin-T 官方权重 → timm swin_tiny_patch4_window7_224 键名映射。

背景：timm 默认从 GitHub releases / HF 拉权重，本机不可达；
torchvision 官方权重在 download.pytorch.org，国内可直连。
两者同为 Swin-T/16 in1k（torchvision swin_t），仅键名不同：

    torchvision                          timm
    features.0.0.*        (patch conv)   patch_embed.proj.*
    features.0.2.*        (patch norm)   patch_embed.norm.*
    features.{1,3,5,7}.B.*(4 级 stage)   layers.{0..3}.blocks.B.*
    features.{2,4,6}.*    (patch merge)  layers.{0..2}.downsample.*
    norm.*                               norm.*
    head.*                               head.fc.*
    attn 内 mlp.0/mlp.3                  attn 内 mlp.fc1/mlp.fc2

用法（在 code/ 目录下）：
    python models/backbones/convert_tv_swin_to_timm.py \
        --src ../../swin_t-704ceda3.pth --dst <输出路径>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def convert_key(k: str) -> str | None:
    if k.startswith("features.0.0."):
        return k.replace("features.0.0.", "patch_embed.proj.", 1)
    if k.startswith("features.0.2."):
        return k.replace("features.0.2.", "patch_embed.norm.", 1)
    m = re.match(r"features\.(\d+)\.(.+)", k)
    if m:
        feat_idx, rest = int(m.group(1)), m.group(2)
        if feat_idx in (1, 3, 5, 7):  # stage
            layer = (feat_idx - 1) // 2
            rest = re.sub(r"^(\d+)\.", r"blocks.\1.", rest)
            rest = rest.replace("mlp.0.", "mlp.fc1.").replace("mlp.3.", "mlp.fc2.")
            return f"layers.{layer}.{rest}"
        if feat_idx in (2, 4, 6):  # patch merging（timm 把它挂在下一个 stage 上）
            layer = feat_idx // 2
            return f"layers.{layer}.downsample.{rest}"
    if k.startswith("norm."):
        return k
    if k.startswith("head."):
        return "head.fc." + k[len("head."):]
    return None  # relative_position_index 等 buffer 不需要


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="torchvision swin_t 权重 .pth")
    ap.add_argument("--dst", required=True, help="输出 timm 格式权重路径")
    args = ap.parse_args()

    sd = torch.load(args.src, map_location="cpu", weights_only=True)
    out = {}
    skipped = []
    for k, v in sd.items():
        # relative_position_index 在 timm 里是非持久 buffer（前向时现算），跳过
        if "relative_position_index" in k:
            skipped.append(k)
            continue
        nk = convert_key(k)
        if nk is None:
            skipped.append(k)
        else:
            out[nk] = v

    import timm
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    missing, unexpected = model.load_state_dict(out, strict=False)
    print(f"转换 {len(out)} 键，跳过 {len(skipped)}（{skipped[:3]}…）")
    print(f"timm 侧缺失 {len(missing)}，多余 {len(unexpected)}")
    assert not unexpected and not missing, "键名映射不完整"

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"已存 {dst}（{dst.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
