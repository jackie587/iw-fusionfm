"""真实全场景推理：对已切块的两景南海 S1 场景跑分割模型并拼接概率图。

注意域差：训练集（Tao/S1-IW-2023）是 8-bit 显示图（三通道相同），
真实场景切块是 [VV_dB, VH_dB, 入射角] 归一化张量。为贴近训练分布，
推理只取 VV 通道复制三份输入模型。

用法（在 code/ 目录下）：
    python tests/run_scene_inference.py \
        --tiles-dir ../data/processed/sar_tiles/<场景>.SAFE \
        --ckpt ../checkpoints/experiments/swin_unet_v2/best.pth \
        --model-config configs/model_swin_strip.yaml \
        --out ../results/scene_eval/<场景名>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from evaluation.evaluate import build_model


def _feather_window(tile_size: int, taper_ratio: float = 0.25) -> np.ndarray:
    """拼接权重窗：中心为 1，边缘按升余弦 taper 到 ~0，消除 tile 边界伪影。"""
    taper = int(tile_size * taper_ratio)
    w = np.ones(tile_size, np.float32)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(taper) / taper))
    w[:taper] = ramp
    w[-taper:] = ramp[::-1]
    return np.outer(w, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stretch", choices=["none", "scene"], default="none",
                    help="scene=按全场景海面 p2~p98 拉伸 VV（对齐训练集"
                         "显示式亮度域，域差消融用）；none=固定 dB 归一化")
    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir)
    files = sorted((tiles_dir / "images").glob("*.npy"))
    pat = re.compile(r"_y(\d+)_x(\d+)\.npy$")
    coords = [tuple(int(v) for v in pat.search(f.name).groups()) for f in files]
    h = max(y for y, _ in coords) + 512
    w = max(x for _, x in coords) + 512
    print(f"{len(files)} 块，画布 {h}×{w}")

    model, device = build_model(args.model_config, args.ckpt)

    # 场景级亮度拉伸参数：从全部切块抽样估计海面 p2/p98
    lo = hi = None
    if args.stretch == "scene":
        sample = np.concatenate([
            np.load(f)[0].ravel() for f in files[::max(1, len(files) // 50)]])
        sample = sample[~np.isnan(sample)]
        lo, hi = np.percentile(sample, [2, 98])
        print(f"场景拉伸：p2={lo:.3f} p98={hi:.3f}")

    prob = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    win = _feather_window(512)
    for i in range(0, len(files), args.batch_size):
        batch_files = files[i:i + args.batch_size]
        batch_coords = coords[i:i + args.batch_size]
        # 取 VV 通道复制三份，贴近训练集（三通道相同）分布；
        # 刈幅外区域为 NaN：全 NaN 块直接跳过，部分 NaN 置 0
        tiles, kept_coords = [], []
        for f, co in zip(batch_files, batch_coords):
            vv = np.load(f)[0:1]
            if np.isnan(vv).mean() > 0.5:
                continue  # 刈幅外，不参与拼接
            if lo is not None:
                vv = np.clip((vv - lo) / max(hi - lo, 1e-6), 0, 1) \
                    .astype(np.float32)
            tiles.append(np.nan_to_num(vv, nan=0.0).repeat(3, axis=0))
            kept_coords.append(co)
        if not tiles:
            continue
        x = torch.from_numpy(np.stack(tiles)).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x)).cpu().numpy()
        for (y, xx), pp in zip(kept_coords, p):
            prob[y:y + 512, xx:xx + 512] += pp[0] * win
            weight[y:y + 512, xx:xx + 512] += win
        if (i // args.batch_size) % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    prob /= np.maximum(weight, 1e-6)
    prob[weight < 1e-3] = np.nan  # 无块覆盖区域标 NaN（下游统计排除）
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "prob.npy", prob.astype(np.float16))

    # 缩略叠合图（原图太大，降到 ~2000 px 宽出图）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first = np.load(files[0])
    # 用 meta 的 transform 不需要——出图只是示意，VV 背景用概率图画布重建太慢，
    # 直接画概率热力图 + 阈值轮廓
    scale = max(1, w // 2000)
    small = prob[::scale, ::scale]
    fig, ax = plt.subplots(figsize=(small.shape[1] / 100, small.shape[0] / 100))
    ax.imshow(small, cmap="viridis", vmin=0, vmax=1)
    ax.contour(small, levels=[0.6], colors="red", linewidths=0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out / "prob_overview.png", dpi=100)
    plt.close(fig)

    stats = {"n_tiles": len(files), "canvas": [h, w],
             "prob_mean": float(prob.mean()),
             "area_frac_gt0.6": float((prob > 0.6).mean())}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"→ {out}/prob.npy, prob_overview.png")


if __name__ == "__main__":
    main()
