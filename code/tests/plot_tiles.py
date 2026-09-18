"""切块抽查：把若干 sar_tiles 的 npy 画成 PNG 存到 results/figures/。

VV/VH 通道是归一化后的 dB（db_clip=[-30,0] → [0,1]），
第 3 通道是归一化入射角。画出 VV 灰度图 + 三分量拼图。
用法（在 code/ 目录下）：
    python tests/plot_tiles.py <tile_dir> [--n 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = CODE_ROOT.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tile_dir", help="场景切块目录（含 images/ meta/）")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/figures"))
    args = ap.parse_args()

    tiles = sorted((Path(args.tile_dir) / "images").glob("*.npy"))
    if not tiles:
        sys.exit(f"没有找到切块：{args.tile_dir}/images")
    picks = [tiles[0], tiles[len(tiles) // 2]][:args.n]  # 首块 + 中间块
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in picks:
        img = np.load(p)  # [3,H,W]，通道 [VV,VH,入射角] 已归一
        vv_db = img[0] * 30.0 - 30.0  # 反归一化回 dB 便于肉眼判读
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(vv_db, cmap="gray", vmin=-25, vmax=-5)
        axes[0].set_title("VV dB")
        axes[1].imshow(img[1] * 30.0 - 30.0, cmap="gray", vmin=-35, vmax=-15)
        axes[1].set_title("VH dB")
        im = axes[2].imshow(img[2] * 25.0 + 25.0, cmap="viridis")
        axes[2].set_title("incidence deg")
        fig.colorbar(im, ax=axes[2])
        fig.suptitle(p.stem)
        fig.savefig(out / f"{p.stem}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"已存 {out / (p.stem + '.png')}")


if __name__ == "__main__":
    main()
