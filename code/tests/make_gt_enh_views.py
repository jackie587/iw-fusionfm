"""GT 标注辅助：生成局部对比度增强的 VV 检视图（带坐标刻度）。

斑点噪声会淹没弱条纹，直接 p2~p98 拉伸看不清楚。增强方法：
高斯背景（σ=24）归一化（VV/背景）+ 2~98 百分位拉伸；
图上带 64px 间隔坐标刻度，便于在 overrides.json 里写 prompt 框。

产出 results/gt_fine/vv_enh/<id>.png。

用法（code/ 目录下）：python tests/make_gt_enh_views.py [--only gt001]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"
OUT = PROJECT_ROOT / "results/gt_fine/vv_enh"


def enhance(vv: np.ndarray) -> np.ndarray:
    """局部 z-score：去 σ=64 大尺度背景（保留 λ≤~900m 条纹），2~98 拉伸。"""
    from scipy import ndimage
    v = np.nan_to_num(vv.astype(np.float32), nan=0.0)
    bg = ndimage.gaussian_filter(v, 64)
    r = v - bg
    p2, p98 = np.percentile(r, [2, 98])
    return np.clip((r - p2) / max(p98 - p2, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection = json.loads(
        (GT_ROOT / "selection.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for s in selection:
        if args.only and s["id"] not in args.only:
            continue
        vv = np.load(GT_ROOT / "images" / f"{s['id']}.npy")
        en = enhance(vv)
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.imshow(en, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks(np.arange(0, 513, 64))
        ax.set_yticks(np.arange(0, 513, 64))
        ax.tick_params(labelsize=8, colors="red")
        ax.grid(color="red", alpha=0.25, lw=0.5)
        wl = s.get("wavelength_m") or "-"
        ax.set_title(f"{s['id']} {s['source']} {s['scene'][-4:]} "
                     f"λ={wl}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{s['id']}.png", dpi=100)
        plt.close(fig)
    print(f"→ {OUT}/ ({len(selection)} 张)")


if __name__ == "__main__":
    main()
