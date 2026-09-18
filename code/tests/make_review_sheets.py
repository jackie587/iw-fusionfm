"""生成人工核查用编号网格图：从全场景概率图抽检出斑块，裁窗口+编号拼 sheet。

用法（在 code/ 目录下，需先跑 run_scene_inference.py）：
    python tests/make_review_sheets.py \
        --tiles-dir ../data/processed/sar_tiles/<场景>.SAFE \
        --scene-out ../results/scene_eval/<场景名> --n 150
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

WINDOW = 384          # 核查窗口边长（像素）
GRID = 5              # 每张 sheet 5×5


def build_vv_canvas(tiles_dir: Path, canvas_hw: tuple[int, int]) -> np.ndarray:
    """从切块重建 VV 通道全场景图（带重叠区均值），缓存为 vv.npy。"""
    cache = tiles_dir / "_vv_canvas.npy"
    if cache.exists():
        return np.load(cache)
    h, w = canvas_hw
    canvas = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    pat = re.compile(r"_y(\d+)_x(\d+)\.npy$")
    files = sorted((tiles_dir / "images").glob("*.npy"))
    for f in files:
        y, x = (int(v) for v in pat.search(f.name).groups())
        tile = np.load(f)[0]
        valid = np.isfinite(tile)
        canvas[y:y + 512, x:x + 512] += np.where(valid, tile, 0.0)
        weight[y:y + 512, x:x + 512] += valid
    canvas /= np.maximum(weight, 1e-6)
    np.save(cache, canvas)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--scene-out", required=True)
    ap.add_argument("--n", type=int, default=150, help="抽样斑块数")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from scipy import ndimage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_out = Path(args.scene_out)
    # 优先用陆地掩膜后的概率图（run_scene_inference + apply_land_mask 产物）
    prob_file = scene_out / "prob_masked.npy"
    if not prob_file.exists():
        prob_file = scene_out / "prob.npy"
    prob = np.load(prob_file).astype(np.float32)
    prob = np.nan_to_num(prob, nan=0.0)
    h, w = prob.shape

    lab, n = ndimage.label(prob > args.threshold)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    comps = [{"cid": i + 1, "area": float(sizes[i])}
             for i in range(n) if sizes[i] >= args.min_area]
    print(f"连通域 {n} 个，≥{args.min_area}px 的 {len(comps)} 个")

    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(comps), size=min(args.n, len(comps)),
                       replace=False)
    picked = [comps[i] for i in sorted(picks)]
    cents = ndimage.center_of_mass(lab > 0, lab,
                                   [c["cid"] for c in picked])
    for c, (cy, cx) in zip(picked, cents):
        c["cy"], c["cx"] = float(cy), float(cx)

    vv = build_vv_canvas(Path(args.tiles_dir), (h, w))

    review_dir = scene_out / "review"
    review_dir.mkdir(exist_ok=True)
    hw = WINDOW // 2
    for si in range(0, len(picked), GRID * GRID):
        batch = picked[si:si + GRID * GRID]
        fig, axes = plt.subplots(GRID, GRID, figsize=(GRID * 3, GRID * 3))
        for ax, c in zip(axes.flat, batch):
            cy, cx = int(c["cy"]), int(c["cx"])
            y0, x0 = np.clip(cy - hw, 0, h - WINDOW), np.clip(cx - hw, 0, w - WINDOW)
            patch = vv[y0:y0 + WINDOW, x0:x0 + WINDOW]
            pcont = prob[y0:y0 + WINDOW, x0:x0 + WINDOW]
            # 自适应对比度：条纹仅 1~2 dB 对比，固定 [0,1] 显示会全灰
            lo, hi = np.percentile(patch, [2, 98])
            ax.imshow(patch, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-3))
            ax.contour(pcont, levels=[args.threshold], colors="red",
                       linewidths=0.8)
            # 核查编号 = sheet 内序号 + 全局序号，双标注防错位
            ax.set_title(f"#{si + batch.index(c) + 1} (域{c['cid']}, {int(c['area'])}px)",
                         fontsize=9)
            ax.axis("off")
        for ax in axes.flat[len(batch):]:
            ax.axis("off")
        out = review_dir / f"sheet_{si // (GRID * GRID) + 1:02d}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
    with open(review_dir / "components.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": args.threshold, "picked": picked}, f,
                  ensure_ascii=False, indent=2)
    print(f"→ {review_dir}/ ({(len(picked) + GRID*GRID - 1) // (GRID*GRID)} 张 sheet"
          f" + components.json)")


if __name__ == "__main__":
    main()
