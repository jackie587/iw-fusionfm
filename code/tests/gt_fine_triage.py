"""GT 补样分诊 sheet：块级（无上下文）窗口拉伸图，2×3 拼版快速分级。

对 backfill2.json 候选出 triage sheet（每块 512px 原图 p2~p98 拉伸，
块级拉伸比全 sheet 拉伸更能显出条纹），AI 目视按 clear/marginal/no 分级。

用法（code/ 目录下）：python tests/gt_fine_triage.py [--scenes 7E00 D11C]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
OUT = PROJECT_ROOT / "results/gt_fine"
GRID_R, GRID_C = 2, 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="只出这些场景尾缀的候选")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cands = json.loads((OUT / "backfill2.json").read_text(encoding="utf-8"))
    if args.scenes:
        cands = [c for c in cands
                 if any(c["scene"].endswith(s) for s in args.scenes)]
    print(f"{len(cands)} 候选")

    cache: dict[str, dict] = {}
    for i in range(0, len(cands), GRID_R * GRID_C):
        batch = cands[i:i + GRID_R * GRID_C]
        fig, axes = plt.subplots(GRID_R, GRID_C,
                                 figsize=(GRID_C * 3.4, GRID_R * 3.4))
        for ax, c in zip(axes.flat, batch):
            key = c["tile"]
            if key not in cache:
                vv = np.load(TILES_ROOT / f"{c['scene']}.SAFE/images/"
                             f"{c['tile']}.npy")[0]
                vv = np.nan_to_num(vv, nan=0.0)
                finite = vv[vv > 0]
                p2, p98 = np.percentile(finite, [2, 98])
                cache[key] = (vv, p2, p98)
            vv, p2, p98 = cache[key]
            ax.imshow(vv, cmap="gray", vmin=p2, vmax=p98)
            ax.set_title(f"{c['id']} {c['scene'][-4:]}", fontsize=9)
            ax.axis("off")
        for ax in axes.flat[len(batch):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"triage_{i // (GRID_R * GRID_C) + 1:02d}.png",
                    dpi=120)
        plt.close(fig)
    print(f"→ {OUT}/triage_*.png")


if __name__ == "__main__":
    main()
