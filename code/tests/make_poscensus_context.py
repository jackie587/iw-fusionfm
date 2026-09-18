# 生成普查斑块的 ±768px 上下文判读图（VV 灰度 + 概率 0.35 等值线），
# 用于对子代理判为 TP/uncertain 的斑块做人工复审。
# 用法: python tests/make_poscensus_context.py <scene> <cid> [<cid> ...]
# 默认读 scene_eval_poscensus；设环境变量 CENSUS_OUTROOT 可切换普查批次
# （如 CENSUS_OUTROOT=scene_eval_springcensus），输出文件名前缀同步替换。
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_review_sheets import build_vv_canvas

ROOT = Path(__file__).resolve().parents[2]
BATCH = os.environ.get("CENSUS_OUTROOT", "scene_eval_poscensus")
OUTROOT = ROOT / "results" / BATCH
TILES = ROOT / "data" / "processed" / "sar_tiles"
FIGDIR = ROOT / "results" / "figures"
HALF = 768


def main() -> None:
    scene = sys.argv[1]
    cids = [int(v) for v in sys.argv[2:]]
    scene_out = OUTROOT / scene
    comps = json.load(open(scene_out / "review" / "components.json"))["picked"]
    by_cid = {c["cid"]: c for c in comps}

    prob_file = scene_out / "prob_masked.npy"
    if not prob_file.exists():
        prob_file = scene_out / "prob.npy"
    prob = np.nan_to_num(np.load(prob_file).astype(np.float32), nan=0.0)
    h, w = prob.shape
    vv = build_vv_canvas(TILES / f"{scene}.SAFE", (h, w))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cid in cids:
        c = by_cid[cid]
        cy, cx = int(c["cy"]), int(c["cx"])
        y0, x0 = np.clip(cy - HALF, 0, h - 2 * HALF), np.clip(cx - HALF, 0, w - 2 * HALF)
        patch = vv[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
        pcont = prob[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
        lo, hi = np.percentile(patch, [2, 98])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(patch, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-3))
        ax.contour(pcont, levels=[0.35], colors="red", linewidths=0.8)
        ax.plot(cx - x0, cy - y0, "r+", markersize=14, markeredgewidth=2)
        ax.set_title(f"{scene[:23]} 域{cid} ({int(c['area'])}px) @({cy},{cx})")
        ax.axis("off")
        FIGDIR.mkdir(exist_ok=True)
        out = FIGDIR / f"{BATCH.replace('scene_eval_', '')}_ctx_{scene[-4:]}_cid{cid}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"→ {out}")


if __name__ == "__main__":
    main()
