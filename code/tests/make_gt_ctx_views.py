"""GT 标注辅助 v2：1024×1024 上下文窗口检视图（红框=512 切块）。

事件库 QA 经验：条纹在大窗口+窗口级 p2~p98 拉伸下最清晰（512 块内
局部增强反而淹没在斑点里）。本脚本对 selection.json 每块取其上下左右
各扩 256px 的画布窗口（_vv_canvas.npy 缓存），窗口级拉伸，红框标出
切块边界，框内 64px 网格用切块局部坐标，便于写 overrides.json 框。

产出 results/gt_fine/vv_ctx/<id>.png。

用法（code/ 目录下）：python tests/make_gt_ctx_views.py [--only gt001]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code/tests"))
from make_review_sheets import build_vv_canvas  # noqa: E402

TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"
OUT = PROJECT_ROOT / "results/gt_fine/vv_ctx"
MARGIN = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--selection", default=None,
                    help="自定义选样清单路径（默认 gt_fine/selection.json）")
    ap.add_argument("--out", default=None, help="输出目录（默认 vv_ctx）")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection = json.loads(
        (Path(args.selection) if args.selection
         else GT_ROOT / "selection.json").read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas_cache: dict[str, np.ndarray] = {}

    for s in selection:
        if args.only and s["id"] not in args.only:
            continue
        scene, y, x = s["scene"], s["y"], s["x"]
        if scene not in canvas_cache:
            tiles_dir = TILES_ROOT / f"{scene}.SAFE"
            cache = tiles_dir / "_vv_canvas.npy"
            if cache.exists():
                canvas_cache[scene] = np.load(cache)
            else:
                pat = re.compile(r"_y(\d+)_x(\d+)\.npy$")
                coords = [tuple(int(v) for v in pat.search(f.name).groups())
                          for f in (tiles_dir / "images").glob("*.npy")]
                hw = (max(y0 for y0, _ in coords) + 512,
                      max(x0 for _, x0 in coords) + 512)
                canvas_cache[scene] = build_vv_canvas(tiles_dir, hw)
        canvas = canvas_cache[scene]
        h, w = canvas.shape
        Y1, Y2 = max(0, y - MARGIN), min(h, y + 512 + MARGIN)
        X1, X2 = max(0, x - MARGIN), min(w, x + 512 + MARGIN)
        win = canvas[Y1:Y2, X1:X2]
        finite = win[np.isfinite(win)]
        p2, p98 = np.percentile(finite, [2, 98]) if finite.size else (0, 1)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np.nan_to_num(win), cmap="gray", vmin=p2,
                  vmax=max(p98, p2 + 1e-3),
                  extent=[X1 - x, X2 - x, Y2 - y, Y1 - y])  # 切块局部坐标
        ax.add_patch(plt.Rectangle((0, 0), 512, 512, fill=False,
                                   edgecolor="red", lw=1.5))
        ax.set_xticks(np.arange(-256, 769, 64))
        ax.set_yticks(np.arange(-256, 769, 64))
        ax.tick_params(labelsize=7, colors="red")
        ax.grid(color="red", alpha=0.2, lw=0.4)
        wl = s.get("wavelength_m") or "-"
        ax.set_title(f"{s['id']} {s['source']} {scene[-4:]} λ={wl}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{s['id']}.png", dpi=100)
        plt.close(fig)
    print(f"→ {out_dir}/")


if __name__ == "__main__":
    main()
