"""GT 补样候选生成：从强事件（mean_prob 排序）取切块，出增强检视图。

第一轮选样按事件密度配额，目视后发现 7 月噪声场景多数切块条纹不可靠。
本脚本改为按 mean_prob 降序（高置信事件优先）从指定场景补候选，
出带坐标刻度的增强图供 AI 目视分级，接受的再并入 selection.json。

用法（code/ 目录下）：
    python tests/gt_fine_backfill.py --scenes 7E00 F792 D11C E318 A6CD DC8D \
        --per-scene 8 --tag c1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code/tests"))
from select_gt_fine import event_tile, load_tile, tile_ok  # noqa: E402
from make_gt_enh_views import enhance  # noqa: E402

EVENT_DB = PROJECT_ROOT / "results/event_database"
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"
OUT = PROJECT_ROOT / "results/gt_fine/vv_enh_cand"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="场景名末尾 4 位（如 7E00）")
    ap.add_argument("--per-scene", type=int, default=8)
    ap.add_argument("--tag", default="c1")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection = json.loads(
        (GT_ROOT / "selection.json").read_text(encoding="utf-8"))
    used = {s["tile"] for s in selection}

    with open(EVENT_DB / "events.csv", encoding="utf-8") as f:
        events = [r for r in csv.DictReader(f)
                  if r["quality"] == "high" and r["wavelength_m"]]

    OUT.mkdir(parents=True, exist_ok=True)
    cands = []
    n = 0
    for suffix in args.scenes:
        es = [e for e in events if e["scene"].endswith(suffix)]
        es.sort(key=lambda e: -float(e["mean_prob"]))
        got = 0
        for e in es:
            if got >= args.per_scene:
                break
            loc = event_tile(e["scene"], float(e["lon"]), float(e["lat"]))
            if loc is None:
                continue
            stem, y, x = loc
            if stem in used:
                continue
            tile = load_tile(e["scene"], stem)
            if tile is None or not tile_ok(tile[0], e["scene"], y, x):
                continue
            used.add(stem)
            got += 1
            cid = f"{args.tag}{n:03d}"
            n += 1
            cands.append({
                "id": cid, "source": "scene_event",
                "scene": e["scene"], "tile": stem, "y": y, "x": x,
                "event_id": e["event_id"],
                "cluster": "west" if float(e["lon"]) < 109.3 else "east",
                "wavelength_m": e["wavelength_m"],
                "direction_deg": e["direction_deg"],
                "mean_prob": e["mean_prob"],
                "reason": f"补样：high 事件 mean_prob={e['mean_prob']}",
            })
            en = enhance(np.nan_to_num(tile[0], nan=0.0))
            fig, ax = plt.subplots(figsize=(6.4, 6.4))
            ax.imshow(en, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks(np.arange(0, 513, 64))
            ax.set_yticks(np.arange(0, 513, 64))
            ax.tick_params(labelsize=8, colors="red")
            ax.grid(color="red", alpha=0.25, lw=0.5)
            ax.set_title(f"{cid} {suffix} λ={e['wavelength_m']}m "
                         f"p={float(e['mean_prob']):.2f}", fontsize=10)
            fig.tight_layout()
            fig.savefig(OUT / f"{cid}.png", dpi=100)
            plt.close(fig)
    (OUT / f"cands_{args.tag}.json").write_text(
        json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cands)} 候选 → {OUT}/ ({args.tag}*)")


if __name__ == "__main__":
    main()
