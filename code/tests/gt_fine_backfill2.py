"""GT 补样 v2：已确认清晰块的邻域扩展 + 同场景高置信事件补样。

第一轮目视分级后清晰正样本集中在 7 个场景（E318/7E00/D11C/DC8D/A6CD/
55E0/E4CD）。波包跨多个切块，邻域块大概率也有清晰条纹。
本脚本：
1. 对已确认清晰的种子块取 ±384 邻域（存在且未用过的块）；
2. 加这些场景 mean_prob 最高的 high 事件所在块（去重）；
产出候选清单 results/gt_fine/backfill2.json + ctx 检视图
（用 make_gt_ctx_views.py --selection 生成）。

用法（code/ 目录下）：python tests/gt_fine_backfill2.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code/tests"))
from select_gt_fine import event_tile, load_tile, tile_ok  # noqa: E402

TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
EVENT_DB = PROJECT_ROOT / "results/event_database"
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"
OUT = PROJECT_ROOT / "results/gt_fine"

# 目视确认清晰的种子块（id, 等级）
SEEDS = ["gt000", "gt004", "gt006", "gt007", "gt012", "gt014", "gt015",
         "gt016", "gt017", "gt018", "gt020", "gt021", "gt022", "gt025",
         "gt028"]


def main() -> None:
    selection = json.loads(
        (GT_ROOT / "selection.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in selection}
    used = {s["tile"] for s in selection}

    with open(EVENT_DB / "events.csv", encoding="utf-8") as f:
        events = [r for r in csv.DictReader(f) if r["quality"] == "high"]

    cands: list[dict] = []
    n = 0

    def add(scene, y, x, reason, event_id=None, wl=None, prob=None):
        nonlocal n
        stem = f"{scene}_y{y:05d}_x{x:05d}"
        if stem in used or y < 0 or x < 0:
            return
        tile = load_tile(scene, stem)
        if tile is None or not tile_ok(tile[0], scene, y, x):
            return
        used.add(stem)
        cid = f"gt{100 + n:03d}"
        n += 1
        cands.append({
            "id": cid, "source": "scene_event",
            "scene": scene, "tile": stem, "y": y, "x": x,
            "event_id": event_id,
            "cluster": None, "wavelength_m": wl, "direction_deg": None,
            "reason": reason,
        })

    # 1. 种子块 8 邻域
    for sid in SEEDS:
        s = by_id[sid]
        for dy in (-384, 0, 384):
            for dx in (-384, 0, 384):
                if dy == 0 and dx == 0:
                    continue
                add(s["scene"], s["y"] + dy, s["x"] + dx,
                    f"补样：{sid}（已确认清晰）的邻域块")

    # 2. 种子场景 top-mean_prob high 事件
    seed_scenes = sorted({by_id[s]["scene"] for s in SEEDS})
    for scene in seed_scenes:
        es = [e for e in events if e["scene"] == scene]
        es.sort(key=lambda e: -float(e["mean_prob"]))
        for e in es[:6]:
            loc = event_tile(scene, float(e["lon"]), float(e["lat"]))
            if loc is None:
                continue
            stem, y, x = loc
            add(scene, y, x,
                f"补样：{scene[-4:]} top-prob high 事件 "
                f"{e['event_id'][-8:]} λ={e['wavelength_m'] or '-'}m",
                event_id=e["event_id"], wl=e["wavelength_m"])

    (OUT / "backfill2.json").write_text(
        json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cands)} 候选 → {OUT}/backfill2.json")


if __name__ == "__main__":
    main()
