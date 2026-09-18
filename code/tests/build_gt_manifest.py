"""生成 gt_fine/manifest.json：每块的来源、地理范围、获取时间、标注口径。

用法（code/ 目录下）：python tests/build_gt_manifest.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
L2_ROOT = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"

SCENE_RE = re.compile(
    r"S1A_IW_GRDH_1SDV_(\d{8}T\d{6})_(\d{8}T\d{6})_.*")


def scene_geo(scene: str):
    m = json.loads(next((TILES_ROOT / f"{scene}.SAFE" / "meta").glob(
        "*_y00000_x00000.json")).read_text(encoding="utf-8"))
    a, _, lon0, _, b, lat0 = m["transform"]
    return lon0, lat0, a, b


def main() -> None:
    final = json.loads((GT_ROOT / "final_list.json").read_text(
        encoding="utf-8"))
    l2_man = {m["tile"]: m for m in json.loads(
        (L2_ROOT / "manifest_all.json").read_text(encoding="utf-8"))}

    entries = []
    for s in final:
        scene = s["scene"]
        lon0, lat0, a, b = scene_geo(scene)
        x0, y0 = s["x"], s["y"]
        lon_w = lon0 + x0 * a
        lon_e = lon0 + (x0 + 512) * a
        lat_s = lat0 + y0 * b
        lat_n = lat0 + (y0 + 512) * b
        t = SCENE_RE.match(scene)
        l2 = l2_man.get(s["tile"])
        notes = []
        if s["id"] == "gt048":
            notes.append("右侧含海岸/陆地")
        if s["id"] == "gt052":
            notes.append("底部含亮目标（疑似陆地/船只）")
        if s["id"] == "gt053":
            notes.append("含暗色条状油污/低风带（非内波），难负样本")
        entries.append({
            "id": s["id"],
            "image": f"images/{s['id']}.npy",
            "image_png": f"images_png/{s['id']}.png",
            "mask": f"masks/{s['id']}.png",
            "label": s["label"],           # positive / negative
            "grade": s["grade"],           # A 清晰 / B 较弱 / neg
            "source": s["source"],
            "scene": scene,
            "acq_start_utc": t.group(1) if t else None,
            "tile": s["tile"], "tile_y": y0, "tile_x": x0,
            "lon_range": [round(lon_w, 5), round(lon_e, 5)],
            "lat_range": [round(lat_s, 5), round(lat_n, 5)],
            "cluster": "west" if (lon_w + lon_e) / 2 < 109.3 else "east",
            "event_id": s.get("event_id"),
            "wavelength_m": s.get("wavelength_m"),
            "boxes_xy": s.get("boxes", []),
            "annotation": ("packet-box 波包级框填充（AI 目视圈定，"
                           "上下文窗口辅助）" if s["label"] == "positive"
                           else "全零（AI 目视确认无波）"),
            "l2_file": l2["file"] if l2 else None,
            "swot_available": bool(l2),
            "select_reason": s.get("reason", ""),
            "notes": "; ".join(notes),
        })
    (GT_ROOT / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8")
    n_pos = sum(1 for e in entries if e["label"] == "positive")
    n_swot = sum(1 for e in entries if e["swot_available"])
    from collections import Counter
    print(f"{len(entries)} 张（正 {n_pos} / 负 {len(entries) - n_pos}），"
          f"含 SWOT {n_swot} 张")
    print("丛集:", Counter(e["cluster"] for e in entries))
    print("场景:", Counter(e["scene"][-4:] for e in entries))
    print("分级:", Counter(e["grade"] for e in entries))


if __name__ == "__main__":
    main()
