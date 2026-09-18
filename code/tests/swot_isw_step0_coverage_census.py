"""事件距离级 SWOT 覆盖普查：每个 event_day × 每个 Expert granule，
计算事件到刈幅有效像元的最近距离分布（bbox 重叠会虚报覆盖）。

输出 results/swot_isw_detection/step0_coverage_census.json
用法（code/ 目录下）：python tests/swot_isw_step0_coverage_census.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

from data_preprocessing.swot_preprocess.quality_control import load_and_qc

SWOT_DIR = PROJECT_ROOT / "data/raw/swot"
EVENTS_CSV = PROJECT_ROOT / "results/event_database/events.csv"
OUT = PROJECT_ROOT / "results/swot_isw_detection"


def parse_t(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def main() -> None:
    days: dict[str, dict] = {}
    with open(EVENTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r["time_utc"][:10]
            e = days.setdefault(d, {"lon": [], "lat": [], "t": []})
            e["lon"].append(float(r["lon"]))
            e["lat"].append(float(r["lat"]))
            e["t"].append(r["time_utc"])
    for d, e in days.items():
        e["lon"] = np.array(e["lon"]); e["lat"] = np.array(e["lat"])
        e["t_mid"] = parse_t(sorted(e["t"])[len(e["t"]) // 2])

    report = []
    files = sorted(SWOT_DIR.glob("*.nc"))
    for i, fp in enumerate(files):
        m = re.search(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_", fp.name)
        t0 = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc)
        t1 = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc)
        swot_t = t0 + (t1 - t0) / 2
        try:
            qc = load_and_qc(str(fp))
        except Exception as exc:  # 坏文件不中断普查
            print(f"[err] {fp.name}: {exc}")
            continue
        lon = qc["lon"].ravel().astype(np.float64)
        lat = qc["lat"].ravel().astype(np.float64)
        ok = (qc["mask"].ravel() > 0) & np.isfinite(lon) & np.isfinite(lat)
        tree = cKDTree(np.stack([lon[ok], lat[ok]], axis=1))
        for d, e in days.items():
            pts = np.stack([e["lon"], e["lat"]], axis=1)
            dist, _ = tree.query(pts)
            km = dist * 111.0
            dt_h = (swot_t - e["t_mid"]).total_seconds() / 3600.0
            if abs(dt_h) > 36:      # 只看同日/邻日
                continue
            n5 = int((km <= 5).sum()); n10 = int((km <= 10).sum())
            if n5 or n10:
                report.append({
                    "day": d, "granule": fp.name, "swot_t":
                    swot_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "dt_h": round(dt_h, 2), "n_events": len(km),
                    "n_within_5km": n5, "n_within_10km": n10,
                    "dist_p50_km": round(float(np.percentile(km, 50)), 1)})
        print(f"[{i + 1}/{len(files)}] {fp.name[:60]}")
    report.sort(key=lambda r: (r["day"], -r["n_within_5km"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "step0_coverage_census.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"→ {OUT}/step0_coverage_census.json")
    for r in report:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
