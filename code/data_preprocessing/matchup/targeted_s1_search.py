"""反向定向配对：SWOT 有效刈幅 × CDSE S1 全量脚印，找出真正值得下载的场景。

背景（2026-08-31 相交矩阵结论）：现有 15 景 S1 与 68 个 granule 无一可用
配对（唯一窗内对 est_tiles=0.5）——需按刈幅反查 S1 脚印，定向下载。

流程：
1. CDSE OData 检索 AOI（108~116E, 17~24N）2023-08-01~10-15 全部
   S1 IW GRDH 产品（含 Footprint，缓存 s1_footprints_cdse.json）；
2. 逐 S1 场景 × SWOT 抽稀有效点云（swot_footprint_cache.npz）：
   Δt≤6h 且点在多边形内 → 估计配对块数；
3. 输出排名清单 targeted_pairs.json（按估计块数排序）。

用法（在 code/ 目录下）：
    python data_preprocessing/matchup/targeted_s1_search.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from data_preprocessing.download.download_sentinel1 import (
    load_credentials, CDSE_ODATA, CDSE_TOKEN_URL)  # noqa: F401
from utils.logger import get_logger

logger = get_logger("targeted_search")

AOI = (108.0, 17.0, 116.0, 24.0)
START, END = "2023-08-01", "2023-10-15"
MAX_DT_H = 6.0
DECIM = 10
SWOT_CACHE = PROJECT_ROOT / "data/processed/swot_patches/swot_footprint_cache.npz"
S1_CACHE = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/s1_footprints_cdse.json"
OUT = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/targeted_pairs.json"


def parse_time(name: str) -> datetime:
    m = re.search(r"(\d{8}T\d{6})", name)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")


def fetch_s1_footprints() -> list[dict]:
    if S1_CACHE.exists():
        logger.info("S1 脚印缓存命中")
        return json.loads(S1_CACHE.read_text(encoding="utf-8"))

    import requests
    lon1, lat1, lon2, lat2 = AOI
    wkt = (f"POLYGON(({lon1} {lat1},{lon2} {lat1},{lon2} {lat2},"
           f"{lon1} {lat2},{lon1} {lat1}))")
    filt = ("Collection/Name eq 'SENTINEL-1' and "
            "OData.CSC.Intersects(area=geography'SRID=4326;" + wkt + "') and "
            f"ContentDate/Start ge {START}T00:00:00.000Z and "
            f"ContentDate/Start le {END}T23:59:59.999Z")
    items, skip = [], 0
    while True:
        r = requests.get(f"{CDSE_ODATA}/Products", params={
            "$filter": filt, "$top": 200, "$skip": skip,
            "$orderby": "ContentDate/Start asc",
        }, timeout=120)
        r.raise_for_status()
        batch = r.json().get("value", [])
        if not batch:
            break
        items += batch
        skip += len(batch)
        if len(batch) < 200:
            break
    logger.info("CDSE 检索到 %d 个产品（过滤前）", len(items))

    out = []
    for it in items:
        name = it.get("Name", "")
        if "IW_GRDH" not in name or name.endswith(".COG.SAFE"):
            continue  # 与下载脚本口径一致：经典 .SAFE，排除 COG 版
        fp = it.get("Footprint") or ""
        m = re.search(r"POLYGON\s*\(\(([^)]+)\)\)", fp)
        if not m:
            continue
        coords = [[float(v) for v in p.split()]
                  for p in m.group(1).split(",")]
        out.append({"name": name.replace(".SAFE", ""),
                    "time": parse_time(name).isoformat(),
                    "polygon": coords})
    # 同名去重（COG/经典重复已在名称层排除，双保险按 name 去重）
    out = list({o["name"]: o for o in out}.values())
    S1_CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    logger.info("有效 S1 脚印 %d 个 → %s", len(out), S1_CACHE)
    return out


def main():
    from matplotlib.path import Path as MplPath

    s1 = fetch_s1_footprints()
    z = np.load(SWOT_CACHE, allow_pickle=True)
    granules = {n: {"lon": z[f"{n}__lon"], "lat": z[f"{n}__lat"],
                    "time": z[f"{n}__time"].item()} for n in z["names"]}

    rows = []
    for s in s1:
        st = datetime.fromisoformat(s["time"])
        poly = np.array(s["polygon"])
        path = MplPath(poly)
        for gname, g in granules.items():
            gt = datetime.fromisoformat(g["time"])
            dt_h = abs((gt - st).total_seconds()) / 3600.0
            if dt_h > MAX_DT_H:
                continue
            pts = np.stack([g["lon"], g["lat"]], axis=1)
            inside = int(path.contains_points(pts).sum())
            if inside == 0:
                continue
            rows.append({
                "s1": s["name"], "granule": gname,
                "dt_h": round(dt_h, 2),
                "valid_pts_decim": inside,
                "est_valid_pts": inside * DECIM * DECIM,
                "est_tiles_ge50": round(inside * DECIM * DECIM / 400.0, 1),
            })
    rows.sort(key=lambda r: -r["est_tiles_ge50"])
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    logger.info("窗内且真实相交 %d 对 → %s", len(rows), OUT)

    print(f"\n{'S1 场景':<46} {'granule':<30} {'Δt(h)':>6} "
          f"{'pts':>5} {'估计块数':>8}")
    for r in rows[:30]:
        print(f"{r['s1'][:46]:<46} {r['granule'][17:47]:<30} "
              f"{r['dt_h']:>6.2f} {r['valid_pts_decim']:>5} "
              f"{r['est_tiles_ge50']:>8.0f}")


if __name__ == "__main__":
    main()
