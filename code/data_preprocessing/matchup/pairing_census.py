"""真实元数据的 S1×SWOT 配对普查：直接查 CDSE/CMR 元数据并跑 match()。

验证 spatiotemporal_match.py 在真实轨道数据上的表现，
同时评估目标 AOI 的配对机会密度（决定 L2 配对集规模上限）。

用法（在 code/ 目录下）：
    python data_preprocessing/matchup/pairing_census.py \
        --aoi "109,18,114,23" --start 2023-09-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "data_preprocessing" / "download"))

from data_preprocessing.matchup.spatiotemporal_match import match
from utils.logger import get_logger

logger = get_logger("pairing_census")


def fetch_s1(aoi: str, start: str, end: str) -> list[dict]:
    from download_sentinel1 import _cdse_search

    lon1, lat1, lon2, lat2 = [float(v) for v in aoi.split(",")]
    wkt = (f"POLYGON(({lon1} {lat1},{lon2} {lat1},{lon2} {lat2},"
           f"{lon1} {lat2},{lon1} {lat1}))")
    items = _cdse_search(wkt, start, end, 500)
    return [{"path": it["name"], "time": it["start"][:19],
             "lon_min": lon1, "lat_min": lat1,
             "lon_max": lon2, "lat_max": lat2} for it in items]


def fetch_swot(aoi: str, start: str, end: str,
               short_name: str = "SWOT_L2_LR_SSH_EXPERT_2.0") -> list[dict]:
    import earthaccess

    lon1, lat1, lon2, lat2 = [float(v) for v in aoi.split(",")]
    granules = earthaccess.search_data(
        short_name=short_name,
        bounding_box=(lon1, lat1, lon2, lat2),
        temporal=(start, end))
    out = []
    for g in granules:
        umm = g["umm"]
        t = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        geom = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        # SWOT 轨道条带用 GPolygon 表示，取其经纬度包围盒做粗匹配即可
        lons, lats = [], []
        for poly in geom.get("GPolygons", []):
            for pt in poly["Boundary"]["Points"]:
                lons.append(pt["Longitude"])
                lats.append(pt["Latitude"])
        for rect in geom.get("BoundingRectangles", []):
            lons += [rect["WestBoundingCoordinate"], rect["EastBoundingCoordinate"]]
            lats += [rect["SouthBoundingCoordinate"], rect["NorthBoundingCoordinate"]]
        if not lons:
            continue
        # granule 是半轨条带，包围盒巨大；用与查询 AOI 的交集作为有效范围
        out.append({"path": g["meta"]["native-id"], "time": t[:19],
                    "lon_min": max(min(lons), lon1), "lat_min": max(min(lats), lat1),
                    "lon_max": min(max(lons), lon2), "lat_max": min(max(lats), lat2)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="109,18,114,23")
    ap.add_argument("--start", default="2023-09-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--swot-short-name", default="SWOT_L2_LR_SSH_EXPERT_2.0",
                    help="科学期用 EXPERT_2.0；calval 期（2023-03~07）"
                         "EXPERT 无 granule，需用 SWOT_L2_LR_SSH_2.0")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/pairs_census.json"))
    args = ap.parse_args()

    s1 = fetch_s1(args.aoi, args.start, args.end)
    sw = fetch_swot(args.aoi, args.start, args.end, args.swot_short_name)
    logger.info("%s ~ %s：S1 %d 景，SWOT %d 个 granule",
                args.start, args.end, len(s1), len(sw))

    pairs = match(s1, sw)
    strict = [p for p in pairs if p["type"] == "strict"]
    relaxed = [p for p in pairs if p["type"] == "relaxed"]
    logger.info("配对：严格 %d 对，放宽 %d 对", len(strict), len(relaxed))
    for p in pairs[:20]:
        logger.info("  [%s] Δt=%.0f min  S1=%s  SWOT=%s", p["type"],
                    p["dt_min"], p["s1"][:44], p["swot"][:44])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"s1": s1, "swot": sw, "pairs": pairs},
                  f, ensure_ascii=False, indent=2)
    logger.info("写出 → %s", out)


if __name__ == "__main__":
    main()
