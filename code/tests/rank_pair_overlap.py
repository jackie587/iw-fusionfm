"""科学期 34 对放宽配对的刈幅×内波高发区重叠排名（CMR 元数据，免下载）。

内波高发子区：110~113E, 19~21.5N（0930 景真实波场所在南海东北部深水区）。
对每个配对取 SWOT granule 的 GPolygon（元数据），计算其与高发子区的
相交面积占比；结合 Δt 给出优先级排序，指导后续数据下载。

用法（在 code/ 目录下）：python tests/rank_pair_overlap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

HOTSPOT = (110.0, 19.0, 113.0, 21.5)  # lon1, lat1, lon2, lat2

from utils.logger import get_logger  # noqa: E402
logger = get_logger("rank_pair_overlap")


def gpolygon_of(native_id: str):
    """earthaccess 按 native-id 查 granule 元数据，返回 shapely Polygon。"""
    import earthaccess
    from shapely.geometry import Polygon
    res = earthaccess.search_data(
        short_name="SWOT_L2_LR_SSH_EXPERT_2.0",
        granule_name=native_id + "*")
    if not res:
        return None
    umm = res[0]["umm"]
    polys = []
    try:
        gs = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        for g in gs.get("GPolygons", []):
            bnd = g["Boundary"]["Points"]
            pts = [(p["Longitude"], p["Latitude"]) for p in bnd]
            polys.append(Polygon(pts))
    except KeyError:
        return None
    if not polys:
        return None
    from shapely.ops import unary_union
    return unary_union(polys)


def main():
    from shapely.geometry import box
    hotspot = box(*HOTSPOT)
    census = json.loads(
        (PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/pairs_census.json")
        .read_text())
    # 每个 SWOT pass 只查一次
    rows = []
    cache = {}
    for p in census["pairs"]:
        sw = p["swot"].replace("_swot", "")
        if sw not in cache:
            try:
                cache[sw] = gpolygon_of(sw)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s 元数据查询失败: %s", sw[:40], e)
                cache[sw] = None
        poly = cache[sw]
        if poly is None:
            continue
        inter = poly.intersection(hotspot).area
        frac = inter / hotspot.area
        rows.append({"s1": p["s1"], "swot": sw, "dt_min": p["dt_min"],
                     "hotspot_overlap_frac": round(frac, 3)})
        print(f"{p['s1'][17:32]} ↔ {sw[19:38]} Δt={p['dt_min']:5.0f}min "
              f"高发区重叠 {frac*100:4.1f}%")

    rows.sort(key=lambda r: (-r["hotspot_overlap_frac"], r["dt_min"]))
    out = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/pair_overlap_rank.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print("\n== 排名前 8 ==")
    for r in rows[:8]:
        print(f"  重叠{r['hotspot_overlap_frac']*100:4.1f}% Δt={r['dt_min']:4.0f}min "
              f"{r['s1'][17:32]} ↔ {r['swot'][19:38]}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
