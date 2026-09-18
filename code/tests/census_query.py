"""多季节负样本普查（round5_census）：CDSE 检索 + 按热点区重叠率选景 + 选择性下载。

与 download_sentinel1.py 的区别：检索结果带 GeoFootprint，计算与内波高发区
（110~113E, 19~21.5N）的 bbox 重叠率供选景；--download 只下载 --pick 指定的景
（CLI 原版只能整段时间窗全下）。

用法（在 code/ 目录下）：
    python tests/census_query.py --start 2023-10-01 --end 2023-11-30          # 只列出
    python tests/census_query.py --start 2023-10-01 --end 2023-11-30 \
        --pick 050629 050658 --download                                       # 按绝对轨道号下载
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "data_preprocessing/download"))

import requests  # noqa: E402

from download_sentinel1 import (  # noqa: E402
    CDSE_ODATA, _cdse_download, _cdse_token,
)
from utils.logger import get_logger  # noqa: E402

logger = get_logger("census_query")

AOI = "POLYGON((109 18,114 18,114 23,109 23,109 18))"
HOTSPOT = (110.0, 19.0, 113.0, 21.5)  # 内波高发区陆坡带


def search(start: str, end: str):
    filt = ("Collection/Name eq 'SENTINEL-1' and "
            "OData.CSC.Intersects(area=geography'SRID=4326;" + AOI + "') and "
            f"ContentDate/Start ge {start}T00:00:00.000Z and "
            f"ContentDate/Start le {end}T23:59:59.999Z")
    r = requests.get(f"{CDSE_ODATA}/Products", params={
        "$filter": filt, "$top": 100, "$orderby": "ContentDate/Start asc",
        "$select": "Id,Name,ContentLength,ContentDate,GeoFootprint",
    }, timeout=120)
    r.raise_for_status()
    items = []
    seen = set()
    for p in r.json().get("value", []):
        n = p["Name"]
        if "_IW_GRDH_" not in n or n.endswith("_COG.SAFE"):
            continue
        key = n[:-5].rsplit("_", 1)[0]  # 采集标识（去校验码）
        if key in seen:
            continue
        seen.add(key)
        coords = p["GeoFootprint"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        ix = max(0.0, min(max(lons), HOTSPOT[2]) - max(min(lons), HOTSPOT[0]))
        iy = max(0.0, min(max(lats), HOTSPOT[3]) - max(min(lats), HOTSPOT[1]))
        tot = max((max(lons) - min(lons)) * (max(lats) - min(lats)), 1e-9)
        items.append({
            "id": p["Id"], "name": n, "size": p.get("ContentLength"),
            "start": p.get("ContentDate", {}).get("Start"),
            "orbit": n[49:55],
            "bbox": (min(lons), min(lats), max(lons), max(lats)),
            "hotspot_ov": ix * iy / tot,
        })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--pick", nargs="*", default=None,
                    help="要下载的绝对轨道号（如 050629）或场景名片段")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="../data/raw/sentinel1")
    args = ap.parse_args()

    items = search(args.start, args.end)
    logger.info("检索到 %d 景 IW GRDH（去 COG 重）", len(items))
    for it in items:
        b = it["bbox"]
        logger.info("  %s | 轨道%s | bbox[%.1f,%.1f,%.1f,%.1f] | 热点重叠 %d%% | %.2f GB",
                    it["name"][17:41], it["orbit"], b[0], b[1], b[2], b[3],
                    round(it["hotspot_ov"] * 100), (it["size"] or 0) / 1e9)
    if not args.download:
        return
    picks = [it for it in items
             if any(s in it["orbit"] or s in it["name"] for s in (args.pick or []))]
    if not picks:
        logger.error("--pick 未匹配到任何景")
        sys.exit(1)
    logger.info("将下载 %d 景", len(picks))
    _cdse_download(picks, Path(args.out), _cdse_token())


if __name__ == "__main__":
    main()
