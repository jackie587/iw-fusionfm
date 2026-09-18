"""春季（4~5 月）正样本普查四层筛（固化版）。

四层筛（沿用冬季普查口径，见 文档/工作进度记录.txt 第 9 天续三）：
  ① 丛集 AOI：110~115E, 19~21.5N（吕宋海峡西传内波路径）；
  ② 大潮窗口：朔望日（UTC）后第 1~5 天（含）；
  ③ ERA5 丛集区 10/11 UTC（S1 降轨过境附近）区域平均风速 ∈ [2, 10] m/s；
  ④ CDSE 可获取性：Sentinel-1 IW GRDH 景数（--cdse 联网查询，复用
     data_preprocessing/download/download_sentinel1.py 的 _cdse_search，
     检索窗口 --start <日> --end <次日>，与冬季普查口径一致）。

朔望日（UTC，已用公开月相历核实）：
  2024：04-08 新月 18:21 / 04-23 满月 23:49 / 05-08 新月 03:22 / 05-23 满月 13:53
  2023：04-20 新月 / 05-05 满月 / 05-19 新月

用法（在 code/ 目录下）：
    python tests/spring_census_screen.py --year 2024           # 风筛
    python tests/spring_census_screen.py --year 2024 --cdse    # 追加 CDSE 检索
输出：results/scene_eval_poscensus/spring_candidates_<year>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from utils.logger import get_logger  # noqa: E402

logger = get_logger("spring_census_screen")

ERA5_DIR = PROJECT_ROOT / "data/raw/auxiliary/era5"
OUT_DIR = PROJECT_ROOT / "results/scene_eval_poscensus"

AOI = (110.0, 19.0, 115.0, 21.5)   # lon_min, lat_min, lon_max, lat_max
WIND_LO, WIND_HI = 2.0, 10.0
HOURS_UTC = (10, 11)               # S1 降轨过境时刻附近两个时次

# (朔望日, 相位)；大潮窗口 = 朔望日后第 1~5 天（含）
SYZYGY_UTC = {
    2024: [("2024-04-08", "new"), ("2024-04-23", "full"),
           ("2024-05-08", "new"), ("2024-05-23", "full")],
    2023: [("2023-04-20", "new"), ("2023-05-05", "full"),
           ("2023-05-19", "new")],
}


def tide_window_days(year: int) -> dict[date, str]:
    """{候选日: 所属朔望日 ISO}（大潮窗口内逐日）。"""
    days = {}
    for s, _phase in SYZYGY_UTC[year]:
        s0 = date.fromisoformat(s)
        for k in range(1, 6):
            days[s0 + timedelta(days=k)] = s
    return dict(sorted(days.items()))


def aoi_mean_wind(day: date, hour: int) -> float:
    """ERA5 丛集 AOI 区域平均风速 (m/s)：逐格点 hypot(u10,v10) 后区域平均。"""
    nc = ERA5_DIR / f"era5_wind10m_{day:%Y%m}.nc"
    ds = xr.open_dataset(nc)
    sub = ds.sel(valid_time=np.datetime64(f"{day.isoformat()}T{hour:02d}:00"),
                 longitude=slice(AOI[0], AOI[2]),
                 latitude=slice(AOI[3], AOI[1]))  # latitude 降序存储
    spd = np.hypot(sub["u10"], sub["v10"])
    out = float(spd.mean())
    ds.close()
    return out


def daily_wind(day: date) -> float:
    """10 与 11 UTC 两时次区域平均风速的均值。"""
    return float(np.mean([aoi_mean_wind(day, h) for h in HOURS_UTC]))


def cdse_query(day: date) -> list[dict]:
    """复用 download_sentinel1 的 CDSE 检索：--start <日> --end <次日>。"""
    from data_preprocessing.download.download_sentinel1 import _cdse_search
    wkt = (f"POLYGON(({AOI[0]} {AOI[1]},{AOI[2]} {AOI[1]},{AOI[2]} {AOI[3]},"
           f"{AOI[0]} {AOI[3]},{AOI[0]} {AOI[1]}))")
    return _cdse_search(wkt, day.isoformat(),
                        (day + timedelta(days=1)).isoformat(), 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024, choices=sorted(SYZYGY_UTC))
    ap.add_argument("--cdse", action="store_true",
                    help="对风筛通过的候选日追加 CDSE 景数检索")
    args = ap.parse_args()

    out_file = OUT_DIR / f"spring_candidates_{args.year}.json"
    if out_file.exists():
        rec = json.loads(out_file.read_text(encoding="utf-8"))
    else:
        rec = {"year": args.year,
               "aoi": list(AOI),
               "wind_window_ms": [WIND_LO, WIND_HI],
               "syzygy_utc": [{"date": s, "phase": p}
                              for s, p in SYZYGY_UTC[args.year]],
               "candidates": []}
    by_date = {c["date"]: c for c in rec["candidates"]}

    for day, s in tide_window_days(args.year).items():
        key = day.isoformat()
        cand = by_date.get(key)
        if cand is None:
            wind = daily_wind(day)
            cand = {"date": key,
                    "wind_ms": round(wind, 2),
                    "in_wind_window": bool(WIND_LO <= wind <= WIND_HI),
                    "tide_window_of": s}
            by_date[key] = cand
            logger.info("%s 风 %.2f m/s %s（大潮窗口 of %s）", key, wind,
                        "通过" if cand["in_wind_window"] else "剔除", s)
        if args.cdse and cand["in_wind_window"] and "cdse_scenes" not in cand:
            items = cdse_query(day)
            cand["cdse_scenes"] = len(items)
            cand["cdse_names"] = [it["name"] for it in items]
            logger.info("%s CDSE %d 景", key, len(items))

    rec["candidates"] = [by_date[k] for k in sorted(by_date)]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    n_ok = sum(c["in_wind_window"] for c in rec["candidates"])
    logger.info("候选 %d 天，风筛通过 %d 天 → %s",
                len(rec["candidates"]), n_ok, out_file)


if __name__ == "__main__":
    main()
