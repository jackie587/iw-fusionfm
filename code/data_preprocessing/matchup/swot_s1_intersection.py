"""SWOT 有效刈幅 × S1 场景脚印相交矩阵。

动机（2026-08-28 诊断结论）：GPolygon 元数据重叠 ≠ 有效数据重叠，
必须基于 granule 内 qual==0 的真实有效点计算相交。

流程：
1. 逐 granule 读 lon/lat/qual，取有效点并按 DECIM 抽稀成点云
   （缓存到 swot_footprint_cache.npz，重跑秒级）；
2. S1 场景脚印 = 全部切块 meta transform 的 lon/lat 包围矩形
   （EPSG:4326 轴对齐，矩形即脚印）；
3. 逐对（granule × scene）统计场景矩形内有效点数，附 |Δt|；
4. 输出相交矩阵 JSON + 按"估计配对块数 × Δt 达标"排名的候选清单。

用法（在 code/ 目录下）：
    python data_preprocessing/matchup/swot_s1_intersection.py
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
from utils.logger import get_logger

logger = get_logger("intersection")

SWOT_DIR = PROJECT_ROOT / "data/raw/swot"
TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
CACHE = PROJECT_ROOT / "data/processed/swot_patches/swot_footprint_cache.npz"
OUT = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/intersection_matrix.json"

DECIM = 10            # 点云抽稀步长（250m 产品 → ~2.5km 采样）
MAX_DT_H = 6.0        # 放宽配对窗口
TILE_AREA_KM2 = 25.0  # 512px × ~10m ≈ 5km 见方（粗略，用于估计块数）


def parse_time(name: str) -> datetime:
    m = re.search(r"(\d{8}T\d{6})", name)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")


def build_swot_clouds() -> dict:
    """granule 名 → {"lon": 1D, "lat": 1D, "time": str}（抽稀有效点云）。"""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        logger.info("点云缓存命中（%d granule）", len(z["names"]))
        return {n: {"lon": z[f"{n}__lon"], "lat": z[f"{n}__lat"],
                    "time": z[f"{n}__time"].item()} for n in z["names"]}

    import xarray as xr
    clouds, save = {}, {}
    for nc in sorted(SWOT_DIR.glob("*.nc")):
        ds = xr.open_dataset(nc)
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        qual = (ds["ssh_karin_qual"].values
                if "ssh_karin_qual" in ds else np.zeros_like(lon))
        valid = (qual == 0) & np.isfinite(lon) & np.isfinite(lat)
        # 抽稀：先二维步进再取有效点，保持空间代表性
        v2 = np.zeros_like(valid)
        v2[::DECIM, ::DECIM] = valid[::DECIM, ::DECIM]
        ys, xs = np.nonzero(v2)
        name = nc.stem
        clouds[name] = {"lon": lon[ys, xs].astype(np.float32),
                        "lat": lat[ys, xs].astype(np.float32),
                        "time": parse_time(nc.name).isoformat()}
        save[f"{name}__lon"] = clouds[name]["lon"]
        save[f"{name}__lat"] = clouds[name]["lat"]
        save[f"{name}__time"] = np.array(clouds[name]["time"])
        ds.close()
        logger.info("%s: 有效点云 %d", name[:60], len(ys))
    save["names"] = np.array(list(clouds.keys()))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **save)
    return clouds


def scene_footprints() -> dict:
    """场景名 → {"lon1","lat1","lon2","lat2","time"}。"""
    fps = {}
    for d in sorted(TILES_ROOT.iterdir()):
        metas = list((d / "meta").glob("*.json"))
        if not metas:
            continue
        lons, lats, ts = [], [], None
        for mp in metas:
            m = json.loads(mp.read_text(encoding="utf-8"))
            a, _, lon0, _, b, lat0 = m["transform"]
            size = m["tile_size"]
            lons += [lon0, lon0 + a * size]
            lats += [lat0, lat0 + b * size]
        fps[d.name.replace(".SAFE", "")] = {
            "lon1": min(lons), "lon2": max(lons),
            "lat1": min(lats), "lat2": max(lats),
            "time": parse_time(d.name).isoformat(),
        }
    return fps


def main():
    clouds = build_swot_clouds()
    scenes = scene_footprints()
    logger.info("granule %d × 场景 %d", len(clouds), len(scenes))

    rows = []
    for gname, g in clouds.items():
        gt = datetime.fromisoformat(g["time"])
        for sname, s in scenes.items():
            st = datetime.fromisoformat(s["time"])
            dt_h = abs((gt - st).total_seconds()) / 3600.0
            inside = int(((g["lon"] >= s["lon1"]) & (g["lon"] <= s["lon2"])
                          & (g["lat"] >= s["lat1"]) & (g["lat"] <= s["lat2"])).sum())
            if inside == 0:
                continue
            # 估计配对块数：有效点数×(DECIM²) / 每块满覆盖点数(500m网格400点)
            n_real = inside * DECIM * DECIM
            est_tiles = n_real / 400.0
            rows.append({
                "granule": gname, "scene": sname,
                "dt_h": round(dt_h, 2),
                "valid_pts_decim": inside,
                "est_valid_pts": n_real,
                "est_tiles_ge50": round(est_tiles, 1),
                "in_window": dt_h <= MAX_DT_H,
            })
    rows.sort(key=lambda r: (not r["in_window"], -r["est_tiles_ge50"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    in_win = [r for r in rows if r["in_window"]]
    logger.info("相交非空 %d 对，其中 |Δt|≤%.0fh 的 %d 对 → %s",
                len(rows), MAX_DT_H, len(in_win), OUT)
    print(f"\n{'granule':<44} {'scene':<16} {'Δt(h)':>6} "
          f"{'有效点(抽稀)':>12} {'估计块数':>8}")
    for r in rows[:40]:
        flag = "✓" if r["in_window"] else " "
        print(f"{r['granule'][:44]:<44} {r['scene'][-16:]:<16} "
              f"{r['dt_h']:>6.2f} {r['valid_pts_decim']:>12} "
              f"{r['est_tiles_ge50']:>8.0f} {flag}")


if __name__ == "__main__":
    main()
