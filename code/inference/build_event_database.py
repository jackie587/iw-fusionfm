"""事件数据库构建：v7 场景概率图 → 波峰线矢量 → 波包事件 → ERA5 风场关联。

对 results/scene_eval_v7sam/ 下每景：
  prob(>0.5) → 连通域过滤 → 骨架化 → 波峰线拟合（复用 inference/postprocess.py）
  骨架距离变换聚类（4 倍降采样）→ 波包事件；事件属性：质心经纬度、UTC 时间、
  波峰线条数、总长度(km)、传播方位角(地理)、波长(m，相邻波峰间距中位数)、
  ERA5 10m 风速与风窗标记（2~10 m/s）。

产物（results/event_database/）：
  events.csv                每事件一行
  crest_lines/<场景>.geojson 波峰线矢量（WGS84 LineString）
  summary.json              总量与分布统计

用法（在 code/ 目录下）：
    python inference/build_event_database.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import filter_small_components, skeletonize_mask, fit_crest_lines
from wind_filter import scene_transform, WIND_LO, WIND_HI, ERA5_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROOT = PROJECT_ROOT / "results/scene_eval_v7sam"
TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
OUT_ROOT = PROJECT_ROOT / "results/event_database"

THRESH = 0.5
MIN_AREA = 20          # 碎斑过滤（px）
MIN_CREST_PX = 30      # 波峰线最短长度（px）
GROUP_RADIUS = 120     # 波包聚类半径（px，≈1.2 km@10m/px）
GROUP_DS = 4           # 聚类降采样因子（控制距离变换内存）
WIND_STEP = 64         # ERA5 风速采样间隔（px）
WAVELEN_RANGE = (100.0, 3000.0)  # 物理合理波长区间（m），之外记 NaN
# 质量分级校准（2026-08-31 QA 抽验结论，见工作进度记录）：
# λ∈[100,1500] 之外不可靠（骨架碎裂伪间距 / 跨波包误配）；
# 面积>2000 km² 为聚类合并的巨块，λ 无意义。
QUALITY_WAVELEN = (100.0, 1500.0)
QUALITY_MAX_AREA_KM2 = 2000.0


def assign_quality(e: dict) -> str:
    """事件质量分级：high（≥3 峰 + λ 物理有效 + 风窗内 + 非巨块）/
    medium（≥2 峰 + 风窗内）/ low（其余碎包）。"""
    if e["n_crest"] >= 2 and e["wind_flag"] == "in_window":
        try:
            wl = float(e["wavelength_m"])
        except (TypeError, ValueError):
            wl = float("nan")
        if (e["n_crest"] >= 3
                and QUALITY_WAVELEN[0] <= wl <= QUALITY_WAVELEN[1]
                and float(e["area_km2"]) <= QUALITY_MAX_AREA_KM2):
            return "high"
        return "medium"
    return "low"

DEG_LAT_M = 110574.0   # 1° 纬度 ≈ m


def px_meter(a: float, e: float, lat: float) -> tuple[float, float]:
    """像素尺寸（m/px）：(x 方向, y 方向)。"""
    return abs(a) * 111320.0 * math.cos(math.radians(lat)), abs(e) * DEG_LAT_M


def image_vec_to_azimuth(vy: float, vx: float, mx: float, my: float) -> float:
    """图像坐标方向向量 (vy,vx) → 地理方位角（北 0°，顺时针，0~360）。"""
    dn, de = vy * my, vx * mx
    return math.degrees(math.atan2(de, dn)) % 360.0


def mean_direction_180(dirs: list[float]) -> float:
    """180° 模糊方向的圆均值（倍角法），返回 0~180。"""
    s = sum(math.sin(math.radians(2 * d)) for d in dirs)
    c = sum(math.cos(math.radians(2 * d)) for d in dirs)
    return (math.degrees(math.atan2(s, c)) / 2.0) % 180.0


def group_packets(skel: np.ndarray, radius: int) -> np.ndarray:
    """骨架聚类（粗网格）：距骨架 ≤radius 的连通区 = 同一波包事件。
    返回降采样 GROUP_DS 倍的标签图，用 (y//DS, x//DS) 索引。"""
    from scipy import ndimage
    from skimage.measure import block_reduce
    s = block_reduce(skel, (GROUP_DS, GROUP_DS), np.max)
    near = ndimage.distance_transform_edt(~(s > 0)) <= radius / GROUP_DS
    lab, _ = ndimage.label(near)
    return lab


def wind_grid(tiles_dir: Path, hw: tuple[int, int], tstamp: str) -> np.ndarray:
    """粗网格（WIND_STEP px）ERA5 10m 风速场 (m/s)。"""
    import xarray as xr
    t = scene_transform(tiles_dir)
    h, w = hw
    ys, xs = np.mgrid[0:h:WIND_STEP, 0:w:WIND_STEP]
    lons = t.c + xs * t.a + ys * t.b
    lats = t.f + xs * t.d + ys * t.e
    ts = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})", tstamp)
    nc = ERA5_DIR / f"era5_wind10m_{ts.group(1)}{ts.group(2)}.nc"
    ds = xr.open_dataset(nc).sel(
        valid_time=np.datetime64(
            f"{ts.group(1)}-{ts.group(2)}-{ts.group(3)}"
            f"T{ts.group(4)}:{ts.group(5)}"), method="nearest")
    pts = ds.interp(longitude=xr.DataArray(lons.ravel(), dims="p"),
                    latitude=xr.DataArray(lats.ravel(), dims="p"))
    spd = np.hypot(pts["u10"].values, pts["v10"].values).reshape(lons.shape)
    ds.close()
    return spd


def process_scene(scene_dir: Path) -> tuple[list[dict], list[dict]]:
    """处理一景，返回 (事件记录列表, 波峰线 geojson feature 列表)。"""
    from skimage.measure import block_reduce
    scene = scene_dir.name
    tiles_dir = TILES_ROOT / f"{scene}.SAFE"
    if not tiles_dir.exists():
        print(f"[skip] {scene}：无切块目录", flush=True)
        return [], []
    pf = scene_dir / "prob_masked.npy"
    if not pf.exists():
        pf = scene_dir / "prob.npy"
    if not pf.exists():
        print(f"[skip] {scene}：无概率图", flush=True)
        return [], []
    prob = np.nan_to_num(np.load(pf).astype(np.float32), nan=0.0)

    t = scene_transform(tiles_dir)
    ts = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", scene)
    iso_time = (f"{ts.group(1)}-{ts.group(2)}-{ts.group(3)}"
                f"T{ts.group(4)}:{ts.group(5)}:{ts.group(6)}Z")
    # 场景中心纬度，用于 m/px 换算
    cy, cx = prob.shape[0] // 2, prob.shape[1] // 2
    lat0 = t.f + cx * t.d + cy * t.e
    mx, my = px_meter(t.a, t.e, lat0)

    mask = filter_small_components((prob > THRESH).astype(np.uint8), MIN_AREA)
    if not mask.any():
        return [], []
    # 裁到掩膜包围盒再骨架化（掩膜通常很稀疏，全画布骨架化是分钟级开销）
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    y0 = max(rows[0] - 256, 0); y1 = min(rows[-1] + 256, mask.shape[0])
    x0 = max(cols[0] - 256, 0); x1 = min(cols[-1] + 256, mask.shape[1])
    mask_c_img = mask[y0:y1, x0:x1]
    skel = skeletonize_mask(mask_c_img)
    lines = fit_crest_lines(skel, MIN_CREST_PX)
    if not lines:
        return [], []
    # 坐标偏移回全画布
    for ln in lines:
        ln["points"] = [(y + y0, x + x0) for y, x in ln["points"]]
    pkt = group_packets(skel, GROUP_RADIUS)
    mask_c = block_reduce(mask_c_img, (GROUP_DS, GROUP_DS), np.max)

    # ERA5 风速场（整景粗网格一次，事件质心最近邻采样）
    spd = wind_grid(tiles_dir, prob.shape, scene)

    # 波峰线 → 事件分组（按线内像素众数标签）
    groups: dict[int, list[dict]] = {}
    feats: list[dict] = []
    for ln in lines:
        pts = np.array(ln["points"])
        gi = pkt[(pts[:, 0] - y0) // GROUP_DS, (pts[:, 1] - x0) // GROUP_DS]
        gid = int(np.bincount(gi.ravel()).argmax())
        ln["gid"] = gid
        groups.setdefault(gid, []).append(ln)

    events: list[dict] = []
    for gid, glines in sorted(groups.items()):
        allpts = np.concatenate([np.array(l["points"]) for l in glines])
        ey, ex = allpts.mean(axis=0)
        lon = t.c + ex * t.a + ey * t.b
        lat = t.f + ex * t.d + ey * t.e
        wy, wx = int(ey) // WIND_STEP, int(ex) // WIND_STEP
        wind = float(spd[min(wy, spd.shape[0] - 1),
                         min(wx, spd.shape[1] - 1)])

        # 传播方向：各线方向（图像系）圆均值 → 地理方位角
        dirs = [l["direction_deg"] for l in glines]
        d_img = mean_direction_180(dirs)
        az = image_vec_to_azimuth(math.cos(math.radians(d_img)),
                                  math.sin(math.radians(d_img)), mx, my)

        # 波长：波峰质心投影到传播轴，相邻间距中位数
        wavelength = float("nan")
        if len(glines) >= 2:
            ax = np.array([math.cos(math.radians(d_img)),
                           math.sin(math.radians(d_img))])
            cents = np.array([np.array(l["points"]).mean(axis=0)
                              for l in glines])
            s = np.sort(cents @ ax)
            dpx = np.diff(s)
            dpx = dpx[dpx > 10]  # 去掉同线碎裂产生的伪间距
            if len(dpx):
                m_px = math.hypot(ax[0] * my, ax[1] * mx)
                w = float(np.median(dpx) * m_px)
                if WAVELEN_RANGE[0] <= w <= WAVELEN_RANGE[1]:
                    wavelength = w

        crest_km = sum(l["length_px"] for l in glines) * (mx + my) / 2 / 1000.0
        ys, xs = allpts[:, 0], allpts[:, 1]
        pc = pkt == gid
        area_km2 = float((pc & (mask_c > 0)).sum()) \
            * (GROUP_DS * mx) * (GROUP_DS * my) / 1e6
        wind_flag = ("in_window" if WIND_LO <= wind <= WIND_HI
                     else ("low" if wind < WIND_LO else "high"))
        eid = f"{scene[:15]}_{ts.group(1)}{ts.group(2)}{ts.group(3)}_E{gid:03d}"
        events.append({
            "event_id": eid, "scene": scene, "time_utc": iso_time,
            "lon": round(lon, 5), "lat": round(lat, 5),
            "n_crest": len(glines),
            "crest_length_km": round(crest_km, 2),
            "direction_deg": round(az, 1),
            "wavelength_m": (round(wavelength, 0)
                             if not math.isnan(wavelength) else ""),
            "mean_prob": round(float(prob[ys, xs].mean()), 3),
            "wind_ms": round(wind, 2), "wind_flag": wind_flag,
            "bbox_lon": f"{t.c + xs.min()*t.a:.4f},{t.c + xs.max()*t.a:.4f}",
            "bbox_lat": f"{t.f + ys.min()*t.e:.4f},{t.f + ys.max()*t.e:.4f}",
            "area_km2": round(area_km2, 2),
        })
        for li, l in enumerate(glines):
            pts = np.array(l["points"])
            order = np.argsort(pts @ np.array(
                [math.cos(math.radians(l["crest_angle_deg"])),
                 math.sin(math.radians(l["crest_angle_deg"]))]))
            coords = [[round(t.c + float(pts[i, 1]) * t.a, 6),
                       round(t.f + float(pts[i, 0]) * t.e, 6)]
                      for i in order]
            feats.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"event_id": eid, "crest_idx": li,
                               "length_px": l["length_px"]},
            })
    return events, feats


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "crest_lines").mkdir(exist_ok=True)
    cache_dir = OUT_ROOT / "scene_events"
    cache_dir.mkdir(exist_ok=True)
    all_events: list[dict] = []
    for scene_dir in sorted(SCENE_ROOT.iterdir()):
        if not scene_dir.is_dir():
            continue
        cache = cache_dir / f"{scene_dir.name}.json"
        if cache.exists():
            events = json.loads(cache.read_text())
            all_events.extend(events)
            print(f"[cache] {scene_dir.name}: {len(events)} 事件", flush=True)
            continue
        events, feats = process_scene(scene_dir)
        cache.write_text(json.dumps(events))
        if feats:
            gj = {"type": "FeatureCollection", "features": feats}
            (OUT_ROOT / "crest_lines" / f"{scene_dir.name}.geojson").write_text(
                json.dumps(gj))
        all_events.extend(events)
        print(f"[ok] {scene_dir.name}: {len(events)} 事件, "
              f"{len(feats)} 波峰线", flush=True)

    if not all_events:
        print("无任何事件")
        return
    for e in all_events:
        e["quality"] = assign_quality(e)
    cols = list(all_events[0].keys())
    with open(OUT_ROOT / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_events)

    wl = [e["wavelength_m"] for e in all_events if e["wavelength_m"] != ""]
    summary = {
        "n_scenes": len({e["scene"] for e in all_events}),
        "n_events": len(all_events),
        "n_events_in_wind_window": sum(
            e["wind_flag"] == "in_window" for e in all_events),
        "wavelength_median_m": (float(np.median(wl)) if wl else None),
        "time_span": [min(e["time_utc"] for e in all_events),
                      max(e["time_utc"] for e in all_events)],
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"→ {OUT_ROOT}/events.csv, crest_lines/*.geojson, summary.json")


if __name__ == "__main__":
    main()
