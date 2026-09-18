"""SWOT 面片重投影到 SAR 切块网格。

原则（数据输入处理方案七）：保留各自原始分辨率，不拉到 10 m。
做法：把 SWOT 散点/栅格重采样到以 SAR 切块为范围的 ~500 m 栅格，
输出 [3,H',W']（SSHA, |∇SSH|, 有效掩膜）+ 地理配准元数据，
写入 data/processed/swot_patches/。融合时在特征层对齐，不做像素级拉平。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def reproject_to_grid(ch: dict, lon_min: float, lat_min: float,
                      lon_max: float, lat_max: float,
                      res_m: float = 500.0) -> dict:
    """把 SWOT 通道重采样到指定经纬度范围的规则网格（最近邻，不造细节）。

    ch: make_swot_channels 的输出。
    返回 {"ssha","grad","mask": 2D 网格, "meta": {...}}；掩膜 0 的区域值为 0。
    """
    from scipy.spatial import cKDTree

    from utils.geo import meters_per_degree

    lat_mid = (lat_min + lat_max) / 2
    m_lon, m_lat = meters_per_degree(lat_mid)
    nx = max(2, int((lon_max - lon_min) * m_lon / res_m))
    ny = max(2, int((lat_max - lat_min) * m_lat / res_m))

    lon = ch["lon"].ravel()
    lat = ch["lat"].ravel()
    valid = ch["mask"].ravel() > 0

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, nx), np.linspace(lat_min, lat_max, ny))
    out = {}
    if valid.any():
        tree = cKDTree(np.stack([lon[valid], lat[valid]], axis=1))
        dist, idx = tree.query(np.stack([grid_lon.ravel(), grid_lat.ravel()], axis=1))
        # 距离超过一个网格尺寸视为无覆盖（保持掩膜诚实）
        max_deg = (res_m / min(m_lon, m_lat)) * 1.5
        near = dist.reshape(ny, nx) <= max_deg
        idx = idx.reshape(ny, nx)
        ssha = np.where(near, ch["ssha"].ravel()[valid][idx], 0.0)
        grad = np.where(near, ch["grad"].ravel()[valid][idx], 0.0)
        mask = near.astype(np.uint8)
    else:
        ssha = np.zeros((ny, nx), np.float64)
        grad = np.zeros((ny, nx), np.float64)
        mask = np.zeros((ny, nx), np.uint8)

    out["ssha"] = np.nan_to_num(ssha, nan=0.0).astype(np.float32)
    out["grad"] = np.nan_to_num(grad, nan=0.0).astype(np.float32)
    out["mask"] = mask
    out["meta"] = {"lon_min": lon_min, "lat_min": lat_min,
                   "lon_max": lon_max, "lat_max": lat_max,
                   "res_m": res_m, "ny": ny, "nx": nx}
    return out


def save_patch(grid: dict, out_dir: str | Path, name: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.stack([grid["ssha"], grid["grad"], grid["mask"].astype(np.float32)])
    np.save(out / f"{name}.npy", arr)
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(grid["meta"], f, ensure_ascii=False)
