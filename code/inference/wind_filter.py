"""ERA5 风速门控后处理：剔除物理上不可能成像内波条纹的检出。

物理先验（方案既定）：SAR 内波条纹只在 2~10 m/s 风速窗口成像——
<2 m/s 海面过光滑条纹不显影，>10 m/s 被风浪背景淹没。
用 ERA5 逐小时 10m 风场（0.25°）双线性插值到场景网格，
窗口外区域概率置 0。对整景低风（如 0604 景中位 1.5 m/s）等效整景抑制。

注意：ERA5 分辨率粗，不能替代纹理判别（auto_review_wind.py 验证：
人工虚警仅 3/223 落在窗外），它是免费的第一道物理滤网。

用法（在 code/ 目录下）：
    python inference/wind_filter.py \
        --tiles-dir ../data/processed/sar_tiles/<场景>.SAFE \
        --scene-out ../results/scene_eval_v5/<场景名>
产物：scene-out 下 wind_ms.npy（float16 风速场）与 prob_windgated.npy。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ERA5_DIR = PROJECT_ROOT / "data/raw/auxiliary/era5"

WIND_LO, WIND_HI = 2.0, 10.0
STEP = 64  # 风速场按 64px 间隔计算再最近邻放大，ERA5 原生 ~0.25° 无需更密


def scene_transform(tiles_dir: Path) -> Affine:
    metas = sorted((tiles_dir / "meta").glob("*.json"))
    return Affine(*json.loads(metas[0].read_text())["transform"])


def wind_field(tiles_dir: Path, hw: tuple[int, int],
               tstamp: str) -> np.ndarray:
    """场景网格上的 ERA5 10m 风速场 (m/s)。"""
    t = scene_transform(tiles_dir)
    h, w = hw
    ys, xs = np.mgrid[0:h:STEP, 0:w:STEP]
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
    return np.kron(spd, np.ones((STEP, STEP), np.float32))[:h, :w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--scene-out", required=True)
    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir)
    scene_out = Path(args.scene_out)
    prob_file = scene_out / "prob_masked.npy"
    if not prob_file.exists():
        prob_file = scene_out / "prob.npy"
    prob = np.nan_to_num(np.load(prob_file).astype(np.float32), nan=0.0)

    tstamp = re.search(r"\d{8}T\d{6}", tiles_dir.name).group(0)
    spd = wind_field(tiles_dir, prob.shape, tstamp)
    valid = (spd >= WIND_LO) & (spd <= WIND_HI)
    gated = np.where(valid, prob, 0.0)

    np.save(scene_out / "wind_ms.npy", spd.astype(np.float16))
    np.save(scene_out / "prob_windgated.npy", gated.astype(np.float16))
    stats = {"wind_window": [WIND_LO, WIND_HI],
             "sea_wind_median": float(np.median(spd)),
             "valid_frac": float(valid.mean()),
             "area_frac_gt0.6_before": float((prob > 0.6).mean()),
             "area_frac_gt0.6_after": float((gated > 0.6).mean())}
    (scene_out / "wind_gate_stats.json").write_text(
        json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
