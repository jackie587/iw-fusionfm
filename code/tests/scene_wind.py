"""查场景成像时刻的 ERA5 风速（判读辅助：风速 <2 或 >10 m/s 时条纹不可见）。

用法：python tests/scene_wind.py <场景名前缀，如 20230703T103344>
"""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
scene_prefix = sys.argv[1]

# 从切块 meta 拿场景中心经纬度
tiles = sorted(PROJECT_ROOT.glob(
    f"data/processed/sar_tiles/*{scene_prefix}*/meta/*.json"))
meta = [json.loads(f.read_text()) for f in tiles]
from affine import Affine
from rasterio.transform import xy
lons, lats = [], []
for m in meta[::200]:
    t = Affine(*m["transform"])
    lon, lat = xy(t, m["row"] + 256, m["col"] + 256)
    lons.append(lon)
    lats.append(lat)
clon, clat = float(np.mean(lons)), float(np.mean(lats))

# 场景时刻 → ERA5 最近小时
import re
ts = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})", scene_prefix)
month_file = PROJECT_ROOT / f"data/raw/auxiliary/era5/era5_wind10m_{ts.group(1)}{ts.group(2)}.nc"
ds = xr.open_dataset(month_file)
t_target = np.datetime64(f"{ts.group(1)}-{ts.group(2)}-{ts.group(3)}T{ts.group(4)}:{ts.group(5)}")
sub = ds.sel(longitude=slice(clon - 1, clon + 1))
lat_arr = ds["latitude"].values
lo, hi = (clat - 1, clat + 1) if lat_arr[0] < lat_arr[-1] else (clat + 1, clat - 1)
sub = sub.sel(latitude=slice(lo, hi))
sub = sub.sel(valid_time=t_target, method="nearest")
spd = np.hypot(sub["u10"], sub["v10"])
print(f"场景 {scene_prefix}")
print(f"中心 ~({clon:.1f}E, {clat:.1f}N)，时刻 {t_target}")
print(f"ERA5 风速（±1° 范围）：中位 {float(spd.median()):.1f} m/s，"
      f"min {float(spd.min()):.1f}，max {float(spd.max()):.1f}")
print("解读：<2 m/s 条纹不成像；2~10 适宜；>10 条纹被淹没")
