"""ERA5 CDS 凭据最小验证：1 天 1 小时 2 变量小区域请求。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "data_preprocessing" / "download"))

from download_era5 import _cds_client

client = _cds_client()
out = PROJECT_ROOT / "data/raw/auxiliary/era5/_auth_test.nc"
out.parent.mkdir(parents=True, exist_ok=True)
client.retrieve("reanalysis-era5-single-levels", {
    "product_type": ["reanalysis"],
    "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    "year": ["2023"], "month": ["06"], "day": ["04"],
    "time": ["10:00"],
    "area": [23, 109, 18, 114],
    "data_format": "netcdf",
    "download_format": "unarchived",
}, str(out))
print("OK", out, out.stat().st_size, "bytes")

import xarray as xr
ds = xr.open_dataset(out)
print(ds)
import numpy as np
spd = np.hypot(ds["u10"], ds["v10"])
print(f"风速模长：min={float(spd.min()):.1f} max={float(spd.max()):.1f} m/s")
