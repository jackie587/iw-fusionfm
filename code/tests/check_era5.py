"""ERA5 下载完整性验证：7 个月文件的变量/时间/空间覆盖与风速值域。"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

era5_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "../data/raw/auxiliary/era5")

for f in sorted(era5_dir.glob("era5_wind10m_*.nc")):
    ds = xr.open_dataset(f)
    dims = {k: int(v) for k, v in ds.sizes.items()}
    spd = np.hypot(ds["u10"], ds["v10"])
    print(f"{f.name}: {dims} "
          f"lon[{float(ds.longitude.min())}~{float(ds.longitude.max())}] "
          f"lat[{float(ds.latitude.min())}~{float(ds.latitude.max())}] "
          f"风速 p5={float(spd.quantile(0.05)):.1f} "
          f"中位={float(spd.median()):.1f} "
          f"p95={float(spd.quantile(0.95)):.1f} m/s")
    ds.close()
