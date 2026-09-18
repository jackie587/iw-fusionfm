"""SWOT Unsmoothed 250m granule 结构勘察（任务 1）。

打印 left/right 半刈幅的变量、时空覆盖、AOI 内有效点统计。
用法（code/ 目录下）：
    python tests/swot_isw_step1_inspect.py [granule_nc]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent

DEFAULT_NC = (PROJECT_ROOT / "data/raw/swot_unsmoothed"
              / "SWOT_L2_LR_SSH_Unsmoothed_002_271_20230820T175019"
              "_20230820T184101_PGC0_01.nc")
AOI = (109.0, 18.0, 117.0, 23.0)  # lon1, lat1, lon2, lat2 南海北部


def main() -> None:
    nc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NC
    print(f"granule: {nc_path.name}")
    for side in ("left", "right"):
        ds = xr.open_dataset(nc_path, group=side)
        lon = ds["longitude"].values.astype(np.float64)
        lat = ds["latitude"].values.astype(np.float64)
        t = ds["time"].values
        print(f"\n== {side} 半刈幅 ==")
        print(f"  形状 num_lines×num_pixels = {lon.shape}")
        print(f"  时间范围 {str(t[0])[:19]} ~ {str(t[-1])[:19]}")
        print(f"  经度范围 {np.nanmin(lon):.2f} ~ {np.nanmax(lon):.2f}"
              f"  纬度范围 {np.nanmin(lat):.2f} ~ {np.nanmax(lat):.2f}")

        inb = ((lon >= AOI[0]) & (lon <= AOI[2])
               & (lat >= AOI[1]) & (lat <= AOI[3]))
        print(f"  AOI {AOI} 内像素数: {inb.sum():,} / {lon.size:,}"
              f"  ({100 * inb.sum() / lon.size:.1f}%)")
        if not inb.any():
            ds.close()
            continue
        rows = np.nonzero(inb.any(axis=1))[0]
        print(f"  AOI 覆盖沿轨行 {rows[0]}~{rows[-1]}"
              f"（{str(t[rows[0]])[:19]} ~ {str(t[rows[-1]])[:19]}）")

        ssh = ds["ssh_karin_2"].values.astype(np.float64)
        qual = ds["ssh_karin_2_qual"].values
        mss = ds["mean_sea_surface_cnescls"].values.astype(np.float64)
        fill = ds["ssh_karin_2"].attrs.get("_FillValue")
        ssha = ssh - mss
        if fill is not None:
            ssha = np.where(ssh == fill, np.nan, ssha)
        inb_q = inb & (qual == 0) & np.isfinite(ssha)
        print(f"  AOI 内 qual==0 有效像素: {inb_q.sum():,}"
              f"  (占 AOI 内 {100 * inb_q.sum() / max(inb.sum(), 1):.1f}%)")
        v = ssha[inb_q]
        if v.size:
            qs = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
            print(f"  SSHA(=ssh_karin_2−MSS) AOI 内统计(m): "
                  f"mean={v.mean():.4f} std={v.std():.4f}")
            print(f"    p1/p5/p25/p50/p75/p95/p99 = "
                  + " ".join(f"{x:.4f}" for x in qs))
        # 质量旗标分布（AOI 内）
        uq, cnt = np.unique(qual[inb], return_counts=True)
        top = sorted(zip(cnt, uq), reverse=True)[:5]
        print(f"  qual 取值分布(top5): "
              + ", ".join(f"{int(q)}:{c:,}" for c, q in top))
        ds.close()


if __name__ == "__main__":
    main()
