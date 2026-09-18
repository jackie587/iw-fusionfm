"""SWOT L2 KaRIn SSH 质量控制。

规则（见数据输入处理方案三）：
- 按 ssh_karin_qual 剔除坏点（仅保留 qual==0 的好点）；
- 星下点 ~20 km 空白带显式掩膜，**绝不插值填充**；
- KaRIn 随机噪声小尺度很大 → 沿轨/跨轨平滑到有效公里级分辨率，
  用大尺度低通会抹掉 ISW 信号，这里用保守的中值+小窗口高斯。
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def load_and_qc(nc_path: str,
                ssh_var: str = "ssh_karin",
                qual_var: str = "ssh_karin_qual") -> dict:
    """读 SWOT L2 产品并质控。

    返回 {"lon": 2D, "lat": 2D, "ssh": 2D(坏点NaN), "mask": 2D(1=有效)}。
    """
    ds = xr.open_dataset(nc_path, group=None)
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    ssh = ds[ssh_var].values.astype(np.float64)
    qual = ds[qual_var].values if qual_var in ds else np.zeros_like(ssh)

    fill = getattr(ds[ssh_var], "_FillValue", None)
    valid = (qual == 0) & np.isfinite(ssh)
    if fill is not None:
        valid &= ssh != fill

    ssh_clean = np.where(valid, ssh, np.nan)
    return {"lon": lon, "lat": lat, "ssh": ssh_clean,
            "mask": valid.astype(np.uint8)}


def denoise_ssh(ssh: np.ndarray, mask: np.ndarray,
                median_size: int = 3, gauss_sigma: float = 1.0) -> np.ndarray:
    """保守降噪：中值去野点 + 小 σ 高斯。坏点（NaN）位置保持 NaN。"""
    from scipy import ndimage
    valid = mask > 0
    filled = np.where(valid, ssh, np.nanmedian(ssh[valid]) if valid.any() else 0.0)
    out = ndimage.median_filter(filled, size=median_size)
    out = ndimage.gaussian_filter(out, sigma=gauss_sigma)
    out[~valid] = np.nan  # 无效区（含 nadir gap）恢复 NaN，不欺骗模型
    return out


def load_unsmoothed_half(nc_path: str, side: str,
                         bbox: tuple | None = None,
                         margin: float = 0.2) -> dict | None:
    """读 SWOT Unsmoothed 250m 产品的半刈幅（left/right）。

    该产品无 ssha 变量，用 ssh_karin_2 − mean_sea_surface_cnescls 近似
    （残留大尺度由下游沿轨去趋势去除，2026-08-31 验证 |p90|≈5.5cm）。
    bbox=(lon1,lat1,lon2,lat2) 时先把刈幅网格裁到包围盒（含 margin 度），
    避免 20M 像素全图降噪的开销。返回 {"lon","lat","ssh"(=SSHA 初值),
    "mask"}；半边不存在或完全在 bbox 外返回 None。
    """
    try:
        ds = xr.open_dataset(nc_path, group=side)
    except OSError:
        return None
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    if bbox is not None:
        lon1, lat1, lon2, lat2 = bbox
        inb = ((lon >= lon1 - margin) & (lon <= lon2 + margin)
               & (lat >= lat1 - margin) & (lat <= lat2 + margin))
        if not inb.any():
            ds.close()
            return None
        rows = np.nonzero(inb.any(axis=1))[0]
        cols = np.nonzero(inb.any(axis=0))[0]
        r0, r1 = max(rows[0] - 50, 0), min(rows[-1] + 51, lon.shape[0])
        c0, c1 = max(cols[0] - 50, 0), min(cols[-1] + 51, lon.shape[1])
        lon, lat = lon[r0:r1, c0:c1], lat[r0:r1, c0:c1]
        sl = np.s_[r0:r1, c0:c1]
    else:
        sl = np.s_[:, :]
    ssh = ds["ssh_karin_2"].values[sl].astype(np.float64)
    qual = ds["ssh_karin_2_qual"].values[sl]
    mss = ds["mean_sea_surface_cnescls"].values[sl].astype(np.float64)
    ssha = ssh - mss
    valid = (qual == 0) & np.isfinite(ssha)
    ssha_clean = np.where(valid, ssha, np.nan)
    ds.close()
    return {"lon": lon, "lat": lat, "ssh": ssha_clean,
            "mask": valid.astype(np.uint8)}
