"""SSHA 与梯度特征加工。

- SSHA = SSH − 沿轨滑动均值（去掉大尺度信号，突出内波/内潮波动）；
- |∇SSH|：SSH 梯度模，内波信号在梯度场更突出；
- 输出供 reproject.py 重投影到 SAR 网格。
"""
from __future__ import annotations

import numpy as np


def compute_ssha(ssh: np.ndarray, along_track_window: int = 81) -> np.ndarray:
    """沿轨（第 0 轴）去趋势得 SSHA。NaN 位置保持 NaN。"""
    from scipy import ndimage
    valid = np.isfinite(ssh)
    filled = np.where(valid, ssh, 0.0)
    # 沿轨均值用带掩膜的均值滤波
    wsum = ndimage.uniform_filter1d(valid.astype(np.float64),
                                    size=along_track_window, axis=0, mode="nearest")
    ssum = ndimage.uniform_filter1d(filled, size=along_track_window,
                                    axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = np.where(wsum > 1e-6, ssum / np.maximum(wsum, 1e-6), np.nan)
    ssha = ssh - trend
    ssha[~valid] = np.nan
    return ssha


def compute_ssh_gradient(ssha: np.ndarray, res_m: float = 250.0) -> np.ndarray:
    """梯度模 |∇SSHA|（单位 m/m），NaN 传播为 NaN。"""
    valid = np.isfinite(ssha)
    filled = np.where(valid, ssha, 0.0)
    gy, gx = np.gradient(filled, res_m)
    grad = np.hypot(gx, gy)
    grad[~valid] = np.nan
    return grad


def make_swot_channels(qc: dict, res_m: float = 250.0) -> dict:
    """从质控结果生成模型输入通道：SSHA / |∇SSH| / 有效掩膜。"""
    ssha = compute_ssha(qc["ssh"])
    grad = compute_ssh_gradient(ssha, res_m)
    return {"lon": qc["lon"], "lat": qc["lat"],
            "ssha": ssha, "grad": grad, "mask": qc["mask"]}
