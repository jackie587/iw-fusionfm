"""入射角归一化（SAR 新手最常见的精度杀手，必须做）。

IW 模式跨刈幅入射角 29°~46°，σ0 随入射角系统性变暗。
方法：按入射角分 bin 统计全场景亮度中位数，拟合随角度变化的
增益曲线，把每个像素归一到参考角（默认 35°）的等效亮度。

同时输出归一化后的入射角图，作为模型第 3 输入通道。
"""
from __future__ import annotations

import numpy as np


def normalize_incidence(sigma0_db: np.ndarray, incidence_deg: np.ndarray,
                        ref_deg: float = 35.0, bin_deg: float = 1.0,
                        min_samples: int = 500) -> np.ndarray:
    """把 σ0_dB 归一到参考入射角。

    sigma0_db: [H,W]；incidence_deg: [H,W] 入射角图。
    返回归一化后的 σ0_dB（同形状）。
    """
    out = sigma0_db.copy()
    valid = np.isfinite(sigma0_db) & np.isfinite(incidence_deg)
    if valid.sum() < min_samples:
        return out  # 样本太少不做，避免引入伪校正

    bins = np.arange(incidence_deg[valid].min(),
                     incidence_deg[valid].max() + bin_deg, bin_deg)
    centers, medians = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = valid & (incidence_deg >= lo) & (incidence_deg < hi)
        if sel.sum() >= min_samples:
            centers.append((lo + hi) / 2)
            medians.append(np.median(sigma0_db[sel]))
    if len(centers) < 3:
        return out

    # 线性拟合 σ0(dB) ~ 入射角（文献：海面上近似线性，约 -0.2~-0.5 dB/°）
    coef = np.polyfit(centers, medians, 1)
    correction = np.polyval(coef, ref_deg) - np.polyval(coef, incidence_deg)
    out[valid] = sigma0_db[valid] + correction[valid]
    return out


def normalize_incidence_channel(incidence_deg: np.ndarray,
                                lo: float = 25.0, hi: float = 50.0) -> np.ndarray:
    """入射角图归一到 [0,1]，作为模型输入通道。"""
    return np.clip((incidence_deg - lo) / (hi - lo), 0, 1).astype(np.float32)
