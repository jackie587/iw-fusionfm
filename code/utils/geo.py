"""地理工具：UTM 换算、内波传播位移校正。

用于 S1×SWOT 配对时按传播方向/速度把观测时刻对齐（见数据输入处理方案四）。
"""
from __future__ import annotations

import math


def lonlat_to_utm_zone(lon: float) -> int:
    """经度 → UTM 带号。"""
    return int((lon + 180.0) / 6.0) + 1


def meters_per_degree(lat: float) -> tuple[float, float]:
    """给定纬度，返回 (每度经度米数, 每度纬度米数) 的局部近似。"""
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) \
        + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_per_deg_lon, m_per_deg_lat


def propagation_shift(lat: float, direction_deg: float, speed_ms: float,
                      dt_seconds: float) -> tuple[float, float]:
    """内波传播位移校正。

    沿传播方向 direction_deg（自北顺时针，度）以速度 speed_ms 传播 dt_seconds，
    返回 (d_lon, d_lat) 位移量。用于把不同时刻的 SWOT/SAR 观测对齐到同一参考时刻。
    """
    dist = speed_ms * dt_seconds
    rad = math.radians(direction_deg)
    dx = dist * math.sin(rad)   # 东向
    dy = dist * math.cos(rad)   # 北向
    m_lon, m_lat = meters_per_degree(lat)
    return dx / m_lon, dy / m_lat


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """两点大圆距离（km），用于时空匹配初筛。"""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
