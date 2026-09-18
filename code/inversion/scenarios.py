"""南海北部季节性两层分层场景 + SRTM15+ 水深采样（反演环境参数输入）。

分层假设（无实测温盐，文献气候态区间，用于敏感性扫描而非单点反演）：
  事件库 2023-06~10 属夏季→初秋。南海北部陆架/陆坡实测个例
  （Ramp et al. 2010 JPO ASIAEX；Cai et al. 2012；Jia et al. 2019 RS）
  混合层夏季 20~60 m、秋季加深到 40~90 m；跨跃层密度差夏季 Δσ≈2~5 kg/m³、
  秋季减弱。方案文档指出浅水区振幅反演季节差异可达 37~50%，故按季节分档、
  每档给 (h1, Δρ/ρ) 3×3 敏感性网格，反演输出区间（min/median/max）。

水深：GMT 远端数据 SRTM15+ v2.7（Tozer et al. 2019，海洋部分源自 GEBCO）
  01m 象限瓦片裁剪的南海 AOI（105~120E, 15~25N），
  下载见 batch_invert.py 顶部说明或 logs；像素配准、高程正值/水深负值。

有效性门槛（超出则该组合剔除，全部剔除则事件跳过并标记）：
  H2_MIN_M      下层至少 15 m；
  CRIT_RATIO    h2/h1 > 1.2 —— h1≈h2 时 α→0 临界分层，KdV 失效
                （方案文档"临界/大振幅需 DJL"，此处只标记不硬算）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATHY_TIF = PROJECT_ROOT / "data/raw/auxiliary/bathymetry/scs_srtm15_01m.tif"
BATHY_SRC = ("SRTM15+ v2.7 (Tozer et al. 2019) via GMT remote "
             "earth_relief_01m_p, tile N00E090, 裁剪 105~120E/15~25N")

H2_MIN_M = 15.0      # 下层最小厚度
H_MIN_M = 30.0       # 总水深最小值（再浅两层近似无意义）
CRIT_RATIO = 1.2     # h2/h1 下限（避开 α≈0 临界分层）
DJL_AMP_RATIO = 1.0  # η0 > h1 判为大振幅（KdV/eKdV 超域，需 DJL，仅标记）
SHORT_WAVE = 3.0     # λ < 3H 时破长波假设（l<H），KdV 不适用的短波区，仅标记

# 季节分层场景：月份 → (h1 网格 m, Δρ/ρ 网格)
# 来源：WOA23 南海北部气候态 + Ramp et al. 2010 / Cai et al. 2012 /
# Jia et al. 2019 个例值，取包络区间（中心=中值）。
SEASONAL_SCENARIOS = {
    "summer": {"months": (6, 7, 8, 9),
               "h1": (25.0, 40.0, 60.0),
               "drho": (2.0e-3, 3.5e-3, 5.0e-3)},
    "autumn": {"months": (10, 11),
               "h1": (40.0, 60.0, 90.0),
               "drho": (1.5e-3, 2.5e-3, 4.0e-3)},
    "winter": {"months": (12, 1, 2),
               "h1": (60.0, 80.0, 110.0),
               "drho": (1.0e-3, 2.0e-3, 3.0e-3)},
    "spring": {"months": (3, 4, 5),
               "h1": (30.0, 50.0, 80.0),
               "drho": (1.5e-3, 3.0e-3, 4.5e-3)},
}


def season_of(month: int) -> str:
    for name, sc in SEASONAL_SCENARIOS.items():
        if month in sc["months"]:
            return name
    raise ValueError(f"非法月份 {month}")


def scenario_grid(month: int) -> list[dict]:
    """给定月份返回 (h1, drho) 敏感性网格（9 组）。"""
    sc = SEASONAL_SCENARIOS[season_of(month)]
    return [{"h1": h1, "drho": dr} for h1 in sc["h1"] for dr in sc["drho"]]


class Bathymetry:
    """SCS AOI 水深栅格（正值=水深 m），双线性采样。"""

    def __init__(self, tif: Path = BATHY_TIF):
        import rasterio
        ds = rasterio.open(tif)
        self.res = ds.res[0]
        self.lon0 = ds.bounds.left           # 像素中心 lon0+res/2
        self.lat_top = ds.bounds.top
        self.z = -ds.read(1).astype(np.float64)  # 高程负值 → 水深正值
        ds.close()

    def depth(self, lon: float, lat: float) -> float:
        """双线性插值水深（m，正值；陆地为负）。"""
        fx = (lon - self.lon0) / self.res - 0.5
        fy = (self.lat_top - lat) / self.res - 0.5
        ny, nx = self.z.shape
        fx = min(max(fx, 0.0), nx - 1.001)
        fy = min(max(fy, 0.0), ny - 1.001)
        x0, y0 = int(fx), int(fy)
        dx, dy = fx - x0, fy - y0
        z = self.z
        return float(z[y0, x0] * (1 - dx) * (1 - dy)
                     + z[y0, x0 + 1] * dx * (1 - dy)
                     + z[y0 + 1, x0] * (1 - dx) * dy
                     + z[y0 + 1, x0 + 1] * dx * dy)
