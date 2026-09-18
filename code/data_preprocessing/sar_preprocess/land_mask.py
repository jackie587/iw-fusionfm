"""陆地掩膜：Natural Earth 10m land 多边形栅格化到场景网格。

补齐预处理链路的已知缺口（2026-08-26 日志："链路不做陆地掩膜"）。
数据源：data/raw/auxiliary/naturalearth/ne_10m_land.shp（Natural Earth，
公有领域，https://www.naturalearthdata.com/downloads/10m-physical-vectors/）。

用法：
    from data_preprocessing.sar_preprocess.land_mask import scene_land_mask
    mask = scene_land_mask(tiles_dir)   # True=陆地，形状=场景画布
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

SHP = PROJECT_ROOT / "data/raw/auxiliary/naturalearth/ne_10m_land.shp"


def _canvas_geo(tiles_dir: Path) -> tuple[tuple[int, int], list]:
    """从任意一个切块 meta 推出场景画布的 (h, w) 与仿射变换。"""
    import re

    meta_dir = Path(tiles_dir) / "meta"
    pat = re.compile(r"_y(\d+)_x(\d+)\.json$")
    max_y = max_x = 0
    transform = None
    for f in meta_dir.glob("*.json"):
        m = pat.search(f.name)
        y, x = int(m.group(1)), int(m.group(2))
        max_y, max_x = max(max_y, y), max(max_x, x)
        if transform is None:
            transform = json.loads(f.read_text())["transform"]
    ts = json.loads((meta_dir / f.name).read_text())["tile_size"]
    return (max_y + ts, max_x + ts), transform


def scene_land_mask(tiles_dir: str | Path,
                    shp_path: str | Path = SHP) -> np.ndarray:
    """栅格化陆地为场景画布大小的布尔掩膜（True=陆地）。"""
    import shapefile  # pyshp，rasterio 的依赖之一
    from affine import Affine

    (h, w), t = _canvas_geo(Path(tiles_dir))
    transform = Affine(*t)

    # 画布经纬度范围，只读相交的多边形（10m 海岸线全球有 4000+ 要素）
    from rasterio.transform import xy
    lons, lats = [], []
    for r, c in ((0, 0), (0, w), (h, 0), (h, w)):
        lon, lat = xy(transform, r, c)
        lons.append(lon)
        lats.append(lat)
    bbox = (min(lons) - 0.5, min(lats) - 0.5, max(lons) + 0.5, max(lats) + 0.5)

    reader = shapefile.Reader(str(shp_path))
    polys = []
    for sr in reader.iterShapeRecords(bbox=bbox):
        polys.append(sr.shape.__geo_interface__)

    from rasterio.features import geometry_mask
    # geometry_mask 默认 inside=True 要反转：我们要 land=True
    ocean = geometry_mask(polys, out_shape=(h, w), transform=transform,
                          invert=False)
    return ~ocean


def main():
    """CLI：python land_mask.py <tiles_dir> <out.npy>"""
    tiles_dir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else tiles_dir / "_land_mask.npy"
    mask = scene_land_mask(tiles_dir)
    np.save(out, mask)
    print(f"陆地占比 {mask.mean():.1%}，形状 {mask.shape} → {out}")


if __name__ == "__main__":
    main()
