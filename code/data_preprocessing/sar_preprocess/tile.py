"""SAR 切块张量化：定标 GeoTIFF → [3,512,512] 训练张量 + 地理配准元数据。

每块输出（写入 data/processed/sar_tiles/）：
    images/<name>.npy   [3,H,W] float32，通道 [VV_dB, VH_dB, 入射角]，已归一
    meta/<name>.json    经纬度范围、仿射变换、入射角统计、源影像、块内坐标
约定：所有切块保留地理配准（目录约定第 3 条 + 事件数据库的前提）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from data_preprocessing.sar_preprocess.incidence_normalize import (
    normalize_incidence, normalize_incidence_channel)
from utils.config import load_yaml
from utils.logger import get_logger

logger = get_logger("tile")


def read_bands(tif_path: Path) -> dict:
    """读 SNAP 输出的 GeoTIFF：VV/VH σ0 与（若有）入射角波段。"""
    import rasterio
    with rasterio.open(tif_path) as src:
        desc = [d or "" for d in src.descriptions]
        bands = {"crs": str(src.crs), "transform": tuple(src.transform)[:6]}
        for i, d in enumerate(desc, start=1):
            arr = src.read(i).astype(np.float32)
            dl = d.lower()
            if "sigma0_vv" in dl or dl == "vv":
                bands["vv"] = arr
            elif "sigma0_vh" in dl or dl == "vh":
                bands["vh"] = arr
            elif "incidence" in dl:
                bands["incidence"] = arr
        if "vv" not in bands:  # 无描述时按惯例前两波段
            bands["vv"] = src.read(1).astype(np.float32)
            if src.count > 1:
                bands.setdefault("vh", src.read(2).astype(np.float32))
    return bands


def to_db(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, eps))


def normalize_db(db: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    return np.clip((db - clip[0]) / (clip[1] - clip[0]), 0, 1).astype(np.float32)


def tile_bands(bands: dict, source_name: str, out_dir: Path, cfg: dict) -> int:
    """把 {"vv","vh","incidence","crs","transform"} 切块写盘。

    bands 可来自 read_bands（SNAP GeoTIFF）或 read_safe（直接读 SAFE）。
    """
    sar = cfg["sar"]
    ts, stride = sar["tile_size"], sar["tile_stride"]
    clip = tuple(sar["db_clip"])

    vv_db, vh_db = to_db(bands["vv"]), to_db(bands.get("vh", bands["vv"]))
    h, w = vv_db.shape

    inc = bands.get("incidence")
    if inc is None:  # 无入射角时，按 IW 刈幅范围行向线性近似
        logger.warning("%s 无入射角波段，用 29~46° 行向线性近似", source_name)
        inc = np.linspace(29, 46, w, dtype=np.float32)[None, :].repeat(h, 0)
    vv_db = normalize_incidence(vv_db, inc, sar["incidence_ref_deg"],
                                sar["incidence_bin_deg"])
    vh_db = normalize_incidence(vh_db, inc, sar["incidence_ref_deg"],
                                sar["incidence_bin_deg"])
    inc_norm = normalize_incidence_channel(inc)

    img = np.stack([normalize_db(vv_db, clip), normalize_db(vh_db, clip),
                    inc_norm])  # [3,H,W]

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    tf = bands["transform"]  # (a,b,c,d,e,f) 仿射
    stem = Path(source_name).stem
    n_tiles = 0
    for y in range(0, h - ts + 1, stride):
        for x in range(0, w - ts + 1, stride):
            tile = img[:, y:y + ts, x:x + ts]
            name = f"{stem}_y{y:05d}_x{x:05d}"
            np.save(out_dir / "images" / f"{name}.npy", tile)
            # 该块的仿射变换 = 平移到块原点
            block_tf = (tf[0], tf[1], tf[2] + x * tf[0] + y * tf[1],
                        tf[3], tf[4], tf[5] + x * tf[3] + y * tf[4])
            with open(out_dir / "meta" / f"{name}.json", "w",
                      encoding="utf-8") as f:
                json.dump({
                    "source": source_name, "row": y, "col": x,
                    "tile_size": ts, "crs": bands["crs"],
                    "transform": block_tf,
                    "incidence_deg_mean": float(inc[y:y + ts, x:x + ts].mean()),
                }, f, ensure_ascii=False)
            n_tiles += 1
    return n_tiles


def tile_scene(tif_path: Path, out_dir: Path, cfg: dict) -> int:
    bands = read_bands(tif_path)
    return tile_bands(bands, tif_path.name, out_dir, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="SNAP 定标后 GeoTIFF 目录")
    ap.add_argument("--out-dir", default="data/processed/sar_tiles")
    ap.add_argument("--config", default="code/configs/data.yaml")
    args = ap.parse_args()

    cfg = load_yaml(CODE_ROOT.parent / args.config
                    if not Path(args.config).is_absolute() else args.config)
    tifs = sorted(Path(args.in_dir).glob("*.tif"))
    if not tifs:
        logger.error("没有找到 GeoTIFF：%s", args.in_dir)
        sys.exit(1)

    total = 0
    for tif in tifs:
        n = tile_scene(tif, Path(args.out_dir), cfg)
        logger.info("%s → %d 块", tif.name, n)
        total += n
    logger.info("共切 %d 块 → %s", total, args.out_dir)


if __name__ == "__main__":
    main()
