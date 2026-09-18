"""SAFE → 训练切块 的纯 Python 降级链路（无 SNAP）。

流程：read_safe（读 SAFE + 定标 + 入射角 + 配准）
    → tile.tile_bands（to_db → 入射角归一化 → 切块 + meta 写出，与
      SNAP 链路完全同一套代码）。
输出：--out/<场景名>/images/*.npy + meta/*.json

用法：
    python code/data_preprocessing/sar_preprocess/preprocess_safe.py \
        --safe data/raw/sentinel1/S1A_....SAFE \
        --out data/processed/sar_tiles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from data_preprocessing.sar_preprocess.read_safe import read_safe
from data_preprocessing.sar_preprocess.tile import tile_bands
from utils.config import load_yaml
from utils.logger import get_logger

logger = get_logger("preprocess_safe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", required=True,
                    help="SAFE 目录或 zip 流（.zip / .SAFE 文件）")
    ap.add_argument("--out", default="data/processed/sar_tiles",
                    help="输出根目录（其下按场景名建子目录）")
    ap.add_argument("--config", default="code/configs/data.yaml")
    ap.add_argument("--tile-size", type=int, default=None,
                    help="覆盖配置中的 sar.tile_size")
    ap.add_argument("--stride", type=int, default=None,
                    help="覆盖配置中的 sar.tile_stride")
    ap.add_argument("--extract-dir", default=None,
                    help="zip 解压到此目录并保留（默认用临时目录、用完删除）")
    args = ap.parse_args()

    cfg = load_yaml(CODE_ROOT.parent / args.config
                    if not Path(args.config).is_absolute() else args.config)
    if args.tile_size:
        cfg["sar"]["tile_size"] = args.tile_size
    if args.stride:
        cfg["sar"]["tile_stride"] = args.stride

    safe_path = Path(args.safe)
    scene = safe_path.name
    if scene.lower().endswith(".zip"):
        scene = scene[:-4]
    out_dir = Path(args.out) / scene

    bands = read_safe(safe_path, extract_dir=args.extract_dir)
    n = tile_bands(bands, scene, out_dir, cfg)
    logger.info("%s → %d 块 → %s", scene, n, out_dir)


if __name__ == "__main__":
    main()
