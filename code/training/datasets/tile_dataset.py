"""通用切块 Dataset：读取 data/processed/sar_tiles/ 下切好的 npy/GeoTIFF 块。

配合 sar_preprocess/tile.py 的输出格式：
    <tile_dir>/images/*.npy   [3, H, W] 已归一化
    <tile_dir>/masks/*.npy    [1, H, W] 0/1（无标注集可缺省 → 全零）
    <tile_dir>/meta/*.json    地理配准与元数据
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TileDataset(Dataset):
    def __init__(self, tile_dir: str | Path, with_mask: bool = True):
        self.tile_dir = Path(tile_dir)
        self.with_mask = with_mask
        self.files = sorted((self.tile_dir / "images").glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"没有找到切块：{self.tile_dir}/images/*.npy")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.files[idx]
        img = np.load(img_path).astype(np.float32)
        if self.with_mask:
            mask_path = self.tile_dir / "masks" / img_path.name
            mask = np.load(mask_path).astype(np.float32) if mask_path.exists() \
                else np.zeros((1, *img.shape[1:]), np.float32)
        else:
            mask = np.zeros((1, *img.shape[1:]), np.float32)
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "tile": img_path.stem,
            "has_stripe": bool(mask.sum() > 0),
        }
