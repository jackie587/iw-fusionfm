"""大图切块推理：重叠切块 + 均值融合拼接，结果可回写 GeoTIFF。

配合 sar_preprocess/tile.py 的切块参数（512/stride 384），
保证检测条纹能通过地理配准映射回经纬度（事件数据库的前提）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def predict_large_image(model: nn.Module, image: np.ndarray,
                        tile_size: int = 512, stride: int = 384,
                        device: str | torch.device | None = None,
                        amp: bool = True) -> np.ndarray:
    """对任意大小的 [C,H,W] 归一化图像做切块推理，返回 [H,W] 概率图。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    c, h, w = image.shape
    prob = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)

    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if ys and ys[-1] != h - tile_size:
        ys.append(max(0, h - tile_size))
    if xs and xs[-1] != w - tile_size:
        xs.append(max(0, w - tile_size))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    for y in ys:
        for x in xs:
            tile = image[:, y:y + tile_size, x:x + tile_size]
            pad_h, pad_w = tile_size - tile.shape[1], tile_size - tile.shape[2]
            if pad_h or pad_w:
                tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)))
            t = torch.from_numpy(tile[None]).to(device)
            with torch.amp.autocast("cuda", enabled=amp and str(device) != "cpu"):
                logits = model(t)
            p = torch.sigmoid(logits)[0, 0].cpu().numpy()
            ph, pw = tile_size - pad_h, tile_size - pad_w
            prob[y:y + ph, x:x + pw] += p[:ph, :pw]
            weight[y:y + ph, x:x + pw] += 1.0

    return prob / np.maximum(weight, 1e-6)


def save_geotiff(prob: np.ndarray, ref_meta: dict, out_path: str | Path) -> None:
    """按参考地理配准信息把概率图写成 GeoTIFF。"""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    transform = Affine(*ref_meta["transform"][:6]) if "transform" in ref_meta else None
    with rasterio.open(
        out_path, "w", driver="GTiff", height=prob.shape[0], width=prob.shape[1],
        count=1, dtype="float32",
        crs=CRS.from_string(ref_meta.get("crs", "EPSG:4326")),
        transform=transform,
    ) as dst:
        dst.write(prob.astype(np.float32), 1)
