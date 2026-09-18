"""后处理：骨架化 → 连通域过滤 → 波峰线拟合。

断线修复（Lu 2025 距离-角度-竞争优化）与波包分组为第二期增强项，
本期先把标准链路搭通，输出可供事件数据库使用的波峰线矢量。
"""
from __future__ import annotations

import math

import numpy as np


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """二值掩膜骨架化（条纹中心线）。"""
    from skimage.morphology import skeletonize
    return skeletonize(mask > 0).astype(np.uint8)


def filter_small_components(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """去掉过小连通域（碎噪声斑），保留条纹尺度结构。"""
    from scipy import ndimage
    lab, _ = ndimage.label(mask > 0)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (counts >= min_area)[lab].astype(np.uint8)


def fit_crest_lines(skeleton: np.ndarray, min_length: int = 30) -> list[dict]:
    """从骨架提取波峰线：每个连通域拟合一条主线，返回端点与方向。

    返回 [{"points": [(y,x),...], "direction_deg": 传播方向(待速度估计),
           "length_px": 长度}, ...]
    """
    from scipy import ndimage
    lab, n = ndimage.label(skeleton > 0)
    counts = np.bincount(lab.ravel())
    slices = ndimage.find_objects(lab)
    lines = []
    for i in np.nonzero(counts >= min_length)[0]:
        if i == 0:
            continue
        sl = slices[i - 1]
        ys, xs = np.nonzero(lab[sl] == i)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        # PCA 主线方向（条纹走向，传播方向与其垂直）
        pts = np.stack([ys - ys.mean(), xs - xs.mean()], axis=1)
        cov = pts.T @ pts / len(pts)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]  # 条纹走向
        crest_angle = math.degrees(math.atan2(axis[1], axis[0]))
        direction = (crest_angle + 90.0) % 180.0  # 传播方向（180° 模糊，后续用波包时序解算）
        lines.append({
            "points": list(zip(ys.tolist(), xs.tolist())),
            "crest_angle_deg": crest_angle,
            "direction_deg": direction,
            "length_px": int(len(ys)),
        })
    return lines


def postprocess(prob: np.ndarray, threshold: float = 0.5,
                min_area: int = 20, min_length: int = 30) -> dict:
    """完整后处理：概率图 → 掩膜 → 过滤 → 骨架 → 波峰线。"""
    mask = (prob > threshold).astype(np.uint8)
    mask = filter_small_components(mask, min_area)
    skel = skeletonize_mask(mask)
    lines = fit_crest_lines(skel, min_length)
    return {"mask": mask, "skeleton": skel, "crest_lines": lines}
