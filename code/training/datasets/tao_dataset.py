"""Tao et al. 2022 / S1-IW-2023 数据集加载器。

公开数据集为框级标注（COCO 格式），起步策略：
- 框 → 二值掩膜（矩形填充），先训出基线与教师模型；
- 第二期用 SAM 辅助人工修校替换为精标像素掩膜后，只需换掉 mask 目录。

目录约定（data/datasets/L1_sar_stripe/tao2022/）：
    images/*.png|tif     SAR 切片
    annotations.json     COCO 格式标注（bbox: [x, y, w, h]）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None


def load_image(path: Path) -> np.ndarray:
    """读图并归一到 [0,1] float32。支持 png/jpg/tif。"""
    if cv2 is None:
        raise ImportError("需要 opencv-python-headless")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if img.ndim == 2:
        img = img[..., None].repeat(3, axis=2)  # 单通道 → 3 通道
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


class TaoIWDataset(Dataset):
    """COCO 框标注 → 像素掩膜的 SAR 内波数据集。

    Tao 原图约 1680×2540 且逐张不同，需裁成固定尺寸才能组 batch：
    - 训练（augment=True）：随机 512 裁剪，有条纹的图 80% 概率裁在条纹框附近；
    - 验证：中心 512 裁剪（确定性，便于复现）。
    """

    def __init__(self, root: str | Path, split_ids: list[int] | None = None,
                 augment: bool = False, crop_size: int = 512,
                 domain_aug: bool = False):
        self.root = Path(root)
        self.augment = augment
        self.domain_aug = domain_aug
        self.crop_size = crop_size
        with open(self.root / "annotations.json", encoding="utf-8") as f:
            coco = json.load(f)
        self.images = {im["id"]: im for im in coco["images"]}
        self.anns: dict[int, list] = {}
        for ann in coco["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)
        self.ids = split_ids if split_ids is not None else sorted(self.images.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def _mask_from_boxes(self, image_id: int, h: int, w: int) -> np.ndarray:
        # 优先读 SAM 精标掩膜（masks/<图名主干>.png），缺失则退回框填充
        if (self.root / "masks").exists():
            stem = Path(self.images[image_id]["file_name"]).stem
            mp = self.root / "masks" / f"{stem}.png"
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape == (h, w):
                    return (m > 127).astype(np.float32)
        mask = np.zeros((h, w), np.float32)
        for ann in self.anns.get(image_id, []):
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0
        return mask

    def _crop(self, img: np.ndarray, mask: np.ndarray,
              image_id: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        cs = self.crop_size
        if h <= cs and w <= cs:
            return img, mask
        if self.augment:
            boxes = self.anns.get(image_id, [])
            if boxes and np.random.rand() < 0.8:
                # 以随机一个条纹框为锚点，但框中心在裁剪窗内均匀分布
                # （2026-08-27 修正：原先 jitter 仅 ±cs/4，正样本系统性偏向
                # 画面中心，模型学到位置先验，大图推理出现网格伪影）
                x, y, bw, bh = boxes[np.random.randint(len(boxes))]["bbox"]
                cx, cy = x + bw / 2, y + bh / 2
                x0 = int(np.clip(cx - np.random.uniform(0.05, 0.95) * cs,
                                 0, w - cs))
                y0 = int(np.clip(cy - np.random.uniform(0.05, 0.95) * cs,
                                 0, h - cs))
            else:
                x0 = np.random.randint(0, w - cs + 1)
                y0 = np.random.randint(0, h - cs + 1)
        else:
            x0, y0 = (w - cs) // 2, (h - cs) // 2
        return img[y0:y0 + cs, x0:x0 + cs], mask[y0:y0 + cs, x0:x0 + cs]

    @staticmethod
    def _domain_aug(img: np.ndarray) -> np.ndarray:
        """域对齐增强（v9）：模拟跨场景/跨季节的辐射差异。

        均在 [0,1] 强度域操作，不改几何（内波方向先验）：
        - 随机百分位拉伸：模拟场景级 p2~p98 拉伸差异（域差主因）；
        - gamma：模拟定标/海况引起的非线性灰度偏移；
        - 加性高斯噪声：模拟斑点滤波残余差异。
        """
        if np.random.rand() < 0.5:  # 随机百分位拉伸
            lo, hi = np.percentile(img, [np.random.uniform(0, 4),
                                           np.random.uniform(96, 100)])
            img = np.clip((img - lo) / max(hi - lo, 1e-3), 0, 1)
        if np.random.rand() < 0.5:  # gamma
            img = np.clip(img, 0, 1) ** np.random.uniform(0.7, 1.4)
        if np.random.rand() < 0.3:  # 加性噪声
            img = np.clip(img + np.random.normal(
                0, np.random.uniform(0, 0.02), img.shape), 0, 1)
        return img.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.ids[idx]
        info = self.images[image_id]
        img = load_image(self.root / "images" / info["file_name"])
        h, w = img.shape[:2]
        mask = self._mask_from_boxes(image_id, h, w)
        img, mask = self._crop(img, mask, image_id)

        if self.augment:
            # 注意：不做翻转/旋转——内波有方向先验，见《最强技术方案》二（六）
            if np.random.rand() < 0.5:  # 亮度微扰，模拟入射角/海况差异
                img = np.clip(img * (0.9 + 0.2 * np.random.rand()), 0, 1)
            if self.domain_aug:
                img = self._domain_aug(img)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            "mask": torch.from_numpy(mask[None].copy()),
            "image_id": image_id,
            "has_stripe": bool(mask.sum() > 0),
        }


def train_val_split(dataset: TaoIWDataset, train_ratio: float = 0.8,
                    seed: int = 42) -> tuple[list[int], list[int]]:
    """按图像 id 划分训练/验证集。"""
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(dataset.images.keys()))
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    return ids[:n_train].tolist(), ids[n_train:].tolist()
