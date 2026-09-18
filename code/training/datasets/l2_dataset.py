"""L2 S1×SWOT 配对块数据集加载器（SWOT 弱融合实验用）。

数据：data/datasets/L2_s1_swot_matched/tiles/*.npz，每块含
- sar   (3,512,512) float16  原始切块三通道 [VV_dB, VH_dB, 入射角] 归一化
  （注意：并非 VV×3；v7 弱标签由 run_scene_inference.py 以 VV 通道复制
  三份输入生成，本数据集对齐该约定，取 sar[0] 复制三份喂模型）；
- swot  (3,20,20)   float32  [ssha(m), |grad|, mask(1=有效)]，5km 块 20×20；
- label (512,512)   uint8    v7 模型弱标签（0/1）；
- coverage 标量。

划分（防泄漏）：按天划分——20230830 全天验证，其余 4 天训练；
只用 label_quality ∈ {clean, none}（suspect 整块可疑开火，剔除）。

通道归一化（据 sanity check 量级）：ssha×100（中位 ~0.0055m→~0.55），
|grad|×1e5（中位 ~1.2e-5→~1.2），mask 不变。

swot_dropout：训练时按概率把 swot 整块清零，模拟缺失模态
（FiLM 门控下退化为 SAR-only，防止模型过度依赖 SWOT）。
swot_zero：SAR-only 对照组开关，swot 恒零（其余完全相同）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SSHA_SCALE = 100.0
GRAD_SCALE = 1e5
VAL_DAY = "20230830"


class L2PairDataset(Dataset):
    """L2 配对块数据集。split="train"/"val" 按天划分。"""

    def __init__(self, root: str | Path, split: str = "train",
                 swot_dropout: float = 0.0, swot_zero: bool = False,
                 augment: bool = False):
        self.root = Path(root)
        self.swot_dropout = swot_dropout if split == "train" else 0.0
        self.swot_zero = swot_zero
        self.augment = augment and split == "train"
        with open(self.root / "manifest_all.json", encoding="utf-8") as f:
            manifest = json.load(f)
        keep = [
            m for m in manifest
            if m["label_quality"] in ("clean", "none")
            and (m["day"] == VAL_DAY) == (split == "val")
        ]
        self.entries = sorted(keep, key=lambda m: m["file"])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        m = self.entries[idx]
        d = np.load(self.root / m["file"])
        sar = d["sar"].astype(np.float32)
        # 剔除预处理残留的 inf/NaN（float16 溢出等）
        sar = np.nan_to_num(sar, nan=0.0, posinf=1.0, neginf=0.0)
        # 对齐弱标签生成管线（run_scene_inference.py）：只取 VV 通道复制三份，
        # 原始块另两通道是 VH/入射角，直接喂会与 v7 训练分布不一致
        sar = sar[0:1].repeat(3, axis=0)
        swot = d["swot"].astype(np.float32)
        swot = np.nan_to_num(swot, nan=0.0, posinf=0.0, neginf=0.0)
        label = d["label"].astype(np.float32)

        swot = swot.copy()
        swot[0] *= SSHA_SCALE
        swot[1] *= GRAD_SCALE

        has_swot = bool(swot[2].sum() > 0)
        if self.swot_zero or (self.swot_dropout > 0
                              and np.random.rand() < self.swot_dropout):
            swot[:] = 0.0
            has_swot = False

        if self.augment:
            # 不做翻转/旋转（内波有方向先验），仅亮度微扰
            if np.random.rand() < 0.5:
                sar = np.clip(sar * (0.9 + 0.2 * np.random.rand()), 0, 1)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(sar)),
            "mask": torch.from_numpy(label[None].copy()),
            "swot": torch.from_numpy(np.ascontiguousarray(swot)),
            "has_swot": has_swot,
            "tile": m["tile"],
        }


def train_val_day_split(root: str | Path) -> tuple[int, int]:
    """返回 (训练块数, 验证块数)，供日志核对。"""
    with open(Path(root) / "manifest_all.json", encoding="utf-8") as f:
        manifest = json.load(f)
    tr = sum(1 for m in manifest if m["label_quality"] in ("clean", "none")
             and m["day"] != VAL_DAY)
    va = sum(1 for m in manifest if m["label_quality"] in ("clean", "none")
             and m["day"] == VAL_DAY)
    return tr, va
