"""tao2022_negmix 数据集自检。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.datasets.tao_dataset import TaoIWDataset

ds = TaoIWDataset("../data/datasets/L1_sar_stripe/tao2022_negmix")
print("总数:", len(ds))
b = ds[500]  # 裁块区
print("裁块:", tuple(b["image"].shape), "mask>0:", bool(b["mask"].sum() > 0),
      "has_stripe:", b["has_stripe"])
pos = sum(1 for i in ds.ids if i >= 100000 and ds.anns.get(i))
neg = sum(1 for i in ds.ids if i >= 100000 and not ds.anns.get(i))
print(f"裁块: 正样本(有掩膜)={pos} 负样本(空掩膜)={neg}")
# 验证裁块像素值域正常
import numpy as np
print("裁块像素: min=%.3f max=%.3f mean=%.3f" % (
    b["image"].min(), b["image"].max(), b["image"].mean()))
