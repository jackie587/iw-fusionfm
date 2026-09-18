"""出一张"内波识别参考图"：从 Tao 2022 裁带标注的真条纹区域。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from training.datasets.tao_dataset import load_image

root = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022"
coco = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
imgs = {im["id"]: im for im in coco["images"]}

# 挑几个不同海区、框大小适中的样本
picks, seen = [], 0
for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    if 200 < w < 700 and 200 < h < 700:
        picks.append(ann)
        seen += 1
        if seen == 6:
            break

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, ann in zip(axes.flat, picks):
    info = imgs[ann["image_id"]]
    img = load_image(root / "images" / info["file_name"])
    x, y, w, h = [int(v) for v in ann["bbox"]]
    pad = int(max(w, h) * 0.35)
    H, W = img.shape[:2]
    y0, y1 = max(0, y - pad), min(H, y + h + pad)
    x0, x1 = max(0, x - pad), min(W, x + w + pad)
    ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
    ax.add_patch(plt.Rectangle((x - x0, y - y0), w, h, fill=False,
                               edgecolor="red", linewidth=1.5))
    ax.set_title(info["file_name"], fontsize=9)
    ax.axis("off")
fig.suptitle("真内波长这样（红框=专家标注位置）：明暗相间、大致平行的条纹群")
fig.tight_layout()
out = PROJECT_ROOT / "results/figures/iw_reference_examples.png"
fig.savefig(out, dpi=110)
print("→", out)
