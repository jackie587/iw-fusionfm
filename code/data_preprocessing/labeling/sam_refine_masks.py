"""SAM 辅助精标：把 COCO 框标注升级为条纹级像素掩膜。

对每个标注框：SAM box prompt 预测 → 取置信最高的掩膜 → 与框区域求交
（SAM 可能把掩膜溢出到框外，框是可信先验，交集最稳）→ 全图掩膜取并集。
输出 <数据集>/masks/<图名主干>.png（0/255）。

用法（在 code/ 目录下）：
    python data_preprocessing/labeling/sam_refine_masks.py \
        --dataset ../data/datasets/L1_sar_stripe/tao2022 \
        --ckpt ../checkpoints/sam/sam_vit_b_01ec64.pth [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--multimask", action="store_true", default=True)
    args = ap.parse_args()

    from segment_anything import SamPredictor, sam_model_registry

    root = Path(args.dataset)
    coco = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)

    out_dir = root / "masks"
    out_dir.mkdir(exist_ok=True)
    images = coco["images"][:args.limit]
    for i, im in enumerate(images):
        stem = Path(im["file_name"]).stem
        out_file = out_dir / f"{stem}.png"
        if out_file.exists():
            continue
        boxes = anns.get(im["id"], [])
        img = cv2.imread(str(root / "images" / im["file_name"]),
                         cv2.IMREAD_UNCHANGED)
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)  # SAM 吃 8-bit
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        h, w = img.shape[:2]
        full = np.zeros((h, w), np.uint8)
        if boxes:
            predictor.set_image(img)
            for a in boxes:
                x, y, bw, bh = a["bbox"]
                masks, scores, _ = predictor.predict(
                    box=np.array([x, y, x + bw, y + bh]),
                    multimask_output=True)
                best = masks[int(np.argmax(scores))]
                boxm = np.zeros((h, w), bool)
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
                boxm[y1:y2, x1:x2] = True
                sam_in_box = best & boxm
                # 混合回退：SAM 分割不足（掩膜面积 < 框面积 30%）时退回框填充
                if sam_in_box.mean() < 0.3 * boxm.mean():
                    sam_in_box = boxm
                full |= sam_in_box.astype(np.uint8)
        cv2.imwrite(str(out_file), full * 255)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(images)}", flush=True)
    print(f"完成 {len(images)} 张 → {out_dir}")


if __name__ == "__main__":
    main()
