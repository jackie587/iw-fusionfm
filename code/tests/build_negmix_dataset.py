"""构建负样本混合数据集 tao2022_negmix：

- Tao 2022 全部 391 图（硬链接，不复制数据），id 保持不变；
- 人工判读的 122 个虚警裁块（空掩膜）+ 101 个真内波裁块
  （掩膜=检出区域，伪标签但内容经人工确认），id 从 100000 起
  （训练脚本据此只把它们放进训练集，验证集保持与原实验一致）；
- 裁块窗口 512×512（对齐训练 crop_size），边缘反射填充。

用法（在 code/ 目录下）：python tests/build_negmix_dataset.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "tests"))

from make_review_sheets import build_vv_canvas  # noqa: E402

WINDOW = 512
OUT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022_negmix"
TAO = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022"


def crop_center(canvas: np.ndarray, cy: int, cx: int,
                size: int, pad_value: float = 0.0) -> np.ndarray:
    """以 (cy,cx) 为中心裁 size×size，越界部分反射填充（图像）或置 0（掩膜）。"""
    h, w = canvas.shape
    y0, x0 = cy - size // 2, cx - size // 2
    ys, xs = np.clip([y0, y0 + size], 0, h), np.clip([x0, x0 + size], 0, w)
    patch = canvas[ys[0]:ys[1], xs[0]:xs[1]]
    pt, pl = ys[0] - y0, xs[0] - x0
    pb, pr = y0 + size - ys[1], x0 + size - xs[1]
    if pt or pb or pl or pr:
        mode = cv2.BORDER_REPLICATE if pad_value is None else cv2.BORDER_CONSTANT
        patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, mode,
                                   value=pad_value)
    return patch


def main():
    verdicts = json.loads(
        (PROJECT_ROOT / "results/scene_eval/human_review_far.json").read_text())

    out_img = OUT / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    # 1) Tao 图像硬链接 + 原标注
    coco = json.loads((TAO / "annotations.json").read_text(encoding="utf-8"))
    for im in coco["images"]:
        dst = out_img / im["file_name"]
        if not dst.exists():
            os.link(TAO / "images" / im["file_name"], dst)
    images = list(coco["images"])
    anns = list(coco["annotations"])
    next_id = 100000
    next_ann_id = max(a["id"] for a in anns) + 1

    # 2) 判读裁块
    n_pos = n_neg = 0
    for scene, v in verdicts.items():
        if scene == "pooled":
            continue
        tiles_dir = PROJECT_ROOT / "data/processed/sar_tiles" / (scene + ".SAFE")
        scene_out = PROJECT_ROOT / "results/scene_eval" / scene
        picked = json.loads((scene_out / "review/components.json")
                            .read_text())["picked"]
        prob = np.nan_to_num(
            np.load(scene_out / "prob_masked.npy").astype(np.float32))
        vv = build_vv_canvas(tiles_dir, prob.shape)

        for label, ids in (("positive", v["tp_ids"]),
                           ("negative", v["fp_ids"])):
            for sheet_no in ids:
                c = picked[sheet_no - 1]
                cy, cx = int(c["cy"]), int(c["cx"])
                img_patch = crop_center(vv, cy, cx, WINDOW, pad_value=None)
                img_patch = np.nan_to_num(img_patch, nan=0.0)
                stag = re.search(r"\d{8}T\d{6}", scene).group(0)  # 区分两景
                fname = f"negmix_{label}_{stag}_n{sheet_no:03d}.png"
                cv2.imwrite(str(out_img / fname),
                            (img_patch * 255).astype(np.uint8))
                images.append({"id": next_id, "file_name": fname,
                               "width": WINDOW, "height": WINDOW})
                if label == "positive":
                    # 伪标签掩膜：检出区域存回 COCO（框用检出区包围盒）
                    det = crop_center(prob, cy, cx, WINDOW,
                                      pad_value=0.0) > 0.6
                    ys, xs = np.where(det)
                    if len(ys):
                        anns.append({
                            "id": next_ann_id, "image_id": next_id,
                            "category_id": 1, "iscrowd": 0,
                            "bbox": [int(xs.min()), int(ys.min()),
                                     int(xs.max() - xs.min() + 1),
                                     int(ys.max() - ys.min() + 1)],
                            "area": float(det.sum()),
                            "segmentation": []})
                        next_ann_id += 1
                    n_pos += 1
                else:
                    n_neg += 1
                next_id += 1

    with open(OUT / "annotations.json", "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": coco["categories"]}, f)
    print(f"完成：Tao {len(coco['images'])} 图 + 正样本 {n_pos} + 负样本 {n_neg}"
          f" → {OUT}")


if __name__ == "__main__":
    main()
