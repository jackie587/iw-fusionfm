"""框级标注 → 像素掩膜转换工具。

起步策略：公开数据集（Tao 2022 / S1-IW-2023）都是框级标注，
先把框填成矩形掩膜训基线；第二期用 SAM 辅助人工修校替换为精标掩膜。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def coco_to_masks(ann_json: str | Path, out_dir: str | Path) -> int:
    """把 COCO annotations.json 里每张图的框转成同名 .npy 掩膜。"""
    with open(ann_json, encoding="utf-8") as f:
        coco = json.load(f)
    anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns.setdefault(ann["image_id"], []).append(ann)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in coco["images"]:
        h, w = im["height"], im["width"]
        mask = np.zeros((1, h, w), np.float32)
        for ann in anns.get(im["id"], []):
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            if x2 > x1 and y2 > y1:
                mask[0, y1:y2, x1:x2] = 1.0
        np.save(out / (Path(im["file_name"]).stem + ".npy"), mask)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = coco_to_masks(args.annotations, args.out)
    print(f"转换完成：{n} 张掩膜 → {args.out}")


if __name__ == "__main__":
    main()
