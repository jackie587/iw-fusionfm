"""YOLO txt 标注 → COCO annotations.json 转换工具。

Tao et al. 2022 数据集（figshare 21365835）是 390 张 PNG + YOLO 格式 txt
标注（每行：class x_center y_center w h，归一化坐标）。本工具把它转成
code/training/datasets/tao_dataset.py 期望的 images/ + annotations.json
（COCO bbox: [x, y, w, h] 像素坐标）结构。

用法：
    python data_preprocessing/labeling/yolo_to_coco.py \
        --images <图目录> --labels <txt目录> \
        --out data/datasets/L1_sar_stripe/tao2022
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from utils.logger import get_logger

logger = get_logger("yolo_to_coco")


def image_size(path: Path) -> tuple[int, int]:
    """读图像宽高（不整图解码，优先 cv2，退化 PIL）。"""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.shape[1], img.shape[0]
    except ImportError:
        pass
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图像目录（png/jpg/tif）")
    ap.add_argument("--labels", required=True, help="YOLO txt 目录（与图像同名）")
    ap.add_argument("--out", required=True,
                    help="输出根目录（生成 images/ 软拷贝 + annotations.json）")
    ap.add_argument("--copy", action="store_true",
                    help="把图像复制到 out/images/（默认只做引用清单不复制）")
    args = ap.parse_args()

    img_dir, lbl_dir = Path(args.images), Path(args.labels)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    # 递归匹配：Tao 数据集按海区分子目录存放，文件名（含时间戳）本身唯一
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        logger.error("在 %s 没有找到图像", img_dir)
        sys.exit(1)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "internal_wave"}]}
    ann_id = 0
    n_boxes = 0
    for img_id, img_path in enumerate(images, start=1):
        w, h = image_size(img_path)
        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })
        txt = lbl_dir / (img_path.stem + ".txt")
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, xc, yc, bw, bh = float(parts[0]), *map(float, parts[1:5])
                # YOLO 归一化中心坐标 → COCO 像素左上角坐标
                x = (xc - bw / 2) * w
                y = (yc - bh / 2) * h
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, bw * w, bh * h],
                })
                ann_id += 1
                n_boxes += 1
        if args.copy:
            dest = out / "images" / img_path.name
            if not dest.exists():
                import shutil
                shutil.copy2(img_path, dest)

    with open(out / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=1)
    logger.info("转换完成：%d 张图，%d 个框 → %s", len(images), n_boxes, out)
    if not args.copy:
        logger.warning("未复制图像：请自行把图像放到 %s/images/ 下", out)


if __name__ == "__main__":
    main()
