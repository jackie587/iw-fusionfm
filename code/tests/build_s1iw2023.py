"""构建 S1-IW-2023 泛化测试集：合并官方 train/val/test 三个 COCO json，
按文件名重排唯一 image_id（三个 split 的 id 都是从 1 开始编的，会撞），
并链接/复制图像到标准目录结构（images/ + annotations.json）。

用途：模型在 Tao 2022 上训练，S1-IW-2023 全部 742 图作为独立泛化测试集
（无泄漏顾虑，不需要保留官方 split）。

用法（在 code/ 目录下）：
    python tests/build_s1iw2023.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# TODO: point this at your local copy of the S1-IW-2023 COCO annotations
ANN_DIR = PROJECT_ROOT / "data/raw/s1_iw_2023_annotations"
IMG_SRC = PROJECT_ROOT / "data/datasets/L1_sar_stripe/s1_iw_2023_raw/IW_Image"
OUT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/s1_iw_2023"


def main():
    images, anns = [], []
    fn2newid = {}
    for split in ("train", "val", "test"):
        d = json.loads((ANN_DIR / f"instances_{split}.json").read_text(
            encoding="utf-8"))
        id2fn = {im["id"]: im["file_name"] for im in d["images"]}
        for im in d["images"]:
            if im["file_name"] not in fn2newid:
                fn2newid[im["file_name"]] = len(fn2newid) + 1
                images.append({"id": fn2newid[im["file_name"]],
                               "file_name": im["file_name"],
                               "width": im["width"], "height": im["height"]})
        for a in d["annotations"]:
            a = dict(a)
            a["image_id"] = fn2newid[id2fn[a["image_id"]]]
            a["id"] = len(anns) + 1
            anns.append(a)

    # 校验：图像齐全
    missing = [im["file_name"] for im in images
               if not (IMG_SRC / im["file_name"]).exists()]
    assert not missing, f"缺图 {len(missing)} 张，例：{missing[:5]}"

    out_img = OUT / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    for im in images:
        dst = out_img / im["file_name"]
        if not dst.exists():
            shutil.copy2(IMG_SRC / im["file_name"], dst)

    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "internal wave",
                            "supercategory": "none"}]}
    with open(OUT / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"完成：{len(images)} 图 / {len(anns)} 框 → {OUT}")


if __name__ == "__main__":
    main()
