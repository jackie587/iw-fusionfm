"""构建 tao2022_negmix6_sam：v9 训练集。

组成：
- tao2022_negmix3_sam 全量硬链接（Tao 391 图 + SAM 掩膜 + rounds 1~3 判读裁块，
  与 v7 训练集逐字节一致）；
- 负样本库 rounds 4/5/6 裁块，直接从 negative_samples 的 npz 读出：
  positive 附检出区伪标签框、negative 无标注、uncertain 不进训练；
- **修正 negmix5_sam 的 annotations bug**（原脚本写回未扩展的 coco dict，
  追加的 images/anns 全部丢失）；
- **冬季留出**：round6 的 03-28 两景 + 03-30 一景不进训练，留作级联评估
  的诚实冬季测试集；
- 新增裁块 id 从 200000 起（≥100000 只进训练集，验证集与 v2~v8 严格一致）。

用法（在 code/ 目录下）：python tests/build_negmix6_sam.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022_negmix3_sam"
OUT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022_negmix6_sam"
LIB = PROJECT_ROOT / "data/datasets/negative_samples"
ADD_ROUNDS = (4, 5, 6)
NEW_ID_START = 200000
HOLDOUT_SCENES = (
    "S1A_IW_GRDH_1SDV_20240328T104129_20240328T104158_053181_06718B_138B",
    "S1A_IW_GRDH_1SDV_20240328T104158_20240328T104223_053181_06718B_DF34",
    "S1A_IW_GRDH_1SDV_20240330T102532_20240330T102601_053210_0672A0_8533",
)


def main():
    out_img = OUT / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    # 1) negmix3_sam 全量硬链接（images + masks）
    (OUT / "masks").mkdir(exist_ok=True)
    n_link = 0
    for sub in ("images", "masks"):
        for f in (BASE / sub).iterdir():
            dst = OUT / sub / f.name
            if not dst.exists():
                os.link(f, dst)
                n_link += 1
    coco = json.loads((BASE / "annotations.json").read_text(encoding="utf-8"))
    images = list(coco["images"])
    anns = list(coco["annotations"])
    print(f"硬链接 {n_link} 文件；基础 images={len(images)} anns={len(anns)}")

    # 2) 追加 round4/5/6 库裁块（跳过留出冬景与存疑样本）
    manifest = json.loads((LIB / "manifest.json").read_text(encoding="utf-8"))
    next_id = NEW_ID_START
    next_ann_id = max(a["id"] for a in anns) + 1
    n_pos = n_neg = n_holdout = 0
    for e in manifest:
        if e["round"] not in ADD_ROUNDS or e["label"] == "uncertain":
            continue
        if e["scene"] in HOLDOUT_SCENES:
            n_holdout += 1
            continue
        z = np.load(LIB / e["file"])
        vv = np.asarray(z["vv"], dtype=np.float32)
        fname = Path(e["file"]).stem + ".png"
        cv2.imwrite(str(out_img / fname),
                    (np.clip(vv, 0, 1) * 255).astype(np.uint8))
        images.append({"id": next_id, "file_name": fname,
                       "width": int(vv.shape[1]), "height": int(vv.shape[0])})
        if e["label"] == "positive":
            mask = np.asarray(z["mask"])
            ys, xs = np.where(mask)
            if len(ys):
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                anns.append({
                    "id": next_ann_id, "image_id": next_id,
                    "category_id": 1,
                    "bbox": [int(x0), int(y0), int(x1 - x0 + 1),
                             int(y1 - y0 + 1)],
                    "area": float((x1 - x0 + 1) * (y1 - y0 + 1)),
                    "iscrowd": 0})
                next_ann_id += 1
            n_pos += 1
        else:
            n_neg += 1
        next_id += 1

    # 3) 写回【扩展后】的 annotations（negmix5 的 bug 就是这里写了原始 coco）
    coco["images"] = images
    coco["annotations"] = anns
    (OUT / "annotations.json").write_text(
        json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"追加 round{ADD_ROUNDS}：positive {n_pos}、negative {n_neg}、"
          f"留出跳过 {n_holdout}；总计 images={len(images)} anns={len(anns)}"
          f" → {OUT}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
