"""构建 tao2022_negmix2：Tao + 两轮人工判读裁块（round1 v2@0.6 + round2 v3@0.85）。

同时把 negative_samples 库统一重建为 512×512 裁块（原 round1 是 384，作废重建）。
输出：
- data/datasets/negative_samples/{positive,negative,uncertain}/*.npz + manifest.json
- data/datasets/L1_sar_stripe/tao2022_negmix2/（images/ + annotations.json，
  裁块 id≥100000 只进训练集）

用法（在 code/ 目录下）：python tests/build_negmix2.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "tests"))

from build_negmix_dataset import crop_center  # noqa: E402
from make_review_sheets import build_vv_canvas  # noqa: E402

WINDOW = 512
TAO = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022"
OUT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022_negmix2"
LIB = PROJECT_ROOT / "data/datasets/negative_samples"

# (判读文件, scene_eval 目录, 该轮正样本伪标签阈值)
ROUNDS = [
    ("results/scene_eval/human_review_far.json", "results/scene_eval", 0.6),
    ("results/scene_eval_v3negmix/human_review_far_round2.json",
     "results/scene_eval_v3negmix", 0.85),
]


def main():
    # 清空旧库（round1 的 384 裁块作废，统一 512 重建）
    if LIB.exists():
        shutil.rmtree(LIB)
    manifest = []

    out_img = OUT / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    coco = json.loads((TAO / "annotations.json").read_text(encoding="utf-8"))
    import os
    for im in coco["images"]:
        dst = out_img / im["file_name"]
        if not dst.exists():
            os.link(TAO / "images" / im["file_name"], dst)
    images = list(coco["images"])
    anns = list(coco["annotations"])
    next_id = 100000
    next_ann_id = max(a["id"] for a in anns) + 1

    counts = {"positive": 0, "negative": 0, "uncertain": 0}
    for ri, (vf, sdir, thr) in enumerate(ROUNDS, start=1):
        verdicts = json.loads((PROJECT_ROOT / vf).read_text())
        for scene, v in verdicts.items():
            if scene == "pooled":
                continue
            tiles_dir = (PROJECT_ROOT / "data/processed/sar_tiles"
                         / (scene + ".SAFE"))
            scene_out = PROJECT_ROOT / sdir / scene
            picked = json.loads((scene_out / "review/components.json")
                                .read_text())["picked"]
            prob = np.nan_to_num(
                np.load(scene_out / "prob_masked.npy").astype(np.float32))
            vv = build_vv_canvas(tiles_dir, prob.shape)

            for label, key in (("positive", "tp_ids"),
                               ("negative", "fp_ids"),
                               ("uncertain", "uncertain_ids")):
                (LIB / label).mkdir(parents=True, exist_ok=True)
                for sheet_no in v[key]:
                    c = picked[sheet_no - 1]
                    cy, cx = int(c["cy"]), int(c["cx"])
                    img = crop_center(vv, cy, cx, WINDOW, pad_value=None)
                    img = np.nan_to_num(img, nan=0.0)
                    det = crop_center(prob, cy, cx, WINDOW,
                                      pad_value=0.0) > thr
                    stag = re.search(r"\d{8}T\d{6}", scene).group(0)
                    name = f"r{ri}_{stag}_n{sheet_no:03d}"
                    np.savez_compressed(LIB / label / f"{name}.npz",
                                        vv=img.astype(np.float16),
                                        mask=det.astype(np.bool_))
                    manifest.append({"file": f"{label}/{name}.npz",
                                     "label": label, "scene": scene,
                                     "round": ri, "cid": c["cid"],
                                     "area_px": c["area"]})
                    counts[label] += 1
                    if label == "uncertain":
                        continue  # 拿不准的只入样本库，不进训练集
                    # 进训练集
                    fname = f"negmix_{name}.png"
                    cv2.imwrite(str(out_img / fname),
                                (img * 255).astype(np.uint8))
                    images.append({"id": next_id, "file_name": fname,
                                   "width": WINDOW, "height": WINDOW})
                    if label == "positive":
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
                    next_id += 1

    (LIB / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    with open(OUT / "annotations.json", "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": coco["categories"]}, f)
    print(f"负样本库：{counts}（含 uncertain）→ {LIB}")
    print(f"训练集：Tao {len(coco['images'])} + 裁块 "
          f"{counts['positive'] + counts['negative']} → {OUT}")


if __name__ == "__main__":
    main()
