"""构建 tao2022_negmix3：Tao + 两轮人工判读裁块 + 留出场景噪声负样本（round3）。

round3 来源：4 景留出场景（07-03 A4EB/9E6B、07-08 A6CD/8714，未参与任何
训练）的 v4@0.6 检出连通域，经逐 sheet 目视确认全部为斑点噪声/雨团/
近岸虚警（含 1.3~28 万 px 的巨块），故全部作 negative 采样
（每场景上限 MAX_PER_SCENE），不挖正样本。
注：频域自动判据在此类噪声上失效（feature_study2.py：斑点与条纹同为
高细频能量，条纹波段方向性 coh_strip AUC 仅 0.80），auto_verdicts.json
的判读字段弃用，只取连通域坐标。

输出：
- data/datasets/negative_samples/{positive,negative,uncertain}/*.npz + manifest.json
- data/datasets/L1_sar_stripe/tao2022_negmix3/（images/ + annotations.json，
  裁块 id≥100000 只进训练集，验证集与 v2/v3/v4 严格一致）

用法（在 code/ 目录下）：python tests/build_negmix3.py
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
MAX_PER_SCENE = 150          # round3 每场景每类采样上限
TAO = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022"
OUT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/tao2022_negmix3"
LIB = PROJECT_ROOT / "data/datasets/negative_samples"

# (判读文件, scene_eval 目录, 正样本伪标签阈值) —— 前两轮人工判读
ROUNDS_HUMAN = [
    ("results/scene_eval/human_review_far.json", "results/scene_eval", 0.6),
    ("results/scene_eval_v3negmix/human_review_far_round2.json",
     "results/scene_eval_v3negmix", 0.85),
]
# round3：自动判读的留出场景（scene_eval_v4negmix2 下含 auto_verdicts.json，
# 排除参与过训练的 0604/0930 与全陆地 0F9E）
AUTO_SCENE_EVAL = PROJECT_ROOT / "results/scene_eval_v4negmix2"
AUTO_SCENES = ["A4EB", "9E6B", "A6CD", "8714"]
AUTO_PROB_THR = 0.6


def add_patch(vv, prob, cy, cx, prob_thr, label, name, round_no, scene,
              cid, area, counts, manifest, out_img, images, anns,
              next_id, next_ann_id):
    """裁 512 斑块：入样本库（全部）+ 训练集（非 uncertain）。返回下一个 id。"""
    img = crop_center(vv, cy, cx, WINDOW, pad_value=None)
    img = np.nan_to_num(img, nan=0.0)
    det = crop_center(prob, cy, cx, WINDOW, pad_value=0.0) > prob_thr
    np.savez_compressed(LIB / label / f"{name}.npz",
                        vv=img.astype(np.float16),
                        mask=det.astype(np.bool_))
    manifest.append({"file": f"{label}/{name}.npz", "label": label,
                     "scene": scene, "round": round_no, "cid": cid,
                     "area_px": area})
    counts[label] += 1
    if label == "uncertain":
        return next_id, next_ann_id  # 拿不准的只入库，不进训练集
    fname = f"negmix_{name}.png"
    cv2.imwrite(str(out_img / fname), (img * 255).astype(np.uint8))
    images.append({"id": next_id, "file_name": fname,
                   "width": WINDOW, "height": WINDOW})
    if label == "positive":
        ys, xs = np.where(det)
        if len(ys):
            anns.append({"id": next_ann_id, "image_id": next_id,
                         "category_id": 1, "iscrowd": 0,
                         "bbox": [int(xs.min()), int(ys.min()),
                                  int(xs.max() - xs.min() + 1),
                                  int(ys.max() - ys.min() + 1)],
                         "area": float(det.sum()), "segmentation": []})
            next_ann_id += 1
    return next_id + 1, next_ann_id


def main():
    if LIB.exists():
        shutil.rmtree(LIB)  # 全量重建，保持 512 统一规格
    for label in ("positive", "negative", "uncertain"):
        (LIB / label).mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = {"positive": 0, "negative": 0, "uncertain": 0}

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

    # ---- round 1/2：人工判读（与 build_negmix2 一致）----
    for ri, (vf, sdir, thr) in enumerate(ROUNDS_HUMAN, start=1):
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
                for sheet_no in v[key]:
                    c = picked[sheet_no - 1]
                    stag = re.search(r"\d{8}T\d{6}", scene).group(0)
                    name = f"r{ri}_{stag}_n{sheet_no:03d}"
                    next_id, next_ann_id = add_patch(
                        vv, prob, int(c["cy"]), int(c["cx"]), thr, label,
                        name, ri, scene, c["cid"], c["area"], counts,
                        manifest, out_img, images, anns, next_id, next_ann_id)

    # ---- round 3：留出场景噪声检出，全部作负样本 ----
    rng = np.random.RandomState(42)
    for scene_dir in sorted(AUTO_SCENE_EVAL.glob("S1A*")):
        if not any(s in scene_dir.name for s in AUTO_SCENES):
            continue
        av_file = scene_dir / "auto_verdicts.json"
        if not av_file.exists():
            print(f"跳过 {scene_dir.name}：无 auto_verdicts.json")
            continue
        comps = json.loads(av_file.read_text())["components"]
        if len(comps) > MAX_PER_SCENE:
            idx = rng.choice(len(comps), MAX_PER_SCENE, replace=False)
            comps = [comps[i] for i in sorted(idx)]
        tiles_dir = (PROJECT_ROOT / "data/processed/sar_tiles"
                     / (scene_dir.name + ".SAFE"))
        prob = np.nan_to_num(
            np.load(scene_dir / "prob_masked.npy").astype(np.float32))
        vv = build_vv_canvas(tiles_dir, prob.shape)
        stag = re.search(r"\d{8}T\d{6}", scene_dir.name).group(0)
        for j, c in enumerate(comps):
            name = f"r3_{stag}_{scene_dir.name[-4:]}_neg{j:03d}"
            next_id, next_ann_id = add_patch(
                vv, prob, int(c["cy"]), int(c["cx"]), AUTO_PROB_THR,
                "negative", name, 3, scene_dir.name, c["cid"], c["area"],
                counts, manifest, out_img, images, anns,
                next_id, next_ann_id)
        print(f"round3 {scene_dir.name[:44]}: {len(comps)} 个负样本")

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
