"""GT 精标：SAM box/point prompt → 逐波峰级掩膜，逐张 QC 叠合图。

流程（对 selection.json 每张切块）：
1. VV 通道 p2~p98 拉伸成 8-bit 灰度（复制 3 通道喂 SAM）；
2. 自动 box prompt：
   - scene_event 块：场景概率图（results/scene_eval_v7sam/<scene>/prob.npy）
     对应切块区域 >0.5 的连通域外接框（min_area 15，pad 8px）；
   - l2_clean 块：v7 弱标签连通域外接框（同参数）；
   - 负样本块：无 prompt，掩膜全零；
3. SAM vit_b box prompt 逐框预测（multimask 取最优）→ 掩膜 ∩ 框 → 并集；
4. overrides.json 人工修正（第二轮起）：
   {"gt001": {"extra_boxes": [[x0,y0,x1,y1], ...],      # 追加框 prompt
              "drop_boxes":  [[x0,y0,x1,y1], ...],      # 按中心匹配移除自动框
              "points":      [[x,y,1], ...],            # 正/负点 prompt（配合框）
              "zero": true}}                            # 强制全零（判负）
5. 产出：
   - 数据集：data/datasets/L1_sar_stripe/gt_fine/images/<id>.npy（float32 VV，
     dB 归一化 [0,1]，与推理管线口径一致）、images_png/<id>.png（8-bit 拉伸
     显示图）、masks/<id>.png（uint8 0/255）；
   - QC：results/gt_fine/qc/<id>.png（VV | prompt框 | 掩膜叠合 三联图）。

用法（code/ 目录下）：
    python data_preprocessing/labeling/sam_label_gt.py [--only gt001 gt002]
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
TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
SCENE_EVAL = PROJECT_ROOT / "results/scene_eval_v7sam"
L2_ROOT = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"
QC_DIR = PROJECT_ROOT / "results/gt_fine/qc"
SAM_CKPT = PROJECT_ROOT / "checkpoints/sam/sam_vit_b_01ec64.pth"

PROB_TH = 0.5
MIN_AREA = 15
PAD = 8
TILE = 512


def vv_to_uint8(vv: np.ndarray) -> np.ndarray:
    finite = vv[np.isfinite(vv)]
    p2, p98 = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
    img = np.clip((np.nan_to_num(vv) - p2) / max(p98 - p2, 1e-3), 0, 1)
    return (img * 255).astype(np.uint8)


def boxes_from_binary(binary: np.ndarray) -> list[list[int]]:
    """二值图连通域 → 外接框（min_area 过滤，pad 8px，裁剪到图内）。"""
    from scipy import ndimage
    lab, n = ndimage.label(binary > 0)
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        if (lab[sl] == i).sum() < MIN_AREA:
            continue
        ys, xs = sl
        boxes.append([max(0, xs.start - PAD), max(0, ys.start - PAD),
                      min(TILE, xs.stop + PAD), min(TILE, ys.stop + PAD)])
    return boxes


def auto_boxes(s: dict) -> list[list[int]]:
    if s["source"] == "scene_event":
        prob = np.load(SCENE_EVAL / s["scene"] / "prob.npy")
        sub = np.asarray(prob[s["y"]:s["y"] + TILE, s["x"]:s["x"] + TILE],
                         np.float32)
        return boxes_from_binary(np.nan_to_num(sub) > PROB_TH)
    if s["source"] == "l2_clean":
        lab = np.load(L2_ROOT / s["l2_file"])["label"]
        return boxes_from_binary(lab > 0)
    return []


def match_drop(box: list[int], drops: list[list[int]]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for d in drops:
        if d[0] <= cx <= d[2] and d[1] <= cy <= d[3]:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    from segment_anything import SamPredictor, sam_model_registry

    selection = json.loads(
        (GT_ROOT / "selection.json").read_text(encoding="utf-8"))
    ov_file = GT_ROOT / "overrides.json"
    overrides = json.loads(ov_file.read_text(encoding="utf-8")) \
        if ov_file.exists() else {}

    (GT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    (GT_ROOT / "images_png").mkdir(parents=True, exist_ok=True)
    (GT_ROOT / "masks").mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    sam.to(device)
    predictor = SamPredictor(sam)

    for s in selection:
        if args.only and s["id"] not in args.only:
            continue
        ov = overrides.get(s["id"], {})
        # VV 来源：L2 块用 npz（与弱标签同源），场景块用 tiles npy
        if s.get("l2_file"):
            vv = np.load(L2_ROOT / s["l2_file"])["sar"][0].astype(np.float32)
        else:
            vv = np.load(TILES_ROOT / f"{s['scene']}.SAFE/images/"
                         f"{s['tile']}.npy")[0]
        vv = np.nan_to_num(vv, nan=0.0)
        u8 = vv_to_uint8(vv)
        rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)

        mask = np.zeros((TILE, TILE), np.uint8)
        boxes = [] if ov.get("zero") else auto_boxes(s)
        boxes = [b for b in boxes
                 if not match_drop(b, ov.get("drop_boxes", []))]
        boxes += ov.get("extra_boxes", [])
        pts = np.array([[p[0], p[1]] for p in ov.get("points", [])]) \
            if ov.get("points") else None
        plab = np.array([p[2] for p in ov.get("points", [])]) \
            if ov.get("points") else None

        if boxes or pts is not None:
            predictor.set_image(rgb)
            for b in boxes:
                kw = {}
                if pts is not None:
                    kw = {"point_coords": pts, "point_labels": plab}
                masks, scores, _ = predictor.predict(
                    box=np.array(b, dtype=np.float32),
                    multimask_output=True, **kw)
                best = masks[int(np.argmax(scores))]
                boxm = np.zeros((TILE, TILE), bool)
                boxm[b[1]:b[3], b[0]:b[2]] = True
                mask |= (best & boxm).astype(np.uint8)

        np.save(GT_ROOT / "images" / f"{s['id']}.npy",
                vv.astype(np.float32))
        cv2.imwrite(str(GT_ROOT / "images_png" / f"{s['id']}.png"), u8)
        cv2.imwrite(str(GT_ROOT / "masks" / f"{s['id']}.png"), mask * 255)

        # QC 三联图：VV | prompt 框 | 掩膜叠合
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
        axes[0].imshow(u8, cmap="gray")
        axes[0].set_title(f"{s['id']} VV  {s['scene'][-4:]}")
        axes[1].imshow(u8, cmap="gray")
        for b in boxes:
            axes[1].add_patch(plt.Rectangle(
                (b[0], b[1]), b[2] - b[0], b[3] - b[1],
                fill=False, edgecolor="lime", lw=1.0))
        if pts is not None:
            axes[1].scatter(pts[:, 0], pts[:, 1],
                            c=["lime" if l else "red" for l in plab], s=30,
                            marker="x")
        axes[1].set_title(f"prompt 框 ×{len(boxes)}")
        overlay = np.stack([u8, u8, u8], -1).astype(np.float32)
        overlay[..., 0] = np.clip(overlay[..., 0] + mask * 120, 0, 255)
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title(f"掩膜叠合 frac={mask.mean():.3f}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(QC_DIR / f"{s['id']}.png", dpi=110)
        plt.close(fig)
        print(f"{s['id']}: boxes={len(boxes)} frac={mask.mean():.3f}",
              flush=True)
    print("完成")


if __name__ == "__main__":
    main()
