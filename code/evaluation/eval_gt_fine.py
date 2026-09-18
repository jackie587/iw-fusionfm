"""独立精标 GT 子集（gt_fine）诚实评估。

gt_fine：71 块 512×512 单通道 VV（npy，与场景推理/训练同源同口径，
无归一化）+ 波包级框填充 mask（0/255 PNG）+ manifest.json（grade
A/B/neg 等 23 字段）。本脚本对单个模型评估：
- 像素级 dice / IoU / F1 / 连通域虚警率（FAR）；
- 容差条纹级检出率（默认 3 px）；
- 总体 + 按 grade 分层（A/B/neg）+ 逐块明细，输出 json；
- 可选输出 "VV | GT | 预测叠合" 三联图。

输入口径：单通道 VV 复制成 3 通道（v7 训练分布即 VV×3）；swin_film
模型额外喂 swot 全零（3×20×20），此时 FiLM 严格恒等退化。

用法（code/ 目录下）：
    python evaluation/eval_gt_fine.py \
        --ckpt ../checkpoints/final/swin_unet_v7_sam/best.pth \
        --model-config configs/model_swin_sam.yaml --tag v7 \
        --out ../results/gt_fine_eval/v7_eval.json --triptychs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

import numpy as np
import torch
from PIL import Image

from evaluation.evaluate import build_model
from evaluation.metrics import (dice_score, false_alarm_rate, f1_score,
                                iou_score, stripe_recall_with_tolerance)
from utils.config import load_yaml

GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"


def build_film_model(model_config_path, ckpt_path):
    """SwinFilmUNet 构建 + 权重加载（对齐 eval_film_experiments.py 的做法）。"""
    from models.segmentation.swin_film import build_swin_film
    model_cfg = load_yaml(model_config_path)
    model = build_swin_film(model_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[film] 加载 {len(sd) - len(unexpected)} 层，缺失 {len(missing)} 层")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device


def load_inputs(entry):
    """读一块 gt_fine：VV npy → 3 通道 tensor，mask PNG → 0/1 ndarray。"""
    vv = np.load(GT_ROOT / entry["image"]).astype(np.float32)
    vv = np.nan_to_num(vv, nan=0.0)
    img = torch.from_numpy(np.repeat(vv[None], 3, axis=0))[None]  # 1×3×H×W
    mask = (np.array(Image.open(GT_ROOT / entry["mask"])) > 127).astype(np.float32)
    return img, mask


def render_triptych(vv, mask, pred, prob, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = vv[np.isfinite(vv)]
    p2, p98 = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
    vis = np.clip((np.nan_to_num(vv) - p2) / max(p98 - p2, 1e-3), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(vis, cmap="gray")
    axes[0].set_title("VV")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    overlay = np.dstack([vis, vis, vis])
    overlay[..., 0] = np.maximum(overlay[..., 0], pred * 0.9)  # 预测=红
    overlay[..., 1] = np.maximum(overlay[..., 1], mask * 0.9)  # GT=绿（重叠→黄）
    axes[2].imshow(overlay)
    axes[2].set_title(f"pred∩GT 叠合（prob_max={prob.max():.2f}）")
    fig.suptitle(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def aggregate(records, keys=("dice", "iou", "f1", "far", "stripe_recall")):
    out = {"n": len(records)}
    for k in keys:
        vals = [r[k] for r in records if not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--tag", default=None, help="模型标签（三联图文件名用），"
                    "缺省取 ckpt 父目录名")
    ap.add_argument("--out", required=True, help="结果 json 输出路径")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tolerance", type=int, default=3)
    ap.add_argument("--triptychs", action="store_true",
                    help="输出三联图到 <out 目录>/triptychs/")
    ap.add_argument("--mask-subdir", default=None,
                    help="改用 gt_fine 下的替代掩膜目录（如 masks_percrest）；"
                         "无对应掩膜的块跳过")
    args = ap.parse_args()

    tag = args.tag or Path(args.ckpt).parent.name
    model_cfg = load_yaml(args.model_config)
    if model_cfg["name"] == "swin_film":
        model, device = build_film_model(args.model_config, args.ckpt)
        is_film = True
    else:
        model, device = build_model(args.model_config, args.ckpt)
        is_film = False

    manifest = json.load(open(GT_ROOT / "manifest.json", encoding="utf-8"))
    if args.mask_subdir:
        filtered = []
        for entry in manifest:
            alt = Path(args.mask_subdir) / Path(entry["mask"]).name
            if (GT_ROOT / alt).exists():
                entry = dict(entry, mask=str(alt).replace("\\", "/"))
                filtered.append(entry)
        print(f"[mask-subdir] {args.mask_subdir}: {len(filtered)}/{len(manifest)} 块有掩膜")
        manifest = filtered

    trip_dir = None
    if args.triptychs:
        trip_dir = Path(args.out).parent / "triptychs"
        trip_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with torch.no_grad():
        for entry in manifest:
            img, mask = load_inputs(entry)
            img = img.to(device)
            swot = (torch.zeros(1, model_cfg.get("swot_channels", 3), 20, 20,
                                device=device) if is_film else None)
            logits = model(img, swot=swot) if is_film else model(img)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob > args.threshold).astype(np.float32)
            pred_t = torch.from_numpy(pred)
            mask_t = torch.from_numpy(mask)

            rec = {
                "id": entry["id"],
                "grade": entry["grade"],
                "label": entry["label"],
                "scene": entry["scene"],
                "swot_available": entry["swot_available"],
                "dice": dice_score(pred_t, mask_t),
                "iou": iou_score(pred_t, mask_t),
                "f1": f1_score(pred_t, mask_t),
                "far": false_alarm_rate(pred, mask),
                "stripe_recall": stripe_recall_with_tolerance(
                    pred, mask, args.tolerance),
                "pred_area_frac": float(pred.mean()),
                "gt_area_frac": float(mask.mean()),
            }
            records.append(rec)
            print(f"[{entry['id']}] grade={entry['grade']} "
                  f"dice={rec['dice']:.3f} far={rec['far']:.3f} "
                  f"stripe_recall={rec['stripe_recall']}")

            if trip_dir is not None:
                render_triptych(
                    np.load(GT_ROOT / entry["image"]).astype(np.float32),
                    mask, pred, prob,
                    f"{entry['id']} [{tag}] grade={entry['grade']} "
                    f"dice={rec['dice']:.3f}",
                    trip_dir / f"{entry['id']}_{tag}.png")

    by_grade = {g: aggregate([r for r in records if r["grade"] == g])
                for g in ("A", "B", "neg")}
    positives = [r for r in records if r["label"] == "positive"]
    report = {
        "ckpt": str(args.ckpt),
        "model_config": str(args.model_config),
        "tag": tag,
        "threshold": args.threshold,
        "stripe_tolerance_px": args.tolerance,
        "swot_input": "zeros" if is_film else None,
        "n_images": len(records),
        "aggregate": aggregate(records),
        "aggregate_positives_only": aggregate(positives),
        "by_grade": by_grade,
        "per_image": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n===== [{tag}] gt_fine 评估汇总 =====")
    print(json.dumps({"all": report["aggregate"],
                      "pos_only": report["aggregate_positives_only"],
                      "by_grade": by_grade}, ensure_ascii=False, indent=2))
    print(f"结果已存 {out_path}")


if __name__ == "__main__":
    main()
