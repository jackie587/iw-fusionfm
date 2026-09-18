"""推理阈值扫描：一次推理缓存概率图，在多个阈值上算全套指标。

背景：swin_unet_v2 精度高但条纹级召回偏低（83.3% vs U-Net 97.9%），
固定 0.5 阈值未必最优。本脚本对同一权重扫多个阈值，找 dice-召回平衡点。
逐图推理后立刻在各阈值上算指标并释放概率图，不占大内存。

用法：
    python evaluation/threshold_sweep.py --ckpt <权重> --model-config <模型yaml> \
        --data <数据集目录> --thresholds 0.30 0.35 ... --out <结果json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.metrics import (dice_score, false_alarm_rate, iou_score,
                                stripe_recall_with_tolerance)
from evaluation.evaluate import build_model  # 复用同一加载逻辑，保证口径一致
from training.datasets.tao_dataset import TaoIWDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--data", required=True, help="Tao 数据集目录")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[round(0.30 + 0.05 * i, 2) for i in range(9)])
    ap.add_argument("--tolerance", type=int, default=3)
    ap.add_argument("--out", default=None, help="结果 json 输出路径")
    args = ap.parse_args()

    model, device = build_model(args.model_config, args.ckpt)

    ds = TaoIWDataset(args.data)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    ths = sorted(args.thresholds)
    # per_threshold[t] = [per-image metric dict, ...]
    per_threshold = {t: [] for t in ths}

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img = batch["image"].to(device)
            mask = batch["mask"][0, 0]
            mask_np = mask.numpy()
            prob = torch.sigmoid(model(img))[0, 0].cpu()
            for t in ths:
                pred = (prob > t).float()
                per_threshold[t].append({
                    "image_id": int(batch["image_id"][0]),
                    "dice": dice_score(pred, mask),
                    "iou": iou_score(pred, mask),
                    "far": false_alarm_rate(pred.numpy(), mask_np),
                    "stripe_recall": stripe_recall_with_tolerance(
                        pred.numpy(), mask_np, args.tolerance),
                })
            if (bi + 1) % 50 == 0:
                print(f"  已推理 {bi + 1}/{len(ds)} 张", flush=True)
            del prob, img

    summary = []
    for t in ths:
        agg = {k: float(np.nanmean([r[k] for r in per_threshold[t]]))
               for k in ("dice", "iou", "far", "stripe_recall")}
        summary.append({"threshold": t, **agg})

    print(f"\n{'阈值':>6} {'dice':>7} {'IoU':>7} {'虚警率':>8} {'条纹检出率':>10}")
    for s in summary:
        print(f"{s['threshold']:>6.2f} {s['dice']:>7.3f} {s['iou']:>7.3f} "
              f"{s['far']:>8.1%} {s['stripe_recall']:>10.1%}")

    report = {"ckpt": args.ckpt, "model_config": args.model_config,
              "n_images": len(ds), "tolerance_px": args.tolerance,
              "sweep": summary, "per_image": per_threshold}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
