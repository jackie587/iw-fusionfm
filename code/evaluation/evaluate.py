"""评估入口：对一个 (图像, 掩膜) 数据集算全套指标并输出报告。

用法：
    python evaluation/evaluate.py --ckpt <权重> --model-config <模型yaml> --data <数据集目录>
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
from models.segmentation.swin_unet import build_swin_unet
from models.segmentation.unet import build_unet
from training.datasets.tao_dataset import TaoIWDataset
from utils.config import load_yaml


def build_model(model_config_path, ckpt_path):
    """按模型配置建模型并加载权重，返回 (model, device)。

    供 evaluate.py 与 threshold_sweep.py 共用，保证评估口径一致。
    """
    model_cfg = load_yaml(model_config_path)
    # 权重路径按项目根解析（配置里写的是相对项目根的路径）
    for key in ("imagenet_weights", "ssl4eo_weights",
                    "warm_start_unet", "warm_start_swin"):
        if model_cfg.get(key):
            p = Path(model_cfg[key])
            model_cfg[key] = str(p if p.is_absolute() else CODE_ROOT.parent / p)
    if model_cfg["name"] == "unet":
        model = build_unet(model_cfg)
    elif model_cfg["name"] == "swin_unet":
        model = build_swin_unet(model_cfg)
    elif model_cfg["name"] == "dual_fuse":
        from models.segmentation.dual_fuse import build_dual_fuse
        model = build_dual_fuse(model_cfg)
    else:
        raise ValueError(f"未知模型：{model_cfg['name']}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--data", required=True, help="Tao 数据集目录")
    ap.add_argument("--tolerance", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="结果 json 输出路径")
    args = ap.parse_args()

    model, device = build_model(args.model_config, args.ckpt)

    ds = TaoIWDataset(args.data)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"][0, 0].numpy()
            logits = model(img)
            pred = (torch.sigmoid(logits)[0, 0].cpu() > args.threshold).float()
            results.append({
                "image_id": int(batch["image_id"][0]),
                "dice": dice_score(pred, batch["mask"][0, 0]),
                "iou": iou_score(pred, batch["mask"][0, 0]),
                "far": false_alarm_rate(pred.numpy(), mask),
                "stripe_recall": stripe_recall_with_tolerance(
                    pred.numpy(), mask, args.tolerance),
            })

    agg = {k: float(np.nanmean([r[k] for r in results])) for k in
           ("dice", "iou", "far", "stripe_recall")}
    report = {"n_images": len(results), "aggregate": agg, "per_image": results}
    print(json.dumps(agg, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
