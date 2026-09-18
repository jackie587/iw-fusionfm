"""SWOT-FiLM 实验评估：融合组 vs SAR-only 对照组。

对两个实验的 best.pth 在留出天（20230830）验证集上计算：
- 全部验证块 dice/iou；
- 仅 clean 块（含弱标签正样本）dice/iou——none 块标签全 0，
  预测全负即 dice≈1，会稀释差异，故单独报；
- 融合模型的 FiLM 调制强度（|γ-1|、|β| 均值，0=恒等退化）。

用法（code/ 目录下）：
    python evaluation/eval_film_experiments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

from evaluation.metrics import dice_score, iou_score
from models.segmentation.swin_film import build_swin_film
from training.datasets.l2_dataset import L2PairDataset
from utils.config import load_yaml

EXPERIMENTS = ["swin_film_l2_v1", "swin_film_l2_v1_saronly"]


def build(exp: str):
    model_cfg = load_yaml(PROJECT_ROOT / "code/configs/model_swin_film.yaml")
    model = build_swin_film(model_cfg)
    ckpt = torch.load(PROJECT_ROOT / f"checkpoints/experiments/{exp}/best.pth",
                      map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{exp}] best.pth ← epoch {ckpt.get('epoch', '?') + 1}，"
          f"加载 {len(sd) - len(unexpected)} 层，缺失 {len(missing)} 层")
    return model.cuda().eval()


@torch.no_grad()
def run(model, loader, collect_film: bool):
    dices, ious, films = [], [], []
    for batch in loader:
        img = batch["image"].cuda()
        mask = batch["mask"].cuda()
        swot = batch["swot"].cuda()
        with torch.amp.autocast("cuda"):
            logits = model(img, swot=swot)
        pred = (torch.sigmoid(logits) > 0.5).float()
        dices.append(dice_score(pred, mask))
        ious.append(iou_score(pred, mask))
        if collect_film:
            films.append(model.film_modulation_stats(swot))
    out = {"dice": sum(dices) / len(dices), "iou": sum(ious) / len(ious)}
    if films:
        out["film_gamma_dev"] = sum(f["gamma_dev"] for f in films) / len(films)
        out["film_beta_abs"] = sum(f["beta_abs"] for f in films) / len(films)
    return out


def main() -> None:
    root = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"
    results = {}
    for exp in EXPERIMENTS:
        saronly = exp.endswith("saronly")
        full = L2PairDataset(root, split="val", swot_zero=saronly)
        clean_idx = [i for i, m in enumerate(full.entries)
                     if m["label_quality"] == "clean"]
        mk = lambda ds: DataLoader(ds, batch_size=8, shuffle=False,
                                   num_workers=4, pin_memory=True)
        model = build(exp)
        all_m = run(model, mk(full), collect_film=not saronly)
        clean_m = run(model, mk(Subset(full, clean_idx)),
                      collect_film=not saronly)
        results[exp] = {"all": all_m, "clean": clean_m}
        del model
        torch.cuda.empty_cache()

    print("\n===== 留出天 20230830 验证集对比 =====")
    print(f"{'实验':<28} {'dice(全部)':>10} {'dice(clean)':>11} "
          f"{'iou(clean)':>10} {'|γ-1|':>8} {'|β|':>8}")
    for exp, r in results.items():
        a, c = r["all"], r["clean"]
        g = f"{a.get('film_gamma_dev', 0):.4f}" if "film_gamma_dev" in a else "-"
        b = f"{a.get('film_beta_abs', 0):.4f}" if "film_beta_abs" in a else "-"
        print(f"{exp:<28} {a['dice']:>10.4f} {c['dice']:>11.4f} "
              f"{c['iou']:>10.4f} {g:>8} {b:>8}")


if __name__ == "__main__":
    main()
