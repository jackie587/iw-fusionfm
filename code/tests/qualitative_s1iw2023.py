"""跨数据集定性泛化演示：用 swin_unet_v2 对 S1-IW-2023 图像推理并出叠合图。

S1-IW-2023：742 张 1024×1024 RGB jpeg（已切块），与 Tao 2022 分布不同
（不同海区/成像条件），用于检验模型泛化。标注文件未到，先出定性结果。

用法（在 code/ 目录下）：
    python tests/qualitative_s1iw2023.py --n 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from PIL import Image

from evaluation.evaluate import build_model

IMG_DIR = PROJECT_ROOT / "data/datasets/L1_sar_stripe/s1_iw_2023_raw/IW_Image"
OUT_DIR = PROJECT_ROOT / "results/figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="抽图数量（均匀抽）")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="swin_unet_v2 阈值扫描的 dice 最优点")
    args = ap.parse_args()

    model, device = build_model(
        CODE_ROOT / "configs/model_swin_strip.yaml",
        PROJECT_ROOT / "checkpoints/experiments/swin_unet_v2/best.pth")

    files = sorted(IMG_DIR.glob("*.jpg"), key=lambda p: int(p.stem))
    picks = [files[int(i)] for i in np.linspace(0, len(files) - 1, args.n)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = []
    for f in picks:
        img = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        pred = prob > args.threshold
        stats.append((f.name, float(prob.max()), float(pred.mean()) * 100))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img); axes[0].set_title(f"{f.name} 原图")
        axes[1].imshow(prob, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("条纹概率图")
        axes[2].imshow(img)
        axes[2].contour(pred, levels=[0.5], colors="red", linewidths=1.5)
        axes[2].set_title(f"检出（>{args.threshold}，占比 {pred.mean()*100:.1f}%）")
        for ax in axes:
            ax.axis("off")
        out = OUT_DIR / f"s1iw2023_generalize_{f.stem}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"{f.name}: prob_max={prob.max():.2f} 检出像素占比={pred.mean()*100:.1f}% → {out.name}")


if __name__ == "__main__":
    main()
