"""GT 掩膜生成（终版口径）：目视框定波包范围 → 波包级掩膜（框填充并集）。

口径说明（诚实记录，详见 README）：
- 曾尝试 SAM box prompt（对斑点噪声 SAR 块返回整框填充，无法给出条纹
  结构，见 results/gt_fine/qc/ 第一轮产物）与 DoG/傅里叶自动波峰提取
  （信噪比不足，掩膜呈碎斑，见 stripe_test/packet_test*.png），均否决；
- 最终口径：AI 目视在 1024px 上下文窗口（results/gt_fine/vv_ctx*/）圈定
  每个可见波包的范围框，掩膜 = 框并集（波包级，含峰间背景）。
  负样本掩膜全零。
- 产物：masks/<id>.png（uint8 0/255）+ mask_qc/<id>.png 三联验收图。

用法（code/ 目录下）：
    python data_preprocessing/labeling/crest_extract_gt.py [--only gt000]
（文件名保留 crest_extract 仅为延续会话上下文，实际为波包级框填充。）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"
QC_DIR = PROJECT_ROOT / "results/gt_fine/mask_qc"
TILE = 512


def enhance(vv: np.ndarray) -> np.ndarray:
    """QC 背景：块级 p2~p98 拉伸。"""
    finite = vv[np.isfinite(vv)]
    p2, p98 = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    return np.clip((np.nan_to_num(vv) - p2) / max(p98 - p2, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_list = json.loads(
        (GT_ROOT / "final_list.json").read_text(encoding="utf-8"))
    (GT_ROOT / "masks").mkdir(exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    for s in final_list:
        if args.only and s["id"] not in args.only:
            continue
        vv = np.load(GT_ROOT / "images" / f"{s['id']}.npy")
        vv = np.nan_to_num(vv, nan=0.0)
        boxes = s.get("boxes", [])
        mask = np.zeros((TILE, TILE), np.uint8)
        for x0, y0, x1, y1 in boxes:
            mask[y0:y1, x0:x1] = 1
        cv2.imwrite(str(GT_ROOT / "masks" / f"{s['id']}.png"), mask * 255)

        en = (enhance(vv) * 255).astype(np.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
        axes[0].imshow(en, cmap="gray")
        axes[0].set_title(f"{s['id']} VV {s['scene'][-4:]} {s['label']}")
        axes[1].imshow(en, cmap="gray")
        for x0, y0, x1, y1 in boxes:
            axes[1].add_patch(plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor="lime", lw=1.2))
        axes[1].set_title(f"boxes x{len(boxes)}")
        overlay = np.stack([en] * 3, -1).astype(np.float32)
        overlay[..., 0] = np.clip(overlay[..., 0] + mask * 120, 0, 255)
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title(f"mask frac={mask.mean():.3f}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(QC_DIR / f"{s['id']}.png", dpi=110)
        plt.close(fig)
        print(f"{s['id']}: frac={mask.mean():.3f}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
