"""L2 配对数据 Sanity Check：SWOT 信号与 v7 弱标签是否相关。

在写融合模型之前先回答一个廉价问题：clean 块中，label 区（内波弱标签）
内/外的 SWOT |ssha| 与 |grad| 是否有系统性差异。若完全不相关，FiLM 融合
无从获益，实验结论应预期"无增益"。

做法：
- 只用 label_quality == "clean" 的块；
- label (512×512) 面积池化到 20×20 得到每格弱标签占比 frac；
- swot mask>0.5 的格为有效观测；frac>0.5 视为"label 内"，frac==0 视为"label 外"；
- 对比两组 |ssha|、|grad| 均值，并给 frac 与 |ssha|/|grad| 的 Spearman 相关。

用法（code/ 目录下）：
    python evaluation/sanity_swot_label_corr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

TILE_DIR = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"


def main() -> None:
    with open(TILE_DIR / "manifest_all.json", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = [m for m in manifest if m["label_quality"] == "clean"]
    print(f"clean 块数：{len(entries)}")

    in_ssha, out_ssha = [], []
    in_grad, out_grad = [], []
    frac_all, ssha_all, grad_all = [], [], []

    for i, m in enumerate(entries):
        d = np.load(TILE_DIR / m["file"])
        swot = d["swot"].astype(np.float32)          # (3,20,20) ssha/|grad|/mask
        label = d["label"].astype(np.float32)        # (512,512)
        # 512 不能被 20 整除，用面积插值近似池化到 swot 网格
        import cv2
        frac = cv2.resize(label, (20, 20), interpolation=cv2.INTER_AREA)
        valid = swot[2] > 0.5
        a_ssha = np.abs(swot[0])
        grad = swot[1]
        pos = valid & (frac > 0.5)
        neg = valid & (frac == 0)
        if pos.any():
            in_ssha.append(a_ssha[pos].mean())
            in_grad.append(grad[pos].mean())
        if neg.any():
            out_ssha.append(a_ssha[neg].mean())
            out_grad.append(grad[neg].mean())
        v = valid & (frac >= 0)  # 全部有效格，含 0<frac<=0.5 的边缘格
        frac_all.append(frac[v])
        ssha_all.append(a_ssha[v])
        grad_all.append(grad[v])
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i + 1}/{len(entries)}")

    frac_all = np.concatenate(frac_all)
    ssha_all = np.concatenate(ssha_all)
    grad_all = np.concatenate(grad_all)

    def spearman(x: np.ndarray, y: np.ndarray) -> float:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)

    print("\n===== Sanity Check：SWOT 信号 × v7 弱标签 =====")
    print(f"有效格总数：{frac_all.size}，label 内块数：{len(in_ssha)}，"
          f"label 外块数：{len(out_ssha)}")
    print(f"|ssha|  label内均值 {np.mean(in_ssha):.4f} m  vs  "
          f"label外均值 {np.mean(out_ssha):.4f} m")
    print(f"|grad|  label内均值 {np.mean(in_grad):.3e}     vs  "
          f"label外均值 {np.mean(out_grad):.3e}")
    print(f"Spearman(frac, |ssha|) = {spearman(frac_all, ssha_all):+.4f}")
    print(f"Spearman(frac, |grad|) = {spearman(frac_all, grad_all):+.4f}")
    # 通道量级（供数据集归一化参考）
    print(f"\n通道量级：ssha |median|={np.median(np.abs(ssha_all)):.4f} m，"
          f"grad median={np.median(grad_all):.3e}")


if __name__ == "__main__":
    main()
