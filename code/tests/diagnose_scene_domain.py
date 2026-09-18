"""诊断真实场景推理失效原因：
1. 切块 NaN 来源统计；
2. 当前 VV dB 归一化 vs 显示式百分位拉伸 vs 训练集图像 的分布对比；
3. 同一切块在两种归一化下的模型响应对比图。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from evaluation.evaluate import build_model

tiles_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "../data/processed/sar_tiles/S1A_IW_GRDH_1SDV_20230604T102558_20230604T102623_048835_05DF6C_178C.SAFE")

files = sorted((tiles_dir / "images").glob("*.npy"))
print(f"共 {len(files)} 块，抽查 NaN 与 VV 分布...")

nan_tiles, vv_vals = 0, []
for f in files[::50]:  # 抽 1/50
    a = np.load(f)
    if np.isnan(a).any():
        nan_tiles += 1
    vv_vals.append(np.nan_to_num(a[0], nan=0.0).flatten()[::97])
vv_all = np.concatenate(vv_vals)
print(f"含 NaN 的块：{nan_tiles}/{len(files[::50])}（抽样）")
print(f"VV(归一化后) 分布: min={vv_all.min():.3f} p5={np.percentile(vv_all,5):.3f} "
      f"中位={np.median(vv_all):.3f} p95={np.percentile(vv_all,95):.3f} max={vv_all.max():.3f}")

# 训练集图像分布参照
from training.datasets.tao_dataset import load_image
for p in ["../data/datasets/L1_sar_stripe/tao2022/images",
          "../data/datasets/L1_sar_stripe/s1_iw_2023/images"]:
    d = Path(p)
    fs = sorted(d.iterdir())[:20]
    vals = np.concatenate([load_image(f)[..., 0].flatten()[::97] for f in fs])
    print(f"{d.parent.name} 图像分布: p5={np.percentile(vals,5):.3f} "
          f"中位={np.median(vals):.3f} p95={np.percentile(vals,95):.3f}")

# 同一切块两种归一化的模型响应对比
model, device = build_model(
    CODE_ROOT / "configs/model_swin_strip.yaml",
    PROJECT_ROOT / "checkpoints/experiments/swin_unet_v2/best.pth")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 场景级 p2/p98（用抽样估计）
p2, p98 = np.percentile(vv_all, 2), np.percentile(vv_all, 98)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for j, f in enumerate(files[::900][:3]):  # 抽 3 块
    a = np.load(f)
    vv = np.nan_to_num(a[0], nan=0.0)
    disp = np.clip((vv - p2) / (p98 - p2 + 1e-8), 0, 1)
    outs = []
    for inp in (vv, disp):
        x = torch.from_numpy(inp[None].repeat(3, 0)[None]).float().to(device)
        with torch.no_grad():
            outs.append(torch.sigmoid(model(x))[0, 0].cpu().numpy())
    axes[j, 0].imshow(vv, cmap="gray", vmin=0, vmax=1); axes[j, 0].set_title("VV dB归一化")
    axes[j, 1].imshow(outs[0], cmap="viridis", vmin=0, vmax=1)
    axes[j, 1].set_title(f"响应 mean={outs[0].mean():.2f}")
    axes[j, 2].imshow(disp, cmap="gray", vmin=0, vmax=1); axes[j, 2].set_title("百分位拉伸")
    axes[j, 3].imshow(outs[1], cmap="viridis", vmin=0, vmax=1)
    axes[j, 3].set_title(f"响应 mean={outs[1].mean():.2f}")
    for ax in axes[j]:
        ax.axis("off")
out = PROJECT_ROOT / "results/figures/scene_domain_diagnosis.png"
fig.tight_layout()
fig.savefig(out, dpi=100)
print("→", out)
