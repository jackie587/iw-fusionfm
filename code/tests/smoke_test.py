"""冒烟测试：用合成条纹数据跑通 训练→推理→后处理→评估 全链路。

不需要真实数据与账号，CPU 也能跑（有 GPU 自动用 GPU）。
用法（在 code/ 目录下）：
    python tests/smoke_test.py [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import dice_score, iou_score, stripe_recall_with_tolerance
from inference.postprocess import postprocess
from inference.predict import predict_large_image
from models.segmentation.unet import build_unet
from training.losses.composite import CompositeLoss


def make_synthetic_stripes(n: int = 8, size: int = 256,
                           seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """造 n 张带弧形亮条纹的 [3,H,W] 假图和对应掩膜，模拟内波条纹。"""
    rng = np.random.RandomState(seed)
    imgs, masks = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n):
        img = rng.rand(3, size, size).astype(np.float32) * 0.3  # 背景
        mask = np.zeros((size, size), np.float32)
        cx, cy = rng.randint(40, size - 40, 2)
        for k in range(3):  # 3 条同心弧条纹
            r = 30 + k * 18
            band = np.abs(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - r) < 2.5
            mask[band] = 1.0
            img[:, band] += 0.5
        imgs.append(np.clip(img, 0, 1))
        masks.append(mask[None])
    return torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(masks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"[smoke] device = {device}")

    # 1. 模型 + 损失
    model = build_unet({"in_channels": 3, "out_channels": 1,
                        "base_channels": 16, "depth": 3}).to(device)
    criterion = CompositeLoss({
        "focal_tversky": {"alpha": 0.3, "beta": 0.7, "gamma": 1.333, "weight": 1.0},
        "cldice": {"weight": 0.5, "iters": 5},
        "mcc": {"weight": 0.5},
    })
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 2. 训练 3 个 epoch（小模型小数据，验证能收敛即可）
    imgs, masks = make_synthetic_stripes()
    loader = DataLoader(TensorDataset(imgs, masks), batch_size=4, shuffle=True)
    model.train()
    for epoch in range(3):
        tot = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss, detail = criterion(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += detail["total"]
        print(f"[smoke] epoch {epoch + 1}/3 loss={tot / len(loader):.4f}")

    # 3. 大图推理（比 tile 大的图验证切块拼接）
    big = make_synthetic_stripes(1, 700, seed=99)[0][0].numpy()
    prob = predict_large_image(model, big, tile_size=256, stride=192,
                               device=device, amp=(device.type == "cuda"))
    assert prob.shape == (700, 700), f"推理输出尺寸错误：{prob.shape}"
    print(f"[smoke] 大图推理 ok，概率范围 [{prob.min():.3f}, {prob.max():.3f}]")

    # 4. 后处理 + 评估
    truth = make_synthetic_stripes(1, 700, seed=99)[1][0, 0].numpy()
    result = postprocess(prob, threshold=0.3, min_area=10, min_length=10)
    pred = torch.from_numpy(result["mask"].astype(np.float32))
    t = torch.from_numpy(truth)
    print(f"[smoke] 波峰线条数 = {len(result['crest_lines'])}")
    print(f"[smoke] dice={dice_score(pred, t):.4f} "
          f"iou={iou_score(pred, t):.4f} "
          f"stripe_recall={stripe_recall_with_tolerance(result['mask'], truth):.4f}")

    print("[smoke] 全链路通过 ✔")


if __name__ == "__main__":
    main()
