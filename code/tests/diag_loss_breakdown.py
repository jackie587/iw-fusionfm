"""诊断 v3：v7 热启动模型在 L2 训练批上的损失分解 + 梯度量级。

目的：找出为何 lr 5e-5 一个 epoch 就把 dice 从 0.80 打到 0.20。
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from torch.utils.data import DataLoader

from models.segmentation.swin_film import build_swin_film
from training.datasets.l2_dataset import L2PairDataset
from training.losses.composite import CompositeLoss
from utils.config import load_yaml


def main():
    mcfg = load_yaml('configs/model_swin_film.yaml')
    model = build_swin_film(mcfg)
    ckpt = torch.load('../checkpoints/final/swin_unet_v7_sam/best.pth',
                      map_location='cpu', weights_only=True)
    sd = ckpt.get('model', ckpt)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items()
                           if k in msd and msd[k].shape == v.shape},
                          strict=False)
    model.cuda().train()

    cfg = load_yaml('configs/train_swin_film.yaml')
    criterion = CompositeLoss(cfg['loss'])
    ds = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles',
                       split='train', swot_dropout=0.25)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)

    for i, b in enumerate(loader):
        if i >= 6:
            break
        img = b['image'].cuda()
        mask = b['mask'].cuda()
        sw = b['swot'].cuda()
        with torch.amp.autocast('cuda'):
            logits = model(img, swot=sw)
            loss, detail = criterion(logits, mask)
        grad_norm = torch.autograd.grad(loss, model.head.weight,
                                        retain_graph=False)[0].norm()
        pos = float(mask.mean())
        print(f"batch{i} label_frac={pos:.4f} "
              + " ".join(f"{k}={v:.4f}" for k, v in detail.items())
              + f" | head 梯度范数={float(grad_norm):.4f}")


if __name__ == '__main__':
    main()
