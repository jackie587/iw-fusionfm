"""诊断 v9：两个独立构造的 v7 初始模型间的 logits 差异（前向可复现性检查）。"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from torch.utils.data import DataLoader, Subset

from models.segmentation.swin_film import build_swin_film
from training.datasets.l2_dataset import L2PairDataset
from utils.config import load_yaml


def load_model(ckpt_path):
    mcfg = load_yaml('configs/model_swin_film.yaml')
    model = build_swin_film(mcfg)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = ckpt.get('model', ckpt)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items()
                           if k in msd and msd[k].shape == v.shape},
                          strict=False)
    return model.cuda().eval()


def main():
    m1 = load_model('../checkpoints/final/swin_unet_v7_sam/best.pth')
    m2 = load_model('../checkpoints/final/swin_unet_v7_sam/best.pth')
    full = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles',
                         split='val')
    idx = np.random.RandomState(0).choice(len(full), 16,
                                          replace=False).tolist()
    loader = DataLoader(Subset(full, idx), batch_size=4, shuffle=False,
                        num_workers=0)
    with torch.no_grad():
        for i, b in enumerate(loader):
            img = b['image'].cuda()
            sw = b['swot'].cuda()
            with torch.amp.autocast('cuda'):
                l1 = m1(img, swot=sw).float()
                l2 = m2(img, swot=sw).float()
            print(f'batch{i}: 同权重两次前向 logit 差 '
                  f'max={float((l1 - l2).abs().max()):.6f} '
                  f'mean={float((l1 - l2).abs().mean()):.6f}')
            if i >= 2:
                break


if __name__ == '__main__':
    main()
