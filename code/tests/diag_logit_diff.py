"""诊断 v8：同一批验证块上，v7 初始模型 vs 训练中 best.pth 的 logits 差异。"""
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
    m_init = load_model('../checkpoints/final/swin_unet_v7_sam/best.pth')
    m_now = load_model('../checkpoints/experiments/swin_film_l2_v1/best.pth')
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
                l0 = m_init(img, swot=sw).float()
                l1 = m_now(img, swot=sw).float()
            print(f'batch{i}: logit 差 max={float((l0-l1).abs().max()):.4f} '
                  f'mean={float((l0-l1).abs().mean()):.4f} | '
                  f'init prob_mean={float(l0.sigmoid().mean()):.4f} '
                  f'now prob_mean={float(l1.sigmoid().mean()):.4f}')
            if i >= 3:
                break


if __name__ == '__main__':
    main()
