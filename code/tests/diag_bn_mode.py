"""诊断 v5：best.pth 在 eval/train 两种 BN 模式下的留出天 dice 对比。"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from torch.utils.data import DataLoader, Subset

from evaluation.metrics import dice_score
from models.segmentation.swin_film import build_swin_film
from training.datasets.l2_dataset import L2PairDataset
from utils.config import load_yaml


def evaluate(model, loader, mode):
    getattr(model, mode)()
    dices, pfs = [], []
    with torch.no_grad():
        for b in loader:
            img = b['image'].cuda()
            mask = b['mask'].cuda()
            with torch.amp.autocast('cuda'):
                logits = model(img, swot=b['swot'].cuda())
            pred = (torch.sigmoid(logits) > 0.5).float()
            dices.append(dice_score(pred, mask))
            pfs.append(float(pred.mean()))
    print(f'{mode} 模式: dice={np.mean(dices):.4f} '
          f'pred_frac={np.mean(pfs):.4f}')


def main():
    mcfg = load_yaml('configs/model_swin_film.yaml')
    model = build_swin_film(mcfg)
    ckpt = torch.load('../checkpoints/experiments/swin_film_l2_v1/best.pth',
                      map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'], strict=False)
    model.cuda()
    full = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles',
                         split='val')
    rng = np.random.RandomState(0)
    idx = rng.choice(len(full), 120, replace=False).tolist()
    loader = DataLoader(Subset(full, idx), batch_size=8, shuffle=False,
                        num_workers=2)
    evaluate(model, loader, 'eval')
    evaluate(model, loader, 'train')


if __name__ == '__main__':
    main()
