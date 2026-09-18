"""诊断 v4：当前训练中的 best.pth 在留出天抽测——pred_frac vs label_frac。"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from torch.utils.data import DataLoader, Subset

from evaluation.metrics import dice_score
from models.segmentation.swin_film import build_swin_film
from training.datasets.l2_dataset import L2PairDataset
from utils.config import load_yaml


def main():
    mcfg = load_yaml('configs/model_swin_film.yaml')
    model = build_swin_film(mcfg)
    ckpt = torch.load('../checkpoints/experiments/swin_film_l2_v1/best.pth',
                      map_location='cpu', weights_only=True)
    sd = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"加载 epoch {ckpt.get('epoch', -1) + 1} best.pth, "
          f"缺失 {len(missing)} 多余 {len(unexpected)}")
    model.cuda().eval()

    full = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles', split='val')
    rng = np.random.RandomState(0)
    idx = rng.choice(len(full), 120, replace=False).tolist()
    loader = DataLoader(Subset(full, idx), batch_size=8, shuffle=False,
                        num_workers=2)
    dices, pfs, lfs, probs_mean = [], [], [], []
    with torch.no_grad():
        for b in loader:
            img = b['image'].cuda()
            mask = b['mask'].cuda()
            with torch.amp.autocast('cuda'):
                logits = model(img, swot=b['swot'].cuda())
            prob = torch.sigmoid(logits).float()
            pred = (prob > 0.5).float()
            dices.append(dice_score(pred, mask))
            pfs.append(float(pred.mean()))
            lfs.append(float(mask.mean()))
            probs_mean.append(float(prob.mean()))
    print(f'dice={np.mean(dices):.4f} pred_frac={np.mean(pfs):.4f} '
          f'label_frac={np.mean(lfs):.4f} prob_mean={np.mean(probs_mean):.4f}')


if __name__ == '__main__':
    main()
