"""诊断 v2：v7 热启动的 SwinFilmUNet（未训练）在留出天验证集上的 dice 抽测。"""
import sys

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
    ckpt = torch.load('../checkpoints/final/swin_unet_v7_sam/best.pth',
                      map_location='cpu', weights_only=True)
    sd = ckpt.get('model', ckpt)
    msd = model.state_dict()
    sub = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(sub, strict=False)
    model.cuda().eval()

    full = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles', split='val')
    import numpy as np
    rng = np.random.RandomState(0)
    idx = rng.choice(len(full), 120, replace=False).tolist()
    loader = DataLoader(Subset(full, idx), batch_size=8, shuffle=False,
                        num_workers=2)
    dices = []
    with torch.no_grad():
        for b in loader:
            img = b['image'].cuda()
            mask = b['mask'].cuda()
            with torch.amp.autocast('cuda'):
                logits = model(img, swot=b['swot'].cuda())
            pred = (torch.sigmoid(logits) > 0.5).float()
            dices.append(dice_score(pred, mask))
    print(f'v7 热启动（未训练）留出天抽测 dice = {sum(dices)/len(dices):.4f} '
          f'(120 块)')


if __name__ == '__main__':
    main()
