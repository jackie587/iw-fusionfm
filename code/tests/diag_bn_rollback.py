"""诊断 v7：把训练中 best.pth 的 BN running stats 换回 v7 原值，看 dice 是否恢复。

若是 → 坐实"BN 统计量漂移摧毁热启动"，修法为微调时冻结 BN。
"""
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
    model.load_state_dict(ckpt['model'], strict=False)
    # BN running stats 回滚到 v7 原值
    v7 = torch.load('../checkpoints/final/swin_unet_v7_sam/best.pth',
                    map_location='cpu', weights_only=True)['model']
    sd = model.state_dict()
    n = 0
    for k in list(sd):
        if ('running_mean' in k or 'running_var' in k) and k in v7 \
                and not k.startswith('swot_encoder'):
            sd[k] = v7[k]
            n += 1
    model.load_state_dict(sd)
    print(f'回滚 {n} 个 BN running stats')
    model.cuda().eval()

    full = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles',
                         split='val')
    rng = np.random.RandomState(0)
    idx = rng.choice(len(full), 120, replace=False).tolist()
    loader = DataLoader(Subset(full, idx), batch_size=8, shuffle=False,
                        num_workers=2)
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
    print(f'BN 回滚后: dice={np.mean(dices):.4f} pred_frac={np.mean(pfs):.4f}')


if __name__ == '__main__':
    main()
