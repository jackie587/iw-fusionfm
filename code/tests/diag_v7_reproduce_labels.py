"""诊断：v7 原始模型（未经 L2 微调）在 L2 验证块上复现自身弱标签的程度。"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from models.segmentation.swin_unet import build_swin_unet
from training.datasets.l2_dataset import L2PairDataset
from utils.config import load_yaml
from evaluation.metrics import dice_score


def main():
    mcfg = load_yaml('configs/model_swin_sam.yaml')
    model = build_swin_unet(mcfg)
    ckpt = torch.load('../checkpoints/final/swin_unet_v7_sam/best.pth',
                      map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()
    ds = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles', split='val')
    idxs = [i for i, m in enumerate(ds.entries)
            if m['label_quality'] == 'clean'][:6]
    idxs += [i for i, m in enumerate(ds.entries)
             if m['label_quality'] == 'none'][:4]
    dices = []
    with torch.no_grad():
        for i in idxs:
            b = ds[i]
            img = b['image'][None]
            logits = model(img)
            pred = (torch.sigmoid(logits) > 0.5).float()
            d = dice_score(pred, b['mask'][None])
            m = ds.entries[i]
            print(f"{m['label_quality']:>6} label_frac={m['label_frac']:.4f} "
                  f"pred_frac={float(pred.mean()):.4f} dice={d:.4f}")
            dices.append(d)
    print('mean dice:', np.mean(dices))


if __name__ == '__main__':
    main()
