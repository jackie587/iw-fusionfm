"""诊断 v11：逐层对比 m_init vs m_now 的中间特征，定位 Δlogit 爆点。"""
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


@torch.no_grad()
def trunk(model, img):
    f1, f2, f3, f4 = model.encoder(img)
    if model.use_strip:
        f4 = model.strip_bot(f4)
        f3 = model.strip_mid(f3)
    d3 = model.dec3(f4, f3)
    d2 = model.dec2(d3, f2)
    d1 = model.dec1(d2, f1)
    d0 = model.dec0(d1, None)
    out = model.head(d0)
    return [f1, f2, f3, f4, d3, d2, d1, d0, out]


def main():
    m_init = load_model('../checkpoints/final/swin_unet_v7_sam/best.pth')
    m_now = load_model('../checkpoints/experiments/swin_film_l2_v1/best.pth')
    full = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles',
                         split='val')
    idx = np.random.RandomState(0).choice(len(full), 4,
                                          replace=False).tolist()
    loader = DataLoader(Subset(full, idx), batch_size=4, shuffle=False,
                        num_workers=0)
    names = ['f1', 'f2', 'f3', 'f4', 'd3', 'd2', 'd1', 'd0', 'logits']
    b = next(iter(loader))
    img = b['image'].cuda()
    with torch.amp.autocast('cuda'):
        a = trunk(m_init, img)
        c = trunk(m_now, img)
    for n, x, y in zip(names, a, c):
        x, y = x.float(), y.float()
        rel = float((x - y).abs().mean() / (x.abs().mean() + 1e-9))
        print(f'{n:>6}: 特征均值 {float(x.abs().mean()):.4f} '
              f'Δmean {float((x - y).abs().mean()):.4f} 相对 {rel:.4f}')


if __name__ == '__main__':
    main()
