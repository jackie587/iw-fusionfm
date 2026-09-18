"""诊断 v10：隔离 Δlogit 来源——m_now 带/不带 swot（FiLM 是否元凶）。"""
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
    idx = np.random.RandomState(0).choice(len(full), 8,
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
                l2 = m_now(img, swot=None).float()      # 绕过 FiLM
                l3 = m_init(img, swot=None).float()
            print(f'batch{i}: init↔now(swot) '
                  f'mean|Δ|={float((l0 - l1).abs().mean()):.4f} | '
                  f'now: swot↔无swot mean|Δ|={float((l1 - l2).abs().mean()):.4f} | '
                  f'init↔now(均无swot) mean|Δ|={float((l3 - l2).abs().mean()):.4f}')
            if i >= 1:
                break
        # FiLM γ/β 实际量级
        st = m_now.film_modulation_stats(sw)
        print('m_now film stats on last batch:', st)
        cond = m_now.swot_encoder(sw.float())
        gb = m_now.film.mlp(cond)
        g = 1 + gb[:, :m_now.film.channels]
        print(f'γ range [{float(g.min()):.3f},{float(g.max()):.3f}] '
              f'β range [{float(gb[:, m_now.film.channels:].min()):.3f},'
              f'{float(gb[:, m_now.film.channels:].max()):.3f}]')


if __name__ == '__main__':
    main()
