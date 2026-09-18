"""冒烟计时 v2：真实 CompositeLoss + batch 8，估算每 epoch 耗时与显存。"""
import sys
import time

import torch

sys.path.insert(0, '.')
from torch.utils.data import DataLoader

from models.segmentation.swin_film import build_swin_film
from training.datasets.l2_dataset import L2PairDataset
from training.losses.composite import CompositeLoss
from utils.config import load_yaml


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bs = 8
    ds = L2PairDataset('../data/datasets/L2_s1_swot_matched/tiles',
                       split='train', swot_dropout=0.25, augment=True)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    mcfg = load_yaml('configs/model_swin_film.yaml')
    mcfg['use_gradient_checkpointing'] = False  # 显存富余，关掉换速度
    model = build_swin_film(mcfg).cuda()
    cfg = load_yaml('configs/train_swin_film.yaml')
    criterion = CompositeLoss(cfg['loss'])
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler()
    nstep = 12
    t0 = time.time()
    model.train()
    for i, batch in enumerate(loader):
        if i >= nstep:
            break
        img = batch['image'].cuda(non_blocking=True)
        mask = batch['mask'].cuda(non_blocking=True)
        sw = batch['swot'].cuda(non_blocking=True)
        with torch.amp.autocast('cuda'):
            logits = model(img, swot=sw)
            loss, _ = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / nstep
    n_train, n_val = 5485, 1472
    ep = dt * (n_train // bs) / 60
    print(f'{dt:.2f}s/step(batch{bs}) → 训练 {ep:.1f} min/epoch,'
          f' 20 epoch ≈ {ep * 20 / 60:.1f} h（不含验证）')
    print(f'显存峰值 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB')


if __name__ == '__main__':
    main()
