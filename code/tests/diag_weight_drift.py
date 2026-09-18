"""诊断 v6：训练中 best.pth 与 v7 初始权重的逐层漂移量。"""
import sys

import torch

sys.path.insert(0, '.')


def main():
    a = torch.load('../checkpoints/final/swin_unet_v7_sam/best.pth',
                   map_location='cpu', weights_only=True)['model']
    b = torch.load('../checkpoints/experiments/swin_film_l2_v1/best.pth',
                   map_location='cpu', weights_only=True)['model']
    diffs = []
    for k in a:
        if k in b and a[k].shape == b[k].shape and a[k].is_floating_point():
            d = (a[k].float() - b[k].float()).abs()
            diffs.append((float(d.max()), float(d.mean()), k))
    diffs.sort(reverse=True)
    print('漂移最大的 15 层（max |Δw|, mean |Δw|）：')
    for mx, mn, k in diffs[:15]:
        print(f'  {mx:.4f} {mn:.6f}  {k}')
    import numpy as np
    print(f'全部 {len(diffs)} 层 max|Δw| 中位数 '
          f'{np.median([d[0] for d in diffs]):.6f}')


if __name__ == '__main__':
    main()
