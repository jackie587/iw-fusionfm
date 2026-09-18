"""查看场景概率图/陆地掩膜覆盖情况（调试用）。"""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
p = np.load(d / "prob.npy")
m = np.load(d / "land_mask.npy")
print("shape", p.shape, "NaN 占比 %.1f%%" % (100 * np.isnan(p).mean()),
      "陆地占比 %.1f%%" % (100 * m.mean()))
ocean = ~m & ~np.isnan(p)
if ocean.sum():
    print("海面像素", int(ocean.sum()),
          "海面>0.6 占比 %.3f" % float((p[ocean] > 0.6).mean()))
else:
    print("海面像素 0 —— 全是陆地或 NaN")
