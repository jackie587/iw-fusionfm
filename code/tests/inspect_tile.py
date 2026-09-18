"""查看切块张量与 meta 结构（调试一次性脚本）。"""
import json
import sys
from pathlib import Path

import numpy as np

t = Path(sys.argv[1])
scene = t.name.removesuffix(".SAFE")  # 目录名带 .SAFE，文件名不带
a = np.load(t / "images" / f"{scene}_y00000_x00000.npy")
m = json.loads((t / "meta" / f"{scene}_y00000_x00000.json").read_text())
print("tile:", a.shape, a.dtype,
      [round(float(a[i].mean()), 3) for i in range(a.shape[0])])
print("meta keys:", sorted(m.keys()))
print({k: m[k] for k in sorted(m)[:10]})
