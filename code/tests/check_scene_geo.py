"""打印场景画布的经纬度范围（从切块 meta 的 transform 推）。"""
import json
import sys
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.transform import xy

meta_dir = Path(sys.argv[1]) / "meta"
f = sorted(meta_dir.glob("*.json"))[0]
m = json.loads(f.read_text())
t = Affine(*m["transform"])
# 画布大小：从 prob.npy 读
scene = Path(sys.argv[2])
prob = np.load(scene / "prob.npy", mmap_mode="r")
h, w = prob.shape
print("画布", h, w, "transform", list(t)[:6])
for r, c in [(0, 0), (0, w), (h, 0), (h, w)]:
    lon, lat = xy(t, r, c)
    print(f"  角点 (row={r},col={c}) → lon={lon:.3f} lat={lat:.3f}")
