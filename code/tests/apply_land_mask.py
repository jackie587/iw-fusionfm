"""应用陆地掩膜 + NaN 清理到全场景概率图，重出统计与叠合图。"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from data_preprocessing.sar_preprocess.land_mask import scene_land_mask

tiles_dir = Path(sys.argv[1])
scene_out = Path(sys.argv[2])

prob = np.load(scene_out / "prob.npy").astype(np.float32)
mask = scene_land_mask(tiles_dir)
assert mask.shape == prob.shape, (mask.shape, prob.shape)

nan_frac = float(np.isnan(prob).mean())
prob_clean = np.where(np.isnan(prob), 0.0, prob)
prob_clean[mask] = 0.0
np.save(scene_out / "prob_masked.npy", prob_clean.astype(np.float16))
np.save(scene_out / "land_mask.npy", mask)

ocean = ~mask & ~np.isnan(prob)
stats = {
    "nan_frac": nan_frac,
    "land_frac": float(mask.mean()),
    "ocean_px": int(ocean.sum()),
    "ocean_area_frac_gt0.6": float((prob_clean[ocean] > 0.6).mean()),
    "ocean_prob_median": float(np.median(prob_clean[ocean])),
}
print(json.dumps(stats, indent=2, ensure_ascii=False))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
scale = max(1, prob.shape[1] // 2000)
small = prob_clean[::scale, ::scale]
fig, ax = plt.subplots(figsize=(small.shape[1] / 100, small.shape[0] / 100))
ax.imshow(small, cmap="viridis", vmin=0, vmax=1)
ax.contour(small, levels=[0.6], colors="red", linewidths=0.5)
ax.axis("off")
fig.tight_layout(pad=0)
fig.savefig(scene_out / "prob_overview_masked.png", dpi=100)
plt.close(fig)
print("→", scene_out / "prob_overview_masked.png")
