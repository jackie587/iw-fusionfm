"""把人工判读过的检出斑块导出为负样本/正样本库（npz 裁块 + manifest）。

负样本（确认虚警）是抗虚警训练的关键资产（Lu et al. 2025 经验：
真实海况虚警率 50%→6% 主要靠负样本库）。
输出：data/datasets/negative_samples/{positive,negative,uncertain}/<场景>_<编号>.npy
      + manifest.json
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from make_review_sheets import build_vv_canvas  # noqa: E402

WINDOW = 384

verdicts = json.loads(
    (PROJECT_ROOT / "results/scene_eval/human_review_far.json").read_text())
out_root = PROJECT_ROOT / "data/datasets/negative_samples"
manifest = []

for scene, v in verdicts.items():
    if scene == "pooled":
        continue
    tiles_dir = (PROJECT_ROOT / "data/processed/sar_tiles" / (scene + ".SAFE"))
    scene_out = PROJECT_ROOT / "results/scene_eval" / scene
    picked = json.loads((scene_out / "review/components.json")
                        .read_text())["picked"]
    prob = np.load(scene_out / "prob_masked.npy").astype(np.float32)
    prob = np.nan_to_num(prob, nan=0.0)
    vv = build_vv_canvas(tiles_dir, prob.shape)
    h, w = prob.shape

    groups = {"positive": v["tp_ids"], "negative": v["fp_ids"],
              "uncertain": v["uncertain_ids"]}
    for label, ids in groups.items():
        (out_root / label).mkdir(parents=True, exist_ok=True)
        for sheet_no in ids:
            c = picked[sheet_no - 1]  # sheet 编号 = picked 顺序
            cy, cx = int(c["cy"]), int(c["cx"])
            y0 = int(np.clip(cy - WINDOW // 2, 0, h - WINDOW))
            x0 = int(np.clip(cx - WINDOW // 2, 0, w - WINDOW))
            patch = vv[y0:y0 + WINDOW, x0:x0 + WINDOW]
            pprob = prob[y0:y0 + WINDOW, x0:x0 + WINDOW]
            if np.isnan(patch).any():
                patch = np.nan_to_num(patch, nan=0.0)
            name = f"{scene[:40]}_n{sheet_no:03d}.npz"
            np.savez_compressed(out_root / label / name,
                                vv=patch.astype(np.float16),
                                prob=pprob.astype(np.float16))
            manifest.append({"file": f"{label}/{name}", "label": label,
                             "scene": scene, "cid": c["cid"],
                             "area_px": c["area"]})
    print(f"{scene[:44]}: " +
          " ".join(f"{k}={len(ids)}" for k, ids in groups.items()))

(out_root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"共 {len(manifest)} 个裁块 → {out_root}")
