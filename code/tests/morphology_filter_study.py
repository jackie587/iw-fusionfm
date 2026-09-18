"""从两轮人工判读（505 个已标斑块）挖掘形态学过滤规则：
计算每个检出连通域的形态特征（面积/长宽比/实心度/细长度），
对比 TP 与 FP 的分布，找出能杀虚警又保住真内波的阈值。

用法（在 code/ 目录下）：python tests/morphology_filter_study.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROUNDS = [
    ("results/scene_eval/human_review_far.json", "results/scene_eval", 0.6),
    ("results/scene_eval_v3negmix/human_review_far_round2.json",
     "results/scene_eval_v3negmix", 0.85),
]


def shape_feats(mask):
    """mask: 单连通域布尔图 → (area, aspect, solidity, thinness)"""
    area = int(mask.sum())
    ys, xs = np.where(mask)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    aspect = float(max(h, w) / max(1, min(h, w)))
    # 实心度近似：面积/包围盒
    bbox_fill = float(area / (h * w))
    # 细长度：周长²/面积（圆≈12.6 最小；越细长越大）
    er = ndimage.binary_erosion(mask)
    perim = area - int(er.sum()) if er.any() else area
    thinness = float((perim ** 2) / max(area, 1))
    return area, aspect, bbox_fill, thinness


rows = []
for vf, sdir, thr in ROUNDS:
    verdicts = json.loads((PROJECT_ROOT / vf).read_text())
    for scene, v in verdicts.items():
        if scene == "pooled":
            continue
        scene_out = PROJECT_ROOT / sdir / scene
        picked = json.loads((scene_out / "review/components.json")
                            .read_text())["picked"]
        prob = np.nan_to_num(
            np.load(scene_out / "prob_masked.npy").astype(np.float32))
        lab, n = ndimage.label(prob > thr)
        objs = ndimage.find_objects(lab)  # 每个连通域的 bbox 切片
        for label, key in (("TP", "tp_ids"), ("FP", "fp_ids"),
                           ("UNC", "uncertain_ids")):
            for sheet_no in v[key]:
                c = picked[sheet_no - 1]
                sl = objs[c["cid"] - 1]
                feats = shape_feats(lab[sl] == c["cid"])  # 只取包围盒
                rows.append({"label": label, "scene": scene[17:25],
                             "area": feats[0], "aspect": feats[1],
                             "bbox_fill": feats[2], "thinness": feats[3]})

tp = [r for r in rows if r["label"] == "TP"]
fp = [r for r in rows if r["label"] == "FP"]
print(f"TP={len(tp)} FP={len(fp)}")

for key in ("area", "aspect", "bbox_fill", "thinness"):
    t = np.array([r[key] for r in tp])
    f = np.array([r[key] for r in fp])
    print(f"\n{key}: TP 中位 {np.median(t):.2f} (p10~p90 "
          f"{np.percentile(t,10):.2f}~{np.percentile(t,90):.2f}) | "
          f"FP 中位 {np.median(f):.2f} (p10~p90 "
          f"{np.percentile(f,10):.2f}~{np.percentile(f,90):.2f})")

# 扫描组合规则：area<=A_max 且 thinness>=T_min
print("\n规则扫描（FP 剔除率 / TP 保持率）：")
for amax in (20000, 50000, 100000, 200000):
    for tmin in (50, 100, 200):
        fp_kill = np.mean([(r["area"] > amax) or (r["thinness"] < tmin)
                           for r in fp])
        tp_keep = np.mean([(r["area"] <= amax) and (r["thinness"] >= tmin)
                           for r in tp])
        print(f"  area≤{amax:>6} 且 thinness≥{tmin:>3}: "
              f"杀 FP {fp_kill:.0%} / 保 TP {tp_keep:.0%}")

json.dump(rows, open(PROJECT_ROOT / "results/scene_eval/morphology_study.json",
                     "w"), ensure_ascii=False, indent=2)
