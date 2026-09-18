"""场景级阈值扫描：对两景真实场景概率图，扫多个阈值下的
虚警抑制率 / 真内波保持率 / 海面高响应面积占比，找最优工作点。

用法（在 code/ 目录下）：
    python tests/sweep_scene_threshold.py --suffix v3negmix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[round(0.40 + 0.05 * i, 2) for i in range(11)])
    args = ap.parse_args()

    verdicts = json.loads(
        (PROJECT_ROOT / "results/scene_eval/human_review_far.json")
        .read_text())

    # 预载两景概率图与各斑块窗口
    scenes = {}
    for scene, v in verdicts.items():
        if scene == "pooled":
            continue
        scene_out = PROJECT_ROOT / f"results/scene_eval_{args.suffix}" / scene
        picked = json.loads((PROJECT_ROOT / "results/scene_eval" / scene
                             / "review/components.json").read_text())["picked"]
        prob = np.nan_to_num(
            np.load(scene_out / "prob_masked.npy").astype(np.float32))
        mask = np.load(scene_out / "land_mask.npy")
        h, w = prob.shape
        windows = []  # (label, fire_mask_window)
        for label, ids in (("negative", v["fp_ids"]),
                           ("positive", v["tp_ids"]),
                           ("uncertain", v["uncertain_ids"])):
            for sheet_no in ids:
                c = picked[sheet_no - 1]
                cy, cx = int(c["cy"]), int(c["cx"])
                y0 = int(np.clip(cy - WINDOW // 2, 0, h - WINDOW))
                x0 = int(np.clip(cx - WINDOW // 2, 0, w - WINDOW))
                windows.append((label, prob[y0:y0 + WINDOW, x0:x0 + WINDOW]))
        scenes[scene] = {"prob": prob, "ocean": ~mask, "windows": windows}

    print(f"{'阈值':>6} {'虚警残留':>8} {'真波保持':>8} {'海面响应面积':>12}")
    sweep = []
    for t in args.thresholds:
        neg = [w for l, w in sum((s["windows"] for s in scenes.values()), [])
               if l == "negative"]
        pos = [w for l, w in sum((s["windows"] for s in scenes.values()), [])
               if l == "positive"]
        neg_fire = sum((w > t).any() for w in neg)
        pos_keep = sum((w > t).any() for w in pos)
        area = float(np.mean([(s["prob"][s["ocean"]] > t).mean()
                              for s in scenes.values()]))
        row = {"threshold": t, "fp_persist": neg_fire / len(neg),
               "tp_retain": pos_keep / len(pos), "ocean_area_frac": area}
        sweep.append(row)
        print(f"{t:>6.2f} {neg_fire}/{len(neg)} ({row['fp_persist']:.0%})"
              f"  {pos_keep}/{len(pos)} ({row['tp_retain']:.0%})"
              f"  {area:>11.2%}")

    out = PROJECT_ROOT / f"results/scene_eval_{args.suffix}/threshold_sweep.json"
    json.dump(sweep, open(out, "w"), indent=2)
    print("→", out)


if __name__ == "__main__":
    main()
