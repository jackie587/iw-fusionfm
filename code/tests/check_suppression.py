"""微调效果核心检验：人工判读斑块上的虚警抑制率 / 真目标保持率。

对新模型（或任意概率图）逐斑块核查：
- 负样本（确认虚警）：窗口内不再有 >阈值 像素 = 抑制成功；
- 正样本（真内波）：窗口内仍有 >阈值 像素 = 保持检出。
不需要重新人工判读——位置锚定旧斑块。

用法（在 code/ 目录下）：
    python tests/check_suppression.py --suffix v3negmix
（读 ../results/scene_eval_<suffix>/<场景>/prob_masked.npy）
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
    ap.add_argument("--suffix", required=True,
                    help="scene_eval_<suffix> 目录后缀")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    verdicts = json.loads(
        (PROJECT_ROOT / "results/scene_eval/human_review_far.json")
        .read_text())

    rows = []
    for scene, v in verdicts.items():
        if scene == "pooled":
            continue
        scene_out = PROJECT_ROOT / f"results/scene_eval_{args.suffix}" / scene
        picked = json.loads((PROJECT_ROOT / "results/scene_eval" / scene
                             / "review/components.json").read_text())["picked"]
        prob = np.nan_to_num(
            np.load(scene_out / "prob_masked.npy").astype(np.float32))
        h, w = prob.shape

        for label, ids in (("negative", v["fp_ids"]),
                           ("positive", v["tp_ids"]),
                           ("uncertain", v["uncertain_ids"])):
            for sheet_no in ids:
                c = picked[sheet_no - 1]
                cy, cx = int(c["cy"]), int(c["cx"])
                y0 = int(np.clip(cy - WINDOW // 2, 0, h - WINDOW))
                x0 = int(np.clip(cx - WINDOW // 2, 0, w - WINDOW))
                win = prob[y0:y0 + WINDOW, x0:x0 + WINDOW]
                fire_frac = float((win > args.threshold).mean())
                rows.append({"scene": scene[:24], "label": label,
                             "sheet_no": sheet_no, "fire_frac": fire_frac,
                             "fires": fire_frac > 0})

    neg = [r for r in rows if r["label"] == "negative"]
    pos = [r for r in rows if r["label"] == "positive"]
    unc = [r for r in rows if r["label"] == "uncertain"]
    neg_fire = sum(r["fires"] for r in neg)
    pos_keep = sum(r["fires"] for r in pos)
    print(f"负样本（原虚警 {len(neg)} 个）：仍误报 {neg_fire} → "
          f"抑制率 {1 - neg_fire / len(neg):.1%}")
    print(f"正样本（真内波 {len(pos)} 个）：仍检出 {pos_keep} → "
          f"保持率 {pos_keep / len(pos):.1%}")
    if unc:
        print(f"拿不准（{len(unc)} 个）：仍报警 "
              f"{sum(r['fires'] for r in unc)}")
    out = PROJECT_ROOT / f"results/scene_eval_{args.suffix}/suppression.json"
    json.dump({"threshold": args.threshold, "rows": rows},
              open(out, "w"), ensure_ascii=False, indent=2)
    print("→", out)


if __name__ == "__main__":
    main()
