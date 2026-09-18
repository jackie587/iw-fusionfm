"""第二轮人工判读落盘（3 景新场景，v3@0.85 检出）→ human_review_far_round2.json。

口径：用户枚举"不是内波"的编号，未列出的按"是"处理（无"拿不准"项）。
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 用户判读：这些编号"不是"内波
NEGATIVES = {
    "S1A_IW_GRDH_1SDV_20230701T104928_20230701T104947_049229_05EB67_E318":
        [1, 6, 14, 16, 22, 23, 27, 31, 32, 33, 36, 41, 42, 46, 52, 54, 57,
         60, 62, 67, 72, 81, 82, 83, 86, 88, 95, 96, 97, 98, 99, 100],
    "S1A_IW_GRDH_1SDV_20230706T105719_20230706T105744_049302_05EDAA_E4CD":
        [1, 3, 6, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26,
         27, 29, 30, 31, 34, 37, 40, 43, 44, 45, 58, 90, 95],
    "S1A_IW_GRDH_1SDV_20230706T105744_20230706T105800_049302_05EDAA_55E0":
        [1, 3, 5, 6, 8, 9, 10, 11, 12, 15, 17, 18, 19, 20, 22, 23, 24, 26,
         29, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 51, 52, 55, 59, 60,
         61, 62, 72, 73, 76],
}

report = {}
tot_tp = tot_fp = 0
for scene, neg_ids in NEGATIVES.items():
    picked = json.loads((PROJECT_ROOT / "results/scene_eval_v3negmix" / scene
                         / "review/components.json").read_text())["picked"]
    n = len(picked)
    neg = sorted(set(neg_ids))
    assert max(neg) <= n, f"{scene}: 编号 {max(neg)} 超出检出数 {n}"
    pos = [i + 1 for i in range(n) if (i + 1) not in set(neg)]
    far = len(neg) / n
    report[scene] = {"n_picked": n, "tp": len(pos), "fp": len(neg),
                     "uncertain": 0, "far": round(far, 4),
                     "tp_ids": pos, "fp_ids": neg, "uncertain_ids": [],
                     "threshold": 0.85, "model": "swin_unet_v3_negmix"}
    tot_tp += len(pos)
    tot_fp += len(neg)
    print(f"{scene[:44]}...: 检出 {n} → TP={len(pos)} FP={len(neg)} "
          f"FAR={far:.1%}")
report["pooled"] = {"tp": tot_tp, "fp": tot_fp,
                    "far": round(tot_fp / (tot_tp + tot_fp), 4)}
print(f"合并: TP={tot_tp} FP={tot_fp} FAR={tot_fp / (tot_tp + tot_fp):.1%}")

out = PROJECT_ROOT / "results/scene_eval_v3negmix/human_review_far_round2.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("→", out)
