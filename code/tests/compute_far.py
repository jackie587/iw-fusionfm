"""根据人工判读计算真实海况虚警率，并导出负样本/正样本裁块。

判读口径（2026-08-27 用户判读）：
- 0930 景：用户枚举了所有"是"的编号（含"是但不完整"），未提及的按"不是"处理
  （抽查 sheet_01 验证口径一致：船 #15/#19 未列入，真条纹 #16~18 列入）；
- 0604 景：#1~21 拿不准（单列），#22~107 全部"不是"；
- "是但不完整"算真阳性（目标存在，检测不完整是召回问题，不是虚警）。
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENES = {
    "S1A_IW_GRDH_1SDV_20230604T102558_20230604T102623_048835_05DF6C_178C": {
        "positive": [],
        "positive_partial": [],
        "uncertain": list(range(1, 22)),          # #1~21 拿不准
        "n_components": 107,
    },
    "S1A_IW_GRDH_1SDV_20230930T104133_20230930T104202_050556_0616E8_7E00": {
        "positive": ([11] + list(range(39, 51)) + list(range(51, 76))
                     + list(range(76, 101))
                     + [102, 104, 105, 106, 107, 108, 109, 110, 111, 113,
                        114, 115, 117, 118, 119, 120, 121, 122, 123, 124,
                        125, 127, 128, 129, 130, 131, 132, 135, 136, 137]),
        "positive_partial": [16, 17, 18, 22, 24, 33, 35, 37],  # 是但不完整
        "uncertain": [],
        "n_components": 137,
    },
}


def main():
    all_report = {}
    tot_tp = tot_fp = tot_unc = 0
    for scene, v in SCENES.items():
        comp_file = (PROJECT_ROOT / "results/scene_eval" / scene
                     / "review/components.json")
        picked = json.loads(comp_file.read_text())["picked"]
        # picked 的顺序即 sheet 编号 #1..#n（make_review_sheets 按顺序铺格）
        n = v["n_components"]
        assert len(picked) == n, (len(picked), n)
        pos = set(v["positive"]) | set(v["positive_partial"])
        unc = set(v["uncertain"])
        fp_idx = [i + 1 for i in range(n) if (i + 1) not in pos | unc]

        n_tp, n_fp, n_unc = len(pos), len(fp_idx), len(unc)
        far_all = n_fp / (n_tp + n_fp)               # 拿不准剔除
        far_cons = (n_fp + n_unc) / n                 # 拿不准算虚警（保守）
        all_report[scene] = {
            "n_picked": n, "tp": n_tp, "fp": n_fp, "uncertain": n_unc,
            "far_excl_uncertain": round(far_all, 4),
            "far_conservative": round(far_cons, 4),
            "tp_ids": sorted(pos), "fp_ids": fp_idx,
            "uncertain_ids": sorted(unc),
        }
        tot_tp += n_tp; tot_fp += n_fp; tot_unc += n_unc
        print(f"{scene[:44]}...: TP={n_tp} FP={n_fp} 拿不准={n_unc} "
              f"FAR={far_all:.1%}（保守 {far_cons:.1%}）")

    all_report["pooled"] = {
        "tp": tot_tp, "fp": tot_fp, "uncertain": tot_unc,
        "far_excl_uncertain": round(tot_fp / (tot_tp + tot_fp), 4),
        "far_conservative": round((tot_fp + tot_unc) / (tot_tp + tot_fp + tot_unc), 4),
    }
    print(f"合并: TP={tot_tp} FP={tot_fp} 拿不准={tot_unc} "
          f"FAR={all_report['pooled']['far_excl_uncertain']:.1%}"
          f"（保守 {all_report['pooled']['far_conservative']:.1%}）")

    out = PROJECT_ROOT / "results/scene_eval/human_review_far.json"
    out.write_text(json.dumps(all_report, ensure_ascii=False, indent=2))
    print("→", out)


if __name__ == "__main__":
    main()
