"""组装 GT 最终清单 final_list.json + 拷贝图像到 gt_fine/images|images_png。

输入：selection.json（第一轮 57 张）+ backfill2.json（补样 149 张）
+ 本文件内置的 AI 目视分级结论（KEEP_POS / KEEP_NEG / boxes）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
L2_ROOT = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"

# AI 目视验收结论（2026-09-01，vv_ctx + triage 图逐张判定）
# 正样本（含波包，值=框列表 [x0,y0,x1,y1] 切块局部坐标）
POS_BOXES: dict[str, list] = {
    # 第一轮种子（ctx 图判定）
    "gt000": [[0, 0, 512, 448]],
    "gt004": [[128, 96, 512, 448]],
    "gt006": [[256, 64, 512, 448]],
    "gt007": [[0, 0, 512, 320]],
    "gt012": [[0, 0, 512, 512]],
    "gt014": [[0, 0, 512, 470]],
    "gt015": [[64, 64, 512, 448]],
    "gt016": [[0, 0, 512, 512]],
    "gt017": [[64, 32, 448, 416]],
    "gt018": [[0, 0, 448, 448]],
    "gt020": [[64, 32, 448, 384]],
    "gt021": [[0, 0, 512, 448]],
    "gt022": [[32, 0, 512, 400]],
    "gt025": [[0, 0, 512, 448]],
    "gt028": [[0, 0, 512, 384]],
    "gt032": [[0, 32, 320, 448]],
    # 补样（triage 图判定清晰，整框提取）
    "gt132": [[0, 0, 512, 512]], "gt133": [[0, 0, 512, 512]],
    "gt134": [[0, 0, 512, 512]], "gt135": [[0, 0, 512, 512]],
    "gt136": [[0, 0, 512, 512]], "gt137": [[0, 0, 512, 512]],
    "gt138": [[0, 0, 512, 512]], "gt139": [[0, 0, 512, 512]],
    "gt140": [[0, 0, 512, 512]], "gt141": [[0, 0, 512, 512]],
    "gt142": [[256, 256, 512, 512]],
    "gt143": [[0, 0, 512, 512]], "gt144": [[0, 0, 512, 512]],
    "gt153": [[0, 0, 512, 512]], "gt154": [[0, 0, 512, 512]],
    "gt156": [[0, 0, 512, 512]], "gt157": [[0, 0, 512, 512]],
    "gt167": [[0, 0, 512, 512]],
    "gt178": [[0, 0, 384, 384]], "gt179": [[0, 0, 512, 256]],
    "gt181": [[0, 256, 512, 512]],
    "gt182": [[0, 0, 512, 512]], "gt183": [[0, 0, 512, 512]],
    "gt184": [[0, 0, 512, 512]], "gt185": [[0, 0, 512, 512]],
    "gt186": [[0, 0, 512, 512]], "gt187": [[0, 0, 512, 512]],
    "gt188": [[0, 0, 512, 512]], "gt189": [[0, 0, 512, 512]],
    "gt190": [[128, 128, 512, 512]], "gt191": [[0, 128, 512, 512]],
    "gt192": [[0, 0, 512, 512]],
    "gt196": [[0, 0, 512, 512]], "gt197": [[0, 0, 512, 512]],
    "gt212": [[0, 0, 512, 512]], "gt215": [[0, 0, 512, 512]],
    "gt232": [[0, 0, 512, 512]], "gt236": [[0, 0, 512, 512]],
    "gt237": [[0, 0, 512, 448]], "gt238": [[0, 0, 448, 448]],
    "gt209": [[0, 0, 448, 384]], "gt210": [[0, 0, 512, 384]],
    "gt211": [[0, 192, 512, 512]],
}
# 负样本（目视确认无波，掩膜全零）
NEG_IDS = ["gt047", "gt048", "gt049", "gt050", "gt051",
           "gt052", "gt053", "gt054", "gt055", "gt056"]

# 目视分级：A=条纹清晰充满画面；B=可见但较弱/局部
GRADE_A = {"gt012", "gt014", "gt016", "gt021", "gt022",
           "gt132", "gt133", "gt134", "gt135", "gt136", "gt137", "gt138",
           "gt139", "gt140", "gt141", "gt143", "gt144", "gt153", "gt156",
           "gt183", "gt192"}


def main() -> None:
    sel = json.loads((GT_ROOT / "selection.json").read_text(encoding="utf-8"))
    bf = json.loads(
        (PROJECT_ROOT / "results/gt_fine/backfill2.json").read_text(
            encoding="utf-8"))
    by_id = {s["id"]: s for s in sel + bf}

    (GT_ROOT / "images").mkdir(exist_ok=True)
    (GT_ROOT / "images_png").mkdir(exist_ok=True)
    (GT_ROOT / "masks").mkdir(exist_ok=True)

    # 西丛集补充（c1 候选 ctx 复判）：c1002/c1006 → gt300/gt301
    bf_west = json.loads(
        (PROJECT_ROOT / "results/gt_fine/vv_enh_cand/cands_c1.json")
        .read_text(encoding="utf-8"))
    west_add = {"c1002": ("gt300", [[0, 0, 512, 512]]),
                "c1006": ("gt301", [[128, 32, 512, 512]])}
    for c in bf_west:
        if c["id"] in west_add:
            gid, boxes = west_add[c["id"]]
            c = dict(c)
            c["id"] = gid
            by_id[gid] = c
            POS_BOXES[gid] = boxes

    final = []
    for gid in list(POS_BOXES) + NEG_IDS:
        s = dict(by_id[gid])
        if not s.get("id", "").startswith("gt"):
            s["id"] = gid
        s.pop("mean_prob", None)
        s["boxes"] = POS_BOXES.get(gid, [])
        s["label"] = "positive" if s["boxes"] else "negative"
        s["grade"] = ("A" if gid in GRADE_A else "B") \
            if s["label"] == "positive" else "neg"
        # VV 来源：L2 块用 npz，其余用场景 tiles
        if s.get("l2_file"):
            vv = np.load(L2_ROOT / s["l2_file"])["sar"][0].astype(np.float32)
        else:
            vv = np.load(TILES_ROOT / f"{s['scene']}.SAFE/images/"
                         f"{s['tile']}.npy")[0].astype(np.float32)
        np.save(GT_ROOT / "images" / f"{gid}.npy", vv)
        finite = vv[np.isfinite(vv)]
        p2, p98 = np.percentile(finite, [2, 98])
        u8 = (np.clip((np.nan_to_num(vv) - p2) / max(p98 - p2, 1e-3), 0, 1)
              * 255).astype(np.uint8)
        cv2.imwrite(str(GT_ROOT / "images_png" / f"{gid}.png"), u8)
        final.append(s)

    (GT_ROOT / "final_list.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    n_pos = sum(1 for s in final if s["label"] == "positive")
    print(f"final_list: {len(final)} 张（正 {n_pos} / 负 {len(final) - n_pos}）")


if __name__ == "__main__":
    main()
