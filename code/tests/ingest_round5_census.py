"""round5_census 入库：普查场景 AI 判读裁块【追加】进负样本库（不重建、不动旧内容）。

与 build_negmix3/4 的区别：那俩是全量重建脚本（rmtree 负样本库后再生成），
本脚本只做增量——读 results/scene_eval_census/human_review_census.json 的
判读结论，按 picked 坐标裁 512 块写入 data/datasets/negative_samples/<label>/，
并把条目追加到 manifest.json 末尾（round=5）。不重建 tao2022_negmix* 训练集
（v8 训练集构建另行进行）。

判读文件格式（与 human_review_far_round4.json 一致）：
    {scene_name: {"tp_ids": [...], "fp_ids": [...], "uncertain_ids": [...]}}
编号 = review/components.json 的 picked 顺序（1 起）。

用法（在 code/ 目录下）：python tests/ingest_round5_census.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "tests"))

from build_negmix_dataset import crop_center  # noqa: E402
from make_review_sheets import build_vv_canvas  # noqa: E402

WINDOW = 512
ROUND = 5
PROB_THR = 0.6          # 正样本伪标签阈值（与 round1 一致）
LIB = PROJECT_ROOT / "data/datasets/negative_samples"
VERDICTS = PROJECT_ROOT / "results/scene_eval_census/human_review_census.json"
SCENE_EVAL = PROJECT_ROOT / "results/scene_eval_census"


def main():
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
    manifest = json.loads((LIB / "manifest.json").read_text(encoding="utf-8"))
    existing = {e["file"] for e in manifest}
    counts = {"positive": 0, "negative": 0, "uncertain": 0}

    for scene, v in verdicts.items():
        if scene.startswith("_"):
            continue
        tiles_dir = (PROJECT_ROOT / "data/processed/sar_tiles"
                     / (scene + ".SAFE"))
        scene_out = SCENE_EVAL / scene
        picked = json.loads((scene_out / "review/components.json")
                            .read_text())["picked"]
        prob = np.nan_to_num(
            np.load(scene_out / "prob_masked.npy").astype(np.float32))
        vv = build_vv_canvas(tiles_dir, prob.shape)
        stag = re.search(r"\d{8}T\d{6}", scene).group(0)
        scode = scene[-4:]
        for label, key in (("positive", "tp_ids"),
                           ("negative", "fp_ids"),
                           ("uncertain", "uncertain_ids")):
            for sheet_no in v[key]:
                c = picked[sheet_no - 1]
                name = f"r{ROUND}_{stag}_{scode}_n{sheet_no:03d}"
                rel = f"{label}/{name}.npz"
                if rel in existing:
                    print(f"[skip] {rel} 已存在")
                    continue
                img = crop_center(vv, int(c["cy"]), int(c["cx"]), WINDOW,
                                  pad_value=None)
                img = np.nan_to_num(img, nan=0.0)
                det = crop_center(prob, int(c["cy"]), int(c["cx"]), WINDOW,
                                  pad_value=0.0) > PROB_THR
                np.savez_compressed(LIB / label / f"{name}.npz",
                                    vv=img.astype(np.float16),
                                    mask=det.astype(np.bool_))
                manifest.append({"file": rel, "label": label, "scene": scene,
                                 "round": ROUND, "cid": c["cid"],
                                 "area_px": c["area"]})
                existing.add(rel)
                counts[label] += 1
        print(f"{scene[:44]}: 本轮累计 {counts}")

    (LIB / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"round5_census 入库：{counts}，manifest 总条目 {len(manifest)}")


if __name__ == "__main__":
    main()
