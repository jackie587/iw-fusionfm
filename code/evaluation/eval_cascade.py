"""级联评估：v9 提案 + 上下文校验器重打分，在冬季留出场景上的诚实 FAR。

流程（对每景）：
1. 读 v9 推理的 prob_masked.npy，0.35 阈值连通域提案（与判读管线同口径）；
2. 从 tiles 重建 VV 画布，对每个提案裁 512 中心块 + 1536 上下文块
   （降采样 384，与校验器训练输入一致）；
3. 校验器（winter_holdout_context，从未见过 round6）打分；
4. 统计：提案总数、各分数阈值下的残余虚警数（冬季留出 3 景人工判读
   全部为 FP，残余即虚警）。

用法（在 code/ 目录下）：
    python evaluation/eval_cascade.py \
        --prob-root ../results/scene_eval_v9holdout \
        --verifier ../checkpoints/experiments/context_verifier/winter_holdout_context/best.pt \
        --scenes <scene1> <scene2> <scene3>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "tests"))

from models.verification.context_verifier import Verifier  # noqa: E402
from make_review_sheets import build_vv_canvas  # noqa: E402
from build_negmix_dataset import crop_center  # noqa: E402

PROJECT_ROOT = CODE_ROOT.parent
TILES = PROJECT_ROOT / "data" / "processed" / "sar_tiles"
PROB_THR = 0.35
MIN_AREA = 20


def scene_proposals(scene: str, prob_root: Path) -> tuple[np.ndarray, list[dict]]:
    from scipy import ndimage
    prob = np.nan_to_num(
        np.load(prob_root / scene / "prob_masked.npy").astype(np.float32))
    lab, n = ndimage.label(prob > PROB_THR)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = [i + 1 for i in range(n) if sizes[i] >= MIN_AREA]
    cents = ndimage.center_of_mass(lab > 0, lab, keep)
    comps = [{"cid": c, "area": float(sizes[c - 1]), "cy": float(cy), "cx": float(cx)}
             for c, (cy, cx) in zip(keep, cents)]
    return prob, comps


@torch.no_grad()
def score_scene(model: Verifier, vv: np.ndarray, comps: list[dict],
                dev: torch.device, batch_size: int = 32) -> np.ndarray:
    feats = []
    for c in comps:
        cy, cx = int(c["cy"]), int(c["cx"])
        cen = crop_center(vv, cy, cx, 512, pad_value=None)
        ctx = crop_center(vv, cy, cx, 1536, pad_value=None)
        cen = torch.from_numpy(np.nan_to_num(cen).astype(np.float32))[None, None]
        ctx = torch.from_numpy(np.nan_to_num(ctx).astype(np.float32))[None, None]
        cen = F.interpolate(cen, size=384, mode="bilinear", align_corners=False)[0]
        ctx = F.interpolate(ctx, size=384, mode="bilinear", align_corners=False)[0]
        feats.append(torch.cat([cen, ctx], 0))
    scores = []
    for i in range(0, len(feats), batch_size):
        x = torch.stack(feats[i:i + batch_size]).to(dev)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            scores.append(torch.sigmoid(model(x)).float().cpu())
    return torch.cat(scores).numpy() if scores else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-root", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.verifier, map_location="cpu", weights_only=False)
    model = Verifier(ckpt["mode"], vv_mean=ckpt["vv_mean"],
                     vv_std=ckpt["vv_std"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"校验器 ← {args.verifier} (mode={ckpt['mode']}, ep{ckpt['epoch']})")

    report = {"prob_thr": PROB_THR, "min_area": MIN_AREA, "scenes": {}}
    for scene in args.scenes:
        prob, comps = scene_proposals(scene, Path(args.prob_root))
        print(f"[{scene[:44]}] 提案 {len(comps)} 个")
        if comps:
            vv = build_vv_canvas(TILES / f"{scene}.SAFE", prob.shape)
            scores = score_scene(model, vv, comps, dev)
        else:
            scores = np.array([])
        rec = {"n_proposals": len(comps),
               "score_quantiles": ([round(float(v), 4) for v in np.percentile(
                   scores, [50, 75, 90, 95, 99])] if len(scores) else []),
               "remain_at_thr": {str(t): int((scores >= t).sum())
                                 for t in (0.1, 0.3, 0.5, 0.7, 0.9)}}
        report["scenes"][scene] = rec
        for c, s in zip(comps, scores):
            c["score"] = round(float(s), 4)
        report["scenes"][scene]["components"] = comps
        print(f"  残余虚警: {rec['remain_at_thr']}")

    out = Path(args.out) if args.out else (
        Path(args.prob_root) / "cascade_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
