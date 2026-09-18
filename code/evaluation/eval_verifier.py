"""汇总上下文校验器实验：指标 + report.json + 图。

读取 checkpoints/experiments/context_verifier/<split>_<mode>/predictions.json
（train_verifier.py 产物），输出到 results/context_verifier/：
  report.json          划分细节、样本数、全部指标、逐样本分数
  fig_roc.png          两套划分的 ROC（冬季留出走"训练正样本召回-冬季FAR"曲线）
  fig_season_split.png 按 round/季节分组的 FAR/TPR 柱状图（季节捷径检验）
  fig_samples_*.png    各实验测试集典型 TP/FN/FP/TN 样本（中心裁块 vs 上下文）

用法（在 code/ 目录下）：
    python evaluation/eval_verifier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "experiments" / "context_verifier"
OUT_ROOT = PROJECT_ROOT / "results" / "context_verifier"
SAMPLE_ROOT = PROJECT_ROOT / "data" / "datasets" / "negative_samples"

SPLITS = ["winter_holdout", "summer_holdout"]
MODES = ["center", "context"]
SEASON = {1: "summer", 2: "summer", 3: "summer", 4: "summer",
          5: "autumn", 6: "winter"}


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUC；单类返回 None。"""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # 同分取平均秩
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    avg = sums / cnt
    rpos = avg[inv[:len(pos)]].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc(scores: np.ndarray, labels: np.ndarray):
    """返回 (fpr, tpr, thresholds)，阈值从 +inf 扫到 -inf。"""
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    P, N = int(l.sum()), int((1 - l).sum())
    tps = np.cumsum(l)
    fps = np.cumsum(1 - l)
    keep = np.r_[np.diff(s) != 0, True]          # 每个唯一阈值取最后一点
    tpr = np.r_[0, tps[keep] / max(P, 1)]
    fpr = np.r_[0, fps[keep] / max(N, 1)]
    thr = np.r_[np.inf, s[keep]]
    return fpr, tpr, thr


def fpr_at_recall(scores: np.ndarray, labels: np.ndarray, recall: float):
    """测试集上达到目标召回的最小阈值及其 FPR；无正样本返回 None。"""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0:
        return None
    thr = np.quantile(pos, 1 - recall)           # 召回≥recall 的最高阈值
    r = float((pos >= thr).mean())
    f = float((neg >= thr).mean()) if len(neg) else None
    return {"threshold": float(thr), "recall": r, "fpr": f}


def far_at_threshold(neg_scores: np.ndarray, thr: float) -> float | None:
    if len(neg_scores) == 0:
        return None
    return float((neg_scores >= thr).mean())


def recall_at_threshold(pos_scores: np.ndarray, thr: float) -> float | None:
    if len(pos_scores) == 0:
        return None
    return float((pos_scores >= thr).mean())


def load_run(split: str, mode: str) -> dict | None:
    p = CKPT_ROOT / f"{split}_{mode}" / "predictions.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def eval_run(run: dict) -> dict:
    """单实验指标：测试集 AUC/FPR@召回、按 round 分组、季节捷径检验。"""
    preds = run["predictions"]
    test = preds["test"]
    tr_pos = np.array([p["score"] for p in preds["train"] + preds["val"]
                       if p["label"] == "positive"])
    ty = np.array([p["label"] == "positive" for p in test], int)
    ts = np.array([p["score"] for p in test])

    res = {"best_epoch": run["best_epoch"],
           "n_train": run["n_train"], "n_val": run["n_val"],
           "n_test": run["n_test"],
           "n_test_pos": int(ty.sum()), "n_test_neg": int((1 - ty).sum()),
           "test_auc": auc(ts, ty),
           "test_fpr_at_recall90": fpr_at_recall(ts, ty, 0.90),
           "test_fpr_at_recall95": fpr_at_recall(ts, ty, 0.95)}

    # 操作阈值：训练正样本召回 90%/95% 的阈值 → 测试集 FAR / 训练集外推
    thr90 = float(np.quantile(tr_pos, 0.10)) if len(tr_pos) else None
    thr95 = float(np.quantile(tr_pos, 0.05)) if len(tr_pos) else None
    res["train_recall90_threshold"] = thr90
    res["train_recall95_threshold"] = thr95
    if len(ty) and ty.sum() == 0 and thr90 is not None:  # 冬季留出：测试集全负
        res["test_far_at_trainrecall90"] = far_at_threshold(ts, thr90)
        res["test_far_at_trainrecall95"] = far_at_threshold(ts, thr95)

    # 按 round 分组（训练轮次标记 in_train，用 thr90 阈值）
    per_round = {}
    for name in ("train", "val", "test"):
        for p in preds[name]:
            r = str(p["round"])
            d = per_round.setdefault(r, {"season": SEASON[p["round"]],
                                         "in_train": name != "test",
                                         "n_pos": 0, "n_neg": 0,
                                         "scores_pos": [], "scores_neg": []})
            key = "pos" if p["label"] == "positive" else "neg"
            d[f"n_{key}"] += 1
            d[f"scores_{key}"].append(p["score"])
    for r, d in per_round.items():
        sp, sn = np.array(d.pop("scores_pos")), np.array(d.pop("scores_neg"))
        if thr90 is not None:
            d["tpr_at_trainrecall90"] = recall_at_threshold(sp, thr90)
            d["far_at_trainrecall90"] = far_at_threshold(sn, thr90)
        else:
            d["tpr_at_trainrecall90"] = d["far_at_trainrecall90"] = None
        d["far_at_0.5"] = far_at_threshold(sn, 0.5)
    res["per_round"] = dict(sorted(per_round.items()))
    return res


def fig_roc(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, split in zip(axes, SPLITS):
        for mode, style in [("center", "--"), ("context", "-")]:
            if mode not in results[split]["runs"]:
                continue
            run = results[split]["runs"][mode]
            preds = run["predictions"]["predictions"]
            if split == "winter_holdout":
                # 测试集全负：x=冬季测试FAR，y=训练正样本召回
                tr = [p for p in preds["train"] + preds["val"]
                      if p["label"] == "positive"]
                te = [p for p in preds["test"]]
                s = np.array([p["score"] for p in tr + te])
                l = np.array([1] * len(tr) + [0] * len(te))
                xlab, title = "winter test FAR", "winter holdout (test=round6, all-negative)"
            else:
                te = preds["test"]
                s = np.array([p["score"] for p in te])
                l = np.array([p["label"] == "positive" for p in te], int)
                xlab, title = "summer test FPR", "summer holdout (test=round1)"
            fpr, tpr, _ = roc(s, l)
            a = auc(s, l)
            ax.plot(fpr, tpr, style,
                    label=f"{mode} (AUC={a:.3f})" if a else mode)
        ax.set_xlabel(xlab); ax.set_ylabel("TPR / recall")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "fig_roc.png", dpi=130)
    plt.close(fig)


def fig_season_split(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for row, split in enumerate(SPLITS):
        avail = [m for m in MODES if m in results[split]["runs"]]
        if not avail:
            continue
        rounds = sorted(results[split]["runs"][avail[0]]["metrics"]["per_round"])
        x = np.arange(len(rounds))
        for col, (key, name) in enumerate(
                [("far_at_trainrecall90", "FAR @ train-recall-90 threshold"),
                 ("tpr_at_trainrecall90", "TPR @ train-recall-90 threshold")]):
            ax = axes[row][col]
            for k, (mode, color) in enumerate([("center", "#999999"),
                                               ("context", "#d62728")]):
                if mode not in results[split]["runs"]:
                    continue
                pr = results[split]["runs"][mode]["metrics"]["per_round"]
                vals = [pr[r][key] if pr[r][key] is not None else 0
                        for r in rounds]
                bars = ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38,
                              color=color, label=mode)
                for b, r in zip(bars, rounds):
                    if pr[r]["in_train"]:
                        b.set_alpha(0.35)
            ax.set_xticks(x)
            ax.set_xticklabels([f"r{r}\n{SEASON[int(r)]}" for r in rounds])
            ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, axis="y")
            ax.set_title(f"{split} - {name} (faded = in-train rounds)")
            if row == 0 and col == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "fig_season_split.png", dpi=130)
    plt.close(fig)


def fig_samples(split: str, mode: str, run: dict):
    """测试集典型样本：中心裁块 | 上下文 双联图，按类别各取 2 例。"""
    test = run["predictions"]["test"]
    pos = sorted([p for p in test if p["label"] == "positive"],
                 key=lambda p: -p["score"])
    neg = sorted([p for p in test if p["label"] == "negative"],
                 key=lambda p: -p["score"])
    picks = ([("TP", p) for p in pos[:2]] + [("FN", p) for p in pos[-2:]] +
             [("FP", p) for p in neg[:2]] + [("TN", p) for p in neg[-2:]])
    if not picks:
        return
    ctx = np.load(SAMPLE_ROOT / "context_cache" / "context.npy", mmap_mode="r")
    fig, axes = plt.subplots(len(picks), 2,
                             figsize=(7, 1.75 * len(picks)))
    for row, (cat, p) in enumerate(picks):
        vv = np.load(SAMPLE_ROOT / p["file"])["vv"].astype(np.float32)
        cx = np.array(ctx[p["index"]], np.float32)
        for col, (img, name) in enumerate([(vv, "center 512"), (cx, "context 1536")]):
            lo, hi = np.percentile(img, [2, 98])
            axes[row][col].imshow(img, cmap="gray", vmin=lo,
                                  vmax=max(hi, lo + 1e-3))
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(name)
        axes[row][0].set_ylabel(f"{cat} s={p['score']:.2f}", rotation=0,
                                labelpad=64, va="center")
    fig.suptitle(f"{split} / {mode} test samples")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / f"fig_samples_{split}_{mode}.png", dpi=120)
    plt.close(fig)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mapping_meta = json.load(open(SAMPLE_ROOT / "context_cache" / "meta.json",
                                  encoding="utf-8"))
    report = {"mapping_stats": mapping_meta["stats"],
              "notes": {
                  "input": "center=中心512裁块resize384单通道; "
                           "context=中心裁块+同中心1536上下文(4x4块均值降采样384)双通道, "
                           "mask通道未用(保证消融唯一差别是上下文)",
                  "border_rule": "npz中心裁块与上下文均按入库脚本的居中+反射填充规则; "
                                 "上下文图边缘的条带即反射填充伪影",
                  "winter_holdout_test": "round6测试集无正样本(冬季全为虚警), "
                                         "AUC未定义; 报训练正样本召回90/95阈值下的冬季FAR",
                  "season_check": "夏季真波保持率看summer_holdout的round1 held-out TPR; "
                                  "冬季虚警抑制看winter_holdout的round6 held-out FAR",
              },
              "runs": {}, "per_sample_scores": []}
    results = {}
    for split in SPLITS:
        results[split] = {"runs": {}}
        for mode in MODES:
            run = load_run(split, mode)
            if run is None:
                print(f"[跳过] {split}_{mode} 无 predictions.json")
                continue
            metrics = eval_run(run)
            results[split]["runs"][mode] = {"predictions": run,
                                            "metrics": metrics}
            report["runs"][f"{split}_{mode}"] = {
                "split": run["split"], "mode": run["mode"],
                "n_train": run["n_train"], "n_val": run["n_val"],
                "n_test": run["n_test"], "val_scenes": run["val_scenes"],
                "history": run["history"], "metrics": metrics}
            for part in ("train", "val", "test"):
                for p in run["predictions"][part]:
                    report["per_sample_scores"].append(
                        {**p, "run": f"{split}_{mode}", "part": part})
            fig_samples(split, mode, run)

    if not report["runs"]:
        raise SystemExit("没有任何实验产物，先跑 train_verifier.py")
    fig_roc(results)
    fig_season_split(results)

    # 季节捷径检验结论所需的核心数字汇总进 report
    check = {}
    for split in SPLITS:
        if split not in results or not results[split]["runs"]:
            continue
        check[split] = {}
        for mode in MODES:
            if mode not in results[split]["runs"]:
                continue
            m = results[split]["runs"][mode]["metrics"]
            check[split][mode] = {
                "test_auc": m["test_auc"],
                "test_fpr_at_recall90": m["test_fpr_at_recall90"],
                "test_fpr_at_recall95": m["test_fpr_at_recall95"],
                "test_far_at_trainrecall90": m.get("test_far_at_trainrecall90"),
                "test_far_at_trainrecall95": m.get("test_far_at_trainrecall95"),
            }
    report["season_shortcut_check"] = check

    with open(OUT_ROOT / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(check, ensure_ascii=False, indent=2))
    print(f"→ {OUT_ROOT}/report.json, fig_roc.png, fig_season_split.png, fig_samples_*.png")


if __name__ == "__main__":
    main()
