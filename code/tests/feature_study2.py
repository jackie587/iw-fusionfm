"""自动判据 v2 特征研究：找能同时区分「真条纹 / 平滑虚警 / 斑点噪声虚警」的特征。

背景：fine_ratio（细尺度能量比）在人工斑块上 AUC 0.967，但留出场景的
大面积斑点噪声检出同样是高细频能量，被误判 positive（目视确认这些
检出全是噪声）。需要方向性/周期性特征补足。

三类样本：
- human_pos / human_neg：负样本库 505 个人工判读斑块；
- heldout：4 景留出场景 v4@0.6 检出（目视确认为噪声虚警主导）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "tests"))

from make_review_sheets import build_vv_canvas  # noqa: E402
from build_negmix_dataset import crop_center  # noqa: E402

WINDOW = 512


def spec_feats(vv: np.ndarray) -> dict:
    """频域特征：细频能量比 + 各频带方向集中度 + 峰值度。"""
    x = vv.astype(np.float32) - float(np.mean(vv))
    P = np.abs(np.fft.rfft2(x)) ** 2
    h, w = x.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    ang = np.arctan2(fy, fx)
    tot = max(P[(f > 0)].sum(), 1e-9)

    def band_stats(lo, hi):
        m = (f >= lo) & (f < hi)
        pw = P[m].ravel()
        a = ang[m].ravel()
        coh = np.abs((pw * np.exp(2j * a)).sum()) / max(pw.sum(), 1e-9)
        return P[m].sum() / tot, coh

    e_fine, coh_fine = band_stats(1 / 8, 1.0)        # 波长 <8px
    e_strip, coh_strip = band_stats(1 / 100, 1 / 25)  # 条纹波段 25~100px
    e_mid, coh_mid = band_stats(1 / 25, 1 / 8)
    # 峰值度：条纹波段最强频率 / 波段中位功率（周期性强度）
    m = (f >= 1 / 100) & (f < 1 / 25)
    peak = float(P[m].max() / max(np.median(P[m]), 1e-9))
    return {"fine": e_fine, "coh_fine": coh_fine, "strip": e_strip,
            "coh_strip": coh_strip, "mid": e_mid, "coh_mid": coh_mid,
            "peak": float(np.log10(peak + 1))}


def grad_feats(vv: np.ndarray) -> dict:
    """梯度结构张量方向性（条纹=梯度方向一致；斑点=随机）。"""
    from scipy import ndimage
    x = vv.astype(np.float32)
    gy, gx = np.gradient(x)
    for sigma, name in ((4, "g4"), (16, "g16")):
        jxx = ndimage.gaussian_filter(gx * gx, sigma)
        jyy = ndimage.gaussian_filter(gy * gy, sigma)
        jxy = ndimage.gaussian_filter(gx * gy, sigma)
        num = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
        den = jxx + jyy + 1e-9
        # 只在梯度足够强的像素上统计，避免平坦区噪声稀释
        strong = den > np.percentile(den, 75)
        yield name, float((num / den)[strong].mean())


def all_feats(vv: np.ndarray) -> dict:
    d = spec_feats(vv)
    for k, v in grad_feats(vv):
        d[k] = v
    return d


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    rows = []  # (group, feats)
    lib = PROJECT_ROOT / "data/datasets/negative_samples"
    manifest = json.loads((lib / "manifest.json").read_text())
    for m in manifest:
        vv = np.load(lib / m["file"])["vv"]
        rows.append((f"human_{m['label']}", all_feats(vv)))
    print(f"人工斑块 {len(rows)} 个特征已算", flush=True)

    rng = np.random.RandomState(0)
    eval_dir = PROJECT_ROOT / "results/scene_eval_v4negmix2"
    for scene_dir in sorted(eval_dir.glob("S1A*")):
        av = scene_dir / "auto_verdicts.json"
        if not av.exists():
            continue
        comps = json.loads(av.read_text())["components"]
        if len(comps) > 150:
            comps = [comps[i] for i in rng.choice(len(comps), 150, False)]
        tiles_dir = (PROJECT_ROOT / "data/processed/sar_tiles"
                     / (scene_dir.name + ".SAFE"))
        prob = np.load(scene_dir / "prob_masked.npy")
        vv = build_vv_canvas(tiles_dir, prob.shape)
        tag = scene_dir.name[-4:]
        for c in comps:
            patch = crop_center(vv, int(c["cy"]), int(c["cx"]), WINDOW,
                                pad_value=None)
            patch = np.nan_to_num(patch, nan=0.0)
            rows.append((f"heldout_{tag}", all_feats(patch)))
        print(f"{tag}: {len(comps)} 个", flush=True)

    keys = list(rows[0][1].keys())
    groups = np.array([r[0] for r in rows])
    F = np.array([[r[1][k] for k in keys] for r in rows])
    np.savez_compressed(PROJECT_ROOT / "results/feature_study.npz",
                        groups=groups, F=F, keys=keys)

    # 任务：human_positive 为正，其余（human_negative + heldout）为负
    y = (groups == "human_positive").astype(int)
    print(f"\n各特征 AUC（真条纹 vs 其余全部，n_pos={y.sum()}, "
          f"n_neg={(1-y).sum()}）：")
    for i, k in enumerate(keys):
        print(f"  {k:10s} {auc(y, F[:, i]):.3f}")
    print("\n组合：")
    combos = {
        "fine*g16": F[:, keys.index("fine")] * F[:, keys.index("g16")],
        "coh_fine": F[:, keys.index("coh_fine")],
        "g16/(1+fine...": F[:, keys.index("g16")],
        "fine*coh_fine": F[:, keys.index("fine")] * F[:, keys.index("coh_fine")],
        "peak*g16": F[:, keys.index("peak")] * F[:, keys.index("g16")],
    }
    for name, s in combos.items():
        print(f"  {name:14s} {auc(y, s):.3f}")
    # 分组中位值参考
    print("\n分组中位值：")
    for g in sorted(set(groups)):
        m = groups == g
        med = {k: round(float(np.median(F[m, i])), 3)
               for i, k in enumerate(keys)}
        print(f"  {g:16s} n={m.sum():4d} {med}")


if __name__ == "__main__":
    main()
