"""L2 配对块盘点：manifest 合并 + 产量统计 + QC 三联图抽查。

输入 data/datasets/L2_s1_swot_matched/tiles/manifest_*.json + *.npz
输出 tiles/manifest_all.json、results/figures/l2_tile_qc_<日>_<n>.png
用法（在 code/ 目录下）：python tests/merge_l2_manifest.py [--qc-per-day 2]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TILES = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"
FIG = PROJECT_ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_manifests():
    """合并非空 manifest，按场景日期排序；返回 (records, per_scene)。"""
    records, per_scene = [], {}
    for f in sorted(TILES.glob("manifest_*.json")):
        if f.stem == "manifest_all":
            continue
        recs = json.load(open(f, encoding="utf-8"))
        if not recs:
            continue
        for r in recs:
            r["day"] = r["scene"][17:25]  # ..._1SDV_YYYYMMDDTHHMMSS_...
        records.extend(recs)
        per_scene[f.stem.replace("manifest_", "")] = len(recs)
    records.sort(key=lambda r: (r["day"], r["scene"], r["tile"]))
    return records, per_scene


def inventory(records, per_scene):
    days = {}
    for r in records:
        d = days.setdefault(r["day"], {"n": 0, "labeled": 0, "cov": []})
        d["n"] += 1
        d["labeled"] += int(r["label_frac"] > 0)
        d["cov"].append(r["swot_coverage"])
    print(f"{'日期':<10}{'块数':>6}{'含弱标签':>8}{'覆盖率中位':>10}")
    for day in sorted(days):
        d = days[day]
        print(f"{day:<10}{d['n']:>6}{d['labeled']:>8}"
              f"{np.median(d['cov']):>10.2f}")
    n = len(records)
    lab = sum(1 for r in records if r["label_frac"] > 0)
    print(f"合计 {n} 块（含弱标签 {lab}），场景 {len(per_scene)} 个")
    return days


def qc_figures(records, per_day=2):
    """每日抽样：1 个含标签块 + 1 个高覆盖块，画 SAR/SSHA/弱标签三联图。"""
    by_day: dict[str, list[dict]] = {}
    for r in records:
        by_day.setdefault(r["day"], []).append(r)
    for day, recs in sorted(by_day.items()):
        recs.sort(key=lambda r: -r["label_frac"])
        picks = [recs[0]]  # 标签最多的
        rest = [r for r in recs if r is not recs[0] and r["swot_coverage"] > 0.9]
        if rest:
            picks.append(rest[len(rest) // 2])
        for i, r in enumerate(picks[:per_day]):
            d = np.load(TILES / r["file"])
            sar = d["sar"][0].astype(np.float32)  # VV, dB 归一化
            ssha, grad, msk = d["swot"]
            label = d["label"]
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
            ax = axes[0]
            ax.imshow(sar, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"SAR VV\n{r['file'][:28]}", fontsize=9)
            ax = axes[1]
            sm = np.ma.masked_where(msk < 0.5, ssha)
            im = ax.imshow(sm, cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                           extent=[0, 512, 512, 0])
            ax.set_title(f"SWOT SSHA 250m（cov={r['swot_coverage']:.2f}）",
                         fontsize=9)
            fig.colorbar(im, ax=ax, label="m", shrink=0.8)
            ax = axes[2]
            ax.imshow(sar, cmap="gray", vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(label < 1, label), cmap="autumn",
                      alpha=0.55)
            ax.set_title(f"v7 弱标签（frac={r['label_frac']:.3f}）",
                         fontsize=9)
            for a in axes:
                a.axis("off")
            out = FIG / f"l2_tile_qc_{day}_{i}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-per-day", type=int, default=2)
    args = ap.parse_args()
    records, per_scene = load_manifests()
    json.dump(records, open(TILES / "manifest_all.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"manifest_all.json ← {len(records)} 条")
    inventory(records, per_scene)
    qc_figures(records, args.qc_per_day)


if __name__ == "__main__":
    main()
