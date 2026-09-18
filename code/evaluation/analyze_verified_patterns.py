# -*- coding: utf-8 -*-
"""verified 子集波长/方向统计稳健性复核（候选③）。

对过滤前（全部 2816 行）与过滤后（keep_verified=True）分别重算：
- 全库与 high 档的波长中位/分布（仅波长有效事件）；
- 传播方向轴向统计（倍角圆均值、集中度 R）；
- 按经度 109.5E 划分的东西两丛集的波长与方向；
另用 QA 判 TP 的事件子集（n=41）的波长中位做交叉印证。

输入  results/event_database/events_verified.csv
      results/event_database/qa_sheets/（v2db 判读，经 build_verified_subset 合并口径）
输出  results/event_database/verified_pattern_summary.json
      results/figures/verified_pattern_wavelength.png
      results/figures/verified_pattern_direction.png
用法（在 code/ 目录下）：python evaluation/analyze_verified_patterns.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "results/event_database/events_verified.csv"
QA_DIR = PROJECT_ROOT / "results/event_database/qa_sheets"
FIG = PROJECT_ROOT / "results/figures"
OUT = PROJECT_ROOT / "results/event_database/verified_pattern_summary.json"

SPLIT_LON = 109.5


def axial_mean(deg: np.ndarray) -> tuple[float, float]:
    th = np.radians(deg) * 2
    s, c = np.sin(th).mean(), np.cos(th).mean()
    return float(np.degrees(np.arctan2(s, c)) / 2 % 180), float(np.hypot(s, c))


def stats(df: pd.DataFrame) -> dict:
    wl = df["wavelength_m"].dropna().values
    dr = df["direction_deg"].dropna().values
    dmean, R = axial_mean(dr) if len(dr) else (None, None)
    return {
        "n": int(len(df)),
        "n_wavelength_valid": int(len(wl)),
        "wavelength_median_m": round(float(np.median(wl)), 1) if len(wl) else None,
        "wavelength_p25_p75_m": ([round(float(np.percentile(wl, 25)), 1),
                                  round(float(np.percentile(wl, 75)), 1)]
                                 if len(wl) else None),
        "direction_axial_mean_deg": round(dmean, 1) if dmean is not None else None,
        "direction_resultant_R": round(R, 3) if R is not None else None,
    }


def cluster_stats(df: pd.DataFrame) -> dict:
    hq = df[(df["quality"] == "high") & df["wavelength_m"].notna()]
    west = hq[hq["lon"] < SPLIT_LON]
    east = hq[hq["lon"] >= SPLIT_LON]
    return {"west_hainan": stats(west), "east_scs_slope": stats(east)}


def load_qa_tp(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier in ["high", "medium", "low"]:
        recs = json.load(open(QA_DIR / f"qa_{tier}_v2db_records.json",
                              encoding="utf-8"))
        panels = []
        for jf in sorted(QA_DIR.glob(f"qa_{tier}_v2db_*_judgment.json")):
            panels.extend(json.load(open(jf, encoding="utf-8"))["panels"])
        for rec, p in zip(recs, panels):
            if p["judgment"] == "TP":
                rows.append({"event_id": rec["event_id"], "scene": rec["scene"]})
    qa_tp = pd.DataFrame(rows)
    ev = events.copy()
    ev["key"] = ev["event_id"] + "|" + ev["scene"]
    qa_tp["key"] = qa_tp["event_id"] + "|" + qa_tp["scene"]
    return qa_tp.merge(ev, on="key", how="left")


def main():
    df = pd.read_csv(DB)
    df["wavelength_m"] = pd.to_numeric(df["wavelength_m"], errors="coerce")
    before, after = df, df[df["keep_verified"]]
    print(f"过滤前 {len(before)} / 过滤后 {len(after)}")

    summary = {
        "split_lon": SPLIT_LON,
        "before": {"n_events": int(len(before)),
                   "all_quality": stats(before),
                   "high_quality": stats(before[before["quality"] == "high"]),
                   "clusters_high": cluster_stats(before)},
        "after": {"n_events": int(len(after)),
                  "all_quality": stats(after),
                  "high_quality": stats(after[after["quality"] == "high"]),
                  "clusters_high": cluster_stats(after)},
    }

    # QA-TP 子集交叉印证
    qa_tp = load_qa_tp(df)
    wl_tp = qa_tp["wavelength_m"].dropna()
    summary["qa_tp_subset"] = {
        "n": int(len(qa_tp)),
        "n_wavelength_valid": int(len(wl_tp)),
        "wavelength_median_m": round(float(wl_tp.median()), 1),
        "wavelength_p25_p75_m": [round(float(wl_tp.quantile(.25)), 1),
                                 round(float(wl_tp.quantile(.75)), 1)],
    }
    print(f"QA-TP 子集 n={len(qa_tp)}，λ 中位 {wl_tp.median():.0f} m")

    # ---- 图 1：波长分布 before/after（high 档，分丛集）----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    bins = np.arange(0, 2100, 100)
    for ax, lon_lo, lon_hi, title in (
            (axes[0], -np.inf, SPLIT_LON, "West cluster (W of Hainan, lon<109.5E)"),
            (axes[1], SPLIT_LON, np.inf, "East cluster (NE SCS slope, lon>=109.5E)")):
        sel = ((df["quality"] == "high") & df["wavelength_m"].notna()
               & (df["lon"] >= lon_lo) & (df["lon"] < lon_hi))
        b, a = df[sel & ~df["keep_verified"]], df[sel & df["keep_verified"]]
        ax.hist([b["wavelength_m"], a["wavelength_m"]], bins=bins,
                label=[f"removed (n={len(b)})", f"verified (n={len(a)})"],
                color=["lightcoral", "steelblue"], alpha=0.85)
        for d, c, ls in ((b, "lightcoral", "--"), (a, "steelblue", "-")):
            if len(d):
                med = d["wavelength_m"].median()
                ax.axvline(med, color=c, ls=ls,
                           label=f"median {med:.0f} m")
        ax.set_xlabel("Wavelength (m)"); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Event count")
    fig.suptitle("High-quality ISW events: wavelength distribution, "
                 "verified subset vs removed")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(FIG / "verified_pattern_wavelength.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 图 2：方向玫瑰 before/after（high 档，分丛集，倍角显示）----
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"},
                             figsize=(10, 4.8))
    for ax, lon_lo, lon_hi, title in (
            (axes[0], -np.inf, SPLIT_LON, "West cluster"),
            (axes[1], SPLIT_LON, np.inf, "East cluster")):
        sel = ((df["quality"] == "high")
               & (df["lon"] >= lon_lo) & (df["lon"] < lon_hi))
        for d, name, c in ((df[sel & ~df["keep_verified"]], "removed", "lightcoral"),
                           (df[sel & df["keep_verified"]], "verified", "steelblue")):
            th = np.radians(d["direction_deg"].values) * 2
            ax.hist(th, bins=18, color=c, alpha=0.55, label=f"{name} (n={len(d)})")
        dr = df[sel & df["keep_verified"]]["direction_deg"].values
        m, R = axial_mean(dr)
        ax.set_title(f"{title}\nverified axial mean {m:.0f} deg (R={R:.2f})",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.suptitle("Propagation direction (180-deg ambiguous, doubled-angle display), "
                 "high-quality events")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIG / "verified_pattern_direction.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    json.dump(summary, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"→ {OUT}\n→ {FIG}/verified_pattern_*.png")


if __name__ == "__main__":
    main()
