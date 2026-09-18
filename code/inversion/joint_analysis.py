# -*- coding: utf-8 -*-
"""
Joint analysis of KdV/eKdV inversion results with wavelength / cluster /
season / wind, on the QA-verified event subset.

Inputs
------
results/event_database/inversion_results.csv   (181 events, 9-scenario min/med/max)
results/event_database/events_verified.csv     (keep_verified flag, wind_ms)
results/event_database/qa_sheets/qa_judgments_merged_v2db.csv (optional QA tag)

Outputs
-------
results/event_database/inversion_joint.json
results/figures/inversion_joint_{cluster,scatter}.png

Run from code/ directory:
    python inversion/joint_analysis.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ED = os.path.join(ROOT, "results", "event_database")
FIG = os.path.join(ROOT, "results", "figures")

CLUSTER_LON_SPLIT = 109.5   # west = Hainan-west cluster, east = NE-SCS shelf


def q(x, p):
    return float(np.nanpercentile(x, p))


def stats_block(s, col):
    v = s[col].dropna().to_numpy(float)
    if len(v) == 0:
        return None
    return dict(n=int(len(v)), median=float(np.median(v)),
                p25=q(v, 25), p75=q(v, 75))


def corr_block(df, a, b):
    sub = df[[a, b]].dropna()
    if len(sub) < 5:
        return None
    r, p = spearmanr(sub[a], sub[b])
    return dict(n=int(len(sub)), spearman_r=float(r), p=float(p))


def main():
    inv = pd.read_csv(os.path.join(ED, "inversion_results.csv"))
    ver = pd.read_csv(os.path.join(ED, "events_verified.csv"),
                      usecols=["event_id", "scene", "wind_ms", "keep_verified",
                               "dist_coast_km", "n_crest"])
    df = inv.merge(ver, on=["event_id", "scene"], how="left")
    df["month"] = pd.to_datetime(df["time_utc"]).dt.month
    df["cluster"] = np.where(df.lon < CLUSTER_LON_SPLIT, "west", "east")
    df["nonlin_ratio"] = df.c_kdv_ms_med / df.c0_ms_med  # c_nl / c_0

    out = {"meta": dict(
        n_inverted=int(len(df)),
        n_verified_keep=int(df.keep_verified.fillna(False).sum()),
        note="inversion ran on the 207-event high set of the earlier event db; "
             "keep_verified joined from the rebuilt 2816-event db by "
             "(event_id, scene)",
        kdv_ok_only="eta0 statistics restricted to regime==kdv_ok; "
                    "short_wave regime amplifies eta0 systematically"),
    }

    kdv = df[df.regime == "kdv_ok"].copy()
    kdv_v = kdv[kdv.keep_verified == True]  # noqa: E712
    out["kdv_ok"] = dict(
        n=int(len(kdv)),
        n_verified=int(len(kdv_v)),
        eta0_m=stats_block(kdv, "eta0_kdv_m_med"),
        c_ms=stats_block(kdv, "c_kdv_ms_med"),
        wavelength_m=stats_block(kdv, "wavelength_m"),
        depth_m=stats_block(kdv, "depth_m"),
        nonlin_ratio=stats_block(kdv, "nonlin_ratio"),
    )
    out["kdv_ok_verified_subset"] = dict(
        eta0_m=stats_block(kdv_v, "eta0_kdv_m_med"),
        c_ms=stats_block(kdv_v, "c_kdv_ms_med"),
        wavelength_m=stats_block(kdv_v, "wavelength_m"),
        depth_m=stats_block(kdv_v, "depth_m"),
        nonlin_ratio=stats_block(kdv_v, "nonlin_ratio"),
    )

    # by cluster (verified only)
    by_cluster = {}
    for cl, sub in kdv_v.groupby("cluster"):
        by_cluster[cl] = dict(
            n=int(len(sub)),
            eta0_m=stats_block(sub, "eta0_kdv_m_med"),
            c_ms=stats_block(sub, "c_kdv_ms_med"),
            wavelength_m=stats_block(sub, "wavelength_m"),
            depth_m=stats_block(sub, "depth_m"),
            wind_ms=stats_block(sub, "wind_ms"),
        )
    out["by_cluster"] = by_cluster

    # by month (verified kdv_ok)
    by_month = {}
    for m, sub in kdv_v.groupby("month"):
        by_month[int(m)] = dict(n=int(len(sub)),
                                eta0_m=stats_block(sub, "eta0_kdv_m_med"),
                                c_ms=stats_block(sub, "c_kdv_ms_med"))
    out["by_month"] = by_month

    # correlations on verified kdv_ok
    out["correlations"] = {
        "eta0_vs_wavelength": corr_block(kdv_v, "eta0_kdv_m_med", "wavelength_m"),
        "eta0_vs_depth": corr_block(kdv_v, "eta0_kdv_m_med", "depth_m"),
        "eta0_vs_wind": corr_block(kdv_v, "eta0_kdv_m_med", "wind_ms"),
        "c_vs_depth": corr_block(kdv_v, "c_kdv_ms_med", "depth_m"),
        "c_vs_wavelength": corr_block(kdv_v, "c_kdv_ms_med", "wavelength_m"),
        "eta0_vs_nonlin_ratio": corr_block(kdv_v, "eta0_kdv_m_med", "nonlin_ratio"),
    }

    # ekdv subset where at least 3 of 9 stratification scenarios have roots
    ek = kdv_v[(kdv_v.ekdv_valid >= 3) & kdv_v.eta0_ekdv_m_med.notna()]
    out["ekdv_valid_subset"] = dict(
        n=int(len(ek)),
        eta0_ekdv_m=stats_block(ek, "eta0_ekdv_m_med"),
        eta0_kdv_m_same_events=stats_block(ek, "eta0_kdv_m_med"),
    )

    # ---- figures ----------------------------------------------------------
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (col, lab) in zip(axes, [("eta0_kdv_m_med", "eta0 (m)"),
                                     ("c_kdv_ms_med", "c (m/s)"),
                                     ("wavelength_m", "wavelength (m)")]):
        data = [kdv_v[kdv_v.cluster == c][col].dropna() for c in ("west", "east")]
        ax.boxplot(data, tick_labels=["west", "east"], showfliers=False,
                   widths=0.5)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(0)
                       .normal(0, 0.04, len(d)), d, s=6, alpha=0.5)
        ax.set_ylabel(lab)
        ax.set_title(f"kdv_ok verified, west n={len(data[0])}, "
                     f"east n={len(data[1])}", fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(FIG, "inversion_joint_cluster.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    pairs = [("wavelength_m", "eta0_kdv_m_med", "wavelength (m)", "eta0 (m)"),
             ("depth_m", "c_kdv_ms_med", "depth (m)", "c (m/s)"),
             ("wind_ms", "eta0_kdv_m_med", "ERA5 wind (m/s)", "eta0 (m)")]
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        sub = kdv_v[[x, y]].dropna()
        ax.scatter(sub[x], sub[y], s=10, alpha=0.6)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        r = corr_block(kdv_v, x, y)
        if r:
            ax.set_title(f"Spearman r={r['spearman_r']:.2f} "
                         f"(p={r['p']:.2g}, n={r['n']})", fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(FIG, "inversion_joint_scatter.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)

    with open(os.path.join(ED, "inversion_joint.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"\nfigures: {p1}\n         {p2}")


if __name__ == "__main__":
    main()
