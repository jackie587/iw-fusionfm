"""事件数据库统计图：时空分布、波长/方向/风速分布（论文英文版）。

口径：verified 子集（results/event_database/events_verified.csv 中
keep_verified=True 的 2315 事件；场景级剔除 + 岸距 + 几何有效性规则，
见 verified_subset_report.json）。用到 high 档事件级统计的图加注 QA caveat。

输入 results/event_database/events_verified.csv → results/figures/event_db_*.png
用法（在 code/ 目录下）：
    python inference/plot_event_stats.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "results/event_database/events_verified.csv"
FIG = PROJECT_ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)

QA_CAVEAT = ("event-level QA precision ~29% (high tier);\n"
             "scene-level statistics QA-corrected")


def load():
    rows = [r for r in csv.DictReader(open(DB, encoding="utf-8"))
            if r["keep_verified"] == "True"]
    return rows


def main():
    rows = load()
    hq = [r for r in rows if r["quality"] == "high"]
    mq = [r for r in rows if r["quality"] == "medium"]
    lq = [r for r in rows if r["quality"] == "low"]
    wind = np.array([float(r["wind_ms"]) for r in rows])
    direc = np.array([float(r["direction_deg"]) for r in hq + mq])
    wl = np.array([float(r["wavelength_m"]) for r in hq])
    month = [r["time_utc"][5:7] for r in rows]

    # 1) 空间分布散点（high 实心/medium 小点/low 灰点，颜色=风速）
    fig, ax = plt.subplots(figsize=(8, 7))
    for grp, s, alpha, lab in ((lq, 8, 0.3, "low (single-crest fragments)"),
                               (mq, 16, 0.6, "medium"),
                               (hq, 32, 0.9, "high (multi-crest + λ + wind window)")):
        if not grp:
            continue
        gl = np.array([float(r["lon"]) for r in grp])
        ga = np.array([float(r["lat"]) for r in grp])
        gw = np.array([float(r["wind_ms"]) for r in grp])
        sc = ax.scatter(gl, ga, c=gw, cmap="viridis", s=s, alpha=alpha,
                        vmin=2, vmax=10, label=f"{lab} n={len(grp)}")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"ISW event spatial distribution "
                 f"(verified subset, n={len(rows)}; high {len(hq)})")
    ax.legend(fontsize=9)
    fig.colorbar(sc, label="ERA5 10 m wind speed (m/s)")
    fig.savefig(FIG / "event_db_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) 波长直方图 + 方向玫瑰图
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                 subplot_kw={})  # a2 换 polar
    a1.cla(); a1.remove()
    a1 = fig.add_subplot(1, 2, 1)
    a1.hist(wl, bins=20, color="steelblue")
    a1.set_xlabel("Wavelength (m)"); a1.set_ylabel("Event count")
    a1.set_title(f"Wavelength distribution (high tier, n={len(wl)})")
    a2.remove()
    a2 = fig.add_subplot(1, 2, 2, projection="polar")
    th = np.radians(direc) * 2  # 180° 模糊 → 倍角铺满整圆
    a2.hist(th, bins=18, color="darkorange")
    a2.set_title("Propagation direction (180° ambiguous)\n"
                 "high+medium tiers, doubled-angle display", fontsize=10)
    a1.text(0.98, 0.97, QA_CAVEAT, transform=a1.transAxes, ha="right",
            va="top", fontsize=7, color="#555555", style="italic",
            bbox=dict(fc="white", ec="#cccccc", lw=0.5, alpha=0.85, pad=2))
    fig.suptitle(f"Verified subset (n={len(rows)})", y=1.0, fontsize=10)
    fig.savefig(FIG / "event_db_wavelength_direction.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 3) 逐月事件数 + 风速分布
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 4))
    ms, cnt = np.unique(month, return_counts=True)
    b1.bar(ms, cnt, color="seagreen")
    b1.set_xlabel("Month (2023)"); b1.set_ylabel("Event count")
    b1.set_title("Monthly event count")
    b2.hist(wind, bins=15, color="slategray")
    b2.axvline(2, color="r", ls="--"); b2.axvline(10, color="r", ls="--")
    b2.set_xlabel("ERA5 10 m wind speed (m/s)"); b2.set_ylabel("Event count")
    b2.set_title("Wind speed at event locations")
    fig.suptitle(f"Verified subset (n={len(rows)})", y=1.0, fontsize=10)
    fig.savefig(FIG / "event_db_time_wind.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"→ {FIG}/event_db_map.png, event_db_wavelength_direction.png, "
          f"event_db_time_wind.png (verified subset n={len(rows)})")


if __name__ == "__main__":
    main()
