"""事件数据库 high 级事件规律分析（论文英文版，verified 子集口径）。

在 plot_event_stats.py 的总览图之上，对 high 级事件（多峰+波长有效+
风窗内+面积≤2000km²）做分海区规律挖掘：
- 空间上按经度 109.5E 分为西丛集（海南岛以西）与东丛集（南海东北部陆坡）；
- 分别统计传播方向（轴向数据，倍角圆均值）、波长、波包规模、风速；
- 逐月事件数按当月有检出的场景数归一化（观测偏差的一阶修正）。

口径：verified 子集（events_verified.csv 中 keep_verified=True）。
high 档事件级 QA 精度约 29%（详见 verified_subset_report.json），各图加小注。

输入 results/event_database/events_verified.csv
输出 results/figures/event_pattern_*.png + results/event_database/pattern_summary.json
用法（在 code/ 目录下）：python inference/analyze_event_patterns.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "results/event_database/events_verified.csv"
FIG = PROJECT_ROOT / "results/figures"
OUT = PROJECT_ROOT / "results/event_database/pattern_summary.json"
FIG.mkdir(parents=True, exist_ok=True)

SPLIT_LON = 109.5  # 西=海南岛以西丛集，东=南海东北部陆坡丛集

QA_CAVEAT = ("event-level QA precision ~29% (high tier);\n"
             "scene-level statistics QA-corrected")
WEST = "West cluster (W of Hainan)"
EAST = "East cluster (NE SCS slope)"


def axial_mean(deg: np.ndarray) -> tuple[float, float]:
    """轴向（180° 模糊）方向的圆均值与集中度 R。"""
    th = np.radians(deg) * 2
    s, c = np.sin(th).mean(), np.cos(th).mean()
    R = float(np.hypot(s, c))
    mean = np.degrees(np.arctan2(s, c)) / 2 % 180
    return float(mean), R


def region_stats(rows: list[dict]) -> dict:
    wl = np.array([float(r["wavelength_m"]) for r in rows])
    dr = np.array([float(r["direction_deg"]) for r in rows])
    wd = np.array([float(r["wind_ms"]) for r in rows])
    nc = np.array([int(r["n_crest"]) for r in rows])
    dmean, R = axial_mean(dr)
    return {
        "n": len(rows),
        "wavelength_median_m": round(float(np.median(wl)), 1),
        "wavelength_p25_p75_m": [round(float(np.percentile(wl, 25)), 1),
                                 round(float(np.percentile(wl, 75)), 1)],
        "direction_axial_mean_deg": round(dmean, 1),
        "direction_resultant_R": round(R, 3),
        "wind_median_ms": round(float(np.median(wd)), 2),
        "n_crest_median": float(np.median(nc)),
        "corr_wind_wavelength": round(float(np.corrcoef(wd, wl)[0, 1]), 3),
    }


def caveat(ax, **kw):
    ax.text(0.98, 0.97, QA_CAVEAT, transform=ax.transAxes, ha="right",
            va="top", fontsize=7, color="#555555", style="italic",
            bbox=dict(fc="white", ec="#cccccc", lw=0.5, alpha=0.85, pad=2),
            **kw)


def main():
    rows = [r for r in csv.DictReader(open(DB, encoding="utf-8"))
            if r["keep_verified"] == "True"]
    hq = [r for r in rows if r["quality"] == "high" and r["wavelength_m"]]
    west = [r for r in hq if float(r["lon"]) < SPLIT_LON]
    east = [r for r in hq if float(r["lon"]) >= SPLIT_LON]
    print(f"verified 事件 {len(rows)}；high 事件 {len(hq)}："
          f"西丛集 {len(west)} / 东丛集 {len(east)}")

    summary = {
        "subset": "verified (events_verified.csv, keep_verified=True)",
        "n_events_verified": len(rows),
        "n_high": len(hq),
        "split_lon": SPLIT_LON,
        "west_hainan": region_stats(west),
        "east_scs_slope": region_stats(east),
        "qa_note": "event-level QA precision ~29% (high tier, incl. UNCERTAIN "
                   "40%); scene-level statistics QA-corrected",
    }

    # 1) 分海区方向玫瑰图
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"},
                             figsize=(10, 4.6))
    for ax, grp, name, color in (
            (axes[0], west, f"{WEST}, n={len(west)}", "steelblue"),
            (axes[1], east, f"{EAST}, n={len(east)}", "darkorange")):
        th = np.radians([float(r["direction_deg"]) for r in grp]) * 2
        ax.hist(th, bins=18, color=color, alpha=0.8)
        m, R = axial_mean(np.array([float(r["direction_deg"]) for r in grp]))
        ax.set_title(f"{name}\naxial mean {m:.0f}° (R={R:.2f})", fontsize=10)
    fig.suptitle("Propagation direction of high-tier events "
                 "(180° ambiguous, doubled-angle display)")
    fig.text(0.99, 0.01, QA_CAVEAT.replace("\n", "; "), ha="right",
             va="bottom", fontsize=7, color="#555555", style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    fig.savefig(FIG / "event_pattern_region_direction.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 2) 波长：分海区直方图 + 波长-纬度散点
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for grp, name, color in ((west, "West cluster", "steelblue"),
                             (east, "East cluster", "darkorange")):
        wl = [float(r["wavelength_m"]) for r in grp]
        a1.hist(wl, bins=15, alpha=0.6, label=f"{name} n={len(grp)}",
                color=color)
        a2.scatter([float(r["lat"]) for r in grp], wl, s=14, alpha=0.5,
                   color=color, label=name)
    a1.set_xlabel("Wavelength (m)"); a1.set_ylabel("Event count"); a1.legend()
    a1.set_title("Wavelength distribution by cluster")
    a2.set_xlabel("Latitude (°N)"); a2.set_ylabel("Wavelength (m)"); a2.legend()
    a2.set_title("Wavelength vs. latitude")
    caveat(a1)
    fig.suptitle(f"High-tier events, verified subset (n={len(hq)})",
                 y=1.0, fontsize=10)
    fig.savefig(FIG / "event_pattern_wavelength.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 3) 波长-风速关系 + 逐月事件数（按当月有检出场景数归一化）
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for grp, name, color in ((west, "West cluster", "steelblue"),
                             (east, "East cluster", "darkorange")):
        b1.scatter([float(r["wind_ms"]) for r in grp],
                   [float(r["wavelength_m"]) for r in grp],
                   s=14, alpha=0.5, color=color, label=name)
    b1.set_xlabel("ERA5 10 m wind speed (m/s)"); b1.set_ylabel("Wavelength (m)")
    b1.legend(); b1.set_title("Wavelength vs. wind speed (high tier)")
    months = sorted({r["time_utc"][5:7] for r in rows})
    cnt = [sum(1 for r in hq if r["time_utc"][5:7] == m) for m in months]
    nsc = [len({r["scene"] for r in rows if r["time_utc"][5:7] == m})
           for m in months]
    norm = [c / n for c, n in zip(cnt, nsc)]
    x = np.arange(len(months))
    b2.bar(x - 0.2, cnt, width=0.4, label="high-tier events", color="seagreen")
    b2.bar(x + 0.2, norm, width=0.4, label="÷ scenes with detections",
           color="gray")
    b2.set_xticks(x, months); b2.set_xlabel("Month (2023)"); b2.legend()
    b2.set_title("Monthly high-tier events (first-order sampling-bias fix)")
    caveat(b1)
    fig.suptitle(f"High-tier events, verified subset (n={len(hq)})",
                 y=1.0, fontsize=10)
    fig.savefig(FIG / "event_pattern_wind_monthly.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 4) 波包规模：n_crest 分布 + 波峰线总长-波长
    fig, (c1, c2) = plt.subplots(1, 2, figsize=(12, 4.2))
    nc = [int(r["n_crest"]) for r in hq]
    c1.hist(np.clip(nc, 0, 40), bins=20, color="slategray")
    c1.set_xlabel("Crest lines per event (display clipped at 40)")
    c1.set_ylabel("Event count")
    c1.set_title(f"Packet size (median {np.median(nc):.0f} crests)")
    cl = [float(r["crest_length_km"]) for r in hq]
    wl = [float(r["wavelength_m"]) for r in hq]
    c2.scatter(wl, cl, s=14, alpha=0.5, color="teal")
    c2.set_xlabel("Wavelength (m)"); c2.set_ylabel("Total crest length (km)")
    c2.set_title("Wavelength vs. total crest length")
    caveat(c1)
    fig.suptitle(f"High-tier events, verified subset (n={len(hq)})",
                 y=1.0, fontsize=10)
    fig.savefig(FIG / "event_pattern_packetsize.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    json.dump(summary, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"→ {FIG}/event_pattern_*.png, {OUT}")


if __name__ == "__main__":
    main()
