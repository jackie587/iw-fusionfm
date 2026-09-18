"""任务 3b：2023-09-30 近同时配对（Δt=1.06h）逐事件个例检查。

这是全档案中唯一"小 Δt + 事件落在刈幅有效像元 5km 内"的配对（仅 3~5 个事件）。
对每个匹配事件开 20×20 km 窗口：Expert 2km 包络残差 + SAR 事件原始位置、
±dir 平移（3.5 km）后的位置，目视检验是否存在与波包对应的同号起伏。

用法（code/ 目录下）：python evaluation/swot_isw_step3b_case_study_0930.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

from data_preprocessing.swot_preprocess.quality_control import load_and_qc
from utils.geo import propagation_shift

SWOT_DIR = PROJECT_ROOT / "data/raw/swot"
EVENTS_CSV = PROJECT_ROOT / "results/event_database/events.csv"
OUT = PROJECT_ROOT / "results/swot_isw_detection"

DAY, GRANULE = "2023-09-30", "004_243_20230930T111959"
DT_H, C_MED = 1.06, 0.92
WIN_KM = 10.0


def nan_gauss(arr, sigma):
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    w = ndimage.gaussian_filter(valid.astype(np.float64), sigma)
    s = ndimage.gaussian_filter(filled, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 1e-3, s / np.maximum(w, 1e-3), np.nan)


def main() -> None:
    fp = sorted(SWOT_DIR.glob(f"SWOT_L2_LR_SSH_Expert_{GRANULE}*.nc"))[0]
    qc = load_and_qc(str(fp))
    lon2d, lat2d = qc["lon"].astype(np.float64), qc["lat"].astype(np.float64)
    ssha, mask = qc["ssh"], qc["mask"] > 0
    ssha = np.where(mask, ssha, np.nan)
    resid = ssha - nan_gauss(ssha, 2.5)      # 去 >5km 背景
    resid[~mask] = np.nan

    flat = np.isfinite(resid).ravel()
    tree = cKDTree(np.stack([lon2d.ravel()[flat], lat2d.ravel()[flat]], 1))

    ev = []
    with open(EVENTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["time_utc"].startswith(DAY):
                ev.append(r)
    # 匹配事件：原始或平移后位置距有效像元 ≤5km
    matched = []
    for r in ev:
        lo, la = float(r["lon"]), float(r["lat"])
        dg = float(r["direction_deg"])
        for sign in (+1, -1):
            dl, da = propagation_shift(la, dg + (0 if sign > 0 else 180),
                                       C_MED, DT_H * 3600)
            d, _ = tree.query([[lo + dl, la + da]])
            if d[0] * 111 <= 5.0:
                matched.append((r, sign, lo + dl, la + da))
                break
    print(f"匹配事件 {len(matched)} / {len(ev)}")

    n = len(matched)
    if not n:
        return
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.5), squeeze=False)
    summary = []
    for ax, (r, sign, slo, sla) in zip(axes[0], matched):
        lo, la = float(r["lon"]), float(r["lat"])
        # 窗口裁剪（经纬度近似）
        m = WIN_KM / 111.0
        sub = ((lon2d >= slo - m) & (lon2d <= slo + m)
               & (lat2d >= sla - m) & (lat2d <= sla + m))
        rows = np.nonzero(sub.any(axis=1))[0]
        cols = np.nonzero(sub.any(axis=0))[0]
        sl = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        rr = resid[sl]
        okp = np.isfinite(rr)
        vlim = np.nanpercentile(np.abs(rr), 98) if okp.any() else 0.05
        sc = ax.scatter(lon2d[sl][okp], lat2d[sl][okp], c=rr[okp], s=25,
                        cmap="RdBu_r", vmin=-vlim, vmax=vlim, linewidths=0)
        fig.colorbar(sc, ax=ax, label="SSHA 残差 (m)", shrink=0.75)
        # 原始位置、平移位置
        ax.plot(lo, la, "kx", ms=10, mew=2, label="SAR 原始位置")
        ax.plot(slo, sla, "o", mfc="none", mec="r", ms=14, mew=2,
                label=f"{'+' if sign > 0 else '-'}dir 平移 {C_MED * DT_H * 3.6:.1f}km")
        dg = float(r["direction_deg"])
        rad = np.deg2rad(dg)
        ax.annotate("", xy=(lo + 0.03 * np.sin(rad), la + 0.03 * np.cos(rad)),
                    xytext=(lo, la),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
        # 平移点邻域统计（5km 半径内残差均值/极值）
        d2, idx2 = tree.query([[slo, sla]], k=200)
        near = d2[0] * 111 <= 5.0
        vals = resid.ravel()[flat][idx2[0][near]]
        rec = {"event_id": r["event_id"], "sign": sign,
               "lon": lo, "lat": la,
               "direction_deg": dg,
               "wavelength_m": r["wavelength_m"],
               "wind_ms": r["wind_ms"], "quality": r["quality"],
               "n_nb": int(near.sum()),
               "nb_mean_m": round(float(np.nanmean(vals)), 4),
               "nb_max_m": round(float(np.nanmax(vals)), 4),
               "nb_min_m": round(float(np.nanmin(vals)), 4)}
        summary.append(rec)
        ax.set_title(f"{r['event_id'][-4:]} λ={r['wavelength_m']}m "
                     f"dir={dg:.0f}°\n邻域均值 {rec['nb_mean_m'] * 100:+.1f}cm "
                     f"极值[{rec['nb_min_m'] * 100:.0f},"
                     f"{rec['nb_max_m'] * 100:.0f}]cm", fontsize=9)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlabel("经度"); ax.set_ylabel("纬度")
    fig.suptitle(f"{DAY} 近同时配对（Δt={DT_H}h）逐事件检查："
                 f"Expert 2km 残差 ±10km 窗口")
    fig.tight_layout()
    fig.savefig(OUT / "fig_step3b_0930_cases.png", dpi=150)
    plt.close(fig)
    with open(OUT / "step3b_0930_cases.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"→ {OUT}/fig_step3b_0930_cases.png, step3b_0930_cases.json")


if __name__ == "__main__":
    main()
