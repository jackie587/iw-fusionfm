"""反演结果统计图：振幅/相速度空间分布、与波长/水深关系。

输入 results/event_database/inversion_results.csv
输出 results/figures/inversion_*.png
用法（在 code/ 目录下）：python inversion/plot_inversion.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "results/event_database/inversion_results.csv"
FIG = PROJECT_ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

REGIME_STYLE = {"kdv_ok": ("o", "steelblue", "kdv_ok（KdV 适用）"),
                "djl_needed": ("D", "crimson", "djl_needed（η0>h1，仅参考）"),
                "short_wave": ("^", "darkorange", "short_wave（λ<3H，仅参考）")}


def load():
    rows = list(csv.DictReader(open(DB, encoding="utf-8")))
    for r in rows:
        for k in ("lon", "lat", "wavelength_m", "depth_m", "eta0_kdv_m_med",
                  "eta0_kdv_m_lo", "eta0_kdv_m_hi", "c_kdv_ms_med", "c0_ms_med"):
            r[k] = float(r[k])
    return rows


def main():
    rows = load()
    print(f"反演事件 {len(rows)}："
          + ", ".join(f"{rg}={sum(r['regime'] == rg for r in rows)}"
                      for rg in REGIME_STYLE))

    # 1) 振幅空间分布
    fig, ax = plt.subplots(figsize=(8, 7))
    for rg, (mk, color, lab) in REGIME_STYLE.items():
        g = [r for r in rows if r["regime"] == rg]
        if not g:
            continue
        sc = ax.scatter([r["lon"] for r in g], [r["lat"] for r in g],
                        c=[r["eta0_kdv_m_med"] for r in g], cmap="turbo",
                        norm=matplotlib.colors.LogNorm(vmin=1, vmax=200),
                        marker=mk, s=30, alpha=0.85, label=f"{lab} n={len(g)}",
                        edgecolors="k", linewidths=0.3)
    ax.set_xlabel("经度"); ax.set_ylabel("纬度")
    ax.set_title(f"反演振幅空间分布（KdV 中位数，n={len(rows)}）")
    ax.legend(fontsize=9)
    fig.colorbar(sc, label="η0 (m，对数色标)")
    fig.savefig(FIG / "inversion_amp_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) η0-λ（含敏感性区间）+ c-λ
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for rg, (mk, color, lab) in REGIME_STYLE.items():
        g = [r for r in rows if r["regime"] == rg]
        if not g:
            continue
        x = np.array([r["wavelength_m"] for r in g])
        y = np.array([r["eta0_kdv_m_med"] for r in g])
        lo = np.array([r["eta0_kdv_m_lo"] for r in g])
        hi = np.array([r["eta0_kdv_m_hi"] for r in g])
        a1.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk, ms=4, color=color,
                    alpha=0.55, elinewidth=0.6, capsize=0, label=lab)
    lam = np.linspace(100, 1500, 50)
    y0 = 10.0 * (600.0 / lam) ** 2  # η0∝λ⁻² 参考线（过 600m,10m）
    a1.plot(lam, y0, "k--", lw=1, label="η0 ∝ 1/λ^2 参考")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("波长 λ (m)"); a1.set_ylabel("η0 (m)")
    a1.set_title("振幅-波长（误差棒=分层敏感性区间）"); a1.legend(fontsize=8)
    for rg, (mk, color, lab) in REGIME_STYLE.items():
        g = [r for r in rows if r["regime"] == rg]
        if not g:
            continue
        a2.scatter([r["wavelength_m"] for r in g], [r["c_kdv_ms_med"] for r in g],
                   c=[r["depth_m"] for r in g], cmap="viridis", marker=mk,
                   s=18, alpha=0.7, norm=matplotlib.colors.LogNorm())
    a2.set_xscale("log")
    a2.set_xlabel("波长 λ (m)"); a2.set_ylabel("相速度 c (m/s)")
    a2.set_title("相速度-波长（颜色=水深）")
    fig.colorbar(a2.collections[-1], ax=a2, label="水深 (m，对数色标)")
    fig.savefig(FIG / "inversion_amp_wavelength.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 3) η0-水深 + 振幅/相速度直方图
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for rg, (mk, color, lab) in REGIME_STYLE.items():
        g = [r for r in rows if r["regime"] == rg]
        if not g:
            continue
        b1.scatter([r["depth_m"] for r in g], [r["eta0_kdv_m_med"] for r in g],
                   marker=mk, s=18, alpha=0.65, color=color, label=lab)
    b1.set_yscale("log")
    b1.set_ylim(0.3, 2000)
    b1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    b1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b1.set_xlabel("水深 (m)"); b1.set_ylabel("η0 (m)")
    b1.set_title("振幅-水深"); b1.legend(fontsize=8)
    ok = np.array([r["eta0_kdv_m_med"] for r in rows if r["regime"] == "kdv_ok"])
    cc = np.array([r["c_kdv_ms_med"] for r in rows if r["regime"] == "kdv_ok"])
    b2.hist(ok, bins=np.geomspace(0.5, 100, 21), color="steelblue",
            alpha=0.75, label=f"η0（中位 {np.median(ok):.1f} m）")
    b2.set_xscale("log")
    ax2 = b2.twiny()
    ax2.hist(cc, bins=15, color="seagreen", alpha=0.5,
             label=f"c（中位 {np.median(cc):.2f} m/s）")
    b2.set_xlabel("η0 (m，对数轴)"); ax2.set_xlabel("c (m/s)")
    b2.set_ylabel("事件数")
    b2.set_title(f"振幅/相速度分布（kdv_ok，n={len(ok)}）")
    h1_, l1_ = b2.get_legend_handles_labels()
    h2_, l2_ = ax2.get_legend_handles_labels()
    b2.legend(h1_ + h2_, l1_ + l2_, fontsize=9, loc="upper right")
    fig.savefig(FIG / "inversion_amp_depth_hist.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"→ {FIG}/inversion_amp_map.png, inversion_amp_wavelength.png, "
          f"inversion_amp_depth_hist.png")


if __name__ == "__main__":
    main()
