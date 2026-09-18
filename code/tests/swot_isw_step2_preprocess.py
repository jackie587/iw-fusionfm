"""任务 2：SWOT Unsmoothed 250m 产品在 AOI 内的预处理与噪声底量化。

流程（对 5 个 granule × left/right 半刈幅，裁 AOI 后有覆盖的才处理）：
1. 复用 quality_control.load_unsmoothed_half：ssha = ssh_karin_2 − MSS，qual==0 掩膜；
2. NaN 感知大尺度高斯（σ=40 px ≈ 10 km）相减 → 中小尺度 SSHA 残差场；
3. 噪声底量化：
   - 像素级小尺度 RMS（σ=1.5 px 差分，MAD 稳健）；
   - 平均尺度曲线：boxcar b∈{1,2,4,8,16,32} px 后的残差 RMS（噪声随 √N 平均下降）；
   - 高覆盖 256×256 块（64 km）加窗周期图 → 波长带 [0.4-0.7]/[0.7-1.2]/
     [1.2-2.5]/[2.5-5] km 的带内 RMS（ISW 信号带 vs 噪声底的关键数）。
4. 输出 results/swot_isw_detection/ 下：
   - grids/{tag}.npz（lon/lat/ssha/resid/mask，供检测器复用）；
   - fig_step2_{tag}_resid.png（残差图 + 全部 SAR 事件叠加作气候参考）；
   - fig_step2_noise_floor.png（各刈幅 RMS-平均尺度曲线）；
   - step2_noise_report.json。

用法（code/ 目录下）：python tests/swot_isw_step2_preprocess.py
"""
from __future__ import annotations

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

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

from data_preprocessing.swot_preprocess.quality_control import load_unsmoothed_half

UNSM_DIR = PROJECT_ROOT / "data/raw/swot_unsmoothed"
EVENTS_CSV = PROJECT_ROOT / "results/event_database/events.csv"
OUT = PROJECT_ROOT / "results/swot_isw_detection"
GRID_DIR = OUT / "grids"

AOI = (109.0, 18.0, 117.0, 23.0)
RES_M = 250.0
SIGMA_LARGE_PX = 40          # ≈10 km 大尺度背景
MIN_PIXELS = 20000           # AOI 内有效像素下限
BOXES_PX = [1, 2, 4, 8, 16, 32]
BLOCK = 128                  # 周期图块边长（px，跨轨仅 240 px 故取 128=32km）
BANDS_KM = [(0.4, 0.7), (0.7, 1.2), (1.2, 2.5), (2.5, 5.0)]


def nan_gauss(arr: np.ndarray, sigma: float) -> np.ndarray:
    """NaN 感知高斯平滑（无效处输出 NaN）。"""
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    w = ndimage.gaussian_filter(valid.astype(np.float64), sigma)
    s = ndimage.gaussian_filter(filled, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(w > 1e-3, s / np.maximum(w, 1e-3), np.nan)
    return out


def robust_rms(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size < 100:
        return float("nan")
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def boxcar_rms_curve(resid: np.ndarray) -> dict:
    """boxcar 平均 b×b 后的残差稳健 RMS（要求盒内覆盖率≥50%）。"""
    out = {}
    valid = np.isfinite(resid)
    filled = np.where(valid, resid, 0.0)
    for b in BOXES_PX:
        w = ndimage.uniform_filter(valid.astype(np.float64), size=b)
        s = ndimage.uniform_filter(filled, size=b)
        ok = w >= 0.5
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(ok, s / np.maximum(w, 1e-9), np.nan)
        out[f"{b * RES_M / 1000:.2f}km"] = robust_rms(m)
    return out


def band_rms_psd(resid: np.ndarray) -> dict:
    """高覆盖 BLOCK×BLOCK 块的加窗周期图 → 各波长带带内 RMS（m）。"""
    ny, nx = resid.shape
    win = np.outer(np.hanning(BLOCK), np.hanning(BLOCK))
    win2 = (win ** 2).mean()
    psds = []
    for r0 in range(0, ny - BLOCK + 1, BLOCK // 2):
        for c0 in range(0, nx - BLOCK + 1, BLOCK // 2):
            blk = resid[r0:r0 + BLOCK, c0:c0 + BLOCK]
            if np.isfinite(blk).mean() < 0.9:
                continue
            blk = np.where(np.isfinite(blk), blk, np.nanmean(blk[np.isfinite(blk)]))
            blk = blk - blk.mean()
            p = np.abs(np.fft.rfft2(blk * win)) ** 2 / (win2 * BLOCK ** 4)
            psds.append(p)
    bands = {f"{a}-{b}km": float("nan") for a, b in BANDS_KM}
    if not psds:
        return {"n_blocks": 0, "bands": bands}
    psd = np.median(np.stack(psds), axis=0)  # m^2 / (cyc/px)^2
    fy = np.fft.fftfreq(BLOCK)[:, None]
    fx = np.fft.rfftfreq(BLOCK)[None, :]
    f = np.hypot(fy, fx)                     # cyc/px
    for (a, b) in BANDS_KM:
        # cyc/px = RES_M / λ；带 [a,b] km → f ∈ [RES_M/(b·10³), RES_M/(a·10³)]
        f_lo, f_hi = RES_M / (b * 1000), RES_M / (a * 1000)
        sel = (f >= f_lo) & (f < f_hi)
        # 二 sided→总方差：rfft 半平面，×2 近似（除直流/Nyquist 外）
        var = 2.0 * psd[sel].sum()
        bands[f"{a}-{b}km"] = float(np.sqrt(max(var, 0.0)))
    return {"n_blocks": len(psds), "bands": bands}


def load_events() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import csv
    lons, lats, days = [], [], []
    with open(EVENTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lons.append(float(r["lon"]))
            lats.append(float(r["lat"]))
            days.append(r["time_utc"][:10])
    return np.array(lons), np.array(lats), np.array(days)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    ev_lon, ev_lat, ev_day = load_events()
    report: dict = {"aoi": AOI, "res_m": RES_M,
                    "sigma_large_px": SIGMA_LARGE_PX, "granules": {}}
    curves: dict[str, dict] = {}

    # 可指定单个 granule 路径（默认处理 swot_unsmoothed 全部）；
    # --bbox lon1,lat1,lon2,lat2 覆盖默认 AOI（如 07-06 事件群南到 16.8N）
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bbox = AOI
    for a in sys.argv[1:]:
        if a.startswith("--bbox="):
            bbox = tuple(float(x) for x in a.split("=")[1].split(","))
    files = [Path(a) for a in args] or sorted(UNSM_DIR.glob("*.nc"))
    # 与既有报告合并（单文件运行时不清空历史 granule 记录）
    old_report_path = OUT / "step2_noise_report.json"
    if files and old_report_path.exists():
        try:
            old = json.load(open(old_report_path, encoding="utf-8"))
            report["granules"].update(old.get("granules", {}))
            for t, g in report["granules"].items():
                curves[t] = g.get("boxcar_rms_m", {})
        except Exception:
            pass
    for nc_path in files:
        nc_path = Path(nc_path)
        if not nc_path.is_absolute():
            nc_path = PROJECT_ROOT / nc_path
        day = nc_path.name.split("_")[7][:8]
        for side in ("left", "right"):
            tag = f"{day}_{side}"
            d = load_unsmoothed_half(str(nc_path), side, bbox=bbox,
                                     margin=0.15)
            if d is None or int(d["mask"].sum()) < MIN_PIXELS:
                print(f"[skip] {tag}: 无覆盖或有效像素不足")
                continue
            lon, lat, ssha, mask = d["lon"], d["lat"], d["ssh"], d["mask"]
            ssha = np.where(mask > 0, ssha, np.nan)
            # 去大尺度背景 → 中小尺度残差
            low = nan_gauss(ssha, SIGMA_LARGE_PX)
            resid = ssha - low
            resid[~np.isfinite(ssha)] = np.nan
            # 像素级小尺度噪声（σ=1.5px 差分的稳健 RMS / √2 近似白噪水平）
            smooth = nan_gauss(ssha, 1.5)
            hf = ssha - smooth
            pix_noise = robust_rms(hf) / np.sqrt(2)

            curve = boxcar_rms_curve(resid)
            psd = band_rms_psd(resid)
            curves[tag] = curve
            report["granules"][tag] = {
                "file": nc_path.name, "side": side,
                "shape_px": list(ssha.shape),
                "n_valid": int(mask.sum()),
                "lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
                "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
                "ssha_raw_rms_m": robust_rms(ssha),
                "resid_rms_m": robust_rms(resid),
                "pixel_noise_m": pix_noise,
                "boxcar_rms_m": curve,
                "band_rms_m": psd["bands"],
                "n_psd_blocks": psd["n_blocks"],
            }
            np.savez_compressed(
                GRID_DIR / f"{tag}.npz", lon=lon.astype(np.float32),
                lat=lat.astype(np.float32), ssha=ssha.astype(np.float32),
                resid=resid.astype(np.float32), mask=mask)
            print(f"[ok] {tag}: 有效 {mask.sum():,} px, 残差 RMS "
                  f"{report['granules'][tag]['resid_rms_m'] * 100:.2f} cm, "
                  f"像素噪声 {pix_noise * 100:.2f} cm, "
                  f"0.7-1.2km 带 RMS "
                  f"{report['granules'][tag]['band_rms_m']['0.7-1.2km'] * 100:.2f} cm")

            # 残差图 + SAR 事件叠加（气候参考，非同日）；lon/lat 含 NaN 用散点
            step = 4
            fig, ax = plt.subplots(figsize=(9, 7))
            vlim = np.nanpercentile(np.abs(resid), 98)
            ok = np.isfinite(resid[::step, ::step])
            sc = ax.scatter(lon[::step, ::step][ok], lat[::step, ::step][ok],
                            c=resid[::step, ::step][ok], s=3, cmap="RdBu_r",
                            vmin=-vlim, vmax=vlim, linewidths=0)
            fig.colorbar(sc, ax=ax, label="SSHA 残差 (m)")
            ina = ((ev_lon >= lon.min()) & (ev_lon <= lon.max())
                   & (ev_lat >= lat.min()) & (ev_lat <= lat.max()))
            ax.scatter(ev_lon[ina], ev_lat[ina], s=8, c="k", alpha=0.5,
                       label="SAR 事件（全部日期，气候参考）")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel("经度"); ax.set_ylabel("纬度")
            ax.set_title(f"Unsmoothed 250m SSHA 残差 {tag}（σ>10km 已去除）"
                         f"\n注：该日刈幅内无同日 SAR 真值，黑点为其他日期事件")
            fig.tight_layout()
            fig.savefig(OUT / f"fig_step2_{tag}_resid.png", dpi=150)
            plt.close(fig)

    # 噪声-平均尺度曲线汇总图
    if curves:
        fig, ax = plt.subplots(figsize=(7, 5))
        for tag, curve in curves.items():
            xs = [float(k.replace("km", "")) for k in curve]
            ys = [curve[k] * 100 for k in curve]
            ax.plot(xs, ys, "o-", label=tag)
        for a_sig, ls in ((0.02, "--"), (0.05, ":")):
            ax.axhline(a_sig * 100, color="r", ls=ls, alpha=0.6)
            ax.text(xs[0], a_sig * 100 * 1.05, f"预期信号 {a_sig * 100:.0f} cm",
                    color="r", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        ax.set_xlabel("相干平均尺度（km）"); ax.set_ylabel("残差 RMS（cm）")
        ax.set_title("Unsmoothed 250m：噪声底 vs 平均尺度")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig_step2_noise_floor.png", dpi=150)
        plt.close(fig)

    with open(OUT / "step2_noise_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"→ {OUT}/step2_noise_report.json, fig_step2_*.png")


if __name__ == "__main__":
    main()
