"""任务 4：SWOT 250m 残差场上的原型内波检测器（方向相干带通滤波组）。

思路（对步骤 2 的 resid 网格，原生猪厩 250m 网格处理）：
- 12 个方向（0~165°，步长 15°）的楔形带通（波长 0.4~2.5 km）FFT 滤波；
- 每个方向的能量图 E_θ 沿波峰走向各向异性平滑（沿峰 2 km / 垂直峰 0.5 km），
  突出"平行条纹组"的空间相干性；
- 检测统计量：方向对比 C = max_θ E / median_θ E 与强度 A = sqrt(max_θ E)；
- 零假设：同幅谱相位扰乱场（保功率谱、毁相干性）上重跑，取全局 99.9%
  分位作阈值 → 控制虚警；
- 局部极大 + 连通域组团 → 事件清单（lon/lat/方向/强度/面积）。

无同日真值，做气候学合理性检查：检出分布 vs 全部 SAR 事件热区
（陆坡折带 108.9~114.6°E 反复出现内波）。

用法（code/ 目录下）：python evaluation/swot_isw_step4_detector.py [tag ...]
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

GRID_DIR = PROJECT_ROOT / "results/swot_isw_detection/grids"
EVENTS_CSV = PROJECT_ROOT / "results/event_database/events.csv"
OUT = PROJECT_ROOT / "results/swot_isw_detection"

RES_M = 250.0
LAM_LO, LAM_HI = 400.0, 2500.0          # 带通波长范围（m）
N_THETA = 12                             # 方向数（0~165°）
HALF_ANGLE = np.deg2rad(15.0)            # 楔形半角
SIGMA_ALONG_PX = 8.0                     # 沿峰平滑 2 km
SIGMA_ACROSS_PX = 2.0                    # 垂直峰平滑 0.5 km
NULL_Q = 99.9                            # 相位扰乱零分布分位（%）
MIN_AREA_PX = 30                         # 最小连通域（px）


def oriented_wedge_filter(shape: tuple[int, int], theta: float) -> np.ndarray:
    """频域楔形带通：径向波长带 × 方向 θ±HALF_ANGLE（条纹走向=θ，
    即波数方向垂直于 θ）。"""
    ny, nx = shape
    fy = np.fft.fftfreq(ny, d=RES_M)[:, None]
    fx = np.fft.fftfreq(nx, d=RES_M)[None, :]
    f = np.hypot(fy, fx)
    band = (f >= 1.0 / LAM_HI) & (f <= 1.0 / LAM_LO)
    ang = np.arctan2(fy, fx)             # 波数角
    kwave = theta + np.pi / 2            # 条纹走向 θ → 波数 θ+90°
    d = np.angle(np.exp(1j * (ang - kwave)))
    wedge = np.abs(d) <= HALF_ANGLE
    return (band & wedge).astype(np.float64)


def anisotropic_smooth(e: np.ndarray, theta: float) -> np.ndarray:
    """沿 θ 走向的各向异性平滑：横向高斯(σ=0.5km) + 沿峰向线平均(±2km)。"""
    e = ndimage.gaussian_filter(e, SIGMA_ACROSS_PX)
    kern_len = int(2 * SIGMA_ALONG_PX) + 1
    return _line_smooth(e, np.cos(theta), np.sin(theta), kern_len)


def _line_smooth(e: np.ndarray, dx: float, dy: float, kern_len: int) -> np.ndarray:
    """沿线方向 (dx,dy) 的滑动均值：对若干亚像素偏移采样求平均。"""
    offsets = np.linspace(-SIGMA_ALONG_PX, SIGMA_ALONG_PX, kern_len)
    acc = np.zeros_like(e)
    for s in offsets:
        acc += ndimage.shift(e, (-s * dy, -s * dx), order=1, mode="nearest",
                             prefilter=False)
    return acc / kern_len


def coherence_maps(resid: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray]:
    """返回 (方向对比 C, 强度 A, 主导方向 θ*)，NaN 已填 0 前需外部掩膜。"""
    ny, nx = resid.shape
    F = np.fft.fft2(resid)
    es = []
    for k in range(N_THETA):
        theta = k * np.pi / N_THETA
        w = oriented_wedge_filter((ny, nx), theta)
        filt = np.fft.ifft2(F * w).real
        e = anisotropic_smooth(filt ** 2, theta)
        es.append(e)
    E = np.stack(es)                     # (n_theta, ny, nx)
    emax = E.max(axis=0)
    emed = np.median(E, axis=0)
    C = emax / np.maximum(emed, 1e-12)
    A = np.sqrt(emax)
    theta_star = np.argmax(E, axis=0) * (180.0 / N_THETA)
    return C, A, theta_star


def phase_scramble(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    F = np.fft.fft2(x)
    ph = np.angle(F)
    rnd = rng.uniform(0, 2 * np.pi, x.shape)
    # 保持共轭对称：直接随机化后取实部会略降功率，作为零假设足够
    rnd = 0.5 * (rnd - rnd[::-1, ::-1]) + 0.5 * (ph - ph[::-1, ::-1]) * 0
    return np.fft.ifft2(np.abs(F) * np.exp(1j * (ph * 0 + rnd))).real


def detect(tag: str, rng: np.random.Generator) -> dict:
    z = np.load(GRID_DIR / f"{tag}.npz")
    lon, lat = z["lon"].astype(np.float64), z["lat"].astype(np.float64)
    resid = z["resid"].astype(np.float64)
    mask = np.isfinite(resid)
    # 刈幅边缘在 FFT/方向滤波下产生虚假方向相干性（图上呈亮边），
    # 内缩 2.5km 且要求平滑覆盖率>0.8，双重抑制边缘伪检出
    mask = ndimage.binary_erosion(mask, iterations=10, border_value=0)
    fill = np.where(mask, resid, 0.0)

    C, A, theta_star = coherence_maps(fill)
    # 局部有效覆盖率（各向同性高斯近似）
    cov = ndimage.gaussian_filter(mask.astype(np.float64), SIGMA_ALONG_PX)

    S = C * A                              # 联合统计量
    S = np.where(cov > 0.8, S, np.nan)

    # 零假设：相位扰乱
    scr = phase_scramble(fill, rng)
    C0, A0, _ = coherence_maps(scr)
    S0 = C0 * A0
    S0 = np.where(cov > 0.8, S0, np.nan)
    thr = np.nanpercentile(S0, NULL_Q)

    def count_events(Sx: np.ndarray) -> tuple[int, int]:
        detx = (Sx > thr) & np.isfinite(Sx)
        labx, nx_ = ndimage.label(detx)
        sizes = np.bincount(labx.ravel())[1:]
        return int((sizes >= MIN_AREA_PX).sum()), int(nx_)

    det = (S > thr) & np.isfinite(S)
    lab, n = ndimage.label(det)
    events = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < MIN_AREA_PX:
            continue
        k = np.argmax(S[ys, xs])
        y, x = ys[k], xs[k]
        if not (np.isfinite(lon[y, x]) and np.isfinite(lat[y, x])):
            continue
        events.append({
            "lon": round(float(lon[y, x]), 4),
            "lat": round(float(lat[y, x]), 4),
            "direction_deg": round(float(theta_star[y, x]), 1),
            "strength": round(float(S[y, x]), 4),
            "amplitude_m": round(float(A[y, x]), 5),
            "contrast": round(float(C[y, x]), 2),
            "n_px": int(ys.size),
            "area_km2": round(ys.size * RES_M ** 2 / 1e6, 2),
        })
    n_null, _ = count_events(S0)   # 同阈值下零假设场的检出数（直接 FPR 估计）
    return {"tag": tag, "n_events": len(events), "threshold": float(thr),
            "S_null_p999": float(thr), "n_null_events": n_null,
            "events": events, "S": S, "lon": lon, "lat": lat}


def main() -> None:
    tags = sys.argv[1:] or ["20230801_left", "20230801_right",
                            "20230820_right", "20230823_left",
                            "20230823_right", "20230827_left",
                            "20230827_right", "20230830_left",
                            "20230830_right"]
    rng = np.random.default_rng(20260831)
    ev_lon, ev_lat = [], []
    with open(EVENTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ev_lon.append(float(r["lon"])); ev_lat.append(float(r["lat"]))
    ev_lon, ev_lat = np.array(ev_lon), np.array(ev_lat)

    summary = []
    for tag in tags:
        if not (GRID_DIR / f"{tag}.npz").exists():
            print(f"[skip] {tag}: 无网格")
            continue
        res = detect(tag, rng)
        events = res["events"]
        # 气候学一致性：检出点到最近 SAR 事件（任意日期）的中位距离，
        # 与刈幅内随机点零分布比较（内波在陆坡折带反复出现）
        clima = {}
        if events:
            etree = cKDTree(np.stack([ev_lon, ev_lat], axis=1))
            pts = np.array([[e["lon"], e["lat"]] for e in events])
            pts = pts[np.isfinite(pts).all(axis=1)]
        if events and len(pts):
            d_det = float(np.median(etree.query(pts)[0]) * 111.0)
            okp = np.isfinite(res["S"])
            rlon, rlat = res["lon"][okp], res["lat"][okp]
            okf = np.isfinite(rlon) & np.isfinite(rlat)
            rlon, rlat = rlon[okf], rlat[okf]
            null = []
            for _ in range(200):
                idx = rng.integers(0, rlon.size, len(events))
                null.append(float(np.median(
                    etree.query(np.stack([rlon[idx], rlat[idx]],
                                         axis=1))[0]) * 111.0))
            null = np.array(null)
            clima = {"det_median_dist_to_sar_event_km": round(d_det, 1),
                     "null_median_km": round(float(np.median(null)), 1),
                     "null_p05_km": round(float(np.percentile(null, 5)), 1),
                     "p_det_closer": round(float((null <= d_det).mean()), 3)}
        summary.append({"tag": tag, "n_events": len(events),
                        "n_null_events": res["n_null_events"],
                        "threshold": round(res["threshold"], 3),
                        "climatology_check": clima})
        with open(OUT / f"step4_detections_{tag}.json", "w",
                  encoding="utf-8") as f:
            json.dump({k: v for k, v in res.items()
                       if k not in ("S", "lon", "lat")}, f,
                      ensure_ascii=False, indent=2)
        with open(OUT / f"step4_detections_{tag}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(events[0].keys())
                               if events else ["lon"])
            w.writeheader()
            w.writerows(events)

        # 图：联合统计量 S + 检出点 + SAR 事件热区
        fig, ax = plt.subplots(figsize=(9, 7))
        st = 2
        okp = np.isfinite(res["S"][::st, ::st])
        vlim = np.nanpercentile(res["S"], 99)
        sc = ax.scatter(res["lon"][::st, ::st][okp], res["lat"][::st, ::st][okp],
                        c=res["S"][::st, ::st][okp], s=3, cmap="inferno",
                        vmin=0, vmax=vlim, linewidths=0)
        fig.colorbar(sc, ax=ax, label="检测统计量 C×A")
        if events:
            ax.scatter([e["lon"] for e in events], [e["lat"] for e in events],
                       facecolors="none", edgecolors="cyan", s=60,
                       label=f"检出 n={len(events)}")
        ina = ((ev_lon >= np.nanmin(res["lon"])) & (ev_lon <= np.nanmax(res["lon"]))
               & (ev_lat >= np.nanmin(res["lat"])) & (ev_lat <= np.nanmax(res["lat"])))
        ax.scatter(ev_lon[ina], ev_lat[ina], marker="x", c="deepskyblue",
                   s=20, label="SAR 事件（全部日期，气候参考）")
        ax.legend(fontsize=8)
        ax.set_xlabel("经度"); ax.set_ylabel("纬度")
        ax.set_title(f"原型检测器输出 {tag}（阈值=相位扰乱 p99.9）"
                     "\n无同日真值：与 SAR 事件热区的一致性仅为气候学参考")
        fig.tight_layout()
        fig.savefig(OUT / f"fig_step4_{tag}_detect.png", dpi=150)
        plt.close(fig)
        print(f"[{tag}] 检出 {len(events)} 个事件（零假设场同阈值检出 "
              f"{res['n_null_events']} 个）, 阈值 {res['threshold']:.3f}")

    # 与既有 summary 合并（按 tag 覆盖），避免子集运行时清空历史记录
    sum_path = OUT / "step4_detector_summary.json"
    if sum_path.exists():
        try:
            old = json.load(open(sum_path, encoding="utf-8"))
            old_map = {r["tag"]: r for r in old}
            new_map = {r["tag"]: r for r in summary}
            old_map.update(new_map)
            summary = [old_map[k] for k in sorted(old_map)]
        except Exception:
            pass
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"→ {OUT}/step4_*")


if __name__ == "__main__":
    main()
