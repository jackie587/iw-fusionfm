"""任务验收：004_243（2023-09-30，Δt≈1.06h）真值重叠区检验。

三层证据：
A. 盲检共定位：step4 检测器（相位扰乱 p99.9 阈值）候选与平移后 SAR 事件
   （direction ±180° 双假设，c=0.92 m/s）5km 共定位率；零假设=事件方向随机
   旋转后重算命中率（1000 次）。
B. 定向检验（免于盲检阈值）：检测统计量 S=C×A 在平移后事件位置的百分位，
   与刈幅内随机位置零分布比较（二项检验：>p95 的事件比例 vs 5% 期望）。
C. 关键图：全刈幅残差+事件+检出叠加；事件密集区放大；S 百分位分布。

输出 results/swot_isw_detection/：
  step6_validation_004243.json / fig_step6_004243_overview.png /
  fig_step6_004243_zoom.png / fig_step6_004243_targeted.png

用法（code/ 目录下）：
    python evaluation/swot_isw_step6_validation_004243.py
    python evaluation/swot_isw_step6_validation_004243.py \
        --day 2023-07-06 --label 574_008 --tags 20230706_left \
        --granule SWOT_L2_LR_SSH_Unsmoothed_574_008_20230706T122332_20230706T131356_PGC0_02.nc
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
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

from evaluation.swot_isw_step4_detector import (coherence_maps,
                                                phase_scramble)
from utils.geo import propagation_shift

OUT = PROJECT_ROOT / "results/swot_isw_detection"
GRID_DIR = OUT / "grids"
EVENTS_CSV = PROJECT_ROOT / "results/event_database/events.csv"
C_MED = 0.92
HIT_KM = 5.0
N_PERM = 1000
SEED = 20260908


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--day", default="2023-09-30")
    p.add_argument("--granule",
                   default="SWOT_L2_LR_SSH_Unsmoothed_004_243_"
                           "20230930T111956_20230930T121039_PGC0_01.nc")
    p.add_argument("--tags", nargs="+",
                   default=["20230930_left", "20230930_right"])
    p.add_argument("--label", default="004243")
    p.add_argument("--zoom", default=None,
                   help="lon1,lat1,lon2,lat2；缺省=+dir 覆盖事件 bbox+0.15°")
    return p.parse_args()


def load_events(day: str) -> list[dict]:
    with open(EVENTS_CSV, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["time_utc"].startswith(day)]


def granule_dt_h(events: list[dict], granule: str) -> float:
    ts = re.search(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_", granule)
    t0 = datetime.strptime(ts.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc)
    t1 = datetime.strptime(ts.group(2), "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc)
    swot_t = t0 + (t1 - t0) / 2
    sar_t = datetime.strptime(sorted(r["time_utc"] for r in events)
                              [len(events) // 2],
                              "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (swot_t - sar_t).total_seconds() / 3600.0


def propagate(events: list[dict], sign: int, dt_h: float,
              dir_jitter: np.ndarray | None = None,
              dir_offset: float = 0.0,
              fixed_az: float | None = None) -> tuple[np.ndarray, ...]:
    lo, la = [], []
    for i, r in enumerate(events):
        dg = float(r["direction_deg"]) + (0 if sign > 0 else 180) + dir_offset
        if fixed_az is not None:
            dg = fixed_az
        if dir_jitter is not None:
            dg += dir_jitter[i]
        dl, da = propagation_shift(float(r["lat"]), dg, C_MED, dt_h * 3600)
        lo.append(float(r["lon"]) + dl)
        la.append(float(r["lat"]) + da)
    return np.array(lo), np.array(la)


def s_enrichment(plo: np.ndarray, pla: np.ndarray, tree: cKDTree,
                 vS: np.ndarray, s_sorted: np.ndarray,
                 rng: np.random.Generator) -> dict:
    """定向检验：平移后事件位置的 S 百分位富集度 + 双零假设。"""
    d_ev, idx_ev = tree.query(np.stack([plo, pla], axis=1))
    covered = d_ev * 111.0 <= HIT_KM
    n_cov = int(covered.sum())
    if n_cov < 3:
        return {"n_covered": n_cov}
    s_ev = vS[idx_ev]
    pct = np.searchsorted(s_sorted, s_ev) / len(s_sorted)
    pct_cov = pct[covered]
    n_hi = int((pct_cov > 0.95).sum())
    p_binom = float(1.0 - np.sum([
        _binom_pmf(n_cov, k, 0.05) for k in range(n_hi)]))
    null_hi = np.empty(N_PERM)
    p95 = np.percentile(vS, 95)
    for i in range(N_PERM):
        pick = rng.integers(0, len(vS), n_cov)
        null_hi[i] = (vS[pick] > p95).sum()
    return {"n_covered": n_cov, "n_S_gt_p95": n_hi,
            "expected_at_random": round(n_cov * 0.05, 2),
            "p_binomial": round(p_binom, 4),
            "p_perm": round(float((null_hi >= n_hi).mean()), 4),
            "median_S_percentile": round(float(np.median(pct_cov)), 3)}


def main() -> None:
    args = parse_args()
    DAY, GRANULE, TAGS, LABEL = args.day, args.granule, args.tags, args.label
    rng = np.random.default_rng(SEED)
    events = load_events(DAY)
    dt_h = granule_dt_h(events, GRANULE)
    shift_km = C_MED * dt_h * 3.6
    print(f"{DAY}: n_events={len(events)}, Δt={dt_h:.2f}h, 平移 {shift_km:.1f} km")

    report: dict = {"day": DAY, "granule": GRANULE, "dt_h": round(dt_h, 3),
                    "shift_km": round(shift_km, 2), "hit_km": HIT_KM,
                    "halves": {}}
    fig_overview, axes_ov = plt.subplots(1, len(TAGS),
                                         figsize=(7.5 * len(TAGS), 7),
                                         squeeze=False)
    axes_ov = axes_ov[0]
    targeted_rows = []

    for ax, tag in zip(axes_ov, TAGS):
        z = np.load(GRID_DIR / f"{tag}.npz")
        lon, lat = z["lon"].astype(np.float64), z["lat"].astype(np.float64)
        resid = z["resid"].astype(np.float64)
        mask = np.isfinite(resid)
        mask_er = ndimage.binary_erosion(mask, iterations=10,
                                         border_value=0)
        fill = np.where(mask_er, resid, 0.0)
        C, A, theta_star = coherence_maps(fill)
        cov = ndimage.gaussian_filter(mask_er.astype(np.float64), 8.0)
        S = np.where(cov > 0.8, C * A, np.nan)

        okv = np.isfinite(S).ravel()
        vlon, vlat = lon.ravel()[okv], lat.ravel()[okv]
        vS = S.ravel()[okv]
        tree = cKDTree(np.stack([vlon, vlat], axis=1))
        s_sorted = np.sort(vS)

        # 跨轨列内秩归一化 S：KaRIn 跨轨噪声在刈幅两侧边缘系统性偏高
        # （实测边缘列 S 均值≈中间列 2~3 倍），全局百分位会把"向东平移"
        # 的伪富集当成信号；列内秩消除该仪器几何效应。
        ny_, nx_ = S.shape
        Scol = np.full_like(S, np.nan)
        for c in range(nx_):
            v = S[:, c]
            okc = np.isfinite(v)
            if okc.sum() > 10:
                order = np.argsort(v[okc])
                ranks = np.empty(okc.sum())
                ranks[order] = np.arange(okc.sum()) / (okc.sum() - 1)
                Scol[np.nonzero(okc)[0], c] = ranks
        vS_col = Scol.ravel()[okv]
        s_sorted_col = np.sort(vS_col)

        det = json.load(open(OUT / f"step4_detections_{tag}.json",
                             encoding="utf-8"))
        det_pts = np.array([[e["lon"], e["lat"]] for e in det["events"]]) \
            if det["events"] else np.empty((0, 2))

        half = {"n_detections": len(det_pts), "signs": {}}
        for sign, snm in ((+1, "+dir"), (-1, "-dir")):
            plo, pla = propagate(events, sign, dt_h)
            d_ev, _ = tree.query(np.stack([plo, pla], axis=1))
            covered = d_ev * 111.0 <= HIT_KM
            # A. 盲检共定位
            if len(det_pts):
                d_hit, _ = cKDTree(det_pts).query(
                    np.stack([plo, pla], axis=1))
                hits = covered & (d_hit * 111.0 <= HIT_KM)
            else:
                d_hit = np.full(len(events), np.inf)
                hits = np.zeros(len(events), bool)
            # 零分布：方向随机旋转后重算命中数
            null_hits = np.empty(N_PERM)
            for i in range(N_PERM):
                jlo, jla = propagate(events, sign, dt_h,
                                     rng.uniform(0, 360, len(events)))
                dj, _ = tree.query(np.stack([jlo, jla], axis=1))
                covj = dj * 111.0 <= HIT_KM
                if len(det_pts):
                    dhj, _ = cKDTree(det_pts).query(
                        np.stack([jlo, jla], axis=1))
                    null_hits[i] = (covj & (dhj * 111.0 <= HIT_KM)).sum()
                else:
                    null_hits[i] = 0
            # B. 定向检验：事件处 S 百分位（全局百分位 + 跨轨列归一化两版）
            tgt = s_enrichment(plo, pla, tree, vS, s_sorted, rng)
            tgt_col = s_enrichment(plo, pla, tree, vS_col, s_sorted_col, rng)
            n_cov = tgt.get("n_covered", 0)
            n_hi = tgt.get("n_S_gt_p95", 0)
            # B2. 空间梯度对照：同位移量、非传播方向的富集度
            # （若对照也显著富集 → S 场空间梯度伪影，物理结论不成立）
            ctrls = {}
            for cname, kw in (("perp+90", dict(dir_offset=90.0)),
                              ("perp-90", dict(dir_offset=-90.0)),
                              ("fixed_E", dict(fixed_az=90.0)),
                              ("fixed_W", dict(fixed_az=270.0)),
                              ("fixed_N", dict(fixed_az=0.0)),
                              ("fixed_S", dict(fixed_az=180.0))):
                clo, cla = propagate(events, sign, dt_h, **kw)
                ctrls[cname] = s_enrichment(clo, cla, tree, vS, s_sorted, rng)

            half["signs"][snm] = {
                "n_covered": n_cov,
                "n_hits_5km": int(hits.sum()),
                "hit_rate_vs_covered": (round(float(hits.sum() / n_cov), 3)
                                        if n_cov else None),
                "null_hits_median": float(np.median(null_hits)),
                "null_hits_p95": float(np.percentile(null_hits, 95)),
                "p_hits": round(float((null_hits >= hits.sum()).mean()), 4),
                "targeted": tgt,
                "targeted_xtnorm": tgt_col,
                "gradient_controls": ctrls,
            }
            if n_cov >= 3:
                d_all, idx_all = tree.query(np.stack([plo, pla], axis=1))
                pct_all = np.searchsorted(s_sorted, vS[idx_all]) / len(vS)
                pct_col_all = vS_col[idx_all]
                for r, covf, pc, pcc in zip(events, d_all * 111.0 <= HIT_KM,
                                            pct_all, pct_col_all):
                    if covf:
                        targeted_rows.append({
                            "event_id": r["event_id"], "half": tag, "sign": snm,
                            "quality": r["quality"],
                            "wavelength_m": r["wavelength_m"],
                            "S_percentile": round(float(pc), 3),
                            "S_percentile_xtnorm": round(float(pcc), 3)})

        report["halves"][tag] = half

        # 图 1：全刈幅叠加（+dir 平移）
        st = 2
        okp = np.isfinite(resid[::st, ::st])
        vlim = np.nanpercentile(np.abs(resid), 98)
        sc = ax.scatter(lon[::st, ::st][okp], lat[::st, ::st][okp],
                        c=resid[::st, ::st][okp], s=3, cmap="RdBu_r",
                        vmin=-vlim, vmax=vlim, linewidths=0)
        fig_overview.colorbar(sc, ax=ax, label="SSHA 残差 (m)", shrink=0.8)
        elo = np.array([float(r["lon"]) for r in events])
        ela = np.array([float(r["lat"]) for r in events])
        plo, pla = propagate(events, +1, dt_h)
        ax.scatter(elo, ela, marker="x", c="gray", s=25, label="SAR 事件(原始)")
        ax.scatter(plo, pla, facecolors="none", edgecolors="red", s=45,
                   label=f"+dir 平移 {shift_km:.1f}km")
        if len(det_pts):
            ax.scatter(det_pts[:, 0], det_pts[:, 1], marker="*", c="cyan",
                       s=200, edgecolors="k", label=f"检测器候选 n={len(det_pts)}")
        ax.legend(fontsize=8)
        ax.set_xlabel("经度"); ax.set_ylabel("纬度")
        hd = half["signs"]["+dir"]
        ax.set_title(f"{tag}：命中 {hd['n_hits_5km']}/{hd['n_covered']} "
                     f"（零分布中位 {hd['null_hits_median']:.0f}）")
    fig_overview.suptitle(f"{DAY} {LABEL} 250m 残差 × SAR 事件 × 检测器候选"
                          f"（Δt={dt_h:.2f}h）")
    fig_overview.tight_layout()
    fig_overview.savefig(OUT / f"fig_step6_{LABEL}_overview.png", dpi=150)
    plt.close(fig_overview)

    # 图 2：事件密集区放大（取覆盖事件最多的半刈幅，窗口=覆盖事件 bbox+边距）
    best_tag, best_n, best_cov = None, 0, None
    for tag, h in report["halves"].items():
        t = h["signs"]["+dir"]
        if t.get("n_covered", 0) > best_n:
            best_n, best_tag = t["n_covered"], tag
    if args.zoom:
        win = tuple(float(x) for x in args.zoom.split(","))
    else:
        plo_all, pla_all = propagate(events, +1, dt_h)
        # 覆盖事件位置（+dir）的包围盒
        zz0 = np.load(GRID_DIR / f"{best_tag}.npz")
        lons0, lats0 = zz0["lon"], zz0["lat"]
        inb0 = np.isfinite(zz0["resid"])
        tr0 = cKDTree(np.stack([lons0.ravel()[inb0.ravel()],
                                lats0.ravel()[inb0.ravel()]], axis=1))
        dd0, _ = tr0.query(np.stack([plo_all, pla_all], axis=1))
        cov0 = dd0 * 111.0 <= HIT_KM
        if cov0.any():
            win = (float(plo_all[cov0].min()) - 0.15,
                   float(pla_all[cov0].min()) - 0.15,
                   float(plo_all[cov0].max()) + 0.15,
                   float(pla_all[cov0].max()) + 0.15)
        else:
            win = (float(plo_all.min()) - 0.15, float(pla_all.min()) - 0.15,
                   float(plo_all.max()) + 0.15, float(pla_all.max()) + 0.15)
    z = np.load(GRID_DIR / f"{best_tag}.npz")
    lon, lat = z["lon"].astype(np.float64), z["lat"].astype(np.float64)
    resid = z["resid"].astype(np.float64)
    mask_er = ndimage.binary_erosion(np.isfinite(resid), iterations=10,
                                     border_value=0)
    fill = np.where(mask_er, resid, 0.0)
    C, A, _ = coherence_maps(fill)
    cov = ndimage.gaussian_filter(mask_er.astype(np.float64), 8.0)
    S = np.where(cov > 0.8, C * A, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, fld, nm, cm in ((axes[0], resid, "SSHA 残差", "RdBu_r"),
                            (axes[1], S, "检测统计量 C×A", "inferno")):
        inw = ((lon >= win[0]) & (lon <= win[2])
               & (lat >= win[1]) & (lat <= win[3]))
        okw = inw & np.isfinite(fld)
        v = fld[okw]
        vlim = np.nanpercentile(np.abs(v), 98) if nm == "SSHA 残差" \
            else np.nanpercentile(v, 99)
        sc = ax.scatter(lon[okw], lat[okw], c=v, s=6, cmap=cm,
                        vmin=(-vlim if nm == "SSHA 残差" else 0), vmax=vlim,
                        linewidths=0)
        fig.colorbar(sc, ax=ax, label=nm, shrink=0.8)
        elo = np.array([float(r["lon"]) for r in events])
        ela = np.array([float(r["lat"]) for r in events])
        inwe = ((elo >= win[0]) & (elo <= win[2])
                & (ela >= win[1]) & (ela <= win[3]))
        plo, pla = propagate(events, +1, dt_h)
        ax.scatter(elo[inwe], ela[inwe], marker="x", c="gray", s=35,
                   label="SAR 事件(原始)")
        ax.scatter(plo[inwe], pla[inwe], facecolors="none", edgecolors="red",
                   s=55, label="+dir 平移后")
        for r, x0, y0, x1, y1 in zip(events, elo, ela, plo, pla):
            if not (win[0] <= x0 <= win[2] and win[1] <= y0 <= win[3]):
                continue
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-", color="r",
                                        alpha=0.3, lw=0.7))
            ax.annotate(r["event_id"][-4:], (x1, y1), fontsize=6,
                        color="darkred")
        ax.set_xlim(win[0], win[2]); ax.set_ylim(win[1], win[3])
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlabel("经度"); ax.set_ylabel("纬度")
        ax.set_title(f"{nm}（事件密集区放大）")
    fig.suptitle(f"{DAY} {LABEL} 事件密集区放大（红圈=+dir 平移后位置）")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_step6_{LABEL}_zoom.png", dpi=150)
    plt.close(fig)

    # 图 3：定向检验——事件处 S 百分位直方图 vs 均匀零分布
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ttl in ((axes[0], "S_percentile", "全局百分位（含跨轨边缘偏差）"),
                         (axes[1], "S_percentile_xtnorm",
                          "跨轨列归一化百分位（正确零假设）")):
        pcs = [r[key] for r in targeted_rows]
        if not pcs:
            continue
        ax.hist(pcs, bins=20, range=(0, 1), alpha=0.7, color="steelblue",
                label=f"事件处 S 百分位（n={len(pcs)}）")
        ax.axhline(len(pcs) / 20, color="r", ls="--",
                   label="随机位置期望（均匀）")
        ax.set_xlabel("检测统计量 S 的百分位")
        ax.set_ylabel("事件数")
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("定向检验：若 SWOT 可见内波，事件应集中于高分位"
                 "（+dir/-dir 两假设合并展示）")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_step6_{LABEL}_targeted.png", dpi=150)
    plt.close(fig)

    # 图 4：列归一化后仍 >p95 的覆盖事件逐一放大（残差 + S 两联）
    hi_events = [r for r in targeted_rows if r["S_percentile_xtnorm"] > 0.95]
    if hi_events:
        grids = {}
        fig, axes = plt.subplots(len(hi_events), 2,
                                 figsize=(11, 4.2 * len(hi_events)),
                                 squeeze=False)
        for row, hr in enumerate(hi_events):
            tag = hr["half"]
            if tag not in grids:
                zz = np.load(GRID_DIR / f"{tag}.npz")
                rresid = zz["resid"].astype(np.float64)
                mmask = ndimage.binary_erosion(np.isfinite(rresid),
                                               iterations=10, border_value=0)
                ffill = np.where(mmask, rresid, 0.0)
                CC, AA, _ = coherence_maps(ffill)
                ccov = ndimage.gaussian_filter(mmask.astype(np.float64), 8.0)
                grids[tag] = (zz["lon"].astype(np.float64),
                              zz["lat"].astype(np.float64), rresid,
                              np.where(ccov > 0.8, CC * AA, np.nan))
            glon, glat, gresid, gS = grids[tag]
            ev_r = next(r for r in events if r["event_id"] == hr["event_id"])
            sg = 1 if hr["sign"] == "+dir" else -1
            plo, pla = propagate([ev_r], sg, dt_h)
            elo0, ela0 = float(ev_r["lon"]), float(ev_r["lat"])
            mwin = 12.0 / 111.0
            inw = ((glon >= plo[0] - mwin) & (glon <= plo[0] + mwin)
                   & (glat >= pla[0] - mwin) & (glat <= pla[0] + mwin))
            for col, (fld, nm, cm) in enumerate(
                    ((gresid, "SSHA 残差 (m)", "RdBu_r"),
                     (gS, "检测统计量 C×A", "inferno"))):
                ax = axes[row][col]
                okw = inw & np.isfinite(fld)
                vv = fld[okw]
                if not vv.size:
                    continue
                vl = (np.nanpercentile(np.abs(vv), 98) if cm == "RdBu_r"
                      else np.nanpercentile(vv, 99))
                sc = ax.scatter(glon[okw], glat[okw], c=vv, s=10, cmap=cm,
                                vmin=(-vl if cm == "RdBu_r" else 0), vmax=vl,
                                linewidths=0)
                fig.colorbar(sc, ax=ax, label=nm, shrink=0.8)
                ax.plot(elo0, ela0, "kx", ms=10, mew=2, label="SAR 原始位置")
                ax.plot(plo[0], pla[0], "o", mfc="none", mec="r", ms=14,
                        mew=2, label=f"{hr['sign']} 平移 {shift_km:.1f}km")
                ax.legend(fontsize=7, loc="lower left")
                ax.set_title(f"{hr['event_id'][-4:]} {tag} {hr['sign']} "
                             f"λ={ev_r['wavelength_m']}m "
                             f"q={ev_r['quality']} "
                             f"S列分位={hr['S_percentile_xtnorm']:.2f}",
                             fontsize=9)
        fig.suptitle("列归一化后 S>p95 的覆盖事件逐一放大")
        fig.tight_layout()
        fig.savefig(OUT / f"fig_step6_{LABEL}_hievents.png", dpi=150)
        plt.close(fig)

    with open(OUT / f"step6_validation_{LABEL}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": report, "per_event": targeted_rows}, f,
                  ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"→ {OUT}/step6_validation_{LABEL}.json, fig_step6_{LABEL}_*.png")


def _binom_pmf(n: int, k: int, p: float) -> float:
    from math import comb
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


if __name__ == "__main__":
    main()
