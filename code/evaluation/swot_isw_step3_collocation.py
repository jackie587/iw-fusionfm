"""任务 3：SAR 事件 × SWOT Expert 2km 共定位检验（含传播平移校正）。

事件级覆盖普查（step0_coverage_census.json）后，只有三类可用配对：
  A. 2023-09-30 ↔ 004_243  Δt=+1.06h（近同时，但仅 5 个事件在有效像元 5km 内）
  B. 2023-08-20 ↔ 002_284  Δt=+18.83h（次日过顶，89 个事件 ≤5km；c·Δt≈62km）
  C. 2023-10-10 ↔ 004_549  Δt=+23.17h（次日过顶，55 个事件 ≤5km；c·Δt≈77km）
7 月定标轨配对（07-01/06/08）经事件级距离核查实际覆盖≈0，弃用。

方法：
- Expert 质控（ssha_karin_qual==0）；NaN 感知高斯去背景（σ=5km）得残差，
  再做 σ=2km 平滑到"波包包络尺度"（Expert 2km  postings 无法分辨
  0.4~2km 单个波峰，只能检验包络级同号起伏）；
- 事件沿 direction_deg 平移 c·Δt（direction 有 180° 模糊，两符号都检验；
  c 取 0.83/0.92/1.03 m/s 做敏感性）；
- 统计量：平移后事件质心处包络残差均值（下沉型 ISW 预期正凸起）。
  零假设：刈幅内随机位置置换（5000 次，分层于同一刈幅有效区）；
  方向对照：垂直于传播方向平移同距离；
- 每对出图：包络残差底图 + 原始事件(灰×) + 平移事件(红圈+位移箭头)。

用法（code/ 目录下）：python evaluation/swot_isw_step3_collocation.py
"""
from __future__ import annotations

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

from data_preprocessing.swot_preprocess.quality_control import load_and_qc
from utils.geo import propagation_shift

SWOT_DIR = PROJECT_ROOT / "data/raw/swot"
EVENTS_CSV = PROJECT_ROOT / "results/event_database/events.csv"
OUT = PROJECT_ROOT / "results/swot_isw_detection"

PAIRS = [
    {"day": "2023-09-30", "granule": "004_243_20230930T111959",
     "case": "A_near_simultaneous"},
    {"day": "2023-08-20", "granule": "002_284_20230821T045823",
     "case": "B_next_day"},
    {"day": "2023-10-10", "granule": "004_549_20231011T094159",
     "case": "C_next_day"},
]
C_MED, C_P25, C_P75 = 0.92, 0.83, 1.03   # m/s，反演相速度
SIGMA_BG_PX = 2.5                        # 去背景 σ=5km（2km 网格）
SIGMA_ENV_PX = 1.0                       # 包络平滑 σ=2km
MAX_DEG = 0.03                           # 最近邻采样距离上限（≈3.3km）
N_PERM = 5000
SEED = 20260831


def nan_gauss(arr: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    w = ndimage.gaussian_filter(valid.astype(np.float64), sigma)
    s = ndimage.gaussian_filter(filled, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 1e-3, s / np.maximum(w, 1e-3), np.nan)


def parse_t(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def load_events(day: str) -> list[dict]:
    with open(EVENTS_CSV, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["time_utc"].startswith(day)]


def run_pair(pair: dict, rng: np.random.Generator) -> dict:
    day = pair["day"]
    files = sorted(SWOT_DIR.glob(f"SWOT_L2_LR_SSH_Expert_{pair['granule']}*.nc"))
    if not files:
        return {"day": day, "error": "granule 缺失"}
    qc = load_and_qc(str(files[0]))
    lon2d, lat2d = qc["lon"].astype(np.float64), qc["lat"].astype(np.float64)
    ssha, mask = qc["ssh"], qc["mask"] > 0

    # 只保留 AOI 附近的行，减少内存（事件都在 18~23N）
    rowok = ((lat2d > 15.0) & (lat2d < 26.0)).any(axis=1)
    ridx = np.nonzero(rowok)[0]
    if not len(ridx):
        return {"day": day, "error": "刈幅不经过 AOI 纬度带"}
    sl = np.s_[ridx[0]:ridx[-1] + 1, :]
    lon2d, lat2d, ssha, mask = (a[sl] for a in (lon2d, lat2d, ssha, mask))
    ssha = np.where(mask, ssha, np.nan)
    resid = ssha - nan_gauss(ssha, SIGMA_BG_PX)
    resid[~mask] = np.nan
    env = nan_gauss(resid, SIGMA_ENV_PX)       # 波包包络尺度残差

    flat_lon, flat_lat = lon2d.ravel(), lat2d.ravel()
    flat_valid = np.isfinite(env).ravel()
    flat_val = np.where(flat_valid, env.ravel(), 0.0)
    tree = cKDTree(np.stack([flat_lon, flat_lat], axis=1))

    events = load_events(day)
    ev = [r for r in events if r["quality"] in ("high", "medium")] or events
    sar_t = parse_t(events[0]["time_utc"])
    ts = re.search(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_", files[0].name)
    t0 = datetime.strptime(ts.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc)
    t1 = datetime.strptime(ts.group(2), "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc)
    swot_t = t0 + (t1 - t0) / 2
    dt_h = (swot_t - sar_t).total_seconds() / 3600.0

    e_lon = np.array([float(r["lon"]) for r in ev])
    e_lat = np.array([float(r["lat"]) for r in ev])
    e_dir = np.array([float(r["direction_deg"]) for r in ev])

    def shifted(lon, lat, direc, sign, c, perp=False):
        out_lo = np.empty(len(lon)); out_la = np.empty(len(lat))
        for i in range(len(lon)):
            az = direc[i] + (90.0 if perp else 0.0) + (0.0 if sign > 0 else 180.0)
            dl, da = propagation_shift(lat[i], az, c, dt_h * 3600.0)
            out_lo[i] = lon[i] + dl; out_la[i] = lat[i] + da
        return out_lo, out_la

    def sample(lo, la):
        dist, idx = tree.query(np.stack([lo, la], axis=1))
        okk = (dist <= MAX_DEG) & flat_valid[idx]
        return np.where(okk, flat_val[idx], np.nan)

    vidx = np.nonzero(flat_valid)[0]
    result = {"day": day, "case": pair["case"], "granule": files[0].name,
              "n_events_total": len(events), "n_hq": len(ev),
              "sar_time": events[0]["time_utc"],
              "swot_time": swot_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "dt_h": round(dt_h, 2),
              "env_resid_rms_m": float(np.nanstd(env)), "tests": {}}

    for sign, ttl in ((+1, "+dir"), (-1, "-dir")):
        lo, la = shifted(e_lon, e_lat, e_dir, sign, C_MED)
        vals = sample(lo, la)
        ok = np.isfinite(vals)
        n_ok = int(ok.sum())
        shift_km = C_MED * dt_h * 3.6
        rec = {"shift_km_at_c_med": round(shift_km, 1), "n_matched": n_ok}
        if n_ok >= 3:
            obs = float(np.nanmean(vals))
            obs_abs = float(np.nanmean(np.abs(vals)))
            # 置换零分布（随机有效位置，符号/绝对值两种）
            perm = np.empty(N_PERM); perm_abs = np.empty(N_PERM)
            for i in range(N_PERM):
                pick = rng.choice(vidx, size=n_ok, replace=False)
                perm[i] = flat_val[pick].mean()
                perm_abs[i] = np.abs(flat_val[pick]).mean()
            rec.update({
                "obs_mean_m": round(obs, 5),
                "obs_absmean_m": round(obs_abs, 5),
                "perm_mean_m": round(float(perm.mean()), 5),
                "perm_std_m": round(float(perm.std()), 5),
                "z_vs_perm": round(float((obs - perm.mean()) / perm.std()), 2),
                "p_two_perm": round(float((np.abs(perm) >= abs(obs)).mean()), 4),
                "p_abs_perm": round(float((perm_abs >= obs_abs).mean()), 4),
            })
            # 垂直方向对照
            plo, pla = shifted(e_lon, e_lat, e_dir, sign, C_MED, perp=True)
            cv = sample(plo, pla)
            cv = cv[np.isfinite(cv)]
            if cv.size >= 3:
                rec["ctrl_perp_mean_m"] = round(float(cv.mean()), 5)
                rec["diff_vs_perp_m"] = round(obs - float(cv.mean()), 5)
            # c 敏感性
            sens = {}
            for nm, c in (("c_p25", C_P25), ("c_p75", C_P75)):
                lo2, la2 = shifted(e_lon, e_lat, e_dir, sign, c)
                v2 = sample(lo2, la2)
                v2 = v2[np.isfinite(v2)]
                sens[nm] = round(float(v2.mean()), 5) if v2.size >= 3 else None
            rec["c_sensitivity_mean_m"] = sens
        result["tests"][ttl] = rec

    # 图：包络残差 + 事件（+dir 平移）
    fig, ax = plt.subplots(figsize=(9, 7))
    lo, la = shifted(e_lon, e_lat, e_dir, +1, C_MED)
    st = 2
    okp = np.isfinite(env[::st, ::st])
    vlim = np.nanpercentile(np.abs(env), 98) if okp.any() else 0.05
    sc = ax.scatter(lon2d[::st, ::st][okp], lat2d[::st, ::st][okp],
                    c=env[::st, ::st][okp], s=4, cmap="RdBu_r",
                    vmin=-vlim, vmax=vlim, linewidths=0)
    fig.colorbar(sc, ax=ax, label="包络尺度 SSHA 残差 (m)")
    ax.scatter(e_lon, e_lat, marker="x", c="gray", s=30,
               label=f"SAR 事件原始位置（{day}）")
    okv = np.isfinite(sample(lo, la))
    ax.scatter(lo[okv], la[okv], facecolors="none", edgecolors="r", s=60,
               label=f"+dir 平移 {C_MED * dt_h * 3.6:.1f} km 后")
    for a, b, c2, d in zip(e_lon, e_lat, lo, la):
        ax.annotate("", xy=(c2, d), xytext=(a, b),
                    arrowprops=dict(arrowstyle="-", color="r", alpha=0.25,
                                    lw=0.6))
    ax.legend(fontsize=8)
    ax.set_xlabel("经度"); ax.set_ylabel("纬度")
    t = result["tests"]["+dir"]
    extra = (f"，平移后残差均值 {t['obs_mean_m'] * 100:+.2f} cm "
             f"(p={t['p_two_perm']:.3f})" if "obs_mean_m" in t else "")
    ax.set_title(f"{day} <-> {pair['granule'][:7]}  Δt={dt_h:.1f}h "
                 f"n_matched={t['n_matched']}{extra}")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_step3_{day}_collocation.png", dpi=150)
    plt.close(fig)
    print(f"[{day}] Δt={dt_h:.2f}h tests="
          f"{json.dumps(result['tests'], ensure_ascii=False)}")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = [run_pair(p, rng) for p in PAIRS]
    with open(OUT / "step3_collocation_report.json", "w",
              encoding="utf-8") as f:
        json.dump({"note": "事件级共定位检验；direction_deg 有 180° 模糊，"
                           "±dir 均检验；统计量为包络尺度(σ=2km)残差均值",
                   "c_med": C_MED, "c_range": [C_P25, C_P75],
                   "n_perm": N_PERM, "pairs": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"→ {OUT}/step3_collocation_report.json, fig_step3_*.png")


if __name__ == "__main__":
    main()
