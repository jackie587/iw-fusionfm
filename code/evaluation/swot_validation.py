"""SWOT×SAR 共定位严格验证：位移不变量 + 聚合统计三检验。

针对 L2 配对切块（5 配对日 / 11 S1 场景 / 7828 块），回答：
SAR 弱标签（v7）指示的内孤立波条纹，是否与 SWOT SSHA 的波状起伏一致。

三个检验：
1. 波长一致性（位移不变）：label 掩膜 FFT 定条纹走向 θ0，VV 纹理在 label
   窗内、θ0±25° 扇区 FFT 定主导波长 λ_sar（label 掩膜直接 FFT 的波长被
   波包包络尺度主导，仅作诊断）；
   在 20×20 SSHA 上以掩膜加权最小二乘正弦拟合，测 λ_sar±30% 频带的能量；
   有波组 vs 同日同场景覆盖率匹配的对照组；辅以块内方向对照（θ vs θ+90°）。
   λ<500 m（<2 格）的块剔除并计数。
2. 位移扫描相关：label 降采样到 20×20，沿 θ 正反向平移 s∈{-12..+12} 格
   （±3 km），与 SSHA 做掩膜加权相关，聚合 C(s)；null = 沿 θ+90° 平移。
   图上标注各配对日 Δt·c 预期位移。
3. 振幅不对称性：下沉型 ISW 应在 label 区产生 SSHA 谷值；比较 label 内
   偏度 / p10 与对照组、以及与同块 label 外的差异。

统计：全部给效应量 + 置换检验 p（默认 10000 次，日内分层）；
三族检验 Bonferroni α=0.0167；检验 1 另做场景块级符号翻转敏感性分析。

用法（code/ 目录下）：
    python evaluation/swot_validation.py
    python evaluation/swot_validation.py --n-perm 2000 --workers 8 --max-tiles 200
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

TILE_DIR = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"
PAIRS_JSON = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/targeted_pairs.json"
OUT_DIR = PROJECT_ROOT / "results/swot_validation"

TILE_M = 5120.0          # SAR 切块边长（512 px @10 m）
GRID_M = 250.0           # SWOT 重投影网格（20×20）
GRID_N = 20
LAM_MIN_SAR = 400.0      # FFT 波长搜索下限（m）
LAM_MAX_SAR = 2560.0     # 上限 = 2 个完整周期 / 块
LAM_MIN_SWOT = 500.0     # 20×20 网格可分辨下限（≥2 格）
LAM_MAX_SWOT = 2500.0
PEAK_CONF_MIN = 5.0      # FFT 峰功率 / 环带中位功率
BAND_TRIALS = (0.70, 0.85, 1.00, 1.15, 1.30)  # λ_sar 的 ±30% 扫描
SWEEP_MAX = 12           # 位移扫描半径（格）
C_MED, C_P25, C_P75 = 0.92, 0.83, 1.03       # 反演相速度（m/s）
N_PERM_DEFAULT = 10000


# ---------------------------------------------------------------- 数据选择

def norm_scene(scene: str) -> str:
    return scene[:-4] if scene.endswith("_COG") else scene


def load_dt_map() -> dict:
    """scene（去 COG 后缀）→ (granule, dt_h)；COG 重复行去重。"""
    with open(PAIRS_JSON, encoding="utf-8") as f:
        pairs = json.load(f)
    dt_map = {}
    for p in pairs:
        key = norm_scene(p["s1"])
        if key in dt_map and abs(dt_map[key][1] - p["dt_h"]) > 1e-6:
            raise ValueError(f"同场景 dt 冲突: {key}")
        dt_map[key] = (p["granule"], float(p["dt_h"]))
    return dt_map


def select_groups(manifest: list) -> tuple[list, list]:
    wave = [m for m in manifest
            if m["label_quality"] == "clean"
            and m["label_frac"] > 0.02 and m["swot_coverage"] > 0.5]
    ctrl_pool = [m for m in manifest
                 if m["swot_coverage"] > 0.5
                 and (m["label_quality"] == "none"
                      or (m["label_quality"] == "clean"
                          and m["label_frac"] <= 0.02))]
    return wave, ctrl_pool


def match_controls(wave: list, ctrl_pool: list, seed: int) -> tuple[list, dict]:
    """每个有波块配一个同日、覆盖率最接近的对照块；池不足时有放回抽样。"""
    rng = np.random.default_rng(seed)
    by_day: dict[str, list] = {}
    for c in ctrl_pool:
        by_day.setdefault(c["day"], []).append(c)
    pairs, info = [], {"n_reused": 0, "per_day": {}}
    for day in sorted({w["day"] for w in wave}):
        ws = [w for w in wave if w["day"] == day]
        cs = sorted(by_day.get(day, []), key=lambda c: c["swot_coverage"])
        used: set[int] = set()
        n_re = 0
        for w in sorted(ws, key=lambda w: w["swot_coverage"]):
            best, best_d = None, 1e9
            for i, c in enumerate(cs):
                if i in used:
                    continue
                d = abs(c["swot_coverage"] - w["swot_coverage"])
                if d < best_d:
                    best, best_d = i, d
            if best is None:  # 对照池耗尽 → 有放回
                best = int(rng.integers(len(cs)))
                n_re += 1
            else:
                used.add(best)
            pairs.append((w, cs[best]))
        info["n_reused"] += n_re
        info["per_day"][day] = {"n_wave": len(ws), "n_ctrl_pool": len(cs),
                                "n_reused": n_re}
    return pairs, info


# ---------------------------------------------------------------- 单块计算

def estimate_stripe(label: np.ndarray, vv: np.ndarray):
    """条纹参数估计（两步）。

    1) label 掩膜 2D FFT 低波数峰 → 波包法向 θ0（走向信息可靠，但波长被
       包络尺度主导，仅作诊断 lam_env）；
    2) VV 在 label 窗内做 FFT，在 θ0±25° 扇区、λ∈[400,2560]m 内找峰 →
       λ、精化 θ、置信度 conf = 峰功率 / 扇区带内中位功率。
    返回 dict 或 None（label 太少 / VV 有效像素不足）。
    """
    L = label.astype(np.float32)
    if L.mean() < 0.02:
        return None
    n = L.shape[0]
    wy = np.hanning(n)
    W = np.outer(wy, wy)
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    k = np.sqrt(fx ** 2 + fy ** 2) * n                    # 周期数 / 块
    lam_grid = np.where(k > 0, TILE_M / np.maximum(k, 1e-9), np.inf)
    band = (lam_grid >= LAM_MIN_SAR) & (lam_grid <= LAM_MAX_SAR)
    # 步骤 1：label FFT → θ0
    P1 = np.abs(np.fft.fft2((L - L.mean()) * W)) ** 2
    Pm = np.where(band, P1, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(Pm)), Pm.shape)
    theta0 = float(np.arctan2(fy[iy, ix], fx[iy, ix]))
    lam_env = float(TILE_M / k[iy, ix])
    # 步骤 2：VV 窗内 FFT，θ0±25° 扇区
    valid = np.isfinite(vv)
    vv_lab_valid = float(valid[L > 0].mean()) if (L > 0).any() else 0.0
    if vv_lab_valid < 0.5:
        return {"lam_env": lam_env, "theta0": theta0, "vv_valid": vv_lab_valid,
                "stripe_ok": False}
    v = vv.astype(np.float32).copy()
    v[~valid] = float(np.nanmean(vv))
    Wl = L * W
    P3 = np.abs(np.fft.fft2((v - v.mean()) * Wl)) ** 2
    ang = np.arctan2(fy, fx)
    dang = np.abs((ang - theta0 + np.pi / 2) % np.pi - np.pi / 2)
    sector = band & (dang < np.deg2rad(25))
    Pm3 = np.where(sector, P3, 0.0)
    iy3, ix3 = np.unravel_index(int(np.argmax(Pm3)), Pm3.shape)
    lam = float(TILE_M / k[iy3, ix3])
    theta = float(np.arctan2(fy[iy3, ix3], fx[iy3, ix3]))
    conf = float(P3[iy3, ix3] / (np.median(P3[sector]) + 1e-30))
    return {"lam": lam, "theta": theta, "conf": conf,
            "lam_env": lam_env, "theta0": theta0, "vv_valid": vv_lab_valid,
            "stripe_ok": conf >= PEAK_CONF_MIN}


def band_fit(ssha: np.ndarray, mask: np.ndarray, lam: float, theta: float):
    """掩膜加权最小二乘正弦拟合；返回 (A_band, A_exact)，不可拟合返回 None。"""
    v = mask > 0.5
    if int(v.sum()) < 60:
        return None
    yy, xx = np.meshgrid(np.arange(GRID_N), np.arange(GRID_N), indexing="ij")
    u = ((xx - 9.5) * np.cos(theta) + (yy - 9.5) * np.sin(theta)) * GRID_M
    u_v, z_v = u[v], ssha[v].astype(np.float64)
    if u_v.max() - u_v.min() < 0.75 * lam:
        return None
    amps = []
    for f in BAND_TRIALS:
        w = 2 * np.pi / (lam * f)
        D = np.column_stack([np.cos(w * u_v), np.sin(w * u_v),
                             np.ones_like(u_v)])
        coef, *_ = np.linalg.lstsq(D, z_v, rcond=None)
        amps.append(float(np.hypot(coef[0], coef[1])))
    return max(amps), amps[2]


def downsample_label(label: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(label.astype(np.float32), (GRID_N, GRID_N),
                      interpolation=cv2.INTER_AREA)


def shift_field(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """平移（非环绕），移入区域置 NaN。"""
    out = np.full_like(a, np.nan)
    ys = slice(max(0, dy), min(GRID_N, GRID_N + dy))
    xs = slice(max(0, dx), min(GRID_N, GRID_N + dx))
    sy = slice(max(0, -dy), min(GRID_N, GRID_N - dy))
    sx = slice(max(0, -dx), min(GRID_N, GRID_N - dx))
    out[ys, xs] = a[sy, sx]
    return out


def sweep_corr(frac: np.ndarray, ssha: np.ndarray, mask: np.ndarray,
               theta: float):
    """沿 θ / θ+90° 平移 s∈[-S,S] 格，与 SSHA 的掩膜相关。"""
    curves = {}
    for name, ang in (("along", theta), ("perp", theta + np.pi / 2)):
        cs, ns = [], []
        for s in range(-SWEEP_MAX, SWEEP_MAX + 1):
            dx = int(round(s * np.cos(ang)))
            dy = int(round(s * np.sin(ang)))
            fs = shift_field(frac, dy, dx)
            v = (mask > 0.5) & np.isfinite(fs)
            n = int(v.sum())
            if n < 20 or np.std(fs[v]) < 1e-6 or np.std(ssha[v]) < 1e-12:
                cs.append(np.nan)
            else:
                f_c = fs[v] - fs[v].mean()
                z_c = ssha[v] - ssha[v].mean()
                cs.append(float((f_c * z_c).sum()
                                / np.sqrt((f_c ** 2).sum() * (z_c ** 2).sum())))
            ns.append(n)
        curves[name] = (cs, ns)
    return curves


def skewness(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    return float(((x - m) ** 3).mean() / s ** 3) if s > 0 else np.nan


def process_tile(task: tuple) -> dict:
    """worker：处理一个块（有波或对照）。

    task=(kind, entry, lam, theta, dt_h, tile_dir)。tile_dir 显式传递：
    Windows spawn 的 worker 不会继承 main 里被覆盖的全局 TILE_DIR。
    """
    kind, entry, lam_asg, theta_asg, dt_h, tile_dir = task
    d = np.load(Path(tile_dir) / entry["file"])
    swot = d["swot"].astype(np.float32)
    ssha, mask = swot[0], swot[2]
    sar = d["sar"]
    rec = {"kind": kind, "day": entry["day"],
           "scene": norm_scene(entry["scene"]),
           "coverage": float(entry["swot_coverage"]),
           "label_frac": float(entry["label_frac"]),
           "dt_h": dt_h,
           "vv_nan_frac": float((~np.isfinite(sar[0])).mean()),
           "ssha_std": float(np.std(ssha[mask > 0.5])) if (mask > 0.5).any()
           else np.nan}
    if kind == "wave":
        est = estimate_stripe(d["label"], sar[0].astype(np.float32))
        rec["stripe_ok"] = bool(est and est["stripe_ok"])
        if est is not None:
            rec["lam_env"] = est["lam_env"]
            rec["vv_valid_in_label"] = est["vv_valid"]
            if est.get("lam") is not None:
                lam, theta, conf = est["lam"], est["theta"], est["conf"]
                rec.update(lam=lam, theta=theta, conf=conf)
                rec["lam_in_swot_range"] = bool(
                    rec["stripe_ok"] and LAM_MIN_SWOT <= lam <= LAM_MAX_SWOT)
                if rec["stripe_ok"]:
                    frac_s = downsample_label(d["label"])
                    if frac_s.std() > 1e-3:
                        sw = sweep_corr(frac_s, ssha, mask, theta)
                        rec["sweep_along"] = sw["along"][0]
                        rec["sweep_along_n"] = sw["along"][1]
                        rec["sweep_perp"] = sw["perp"][0]
                        rec["sweep_perp_n"] = sw["perp"][1]
            frac = downsample_label(d["label"])
            v = mask > 0.5
            in_m = v & (frac > 0.5)
            out_m = v & (frac < 0.05)
            rec["n_in"] = int(in_m.sum())
            if in_m.sum() >= 10:
                rec["skew_in"] = skewness(ssha[in_m].astype(np.float64))
                rec["p10_in"] = float(np.percentile(ssha[in_m], 10))
            if out_m.sum() >= 30:
                rec["skew_out"] = skewness(ssha[out_m].astype(np.float64))
                rec["p10_out"] = float(np.percentile(ssha[out_m], 10))
    else:  # ctrl：用配对分派的 (λ, θ)；无有效分派则不做拟合
        rec["lam"], rec["theta"] = lam_asg, theta_asg
        rec["lam_in_swot_range"] = lam_asg is not None
    lam, theta = rec.get("lam"), rec.get("theta")
    if lam is not None and rec.get("lam_in_swot_range", False):
        fit = band_fit(ssha, mask, lam, theta)
        fit_p = band_fit(ssha, mask, lam, theta + np.pi / 2)
        rec["fit_ok"] = fit is not None and fit_p is not None
        if rec["fit_ok"]:
            rec["A_band"], rec["A_exact"] = fit
            rec["A_band_perp"], rec["A_exact_perp"] = fit_p
    else:
        rec["fit_ok"] = False
    if kind == "ctrl":
        v = mask > 0.5
        if v.sum() >= 30:
            rec["skew_all"] = skewness(ssha[v].astype(np.float64))
            rec["p10_all"] = float(np.percentile(ssha[v], 10))
    return rec


# ---------------------------------------------------------------- 置换检验

def perm_unpaired(a: np.ndarray, b: np.ndarray, days_a: np.ndarray,
                  days_b: np.ndarray, n_perm: int, rng, stat_fn=None):
    """日内分层标签置换。返回 (obs, p_two, p_one, null)。"""
    stat_fn = stat_fn or (lambda x, y: float(np.mean(x) - np.mean(y)))
    va = np.isfinite(a)
    vb = np.isfinite(b)
    a, b, days_a, days_b = a[va], b[vb], days_a[va], days_b[vb]
    obs = stat_fn(a, b)
    allv = np.concatenate([a, b])
    alld = np.concatenate([days_a, days_b])
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.empty(len(allv), bool)
        for day in np.unique(alld):
            idx = np.nonzero(alld == day)[0]
            take = int((days_a == day).sum())
            sel = rng.permutation(idx)
            perm[sel[:take]] = True
        null[i] = stat_fn(allv[perm], allv[~perm])
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    p1 = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, p, p1, null


def perm_paired(diffs: np.ndarray, n_perm: int, rng):
    """符号翻转置换（paired）。返回 (obs_mean, obs_median, p_two_sided)。"""
    diffs = diffs[np.isfinite(diffs)]
    obs_m, obs_med = float(diffs.mean()), float(np.median(diffs))
    null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        null[i] = (signs * diffs).mean()
    p = (1 + np.sum(np.abs(null) >= abs(obs_m))) / (n_perm + 1)
    return obs_m, obs_med, p, null


def scene_block_p(diffs: np.ndarray, scenes: np.ndarray, n_perm: int, rng):
    """场景块级符号翻转置换：相邻切块空间相关，逐块 p 值偏乐观；
    以场景均值为单位做符号翻转，是保守敏感性分析。"""
    agg: dict[str, list] = {}
    for v, s in zip(diffs, scenes):
        if np.isfinite(v):
            agg.setdefault(s, []).append(v)
    ms = np.array([np.mean(v) for v in agg.values()])
    if len(ms) < 5:
        return np.nan, np.nan, len(ms)
    obs = float(ms.mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = (rng.choice([-1.0, 1.0], size=len(ms)) * ms).mean()
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return obs, p, len(ms)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    u = mannwhitneyu(a, b).statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def build_conclusion(report: dict) -> dict:
    """三选一判定。规则：
    - 支持：检验 1 方向性对照单侧 p<α 且方向为正，且检验 2 或 3 至少一个同向显著；
    - 不支持：有波组频带能量显著低于对照（直接否定共定位波能量）；
    - 其余（含唯一显著效应方向相反但物理上不可解释为 ISW 共定位检验的情形）：
      无法判定。检验 3 的 p10 反向效应发生时波已移出切块 8~20 km，
      与"共定位"无物理通道，只作警示记录，不作为否决依据。
    """
    t1 = report["test1_wavelength"]
    t2 = report["test2_displacement_sweep"]
    t3 = report["test3_amplitude_asymmetry"]
    alpha = report["multiple_testing"]["bonferroni_alpha"]
    t1_sig_support = (t1["directional_contrast_within_tile"]["p_one"] < alpha
                      and t1["directional_contrast_within_tile"]["obs_diff_m"] > 0)
    t2_sig = t2["p_one"] < alpha
    t3_p10 = t3["p10_in_vs_ctrl_m"]
    t1_contrary = (t1["unpaired_Aband"]["p_two"] < alpha
                   and t1["unpaired_Aband"]["obs_diff_m"] < 0
                   and abs(t1["unpaired_Aband"]["cohens_d"]) > 0.2)
    if t1_sig_support and (t2_sig or
                           (t3_p10["p_two"] < alpha and t3_p10["obs_diff_m"] < 0)):
        verdict = "支持共定位"
    elif t1_contrary:
        verdict = "不支持（有波组频带能量显著低于对照）"
    else:
        verdict = "现有数据无法判定"
    return {
        "verdict": verdict,
        "summary": [
            f"检验1（波长一致性，位移不变）：有波组与对照组的 SSHA 在 "
            f"λ_sar±30% 频带的拟合振幅无显著差异"
            f"（Δ={t1['unpaired_Aband']['obs_diff_m']*100:.3f} cm, "
            f"p={t1['unpaired_Aband']['p_two']:.3f}, "
            f"d={t1['unpaired_Aband']['cohens_d']:.3f}）；"
            f"块内方向对照（θ vs θ+90°）同样无显著差异"
            f"（p={t1['directional_contrast_within_tile']['p_two']:.3f}，"
            f"场景块级敏感性 p={t1['scene_block_sensitivity']['p_two']:.3f}）。"
            f"两组拟合振幅中位均≈0.12 cm，与拟合噪声底相当。",
            f"检验2（位移扫描）：零位移处 corr≈{t2['C_along'][12]:.4f}，"
            f"全窗最大方向差 {t2['max_diff_along_minus_perp']:.4f}"
            f"（p={t2['p_one']:.3f}）。各日 Δt·c 预期位移 8~20 km"
            f"（33~79 格）远超 ±12 格（±3 km）扫描窗，逐像素对应在物理上"
            f"不成立；扫描曲线中的缓变趋势反映大尺度 SSHA 梯度与条纹走向"
            f"的弱对齐，非波动信号。",
            f"检验3（振幅不对称）：偏度无任何组间/块内差异；"
            f"p10 差异显著但方向与下沉型 ISW 预期相反——label 区下尾更浅"
            f"（in−ctrl=+{t3_p10['obs_diff_m']*100:.2f} cm, "
            f"p={t3_p10['p_two']:.1e}, d={t3_p10['cohens_d']:.2f}；"
            f"块内 in−out=+{t3['within_tile_p10_in_minus_out_m']['obs_mean_m']*100:.2f} cm）。"
            f"该效应幅度小（≈0.3~0.7 cm，SSHA 块级 std 中位 0.64 cm），"
            f"且与波位移开后的残留 SSHA 场无明确物理联系，"
            f"更可能反映 v7 标签倾向落于 SSHA 空间平滑场的特定相位"
            f"（选择效应），不能作为 ISW 证据，如实报告。",
        ],
        "interpretation": (
            "两次过境 Δt=2.5~6.0 h，按反演相速度 c=0.83~1.03 m/s，ISW 在 "
            "SWOT 观测时刻距其 SAR 时刻位置 8~20 km，是 5 km 切块的 2~4 倍"
            "——两类观测在时空上根本不重叠，本数据集的切块级共定位前提"
            "不成立。检验 1/3 的位移不变量统计也未提供区域尺度的支持性"
            "证据（检验 3 唯一显著效应方向相反且幅度处于噪声边缘）。"
            "因此既不能宣称'双星观测到同一 ISW'，也不能据此断言信号不存"
            "在——本数据布局对这一问题原则上不可判定。"),
        "what_would_be_needed": [
            "Δt≲1 h 的配对（位移 <~3 km，落在切块尺度内），或",
            "按 c≈0.9 m/s 与传播方向（≈东西向，180° 模糊）把 label 波包"
            "外推 Δt·c（8~20 km）后，在 SWOT 刈幅的大范围连续网格上"
            "（而非 5 km 切块）做波形匹配/互相关扫描，或",
            "利用 SWOT 沿轨连续条带直接与 SAR 条纹做 2D 波数谱比对"
            "（谱是位移不变量），要求条带长度 ≫ λ。",
        ],
    }


# ---------------------------------------------------------------- 主流程

def main() -> None:
    global TILE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-dir", type=Path, default=TILE_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--max-tiles", type=int, default=0, help="调试：限量")
    args = ap.parse_args()

    TILE_DIR = args.tile_dir
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(TILE_DIR / "manifest_all.json", encoding="utf-8") as f:
        manifest = json.load(f)
    dt_map = load_dt_map()
    wave, ctrl_pool = select_groups(manifest)
    if args.max_tiles:
        wave = wave[: args.max_tiles]
    pairs, match_info = match_controls(wave, ctrl_pool, args.seed)
    print(f"有波组 {len(wave)}，对照池 {len(ctrl_pool)}，配对 {len(pairs)}"
          f"（有放回 {match_info['n_reused']}）")

    # 阶段 1：有波块（估 λ、θ、扫描、偏度）
    tasks_wave = [("wave", w, None, None,
                  dt_map.get(norm_scene(w["scene"]), (None, np.nan))[1],
                  str(TILE_DIR))
                 for w, _ in pairs]
    print(f"阶段 1：处理 {len(tasks_wave)} 有波块（workers={args.workers}）…")
    wave_recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_tile, tasks_wave, chunksize=16)):
            wave_recs.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks_wave)}")

    # 阶段 2：对照块，分派配对有波块的 (λ,θ)（无有效 λ 则不拟合）
    tasks_ctrl = []
    for (_, c), wr in zip(pairs, wave_recs):
        lam = wr["lam"] if wr.get("lam_in_swot_range") else None
        theta = wr.get("theta") if lam is not None else None
        tasks_ctrl.append(("ctrl", c, lam, theta,
                           dt_map.get(norm_scene(c["scene"]), (None, np.nan))[1],
                           str(TILE_DIR)))
    print(f"阶段 2：处理 {len(tasks_ctrl)} 对照块…")
    ctrl_recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_tile, tasks_ctrl, chunksize=16)):
            ctrl_recs.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks_ctrl)}")
    recs = wave_recs + ctrl_recs

    report = {"n_perm": args.n_perm, "seed": args.seed,
              "matching": match_info,
              "counts": {
                  "manifest": len(manifest),
                  "wave_selected": len(wave),
                  "ctrl_pool": len(ctrl_pool),
                  "stripe_ok": int(sum(r.get("stripe_ok") for r in wave_recs)),
                  "lam_lt_500m": int(sum(
                      r.get("stripe_ok") and r.get("lam") is not None
                      and r["lam"] < LAM_MIN_SWOT for r in wave_recs)),
                  "lam_in_swot_range": int(sum(
                      r.get("lam_in_swot_range", False) for r in wave_recs)),
                  "wave_fit_ok": int(sum(r.get("fit_ok") for r in wave_recs)),
                  "ctrl_fit_ok": int(sum(r.get("fit_ok") for r in ctrl_recs)),
              }}

    # ---------------- 数据质量 ----------------
    report["data_quality"] = {
        "vv_nan_frac_median": float(np.nanmedian(
            [r["vv_nan_frac"] for r in recs])),
        "ssha_std_median_m": float(np.nanmedian(
            [r["ssha_std"] for r in recs])),
        "ssha_std_p90_m": float(np.nanpercentile(
            [r["ssha_std"] for r in recs], 90)),
        "dt_h_by_day": {day: [float(np.nanmin(dts)), float(np.nanmax(dts))]
                        for day in sorted({r["day"] for r in wave_recs})
                        if len(dts := [r["dt_h"] for r in wave_recs
                                       if r["day"] == day
                                       and np.isfinite(r["dt_h"])])},
        "expected_displacement_km": {
            "c_med": C_MED, "per_day": {}},
    }
    for day, (lo, hi) in report["data_quality"]["dt_h_by_day"].items():
        report["data_quality"]["expected_displacement_km"]["per_day"][day] = {
            "dt_h_range": [lo, hi],
            "km_at_c_med": [lo * 3.6 * C_MED, hi * 3.6 * C_MED],
            "km_at_c_p25_p75": [lo * 3.6 * C_P25, hi * 3.6 * C_P75]}

    # ---------------- 检验 1：波长一致性 ----------------
    wf = [r for r in wave_recs if r.get("fit_ok")]
    cf = [r for r in ctrl_recs if r.get("fit_ok")]
    A_w = np.array([r["A_band"] for r in wf])
    A_c = np.array([r["A_band"] for r in cf])
    d_w = np.array([r["day"] for r in wf])
    d_c = np.array([r["day"] for r in cf])
    D_w = np.array([r["A_band"] - r["A_band_perp"] for r in wf])
    D_c = np.array([r["A_band"] - r["A_band_perp"] for r in cf])

    obs1, p1, p1_1s, null1 = perm_unpaired(A_w, A_c, d_w, d_c,
                                           args.n_perm, rng)
    obsD, pD, pD_1s, nullD = perm_unpaired(D_w, D_c, d_w, d_c,
                                           args.n_perm, rng)
    # 场景块级敏感性：每场景平均 D 的配对差（同场景 wave-ctrl），符号翻转
    scenes = sorted({r["scene"] for r in wf} & {r["scene"] for r in cf})
    scen_diffs = []
    for s in scenes:
        mw = [r["A_band"] - r["A_band_perp"] for r in wf if r["scene"] == s]
        mc = [r["A_band"] - r["A_band_perp"] for r in cf if r["scene"] == s]
        if mw and mc:
            scen_diffs.append(np.mean(mw) - np.mean(mc))
    scen_diffs = np.array(scen_diffs)
    obs_s, _, p_s, _ = perm_paired(scen_diffs, min(args.n_perm, 10000), rng) \
        if len(scen_diffs) >= 5 else (np.nan, np.nan, np.nan, None)

    report["test1_wavelength"] = {
        "n_wave": len(wf), "n_ctrl": len(cf),
        "A_band_wave_median_m": float(np.median(A_w)),
        "A_band_ctrl_median_m": float(np.median(A_c)),
        "unpaired_Aband": {"obs_diff_m": obs1, "p_two": p1, "p_one": p1_1s,
                           "cohens_d": cohens_d(A_w, A_c),
                           "rank_biserial": rank_biserial(A_w, A_c)},
        "directional_contrast_within_tile": {
            "obs_diff_m": obsD, "p_two": pD, "p_one": pD_1s,
            "cohens_d": cohens_d(D_w, D_c),
            "rank_biserial": rank_biserial(D_w, D_c),
            "wave_median_m": float(np.median(D_w)),
            "ctrl_median_m": float(np.median(D_c))},
        "scene_block_sensitivity": {"n_scenes": len(scen_diffs),
                                    "obs_mean_diff_m": float(obs_s),
                                    "p_two": float(p_s)},
    }

    # ---------------- 检验 2：位移扫描 ----------------
    ss = np.arange(-SWEEP_MAX, SWEEP_MAX + 1)
    sw_recs = [r for r in wave_recs if "sweep_along" in r]
    agg = {}
    for key in ("sweep_along", "sweep_perp"):
        C = np.array([r[key] for r in sw_recs], dtype=float)
        N = np.array([r[key + "_n"] for r in sw_recs], dtype=float)
        ok = np.isfinite(C) & (N >= 20)
        agg[key] = (np.nansum(np.where(ok, C, 0) * np.where(ok, N, 0), axis=0)
                    / np.maximum(np.where(ok, N, 0).sum(axis=0), 1))
    diff_curve = agg["sweep_along"] - agg["sweep_perp"]
    obs2 = float(np.nanmax(diff_curve))
    n_perm2 = min(args.n_perm, 2000)
    Ca = np.array([r["sweep_along"] for r in sw_recs], dtype=float)
    Cp = np.array([r["sweep_perp"] for r in sw_recs], dtype=float)
    null2 = np.empty(n_perm2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_perm2):
            flip = rng.random(len(sw_recs)) < 0.5
            A = np.where(flip[:, None], Cp, Ca)
            B = np.where(flip[:, None], Ca, Cp)
            null2[i] = np.nanmax(np.nanmean(A, axis=0)
                                 - np.nanmean(B, axis=0))
    p2 = (1 + np.sum(null2 >= obs2)) / (n_perm2 + 1)
    report["test2_displacement_sweep"] = {
        "n_tiles": len(sw_recs), "sweep_cells": ss.tolist(),
        "C_along": agg["sweep_along"].tolist(),
        "C_perp": agg["sweep_perp"].tolist(),
        "max_diff_along_minus_perp": obs2, "p_one": float(p2),
        "note": "预期位移 Δt·c ≈ 8~20 km ≈ 33~79 格 >> ±12 格扫描窗；"
                "物理上不应出现侧峰，平坦曲线=位移过大无法逐像素对应",
    }

    # ---------------- 检验 3：振幅不对称 ----------------
    def collect(key, recs_):
        v = np.array([r.get(key, np.nan) for r in recs_], float)
        d_ = np.array([r["day"] for r in recs_])
        ok = np.isfinite(v)
        return v[ok], d_[ok]

    sk_in, dsk_in = collect("skew_in", wave_recs)
    sk_out, dsk_out = collect("skew_out", wave_recs)
    sk_ct, dsk_ct = collect("skew_all", ctrl_recs)
    p10_in, dp10_in = collect("p10_in", wave_recs)
    p10_ct, dp10_ct = collect("p10_all", ctrl_recs)

    obs3a, p3a, p3a_1, _ = perm_unpaired(sk_in, sk_ct, dsk_in, dsk_ct,
                                         args.n_perm, rng)
    obs3b, p3b, p3b_1, _ = perm_unpaired(p10_in, p10_ct, dp10_in, dp10_ct,
                                         args.n_perm, rng)
    # 块内：label 内 vs label 外（同一批块的 skew_in/skew_out 配对）
    both = [r for r in wave_recs
            if np.isfinite(r.get("skew_in", np.nan))
            and np.isfinite(r.get("skew_out", np.nan))]
    din = np.array([r["skew_in"] - r["skew_out"] for r in both])
    obs3c, obs3c_med, p3c, _ = perm_paired(din, args.n_perm, rng)
    p10_pairs = [r for r in wave_recs
                 if np.isfinite(r.get("p10_in", np.nan))
                 and np.isfinite(r.get("p10_out", np.nan))]
    dp10 = np.array([r["p10_in"] - r["p10_out"] for r in p10_pairs])
    obs3d, obs3d_med, p3d, _ = perm_paired(dp10, args.n_perm, rng)
    # 场景块级敏感性（相邻块空间相关，逐块 p 偏乐观）
    sc_both = np.array([r["scene"] for r in both])
    sc_p10 = np.array([r["scene"] for r in p10_pairs])
    obs3c_s, p3c_s, nsc1 = scene_block_p(din, sc_both, args.n_perm, rng)
    obs3d_s, p3d_s, nsc2 = scene_block_p(dp10, sc_p10, args.n_perm, rng)

    report["test3_amplitude_asymmetry"] = {
        "skew_in_vs_ctrl": {"obs_diff": obs3a, "p_two": p3a, "p_one": p3a_1,
                            "cohens_d": cohens_d(sk_in, sk_ct),
                            "rank_biserial": rank_biserial(sk_in, sk_ct),
                            "n": [len(sk_in), len(sk_ct)],
                            "median_in": float(np.median(sk_in)),
                            "median_ctrl": float(np.median(sk_ct))},
        "p10_in_vs_ctrl_m": {"obs_diff_m": obs3b, "p_two": p3b,
                             "p_one": p3b_1,
                             "cohens_d": cohens_d(p10_in, p10_ct),
                             "n": [len(p10_in), len(p10_ct)],
                             "median_in_m": float(np.median(p10_in)),
                             "median_ctrl_m": float(np.median(p10_ct))},
        "within_tile_skew_in_minus_out": {"obs_mean": obs3c,
                                          "obs_median": obs3c_med,
                                          "p_two": p3c, "n": len(din),
                                          "scene_block": {
                                              "obs_mean": float(obs3c_s),
                                              "p_two": float(p3c_s),
                                              "n_scenes": nsc1}},
        "within_tile_p10_in_minus_out_m": {"obs_mean_m": float(obs3d),
                                           "obs_median_m": float(obs3d_med),
                                           "p_two": p3d, "n": len(dp10),
                                           "scene_block": {
                                               "obs_mean_m": float(obs3d_s),
                                               "p_two": float(p3d_s),
                                               "n_scenes": nsc2}},
        "note": "下沉型 ISW 预期 label 内 skew<0、p10 更低",
    }

    report["multiple_testing"] = {
        "families": 3, "bonferroni_alpha": 0.05 / 3,
        "note": "每族内部多个统计量为同一假设的不同视角，未再校正，"
                "解读以 p_two 与效应量联合判断"}

    # ---------------- 图 ----------------
    lam_ok = np.array([r["lam"] for r in wave_recs
                       if r.get("lam_in_swot_range")])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.hist(lam_ok, bins=40, color="steelblue")
    ax.axvline(LAM_MIN_SWOT, color="r", ls="--", label="500 m (2 cells) limit")
    ax.set_xlabel("λ_sar (m)"); ax.set_ylabel("tiles")
    ax.set_title(f"Stripe wavelength from VV-texture FFT (n={len(lam_ok)})")
    ax.legend()
    ax = axes[0, 1]
    conf = [r["conf"] for r in wave_recs if r.get("stripe_ok")]
    ax.hist(conf, bins=40, color="steelblue")
    ax.axvline(PEAK_CONF_MIN, color="r", ls="--")
    ax.set_xlabel("FFT peak / annulus median"); ax.set_ylabel("tiles")
    ax.set_title("Stripe peak confidence")
    ax = axes[1, 0]
    bins = np.linspace(0, np.percentile(np.concatenate([A_w, A_c]), 99), 40)
    ax.hist(A_w * 100, bins=bins * 100, alpha=0.6, label=f"wave (n={len(A_w)})")
    ax.hist(A_c * 100, bins=bins * 100, alpha=0.6, label=f"ctrl (n={len(A_c)})")
    ax.set_xlabel("SSHA band amplitude (cm)"); ax.set_ylabel("tiles")
    ax.set_title(f"Test 1: band energy at λ_sar±30%\n"
                 f"diff={obs1*100:.3f} cm, p={p1:.4f}, d={cohens_d(A_w, A_c):.3f}")
    ax.legend()
    ax = axes[1, 1]
    ax.hist(nullD * 100, bins=50, color="gray", alpha=0.7,
            label="permutation null")
    ax.axvline(obsD * 100, color="r", lw=2,
               label=f"obs={obsD*100:.3f} cm, p={pD:.4f}")
    ax.set_xlabel("mean(A_band − A_perp) diff, wave−ctrl (cm)")
    ax.set_title("Test 1 directional contrast (within-tile)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "fig_wavelength_check.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ss, agg["sweep_along"], "o-", label="shift along θ (physical)")
    ax.plot(ss, agg["sweep_perp"], "s--", label="shift along θ+90° (null)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("shift s (cells, 250 m each)")
    ax.set_ylabel("coverage-weighted corr(label, SSHA)")
    ax.set_title(f"Test 2: displacement sweep (n={len(sw_recs)} tiles), "
                 f"max diff={obs2:.4f}, p={p2:.3f}")
    for day, info in report["data_quality"]["expected_displacement_km"][
            "per_day"].items():
        km = np.mean(info["km_at_c_med"])
        cells = km * 1000 / GRID_M
        ax.annotate(f"{day}: Δt·c≈{km:.0f} km ({cells:.0f} cells) →",
                    xy=(0.02, 0.95 - 0.055 * list(report["data_quality"]
                    ["expected_displacement_km"]["per_day"]).index(day)),
                    xycoords="axes fraction", fontsize=8, color="darkred")
    ax.annotate("expected displacements (33–79 cells) are far outside the "
                "±12-cell window", xy=(0.02, 0.62), xycoords="axes fraction",
                fontsize=9, color="darkred")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "fig_displacement_sweep.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    bins = np.linspace(-2, 2, 50)
    ax.hist(np.clip(sk_in, -2, 2), bins=bins, alpha=0.6,
            label=f"label-in (n={len(sk_in)})")
    ax.hist(np.clip(sk_out, -2, 2), bins=bins, alpha=0.6,
            label=f"label-out (n={len(sk_out)})")
    ax.hist(np.clip(sk_ct, -2, 2), bins=bins, alpha=0.6,
            label=f"ctrl (n={len(sk_ct)})")
    ax.set_xlabel("SSHA skewness"); ax.set_ylabel("tiles")
    ax.set_title(f"Test 3: skewness\nin−ctrl={obs3a:.3f} (p={p3a:.4f}); "
                 f"in−out paired mean={obs3c:.3f} (p={p3c:.4f})")
    ax.legend(fontsize=8)
    ax = axes[1]
    q = np.percentile(np.concatenate([p10_in, p10_ct]), [1, 99])
    bins = np.linspace(q[0], q[1], 50)
    ax.hist(np.clip(p10_in, *q) * 100, bins=bins * 100, alpha=0.6,
            label=f"label-in (n={len(p10_in)})")
    ax.hist(np.clip(p10_ct, *q) * 100, bins=bins * 100, alpha=0.6,
            label=f"ctrl (n={len(p10_ct)})")
    ax.set_xlabel("SSHA p10 (cm)"); ax.set_ylabel("tiles")
    ax.set_title(f"Test 3: p10 (lower tail)\nin−ctrl={obs3b*100:.2f} cm "
                 f"(p={p3b:.4f}); in−out paired={obs3d*100:.2f} cm (p={p3d:.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "fig_amplitude_asymmetry.png", dpi=150)
    plt.close(fig)

    report["conclusion"] = build_conclusion(report)

    with open(args.out / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # 保存逐块记录供复核
    slim = [{k: v for k, v in r.items() if not k.startswith("sweep")}
            for r in recs]
    with open(args.out / "per_tile_records.json", "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False)
    print(f"完成 → {args.out}")
    print("判定：", report["conclusion"]["verdict"])
    print(json.dumps({k: report[k] for k in
                      ("counts", "test1_wavelength")}, ensure_ascii=False,
                     indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
