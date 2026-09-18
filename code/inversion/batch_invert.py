"""事件数据库 KdV/eKdV 批量反演：几何观测 → 振幅/相速度（带区间）。

链路：events.csv（quality 过滤，默认 high）→ SRTM15+ 水深采样
→ 季节分层 3×3 敏感性网格（scenarios.py）→ physics.py kdv_invert /
ekdv_invert 逐组合反演 → 跨组合聚合 min/median/max（区间而非单点值）。

假设与口径（全部写入输出 JSON meta）：
- wavelength_m 为相邻波峰间距中位数，按 l = λ/κ、κ=π 折算孤子半宽
  （physics.py docstring；κ 敏感性解析：κ=2 → η0×0.41，κ=2π → η0×4.0）；
- 下凹波（h1<h2 门控保证）；eKdV 双解取小振幅根、大振幅根记录在案，
  no_root 回退 KdV；
- η0 > h1 标记 djl_needed（KdV/eKdV 超域，方案文档对应 DJL 分区），
  数值仍给出但仅供参考；
- 分层为文献气候态包络（无实测温盐），结果是敏感性区间，不是真值。

产物（results/event_database/）：
  inversion_results.csv   每事件一行（中位数 + 区间 + 标记）
  inversion_results.json  meta（假设/数据源）+ 全量记录

用法（在 code/ 目录下）：
    python inversion/batch_invert.py [--quality high,medium]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics
import scenarios as S

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENTS_CSV = PROJECT_ROOT / "results/event_database/events.csv"
OUT_DIR = PROJECT_ROOT / "results/event_database"


def invert_event(lam: float, depth: float, month: int) -> dict | None:
    """单事件敏感性网格反演。返回聚合结果或 None（全组合无效）。"""
    if depth < S.H_MIN_M:
        return None
    rows = []
    for sc in S.scenario_grid(month):
        h1, h2 = sc["h1"], depth - sc["h1"]
        if h2 < S.H2_MIN_M or h2 < S.CRIT_RATIO * h1:
            continue
        cf = physics.two_layer_coeffs(h1, h2, sc["drho"])
        kd = physics.kdv_invert(lam, cf["c0"], cf["alpha"], cf["beta"])
        ek = physics.ekdv_invert(lam, cf["c0"], cf["alpha"], cf["alpha1"],
                                 cf["beta"])
        rows.append({"h1": h1, "drho": sc["drho"], "c0": cf["c0"],
                     "eta0_kdv": kd["eta0_m"], "c_kdv": kd["c_ms"],
                     "eta0_ekdv": ek["eta0_m"], "eta0_ekdv_alt": ek["eta0_alt_m"],
                     "c_ekdv": ek["c_ms"], "ekdv_flag": ek["flag"],
                     "a_lim": abs(cf["alpha"] / cf["alpha1"])})
    if not rows:
        return None

    def agg(key):
        v = [r[key] for r in rows if r[key] is not None]
        if not v:
            return {"med": None, "lo": None, "hi": None}
        return {"med": statistics.median(v), "lo": min(v), "hi": max(v)}

    eta_k = agg("eta0_kdv")
    h1_med = statistics.median(r["h1"] for r in rows)
    n_ekdv_ok = sum(r["ekdv_flag"] in ("ok", "dual_root") for r in rows)
    # regime 判定（优先级自上而下；数值照常输出，标记提示超域）
    if lam < S.SHORT_WAVE * depth:
        regime = "short_wave"    # λ < 3H，长波假设破缺（方案文档对应 BO/DJL 区）
    elif eta_k["med"] is not None and eta_k["med"] > h1_med:
        regime = "djl_needed"    # η0 > h1，KdV/eKdV 超域
    else:
        regime = "kdv_ok"
    return {
        "n_scen": len(rows),
        "h1_med_m": h1_med,
        "c0_ms": agg("c0"),
        "eta0_kdv_m": eta_k,
        "c_kdv_ms": agg("c_kdv"),
        "eta0_ekdv_m": agg("eta0_ekdv"),
        "c_ekdv_ms": agg("c_ekdv"),
        "ekdv_valid": n_ekdv_ok,   # 9 组合中 eKdV 有解的个数，其余回退 KdV
        "regime": regime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="high",
                    help="逗号分隔的质量级，默认 high")
    args = ap.parse_args()
    quals = set(args.quality.split(","))

    rows = list(csv.DictReader(open(EVENTS_CSV, encoding="utf-8")))
    sel = [r for r in rows
           if r["quality"] in quals and r["wavelength_m"] != ""]
    print(f"反演事件：{len(sel)}（quality ∈ {sorted(quals)}）", flush=True)

    bathy = S.Bathymetry()
    out, skipped = [], {"shallow": 0, "land": 0}
    for r in sel:
        lon, lat = float(r["lon"]), float(r["lat"])
        depth = bathy.depth(lon, lat)
        month = int(r["time_utc"][5:7])
        if depth <= 0:
            skipped["land"] += 1
            continue
        res = invert_event(float(r["wavelength_m"]), depth, month)
        if res is None:
            skipped["shallow"] += 1
            continue
        rec = {
            "event_id": r["event_id"], "scene": r["scene"],
            "time_utc": r["time_utc"], "lon": lon, "lat": lat,
            "direction_deg": float(r["direction_deg"]),
            "wavelength_m": float(r["wavelength_m"]),
            "quality": r["quality"],
            "depth_m": round(depth, 1),
            "season": S.season_of(month),
            "polarity": "depression",
            "regime": res["regime"],
            "n_scen": res["n_scen"], "ekdv_valid": res["ekdv_valid"],
            "c0_ms_med": round(res["c0_ms"]["med"], 3),
            "eta0_kdv_m_med": round(res["eta0_kdv_m"]["med"], 1),
            "eta0_kdv_m_lo": round(res["eta0_kdv_m"]["lo"], 1),
            "eta0_kdv_m_hi": round(res["eta0_kdv_m"]["hi"], 1),
            "c_kdv_ms_med": round(res["c_kdv_ms"]["med"], 3),
            "c_kdv_ms_lo": round(res["c_kdv_ms"]["lo"], 3),
            "c_kdv_ms_hi": round(res["c_kdv_ms"]["hi"], 3),
            "eta0_ekdv_m_med": (round(res["eta0_ekdv_m"]["med"], 1)
                                if res["eta0_ekdv_m"]["med"] else ""),
            "eta0_ekdv_m_lo": (round(res["eta0_ekdv_m"]["lo"], 1)
                               if res["eta0_ekdv_m"]["lo"] else ""),
            "eta0_ekdv_m_hi": (round(res["eta0_ekdv_m"]["hi"], 1)
                               if res["eta0_ekdv_m"]["hi"] else ""),
            "c_ekdv_ms_med": (round(res["c_ekdv_ms"]["med"], 3)
                              if res["c_ekdv_ms"]["med"] else ""),
        }
        out.append(rec)

    cols = list(out[0].keys())
    with open(OUT_DIR / "inversion_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    meta = {
        "method": "两层流体 KdV/eKdV（physics.py）+ 季节分层 3×3 敏感性网格",
        "quality_filter": sorted(quals),
        "kappa": {"value": physics.KAPPA_DEFAULT,
                  "sensitivity": "η0 ∝ κ²：κ=2 → ×0.41，κ=2π → ×4.0"},
        "bathymetry": S.BATHY_SRC,
        "stratification": {k: {"h1_m": v["h1"], "drho": v["drho"]}
                           for k, v in S.SEASONAL_SCENARIOS.items()},
        "stratification_source": ("WOA23 南海北部气候态 + Ramp et al. 2010 / "
                                  "Cai et al. 2012 / Jia et al. 2019 个例包络；"
                                  "无实测温盐，结果为区间非真值"),
        "validity_gates": {"H_min_m": S.H_MIN_M, "h2_min_m": S.H2_MIN_M,
                           "h2/h1_min": S.CRIT_RATIO,
                           "short_wave": "λ < 3H 标记 short_wave（长波假设破缺）",
                           "djl_needed": "η0 > h1 标记（KdV/eKdV 超域）"},
        "n_events": len(out), "skipped": skipped,
        "regime_counts": {rg: sum(r["regime"] == rg for r in out)
                          for rg in sorted({r["regime"] for r in out})},
    }
    json.dump({"meta": meta, "events": out},
              open(OUT_DIR / "inversion_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 控制台自检摘要
    import numpy as np
    eta = np.array([r["eta0_kdv_m_med"] for r in out])
    cc = np.array([r["c_kdv_ms_med"] for r in out])
    dep = np.array([r["depth_m"] for r in out])
    print(f"完成 {len(out)} 事件（跳过 {skipped}）")
    print(f"水深   中位 {np.median(dep):.0f} m，P5~P95 "
          f"{np.percentile(dep, 5):.0f}~{np.percentile(dep, 95):.0f} m")
    print(f"η0_kdv 中位 {np.median(eta):.1f} m，P5~P95 "
          f"{np.percentile(eta, 5):.1f}~{np.percentile(eta, 95):.1f} m")
    print(f"c_kdv  中位 {np.median(cc):.2f} m/s，P5~P95 "
          f"{np.percentile(cc, 5):.2f}~{np.percentile(cc, 95):.2f} m/s")
    print(f"regime: {meta['regime_counts']}")
    print(f"→ {OUT_DIR}/inversion_results.{{csv,json}}")


if __name__ == "__main__":
    main()
