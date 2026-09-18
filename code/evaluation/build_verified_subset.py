# -*- coding: utf-8 -*-
"""事件数据库"已验证子集"构建：场景级剔除 + 岸距规则 + 几何有效性规则。

规则（阈值均用 234 条 v2db QA 抽样判读校准，详见 verified_subset_report.json）：
1. scene_fp     —— QA 判定抽样全 FP 的三景（178C / C19A / 957B），整景剔除；
2. nearshore    —— 距 Natural Earth 10m 岸线 ≤5 km 的事件（全档），
                   或 ≤20 km 且 quality != high（高档已过严格特征门，5-20 km
                   内 QA 确认的 TP 全为 high 档，非高档近岸检出绝大多数为 FP）；
3. blob_invalid —— 提取器失效"红线团块"的几何特征：
                   骨架总长/面积 > 2.0 km/km²（实心团块的骨架密度异常高），
                   或 n_crest > 100，或 area_km2 > 1000，或 wavelength_m > 2000
                   （超出内波物理合理范围）。

输入  results/event_database/events.csv（不修改）
      results/event_database/qa_sheets/qa_{tier}_v2db_records.json + *_judgment.json
      data/raw/auxiliary/naturalearth/ne_10m_land.shp
输出  results/event_database/events_verified.csv（全 2816 行 + keep_verified/drop_reason/dist_coast_km）
      results/event_database/verified_subset_report.json
用法（在 code/ 目录下）：python evaluation/build_verified_subset.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile  # pyshp
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "results/event_database/events.csv"
QA_DIR = PROJECT_ROOT / "results/event_database/qa_sheets"
SHP = PROJECT_ROOT / "data/raw/auxiliary/naturalearth/ne_10m_land.shp"
OUT_CSV = PROJECT_ROOT / "results/event_database/events_verified.csv"
OUT_JSON = PROJECT_ROOT / "results/event_database/verified_subset_report.json"
COAST_CACHE = PROJECT_ROOT / "results/event_database/coastline_points_scs.npy"

# ---- 规则参数（QA 校准后定值）----
FP_SCENE_CODES = ("178C", "C19A", "957B")   # QA 抽样全 FP 的场景
NEARSHORE_ALL_KM = 5.0                       # 全档剔除岸距
NEARSHORE_NONHIGH_KM = 20.0                  # 非 high 档剔除岸距
BLOB_LEN_PER_AREA = 2.0                      # 骨架总长/面积 (km/km²)
BLOB_N_CREST = 100
BLOB_AREA_KM2 = 1000.0
BLOB_WAVELENGTH_M = 2000.0

# 区域等距圆柱投影近似（南海范围够用，岸线点加密到 0.5 km 后误差 <0.3 km）
LAT0, LON0 = 20.0, 110.0
KX = 111.32 * np.cos(np.radians(LAT0))
KY = 110.57


def load_events() -> pd.DataFrame:
    df = pd.read_csv(DB)
    df["wavelength_m"] = pd.to_numeric(df["wavelength_m"], errors="coerce")
    return df


def load_qa(events: pd.DataFrame) -> pd.DataFrame:
    """合并三档 v2db QA 判读 -> event 级表（按 event_id+scene 关联事件库）。"""
    rows = []
    for tier in ["high", "medium", "low"]:
        recs = json.load(open(QA_DIR / f"qa_{tier}_v2db_records.json",
                              encoding="utf-8"))
        panels = []
        for jf in sorted(QA_DIR.glob(f"qa_{tier}_v2db_*_judgment.json")):
            panels.extend(json.load(open(jf, encoding="utf-8"))["panels"])
        assert len(recs) == len(panels), (tier, len(recs), len(panels))
        for rec, p in zip(recs, panels):
            rows.append({"event_id": rec["event_id"], "scene": rec["scene"],
                         "tier": tier, "judgment": p["judgment"],
                         "reason": p.get("reason", "")})
    qa = pd.DataFrame(rows)
    assert len(qa) == 234, len(qa)
    vc = qa.groupby("tier")["judgment"].value_counts().to_dict()
    expect = {("high", "TP"): 24, ("high", "UNCERTAIN"): 18, ("high", "FP"): 40,
              ("medium", "TP"): 12, ("medium", "UNCERTAIN"): 23,
              ("medium", "FP"): 48, ("low", "TP"): 5, ("low", "UNCERTAIN"): 8,
              ("low", "FP"): 56}
    assert vc == expect, vc
    # event_id 在事件库内不唯一（同日两景共享编号），必须用 event_id+scene 关联
    key = events["event_id"] + "|" + events["scene"]
    assert not key.duplicated().any()
    qa["key"] = qa["event_id"] + "|" + qa["scene"]
    ev = events.copy()
    ev["key"] = ev["event_id"] + "|" + ev["scene"]
    qa = qa.merge(ev.drop(columns=["event_id", "scene"]), on="key", how="left")
    assert len(qa) == 234 and qa["lon"].notna().all()
    return qa


def coastline_points(lon_range=(100, 123), lat_range=(10, 30), step_km=0.5):
    """Natural Earth 陆地多边形边界（含内环）在区域内的加密点云（km 投影）。"""
    if COAST_CACHE.exists():
        return np.load(COAST_CACHE)
    r = shapefile.Reader(str(SHP))
    pts = []
    for sr in r.iterShapeRecords():
        parts = list(sr.shape.parts) + [len(sr.shape.points)]
        p = sr.shape.points
        for i in range(len(parts) - 1):
            ring = p[parts[i]:parts[i + 1]]
            for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
                if (max(x1, x2) < lon_range[0] or min(x1, x2) > lon_range[1]
                        or max(y1, y2) < lat_range[0] or min(y1, y2) > lat_range[1]):
                    continue
                seg = np.hypot((x2 - x1) * KX, (y2 - y1) * KY)
                n = max(1, int(np.ceil(seg / step_km)))
                for t in np.linspace(0, 1, n + 1):
                    pts.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    pts = np.asarray(pts)
    xy = np.column_stack([(pts[:, 0] - LON0) * KX, (pts[:, 1] - LAT0) * KY])
    np.save(COAST_CACHE, xy)
    return xy


def add_coast_distance(df: pd.DataFrame, tree: cKDTree) -> pd.Series:
    q = np.column_stack([(df["lon"].values - LON0) * KX,
                         (df["lat"].values - LAT0) * KY])
    return pd.Series(tree.query(q)[0], index=df.index)


def fp_category(reason: str) -> str:
    if re.search(r"团块|失效|不贴合|满屏", reason):
        return "blob_extractor"
    if re.search(r"陆地|港口|海岸|陆海|岸线|近岸|岛|陆冰", reason):
        return "nearshore"
    if re.search(r"斑点|噪声|风浪|无条带|无波|风条纹", reason):
        return "noise"
    return "other"


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """返回带 keep_verified / drop_reason / 各规则布尔列的副本。"""
    df = df.copy()
    df["len_per_area"] = df["crest_length_km"] / df["area_km2"].replace(0, np.nan)
    r_scene = df["scene"].str.endswith(FP_SCENE_CODES)
    r_near = ((df["dist_coast_km"] <= NEARSHORE_ALL_KM)
              | ((df["dist_coast_km"] <= NEARSHORE_NONHIGH_KM)
                 & (df["quality"] != "high")))
    r_blob = ((df["len_per_area"] > BLOB_LEN_PER_AREA)
              | (df["n_crest"] > BLOB_N_CREST)
              | (df["area_km2"] > BLOB_AREA_KM2)
              | (df["wavelength_m"] > BLOB_WAVELENGTH_M)).fillna(False)
    df["rule_scene_fp"] = r_scene
    df["rule_nearshore"] = r_near
    df["rule_blob_invalid"] = r_blob
    df["drop_reason"] = ""
    df.loc[r_scene, "drop_reason"] = "scene_fp"
    df.loc[~r_scene & r_near, "drop_reason"] = "nearshore"
    df.loc[~r_scene & ~r_near & r_blob, "drop_reason"] = "blob_invalid"
    df["keep_verified"] = df["drop_reason"] == ""
    return df


def calibrate(qa: pd.DataFrame) -> dict:
    """在 234 条 QA 样本上评估每条规则及组合。"""
    out = {"n_qa": len(qa),
           "verdict_by_tier": {f"{t}/{j}": int(n) for (t, j), n in
                               qa.groupby("tier")["judgment"].value_counts().items()}}
    rule_cols = {"scene_fp": "rule_scene_fp", "nearshore": "rule_nearshore",
                 "blob_invalid": "rule_blob_invalid",
                 "combined": None}

    def stat(mask):
        res = {}
        for tier in ["high", "medium", "low", "all"]:
            sub = qa if tier == "all" else qa[qa["tier"] == tier]
            m = mask.loc[sub.index]
            r = {}
            for j in ["TP", "UNCERTAIN", "FP"]:
                n = int((sub["judgment"] == j).sum())
                hit = int((m & (sub["judgment"] == j)).sum())
                r[j] = {"n": n, "dropped": hit,
                        "rate": round(hit / n, 3) if n else None}
            res[tier] = r
        return res

    for name, col in rule_cols.items():
        mask = (qa["drop_reason"] != "") if col is None else qa[col]
        out[name] = stat(mask)
    # 残余 FP 构成（组合规则之后）
    fp = qa[qa["judgment"] == "FP"].copy()
    fp["cat"] = fp["reason"].map(fp_category)
    res_fp = fp[fp["drop_reason"] == ""]
    out["fp_category_total"] = fp["cat"].value_counts().to_dict()
    out["residual_fp_category"] = res_fp["cat"].value_counts().to_dict()
    out["residual_fp_by_tier"] = res_fp["tier"].value_counts().to_dict()
    return out


def weighted_precision(qa: pd.DataFrame, events: pd.DataFrame) -> dict:
    """按各档事件数加权估计全库精度（UNC 按 0.5 计入正确，另报严格口径）。"""
    tier_n = events["quality"].value_counts().to_dict()
    res = {}
    for label, mask in (("before", pd.Series(False, index=qa.index)),
                        ("after", qa["drop_reason"] != "")):
        tp_w = unc_w = n_w = 0.0
        for tier in ["high", "medium", "low"]:
            sub = qa[qa["tier"] == tier]
            keep = ~mask.loc[sub.index]
            n = tier_n.get(tier, 0)
            tp = ((sub["judgment"] == "TP") & keep).sum() / len(sub) * n
            unc = ((sub["judgment"] == "UNCERTAIN") & keep).sum() / len(sub) * n
            fp = ((sub["judgment"] == "FP") & keep).sum() / len(sub) * n
            tp_w += tp; unc_w += unc; n_w += tp + unc + fp
        res[label] = {
            "est_events": round(n_w),
            "est_tp": round(tp_w), "est_unc": round(unc_w),
            "est_fp": round(n_w - tp_w - unc_w),
            "precision_tp_only": round(tp_w / n_w, 3),
            "precision_tp_plus_half_unc": round((tp_w + 0.5 * unc_w) / n_w, 3),
        }
    return res


def main():
    events = load_events()
    print(f"事件库 {len(events)} 行（unique event_id {events['event_id'].nunique()}；"
          f"同日两景共享编号，以 event_id+scene 为键）")
    qa = load_qa(events)
    print(f"QA 判读 {len(qa)} 条合并完成，verdict 分布校验通过")

    tree = cKDTree(coastline_points())
    events["dist_coast_km"] = add_coast_distance(events, tree).round(2)
    qa["dist_coast_km"] = add_coast_distance(qa, tree)

    events = apply_rules(events)
    qa = apply_rules(qa)

    # ---- 输出 verified CSV（全 2816 行）----
    cols = [c for c in pd.read_csv(DB, nrows=0).columns]
    out = events[cols + ["dist_coast_km", "keep_verified", "drop_reason"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"→ {OUT_CSV}（保留 {int(events['keep_verified'].sum())} / {len(events)}）")

    # ---- 过滤前后对照 ----
    ev = events.copy()
    ev["month"] = ev["time_utc"].str[5:7]
    ev["cluster"] = np.where(ev["lon"] < 109.5, "west_hainan", "east_scs_slope")

    def counts(df):
        return {
            "total": int(len(df)),
            "by_quality": {k: int(v) for k, v in df["quality"].value_counts().items()},
            "by_month": {k: int(v) for k, v in
                         sorted(df["month"].value_counts().items())},
            "by_cluster": {k: int(v) for k, v in
                           df["cluster"].value_counts().items()},
        }

    before, after = counts(ev), counts(ev[ev["keep_verified"]])
    drop_by_reason = {k: int(v) for k, v in
                      ev.loc[ev["drop_reason"] != "", "drop_reason"]
                      .value_counts().items()}

    calib = calibrate(qa)
    wp = weighted_precision(qa, ev)
    wp["before"]["actual_events"] = int(len(ev))
    wp["after"]["actual_kept_events"] = int(ev["keep_verified"].sum())

    report = {
        "rules": {
            "scene_fp": {
                "definition": "QA 抽样判全 FP 的场景整景剔除",
                "scenes": list(FP_SCENE_CODES),
                "rationale": "scene_visibility_table.csv 中 has_wave=False 的三景；"
                             "QA 样本 11/11 全为 FP"},
            "nearshore": {
                "definition": f"dist_coast_km <= {NEARSHORE_ALL_KM}（全档）或 "
                              f"<= {NEARSHORE_NONHIGH_KM} 且 quality != high",
                "rationale": "QA 近岸类 FP 岸距中位 11 km（32 个中 26 个 ≤20 km）；"
                             "但 5-20 km 内 QA 确认的 4 个 TP 全为 high 档"
                             "（近岸真实波包存在），故 5-20 km 段只剔除非高档；"
                             "≤5 km 全档剔除。纯距离阈值扫描：5/10/20 km 的 "
                             "TP 保留率 95%/88%/85%，FP 去除率 10%/18%/32%，"
                             "纯距离方案在 ≥90% TP 保留下只能取 5 km，"
                             "分层方案在同样 TP 保留（95.1%）下 FP 去除达 25.7%"},
            "blob_invalid": {
                "definition": f"crest_length_km/area_km2 > {BLOB_LEN_PER_AREA} 或 "
                              f"n_crest > {BLOB_N_CREST} 或 area_km2 > {BLOB_AREA_KM2} "
                              f"或 wavelength_m > {BLOB_WAVELENGTH_M}",
                "rationale": "针对提取器失效的红线团块：实心团块骨架密度"
                             "（总长/面积）异常高；满屏失效事件面积/峰数极端；"
                             "λ>2000 m 超出南海内波物理合理范围。"
                             "阈值取 QA TP 零误伤/低误伤点"},
        },
        "qa_calibration": calib,
        "full_db_precision_estimate": wp,
        "counts_before": before,
        "counts_after": after,
        "dropped_by_reason": drop_by_reason,
        "notes": [
            "events.csv 的 event_id 不唯一（同日两景共享日期+编号），本脚本以 "
            "event_id+scene 为关联键；verified CSV 保留全部 2816 行原记录。",
            "岸距为到 Natural Earth 10m 陆地多边形边界（含岛屿、内环）的最近距离，"
            "岸线加密到 0.5 km 点距后 KDTree 查询，投影误差 <0.3 km。",
            "QA 为分层抽样，全库精度按各档事件数加权估计；残余 FP 以纯斑点噪声类"
            "为主，属规则够不着的部分，未强行过滤。",
            "重要口径说明：QA 抽样是按景均摊配额（每景抽样数≈n/景数，"
            "见 code/tests/make_event_qa_sheets.py），并非档内按事件随机——"
            "事件少的景被过度代表。因此 full_db_precision_estimate 是按"
            "档加权的近似值，规则命中率偏向小事件数景的特征（如近岸景），"
            "估计的去虚警幅度可能偏乐观；过滤前后实际事件数以 "
            "counts_before/counts_after 为准。",
        ],
    }
    json.dump(report, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"→ {OUT_JSON}")

    c = calib["combined"]["all"]
    print(f"\n组合规则 QA 校准（合计 234）：TP 保留 "
          f"{c['TP']['n'] - c['TP']['dropped']}/{c['TP']['n']}，"
          f"UNC 保留 {c['UNCERTAIN']['n'] - c['UNCERTAIN']['dropped']}/{c['UNCERTAIN']['n']}，"
          f"FP 去除 {c['FP']['dropped']}/{c['FP']['n']}")
    print(f"残余 FP 构成: {calib['residual_fp_category']}")
    print(f"全库精度估计（TP+0.5UNC 口径）: "
          f"{wp['before']['precision_tp_plus_half_unc']:.3f} → "
          f"{wp['after']['precision_tp_plus_half_unc']:.3f}")


if __name__ == "__main__":
    main()
