"""SWOT 信号对 真波/虚警 的判别力分析（弱融合价值前置检验）。

问题：v5 的 SAR 检出里，人工确认的真波（TP）与虚警（FP）在 SWOT
SSHA/|∇SSH| 上是否可分？若可分，SWOT 是物理上最硬的虚警过滤器
（内波在 SWOT 上有 2~12 cm 信号，斑点噪声/雨团没有）。

方法：取 round2 人工判读场景（07-01/07-06 三景，均有 SWOT 配对，
Δt 86~141 min）的检出斑块质心 → 经纬度 → 半径 R 内 SWOT 有效点的
|SSHA|/grad 统计（max、p90、mean、覆盖率），对比 TP/FP 分布与 AUC。

注意：Δt 内内波传播位移可达数 km（0.5~1 m/s 相速度），故 R 取 5 km
作粗对齐；严格配对需传播校正，此处只做大样本统计检验。

用法（在 code/ 目录下）：
    python tests/swot_tp_fp_analysis.py [--radius-km 5] [--qual-max 0]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.transform import xy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from data_preprocessing.swot_preprocess.quality_control import (
    load_and_qc, denoise_ssh)
from data_preprocessing.swot_preprocess.ssha import (
    compute_ssha, compute_ssh_gradient)

VERDICTS = PROJECT_ROOT / "results/scene_eval_v3negmix/human_review_far_round2.json"
SCENE_EVAL = PROJECT_ROOT / "results/scene_eval_v3negmix"
SWOT_DIR = PROJECT_ROOT / "data/raw/swot"
# 场景日期 → SWOT pass（pairs_census_scs_calval 的 Expert 配对）
PASS_OF_DATE = {"20230701": "569", "20230703": "571",
                "20230706": "574", "20230708": "576"}


def scene_transform(tiles_dir: Path) -> Affine:
    metas = sorted((tiles_dir / "meta").glob("*.json"))
    return Affine(*json.loads(metas[0].read_text())["transform"])


def load_swot(date: str, qual_max: int):
    granules = sorted(SWOT_DIR.glob(f"*Expert_{PASS_OF_DATE[date]}_*.nc"))
    assert granules, f"无 {date} 的 SWOT granule"
    qc = load_and_qc(str(granules[0]))
    if qual_max > 0:  # 放宽质控：suspect 级也算有效（覆盖率换噪声）
        import xarray as xr
        ds = xr.open_dataset(str(granules[0]))
        qual = ds["ssh_karin_qual"].values
        ssh = ds["ssh_karin"].values.astype(np.float64)
        qc["mask"] = ((qual <= qual_max) & np.isfinite(ssh)).astype(np.uint8)
        qc["ssh"] = np.where(qc["mask"] > 0, ssh, np.nan)
        ds.close()
    qc["ssh"] = denoise_ssh(qc["ssh"], qc["mask"])
    qc["ssha"] = compute_ssha(qc["ssh"])
    qc["grad"] = compute_ssh_gradient(qc["ssha"])
    return qc


def local_stats(lon0, lat0, qc, radius_km):
    """质心半径 R 内 SWOT 有效点的 |SSHA|/grad 统计。"""
    dlat = radius_km / 110.57
    dlon = radius_km / (111.32 * np.cos(np.radians(lat0)))
    near = ((np.abs(qc["lon"] - lon0) <= dlon)
            & (np.abs(qc["lat"] - lat0) <= dlat) & (qc["mask"] > 0))
    n = int(near.sum())
    if n < 20:
        return None
    a = np.abs(qc["ssha"][near])
    g = qc["grad"][near]
    a, g = a[np.isfinite(a)], g[np.isfinite(g)]
    if len(a) < 20:
        return None
    return {"n": n, "ssha_p90": float(np.percentile(a, 90)),
            "ssha_max": float(a.max()),
            "grad_p90": float(np.percentile(g, 90)),
            "grad_mean": float(g.mean())}


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--qual-max", type=int, default=0)
    args = ap.parse_args()

    verdicts = json.loads(VERDICTS.read_text())
    rows = []
    cache = {}
    for scene, v in verdicts.items():
        if scene == "pooled" or scene.startswith("_"):
            continue
        date = re.search(r"(\d{8})T", scene).group(1)
        if date not in PASS_OF_DATE:
            print(f"跳过 {scene[:40]}：无配对 pass")
            continue
        if date not in cache:
            cache[date] = load_swot(date, args.qual_max)
        qc = cache[date]
        tiles_dir = (PROJECT_ROOT / "data/processed/sar_tiles"
                     / (scene + ".SAFE"))
        t = scene_transform(tiles_dir)
        picked = json.loads(((SCENE_EVAL / scene) / "review/components.json")
                            .read_text())["picked"]
        for label, key in (("positive", "tp_ids"), ("negative", "fp_ids")):
            for sheet_no in v[key]:
                c = picked[sheet_no - 1]
                lon, lat = xy(t, c["cy"], c["cx"])
                st = local_stats(lon, lat, qc, args.radius_km)
                if st is None:
                    continue
                rows.append({"label": label, "scene": scene[-4:], **st})
        n_valid = sum(1 for r in rows if r["scene"] == scene[-4:])
        print(f"{scene[:44]}: 有 SWOT 覆盖的判读斑块 {n_valid} 个")

    out = PROJECT_ROOT / "results/swot_tp_fp_analysis.json"
    out.write_text(json.dumps({"radius_km": args.radius_km,
                               "qual_max": args.qual_max, "rows": rows},
                              ensure_ascii=False, indent=2))
    tp = [r for r in rows if r["label"] == "positive"]
    fp = [r for r in rows if r["label"] == "negative"]
    print(f"\n有效样本：TP {len(tp)} / FP {len(fp)}")
    if len(tp) > 10 and len(fp) > 10:
        y = [1] * len(tp) + [0] * len(fp)
        for k in ("ssha_p90", "ssha_max", "grad_p90", "grad_mean"):
            s_tp = [r[k] for r in tp]
            s_fp = [r[k] for r in fp]
            print(f"  {k:10s} TP中位 {np.median(s_tp):.4f}  "
                  f"FP中位 {np.median(s_fp):.4f}  AUC {auc(y, s_tp + s_fp):.3f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
