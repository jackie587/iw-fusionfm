"""条纹纹理自动判读：替代人工对场景检出打标（人机回环自动化）。

判据（在 505 个人工判读斑块上验证，AUC 0.967，分轮 0.94/0.99）：
VV 裁块去均值后 FFT 功率谱的细尺度能量比 fine_ratio（波长 <8px 能量
占全谱比例）——真内波条纹有锐利明暗交替（fine_ratio 中位 0.69），
虚警（油膜/低风暗斑/养殖区）纹理平滑（中位 0.27）。
保守双阈值：>0.65 判 positive，<0.40 判 negative，其间 uncertain
（交叉污染：正判负 ~1%，负判正 ~5%）。
ERA5 风速窗口（2~10 m/s）单独验证过，判别力弱（FP 仅 3/223 落在窗外），
只作元数据记录，不作判据；风控用作推理后处理（见 wind_filter.py）。

用法（在 code/ 目录下）：
    python tests/auto_review_wind.py --validate     # 人工斑块上复验判据
    python tests/auto_review_wind.py --scene-eval-dir ../results/scene_eval_v4negmix2 \
        --scenes A4EB 9E6B A6CD 8714                # 自动打标，出 auto_verdicts.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine
from rasterio.transform import xy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "tests"))

from make_review_sheets import build_vv_canvas  # noqa: E402
from build_negmix_dataset import crop_center  # noqa: E402

ERA5_DIR = PROJECT_ROOT / "data/raw/auxiliary/era5"

WINDOW = 512                 # 纹理分析裁块边长（与负样本库一致）
THR_POS, THR_NEG = 0.65, 0.40   # fine_ratio 双阈值（人工斑块 p95 负 / p1 正）
MIN_AREA = 20                # 连通域最小面积 (px)，与判读图口径一致


def fine_ratio(vv: np.ndarray) -> float:
    """细尺度（波长 <8px）FFT 能量占全谱比例：条纹锐利度的纹理指标。"""
    vv = vv.astype(np.float32) - float(np.mean(vv))
    P = np.abs(np.fft.rfft2(vv)) ** 2
    h, w = vv.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    return float(P[f > 1 / 8].sum() / max(P[(f > 0)].sum(), 1e-9))


def scene_transform(tiles_dir: Path) -> Affine:
    """任一切块 meta 的 transform 即全场景统一的像素→经纬度网格。"""
    metas = sorted((tiles_dir / "meta").glob("*.json"))
    m = json.loads(metas[0].read_text())
    return Affine(*m["transform"])


def era5_wind_at(lon: float, lat: float, tstamp: str) -> float:
    """场景时刻前后最近小时 ERA5 10m 风速，双线性空间插值。"""
    ts = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})", tstamp)
    nc = ERA5_DIR / f"era5_wind10m_{ts.group(1)}{ts.group(2)}.nc"
    ds = xr.open_dataset(nc)
    t = np.datetime64(f"{ts.group(1)}-{ts.group(2)}-{ts.group(3)}"
                      f"T{ts.group(4)}:{ts.group(5)}")
    ds = ds.sel(valid_time=t, method="nearest")
    u = float(ds["u10"].interp(longitude=lon, latitude=lat))
    v = float(ds["v10"].interp(longitude=lon, latitude=lat))
    ds.close()
    return float(np.hypot(u, v))


def coast_distance_map(land_mask: np.ndarray, transform: Affine) -> np.ndarray:
    """每个海面像素到最近陆地的距离 (km)。"""
    from scipy import ndimage
    lat0 = transform.f
    km_per_px_y = abs(transform.e) * 110.57
    km_per_px_x = abs(transform.a) * 111.32 * np.cos(np.radians(lat0))
    return ndimage.distance_transform_edt(
        land_mask == 0, sampling=(km_per_px_y, km_per_px_x))


def label_components(prob: np.ndarray, threshold: float, min_area: int):
    from scipy import ndimage
    lab, n = ndimage.label(prob > threshold)
    if n == 0:
        return []
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    comps = [{"cid": i + 1, "area": float(sizes[i])}
             for i in range(n) if sizes[i] >= min_area]
    cents = ndimage.center_of_mass(lab > 0, lab, [c["cid"] for c in comps])
    for c, (cy, cx) in zip(comps, cents):
        c["cy"], c["cx"] = float(cy), float(cx)
    return comps


def auto_verdict(fr: float) -> str:
    if fr > THR_POS:
        return "positive"
    if fr < THR_NEG:
        return "negative"
    return "uncertain"


def annotate_scene(scene_dir: Path, tiles_dir: Path, threshold: float):
    """对单场景全部连通域自动打标。"""
    prob_file = scene_dir / "prob_masked.npy"
    if not prob_file.exists():
        prob_file = scene_dir / "prob.npy"
    prob = np.nan_to_num(np.load(prob_file).astype(np.float32), nan=0.0)
    transform = scene_transform(tiles_dir)
    tstamp = re.search(r"\d{8}T\d{6}", scene_dir.name).group(0)

    land_file = scene_dir / "land_mask.npy"
    land = np.load(land_file) if land_file.exists() else np.zeros_like(prob)
    coast_km = coast_distance_map(land, transform)
    vv = build_vv_canvas(tiles_dir, prob.shape)

    comps = label_components(prob, threshold, MIN_AREA)
    out = []
    for c in comps:
        cy, cx = int(c["cy"]), int(c["cx"])
        patch = crop_center(vv, cy, cx, WINDOW, pad_value=None)
        if np.isnan(patch).any():
            patch = np.nan_to_num(patch, nan=0.0)
        fr = fine_ratio(patch)
        lon, lat = xy(transform, c["cy"], c["cx"])
        wind = era5_wind_at(lon, lat, tstamp)
        ck = float(coast_km[cy, cx])
        out.append({**c, "lon": round(lon, 5), "lat": round(lat, 5),
                    "wind_ms": round(wind, 2), "coast_km": round(ck, 1),
                    "fine_ratio": round(fr, 3), "verdict": auto_verdict(fr)})
    return out


def cmd_validate():
    """把自动规则应用到人工判读斑块（round1+round2），报告一致率。"""
    import collections
    lib = PROJECT_ROOT / "data/datasets/negative_samples"
    manifest = json.loads((lib / "manifest.json").read_text())
    rows = [{"label": m["label"],
             "auto": auto_verdict(fine_ratio(np.load(lib / m["file"])["vv"]))}
            for m in manifest]
    tab = collections.Counter((r["label"], r["auto"]) for r in rows)
    print("人工结论 × 自动判读 交叉表：")
    for (human, auto), n in sorted(tab.items()):
        print(f"  人工={human:10s} 自动={auto:10s} : {n}")
    tp = [r for r in rows if r["label"] == "positive"]
    fp = [r for r in rows if r["label"] == "negative"]
    print(f"\n真波被误判 negative: {sum(r['auto']=='negative' for r in tp)}/{len(tp)}")
    print(f"虚警被误判 positive: {sum(r['auto']=='positive' for r in fp)}/{len(fp)}")
    print(f"虚警自动捕获(negative): {sum(r['auto']=='negative' for r in fp)}/{len(fp)}")
    print(f"真波自动捕获(positive): {sum(r['auto']=='positive' for r in tp)}/{len(tp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--scene-eval-dir",
                    default=str(PROJECT_ROOT / "results/scene_eval_v4negmix2"))
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="场景名子串过滤（默认目录下全部）")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    if args.validate:
        cmd_validate()
        return

    root = Path(args.scene_eval_dir)
    for scene_dir in sorted(root.glob("S1A*")):
        if args.scenes and not any(s in scene_dir.name for s in args.scenes):
            continue
        tiles_dir = (PROJECT_ROOT / "data/processed/sar_tiles"
                     / (scene_dir.name + ".SAFE"))
        if not tiles_dir.exists():
            print(f"跳过 {scene_dir.name}：无切块目录")
            continue
        verdicts = annotate_scene(scene_dir, tiles_dir, args.threshold)
        out = {"threshold": args.threshold,
               "fine_ratio_thresholds": {"positive": THR_POS, "negative": THR_NEG},
               "components": verdicts}
        (scene_dir / "auto_verdicts.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2))
        n_pos = sum(v["verdict"] == "positive" for v in verdicts)
        n_neg = sum(v["verdict"] == "negative" for v in verdicts)
        print(f"{scene_dir.name[:52]}: 检出 {len(verdicts)}，"
              f"自动 positive {n_pos} / negative {n_neg} / "
              f"uncertain {len(verdicts) - n_pos - n_neg}")


if __name__ == "__main__":
    main()
