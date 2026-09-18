"""L2 配对数据集构建：SAR 切块 × SWOT 通道 × v5 弱标签。

流程（数据输入处理方案四/七）：
1. SWOT granule → 质控 → 降噪 → SSHA/|∇SSH|；
2. 逐 SAR 切块统计 SWOT 有效点覆盖，保留 ≥MIN_PTS 的块；
3. 每块：SWOT 重投影到块范围 500 m 网格（不插值造细节），
   SAR 张量原样，v5 场景概率图裁出弱标签（prob>thr）；
4. 写 data/datasets/L2_s1_swot_matched/tiles/<场景>__<块名>.npz + manifest.json。

用法（在 code/ 目录下）：
    python data_preprocessing/matchup/build_l2_dataset.py \
        --s1-tiles ../data/processed/sar_tiles/<场景>.SAFE \
        --swot ../data/raw/swot/<granule>.nc \
        --prob ../results/scene_eval_v5negmix3/<场景>/prob_masked.npy
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from data_preprocessing.swot_preprocess.quality_control import (
    load_and_qc, load_unsmoothed_half, denoise_ssh)
from data_preprocessing.swot_preprocess.ssha import (
    compute_ssha, compute_ssh_gradient)
from data_preprocessing.swot_preprocess.reproject import reproject_to_grid
from utils.logger import get_logger

logger = get_logger("build_l2")

MIN_PTS = 50          # 块内 SWOT 有效点下限（500m 网格 ~20×20=400 点满覆盖）
LABEL_THR = 0.6
OUT = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"


def load_swot_channels(nc_path: str, bbox: tuple | None = None) -> dict:
    """按产品布局加载 SWOT 通道（SSHA/|∇SSHA|/mask）。

    - Expert/Basic（2km 单组网格）：官方 ssha_karin 出发
      （2026-08-31 修正：ssh_karin 含 geoid，跨轨梯度沿轨滤波去不掉，
      曾致 |SSHA|p90 虚高至 50~70cm；官方 SSHA+降噪+沿轨去趋势后
      |p90|≈5.5cm，落在 ISW 量级 2~12cm）；
    - Unsmoothed（250m，left/right 两半刈幅）：ssh_karin_2 − MSS，
      半边分别降噪去趋势后拼接（重投影按散点处理，nadir gap 天然
      mask=0，无假邻接）；
    - bbox=(lon1,lat1,lon2,lat2) 时 Unsmoothed 半边先裁网格再处理，
      避免 20M 像素全图降噪。
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    is_flat = "ssha_karin" in ds
    ds.close()
    if is_flat:
        qc = load_and_qc(nc_path, ssh_var="ssha_karin",
                         qual_var="ssha_karin_qual")
        qc["ssh"] = denoise_ssh(qc["ssh"], qc["mask"])
        ssha = compute_ssha(qc["ssh"])
        return {"lon": qc["lon"], "lat": qc["lat"], "ssha": ssha,
                "grad": compute_ssh_gradient(ssha, res_m=2000.0),
                "mask": qc["mask"]}
    halves = []
    for side in ("left", "right"):
        qc = load_unsmoothed_half(nc_path, side, bbox=bbox)
        if qc is None:
            continue
        qc["ssh"] = denoise_ssh(qc["ssh"], qc["mask"])
        ssha = compute_ssha(qc["ssh"])
        halves.append({"lon": qc["lon"], "lat": qc["lat"], "ssha": ssha,
                       "grad": compute_ssh_gradient(ssha, res_m=250.0),
                       "mask": qc["mask"]})
    if not halves:
        raise ValueError(f"无法识别的 SWOT 产品布局或刈幅不覆盖 bbox: {nc_path}")
    return {k: np.concatenate([h[k].ravel() for h in halves]).reshape(-1, 1)
            for k in ("lon", "lat", "ssha", "grad", "mask")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-tiles", required=True)
    ap.add_argument("--swot", required=True)
    ap.add_argument("--prob", default=None, help="场景概率图（弱标签，可选）")
    ap.add_argument("--label-thr", type=float, default=LABEL_THR)
    ap.add_argument("--res", type=float, default=250.0,
                    help="SWOT 重投影网格分辨率（m）；250 可分辨 λ≥500m 波形")
    args = ap.parse_args()

    tiles_dir = Path(args.s1_tiles)
    scene = tiles_dir.name.replace(".SAFE", "")

    # 场景 lon/lat 包围盒（Unsmoothed 250m 按此预裁，省全图降噪）
    lons, lats = [], []
    for mp in (tiles_dir / "meta").glob("*.json"):
        m = json.loads(mp.read_text(encoding="utf-8"))
        a, _, lon0, _, b, lat0 = m["transform"]
        size = m["tile_size"]
        lons += [lon0, lon0 + a * size]
        lats += [lat0, lat0 + b * size]
    scene_bbox = (min(lons), min(lats), max(lons), max(lats))

    # 按产品布局自适应加载（2026-08-31：Expert 2km / Unsmoothed 250m）
    ch = load_swot_channels(args.swot, bbox=scene_bbox)

    prob = None
    if args.prob:
        prob = np.nan_to_num(
            np.load(args.prob).astype(np.float32), nan=0.0)

    manifest = []
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"_y(\d+)_x(\d+)\.json$")
    # 有效点预压平 + 每块先取块域点子集再建 KDTree（全量点逐块建树太慢）
    ch_flat = {k: v.ravel() for k, v in ch.items()}
    valid_idx = np.nonzero(ch_flat["mask"] > 0)[0]
    ch_v = {k: v[valid_idx] for k, v in ch_flat.items()}
    for mp in sorted((tiles_dir / "meta").glob("*.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        a, _, lon0, _, b, lat0 = m["transform"]
        ts = m["tile_size"]
        lon1, lon2 = sorted([lon0, lon0 + a * ts])
        lat1, lat2 = sorted([lat0, lat0 + b * ts])
        sel = ((ch_v["lon"] >= lon1) & (ch_v["lon"] <= lon2)
               & (ch_v["lat"] >= lat1) & (ch_v["lat"] <= lat2))
        if int(sel.sum()) < MIN_PTS:
            continue
        sub = {k: v[sel].reshape(-1, 1) for k, v in ch_v.items()}
        sub = {**sub, "mask": np.ones_like(sub["ssha"])}
        rep = reproject_to_grid(sub, lon1, lat1, lon2, lat2,
                                res_m=args.res)
        cov = float(rep["mask"].mean())
        sar = np.load(tiles_dir / "images" / f"{mp.stem}.npy")
        # 刈幅外/全 NaN 切块不入库（SAR 无效则配对无意义）
        if not np.isfinite(sar).any() or (~np.isfinite(sar)).mean() > 0.5:
            continue
        label = np.zeros((ts, ts), np.uint8)
        if prob is not None:
            mo = pat.search(mp.name)
            y, x = int(mo.group(1)), int(mo.group(2))
            label = (prob[y:y + ts, x:x + ts] > args.label_thr).astype(np.uint8)
        name = f"{scene[-4:]}_{mp.stem[-24:]}"
        np.savez_compressed(
            out_dir / f"{name}.npz",
            sar=sar.astype(np.float16),
            swot=np.stack([rep["ssha"], rep["grad"],
                           rep["mask"].astype(np.float32)]).astype(np.float32),
            label=label,
            coverage=np.float32(cov))
        manifest.append({"file": f"{name}.npz", "scene": scene,
                         "tile": mp.stem, "swot_coverage": round(cov, 3),
                         "label_frac": float(label.mean())})
    (out_dir / f"manifest_{scene[-4:]}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info("%s: 配对块 %d（覆盖率中位 %.2f，含标签块 %d）→ %s",
                scene[:44], len(manifest),
                float(np.median([m["swot_coverage"] for m in manifest]))
                if manifest else 0.0,
                sum(m["label_frac"] > 0 for m in manifest), out_dir)


if __name__ == "__main__":
    main()
