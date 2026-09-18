"""L2 配对样本构建演示：SAR 切块 × SWOT 面片的时空对齐与可视化。

流程（数据输入处理方案四/七）：
1. 读 SWOT granule → 质控 → SSHA/|∇SSH| 通道；
2. 读某景 SAR 切块的 meta（地理范围），筛出落在 SWOT 刈幅内的块；
3. 对每个覆盖块，把 SWOT 通道重投影到该块经纬度范围的 ~500 m 网格；
4. 拼接一张 SAR(VV) + SSHA 叠合图存 results/figures/ 供人工核查。

用法（在 code/ 目录下）：
    python data_preprocessing/matchup/pairing_demo.py \
        --s1-tiles <preprocess_safe 输出的场景目录> \
        --swot <granule .nc 路径> [--n 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

import numpy as np

from data_preprocessing.swot_preprocess.quality_control import (
    load_and_qc, denoise_ssh)
from data_preprocessing.swot_preprocess.ssha import (
    make_swot_channels, compute_ssha, compute_ssh_gradient)
from data_preprocessing.swot_preprocess.reproject import reproject_to_grid
from utils.logger import get_logger

logger = get_logger("pairing_demo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-tiles", required=True, help="场景切块目录（images/ meta/）")
    ap.add_argument("--swot", required=True, help="SWOT granule .nc")
    ap.add_argument("--n", type=int, default=3, help="抽查块数")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/figures"))
    args = ap.parse_args()

    # 1. SWOT 通道
    qc = load_and_qc(args.swot)
    qc["ssh"] = denoise_ssh(qc["ssh"], qc["mask"])
    ssha = compute_ssha(qc["ssh"])
    grad = compute_ssh_gradient(ssha)
    ch = make_swot_channels(qc)
    ch["ssha"], ch["grad"] = ssha, grad

    sw_lon = ch["lon"][ch["mask"] > 0]
    sw_lat = ch["lat"][ch["mask"] > 0]
    logger.info("SWOT 刈幅有效范围：lon %.2f~%.2f, lat %.2f~%.2f",
                np.nanmin(sw_lon), np.nanmax(sw_lon),
                np.nanmin(sw_lat), np.nanmax(sw_lat))

    # 2. 逐块统计落在块内的 SWOT 有效点数，按密度排序取 top-n
    #    （刈幅与 SAR 场景往往只部分重叠，均匀抽查会采到零覆盖块）
    meta_dir = Path(args.s1_tiles) / "meta"
    metas = []
    for mp in sorted(meta_dir.glob("*.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        a, _, lon0, _, b, lat0 = m["transform"]
        ts = m["tile_size"]
        lon1, lon2 = sorted([lon0, lon0 + a * ts])
        lat1, lat2 = sorted([lat0, lat0 + b * ts])
        m.update(lon_min=lon1, lon_max=lon2, lat_min=lat1, lat_max=lat2)
        inside = ((ch["lon"] >= lon1) & (ch["lon"] <= lon2)
                  & (ch["lat"] >= lat1) & (ch["lat"] <= lat2)
                  & (ch["mask"] > 0))
        n_pts = int(inside.sum())
        if n_pts > 0:
            metas.append((mp.stem, m, n_pts))
    metas.sort(key=lambda t: -t[2])
    logger.info("切块总数 %d，有 SWOT 有效点覆盖 %d",
                len(list(meta_dir.glob('*.json'))), len(metas))
    if not metas:
        logger.error("无重叠切块，配对为空")
        return

    # 3. 覆盖率最高的 n 块做重投影 + 可视化
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, m, n_pts in metas[:args.n]:
        tile = np.load(Path(args.s1_tiles) / "images" / f"{stem}.npy")
        rep = reproject_to_grid(ch, m["lon_min"], m["lat_min"],
                                m["lon_max"], m["lat_max"])
        cov = rep["mask"].mean()
        logger.info("  %s  有效点 %d  SWOT 覆盖率 %.1f%%", stem[-24:],
                    n_pts, cov * 100)

        fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
        ax[0].imshow(tile[0], cmap="gray", vmin=0, vmax=1)
        ax[0].set_title("VV dB (norm)")
        vv = np.clip(tile[0], 0, 1)
        ssha_n = np.ma.masked_where(rep["mask"] == 0, rep["ssha"])
        ax[1].imshow(ssha_n, cmap="RdBu_r")
        ax[1].set_title(f"SWOT SSHA (coverage {cov*100:.0f}%)")
        ax[2].imshow(vv, cmap="gray", vmin=0, vmax=1)
        ax[2].imshow(np.kron(np.isfinite(ssha_n).astype(float),
                             np.ones((8, 8))), cmap="autumn", alpha=0.35,
                     vmin=0, vmax=1)
        ax[2].set_title("VV + SWOT swath")
        fig.suptitle(stem, fontsize=8)
        fig.savefig(out_dir / f"pair_{stem}.png", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info("  已存 %s", out_dir / f"pair_{stem}.png")


if __name__ == "__main__":
    main()
