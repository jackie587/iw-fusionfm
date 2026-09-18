"""09-30 配对日 SAR×SWOT 共定位分析：v5 检出下的 SWOT 信号 vs 背景。

产出：
- results/figures/swot_coloc_*.png：每个共定位检出的 SAR(VV)×SSHA×|∇SSH| 三联图；
- results/swot_coloc_0930.json：检出区与随机背景区的 SSHA 统计对比。
用法（在 code/ 目录下）：python tests/swot_coloc_0930.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine
from rasterio.transform import xy
from scipy import ndimage
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from data_preprocessing.swot_preprocess.quality_control import (
    load_and_qc, denoise_ssh)
from data_preprocessing.swot_preprocess.ssha import (
    compute_ssha, compute_ssh_gradient)
from make_review_sheets import build_vv_canvas  # noqa: E402
sys.path.insert(0, str(CODE_ROOT / "tests"))

SCENE = "S1A_IW_GRDH_1SDV_20230930T104133_20230930T104202_050556_0616E8_7E00"
GRANULE = "SWOT_L2_LR_SSH_Expert_004_243_20230930T111959_20230930T121040_PGC0_01.nc"
RADIUS_KM = 2.5


def main():
    qc = load_and_qc(str(PROJECT_ROOT / "data/raw/swot" / GRANULE))
    qc["ssh"] = denoise_ssh(qc["ssh"], qc["mask"])
    qc["ssha"] = compute_ssha(qc["ssh"])
    qc["grad"] = compute_ssh_gradient(qc["ssha"])
    m = qc["mask"] > 0
    pts = np.column_stack([qc["lon"][m], qc["lat"][m]])
    tree = cKDTree(pts)

    sd = PROJECT_ROOT / "results/scene_eval_v5negmix3" / SCENE
    prob = np.nan_to_num(np.load(sd / "prob_masked.npy").astype(np.float32))
    lab, n = ndimage.label(prob > 0.6)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    cids = [i + 1 for i in range(n) if sizes[i] >= 20]
    cents = ndimage.center_of_mass(lab > 0, lab, cids)
    meta = json.loads(sorted(
        (PROJECT_ROOT / f"data/processed/sar_tiles/{SCENE}.SAFE/meta")
        .glob("*.json"))[0].read_text())
    t = Affine(*meta["transform"])

    # 共定位检出
    hits = []
    for c, (cy, cx) in zip(cids, cents):
        lo, la = xy(t, cy, cx)
        d, _ = tree.query([lo, la])
        if d * 111.32 * np.cos(np.radians(la)) <= RADIUS_KM:
            hits.append({"cid": int(c), "cy": float(cy), "cx": float(cx),
                         "lon": lo, "lat": la, "area": float(sizes[c - 1])})
    print(f"共定位检出 {len(hits)} 个")

    def swot_stats(lo, la):
        dlat = RADIUS_KM / 110.57
        dlon = RADIUS_KM / (111.32 * np.cos(np.radians(la)))
        near = ((np.abs(qc["lon"] - lo) <= dlon)
                & (np.abs(qc["lat"] - la) <= dlat) & (qc["mask"] > 0))
        a = np.abs(qc["ssha"][near])
        g = qc["grad"][near]
        a, g = a[np.isfinite(a)], g[np.isfinite(g)]
        if len(a) < 5:
            return None
        return {"n": int(near.sum()), "ssha_p90": float(np.percentile(a, 90)),
                "ssha_max": float(a.max()),
                "grad_p90": float(np.percentile(g, 90))}

    rng = np.random.RandomState(0)
    for h in hits:
        st = swot_stats(h["lon"], h["lat"])
        h.update(st or {})
        # 同纬度带、刈幅内、离检出 ≥20 km 的背景对照点
        for _ in range(50):
            i = rng.randint(len(pts))
            blo, bla = pts[i]
            if abs(bla - h["lat"]) < 0.5 and \
               abs(blo - h["lon"]) * 111 > 20:
                bg = swot_stats(blo, bla)
                if bg:
                    h["bg_ssha_p90"] = bg["ssha_p90"]
                    h["bg_grad_p90"] = bg["grad_p90"]
                    break
        print(f"域{h['cid']:4d} ({h['lon']:.2f}E,{h['lat']:.2f}N) "
              f"面积{h['area']:.0f}px 有效点{h.get('n', 0):4d} "
              f"|SSHA|p90 {h.get('ssha_p90', float('nan'))*100:.2f}cm "
              f"(背景 {h.get('bg_ssha_p90', float('nan'))*100:.2f}cm) "
              f"grad_p90 {h.get('grad_p90', float('nan'))*1e6:.2f}e-6 "
              f"(背景 {h.get('bg_grad_p90', float('nan'))*1e6:.2f}e-6)")

    (PROJECT_ROOT / "results/swot_coloc_0930.json").write_text(
        json.dumps(hits, ensure_ascii=False, indent=2))

    # 三联图（前 4 个最大检出）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vv = build_vv_canvas(
        PROJECT_ROOT / f"data/processed/sar_tiles/{SCENE}.SAFE", prob.shape)
    figdir = PROJECT_ROOT / "results/figures"
    for h in sorted(hits, key=lambda x: -x["area"])[:4]:
        cy, cx = int(h["cy"]), int(h["cx"])
        hw = 512
        y0, x0 = np.clip(cy - hw, 0, prob.shape[0] - 2 * hw), \
                 np.clip(cx - hw, 0, prob.shape[1] - 2 * hw)
        patch = vv[y0:y0 + 2 * hw, x0:x0 + 2 * hw]
        pcont = prob[y0:y0 + 2 * hw, x0:x0 + 2 * hw]
        # SWOT 点投到像素坐标
        dlat = 6 / 110.57
        dlon = 6 / (111.32 * np.cos(np.radians(h["lat"])))
        near = ((np.abs(qc["lon"] - h["lon"]) <= dlon)
                & (np.abs(qc["lat"] - h["lat"]) <= dlat))
        plo, pla = qc["lon"][near], qc["lat"][near]
        pssha, pgrad = qc["ssha"][near], qc["grad"][near]
        # 经纬度 → 像素（用 transform 逆变换）
        inv = ~t
        px, py = [], []
        for lo, la in zip(plo, pla):
            c_x, c_y = inv * (lo, la)
            px.append(c_x - x0)
            py.append(c_y - y0)
        px, py = np.array(px), np.array(py)
        ok = (px >= 0) & (px < 2 * hw) & (py >= 0) & (py < 2 * hw)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        lo_p, hi_p = np.percentile(patch[np.isfinite(patch)], [2, 98])
        ax[0].imshow(patch, cmap="gray", vmin=lo_p, vmax=hi_p)
        ax[0].contour(pcont, levels=[0.6], colors="red", linewidths=0.8)
        ax[0].set_title(f"VV  域{h['cid']} ({h['area']:.0f}px)")
        sc = ax[1].scatter(px[ok], py[ok], c=pssha[ok] * 100, cmap="RdBu_r",
                           s=8, vmin=-8, vmax=8)
        ax[1].set_facecolor("black")
        ax[1].set_title(f"SWOT SSHA (cm)  p90|.|={h.get('ssha_p90', 0)*100:.1f}")
        fig.colorbar(sc, ax=ax[1], shrink=0.8)
        sc2 = ax[2].scatter(px[ok], py[ok], c=pgrad[ok] * 1e6, cmap="viridis",
                            s=8)
        ax[2].set_facecolor("black")
        ax[2].set_title("|∇SSHA| (1e-6)")
        fig.colorbar(sc2, ax=ax[2], shrink=0.8)
        for a in ax:
            a.set_xticks([]), a.set_yticks([])
        fig.savefig(figdir / f"swot_coloc_cid{h['cid']}.png", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"→ results/figures/swot_coloc_*.png, results/swot_coloc_0930.json")


if __name__ == "__main__":
    main()
