"""逐峰级精标：把 grade A 波包框掩膜升级为波峰亮带掩膜。

路线（纯图像处理，不用 SAM）：
1. VV 百分位拉伸 + 轻高斯去斑；
2. 2×2 象限各自用 FFT 峰值估计条纹主波长/走向（兼容波包扇形散开与弯曲），
   manifest 有 wavelength_m 时换算成像素做频段约束；
3. 沿条纹拉长的偶对称 Gabor 滤波，响应用局部能量归一化，四象限按高斯权重融合；
4. 正响应阈值（Otsu×0.85，限幅 [0.45,0.9]）取亮峰带，限在原框内，
   连通域面积过滤（< 0.25λ² 的碎斑丢弃）。

输出 <数据集>/masks_percrest/<图名主干>.png（0/255），
QC 叠合图（VV 底 + 逐峰半透明红 + 原框绿轮廓）到 results/figures/percrest_qc/。

用法（在 code/ 目录下）：
    python data_preprocessing/labeling/percrest_refine.py \
        --dataset ../data/datasets/L1_sar_stripe/gt_fine \
        --qc ../results/figures/percrest_qc [--grade A] [--limit 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def px_per_m(entry: dict) -> float:
    """由经纬度范围估算每米像素数（512px 块）。"""
    lon0, lon1 = entry["lon_range"]
    lat = sum(entry["lat_range"]) / 2
    mx = abs(lon1 - lon0) * 111320 * np.cos(np.radians(lat))
    my = abs(entry["lat_range"][1] - entry["lat_range"][0]) * 110540
    return 512 / ((mx + my) / 2)


def estimate_stripe(g: np.ndarray, box: np.ndarray,
                    lam_hint: float | None = None):
    """框内加窗 FFT 峰值 → (主波长 px, 条纹走向 rad)。"""
    win = np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None, :]
    gm = (g - g[box].mean()) * win * box.astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2(gm)))
    H = g.shape[0]
    fy, fx = np.mgrid[-H // 2:H // 2, -H // 2:H // 2]
    lam = 1 / np.maximum(np.sqrt(fx ** 2 + fy ** 2) / H, 1e-9)
    if lam_hint:
        band = (lam >= 0.6 * lam_hint) & (lam <= 1.7 * lam_hint)
    else:
        band = (lam >= 10) & (lam <= 64)  # 约 100~600m
    Fb = np.where(band, F, 0)
    iy, ix = np.unravel_index(np.argmax(Fb), Fb.shape)
    theta = np.arctan2(iy - H // 2, ix - H // 2) + np.pi / 2  # 条纹走向 ⊥ 频率矢量
    return lam[iy, ix], theta


def gabor_resp(g: np.ndarray, lam0: float, theta: float,
               gamma: float = 0.35) -> np.ndarray:
    """偶对称 Gabor 响应，局部能量归一化；正瓣对应亮峰带。"""
    sigma = lam0 / 2.2
    ks = int(max(9, np.ceil(lam0 * 2))) | 1
    k = cv2.getGaborKernel((ks, ks), sigma, np.degrees(theta), lam0, gamma, 0,
                           ktype=cv2.CV_32F)
    k -= k.mean()
    r = cv2.filter2D(g, -1, k)
    loc = cv2.GaussianBlur(r * r, (0, 0), lam0 / 2)
    return r / np.sqrt(loc + 1e-9)


def fused_rn(g: np.ndarray, box: np.ndarray,
             lam_hint: float | None) -> np.ndarray:
    """2×2 象限各估 (λ,θ)，Gabor 响应按到象限中心的高斯权重融合。"""
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rn_sum = np.zeros_like(g)
    w_sum = np.zeros_like(g)
    for cy, cx in [(H / 4, W / 4), (H / 4, 3 * W / 4),
                   (3 * H / 4, W / 4), (3 * H / 4, 3 * W / 4)]:
        w = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (H / 4) ** 2))
        sub = box & (np.abs(yy - cy) < H / 4) & (np.abs(xx - cx) < W / 4)
        if sub.sum() < 500:
            sub = box
        lam0, theta = estimate_stripe(g, sub, lam_hint)
        rn_sum += w * gabor_resp(g, lam0, theta)
        w_sum += w
    return rn_sum / np.maximum(w_sum, 1e-9)


def percrest_mask(g: np.ndarray, box: np.ndarray,
                  lam_hint: float | None) -> tuple[np.ndarray, float]:
    rn = fused_rn(g, box, lam_hint)
    vals = rn[box]
    t = threshold_otsu(vals[vals > 0]) if (vals > 0).any() else 1.0
    thr = float(np.clip(0.85 * t, 0.45, 0.9))
    m = (rn > thr) & box
    lamc = lam_hint if lam_hint else 40.0
    lab = label(m)
    keep = np.zeros_like(m)
    for r in regionprops(lab):
        if r.area < 0.25 * lamc * lamc:
            continue  # 碎斑
        keep[lab == r.label] = True
    return keep, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--qc", default=None, help="QC 叠合图输出目录")
    ap.add_argument("--grade", default="A")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = [e for e in manifest if e.get("grade") == args.grade]
    if args.limit:
        entries = entries[: args.limit]

    out_dir = root / "masks_percrest"
    out_dir.mkdir(exist_ok=True)
    qc_dir = Path(args.qc) if args.qc else None
    if qc_dir:
        qc_dir.mkdir(parents=True, exist_ok=True)

    for e in entries:
        gid = e["id"]
        img = np.load(root / e["image"]).astype(np.float32)
        p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
        g = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
        box = cv2.imread(str(root / e["mask"]), cv2.IMREAD_GRAYSCALE) > 0
        wl = e.get("wavelength_m")
        lam_hint = float(wl) * px_per_m(e) if wl else None

        keep, thr = percrest_mask(g, box, lam_hint)
        cv2.imwrite(str(out_dir / f"{gid}.png"), keep.astype(np.uint8) * 255)
        frac = keep.sum() / max(box.sum(), 1)
        print(f"{gid}: lam_hint={lam_hint and round(lam_hint, 1)} "
              f"thr={thr:.2f} frac={frac:.2f}", flush=True)

        if qc_dir:
            vis = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            ov = vis.copy()
            ov[keep] = (0, 0, 255)
            vis = cv2.addWeighted(vis, 0.55, ov, 0.45, 0)
            cnt, _ = cv2.findContours((box * 255).astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnt, -1, (0, 255, 0), 1)
            cv2.putText(vis, f"{gid} frac={frac:.2f}", (6, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(qc_dir / f"{gid}.png"), vis)
    print(f"完成 {len(entries)} 张 → {out_dir}")


if __name__ == "__main__":
    main()
