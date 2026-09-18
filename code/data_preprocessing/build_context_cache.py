"""构建上下文校验器数据集缓存：为判读样本库每条样本重裁同中心 1536×1536 上下文。

坐标链（三级回退 + 像素校验）：
1. npz 文件名 n{编号:03d} → 该场景 review/components.json picked[编号-1]，核对 cid；
2. 失配时按 manifest cid 在 picked 里反查；
3. 仍失配时按 cid 在同场景 auto_verdicts.json 全量连通域里反查
   （round3 negmix 场景的 neg100+ 样本来自 picked 之外的大连通域）；
4. 最终用 npz 中心裁块与画布重裁 512 块做像素比对（float16 容差）。
   注意入库脚本（build_negmix_dataset.crop_center）的边界规则是
   居中裁切 + 反射填充，不是 clip，校验必须用同一规则。

旧 _vv_canvas.npy 缓存是 NaN 污染版，本脚本不读它，只从切块局部重建。

用法（在 code/ 目录下）：
    python data_preprocessing/build_context_cache.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

SAMPLE_ROOT = PROJECT_ROOT / "data" / "datasets" / "negative_samples"
TILES_ROOT = PROJECT_ROOT / "data" / "processed" / "sar_tiles"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUT_DIR = SAMPLE_ROOT / "context_cache"

CTX = 1536          # 上下文裁块边长（原分辨率）
OUT = 384           # 降采样后边长
TILE = 512
STRIDE = 384
TILE_PAT = re.compile(r"_y(\d+)_x(\d+)\.npy$")


def find_components(scene: str) -> Path | None:
    """在 results/scene_eval*/<scene>/review/components.json 里按场景名匹配。"""
    hits = sorted(RESULTS_ROOT.glob(f"scene_eval*/{scene}/review/components.json"))
    return hits[0] if hits else None


def build_subcanvas(tiles_dir: Path, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """只在 [y0,y1)×[x0,x1) 区域上重建 VV 画布（重叠区均值，NaN 剔除）。"""
    h, w = y1 - y0, x1 - x0
    canvas = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    for f in (tiles_dir / "images").glob("*.npy"):
        m = TILE_PAT.search(f.name)
        if not m:
            continue
        ty, tx = int(m.group(1)), int(m.group(2))
        if ty + TILE <= y0 or ty >= y1 or tx + TILE <= x0 or tx >= x1:
            continue
        tile = np.load(f)[0].astype(np.float32)
        sy0, sy1 = max(ty, y0) - ty, min(ty + TILE, y1) - ty
        sx0, sx1 = max(tx, x0) - tx, min(tx + TILE, x1) - tx
        dy0, dx0 = max(ty, y0) - y0, max(tx, x0) - x0
        sub = tile[sy0:sy1, sx0:sx1]
        valid = np.isfinite(sub)
        canvas[dy0:dy0 + sub.shape[0], dx0:dx0 + sub.shape[1]] += np.where(valid, sub, 0.0)
        weight[dy0:dy0 + sub.shape[0], dx0:dx0 + sub.shape[1]] += valid
    canvas /= np.maximum(weight, 1e-6)
    canvas[weight == 0] = np.nan
    return canvas


def resolve_coord(entry: dict, picked: list[dict],
                  verdicts: list[dict] | None) -> tuple[float, float, str] | None:
    """三级回退解析 (cy, cx)，返回 (cy, cx, source) 或 None。"""
    cid = int(entry["cid"])
    m = re.search(r"_(?:n|neg)(\d+)\.npz$", entry["file"])
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(picked) and int(picked[idx]["cid"]) == cid:
            return float(picked[idx]["cy"]), float(picked[idx]["cx"]), "picked_index"
    for c in picked:
        if int(c["cid"]) == cid:
            return float(c["cy"]), float(c["cx"]), "picked_cid"
    if verdicts:
        for c in verdicts:
            if int(c["cid"]) == cid:
                return float(c["cy"]), float(c["cx"]), "auto_verdicts"
    return None


def crop_center(canvas: np.ndarray, cy: float, cx: float, size: int) -> np.ndarray:
    """以 (cy,cx) 为中心裁 size×size，越界反射填充（与入库脚本 crop_center 同规则）。"""
    import cv2
    h, w = canvas.shape
    y0, x0 = int(cy) - size // 2, int(cx) - size // 2
    ys = np.clip([y0, y0 + size], 0, h)
    xs = np.clip([x0, x0 + size], 0, w)
    patch = canvas[ys[0]:ys[1], xs[0]:xs[1]]
    pt, pl = ys[0] - y0, xs[0] - x0
    pb, pr = y0 + size - ys[1], x0 + size - xs[1]
    if pt or pb or pl or pr:
        patch = cv2.copyMakeBorder(np.nan_to_num(patch), pt, pb, pl, pr,
                                   cv2.BORDER_REPLICATE)
    return patch


def main():
    manifest = json.load(open(SAMPLE_ROOT / "manifest.json", encoding="utf-8"))
    by_scene: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(manifest):
        by_scene[e["scene"]].append(i)

    n = len(manifest)
    ctx_all = np.zeros((n, OUT, OUT), np.float16)
    meta = []
    stats = {"components_found": 0, "components_missing": 0,
             "coord_unresolved": 0, "nan_pixels_total": 0,
             "src_picked_index": 0, "src_picked_cid": 0, "src_auto_verdicts": 0,
             "coord_verified": 0, "coord_verify_failed": 0}

    for scene, idxs in sorted(by_scene.items()):
        comp_path = find_components(scene)
        if comp_path is None:
            print(f"[缺失] {scene} 无 components.json，{len(idxs)} 条退回无上下文")
            stats["components_missing"] += len(idxs)
            for i in idxs:
                meta.append({"index": i, "status": "no_components"})
            continue
        picked = json.load(open(comp_path, encoding="utf-8"))["picked"]
        verdict_path = comp_path.parent.parent / "auto_verdicts.json"
        verdicts = (json.load(open(verdict_path, encoding="utf-8"))["components"]
                    if verdict_path.exists() else None)

        # 三级回退解析每条样本的画布坐标
        coords = {}
        for i in idxs:
            e = manifest[i]
            rc = resolve_coord(e, picked, verdicts)
            if rc is None:
                meta.append({"index": i, "status": "coord_unresolved"})
                stats["coord_unresolved"] += 1
                continue
            cy, cx, source = rc
            stats[f"src_{source}"] += 1
            coords[i] = (cy, cx)

        #  union 矩形上局部重建画布
        half = CTX // 2
        ys = [c[0] for c in coords.values()]
        xs = [c[1] for c in coords.values()]
        y0 = max(int(min(ys)) - half, 0)
        x0 = max(int(min(xs)) - half, 0)
        tiles_dir = TILES_ROOT / f"{scene}.SAFE"
        maxy = maxx = 0
        for f in (tiles_dir / "images").glob("*.npy"):
            mm = TILE_PAT.search(f.name)
            if mm:
                maxy = max(maxy, int(mm.group(1)) + TILE)
                maxx = max(maxx, int(mm.group(2)) + TILE)
        y1 = min(int(max(ys)) + half, maxy)
        x1 = min(int(max(xs)) + half, maxx)
        canvas = build_subcanvas(tiles_dir, y0, x0, y1, x1)
        stats["components_found"] += len(coords)

        for i, (cy, cx) in coords.items():
            # 像素校验：按入库时的居中+反射规则重裁 512 中心块，与 npz 的 vv 比对
            center = crop_center(canvas, cy - y0, cx - x0, 512)
            vv_ref = np.load(SAMPLE_ROOT / manifest[i]["file"])["vv"].astype(np.float32)
            # 容差：旧缓存 NaN→0 污染的零星像素允许 ≤2% 不一致
            if (np.abs(center - vv_ref) > 0.01).mean() > 0.02:
                stats["coord_verify_failed"] += 1
                meta.append({"index": i, "status": "coord_verify_failed",
                             "cy": cy, "cx": cx})
                continue
            stats["coord_verified"] += 1
            patch = crop_center(canvas, cy - y0, cx - x0, CTX)
            stats["nan_pixels_total"] += int(np.isnan(patch).sum())
            patch = np.nan_to_num(patch, nan=0.0)
            # 4×4 块均值降采样 1536→384
            patch = patch.reshape(OUT, CTX // OUT, OUT, CTX // OUT).mean((1, 3))
            ctx_all[i] = patch.astype(np.float16)
            meta.append({"index": i, "status": "ok", "cy": cy, "cx": cx})
        print(f"[完成] {scene}  样本 {len(idxs)}  局部画布 {canvas.shape}  "
              f"({comp_path.relative_to(PROJECT_ROOT)})")

    OUT_DIR.mkdir(exist_ok=True)
    np.save(OUT_DIR / "context.npy", ctx_all)
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "samples": sorted(meta, key=lambda d: d["index"])},
                  f, ensure_ascii=False, indent=1)
    print("统计:", json.dumps(stats, ensure_ascii=False))
    print(f"→ {OUT_DIR}/context.npy {ctx_all.shape}")


if __name__ == "__main__":
    main()
