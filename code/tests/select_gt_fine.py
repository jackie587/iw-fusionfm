"""GT 精标子集选样：事件库引导 + L2 配对块 + 负样本对照。

选样来源三路（全部 512×512）：
1. 场景切块（data/processed/sar_tiles/<scene>.SAFE/images/）：按事件库
   events.csv 的 high/medium 事件质心定位切块，分西丛集（lon<109.3，
   海南岛以西）/东丛集（南海东北陆坡）与波长档分层抽样；
2. L2 配对块（data/datasets/L2_s1_swot_matched/tiles/）：只取 20230830
   留出天（FiLM 实验的验证天，避免其训练泄漏）label_quality==clean 且
   label_frac 适中的块，便于后续融合组带真实 SWOT 重估；
3. 负样本：L2 none 块（v7 无响应）+ 场景低概率海面块，目视确认无波后
   掩膜全零。

产出：
- data/datasets/L1_sar_stripe/gt_fine/selection.json（选样清单，含理由）
- results/gt_fine/select_qc_*.png（VV 缩略 QC sheet，供 AI 目视核样）

用法（code/ 目录下）：
    python tests/select_gt_fine.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"
SCENE_EVAL = PROJECT_ROOT / "results/scene_eval_v7sam"
EVENT_DB = PROJECT_ROOT / "results/event_database"
L2_ROOT = PROJECT_ROOT / "data/datasets/L2_s1_swot_matched/tiles"
GT_ROOT = PROJECT_ROOT / "data/datasets/L1_sar_stripe/gt_fine"
QC_DIR = PROJECT_ROOT / "results/gt_fine"

STRIDE = 384
TILE = 512
WEST_LON = 109.3          # 西丛集（海南岛以西）/ 东丛集分界
N_SCENE_POS = 35          # 场景事件正样本配额
N_L2_POS = 15             # L2 clean 正样本配额
N_NEG = 10                # 负样本配额
SEED = 20260901

PAT = re.compile(r"_y(\d+)_x(\d+)$")


def scene_geo(scene: str):
    m = json.loads(next((TILES_ROOT / f"{scene}.SAFE" / "meta").glob(
        "*_y00000_x00000.json")).read_text(encoding="utf-8"))
    a, _, lon0, _, b, lat0 = m["transform"]
    return lon0, lat0, a, b


def load_tile(scene: str, stem: str) -> np.ndarray | None:
    f = TILES_ROOT / f"{scene}.SAFE/images/{stem}.npy"
    return np.load(f) if f.exists() else None


def tile_ok(vv: np.ndarray, scene: str, y: int, x: int,
            max_nan: float = 0.3, min_sea: float = 0.5) -> bool:
    if np.isnan(vv).mean() > max_nan:
        return False
    lm_file = SCENE_EVAL / scene / "land_mask.npy"
    if lm_file.exists():
        lm = np.load(lm_file)
        sea = 1.0 - lm[y:y + TILE, x:x + TILE].mean()
        if sea < min_sea:
            return False
    return True


def event_tile(scene: str, lon: float, lat: float):
    """事件质心 → 最近的切块栅格位置；返回 (stem, y, x) 或 None。"""
    lon0, lat0, a, b = scene_geo(scene)
    cx, cy = (lon - lon0) / a, (lat - lat0) / b
    ty = int(round((cy - TILE / 2) / STRIDE) * STRIDE)
    tx = int(round((cx - TILE / 2) / STRIDE) * STRIDE)
    for dy, dx in [(0, 0), (0, -STRIDE), (-STRIDE, 0), (0, STRIDE),
                   (STRIDE, 0)]:
        y, x = ty + dy, tx + dx
        if y < 0 or x < 0:
            continue
        stem = f"{scene}_y{y:05d}_x{x:05d}"
        if (TILES_ROOT / f"{scene}.SAFE/images/{stem}.npy").exists():
            return stem, y, x
    return None


def wl_bin(wl: str) -> str:
    try:
        v = float(wl)
    except (TypeError, ValueError):
        return "na"
    return "short" if v < 300 else ("mid" if v <= 600 else "long")


def main() -> None:
    rng = np.random.RandomState(SEED)
    selection: list[dict] = []

    # ---------- 1. 场景事件正样本 ----------
    with open(EVENT_DB / "events.csv", encoding="utf-8") as f:
        events = [r for r in csv.DictReader(f)
                  if r["quality"] in ("high", "medium")]
    # 分层：cluster × 波长档
    strata: dict[tuple[str, str], list[dict]] = {}
    for e in events:
        clu = "west" if float(e["lon"]) < WEST_LON else "east"
        strata.setdefault((clu, wl_bin(e["wavelength_m"])), []).append(e)
    for k in strata:
        rng.shuffle(strata[k])
        # high 优先
        strata[k].sort(key=lambda e: 0 if e["quality"] == "high" else 1)
    quotas = {("west", "short"): 4, ("west", "mid"): 5, ("west", "long"): 3,
              ("west", "na"): 3,
              ("east", "short"): 5, ("east", "mid"): 8, ("east", "long"): 4,
              ("east", "na"): 3}
    per_scene_cap = 4
    scene_count: dict[str, int] = {}
    used_tiles: set[str] = set()
    for key, quota in quotas.items():
        got = 0
        for e in strata.get(key, []):
            if got >= quota:
                break
            scene = e["scene"]
            if scene_count.get(scene, 0) >= per_scene_cap:
                continue
            loc = event_tile(scene, float(e["lon"]), float(e["lat"]))
            if loc is None:
                continue
            stem, y, x = loc
            if stem in used_tiles:
                continue
            tile = load_tile(scene, stem)
            if tile is None or not tile_ok(tile[0], scene, y, x):
                continue
            used_tiles.add(stem)
            scene_count[scene] = scene_count.get(scene, 0) + 1
            got += 1
            selection.append({
                "source": "scene_event",
                "scene": scene, "tile": stem, "y": y, "x": x,
                "event_id": e["event_id"],
                "cluster": key[0], "wavelength_m": e["wavelength_m"],
                "direction_deg": e["direction_deg"],
                "reason": f"事件库{e['quality']}事件 {e['event_id'][-12:]}，"
                          f"{key[0]}丛集 λ={e['wavelength_m'] or '-'}m",
            })
    n_scene = len(selection)
    print(f"场景事件正样本 {n_scene} 个（配额 {N_SCENE_POS}）")

    # ---------- 2. L2 clean 正样本（20230830 留出天） ----------
    manifest = json.loads(
        (L2_ROOT / "manifest_all.json").read_text(encoding="utf-8"))
    l2_clean = [m for m in manifest
                if m["day"] == "20230830" and m["label_quality"] == "clean"
                and 0.01 <= (m["label_frac"] or 0) <= 0.30]
    by_scene: dict[str, list[dict]] = {}
    for m in l2_clean:
        by_scene.setdefault(m["scene"], []).append(m)
    per = max(1, N_L2_POS // max(1, len(by_scene)))
    n_l2 = 0
    for scene, ms in sorted(by_scene.items()):
        rng.shuffle(ms)
        for m in ms[:per + 2]:
            if n_l2 >= N_L2_POS:
                break
            if m["tile"] in used_tiles:
                continue
            used_tiles.add(m["tile"])
            n_l2 += 1
            selection.append({
                "source": "l2_clean",
                "scene": scene, "tile": m["tile"],
                "y": int(PAT.search(m["tile"]).group(1)),
                "x": int(PAT.search(m["tile"]).group(2)),
                "l2_file": m["file"],
                "event_id": None, "cluster": None,
                "wavelength_m": None, "direction_deg": None,
                "reason": f"L2 留出天(0830) clean 块，弱标签占比 "
                          f"{m['label_frac']:.3f} 适中，SWOT 覆盖率 "
                          f"{m['swot_coverage']:.2f}（可供 FiLM 融合组重估）",
            })
    print(f"L2 clean 正样本 {n_l2} 个（配额 {N_L2_POS}）")

    # ---------- 3. 负样本 ----------
    n_neg = 0
    # 3a. L2 none 块（0830 天）
    l2_none = [m for m in manifest
               if m["day"] == "20230830" and m["label_quality"] == "none"]
    rng.shuffle(l2_none)
    for m in l2_none:
        if n_neg >= N_NEG // 2:
            break
        if m["tile"] in used_tiles:
            continue
        used_tiles.add(m["tile"])
        n_neg += 1
        selection.append({
            "source": "l2_none_neg",
            "scene": m["scene"], "tile": m["tile"],
            "y": int(PAT.search(m["tile"]).group(1)),
            "x": int(PAT.search(m["tile"]).group(2)),
            "l2_file": m["file"],
            "event_id": None, "cluster": None,
            "wavelength_m": None, "direction_deg": None,
            "reason": "L2 none 块（v7 无响应），负样本候选，待目视确认无波",
        })
    # 3b. 场景低概率海面块
    scenes = sorted(p.name for p in SCENE_EVAL.iterdir() if p.is_dir())
    rng.shuffle(scenes)
    for scene in scenes:
        if n_neg >= N_NEG:
            break
        pf = SCENE_EVAL / scene / "prob.npy"
        if not pf.exists():
            continue
        prob = np.load(pf)
        h, w = prob.shape
        # 注意：v7 背景概率整体偏高（海面中位 ~0.5），没有 max<0.2 的块；
        # 改为按块内概率均值升序取最低的一小批做负样本候选（目视终审）
        cands = []
        for y in range(0, h - TILE, STRIDE * 2):
            for x in range(0, w - TILE, STRIDE * 2):
                stem = f"{scene}_y{y:05d}_x{x:05d}"
                if stem in used_tiles:
                    continue
                sub = np.asarray(prob[y:y + TILE, x:x + TILE], np.float32)
                if np.isnan(sub).mean() > 0.05:
                    continue
                cands.append((float(np.nanmean(sub)), stem, y, x))
        cands.sort()
        picked = 0
        for _, stem, y, x in cands:
            tile = load_tile(scene, stem)
            if tile is None or not tile_ok(tile[0], scene, y, x,
                                           max_nan=0.05, min_sea=0.9):
                continue
            used_tiles.add(stem)
            n_neg += 1
            picked += 1
            selection.append({
                "source": "scene_neg",
                "scene": scene, "tile": stem, "y": y, "x": x,
                "event_id": None, "cluster": None,
                "wavelength_m": None, "direction_deg": None,
                "reason": "场景内概率均值最低的块之一（背景级响应），"
                          "负样本候选，待目视确认无波",
            })
            if n_neg >= N_NEG or picked >= 2:
                break
    print(f"负样本 {n_neg} 个（配额 {N_NEG}）")

    # ---------- 编号 + 落盘 ----------
    for i, s in enumerate(selection):
        s["id"] = f"gt{i:03d}"
    GT_ROOT.mkdir(parents=True, exist_ok=True)
    (GT_ROOT / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"共 {len(selection)} 张 → {GT_ROOT}/selection.json")

    # ---------- QC sheet（VV 缩略图，供 AI 目视核样） ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GRID = 5
    QC_DIR.mkdir(parents=True, exist_ok=True)
    vv_cache: dict[str, np.ndarray] = {}

    def get_vv(s: dict) -> np.ndarray:
        if "l2_file" in s and s["l2_file"]:
            return np.load(L2_ROOT / s["l2_file"])["sar"][0].astype(np.float32)
        return load_tile(s["scene"], s["tile"])[0]

    for si in range(0, len(selection), GRID * GRID):
        batch = selection[si:si + GRID * GRID]
        fig, axes = plt.subplots(GRID, GRID, figsize=(GRID * 3, GRID * 3))
        for ax, s in zip(axes.flat, batch):
            vv = get_vv(s)
            finite = vv[np.isfinite(vv)]
            p2, p98 = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
            ax.imshow(np.nan_to_num(vv), cmap="gray",
                      vmin=p2, vmax=max(p98, p2 + 1e-3))
            tag = {"scene_event": "EV", "l2_clean": "L2",
                   "l2_none_neg": "N0", "scene_neg": "NS"}[s["source"]]
            ax.set_title(f"{s['id']} {tag} {s['scene'][-4:]}", fontsize=9)
            ax.axis("off")
        for ax in axes.flat[len(batch):]:
            ax.axis("off")
        out = QC_DIR / f"select_qc_{si // (GRID * GRID) + 1:02d}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
    print(f"QC sheet → {QC_DIR}/select_qc_*.png")


if __name__ == "__main__":
    main()
