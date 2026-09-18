"""事件库抽验：high 级事件的波峰线 × SAR 叠合图。

从 events.csv 抽 high/medium 事件，按事件 bbox 裁 VV 画布窗口，
叠合该事件的波峰线（crest_lines/*.geojson，event_id 匹配），
出 5×5 编号 sheet 供 AI 目视抽验，校准 quality 分级阈值。

用法（在 code/ 目录下）：
    python tests/make_event_qa_sheets.py --n 25 --quality high
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_DB = PROJECT_ROOT / "results/event_database"
TILES_ROOT = PROJECT_ROOT / "data/processed/sar_tiles"

GRID = 5
MARGIN = 1.5          # 窗口 = 事件 bbox 的 MARGIN 倍（最小 600px）
MAX_WIN = 5000        # 窗口封顶（巨块事件以质心为中心）


def load_events(quality: str) -> list[dict]:
    with open(EVENT_DB / "events.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["quality"] == quality]


def scene_geo(scene: str):
    """画布原点与像元大小（从 y0x0 切块 meta）。"""
    m = json.loads(next((TILES_ROOT / f"{scene}.SAFE" / "meta").glob(
        "*_y00000_x00000.json")).read_text(encoding="utf-8"))
    a, _, lon0, _, b, lat0 = m["transform"]
    return lon0, lat0, a, b


TILE_SIZE, TILE_STRIDE = 512, 384


def load_vv_patch(scene: str, X1: int, X2: int, Y1: int, Y2: int
                  ) -> np.ndarray:
    """按需拼接窗口内的 VV 像素（重叠区取均值），避免整幅画布占内存。"""
    import re
    img_dir = TILES_ROOT / f"{scene}.SAFE" / "images"
    pat = re.compile(r"_y(\d+)_x(\d+)\.npy$")
    acc = np.zeros((Y2 - Y1, X2 - X1), np.float32)
    wgt = np.zeros((Y2 - Y1, X2 - X1), np.float32)
    y0 = max((Y1 - TILE_SIZE) // TILE_STRIDE, 0) * TILE_STRIDE
    x0 = max((X1 - TILE_SIZE) // TILE_STRIDE, 0) * TILE_STRIDE
    for ty in range(y0, Y2, TILE_STRIDE):
        for tx in range(x0, X2, TILE_STRIDE):
            f = img_dir / f"{scene}_y{ty:05d}_x{tx:05d}.npy"
            if not f.exists():
                continue
            tile = np.load(f)[0]
            iy1, iy2 = max(ty, Y1) - ty, min(ty + TILE_SIZE, Y2) - ty
            ix1, ix2 = max(tx, X1) - tx, min(tx + TILE_SIZE, X2) - tx
            py1, px1 = max(ty - Y1, 0), max(tx - X1, 0)
            sub = tile[iy1:iy2, ix1:ix2]
            valid = np.isfinite(sub)
            acc[py1:py1 + sub.shape[0], px1:px1 + sub.shape[1]] += np.where(
                valid, sub, 0.0)
            wgt[py1:py1 + sub.shape[0], px1:px1 + sub.shape[1]] += valid
    out = acc / np.maximum(wgt, 1e-6)
    out[wgt == 0] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--quality", default="high")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--minwin", type=int, default=600,
                    help="窗口最小边长（px）")
    ap.add_argument("--db", action="store_true",
                    help="dB 拉伸显示（弱条带更清晰）")
    ap.add_argument("--tag", default="",
                    help="输出文件名附加标签（避免覆盖旧 sheet）")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = load_events(args.quality)
    rng = np.random.RandomState(args.seed)
    # 按场景分层抽样，避免单一场景霸屏
    by_scene = {}
    for e in events:
        by_scene.setdefault(e["scene"], []).append(e)
    per = max(1, args.n // len(by_scene))
    picks = []
    for s, es in sorted(by_scene.items()):
        idx = rng.choice(len(es), size=min(per, len(es)), replace=False)
        picks += [es[i] for i in sorted(idx)]
    rng.shuffle(picks)
    picks = picks[:args.n]
    print(f"{args.quality} 事件 {len(events)} 个，抽样 {len(picks)} 个"
          f"（覆盖 {len(by_scene)} 景）")

    out_dir = EVENT_DB / "qa_sheets"
    out_dir.mkdir(exist_ok=True)
    crest_cache, geo_cache, shape_cache = {}, {}, {}

    records = []
    grid = args.grid
    for si in range(0, len(picks), grid * grid):
        batch = picks[si:si + grid * grid]
        fig, axes = plt.subplots(grid, grid, figsize=(grid * 3.6, grid * 3.6))
        for ax, e in zip(axes.flat, batch):
            scene = e["scene"]
            if scene not in geo_cache:
                import re
                ys, xs = [], []
                for mf in (TILES_ROOT / f"{scene}.SAFE" / "meta").glob(
                        "*_y*_x*.json"):
                    mm = re.search(r"_y(\d+)_x(\d+)\.json$", mf.name)
                    ys.append(int(mm.group(1)))
                    xs.append(int(mm.group(2)))
                shape_cache[scene] = (max(ys) + 512, max(xs) + 512)
                geo_cache[scene] = scene_geo(scene)
                g = json.loads((EVENT_DB / "crest_lines"
                                / f"{scene}.geojson").read_text(
                                    encoding="utf-8"))
                crest_cache[scene] = [
                    f for f in g["features"]]
            lon0, lat0, a, b = geo_cache[scene]
            h, w = shape_cache[scene]

            lo1, lo2 = (float(v) for v in e["bbox_lon"].split(","))
            la1, la2 = (float(v) for v in e["bbox_lat"].split(","))
            x1, x2 = (lo1 - lon0) / a, (lo2 - lon0) / a
            y1, y2 = (la1 - lat0) / b, (la2 - lat0) / b
            # 巨块事件（聚类合并的大波场）以质心为中心，窗口封顶 MAX_WIN
            cx, cy = (float(e["lon"]) - lon0) / a, (float(e["lat"]) - lat0) / b
            hw = min(max((x2 - x1) * args.margin, (y2 - y1) * args.margin,
                         args.minwin), MAX_WIN) / 2
            X1, X2 = int(max(cx - hw, 0)), int(min(cx + hw, w))
            Y1, Y2 = int(max(cy - hw, 0)), int(min(cy + hw, h))
            patch = load_vv_patch(scene, X1, X2, Y1, Y2)
            if args.db:
                img = 10 * np.log10(np.maximum(patch, 1e-7))
                v5, v99 = np.percentile(img[np.isfinite(img)], [5, 99])
                ax.imshow(img, cmap="gray", vmin=v5, vmax=v99)
            else:
                p2, p98 = np.percentile(patch[np.isfinite(patch)], [2, 98])
                ax.imshow(patch, cmap="gray", vmin=p2,
                          vmax=max(p98, p2 + 1e-3))

            n_lines = 0
            for f in crest_cache[scene]:
                if f["properties"]["event_id"] != e["event_id"]:
                    continue
                coords = np.array(f["geometry"]["coordinates"])
                xs = (coords[:, 0] - lon0) / a - X1
                ys = (coords[:, 1] - lat0) / b - Y1
                ax.plot(xs, ys, "r-", lw=1.0)
                n_lines += 1
            wl = e["wavelength_m"] or "-"
            ax.set_title(f"#{si + batch.index(e) + 1} {e['event_id'][-8:]}"
                         f" {n_lines}峰 λ={wl}m", fontsize=9)
            ax.axis("off")
            records.append({"event_id": e["event_id"], "scene": scene,
                            "n_lines_drawn": n_lines,
                            "wavelength_m": e["wavelength_m"],
                            "wind_ms": e["wind_ms"]})
        for ax in axes.flat[len(batch):]:
            ax.axis("off")
        tag = f"_{args.tag}" if args.tag else ""
        out = out_dir / f"qa_{args.quality}{tag}_{si // (grid * grid) + 1:02d}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
    (out_dir / f"qa_{args.quality}{tag}_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out_dir}/ ({(len(picks) + grid*grid - 1) // (grid*grid)} 张 sheet)")


if __name__ == "__main__":
    main()
