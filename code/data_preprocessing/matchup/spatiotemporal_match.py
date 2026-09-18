"""S1 × SWOT 时空匹配工具（构建 L2 配对样本清单）。

匹配准则（数据输入处理方案四）：
- 严格配对：|Δt| < 30 min 且空间重叠；
- 放宽配对：Δt 放宽到数小时，但用 SAR 条纹拟合的传播方向 +
  波速范围（0.5~3 m/s）做传播位移校正，要求校正后空间仍重叠。
  注意：传播方向必须用 SAR 条纹拟合结果，不能用模型预测值（避免循环依赖）。

输入：两份元数据清单 json（s1_scenes.json / swot_granules.json），
每条含 {"path", "time"(ISO8601), "lon_min","lat_min","lon_max","lat_max"}。
输出：配对清单 json，供重投影与训练集构建使用。
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from utils.geo import haversine_km, propagation_shift
from utils.logger import get_logger

logger = get_logger("matchup")


def center(rec: dict) -> tuple[float, float]:
    return ((rec["lon_min"] + rec["lon_max"]) / 2,
            (rec["lat_min"] + rec["lat_max"]) / 2)


def boxes_overlap(a: dict, b: dict) -> bool:
    return not (a["lon_max"] < b["lon_min"] or a["lon_min"] > b["lon_max"]
                or a["lat_max"] < b["lat_min"] or a["lat_min"] > b["lat_max"])


def match(s1_scenes: list[dict], swot_granules: list[dict],
          strict_min: float = 30.0, relaxed_hours: float = 6.0,
          speed_range: tuple[float, float] = (0.5, 3.0)) -> list[dict]:
    pairs = []
    for s1, sw in itertools.product(s1_scenes, swot_granules):
        t1 = datetime.fromisoformat(s1["time"])
        t2 = datetime.fromisoformat(sw["time"])
        dt_min = abs((t1 - t2).total_seconds()) / 60.0

        if not boxes_overlap(s1, sw):
            # 放宽情形：中心距过大直接跳过
            c1, c2 = center(s1), center(sw)
            if haversine_km(c1[0], c1[1], c2[0], c2[1]) > 300:
                continue

        if dt_min <= strict_min and boxes_overlap(s1, sw):
            pairs.append({"s1": s1["path"], "swot": sw["path"],
                          "dt_min": dt_min, "type": "strict"})
        elif dt_min <= relaxed_hours * 60:
            # 需要传播校正：由调用方在拿到 SAR 条纹方向后调用
            # propagation_shift() 做位移校正再复核重叠。这里先标记候选。
            dt_s = abs((t1 - t2).total_seconds())
            shift_max_km = speed_range[1] * dt_s / 1000.0
            c1, c2 = center(s1), center(sw)
            if haversine_km(c1[0], c1[1], c2[0], c2[1]) <= 300 + shift_max_km:
                pairs.append({"s1": s1["path"], "swot": sw["path"],
                              "dt_min": dt_min, "type": "relaxed",
                              "note": "需用 SAR 条纹拟合方向做传播位移校正"})
    pairs.sort(key=lambda p: p["dt_min"])
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-meta", required=True, help="S1 场景清单 json")
    ap.add_argument("--swot-meta", required=True, help="SWOT granule 清单 json")
    ap.add_argument("--out", default="data/datasets/L2_s1_swot_matched/pairs.json")
    ap.add_argument("--strict-min", type=float, default=30.0)
    ap.add_argument("--relaxed-hours", type=float, default=6.0)
    args = ap.parse_args()

    with open(args.s1_meta, encoding="utf-8") as f:
        s1 = json.load(f)
    with open(args.swot_meta, encoding="utf-8") as f:
        sw = json.load(f)

    pairs = match(s1, sw, args.strict_min, args.relaxed_hours)
    strict = sum(1 for p in pairs if p["type"] == "strict")
    logger.info("配对结果：严格 %d 对，放宽 %d 对", strict, len(pairs) - strict)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info("写出 → %s", out)


if __name__ == "__main__":
    main()
