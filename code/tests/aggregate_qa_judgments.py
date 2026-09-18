"""汇总 QA 判读：按特征分箱统计 TP 率，寻找数据驱动的过滤规则。

用法：python tests/aggregate_qa_judgments.py [tag]   # tag 如 v2db，默认 v1
"""
import glob
import json
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path(__file__).resolve().parents[2] / "results" / "event_database"
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
TAG_ = f"_{TAG}" if TAG else ""

events = {r["event_id"]: r for r in
          csv.DictReader(open(DB / "events.csv", encoding="utf-8"))}
rows = []
for q in ["high", "medium", "low"]:
    recs = json.load(open(DB / "qa_sheets" / f"qa_{q}{TAG_}_records.json",
                          encoding="utf-8"))
    for jf in sorted(glob.glob(str(DB / "qa_sheets"
                                   / f"qa_{q}{TAG_}_*_judgment.json"))):
        j = json.load(open(jf, encoding="utf-8"))
        for p in j["panels"]:
            num = p["num"]
            if num < 1 or num > len(recs):
                print("编号越界", jf, num)
                continue
            eid = recs[num - 1]["event_id"]
            e = events.get(eid)
            rows.append({"quality": q, "judgment": p["judgment"],
                         "event_id": eid,
                         "n_crest": int(e["n_crest"]),
                         "wl": float(e["wavelength_m"]) if e["wavelength_m"] else np.nan,
                         "area": float(e["area_km2"]),
                         "prob": float(e["mean_prob"]),
                         "wind": float(e["wind_ms"])})
df = pd.DataFrame(rows)
df["score"] = df.judgment.map({"TP": 1.0, "UNCERTAIN": 0.5, "FP": 0.0})

print("总数", len(df), df.judgment.value_counts().to_dict())
for q in ["high", "medium", "low"]:
    sub = df[df.quality == q]
    tp = (sub.judgment == "TP").sum()
    un = (sub.judgment == "UNCERTAIN").sum()
    print(f"--- {q}: n={len(sub)} TP={tp} UNC={un} "
          f"TP率={tp / len(sub):.2f} 含UNC={sub.score.mean():.2f}")

print("\n按 n_crest 分箱:")
for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 9), (10, 20), (21, 999)]:
    sub = df[(df.n_crest >= lo) & (df.n_crest <= hi)]
    if len(sub):
        print(f"  {lo}-{hi}: n={len(sub)} "
              f"TP率={(sub.judgment == 'TP').mean():.2f} "
              f"含UNC={sub.score.mean():.2f}")

print("\n按波长有效性:")
df["wl_ok"] = df.wl.between(100, 1500)
for flag in [True, False]:
    sub = df[df.wl_ok == flag]
    print(f"  wl∈[100,1500]={flag}: n={len(sub)} "
          f"TP率={(sub.judgment == 'TP').mean():.2f} "
          f"含UNC={sub.score.mean():.2f}")

print("\n按面积分箱:")
for lo, hi in [(0, 500), (500, 2000), (2000, 1e9)]:
    sub = df[(df.area >= lo) & (df.area < hi)]
    print(f"  {lo}-{hi}: n={len(sub)} "
          f"TP率={(sub.judgment == 'TP').mean():.2f} "
          f"含UNC={sub.score.mean():.2f}")

print("\n组合规则候选（n_crest≥4 且 λ∈[100,1500] 且 area≤2000）:")
strict = df[(df.n_crest >= 4) & df.wl_ok & (df.area <= 2000)]
print(f"  样本内: n={len(strict)} TP率={(strict.judgment == 'TP').mean():.2f} "
      f"含UNC={strict.score.mean():.2f}")
rest = df[~df.index.isin(strict.index)]
print(f"  其余: n={len(rest)} TP率={(rest.judgment == 'TP').mean():.2f}")

df.to_csv(DB / "qa_sheets" / f"qa_judgments_merged{TAG_}.csv", index=False)
print(f"\nsaved qa_judgments_merged{TAG_}.csv")
