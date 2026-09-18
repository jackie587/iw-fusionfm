"""检查 calval 期南海 S1/SWOT 时刻分布，解释 4~5 月无配对的原因。"""
import json
from collections import Counter
from pathlib import Path

data = json.loads(Path(
    "../data/datasets/L2_s1_swot_matched/pairs_census_scs_calval.json"
).read_text(encoding="utf-8"))

s1_by_month = Counter(s["time"][:7] for s in data["s1"])
print("S1 按月景数:", dict(sorted(s1_by_month.items())))

sw_exp = [g for g in data["swot"] if "Expert" in g["path"]]
sw_by_month = Counter(g["time"][:7] for g in sw_exp)
print("SWOT(Expert) 按月 pass 数:", dict(sorted(sw_by_month.items())))

print("\nS1 各月 UTC 时刻分布（时:分  top5）:")
for m in sorted(s1_by_month):
    hhmm = Counter(s["time"][11:16] for s in data["s1"] if s["time"][:7] == m)
    print(f"  {m}: {hhmm.most_common(5)}")

print("\nSWOT(Expert) 各月 UTC 时刻分布（时  top6）:")
for m in sorted(sw_by_month):
    hh = Counter(g["time"][11:13] for g in sw_exp if g["time"][:7] == m)
    print(f"  {m}: {hh.most_common(6)}")
