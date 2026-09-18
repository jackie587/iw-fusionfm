"""探测 calval 期（1 天轨道，2023-04）SWOT KaRIn SSH 产品的正确 short_name。

背景：Expert 2.0 在 calval 期检索为 0（南海/安达曼海均 0 granule），
怀疑 calval 期数据在另一个 collection/版本里。
"""
import sys

import earthaccess  # CMR 检索免登录，仅下载才需 Earthdata 凭据

CANDIDATES = [
    "SWOT_L2_LR_SSH_EXPERT_2.0",
    "SWOT_L2_LR_SSH_BASIC_2.0",
    "SWOT_L2_LR_SSH_2.0",
    "SWOT_L2_LR_SSH_EXPERT_1.0",
    "SWOT_L2_LR_SSH_BASIC_1.0",
    "SWOT_L2_LR_SSH_1.0",
    "SWOT_L2_LR_SSH_EXPERT_1.1",
    "SWOT_L2_LR_SSH_BASIC_1.1",
]
AOI = (109.0, 18.0, 114.0, 23.0)
CALVAL = ("2023-04-01", "2023-04-30")

print("== calval 期（2023-04，南海 AOI）按 short_name 探测 ==")
for sn in CANDIDATES:
    try:
        gs = earthaccess.search_data(short_name=sn, bounding_box=AOI,
                                     temporal=CALVAL)
        print(f"  {sn}: {len(gs)} granule")
    except Exception as e:
        print(f"  {sn}: 查询失败 {type(e).__name__} {e}")

print("\n== earthaccess 里所有 SWOT_L2_LR_SSH 相关 collection ==")
try:
    cols = earthaccess.search_datasets(keyword="SWOT L2 LR SSH")
    for c in cols:
        print(" ", c.get("umm", {}).get("ShortName"),
              c.get("umm", {}).get("Version"))
except Exception as e:
    print("  查询失败", type(e).__name__, e)
