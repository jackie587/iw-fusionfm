"""交叉验证安达曼海 SWOT 检索为 0 是覆盖问题还是查询问题。"""
import time

import earthaccess

AOI_ANDAMAN = (92.0, 8.0, 98.0, 15.0)
AOI_SCS = (109.0, 18.0, 114.0, 23.0)

tests = [
    ("安达曼 科学期 2023-10", AOI_ANDAMAN, "SWOT_L2_LR_SSH_EXPERT_2.0",
     ("2023-10-01", "2023-10-31")),
    ("安达曼 calval 2023-04", AOI_ANDAMAN, "SWOT_L2_LR_SSH_2.0",
     ("2023-04-01", "2023-04-30")),
    ("南海   calval 2023-04", AOI_SCS, "SWOT_L2_LR_SSH_2.0",
     ("2023-04-01", "2023-04-30")),
    ("安达曼 calval 2023-04 全球无bbox", None, "SWOT_L2_LR_SSH_2.0",
     ("2023-04-10", "2023-04-12")),
]
for name, bbox, sn, temporal in tests:
    for attempt in range(3):
        try:
            kw = dict(short_name=sn, temporal=temporal)
            if bbox:
                kw["bounding_box"] = bbox
            gs = earthaccess.search_data(**kw)
            print(f"{name}: {len(gs)} granule")
            if gs:
                print("   例:", gs[0]["meta"]["native-id"])
            break
        except Exception as e:
            print(f"{name}: 第{attempt+1}次失败 {type(e).__name__}，重试")
            time.sleep(5)
