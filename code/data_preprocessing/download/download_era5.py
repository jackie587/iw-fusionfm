"""ERA5 10m 风场批量下载（Copernicus CDS）。

用途（数据输入处理方案）：S1 场景风速质量筛选——剔除 <2 m/s（条纹不成像）
与 >10 m/s（条纹被淹没）的场景。

凭据（不进版本库）：项目根目录 .secrets/credentials.json 的
{"cds": {"url", "key"}} 节（CDS 个人资料页复制，与 CDSE 是两套系统），
或环境变量 CDS_API_URL / CDS_API_KEY，或 ~/.cdsapirc。

数据集：reanalysis-era5-single-levels（ERA5 逐小时单层再分析），
变量仅 10m u/v 风。按月拆分请求（CDS 对单请求数据量有限制），
已存在的月份文件自动跳过（可断点续传）。

用法（在 code/ 目录下）：
    python data_preprocessing/download/download_era5.py \
        --start 2023-06 --end 2023-12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from utils.logger import get_logger

logger = get_logger("download_era5")

DATASET = "reanalysis-era5-single-levels"
VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
HOURS = [f"{h:02d}:00" for h in range(24)]


def _cds_client():
    import cdsapi

    url = os.environ.get("CDS_API_URL")
    key = os.environ.get("CDS_API_KEY")
    if not url or not key:
        cred_file = PROJECT_ROOT / ".secrets" / "credentials.json"
        if cred_file.exists():
            creds = json.loads(cred_file.read_text(encoding="utf-8")).get("cds")
            if creds:
                url = url or creds["url"]
                key = key or creds["key"]
    if url and key:
        return cdsapi.Client(url=url, key=key)
    return cdsapi.Client()  # 退化：读 ~/.cdsapirc


def month_range(start: str, end: str) -> list[tuple[str, list[str]]]:
    """('2023-06','2023-12') → [('2023', ['06']), ..., ('2023', ['12'])]"""
    y1, m1 = int(start[:4]), int(start[5:7])
    y2, m2 = int(end[:4]), int(end[5:7])
    out = {}
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        out.setdefault(str(y), []).append(f"{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return sorted(out.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-06", help="起始月份 YYYY-MM")
    ap.add_argument("--end", default="2023-12", help="结束月份 YYYY-MM")
    ap.add_argument("--bbox", default="25,105,15,120",
                    help="N,W,S,E（默认外扩版南海北部 AOI）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/raw/auxiliary/era5"))
    args = ap.parse_args()

    north, west, south, east = [float(v) for v in args.bbox.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _cds_client()
    for year, months in month_range(args.start, args.end):
        for month in months:
            out = out_dir / f"era5_wind10m_{year}{month}.nc"
            if out.exists() and out.stat().st_size > 0:
                logger.info("跳过已存在：%s", out.name)
                continue
            logger.info("请求 %s-%s ...", year, month)
            client.retrieve(DATASET, {
                "product_type": ["reanalysis"],
                "variable": VARIABLES,
                "year": [year],
                "month": [month],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": HOURS,
                "area": [north, west, south, east],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }, str(out))
            logger.info("完成 → %s（%.1f MB）", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
