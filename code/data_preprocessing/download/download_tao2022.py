"""Tao et al. 2022 Sentinel-1 内波公开数据集下载与整理。

论文：Tao et al. 2022, Earth and Space Science
（An Internal Waves Data Set From Sentinel-1 SAR Imagery and Preliminary Detection，
数据获取地址见论文 Data Availability 一节）。

注意：数据下载地址需以论文给出的官方链接为准，用 --url 传入
（本机网络曾被 Zenodo 临时封锁，脚本不做硬编码链接）。
S1-IW-2023（COCO 格式，1039 框）同理。

整理目标目录（遵守目录约定）：
    data/datasets/L1_sar_stripe/tao2022/
        images/            SAR 切片
        annotations.json   COCO 格式框标注
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from utils.logger import get_logger

logger = get_logger("download_tao2022")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True,
                    help="数据集压缩包直链（以论文 Data Availability 为准）")
    ap.add_argument("--out", default="data/datasets/L1_sar_stripe/tao2022")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import requests
    dest = out / Path(args.url.split("?")[0]).name
    logger.info("下载 %s → %s", args.url, dest)
    r = requests.get(args.url, stream=True, timeout=600,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fp:
        for chunk in r.iter_content(1 << 20):
            fp.write(chunk)

    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as z:
            z.extractall(out)
        dest.unlink()

    logger.info("完成 → %s。请核对目录是否为 images/ + annotations.json 结构，"
                "若不一致按 training/datasets/tao_dataset.py 的约定整理。", out)


if __name__ == "__main__":
    main()
