"""SNAP GPT 批量预处理驱动脚本。

前置条件（用户待办）：安装 ESA SNAP（https://step.esa.int/main/download/snap-download/），
把 GPT 可执行文件路径传给 --gpt（默认找 C:/Program Files/snap/bin/gpt.exe）。

用法：
    python data_preprocessing/sar_preprocess/run_snap.py \
        --in-dir data/raw/sentinel1 --out-dir data/processed/sar_calibrated \
        --gpt "C:/Program Files/snap/bin/gpt.exe"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from utils.logger import get_logger

logger = get_logger("run_snap")

GRAPH = Path(__file__).parent / "snap_graph.xml"
DEFAULT_GPT = r"C:/Program Files/snap/bin/gpt.exe"


def process_one(gpt: str, src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [gpt, str(GRAPH),
           f"-Pinput={src}", f"-Poutput={dst.with_suffix('')}",
           "-q", "8"]  # 8 线程
    logger.info("处理 %s → %s", src.name, dst.name)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        logger.error("SNAP 处理失败 %s：\n%s", src.name, r.stderr[-2000:])
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpt", default=DEFAULT_GPT)
    args = ap.parse_args()

    if not Path(args.gpt).exists():
        logger.error("未找到 SNAP gpt：%s。请先安装 SNAP 并用 --gpt 指定路径。", args.gpt)
        sys.exit(1)

    in_dir = Path(args.in_dir)
    scenes = sorted(in_dir.glob("*.zip")) + sorted(in_dir.glob("*.SAFE"))
    if not scenes:
        logger.error("在 %s 没有找到 Sentinel-1 产品（.zip/.SAFE）", in_dir)
        sys.exit(1)

    ok = 0
    for s in scenes:
        dst = Path(args.out_dir) / (s.stem + ".tif")
        if dst.exists():
            logger.info("已存在，跳过：%s", dst.name)
            continue
        ok += process_one(args.gpt, s, dst)
    logger.info("完成 %d/%d 景", ok, len(scenes))


if __name__ == "__main__":
    main()
