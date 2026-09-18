#!/usr/bin/env bash
# 春季（2024-04/05）正样本普查 S1 下载循环（幂等可重跑，断点续传）。
# 候选日 = 风筛通过且 CDSE 有景（见 results/scene_eval_poscensus/spring_candidates_2024.json）。
# 必须在项目根目录运行（--out 显式指向 data/raw/sentinel1）。
set -u
cd "$(dirname "$0")/../.." || exit 1   # 项目根目录
PY=python   # 需在运行前激活 environment.yml 创建的 conda 环境

DAYS="2024-04-10 2024-04-11 2024-04-13 2024-04-25 2024-04-26 2024-04-27 2024-04-28 2024-05-26 2024-05-27"

for d in $DAYS; do
    d2=$(date -d "$d +1 day" +%F)
    echo "===== $d ~ $d2 ====="
    "$PY" code/data_preprocessing/download/download_sentinel1.py \
        --aoi 110,19,115,21.5 --start "$d" --end "$d2" \
        --out data/raw/sentinel1
done
echo "ALL DONE"
