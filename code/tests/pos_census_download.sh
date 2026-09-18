#!/bin/bash
# 正样本普查（round6）下载：15 候选日 3 轮续传循环。
# 候选日 = 丛集 AOI(110~115E,19~21.5N) × 大潮窗口 × ERA5 风筛 [2,10] m/s × CDSE 可获取性
# （筛选过程见工作进度记录第 9 天续三；下载脚本自身按文件大小跳过完整文件/断点续传）
set -u
cd "$(dirname "$0")/.."   # code/
PY=${PY:-python}
DAYS="2023-11-17 2023-11-29 2023-12-18 2023-12-28 2023-12-30 \
2024-01-16 2024-01-28 2024-01-30 2024-02-11 2024-02-14 \
2024-02-26 2024-02-28 2024-03-11 2024-03-28 2024-03-30"
for pass in 1 2 3; do
  echo "===== PASS $pass $(date) ====="
  for d in $DAYS; do
    end=$(date -d "$d +1 day" +%F)
    echo "--- day $d (pass $pass) $(date) ---"
    for try in 1 2 3; do
      timeout 1800 $PY data_preprocessing/download/download_sentinel1.py \
        --aoi 110,19,115,21.5 --start "$d" --end "$end" \
        --out ../data/raw/sentinel1 && break
      echo "retry $try for $d after failure/timeout"; sleep 20
    done
  done
done
echo "===== ALL PASSES DONE $(date) ====="
