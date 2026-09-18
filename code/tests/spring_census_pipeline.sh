#!/bin/bash
# 春季普查（2023/2024 年 4~5 月）滚动管线：对春季普查下载的 15 景
# 逐景 预处理切块 → v7 推理 → 陆地掩膜 → 判读图（阈值 0.35 收网，抽样 200）。
# 幂等：已有 tiles/prob/掩膜的步骤自动跳过，可反复重跑直至全部场景完成。
# 用法: bash tests/spring_census_pipeline.sh [场景名过滤子串]
set -u
cd "$(dirname "$0")/.."   # code/
PY=${PY:-python}
RAW=../data/raw/sentinel1
TILES=../data/processed/sar_tiles
OUTROOT=../results/scene_eval_springcensus
CKPT=../checkpoints/final/swin_unet_v7_sam/best.pth
MODELCFG=configs/model_swin_sam.yaml
THRESH=0.35
N=200
# 默认只处理春季普查批次的 15 景（按过境日期匹配）；传参可覆盖
FILTER=${1:-"20240411|20240414|20240426|20240428|20240527|20230422|20230509|20230511|20230521|20230523"}

mkdir -p "$OUTROOT" ../logs
LOG=../logs/spring_census_pipeline_$(date +%Y%m%d_%H%M%S).log

for safe in "$RAW"/*.SAFE; do
    [ -e "$safe" ] || continue   # 可能是 .SAFE 目录，也可能是同名 zip 文件（read_safe 两者皆可）
    scene=$(basename "$safe" .SAFE)
    echo "$scene" | grep -qE "$FILTER" || continue
    tdir="$TILES/$scene.SAFE"
    out="$OUTROOT/$scene"
    echo "=== [scene] $scene $(date) ===" >> "$LOG"
    if [ ! -d "$tdir" ]; then
        echo "[preprocess] $scene" >> "$LOG"
        $PY data_preprocessing/sar_preprocess/preprocess_safe.py \
            --safe "$safe" --out "$TILES" >> "$LOG" 2>&1
        [ $? -ne 0 ] && { echo "[FAIL] preprocess $scene" >> "$LOG"; continue; }
    else
        echo "[skip] tiles 已存在" >> "$LOG"
    fi
    if [ ! -f "$out/prob.npy" ]; then
        echo "[inference] $scene" >> "$LOG"
        $PY tests/run_scene_inference.py --tiles-dir "$tdir" \
            --ckpt "$CKPT" --model-config "$MODELCFG" --out "$out" >> "$LOG" 2>&1
        [ $? -ne 0 ] && { echo "[FAIL] inference $scene" >> "$LOG"; continue; }
    fi
    if [ ! -f "$out/prob_masked.npy" ]; then
        echo "[landmask] $scene" >> "$LOG"
        $PY tests/apply_land_mask.py "$tdir" "$out" >> "$LOG" 2>&1
        [ $? -ne 0 ] && { echo "[FAIL] landmask $scene" >> "$LOG"; continue; }
    fi
    if [ ! -f "$out/review/components.json" ]; then
        echo "[review] $scene @$THRESH" >> "$LOG"
        $PY tests/make_review_sheets.py --tiles-dir "$tdir" --scene-out "$out" \
            --threshold "$THRESH" --n "$N" >> "$LOG" 2>&1
    fi
    echo "[done] $scene" >> "$LOG"
done
echo "=== 滚动管线本轮完成 $(date) ===" >> "$LOG"
