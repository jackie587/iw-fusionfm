#!/bin/bash
# 定向配对场景全链路：预处理 → v7 推理+陆地掩膜 → L2 配对块构建。
# 依赖 targeted_pairs.json 中的 scene↔granule 映射。
set -u
cd "$(dirname "$0")/.."   # code/

RAW=../data/raw/sentinel1
TILES=../data/processed/sar_tiles
OUTROOT=../results/scene_eval_v7sam
SWOT=../data/raw/swot_unsmoothed
CKPT=../checkpoints/final/swin_unet_v7_sam/best.pth
MODELCFG=configs/model_swin_sam.yaml

# scene 日期片段:unsmoothed granule 文件名片段
PAIRS="
20230801:001_312
20230820:002_271
20230823:002_340
20230827:002_465
20230830:002_549
"

for pair in $PAIRS; do
    day="${pair%%:*}"; pass="${pair##*:}"
    granule=$(ls "$SWOT"/*${pass}*.nc 2>/dev/null | head -1)
    if [ -z "$granule" ]; then echo "[skip] $day 无 granule $pass"; continue; fi
    # 1. 预处理：raw 还在且尚未切块的场景先切块
    #   （raw 可能是 .SAFE 目录或 .zip；tiles 目录名带 .SAFE 后缀）
    for d in "$RAW"/S1A_IW_GRDH_1SDV_${day}T*.SAFE "$RAW"/S1A_IW_GRDH_1SDV_${day}T*.zip; do
        [ -e "$d" ] || continue
        base=$(basename "$d"); base="${base%.zip}"
        if [ ! -d "$TILES/$base" ]; then
            python data_preprocessing/sar_preprocess/preprocess_safe.py \
                --safe "$d" --out "$TILES" 2>&1 | tail -1
        fi
    done
    # 2-3. 对当日所有已切块场景做推理 + L2 配对块
    #     （raw 清理后 tiles 仍在，必须遍历 tiles 目录而不是 raw）
    for td in "$TILES"/S1A_IW_GRDH_1SDV_${day}T*.SAFE; do
        [ -d "$td" ] || continue
        scene=$(basename "$td" .SAFE)
        echo "=== $scene × $(basename "$granule" .nc | cut -c18-47) ==="
        # 2. v7 推理 + 陆地掩膜（弱标签）
        out="$OUTROOT/$scene"
        if [ ! -f "$out/prob_masked.npy" ]; then
            python tests/run_scene_inference.py --tiles-dir "$td" \
                --ckpt "$CKPT" --model-config "$MODELCFG" \
                --out "$out" 2>&1 | tail -1
            python tests/apply_land_mask.py "$td" "$out" 2>&1 | tail -1
        else
            echo "[skip] 已推理"
        fi
        if [ ! -f "$out/prob_masked.npy" ]; then
            echo "[FAIL] $scene 推理/掩膜未产出，跳过 L2"; continue
        fi
        # 3. L2 配对块（含 v7 弱标签）
        python data_preprocessing/matchup/build_l2_dataset.py \
            --s1-tiles "$td" --swot "$granule" \
            --prob "$out/prob_masked.npy" 2>&1 | tail -1
    done
done
echo "=== PAIR_PIPELINE_DONE ==="
ls ../data/datasets/L2_s1_swot_matched/tiles/*.npz | wc -l
