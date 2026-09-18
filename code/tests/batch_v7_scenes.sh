#!/bin/bash
# 用最终模型 swin_unet_v7_sam 补齐所有已切块场景的推理 + 陆地掩膜。
# 已存在于 results/scene_eval_v7sam/ 的场景自动跳过。
set -u
cd "$(dirname "$0")/.."   # code/

TILES=../data/processed/sar_tiles
OUTROOT=../results/scene_eval_v7sam
CKPT=../checkpoints/final/swin_unet_v7_sam/best.pth
MODELCFG=configs/model_swin_sam.yaml

mkdir -p "$OUTROOT" ../logs
LOG=../logs/batch_v7_scenes_$(date +%Y%m%d_%H%M%S).log

for d in "$TILES"/*.SAFE; do
    scene=$(basename "$d" .SAFE)
    out="$OUTROOT/$scene"
    if [ -f "$out/prob_masked.npy" ]; then
        echo "[skip] $scene 已完成" | tee -a "$LOG"
        continue
    fi
    echo "=== [run] $scene ===" | tee -a "$LOG"
    python tests/run_scene_inference.py \
        --tiles-dir "$d" \
        --ckpt "$CKPT" \
        --model-config "$MODELCFG" \
        --out "$out" >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
        echo "[FAIL] $scene 推理失败，继续下一景" | tee -a "$LOG"
        continue
    fi
    python tests/apply_land_mask.py "$d" "$out" >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
        echo "[FAIL] $scene 陆地掩膜失败" | tee -a "$LOG"
        continue
    fi
    echo "[done] $scene" | tee -a "$LOG"
done
echo "=== 全部完成 ===" | tee -a "$LOG"
