#!/bin/bash
# round5_census：秋冬春普查场景的 v7 推理 + 陆地掩膜，输出到 results/scene_eval_census/
# （与 batch_v7_scenes.sh 同链路，但只处理 sar_tiles 里属于普查的新场景，
#   且不写 scene_eval_v7sam）。
set -u
cd "$(dirname "$0")/.."   # code/

TILES=../data/processed/sar_tiles
OUTROOT=../results/scene_eval_census
CKPT=../checkpoints/final/swin_unet_v7_sam/best.pth
MODELCFG=configs/model_swin_sam.yaml

# 普查 9 景（秋 2023-10 / 冬 2023-12~2024-01 / 春 2024-03）
SCENES="20231012T104132 20231005T105002 20231007T103348 20240104T104130 20231204T105001 20231206T103347 20240304T104128 20240309T104958 20240311T103344"

mkdir -p "$OUTROOT" ../logs
LOG=../logs/batch_v7_census_$(date +%Y%m%d_%H%M%S).log

for tag in $SCENES; do
    d=$(ls -d "$TILES"/*${tag}*.SAFE 2>/dev/null | head -1)
    if [ -z "$d" ]; then
        echo "[skip] $tag 尚未切块" | tee -a "$LOG"
        continue
    fi
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
echo "=== 普查推理全部完成 ===" | tee -a "$LOG"
