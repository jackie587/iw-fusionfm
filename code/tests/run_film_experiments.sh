#!/bin/bash
# SWOT-FiLM 实验链看护脚本（独立于会话存活）：
# 1. 等 swin_film_l2_v1 训练完成（last.pth 出现）；
# 2. 若 SAR-only 对照未归档且当前无任何 train_baseline 进程（说明原
#    后台 && 链已随会话消亡），则补拉 swin_film_l2_v1_saronly 训练；
# 3. 两组都归档后自动跑对比评估，结果写 results/film_eval_output.txt。
# 幂等：任何一步已完成则跳过；用 mkdir 锁防并发重复拉起。
#
# 拉起方式：setsid nohup bash tests/run_film_experiments.sh > logs/film_chain.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
EXP1_DIR=../checkpoints/experiments/swin_film_l2_v1
EXP2_DIR=../checkpoints/experiments/swin_film_l2_v1_saronly
LOCK=../checkpoints/experiments/.film_chain.lock
EVAL_OUT=../results/film_eval_output.txt

# Git Bash 无 pgrep，用 ps+grep 判断训练进程
train_running() {
    ps -ef | grep -i "[t]rain_baseline\.py" >/dev/null 2>&1
}

echo "[chain] $(date) 看护启动"

# 1. 等实验一完成（最多 4 小时）
for i in $(seq 1 240); do
    [ -f "$EXP1_DIR/last.pth" ] && break
    sleep 60
done
if [ ! -f "$EXP1_DIR/last.pth" ]; then
    echo "[chain] $(date) 实验一 4 小时内未完成，退出"
    exit 1
fi
echo "[chain] $(date) 实验一已归档"

# 2. SAR-only 对照：已归档则跳过；等 10 分钟让原 && 链自己拉起；
#    超时仍无训练进程则说明原链已死，由本脚本补拉
if [ ! -f "$EXP2_DIR/last.pth" ]; then
    for i in $(seq 1 10); do
        train_running && break
        sleep 60
    done
    if ! train_running; then
        if mkdir "$LOCK" 2>/dev/null; then
            echo "[chain] $(date) 原链已死，补拉 SAR-only 对照训练"
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate iw
            python training/scripts/train_baseline.py \
                --config configs/train_swin_film_saronly.yaml
            rmdir "$LOCK"
        else
            echo "[chain] $(date) 锁被占用（另一个看护进程在跑），退出"
            exit 0
        fi
    else
        echo "[chain] $(date) 原链已拉起 SAR-only，继续看护"
    fi
fi

# 3. 等 SAR-only 完成（最多 4 小时）
for i in $(seq 1 240); do
    [ -f "$EXP2_DIR/last.pth" ] && break
    sleep 60
done
if [ ! -f "$EXP2_DIR/last.pth" ]; then
    echo "[chain] $(date) SAR-only 4 小时内未完成，退出"
    exit 1
fi
echo "[chain] $(date) SAR-only 已归档"

# 4. 对比评估
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iw
python evaluation/eval_film_experiments.py > "$EVAL_OUT" 2>&1
echo "[chain] $(date) 评估完成 → $EVAL_OUT"
