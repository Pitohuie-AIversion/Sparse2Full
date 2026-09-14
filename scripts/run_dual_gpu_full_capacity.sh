#!/bin/bash
# ==============================================================================
# SWVT (Video Swin Transformer) 双卡 32GB 满卡训练流水线
# GPU 0: 方案 A (Tin=4 -> Tout=10) 基准稳健型
# GPU 1: 方案 C (Tin=8 -> Tout=16) 深时空长程型
# 纯全场高分辨率 128x128 物理真值输入输出，无任何稀疏退化
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================================="
echo "🚀 启动 SWVT 双卡满载并行实验"
echo "工作目录: $PROJECT_ROOT"
echo "=================================================================="

# 激活 Python 环境 (如果需要)
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# 创建运行目录
mkdir -p runs/SWVT_Tin4_Tout10_GPU0
mkdir -p runs/SWVT_Tin8_Tout16_GPU1

echo "1️⃣  启动 GPU 0 任务: 方案 A (Tin=4 -> Tout=10) [断点恢复模式]..."
CUDA_VISIBLE_DEVICES=0 python train_temporal.py \
    --config-name swvt_tin4_tout10 \
    experiment.name=SWVT_Tin4_Tout10_GPU0 \
    experiment.device=cuda:0 \
    experiment.resume=true \
    >> runs/SWVT_Tin4_Tout10_GPU0/console.log 2>&1 &
PID_GPU0=$!
echo "   ✅ GPU 0 进程 PID: $PID_GPU0"

echo "2️⃣  启动 GPU 1 任务: 方案 C (Tin=8 -> Tout=16)..."
CUDA_VISIBLE_DEVICES=1 python train_temporal.py \
    --config-name swvt_tin8_tout16 \
    experiment.name=SWVT_Tin8_Tout16_GPU1 \
    experiment.device=cuda:0 \
    experiment.resume=true \
    >> runs/SWVT_Tin8_Tout16_GPU1/console.log 2>&1 &
PID_GPU1=$!
echo "   ✅ GPU 1 进程 PID: $PID_GPU1"

echo "=================================================================="
echo "🎯 双卡训练均已在后台启动！"
echo "GPU 0 监控: tail -f runs/SWVT_Tin4_Tout10_GPU0/train.log"
echo "GPU 1 监控: tail -f runs/SWVT_Tin8_Tout16_GPU1/train.log"
echo "=================================================================="

# 保存 PID
echo "$PID_GPU0" > runs/SWVT_Tin4_Tout10_GPU0/process.pid
echo "$PID_GPU1" > runs/SWVT_Tin8_Tout16_GPU1/process.pid
