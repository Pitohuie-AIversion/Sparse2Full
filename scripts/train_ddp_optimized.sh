#!/usr/bin/env bash

set -euo pipefail

# 使用方法：
#   ./scripts/train_ddp_optimized.sh [CONFIG_PATH] [SEEDS]
# 示例：
#   ./scripts/train_ddp_optimized.sh "configs/ar_training_config debug.yaml" "2025,2026,2027"

CONFIG_PATH=${1:-"configs/ar_training_config debug.yaml"}
SEEDS=${2:-"2025"}

echo "[INFO] 使用配置文件: ${CONFIG_PATH}"
echo "[INFO] 使用种子: ${SEEDS}"

# 建议设置可见GPU（如需要）：
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}

# 运行双GPU DDP训练
torchrun --nproc_per_node 2 tools/training/train_real_data_ar.py \
  --config "${CONFIG_PATH}" \
  --seeds "${SEEDS}"

echo "[INFO] 训练进程已启动（DDP，2 GPUs）。"