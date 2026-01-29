#!/bin/bash
# 运行长时 AR 滚动预测评估 (Rollout Evaluation)
# 用法: bash scripts/run_rollout_eval.sh <CHECKPOINT_PATH> [STEPS] [GPU_ID]

CKPT=$1
STEPS=${2:-20}
GPU_ID=${3:-0}

if [ -z "$CKPT" ]; then
  echo "Usage: bash scripts/run_rollout_eval.sh <CHECKPOINT_PATH> [STEPS] [GPU_ID]"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================================"
echo "开始运行长时预测评估 (Steps=$STEPS) - GPU $GPU_ID"
echo "Checkpont: $CKPT"
echo "========================================================"

# 使用 train_real_data_ar.py 的测试模式
# 覆盖 data.T_out 和 validation.rollout_steps 以确保评估长度
python tools/training/train_real_data_ar.py \
  --test-only \
  --ckpt "$CKPT" \
  data.T_out=$STEPS \
  validation.rollout_steps="[$STEPS]" \
  testing.num_visualization_samples=5 \
  logging.experiment_name="Eval-Rollout-T${STEPS}"

echo "========================================================"
echo "评估完成！结果请查看 runs/Eval-Rollout-T${STEPS}/test_results.json"
echo "可视化图表请查看 runs/Eval-Rollout-T${STEPS}/test_visualizations"
echo "========================================================"
