#!/bin/bash

# 三损失配置 1-Epoch 冒烟测试脚本
# 用于快速验证 A3_Full (Rec + Spec + DC) 配置是否稳定

PYTHON="torchrun --nproc_per_node=2"
TRAIN_SCRIPT="tools/training/train_real_data_ar.py"
BASE_CONFIG="thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml"
OUTPUT_DIR="runs_3loss_ablation/SmokeTest_A3"

# 确保输出目录清理干净
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

echo "========================================================"
echo "开始运行 A3_Full 冒烟测试 (1 Epoch)"
echo "目的: 验证 Rec=1.0, Spec=0.05, DC=0.1 的稳定性"
echo "========================================================"

# 构造命令：强制 1 epoch，启用详细日志
# 注意：我们使用新的安全权重 0.05 / 0.1
# 指定随机端口以避免地址冲突
RANDOM_PORT=$((RANDOM % 1000 + 20000))
CMD="$PYTHON --master_port=$RANDOM_PORT $TRAIN_SCRIPT --config $BASE_CONFIG \
    training.epochs=1 \
    training.checkpoint.save_every_n_epochs=1 \
    loss.spectral.weight=0.05 \
    loss.degradation_consistency.weight=0.1 \
    training.loss_weights.spectral=0.05 \
    training.loss_weights.data_consistency=0.1 \
    experiment.name=\"SmokeTest_A3_Full\" \
    experiment.output_dir=\"$OUTPUT_DIR\" \
    logging.log_model=true"

echo "Command: $CMD"
eval $CMD

RET_VAL=$?

if [ $RET_VAL -eq 0 ]; then
    echo "========================================================"
    echo "✅ 冒烟测试成功完成！"
    echo "请检查以下日志文件确认 Loss 数值是否正常:"
    echo "  - $OUTPUT_DIR/training.log"
    echo "  - $OUTPUT_DIR/tensorboard/"
    echo "========================================================"
else
    echo "========================================================"
    echo "❌ 冒烟测试失败！"
    echo "请立即检查日志定位崩溃原因。"
    echo "========================================================"
    exit 1
fi
