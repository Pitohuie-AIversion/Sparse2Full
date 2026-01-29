#!/bin/bash

# CPU 快速验证脚本
# 用于验证 A0 配置下 Train Loss 是否恢复正常（即不包含 Spectral/DC Loss）

PYTHON="python"
TRAIN_SCRIPT="tools/training/train_real_data_ar.py"
BASE_CONFIG="thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml"
OUTPUT_DIR="runs_debug/CPU_Loss_Check"

# 确保输出目录清理干净
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

echo "========================================================"
echo "开始 CPU 验证: A0_Baseline (Rec=1.0, Spec=0.0, DC=0.0)"
echo "目的: 验证 Train Loss 是否大幅下降 (应 < 1.0)"
echo "========================================================"

# 构造命令：强制 CPU，极小数据量
# 关键：training.loss_weights.spectral=0.0
CMD="$PYTHON $TRAIN_SCRIPT --config $BASE_CONFIG \
    device.accelerator=cpu \
    device.devices=1 \
    training.epochs=1 \
    training.batch_size=4 \
    training.checkpoint.save_every_n_epochs=1 \
    data.max_samples=16 \
    data.dataloader.num_workers=0 \
    data.dataloader.batch_size=4 \
    data.dataloader.val_batch_size=4 \
    loss.spectral.weight=0.0 \
    loss.degradation_consistency.weight=0.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=0.0 \
    experiment.name=\"CPU_Debug_A0\" \
    experiment.output_dir=\"$OUTPUT_DIR\" \
    logging.log_model=false"

echo "Command: $CMD"
eval $CMD

RET_VAL=$?

if [ $RET_VAL -eq 0 ]; then
    echo "========================================================"
    echo "✅ CPU 验证完成！"
    echo "请检查日志中的 'Train Loss'。"
    echo "如果修复生效，Train Loss 应该与 Val Loss (Rel-L2) 在同一数量级 (e.g., 0.1 ~ 0.8)。"
    echo "如果仍为 ~23.0，说明参数覆盖依然失败。"
    echo "日志路径: $OUTPUT_DIR/training.log"
    echo "========================================================"
else
    echo "========================================================"
    echo "❌ CPU 验证失败！"
    echo "========================================================"
    exit 1
fi
