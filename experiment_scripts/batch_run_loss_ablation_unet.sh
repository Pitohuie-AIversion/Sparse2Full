#!/bin/bash

# 硕士论文 Triple Loss 消融实验批量运行脚本
# 该脚本会自动按顺序运行 A0, A2, A3 三组实验，并将结果保存在 runs_3loss_ablation 目录下
# 使用方法: bash experiment_scripts/batch_run_loss_ablation.sh

# 基础配置
PYTHON="torchrun --nproc_per_node=2"
TRAIN_SCRIPT="tools/training/train_real_data_ar.py"
BASE_CONFIG="thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml"
OUTPUT_ROOT="runs_3loss_ablation_unet"

# 确保输出目录存在
mkdir -p $OUTPUT_ROOT

echo "========================================================"
echo "开始运行 Triple Loss 消融实验"
echo "基础配置: $BASE_CONFIG"
echo "输出根目录: $OUTPUT_ROOT"
echo "========================================================"

run_ablation() {
    EXP_ID=$1
    SPEC_W=$2
    DC_W=$3
    DESC=$4
    
    EXP_NAME="3loss_ablation/${EXP_ID}"
    
    echo "--------------------------------------------------------"
    echo "正在运行实验: $EXP_NAME ($DESC)"
    echo "参数: Spectral=$SPEC_W, DC=$DC_W"
    
    # 构造完整命令
    # 注意：我们通过 experiment.output_dir 重定向输出目录，使其归档到 runs_3loss_ablation_unet 下
    CMD="$PYTHON $TRAIN_SCRIPT --config $BASE_CONFIG \
        model.name=unet \
        training.epochs=50 \
        training.checkpoint.save_every_n_epochs=10 \
        training.loss_weights.spectral=$SPEC_W \
        training.loss_weights.data_consistency=$DC_W \
        experiment.name=\"$EXP_NAME\" \
        experiment.output_dir=\"$OUTPUT_ROOT/$EXP_ID\" \
        logging.log_model=true"
        
    echo "Command: $CMD"
    
    # 执行命令
    # 使用 eval 执行以正确处理引号
    eval $CMD
    
    RET_VAL=$?
    
    if [ $RET_VAL -eq 0 ]; then
        echo "实验 $EXP_NAME 完成!"
    else
        echo "实验 $EXP_NAME 失败! 请检查日志。"
        exit 1
    fi
}

# ========================================================
# 1. Ablation A0: Baseline (MSE Only)
# ========================================================
# 仅使用重建损失 (Rel-L2 + MAE)，辅助损失权重为 0
run_ablation "A0_Baseline" "0.0" "0.0" "Baseline: MSE Only"

# ========================================================
# 2. Ablation A2: Rec + Spectral
# ========================================================
# 引入频域损失，验证对高频纹理的改善
# NOTE: 降低 Spectral 权重 (0.5 -> 0.05) 以提高训练稳定性
run_ablation "A2_RecSpec" "0.05" "0.0" "Ablation: Rec + Spectral Loss"

# ========================================================
# 3. Ablation A3: Full (Ours)
# ========================================================
# 引入数据一致性损失 (DC)，完整的三重损失
# NOTE: 降低 Spectral (0.5->0.05) 和 DC (1.0->0.1) 权重以提高稳定性
run_ablation "A3_Full" "0.05" "0.1" "Ours: Full Triple Loss"

echo "========================================================"
echo "所有消融实验已完成！"
echo "结果已保存在: $OUTPUT_ROOT"
echo "请查看各实验目录下的 metrics.csv 填写论文表格。"
echo "========================================================"
