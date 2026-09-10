#!/bin/bash

# 硕士论文实验自动化脚本
# 该脚本涵盖了论文所需的核心实验：主性能、消融实验（损失函数、训练策略）
# 使用方法: bash experiment_scripts/run_thesis_experiments.sh [gpu_id]

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 基础设置
PYTHON="python"
TRAIN_SCRIPT="tools/training/train_real_data_ar.py"
BASE_CONFIG="config_name=train_real_data_ar_config" 
# 注意：假设 train_real_data_ar.py 默认加载 train_real_data_ar_config.yaml 或类似配置，
# 如果需要指定 config path，请根据实际 hydra 配置调整。
# 这里假设脚本内部已经有了默认配置或者通过 config_name 指定。
# 根据之前 grep 结果，脚本似乎使用 hydra 或自定义配置加载器。
# 我们通过传递参数覆盖来控制。

echo "========================================================"
echo "开始运行硕士论文实验 (GPU: $GPU_ID)"
echo "日期: $(date)"
echo "========================================================"

# 创建日志目录
LOG_DIR="runs/thesis_experiments"
mkdir -p $LOG_DIR

run_exp() {
    EXP_NAME=$1
    ARGS=$2
    echo "--------------------------------------------------------"
    echo "正在运行实验: $EXP_NAME"
    echo "参数: $ARGS"
    
    # 构造完整命令
    CMD="$PYTHON $TRAIN_SCRIPT experiment_name=$EXP_NAME $ARGS"
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 执行命令 (这里使用 nohup 或者直接执行，为了演示直接执行)
    # 建议在实际运行时使用 nohup 并重定向日志
    echo "Command: $CMD"
    $CMD > "$LOG_DIR/${EXP_NAME}.log" 2>&1
    
    RET_VAL=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $RET_VAL -eq 0 ]; then
        echo "实验 $EXP_NAME 完成! 耗时: ${DURATION}s"
    else
        echo "实验 $EXP_NAME 失败! 查看日志: $LOG_DIR/${EXP_NAME}.log"
    fi
}

# ========================================================
# 1. 主实验 (Main Performance) - Proposed Method
# ========================================================
# 使用完整的三重损失 + 课程学习 + 统一观测算子
echo ">>> 阶段 1: 主性能对比 (Ours)"
run_exp "thesis_main_ours" \
    "loss.spectral.weight=0.1 loss.data_consistency.weight=1.0 training.curriculum.enabled=true"

# ========================================================
# 2. 消融实验 - 损失函数 (Ablation: Loss Functions)
# ========================================================
echo ">>> 阶段 2: 损失函数消融"

# 2.1 w/o Spectral Loss (无频域损失)
run_exp "thesis_ablation_no_spec" \
    "loss.spectral.weight=0.0 loss.data_consistency.weight=1.0 training.curriculum.enabled=true"

# 2.2 w/o DC Loss (无数据一致性损失)
run_exp "thesis_ablation_no_dc" \
    "loss.spectral.weight=0.1 loss.data_consistency.weight=0.0 training.curriculum.enabled=true"

# 2.3 Only Reconstruction (仅重建损失)
run_exp "thesis_ablation_only_rec" \
    "loss.spectral.weight=0.0 loss.data_consistency.weight=0.0 training.curriculum.enabled=true"

# ========================================================
# 3. 消融实验 - 训练策略 (Ablation: Training Strategy)
# ========================================================
echo ">>> 阶段 3: 训练策略消融"

# 3.1 Direct Joint Training (无课程学习，直接训练)
# 注意：可能需要调整 batch size 或 learning rate 以防不收敛，这里保持其他参数一致以控制变量
run_exp "thesis_ablation_direct_train" \
    "loss.spectral.weight=0.1 loss.data_consistency.weight=1.0 training.curriculum.enabled=false"

echo "========================================================"
echo "所有实验计划已提交执行。"
echo "请检查 $LOG_DIR 目录下的日志文件以监控进度。"
echo "========================================================"
