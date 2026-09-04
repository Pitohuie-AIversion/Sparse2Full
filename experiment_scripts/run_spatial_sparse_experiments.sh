#!/bin/bash
# 空间稀疏观测实验批量执行脚本
# 遵循黄金法则：一致性、可重现性、统计显著性

set -e  # 出错即停止

# 实验配置
SEEDS=(42 123 456 789 999)  # 5重随机种子
PDE_TYPES=("2d_diff_react" "ns_incom_inhom" "2d_rdb")
SPARSE_TASKS=("sr_x2" "sr_x4" "crop_20" "crop_40")
MODELS=("swin_unet" "unet" "fno2d" "segformer")

# 基础路径
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$BASE_DIR/paper_package/sparse_observation_experiments"

echo "🚀 开始空间稀疏观测实验..."
echo "📊 实验规模: ${#SEEDS[@]}种子 × ${#PDE_TYPES[@]}PDE × ${#SPARSE_TASKS[@]}任务 × ${#MODELS[@]}模型"
echo "📁 结果目录: $RESULTS_DIR"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 实验计数器
TOTAL_EXPS=$((${#SEEDS[@]} * ${#PDE_TYPES[@]} * ${#SPARSE_TASKS[@]} * ${#MODELS[@]}))
CURRENT_EXP=0

# 批量执行实验
for seed in "${SEEDS[@]}"; do
  for pde in "${PDE_TYPES[@]}"; do
    for task in "${SPARSE_TASKS[@]}"; do
      for model in "${MODELS[@]}"; do
        CURRENT_EXP=$((CURRENT_EXP + 1))
        echo "📍 实验 $CURRENT_EXP/$TOTAL_EXPS: seed=$seed, pde=$pde, task=$task, model=$model"
        
        # 实验名称
        EXP_NAME="${task}-${pde}-256-${model}-s${seed}-$(date +%Y%m%d)"
        EXP_DIR="$RESULTS_DIR/$EXP_NAME"
        
        # 执行训练
        python "$BASE_DIR/train_real_data_ar.py" \
          --config-path "$BASE_DIR/configs" \
          --config-name "train" \
          data.task="$task" \
          data.pde_type="$pde" \
          model.name="$model" \
          training.seed="$seed" \
          experiment.name="$EXP_NAME" \
          hydra.run.dir="$EXP_DIR" \
          > "$EXP_DIR/train.log" 2>&1
        
        # 执行评估
        python "$BASE_DIR/eval.py" \
          --config-path "$EXP_DIR" \
          --config-name "config_merged" \
          hydra.run.dir="$EXP_DIR/eval" \
          > "$EXP_DIR/eval.log" 2>&1
          
        echo "✅ 完成: $EXP_NAME"
        echo ""
      done
    done
  done
done

echo "🎉 所有空间稀疏观测实验完成！"
echo "📊 开始汇总结果..."

# 结果汇总
python "$BASE_DIR/tools/summarize_runs.py" \
  --results_dir "$RESULTS_DIR" \
  --output_file "$RESULTS_DIR/spatial_sparse_summary.md" \
  --latex_table "$RESULTS_DIR/spatial_sparse_table.tex"

echo "📄 结果汇总完成: $RESULTS_DIR/spatial_sparse_summary.md"
