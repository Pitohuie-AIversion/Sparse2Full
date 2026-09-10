#!/bin/bash
# 时间稀疏观测实验批量执行脚本
# 非自回归vs自回归对比实验

set -e

# 实验配置
SEEDS=(42 123 456)
PDE_TYPES=("2d_diff_react" "ns_incom_inhom")
TEMPORAL_STRATEGIES=("TS25" "TS50" "TS75")  # 时间采样率
PREDICTION_MODES=("AR" "NAR")  # 自回归 vs 非自回归
T_OUT_VALUES=(1 3 5)  # 预测长度

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$BASE_DIR/paper_package/temporal_sparse_experiments"

echo "⏰ 开始时间稀疏观测实验..."
echo "📊 对比AR vs NAR在不同时间采样率下的性能"

mkdir -p "$RESULTS_DIR"

# 实验执行
for seed in "${SEEDS[@]}"; do
  for pde in "${PDE_TYPES[@]}"; do
    for temporal in "${TEMPORAL_STRATEGIES[@]}"; do
      for mode in "${PREDICTION_MODES[@]}"; do
        for t_out in "${T_OUT_VALUES[@]}"; do
          
          EXP_NAME="${temporal}-${mode}-T${t_out}-${pde}-s${seed}-$(date +%Y%m%d)"
          EXP_DIR="$RESULTS_DIR/$EXP_NAME"
          
          echo "📍 实验: $EXP_NAME"
          
          # 选择配置文件
          if [ "$mode" = "AR" ]; then
            CONFIG_NAME="ar_training_config_temporal_only"
          else
            CONFIG_NAME="temporal_nar_optimized"
          fi
          
          # 时间采样配置
          case $temporal in
            "TS25")  
              SAMPLE_RATIO=0.25
              ;;  
            "TS50")
              SAMPLE_RATIO=0.50
              ;;
            "TS75")
              SAMPLE_RATIO=0.75
              ;;
          esac
          
          # 执行训练
          python "$BASE_DIR/train_real_data_ar.py" \
            --config-path "$BASE_DIR/configs" \
            --config-name "$CONFIG_NAME" \
            data.pde_type="$pde" \
            temporal.T_out="$t_out" \
            data.temporal_sample_ratio="$SAMPLE_RATIO" \
            training.seed="$seed" \
            experiment.name="$EXP_NAME" \
            hydra.run.dir="$EXP_DIR" \
            > "$EXP_DIR/train.log" 2>&1
            
          # 长序列rollout评估（仅对NAR）
          if [ "$mode" = "NAR" ]; then
            python "$BASE_DIR/eval.py" \
              --config-path "$EXP_DIR" \
              --config-name "config_merged" \
              model.eval.rollout_steps=10 \
              hydra.run.dir="$EXP_DIR/eval_long" \
              > "$EXP_DIR/eval_long.log" 2>&1
          fi
          
          echo "✅ 完成: $EXP_NAME"
          
        done
      done
    done
  done
done

echo "📊 生成时间稀疏实验对比报告..."

# 生成对比报告
python "$BASE_DIR/tools/analyze_temporal_comparison.py" \
  --results_dir "$RESULTS_DIR" \
  --output_file "$RESULTS_DIR/temporal_comparison_report.md" \
  --plot_curves true

echo "📄 时间稀疏实验报告: $RESULTS_DIR/temporal_comparison_report.md"
