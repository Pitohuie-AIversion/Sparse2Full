#!/bin/bash
# 损失函数权重消融实验脚本
# 验证理论分析：L = L_rec + λ_s L_spec + λ_dc L_dc

set -e

# 损失函数权重配置
LOSS_CONFIGS=(
  "rec=1.0,spec=0.5,dc=1.0"      # 基准配置
  "rec=1.0,spec=0.0,dc=1.0"      # 无频域损失
  "rec=1.0,spec=0.5,dc=0.0"      # 无DC损失  
  "rec=1.0,spec=1.0,dc=1.0"      # 强化频域
  "rec=1.0,spec=0.5,dc=2.0"      # 强化DC
  "rec=2.0,spec=0.5,dc=1.0"      # 强化重建
)

SEEDS=(42 123 456)
PDE_TYPES=("2d_diff_react" "ns_incom_inhom")
SPARSE_TASKS=("sr_x4" "crop_20")  # 选择困难任务

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$BASE_DIR/paper_package/loss_ablation_experiments"

echo "🔬 开始损失函数权重消融实验..."
echo "📊 验证理论: L = L_rec + λ_s L_spec + λ_dc L_dc"

mkdir -p "$RESULTS_DIR"

# 梯度冲突分析函数
analyze_gradient_conflict() {
  local exp_dir=$1
  local config_file=$2
  
  python "$BASE_DIR/tools/analyze_gradient_conflict.py" \
    --config_file "$config_file" \
    --output_dir "$exp_dir" \
    --loss_pairs "spec_dc" \
    --save_plots true
}

# 执行消融实验
for loss_config in "${LOSS_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for pde in "${PDE_TYPES[@]}"; do
      for task in "${SPARSE_TASKS[@]}"; do
        
        # 解析损失权重
        IFS=',' read -ra WEIGHTS <<< "$loss_config"
        rec_weight=$(echo "${WEIGHTS[0]}" | cut -d'=' -f2)
        spec_weight=$(echo "${WEIGHTS[1]}" | cut -d'=' -f2)
        dc_weight=$(echo "${WEIGHTS[2]}" | cut -d'=' -f2)
        
        EXP_NAME="loss_${loss_config//,/\_}-${task}-${pde}-s${seed}-$(date +%Y%m%d)"
        EXP_NAME=${EXP_NAME//./p}  # 替换小数点
        EXP_DIR="$RESULTS_DIR/$EXP_NAME"
        
        echo "📍 损失消融: $loss_config | 任务: $task | PDE: $pde | 种子: $seed"
        
        # 执行训练
        python "$BASE_DIR/train_real_data_ar.py" \
          --config-path "$BASE_DIR/configs" \
          --config-name "train" \
          data.task="$task" \
          data.pde_type="$pde" \
          loss.rec_weight="$rec_weight" \
          loss.spec_weight="$spec_weight" \
          loss.dc_weight="$dc_weight" \
          training.seed="$seed" \
          experiment.name="$EXP_NAME" \
          hydra.run.dir="$EXP_DIR" \
          > "$EXP_DIR/train.log" 2>&1
          
        # 梯度冲突分析
        analyze_gradient_conflict "$EXP_DIR" "$EXP_DIR/config_merged.yaml"
          
        echo "✅ 完成: $EXP_NAME"
        
      done
    done
  done
done

echo "📊 生成损失函数消融分析报告..."

# 理论验证分析
python "$BASE_DIR/tools/validate_loss_theory.py" \
  --results_dir "$RESULTS_DIR" \
  --output_file "$RESULTS_DIR/loss_ablation_theory_validation.md" \
  --validate_gradient_conflict true \
  --validate_information_theory true \
  --plot_weights_vs_performance true

echo "📄 损失函数消融报告: $RESULTS_DIR/loss_ablation_theory_validation.md"
