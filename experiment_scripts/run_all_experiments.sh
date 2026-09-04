#!/bin/bash
# 主实验执行入口脚本
# 一键执行所有稀疏观测实验

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_SCRIPTS_DIR="$BASE_DIR/experiment_scripts"
PAPER_PACKAGE_DIR="$BASE_DIR/paper_package"

echo "🎯 Sparse2Full 稀疏观测实验完整执行方案"
echo "📋 实验设计遵循论文理论框架："
echo "   - 空间稀疏观测 (SR×2/×4, Crop-20%/40%)"
echo "   - 时间稀疏观测 (TS25/50/75%, AR vs NAR)"
echo "   - 损失函数消融 (L_rec + λ_s L_spec + λ_dc L_dc)"
echo "   - 理论验证 (信息论、Kolmogorov宽度、Lyapunov稳定性)"
echo ""

# 创建实验目录结构
mkdir -p "$PAPER_PACKAGE_DIR"/{spatial_sparse,temporal_sparse,loss_ablation,robustness,theory_validation}

# 执行选项
echo "请选择要执行的实验："
echo "1) 空间稀疏观测实验 (240组实验)"
echo "2) 时间稀疏观测实验 (108组实验)"  
echo "3) 损失函数消融实验 (216组实验)"
echo "4) 鲁棒性分析实验"
echo "5) 理论验证实验"
echo "6) 执行全部实验 (推荐)"
echo "7) 仅生成论文材料包"
echo ""

read -p "请输入选项 (1-7): " choice

case $choice in
  1)
    echo "🚀 执行空间稀疏观测实验..."
    bash "$EXPERIMENT_SCRIPTS_DIR/run_spatial_sparse_experiments.sh"
    ;;
  2)
    echo "⏰ 执行时间稀疏观测实验..."
    bash "$EXPERIMENT_SCRIPTS_DIR/run_temporal_sparse_experiments.sh"
    ;;
  3)
    echo "🔬 执行损失函数消融实验..."
    bash "$EXPERIMENT_SCRIPTS_DIR/run_loss_ablation_experiments.sh"
    ;;
  4)
    echo "🛡️ 执行鲁棒性分析实验..."
    bash "$EXPERIMENT_SCRIPTS_DIR/run_robustness_experiments.sh"
    ;;
  5)
    echo "📐 执行理论验证实验..."
    bash "$EXPERIMENT_SCRIPTS_DIR/run_theory_validation_experiments.sh"
    ;;
  6)
    echo "🎯 执行全部实验 (预计运行时间: 24-48小时)..."
    echo "⏱️  建议: 使用screen/tmux保持会话，或使用nohup后台运行"
    echo ""
    
    read -p "确认执行全部实验? (y/N): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
      echo "🚀 开始完整实验流程..."
      
      # 记录开始时间
      START_TIME=$(date +%s)
      
      # 依次执行各个实验
      bash "$EXPERIMENT_SCRIPTS_DIR/run_spatial_sparse_experiments.sh"
      bash "$EXPERIMENT_SCRIPTS_DIR/run_temporal_sparse_experiments.sh"  
      bash "$EXPERIMENT_SCRIPTS_DIR/run_loss_ablation_experiments.sh"
      bash "$EXPERIMENT_SCRIPTS_DIR/run_robustness_experiments.sh"
      bash "$EXPERIMENT_SCRIPTS_DIR/run_theory_validation_experiments.sh"
      
      # 计算运行时间
      END_TIME=$(date +%s)
      RUNTIME=$((END_TIME - START_TIME))
      HOURS=$((RUNTIME / 3600))
      MINUTES=$(((RUNTIME % 3600) / 60))
      
      echo "🎉 全部实验完成！"
      echo "⏱️  总运行时间: ${HOURS}小时${MINUTES}分钟"
    else
      echo "❌ 取消执行"
      exit 0
    fi
    ;;
  7)
    echo "📦 生成论文材料包..."
    bash "$EXPERIMENT_SCRIPTS_DIR/generate_paper_package.sh"
    ;;
  *)
    echo "❌ 无效选项"
    exit 1
    ;;
esac

# 生成论文材料包
echo ""
echo "📦 开始生成论文材料包..."

# 汇总所有实验结果
python "$BASE_DIR/tools/generate_paper_package.py" \
  --package_dir "$PAPER_PACKAGE_DIR" \
  --paper_title "稀疏观测驱动的时空流场重建方法研究" \
  --include_experiments "spatial_sparse,temporal_sparse,loss_ablation,robustness,theory_validation" \
  --generate_latex_tables true \
  --generate_plots true \
  --create_archive true

echo ""
echo "🎉 实验执行完成！"
echo "📁 论文材料包位置: $PAPER_PACKAGE_DIR"
echo "📊 主要结果文件:"
echo "   - 空间稀疏实验: $PAPER_PACKAGE_DIR/spatial_sparse/spatial_sparse_summary.md"
echo "   - 时间稀疏实验: $PAPER_PACKAGE_DIR/temporal_sparse/temporal_comparison_report.md"
echo "   - 损失函数消融: $PAPER_PACKAGE_DIR/loss_ablation/loss_ablation_theory_validation.md"
echo "   - 完整论文包: $PAPER_PACKAGE_DIR/Sparse2Full_paper_package.zip"
echo ""
echo "📝 下一步:"
echo "   1. 检查实验结果和统计显著性"
echo "   2. 将结果表格插入论文第6章"
echo "   3. 更新理论验证部分(第4-5章)"
echo "   4. 完善实验分析讨论"
