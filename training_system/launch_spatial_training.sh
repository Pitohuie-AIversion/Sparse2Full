#!/bin/bash
# 纯空间预测训练启动脚本
# Usage: ./launch_spatial_training.sh [experiment_name] [config_file]

set -e  # 出错时退出

# 默认参数
EXPERIMENT_NAME=${1:-"spatial_sr4_experiment"}
CONFIG_FILE=${2:-"../configs/spatial/spatial_sr4_config.yaml"}
SEED=${3:-2025}

# 设置环境
export PYTHONPATH=/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full:$PYTHONPATH
export PYTHONHASHSEED=0
export CUDA_LAUNCH_BLOCKING=1  # 更好的错误信息

# 创建输出目录
OUTPUT_DIR="runs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 记录实验信息
echo "=== 纯空间预测训练启动 ==="
echo "实验名称: $EXPERIMENT_NAME"
echo "配置文件: $CONFIG_FILE"
echo "随机种子: $SEED"
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date)"
echo "=========================="

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 记录环境信息
{
    echo "环境信息:"
    echo "Python版本: $(python --version)"
    echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "CUDA版本: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
    echo "Git提交: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
    echo "Git分支: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
    echo "工作目录: $(pwd)"
} > "$OUTPUT_DIR/experiment_info.txt"

# 检查Python环境
echo "检查Python环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "警告: Python环境检查失败，继续执行..."
fi

# 运行训练
echo "开始纯空间预测训练..."
cd /share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/training_system

# 使用训练系统的主训练脚本
python scripts/train.py \
    --config-path=$(dirname "$CONFIG_FILE") \
    --config-name=$(basename "$CONFIG_FILE" .yaml) \
    experiment.name="$EXPERIMENT_NAME" \
    experiment.seed=$SEED \
    experiment.output_dir="$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "✅ 纯空间预测训练完成成功！"
    echo "结果保存在: $OUTPUT_DIR"
    
    # 生成性能摘要
    if [ -f "$OUTPUT_DIR/paper_package/metrics/summary.csv" ]; then
        echo "性能摘要:"
        cat "$OUTPUT_DIR/paper_package/metrics/summary.csv"
    fi
    
    # 创建完成标记
    echo "$(date): 纯空间预测训练成功完成" > "$OUTPUT_DIR/COMPLETED"
    echo "最佳验证指标: $(cat "$OUTPUT_DIR/best_metric.txt" 2>/dev/null || echo 'N/A')" >> "$OUTPUT_DIR/COMPLETED"
    
else
    echo "❌ 纯空间预测训练失败！检查日志: $OUTPUT_DIR/training.log"
    echo "$(date): 训练失败" > "$OUTPUT_DIR/FAILED"
    exit 1
fi

# 显示关键结果
echo ""
echo "=== 训练结果摘要 ==="
echo "实验名称: $EXPERIMENT_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "配置文件: $CONFIG_FILE"

if [ -f "$OUTPUT_DIR/metrics/final_metrics.json" ]; then
    echo "最终指标:"
    cat "$OUTPUT_DIR/metrics/final_metrics.json" | python -m json.tool
fi

echo ""
echo "=== 纯空间预测训练完成 ==="
echo "查看详细结果: ls -la $OUTPUT_DIR"
echo "查看训练日志: cat $OUTPUT_DIR/training.log"
echo "查看可视化: ls -la $OUTPUT_DIR/visualizations/"
echo "生成论文包: python tools/generate_paper_package.py --run-dir=$OUTPUT_DIR"