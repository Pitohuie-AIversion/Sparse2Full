#!/bin/bash
# 逐个模型调试训练脚本
# 基于100epoch配置，依次运行不同模型进行调试

set -e  # 遇到错误立即退出

# 基础配置
CONFIG_BASE="configs/train/ar_training_config debug copy.yaml"
LOG_DIR="runs/model_debug_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建日志目录
mkdir -p "$LOG_DIR"

# 模型调试顺序（从轻量级到复杂）
MODELS=(
    "UNet:configs/model/unet.yaml"
    "SwinUNet:configs/model/swin_unet.yaml" 
    "FNO2d:configs/model/fno2d.yaml"
    "UFNOUNet:configs/model/ufno_unet.yaml"
    "SegFormer:configs/model/segformer.yaml"
    "UNetFormer:configs/model/unetformer.yaml"
    "MLPMixer:configs/model/mlp_mixer.yaml"
    "MLP:configs/model/mlp.yaml"
    "Transformer:configs/model/transformer.yaml"
    "ViT:configs/model/vit.yaml"
    "Hybrid:configs/model/hybrid.yaml"
)

echo "🚀 开始逐个模型调试训练..."
echo "📊 总计 ${#MODELS[@]} 个模型待测试"
echo "⏱️  预计总时长: ~$(( ${#MODELS[@]} * 15 )) 分钟"
echo ""

# 循环运行每个模型
for i in "${!MODELS[@]}"; do
    IFS=':' read -r MODEL_NAME MODEL_CONFIG <<< "${MODELS[$i]}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔧 模型 $((i+1))/${#MODELS[@]}: $MODEL_NAME"
    echo "📁 配置: $MODEL_CONFIG"
    echo "⏰ 开始时间: $(date)"
    echo ""
    
    # 创建临时配置文件
    TEMP_CONFIG="temp_${MODEL_NAME}_${TIMESTAMP}.yaml"
    
    # 合并基础配置和模型特定配置
    python -c "
import yaml
import sys
from pathlib import Path

# 读取基础配置
with open('$CONFIG_BASE', 'r') as f:
    base_config = yaml.safe_load(f)

# 读取模型配置
with open('$MODEL_CONFIG', 'r') as f:
    model_config = yaml.safe_load(f)

# 更新模型名称和参数
base_config['model']['name'] = '$MODEL_NAME'
if 'model' in model_config:
    # 合并模型特定参数
    for key, value in model_config['model'].items():
        if key != 'name':  # 不覆盖名称
            base_config['model'][key] = value

# 更新实验名称
base_config['experiment']['name'] = f'AR-DR2D-Debug100-${MODEL_NAME}-s2025-${TIMESTAMP}'
base_config['experiment']['description'] = f'100epoch调试 - ${MODEL_NAME}'

# 保存临时配置
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(base_config, f, default_flow_style=False)

print(f'✅ 临时配置已创建: $TEMP_CONFIG')
"
    
    # 运行训练
    LOG_FILE="$LOG_DIR/${MODEL_NAME}_${TIMESTAMP}.log"
    echo "📝 日志文件: $LOG_FILE"
    echo ""
    
    # 执行训练（带错误处理）
    if python tools/training/train_real_data_ar.py --config "$TEMP_CONFIG" > "$LOG_FILE" 2>&1; then
        echo "✅ $MODEL_NAME 训练成功！"
        
        # 提取关键指标
        echo "📊 训练结果摘要:"
        tail -n 20 "$LOG_FILE" | grep -E "(Epoch|Train Loss|Val Loss|Best|Time)" | tail -n 5
        
    else
        echo "❌ $MODEL_NAME 训练失败！"
        echo "📋 错误信息:"
        tail -n 10 "$LOG_FILE"
        
        # 询问是否继续
        echo ""
        read -p "是否继续下一个模型? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "🛑 调试已停止"
            break
        fi
    fi
    
    # 清理临时配置
    rm -f "$TEMP_CONFIG"
    
    echo ""
    echo "⏰ 结束时间: $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # 可选：短暂暂停，避免GPU过热
    sleep 30
    
done

echo ""
echo "🎉 模型调试训练完成！"
echo "📁 日志文件保存在: $LOG_DIR"
echo "📊 可通过以下命令查看所有结果:"
echo "   ls -la $LOG_DIR"
echo ""

# 生成汇总报告
echo "生成汇总报告中..."
cat > "$LOG_DIR/summary_${TIMESTAMP}.txt" << EOF
模型调试训练汇总报告
========================
时间: $(date)
总模型数: ${#MODELS[@]}

各模型结果:
EOF

for log_file in "$LOG_DIR"/*_${TIMESTAMP}.log; do
    if [ -f "$log_file" ]; then
        model_name=$(basename "$log_file" | cut -d'_' -f1)
        echo "模型: $model_name" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"
        
        # 提取最终结果
        if tail -n 50 "$log_file" | grep -q "训练成功"; then
            echo "状态: ✅ 成功" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"
            final_result=$(tail -n 20 "$log_file" | grep "Epoch" | tail -n 1)
            echo "最终结果: $final_result" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"
        else
            echo "状态: ❌ 失败" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"
        fi
        echo "" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"
    fi
done

echo "✅ 汇总报告已生成: $LOG_DIR/summary_${TIMESTAMP}.txt"