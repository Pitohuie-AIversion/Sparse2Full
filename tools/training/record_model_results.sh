#!/bin/bash
# 模型调试训练记录脚本

echo "📊 模型调试训练记录"
echo "===================="
echo ""

# 创建记录文件
RECORD_FILE="runs/model_debug_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p runs

cat > "$RECORD_FILE" << 'EOF'
模型调试训练汇总报告
====================
时间: $(date)

EOF

# 函数：提取模型训练结果
extract_model_results() {
    local model_name=$1
    local pattern="*Debug100*${model_name}*"
    
    echo "🔍 正在查找 $model_name 的训练结果..."
    
    # 查找最新的训练目录
    local latest_dir=$(find runs/ -name "$pattern" -type d 2>/dev/null | sort | tail -1)
    
    if [ -z "$latest_dir" ]; then
        echo "❌ 未找到 $model_name 的训练目录"
        return 1
    fi
    
    echo "📁 训练目录: $latest_dir"
    
    # 检查是否训练完成
    if [ -f "$latest_dir"/training.log ]; then
        # 提取最终结果
        local final_epoch=$(grep -E "Epoch.*100|Training finished" "$latest_dir"/training.log | tail -1)
        local best_val=$(grep -E "最佳验证损失|Best.*Val Loss" "$latest_dir"/training.log | tail -1)
        local total_time=$(grep -E "total time|训练时间" "$latest_dir"/training.log | tail -1)
        
        echo "📈 训练结果:"
        echo "  - 最终状态: $final_epoch"
        echo "  - 最佳验证: $best_val"
        echo "  - 总耗时: $total_time"
        
        # 提取资源使用
        if [ -f "$latest_dir"/resource_summary.json ]; then
            echo "💻 资源使用:"
            local params=$(grep -o '"total_params": [0-9,]*' "$latest_dir"/resource_summary.json | head -1)
            local flops=$(grep -o '"flops_g": [0-9.]*' "$latest_dir"/resource_summary.json | head -1)
            local memory=$(grep -o '"memory_mb": [0-9.]*' "$latest_dir"/resource_summary.json | head -1)
            
            echo "  - 参数量: $params"
            echo "  - FLOPs: $flops G"
            echo "  - 内存: $memory MB"
        fi
        
        # 提取训练曲线趋势
        local train_losses=$(grep -E "Train Loss.*[0-9]" "$latest_dir"/training.log | tail -5 | awk '{print $NF}')
        local val_losses=$(grep -E "Val Loss.*[0-9]" "$latest_dir"/training.log | tail -5 | awk '{print $NF}')
        
        echo "📊 损失趋势 (最近5轮):"
        echo "  - 训练损失: $train_losses"
        echo "  - 验证损失: $val_losses"
        
    else
        echo "⚠️  训练日志不存在，可能仍在训练中"
    fi
    
    echo ""
    echo "---"
    echo ""
}

# 记录SwinUNet结果
echo "📋 记录 SwinUNet 结果:"
extract_model_results "SwinUNet"

# 模型列表
MODELS=("UNet" "FNO2d" "UFNOUNet" "SegFormer" "UNetFormer" "MLPMixer" "MLP" "Transformer" "ViT" "Hybrid")

echo "🎯 待测试模型列表:"
for i in "${!MODELS[@]}"; do
    echo "  $((i+2)). ${MODELS[$i]}"
done

echo ""
echo "📄 详细记录已保存至: $RECORD_FILE"
echo ""
echo "🚀 准备启动UNet训练..."

# 保存当前结果到文件
cat >> "$RECORD_FILE" << EOF

手动调试计划:
===============
1. SwinUNet ✅ (已完成)
$(for i in "${!MODELS[@]}"; do echo "$((i+2)). ${MODELS[$i]} ⏳"; done)

EOF

echo "✅ 记录完成！"