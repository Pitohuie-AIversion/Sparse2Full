#!/bin/bash
# 模型调试训练记录脚本 - 获取完整结果

echo "📊 模型调试训练记录"
echo "===================="
echo ""

# 创建记录文件
RECORD_FILE="runs/model_debug_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p runs

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
        # 提取训练统计
        local total_epochs=$(grep -c "Epoch.*[0-9]" "$latest_dir"/training.log 2>/dev/null || echo "0")
        local final_train_loss=$(grep "Train Loss" "$latest_dir"/training.log | tail -1 | grep -o "[0-9]*\.[0-9]*" | head -1)
        local final_val_loss=$(grep "Val Loss" "$latest_dir"/training.log | tail -1 | grep -o "[0-9]*\.[0-9]*" | head -1)
        local best_val_loss=$(grep "Best" "$latest_dir"/training.log | tail -1 | grep -o "[0-9]*\.[0-9]*" | tail -1)
        
        echo "📈 训练统计:"
        echo "  - 总轮次: $total_epochs"
        echo "  - 最终训练损失: $final_train_loss"
        echo "  - 最终验证损失: $final_val_loss"
        echo "  - 最佳验证损失: $best_val_loss"
        
        # 提取资源使用
        if [ -f "$latest_dir"/resource_summary.json ]; then
            echo "💻 资源使用:"
            local params=$(grep -o '"total_params":[[:space:]]*[0-9]*' "$latest_dir"/resource_summary.json | head -1 | grep -o "[0-9]*")
            local flops=$(grep -o '"flops_g":[[:space:]]*[0-9.]*' "$latest_dir"/resource_summary.json | head -1 | grep -o "[0-9.]*")
            local memory=$(grep -o '"memory_mb":[[:space:]]*[0-9.]*' "$latest_dir"/resource_summary.json | head -1 | grep -o "[0-9.]*")
            
            echo "  - 参数量: ${params:-未知}"
            echo "  - FLOPs: ${flops:-未知} G"
            echo "  - 内存: ${memory:-未知} MB"
        fi
        
        # 检查训练状态
        if grep -q "Training finished" "$latest_dir"/training.log; then
            echo "✅ 训练状态: 已完成"
        elif grep -q "失败\|错误\|Error" "$latest_dir"/training.log; then
            echo "❌ 训练状态: 失败"
        else
            echo "⚠️  训练状态: 未知或中断"
        fi
        
    else
        echo "⚠️  训练日志不存在"
    fi
    
    echo ""
    echo "---"
    echo ""
}

# 记录SwinUNet结果
echo "📋 记录 SwinUNet 结果:"
extract_model_results "SwinUNet"

# 检查UNet是否完成
echo "📋 检查 UNet 状态:"
extract_model_results "UNet"

# 模型列表
MODELS=("UNet" "FNO2d" "UFNOUNet" "SegFormer" "UNetFormer" "MLPMixer" "MLP" "Transformer" "ViT" "Hybrid")

echo "🎯 待测试模型列表:"
for i in "${!MODELS[@]}"; do
    echo "  $((i+2)). ${MODELS[$i]}"
done

echo ""
echo "📄 详细记录已保存"

# 保存结果到文件
cat > "$RECORD_FILE" << EOF
模型调试训练汇总报告
====================
时间: $(date)

SwinUNet 结果:
$(extract_model_results "SwinUNet" 2>&1)

UNet 状态:
$(extract_model_results "UNet" 2>&1)

待测试模型列表:
$(for i in "${!MODELS[@]}"; do echo "$((i+2)). ${MODELS[$i]}"; done)

EOF

echo "✅ 记录完成！文件: $RECORD_FILE"