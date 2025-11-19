#!/bin/bash
# 模型调试训练结果对比分析

echo "📊 模型调试训练结果对比分析"
echo "================================"
echo ""

# 函数：提取模型完整结果
extract_model_complete_results() {
    local model_name=$1
    local pattern="*Debug100*${model_name}*"
    
    echo "🔍 分析 $model_name 训练结果..."
    
    # 查找最新的训练目录
    local latest_dir=$(find runs/ -name "$pattern" -type d 2>/dev/null | sort | tail -1)
    
    if [ -z "$latest_dir" ]; then
        echo "❌ 未找到 $model_name 的训练目录"
        return 1
    fi
    
    echo "📁 训练目录: $(basename "$latest_dir")"
    
    # 检查是否训练完成
    if [ -f "$latest_dir/training.log" ]; then
        # 提取完整训练统计
        local total_epochs=$(grep -c "Epoch.*[0-9].*Train Loss" "$latest_dir/training.log" 2>/dev/null || echo "0")
        local final_train_loss=$(grep "Train Loss" "$latest_dir/training.log" | tail -1 | grep -o "[0-9]*\.[0-9]*" | head -1)
        local final_val_loss=$(grep "Val Loss" "$latest_dir/training.log" | tail -1 | grep -o "[0-9]*\.[0-9]*" | head -1)
        local best_val_loss=$(grep "Best.*Val Loss\|最佳验证损失" "$latest_dir/training.log" | tail -1 | grep -o "[0-9]*\.[0-9]*" | tail -1)
        
        # 提取资源使用
        local params=""
        local flops=""
        local memory=""
        local latency=""
        
        if [ -f "$latest_dir/resource_summary.json" ]; then
            params=$(grep -o '"total_params":[[:space:]]*[0-9]*' "$latest_dir/resource_summary.json" | head -1 | grep -o "[0-9]*")
            flops=$(grep -o '"flops_g":[[:space:]]*[0-9.]*' "$latest_dir/resource_summary.json" | head -1 | grep -o "[0-9.]*")
            memory=$(grep -o '"memory_mb":[[:space:]]*[0-9.]*' "$latest_dir/resource_summary.json" | head -1 | grep -o "[0-9.]*")
            latency=$(grep -o '"latency_ms":[[:space:]]*[0-9.]*' "$latest_dir/resource_summary.json" | head -1 | grep -o "[0-9.]*")
        fi
        
        # 检查训练状态
        local status="未知"
        if grep -q "Training finished" "$latest_dir/training.log"; then
            status="✅ 已完成"
        elif grep -q "失败\|错误\|Error" "$latest_dir/training.log"; then
            status="❌ 失败"
        else
            status="⚠️  进行中/中断"
        fi
        
        # 输出详细结果
        echo "📈 训练统计:"
        echo "  - 状态: $status"
        echo "  - 总轮次: $total_epochs"
        echo "  - 最终训练损失: ${final_train_loss:-未知}"
        echo "  - 最终验证损失: ${final_val_loss:-未知}"
        echo "  - 最佳验证损失: ${best_val_loss:-未知}"
        
        echo "💻 资源使用:"
        echo "  - 参数量: ${params:-未知}"
        echo "  - FLOPs: ${flops:-未知} G"
        echo "  - 内存: ${memory:-未知} MB"
        echo "  - 延迟: ${latency:-未知} ms"
        
        # 计算性能指标
        if [ -n "$best_val_loss" ] && [ -n "$params" ] && [ -n "$flops" ]; then
            echo "🎯 性能评估:"
            echo "  - 参数效率: $(echo "scale=2; $best_val_loss * 1000000 / $params" | bc -l 2>/dev/null || echo "计算错误")"
            echo "  - 计算效率: $(echo "scale=2; $best_val_loss / $flops" | bc -l 2>/dev/null || echo "计算错误")"
        fi
        
        # 保存结果到数组供后续对比
        declare -g "${model_name}_RESULTS"="$status|$total_epochs|$best_val_loss|$params|$flops|$memory|$latency"
        
    else
        echo "⚠️  训练日志不存在"
        declare -g "${model_name}_RESULTS"="⚠️  无日志|0|未知|未知|未知|未知|未知"
    fi
    
    echo ""
    echo "---"
    echo ""
}

# 对比分析函数
compare_models() {
    local models=("$@")
    
    echo "🔍 模型性能对比分析"
    echo "===================="
    echo ""
    
    # 创建对比表格
    printf "%-15s %-10s %-15s %-12s %-10s %-12s %-10s\n" \
        "模型" "状态" "最佳验证损失" "参数量(M)" "FLOPs(G)" "内存(MB)" "延迟(ms)"
    printf "%-15s %-10s %-15s %-12s %-10s %-12s %-10s\n" \
        "---------------" "----------" "---------------" "------------" "----------" "------------" "----------"
    
    for model in "${models[@]}"; do
        local results_var="${model}_RESULTS"
        local results="${!results_var}"
        
        if [ -n "$results" ]; then
            IFS='|' read -r status epochs best_loss params flops memory latency <<< "$results"
            
            # 格式化输出
            printf "%-15s %-10s %-15s %-12s %-10s %-12s %-10s\n" \
                "$model" "$status" "$best_loss" "${params:-N/A}" "${flops:-N/A}" "${memory:-N/A}" "${latency:-N/A}"
        fi
    done
    
    echo ""
    echo "📊 分析总结:"
    echo "1. 验证损失: 数值越小表示预测精度越高"
    echo "2. 参数量: 模型复杂度指标，影响内存占用"
    echo "3. FLOPs: 计算复杂度，影响推理速度"
    echo "4. 延迟: 单次推理时间，影响实时性"
    echo "5. 内存: GPU显存占用情况"
    echo ""
}

# 主分析流程
echo "🚀 开始模型调试结果分析..."
echo ""

# 分析已完成模型
MODELS_TO_ANALYZE=("SwinUNet" "UNet")

for model in "${MODELS_TO_ANALYZE[@]}"; do
    extract_model_complete_results "$model"
done

# 对比分析
compare_models "${MODELS_TO_ANALYZE[@]}"

# 生成建议
echo "💡 模型选择建议:"
echo "===================="
echo ""

# 基于结果生成建议
if [ -n "${SwinUNet_RESULTS}" ] && [ -n "${UNet_RESULTS}" ]; then
    IFS='|' read -r swin_status swin_epochs swin_loss swin_params swin_flops swin_memory swin_latency <<< "${SwinUNet_RESULTS}"
    IFS='|' read -r unet_status unet_epochs unet_loss unet_params unet_flops unet_memory unet_latency <<< "${UNet_RESULTS}"
    
    echo "基于当前训练结果的分析:"
    echo ""
    
    if [ "$swin_status" = "✅ 已完成" ] && [ "$unet_status" = "✅ 已完成" ]; then
        echo "✅ 两个模型都成功完成训练"
        echo ""
        
        # 精度对比
        if [ -n "$swin_loss" ] && [ -n "$unet_loss" ]; then
            echo "🎯 精度对比:"
            if (( $(echo "$swin_loss < $unet_loss" | bc -l 2>/dev/null || echo "0") )); then
                echo "  - SwinUNet 验证损失更低 ($swin_loss < $unet_loss)，精度更高"
            else
                echo "  - UNet 验证损失更低 ($unet_loss < $swin_loss)，精度更高"
            fi
            echo ""
        fi
        
        # 效率对比
        echo "⚡ 效率对比:"
        if [ -n "$swin_params" ] && [ -n "$unet_params" ]; then
            if (( $(echo "$swin_params < $unet_params" | bc -l 2>/dev/null || echo "0") )); then
                echo "  - SwinUNet 参数量更少 ($swin_params M < $unet_params M)，更轻量"
            else
                echo "  - UNet 参数量更少 ($unet_params M < $swin_params M)，更轻量"
            fi
        fi
        
        if [ -n "$swin_flops" ] && [ -n "$unet_flops" ]; then
            if (( $(echo "$swin_flops < $unet_flops" | bc -l 2>/dev/null || echo "0") )); then
                echo "  - SwinUNet 计算量更少 ($swin_flops G < $unet_flops G)，更高效"
            else
                echo "  - UNet 计算量更少 ($unet_flops G < $swin_flops G)，更高效"
            fi
        fi
        
        if [ -n "$swin_latency" ] && [ -n "$unet_latency" ]; then
            if (( $(echo "$swin_latency < $unet_latency" | bc -l 2>/dev/null || echo "0") )); then
                echo "  - SwinUNet 推理延迟更低 ($swin_latency ms < $unet_latency ms)，更快速"
            else
                echo "  - UNet 推理延迟更低 ($unet_latency ms < $swin_latency ms)，更快速"
            fi
        fi
        
        echo ""
        echo "🎯 选择建议:"
        echo "- 如果追求精度: 选择验证损失更低的模型"
        echo "- 如果追求效率: 选择参数量少、FLOPs低、延迟小的模型"
        echo "- 如果资源有限: 优先考虑轻量级模型"
        echo "- 如果要求实时: 优先考虑低延迟模型"
        
    else
        echo "⚠️  部分模型训练未完成或失败，建议:"
        echo "- 检查失败原因并重新训练"
        echo "- 调整超参数或模型配置"
        echo "- 考虑减少训练轮次或数据量"
    fi
fi

echo ""
echo "📝 分析完成！建议继续测试剩余模型以获得更全面的对比结果。"