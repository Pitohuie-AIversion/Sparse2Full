#!/bin/bash
# 模型调试训练实时监控脚本

echo "📊 UNet 训练实时监控"
echo "====================="
echo ""

# 获取最新的UNet训练目录
get_latest_unet_dir() {
    find runs/ -name "*Debug100*UNet*" -type d 2>/dev/null | sort | tail -1
}

# 实时监控函数
monitor_training() {
    local dir=$1
    local model_name=$2
    
    echo "📁 监控目录: $dir"
    echo "🎯 模型: $model_name"
    echo ""
    
    # 检查训练日志是否存在
    if [ ! -f "$dir/training.log" ]; then
        echo "⚠️  训练日志不存在，等待启动..."
        return 1
    fi
    
    echo "📈 实时训练状态:"
    echo "时间: $(date)"
    echo ""
    
    # 显示最近10轮的训练结果
    echo "最近训练轮次:"
    tail -n 50 "$dir/training.log" | grep -E "Epoch.*[0-9].*Train Loss.*Val Loss" | tail -10 | while read -r line; do
        echo "  $line"
    done
    
    echo ""
    
    # 提取关键指标
    local total_epochs=$(grep -c "Epoch.*[0-9].*Train Loss" "$dir/training.log" 2>/dev/null || echo "0")
    local current_epoch=$(grep "Epoch.*[0-9].*Train Loss" "$dir/training.log" | tail -1 | grep -o "Epoch[[:space:]]*[0-9]*" | grep -o "[0-9]*")
    local best_val=$(grep "Best.*Val Loss\|最佳验证损失" "$dir/training.log" | tail -1 | grep -o "[0-9]*\.[0-9]*" | tail -1)
    
    echo "📊 训练统计:"
    echo "  - 总轮次: $total_epochs"
    echo "  - 当前轮次: ${current_epoch:-未知}"
    echo "  - 最佳验证损失: ${best_val:-未知}"
    
    # 检查训练状态
    if grep -q "Training finished" "$dir/training.log"; then
        echo "✅ 训练状态: 已完成"
        return 0
    elif grep -q "失败\|错误\|Error" "$dir/training.log"; then
        echo "❌ 训练状态: 失败"
        return 1
    else
        echo "🔄 训练状态: 进行中"
        return 2
    fi
}

# 主监控循环
while true; do
    clear
    echo "📊 UNet 训练实时监控"
    echo "====================="
    echo "更新时间: $(date)"
    echo ""
    
    LATEST_DIR=$(get_latest_unet_dir)
    
    if [ -n "$LATEST_DIR" ]; then
        monitor_training "$LATEST_DIR" "UNet"
        status=$?
        
        if [ $status -eq 0 ]; then
            echo ""
            echo "🎉 UNet 训练已完成！"
            echo "准备启动下一个模型: FNO2d"
            break
        elif [ $status -eq 1 ]; then
            echo ""
            echo "❌ UNet 训练失败，需要检查错误"
            break
        fi
    else
        echo "❌ 未找到UNet训练目录"
    fi
    
    echo ""
    echo "按 Ctrl+C 停止监控，或等待30秒自动更新..."
    sleep 30
done