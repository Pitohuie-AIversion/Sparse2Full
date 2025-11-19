#!/bin/bash
# 手动模型调试流程控制脚本

echo "🎯 手动逐个模型调试控制面板"
echo "================================"
echo ""

# 模型调试顺序
MODELS=("SwinUNet" "UNet" "FNO2d" "UFNOUNet" "SegFormer" "UNetFormer" "MLPMixer" "MLP" "Transformer" "ViT" "Hybrid")
CURRENT_INDEX=1  # 当前是UNet (索引1)

echo "📋 调试进度:"
echo "✅ 1. SwinUNet - 已完成"
echo "🔄 2. UNet - 正在运行"
echo "⏳ 剩余模型:"
for i in $(seq 3 ${#MODELS[@]}); do
    echo "   $i. ${MODELS[$((i-1))]}"
done

echo ""
echo "🚀 当前操作选项:"
echo "1. 等待UNet完成并检查状态"
echo "2. 手动启动下一个模型 (FNO2d)"
echo "3. 查看所有模型结果"
echo "4. 生成调试报告"
echo ""

# 函数：等待当前训练完成
wait_for_training() {
    echo "⏳ 等待当前训练完成..."
    
    # 查找正在运行的训练进程
    while pgrep -f "train_real_data_ar" > /dev/null; do
        echo -n "."
        sleep 30
    done
    echo ""
    echo "✅ 当前训练已完成"
}

# 函数：检查模型状态
check_model_status() {
    local model_name=$1
    local pattern="*Debug100*${model_name}*"
    
    echo "🔍 检查 $model_name 状态..."
    
    # 查找训练目录
    local latest_dir=$(find runs/ -name "$pattern" -type d 2>/dev/null | sort | tail -1)
    
    if [ -z "$latest_dir" ]; then
        echo "❌ $model_name - 未找到训练记录"
        return 1
    fi
    
    if [ -f "$latest_dir"/training.log ]; then
        if grep -q "Training finished" "$latest_dir"/training.log; then
            echo "✅ $model_name - 训练完成"
            
            # 提取最佳结果
            local best_val=$(grep "Best.*Val Loss" "$latest_dir"/training.log | tail -1 | grep -o "[0-9]*\.[0-9]*" | tail -1)
            echo "   最佳验证损失: ${best_val:-未知}"
            
            return 0
        elif grep -q "失败\|错误\|Error" "$latest_dir"/training.log; then
            echo "❌ $model_name - 训练失败"
            return 1
        else
            echo "🔄 $model_name - 训练中或中断"
            return 2
        fi
    else
        echo "⚠️  $model_name - 无训练日志"
        return 3
    fi
}

# 主循环
echo "请选择操作 (1-4):"
read -r choice

case $choice in
    1)
        wait_for_training
        check_model_status "UNet"
        ;;
    2)
        echo "🚀 准备启动 FNO2d 训练..."
        echo "请手动执行: python tools/training/train_real_data_ar.py --config configs/train/ar_training_config_debug_fno2d.yaml"
        ;;
    3)
        echo "📊 查看所有模型结果:"
        for model in "${MODELS[@]}"; do
            check_model_status "$model"
        done
        ;;
    4)
        echo "📈 生成调试报告..."
        /share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/get_complete_results.sh
        ;;
    *)
        echo "❌ 无效选择"
        ;;
esac

echo ""
echo "📝 提示: 可以随时使用以下命令检查训练状态:"
echo "   ps aux | grep train_real_data_ar"
echo "   tail -f runs/AR-DR2D-Debug100*/training.log"