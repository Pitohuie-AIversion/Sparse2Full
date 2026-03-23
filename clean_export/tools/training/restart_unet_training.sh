#!/bin/bash
# 重新启动UNet训练并获取完整结果

echo "🔄 重新启动UNet训练"
echo "=================="
echo ""

# 检查是否有训练进程在运行
if pgrep -f "train_real_data_ar" > /dev/null; then
    echo "⚠️  检测到训练进程在运行，请先停止"
    exit 1
fi

# 查找最新的UNet训练目录
LATEST_UNET_DIR=$(find runs/ -name "*Debug100*UNet*" -type d 2>/dev/null | sort | tail -1)

if [ -n "$LATEST_UNET_DIR" ]; then
    echo "📁 找到现有UNet训练目录: $(basename "$LATEST_UNET_DIR")"
    echo "📊 当前状态:"
    
    # 检查训练状态
    if [ -f "$LATEST_UNET_DIR/training.log" ]; then
        local total_lines=$(wc -l < "$LATEST_UNET_DIR/training.log")
        echo "  - 日志行数: $total_lines"
        
        if grep -q "Training finished" "$LATEST_UNET_DIR/training.log"; then
            echo "  - 状态: ✅ 已完成"
            echo ""
            echo "🎉 UNet训练已完成，无需重新启动"
            exit 0
        elif grep -q "失败\|错误\|Error" "$LATEST_UNET_DIR/training.log"; then
            echo "  - 状态: ❌ 训练失败"
            echo ""
            echo "📋 错误信息:"
            grep -A 5 -B 5 "失败\|错误\|Error" "$LATEST_UNET_DIR/training.log" | tail -10
        else
            echo "  - 状态: ⚠️  训练中或中断"
            echo ""
            echo "🔄 准备重新启动训练..."
        fi
    else
        echo "  - 状态: ❌ 无训练日志"
        echo ""
        echo "🔄 准备启动训练..."
    fi
else
    echo "❌ 未找到UNet训练目录"
    echo "🔄 准备启动新的训练..."
fi

echo ""
echo "🚀 启动UNet训练"
echo "=================="

# 启动训练
echo "执行命令:"
echo "python tools/training/train_real_data_ar.py --config configs/train/ar_training_config_debug_unet.yaml"
echo ""

# 执行训练
python tools/training/train_real_data_ar.py --config configs/train/ar_training_config_debug_unet.yaml

echo ""
echo "✅ UNet训练启动完成！"
echo ""
echo "📊 监控建议:"
echo "1. 实时查看日志: tail -f runs/AR-DR2D-Debug100-UNet*/training.log"
echo "2. TensorBoard监控: tensorboard --logdir runs/tensorboard"
echo "3. 检查GPU状态: nvidia-smi"
echo ""
echo "⏱️  预计训练时间: 15-20分钟 (100轮)"