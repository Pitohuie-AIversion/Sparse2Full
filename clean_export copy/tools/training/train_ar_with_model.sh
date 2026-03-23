#!/bin/bash
# AR训练脚本 - 支持不同模型切换
# 用于快速切换和训练不同的模型架构

echo "AR训练脚本 - 模型切换工具"
echo "=========================="

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <模型名称> [额外参数]"
    echo ""
    echo "可用模型:"
    echo "  swinunet    - Swin-UNet (默认配置)"
    echo "  unet        - UNet"
    echo "  fno         - FNO2d"
    echo "  segformer   - SegFormer"
    echo ""
    echo "示例:"
    echo "  $0 swinunet"
    echo "  $0 unet --dry-run"
    echo "  $0 fno --epochs 1000"
    exit 1
fi

MODEL_NAME=$1
shift  # 移除第一个参数，保留其他参数

# 根据模型名称选择配置文件
case $MODEL_NAME in
    swinunet)
        CONFIG_FILE="configs/train/ar_training_config debug.yaml"
        echo "使用SwinUNet模型配置"
        ;;
    unet)
        CONFIG_FILE="configs/train/ar_training_config_unet.yaml"
        echo "使用UNet模型配置"
        ;;
    fno)
        CONFIG_FILE="configs/train/ar_training_config_fno.yaml"
        echo "使用FNO2d模型配置"
        ;;
    segformer)
        CONFIG_FILE="configs/train/ar_training_config_segformer.yaml"
        echo "使用SegFormer模型配置"
        ;;
    *)
        echo "错误: 未知的模型名称 '$MODEL_NAME'"
        echo "可用模型: swinunet, unet, fno, segformer"
        exit 1
        ;;
esac

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 '$CONFIG_FILE' 不存在"
    exit 1
fi

echo "配置文件: $CONFIG_FILE"
echo ""

# 构建训练命令
TRAIN_CMD="python tools/training/train_real_data_ar.py --config $CONFIG_FILE $@"

echo "执行命令:"
echo "$TRAIN_CMD"
echo ""

# 执行训练
eval $TRAIN_CMD