#!/bin/bash
# 一键训练脚本
# 使用方法: bash train_all.sh

set -e

echo "开始训练所有模型..."

# 设置环境
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 训练配置列表
configs=(
    "configs/sr_darcy_swinunet.yaml"
    "configs/sr_darcy_unet.yaml"
    "configs/crop_cfd_hybrid.yaml"
)

# 种子列表
seeds=(42 123 456)

# 训练所有配置
for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "训练配置: $config, 种子: $seed"
        python tools/train.py --config $config --seed $seed
    done
done

echo "训练完成！"
