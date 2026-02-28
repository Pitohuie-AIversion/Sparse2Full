#!/bin/bash
# 运行 UNet 在 Crop (Inpainting) 任务上的快速验证实验
# 目标：Center Crop 48x48 (37.5% 面积) -> 恢复 128x128
# 模型：UNet (Classic Encoder-Decoder)
# 策略：Masking Inpainting

# 1. 设置实验名称与关键参数
EXP_NAME="AR-DR2D-Crop-Inpainting-UNet-Size48"
TASK="Crop"
CROP_SIZE="[48,48]"
MODEL="unet"
BATCH_SIZE=128
EPOCHS=100  # 快速验证跑100轮足够看趋势

# 2. 运行训练脚本
# 使用 overrides 覆盖默认配置
# - task=Crop: 启用 Crop 任务
# - model=unet: 切换到 UNet 模型
# - training.max_epochs=100: 缩短训练时间
# - data.dataloader.num_workers=8: 加速数据加载 (增加 worker 数以匹配大 Batch)

# 使用 torchrun 启动双卡分布式训练
# --nproc_per_node=2: 使用 2 张显卡
# --master_port: 指定端口避免冲突
torchrun --nproc_per_node=2 --master_port=29500 tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_aligned_crop.yaml \
    --model ${MODEL} \
    experiment.name=${EXP_NAME} \
    data.dataloader.batch_size=${BATCH_SIZE} \
    training.max_epochs=${EPOCHS} \
    data.dataloader.num_workers=8 \
    data.observation.crop_size=${CROP_SIZE} \
    data.observation.crop_mode="center" \
    model.img_size=128

# 3. 打印完成信息
echo "✅ 实验 ${EXP_NAME} 已启动/完成。"
echo "请检查 runs/${EXP_NAME}/training.log 查看训练进度。"
