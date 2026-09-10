#!/bin/bash
# PartialConvUNet Crop Size Scan Experiment Script
# Runs 4 experiments on 2 GPUs (2 per GPU, sequential)

# Common arguments
ARGS="--config thesis_paper/configs/ar_paper_aligned_crop.yaml \
--model PartialConvUNet \
training.max_epochs=100 \
data.dataloader.num_workers=4 \
data.dataloader.batch_size=256 \
model.img_size=128 \
model.input_includes_mask=True \
model.in_channels=2 \
data.observation.crop_mode=center"

# GPU 0 Tasks
(
    echo "Starting Size 32 on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python tools/training/train_real_data_ar.py $ARGS \
        experiment.name=AR-DR2D-Crop-Inpainting-PartialConvUNet-Size32 \
        "data.observation.crop_size=[32,32]"
        
    echo "Starting Size 64 on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python tools/training/train_real_data_ar.py $ARGS \
        experiment.name=AR-DR2D-Crop-Inpainting-PartialConvUNet-Size64 \
        "data.observation.crop_size=[64,64]"
) &

# GPU 1 Tasks
(
    echo "Starting Size 48 on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python tools/training/train_real_data_ar.py $ARGS \
        experiment.name=AR-DR2D-Crop-Inpainting-PartialConvUNet-Size48 \
        "data.observation.crop_size=[48,48]"
        
    echo "Starting Size 80 on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python tools/training/train_real_data_ar.py $ARGS \
        experiment.name=AR-DR2D-Crop-Inpainting-PartialConvUNet-Size80 \
        "data.observation.crop_size=[80,80]"
) &

wait
echo "All experiments completed."
