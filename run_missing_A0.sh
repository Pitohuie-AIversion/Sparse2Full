#!/bin/bash
export PYTHONUNBUFFERED=1

# 1. UNet A0 (MSE Only)
# 缺失组：纯重建损失 (A0)，作为 Baseline
# 路径：runs_3loss_ablation_unet/A0_Rec
mkdir -p runs_3loss_ablation_unet/A0_Rec
echo "Starting UNet A0 (Rec Only)..."
nohup torchrun --nproc_per_node=2 tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml \
    model.name=unet \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=0.0 \
    experiment.name="3loss_ablation/A0_Rec" \
    experiment.output_dir="runs_3loss_ablation_unet/A0_Rec" \
    logging.log_model=true > runs_3loss_ablation_unet/nohup_A0_final.log 2>&1 &
PID_UNET=$!
echo "UNet A0 PID: $PID_UNET"

# 2. EDSR A0 (MSE Only)
# 缺失组：纯重建损失 (A0)，作为 Baseline
# 路径：runs_3loss_ablation/A0_Rec
mkdir -p runs_3loss_ablation/A0_Rec
echo "Starting EDSR A0 (Rec Only)..."
# 使用 EDSR 专用配置 (NoSpec模板) 并覆盖参数
nohup python tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_ablation_nospec_edsr_sr4.yaml \
    model.name=edsr \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=0.0 \
    experiment.name="3loss_ablation/A0_Rec_EDSR" \
    experiment.output_dir="runs_3loss_ablation/A0_Rec" \
    logging.log_model=true > runs_3loss_ablation/nohup_A0_edsr.log 2>&1 &
PID_EDSR=$!
echo "EDSR A0 PID: $PID_EDSR"

echo "All missing ablation groups (A0) started."
