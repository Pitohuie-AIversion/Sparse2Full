#!/bin/bash
export PYTHONUNBUFFERED=1

# ==========================================
# 1. UNet: Rec + DC (No Spec)
# ==========================================
# 目标文件夹: runs_3loss_ablation_unet/A2_RecDC
# 配置: Rec=1.0, Spec=0.0, DC=1.0
echo "Starting UNet Rec+DC (No Spec)..."
mkdir -p runs_3loss_ablation_unet/A2_RecDC

nohup torchrun --nproc_per_node=2 tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml \
    model.name=unet \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="3loss_ablation/A2_RecDC" \
    experiment.output_dir="runs_3loss_ablation_unet/A2_RecDC" \
    logging.log_model=true > runs_3loss_ablation_unet/nohup_A2_RecDC.log 2>&1 &

PID_UNET=$!
echo "UNet Rec+DC PID: $PID_UNET"


# ==========================================
# 2. EDSR: Rec + DC (No Spec)
# ==========================================
# 目标文件夹: runs_3loss_ablation/A2_RecDC
# 配置: Rec=1.0, Spec=0.0, DC=1.0
echo "Starting EDSR Rec+DC (No Spec)..."
mkdir -p runs_3loss_ablation/A2_RecDC

# 使用 EDSR NoSpec 配置文件 (该配置默认已是 Spec=0, DC=1, 但我们显式指定以防万一)
nohup python tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_ablation_nospec_edsr_sr4.yaml \
    model.name=edsr \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="3loss_ablation/A2_RecDC_EDSR" \
    experiment.output_dir="runs_3loss_ablation/A2_RecDC" \
    logging.log_model=true > runs_3loss_ablation/nohup_A2_RecDC_EDSR.log 2>&1 &

PID_EDSR=$!
echo "EDSR Rec+DC PID: $PID_EDSR"

echo "✅ Missing Rec+DC experiments started."
