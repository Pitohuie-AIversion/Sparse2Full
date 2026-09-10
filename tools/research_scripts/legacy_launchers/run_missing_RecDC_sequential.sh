#!/bin/bash
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "Starting Sequential Training for Missing Ablation Groups"
echo "Order: 1. UNet (Rec+DC) -> 2. EDSR (Rec+DC)"
echo "========================================================"

# 1. UNet: Rec + DC (No Spec)
echo "[1/2] Starting UNet Rec+DC (No Spec)..."
mkdir -p runs_3loss_ablation_unet/A2_RecDC

python tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml \
    model.name=unet \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="3loss_ablation/A2_RecDC" \
    experiment.output_dir="runs_3loss_ablation_unet/A2_RecDC" \
    logging.log_model=true

echo "[1/2] UNet Rec+DC Finished (Exit Code: $?)"
echo "--------------------------------------------------------"

# 2. EDSR: Rec + DC (No Spec)
echo "[2/2] Starting EDSR Rec+DC (No Spec)..."
mkdir -p runs_3loss_ablation/A2_RecDC

python tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_ablation_nospec_edsr_sr4.yaml \
    model.name=edsr \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="3loss_ablation/A2_RecDC_EDSR" \
    experiment.output_dir="runs_3loss_ablation/A2_RecDC" \
    logging.log_model=true

echo "[2/2] EDSR Rec+DC Finished (Exit Code: $?)"
echo "========================================================"
echo "All sequential tasks completed."
