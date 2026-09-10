#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

echo "========================================================"
echo "Starting Full Reproduction of UNet Ablation Experiments (DDP)"
echo "========================================================"

# 1. UNet: A0 Baseline (MSE Only)
echo "[1/4] Starting UNet A0_Baseline (MSE Only)..."
mkdir -p runs_3loss_ablation_unet/A0_Baseline_Repro_v2

torchrun --nproc_per_node=2 --master_port=29505 tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation_unet/A0_Baseline/config_merged.yaml \
    experiment.name="3loss_ablation/A0_Baseline_Repro_UNet" \
    experiment.output_dir="runs_3loss_ablation_unet/A0_Baseline_Repro_v2" \
    logging.log_model=true

echo "[1/4] UNet A0_Baseline Finished (Exit Code: $?)"
echo "--------------------------------------------------------"

# 2. UNet: A2 Rec+Spec
echo "[2/4] Starting UNet A2_RecSpec (Rec+Spec)..."
mkdir -p runs_3loss_ablation_unet/A2_RecSpec_Repro_v2

torchrun --nproc_per_node=2 --master_port=29506 tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation_unet/A2_RecSpec/config_merged.yaml \
    experiment.name="3loss_ablation/A2_RecSpec_Repro_UNet" \
    experiment.output_dir="runs_3loss_ablation_unet/A2_RecSpec_Repro_v2" \
    logging.log_model=true

echo "[2/4] UNet A2_RecSpec Finished (Exit Code: $?)"
echo "--------------------------------------------------------"

# 3. UNet: A2 Rec+DC
echo "[3/4] Starting UNet A2_RecDC (Rec+DC)..."
mkdir -p runs_3loss_ablation_unet/A2_RecDC_Repro_v2

torchrun --nproc_per_node=2 --master_port=29507 tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation_unet/A2_RecDC/config_merged.yaml \
    experiment.name="3loss_ablation/A2_RecDC_Repro_UNet" \
    experiment.output_dir="runs_3loss_ablation_unet/A2_RecDC_Repro_v2" \
    logging.log_model=true

echo "[3/4] UNet A2_RecDC Finished (Exit Code: $?)"
echo "--------------------------------------------------------"

# 4. UNet: A3 Full (Rec+Spec+DC)
echo "[4/4] Starting UNet A3_Full (Rec+Spec+DC)..."
mkdir -p runs_3loss_ablation_unet/A3_Full_Repro_v2

torchrun --nproc_per_node=2 --master_port=29508 tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation_unet/A3_Full/config_merged.yaml \
    experiment.name="3loss_ablation/A3_Full_Repro_UNet" \
    experiment.output_dir="runs_3loss_ablation_unet/A3_Full_Repro_v2" \
    logging.log_model=true

echo "[4/4] UNet A3_Full Finished (Exit Code: $?)"
echo "========================================================"
echo "All UNet reproduction tasks completed."