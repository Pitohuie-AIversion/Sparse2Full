#!/bin/bash
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "Starting Reproduction of EDSR A0 (Baseline) and A3 (Full)"
echo "========================================================"

# 1. EDSR: A0 Baseline (MSE Only)
echo "[1/2] Starting EDSR A0_Baseline (MSE Only)..."
mkdir -p runs_3loss_ablation/A0_Baseline_Repro

python tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A0_Baseline/config_merged.yaml \
    experiment.name="3loss_ablation/A0_Baseline_Repro" \
    experiment.output_dir="runs_3loss_ablation/A0_Baseline_Repro" \
    logging.log_model=true

echo "[1/2] EDSR A0_Baseline Finished (Exit Code: $?)"
echo "--------------------------------------------------------"

# 2. EDSR: A3 Full (Rec+Spec+DC)
echo "[2/2] Starting EDSR A3_Full (Rec+Spec+DC)..."
mkdir -p runs_3loss_ablation/A3_Full_Repro

python tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A3_Full/config_merged.yaml \
    experiment.name="3loss_ablation/A3_Full_Repro" \
    experiment.output_dir="runs_3loss_ablation/A3_Full_Repro" \
    logging.log_model=true

echo "[2/2] EDSR A3_Full Finished (Exit Code: $?)"
echo "========================================================"
echo "All reproduction tasks completed."
