#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

echo "========================================================"
echo "Starting Full Reproduction of EDSR Ablation Experiments"
echo "========================================================"

# 1. EDSR: A0 Baseline (MSE Only)
echo "[1/4] Starting EDSR A0_Baseline (MSE Only)..."
mkdir -p runs_3loss_ablation/A0_Baseline_Repro_v2

torchrun --nproc_per_node=2 --master_port=29500 tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A0_Baseline/config_merged.yaml \
    experiment.name="3loss_ablation/A0_Baseline_Repro" \
    experiment.output_dir="runs_3loss_ablation/A0_Baseline_Repro_v2" \
    logging.log_model=true

echo "[1/4] EDSR A0_Baseline Finished (Exit Code: $?)"
echo "--------------------------------------------------------"

# 2. EDSR: A2 Rec+Spec
echo "[2/4] Starting EDSR A2_RecSpec (Rec+Spec)..."
mkdir -p runs_3loss_ablation/A2_RecSpec_Repro_v2

torchrun --nproc_per_node=2 --master_port=29501 tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A2_RecSpec/config_merged.yaml \
    experiment.name="3loss_ablation/A2_RecSpec_Repro" \
    experiment.output_dir="runs_3loss_ablation/A2_RecSpec_Repro_v2" \
    logging.log_model=true

echo "[2/4] EDSR A2_RecSpec Finished (Exit Code: $?)"
echo "--------------------------------------------------------"

# 3. EDSR: A2 Rec+DC
echo "[3/4] Starting EDSR A2_RecDC (Rec+DC)..."
mkdir -p runs_3loss_ablation/A2_RecDC_Repro_v2

torchrun --nproc_per_node=2 --master_port=29502 tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A2_RecDC/config_merged.yaml \
    experiment.name="3loss_ablation/A2_RecDC_Repro" \
    experiment.output_dir="runs_3loss_ablation/A2_RecDC_Repro_v2" \
    logging.log_model=true

echo "[3/4] EDSR A2_RecDC Finished (Exit Code: $?)"
echo "--------------------------------------------------------"

# 4. EDSR: A3 Full (Rec+Spec+DC)
echo "[4/4] Starting EDSR A3_Full (Rec+Spec+DC)..."
mkdir -p runs_3loss_ablation/A3_Full_Repro_v2

torchrun --nproc_per_node=2 --master_port=29503 tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A3_Full/config_merged.yaml \
    experiment.name="3loss_ablation/A3_Full_Repro" \
    experiment.output_dir="runs_3loss_ablation/A3_Full_Repro_v2" \
    logging.log_model=true

echo "[4/4] EDSR A3_Full Finished (Exit Code: $?)"
echo "========================================================"
echo "All reproduction tasks completed."
