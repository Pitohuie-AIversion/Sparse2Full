#!/bin/bash
set -e

echo "Starting ablation re-run with fixed loss logic..."

# A0: Baseline (Rec only)
# Expect: Rel-L2 ~0.0135 (Baseline)
echo "Running A0_Baseline..."
python tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A0_Baseline/config_merged.yaml \
    experiment.output_dir=runs_3loss_ablation_fixed/A0_Baseline \
    training.epochs=50

# A2: Rec + Spec
# Expect: Rel-L2 < 0.0135 (Better than Baseline)
echo "Running A2_RecSpec..."
python tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A2_RecSpec/config_merged.yaml \
    experiment.output_dir=runs_3loss_ablation_fixed/A2_RecSpec \
    training.epochs=50

# A3: Full (Rec + Spec + DC)
# Expect: Rel-L2 < 0.0129 (Best)
echo "Running A3_Full..."
python tools/training/train_real_data_ar.py \
    --config runs_3loss_ablation/A3_Full/config_merged.yaml \
    experiment.output_dir=runs_3loss_ablation_fixed/A3_Full \
    training.epochs=50

echo "Ablation re-run completed."
