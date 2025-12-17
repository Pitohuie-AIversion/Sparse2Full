#!/usr/bin/env bash
set -euo pipefail

CONFIG="thesis_paper/configs/spatial_training_template.yaml"
SEEDS=(2025 2026 2027)

MODELS=(
  "@thesis_paper/configs/model/unet_10m.yaml SR-UNet-10M"
  "@thesis_paper/configs/model/unetpp_10m.yaml SR-UNetPP-10M"
  "@thesis_paper/configs/model/fno2d_10m.yaml SR-FNO2d-10M"
  "@thesis_paper/configs/model/segformer_10m.yaml SR-SegFormer-10M"
  "@thesis_paper/configs/model/mlp_mixer_10m.yaml SR-MLPMixer-10M"
  "@thesis_paper/configs/model/segformer_unetformer_10m.yaml SR-SegFormerUNet-10M"
  "@thesis_paper/configs/model/swin_tiny_10m.yaml SR-SwinTiny-10M"
  "@thesis_paper/configs/model/vit_10m.yaml SR-ViT-10M"
  "@thesis_paper/configs/model/swin_unet_10m.yaml SR-SwinUNet-10M"
  "@thesis_paper/configs/model/hybrid_10m.yaml SR-Hybrid-10M"
)

for entry in "${MODELS[@]}"; do
  read -r MODEL EXP <<<"${entry}"
  for SEED in "${SEEDS[@]}"; do
    echo "Running ${EXP} seed=${SEED}"
    python train.py +config=${CONFIG} +model=${MODEL} experiment.seed=${SEED} experiment.name=${EXP}-s${SEED}
  done
done

echo "Batch runs completed"
