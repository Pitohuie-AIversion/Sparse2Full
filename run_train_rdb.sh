#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 train.py \
  data.data_path="/root/autodl-tmp/datasets/2D_rdb_NA_NA.h5" \
  "data.keys=['data']" \
  data.splits_dir=splits_shallow \
  data.image_size=128 \
  data.observation.mode=SR \
  data.observation.sr.scale_factor=4 \
  data.dataloader.batch_size=8 \
  data.dataloader.num_workers=2 \
  training.epochs=20 \
  training.log_interval=5 \
  training.save_interval=5 \
  model.name="SwinUNet" \
  experiment.name="2D_RDB_SwinUNet_SR4" \
  experiment.device="cuda:0"
