#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1   # 双卡

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="runs/swin_t_enc_train.log"
mkdir -p runs/checkpoints

echo "Starting SwinTWithEncoder dual-GPU training at $(date)" | tee "$LOG"

# 用 ++ 强制覆盖 Hydra struct（支持新 key）
nohup python3 train.py \
  data.data_path="/root/autodl-tmp/datasets/2D_rdb_NA_NA.h5" \
  "data.keys=['data']" \
  data.splits_dir=splits_shallow \
  data.image_size=128 \
  data.observation.mode=SR \
  data.observation.sr.scale_factor=4 \
  data.observation.sr.blur_sigma=1.0 \
  data.observation.sr.blur_kernel_size=5 \
  data.observation.sr.boundary_mode=mirror \
  data.dataloader.batch_size=16 \
  data.dataloader.num_workers=4 \
  data.dataloader.pin_memory=true \
  training.epochs=100 \
  training.log_interval=10 \
  training.save_interval=10 \
  training.use_amp=true \
  training.grad_clip_norm=1.0 \
  training.optimizer.name=AdamW \
  training.optimizer.params.lr=1e-4 \
  training.optimizer.params.weight_decay=1e-4 \
  "training.optimizer.params.betas=[0.9,0.999]" \
  training.scheduler.name=cosine_warmup \
  training.scheduler.params.T_max=200 \
  training.scheduler.params.warmup_steps=1000 \
  training.scheduler.params.eta_min=1e-6 \
  "+training.distributed.enabled=true" \
  model.name="SwinTWithEncoder" \
  "++model.params.in_channels=4" \
  "++model.params.out_channels=1" \
  "++model.params.img_size=128" \
  "++model.params.encoder_out_channels=4" \
  "++model.params.use_coords=true" \
  "++model.params.use_mask=true" \
  "++model.params.patch_size=4" \
  "++model.params.embed_dim=96" \
  "++model.params.depths=[2,2,6,2]" \
  "++model.params.num_heads=[3,6,12,24]" \
  "++model.params.window_size=8" \
  "++model.params.mlp_ratio=4.0" \
  "++model.params.drop_path_rate=0.1" \
  "++model.params.post_conv3x3=true" \
  loss.rec_weight=1.0 \
  loss.spec_weight=0.0 \
  loss.dc_weight=0.0 \
  loss.rec_loss_type=l2 \
  experiment.name="2D_RDB_SwinTWithEncoder_SR4_DDP" \
  experiment.device="cuda:0" \
  experiment.seed=2025 \
  experiment.output_dir=runs \
  >> "$LOG" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID" | tee -a "$LOG"
echo "Log: $LOG"

sleep 12
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training is running OK. Last log:"
    tail -n 30 "$LOG"
else
    echo "ERROR: process died. Last log:"
    tail -n 50 "$LOG"
    exit 1
fi
