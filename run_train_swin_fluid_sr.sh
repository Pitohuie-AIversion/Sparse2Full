#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1   # 双卡并行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="runs/swin_fluid_sr_train.log"
mkdir -p runs/checkpoints
mkdir -p runs_swin_fluid_sr/checkpoints

echo "=========================================================="
echo "启动全新 SwinFluidSR 科学流体超分辨率双卡训练"
echo "架构改进亮点："
echo " 1. 直接低维输入 (Direct 32x32 Input, Token计算量骤减 16 倍)"
echo " 2. RSTB 移位窗口自注意力 (Residual Swin Transformer Blocks)"
echo " 3. 亚像素卷积上采样 (PixelShuffle 4x Upsampling)"
echo " 4. 全局物理插值残差学习 (Global Bicubic Residual)"
echo " 5. 激活频域能谱物理损失 (Spectral Loss = 0.1)"
echo "=========================================================="

python3 train.py \
  "data.data_path=/root/autodl-tmp/datasets/2D_rdb_NA_NA.h5" \
  "data.keys=['data']" \
  "data.splits_dir=splits_shallow" \
  "data.image_size=128" \
  "data.observation.mode=SR" \
  "data.observation.sr.scale_factor=4" \
  "data.observation.sr.blur_sigma=1.0" \
  "data.observation.sr.blur_kernel_size=5" \
  "data.observation.sr.boundary_mode=mirror" \
  "data.dataloader.batch_size=16" \
  "data.dataloader.num_workers=4" \
  "data.dataloader.pin_memory=true" \
  "training.epochs=100" \
  "training.log_interval=10" \
  "training.save_interval=10" \
  "training.use_amp=true" \
  "training.grad_clip_norm=1.0" \
  "training.optimizer.name=AdamW" \
  "training.optimizer.params.lr=2e-4" \
  "training.optimizer.params.weight_decay=1e-4" \
  "training.optimizer.params.betas=[0.9,0.999]" \
  "training.scheduler.name=cosine_warmup" \
  "training.scheduler.params.T_max=100" \
  "training.scheduler.params.warmup_steps=50" \
  "training.scheduler.params.eta_min=1e-6" \
  "+training.distributed.enabled=true" \
  "model.name=SwinFluidSR" \
  "++model.params.in_channels=1" \
  "++model.params.out_channels=1" \
  "++model.params.img_size=32" \
  "++model.params.upscale_factor=4" \
  "++model.params.embed_dim=96" \
  "++model.params.depths=[4,4,4,4]" \
  "++model.params.num_heads=[4,4,4,4]" \
  "++model.params.window_size=8" \
  "++model.params.mlp_ratio=2.0" \
  "++model.params.drop_path_rate=0.05" \
  "++model.params.global_residual=true" \
  "++model.params.use_lowres_input=true" \
  "loss.rec_weight=1.0" \
  "loss.spec_weight=0.1" \
  "loss.dc_weight=0.0" \
  "loss.rec_loss_type=l2" \
  "experiment.name=2D_RDB_SwinFluidSR_DirectLR_DDP" \
  "experiment.device=cuda:0" \
  "experiment.seed=2025" \
  "experiment.output_dir=runs_swin_fluid_sr" \
  2>&1 | tee "$LOG"
