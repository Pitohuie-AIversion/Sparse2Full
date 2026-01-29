#!/bin/bash
# 运行消融实验 A0-A3 (Loss Ablation Study)
# 用法: bash scripts/run_ablation_study.sh [GPU_ID]

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 基础命令
CMD="python tools/training/train_real_data_ar.py"

# 通用配置 (减少 epoch 以便快速验证，正式跑建议 100)
EPOCHS=50
BATCH_SIZE=32

echo "========================================================"
echo "开始运行消融实验 (Ablation Study) - GPU $GPU_ID"
echo "========================================================"

# A0: 仅 L_rec (Baseline)
# loss.spectral.weight=0.0 loss.data_consistency.weight=0.0
echo "[1/4] Running A0: Reconstruction Loss Only..."
$CMD experiment.name="Ablation-A0-RecOnly" \
     training.epochs=$EPOCHS \
     data.dataloader.batch_size=$BATCH_SIZE \
     loss.spectral.weight=0.0 \
     loss.data_consistency.weight=0.0 \
     logging.experiment_name="Ablation-A0"

# A1: L_rec + L_dc
# loss.spectral.weight=0.0 loss.data_consistency.weight=0.5
echo "[2/4] Running A1: Rec + DC Loss..."
$CMD experiment.name="Ablation-A1-RecDC" \
     training.epochs=$EPOCHS \
     data.dataloader.batch_size=$BATCH_SIZE \
     loss.spectral.weight=0.0 \
     loss.data_consistency.weight=0.5 \
     logging.experiment_name="Ablation-A1"

# A2: L_rec + L_spec
# loss.spectral.weight=0.1 loss.data_consistency.weight=0.0
echo "[3/4] Running A2: Rec + Spectral Loss..."
$CMD experiment.name="Ablation-A2-RecSpec" \
     training.epochs=$EPOCHS \
     data.dataloader.batch_size=$BATCH_SIZE \
     loss.spectral.weight=0.1 \
     loss.data_consistency.weight=0.0 \
     logging.experiment_name="Ablation-A2"

# A3: L_rec + L_spec + L_dc (Ours)
# loss.spectral.weight=0.1 loss.data_consistency.weight=0.5
echo "[4/4] Running A3: All Losses (Ours)..."
$CMD experiment.name="Ablation-A3-Ours" \
     training.epochs=$EPOCHS \
     data.dataloader.batch_size=$BATCH_SIZE \
     loss.spectral.weight=0.1 \
     loss.data_consistency.weight=0.5 \
     logging.experiment_name="Ablation-A3"

echo "========================================================"
echo "消融实验 A0-A3 全部完成！"
echo "请使用 tensorboard --logdir runs 查看对比曲线"
echo "========================================================"
