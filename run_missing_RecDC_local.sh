#!/bin/bash
export PYTHONUNBUFFERED=1

# 移除手动指定的 CUDA_VISIBLE_DEVICES，让 PyTorch 自动检测
unset CUDA_VISIBLE_DEVICES

# 打印环境信息
echo "Checking CUDA environment..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# ==========================================
# 1. UNet: Rec + DC (No Spec)
# ==========================================
echo "Starting UNet Rec+DC (No Spec)..."
mkdir -p runs_3loss_ablation_unet/A2_RecDC

# 使用 python 直接启动 (单卡模式，避免 torchrun 的复杂性)
nohup python tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml \
    model.name=unet \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="3loss_ablation/A2_RecDC" \
    experiment.output_dir="runs_3loss_ablation_unet/A2_RecDC" \
    logging.log_model=true > runs_3loss_ablation_unet/nohup_A2_RecDC_local.log 2>&1 &

PID_UNET=$!
echo "UNet Rec+DC PID: $PID_UNET"
sleep 5
if ps -p $PID_UNET > /dev/null; then
   echo "✅ UNet process $PID_UNET is running."
else
   echo "❌ UNet process $PID_UNET failed to start. Check runs_3loss_ablation_unet/nohup_A2_RecDC_local.log"
   tail -n 20 runs_3loss_ablation_unet/nohup_A2_RecDC_local.log
fi

# ==========================================
# 2. EDSR: Rec + DC (No Spec)
# ==========================================
echo "Starting EDSR Rec+DC (No Spec)..."
mkdir -p runs_3loss_ablation/A2_RecDC

# 使用 python 直接启动
nohup python tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_ablation_nospec_edsr_sr4.yaml \
    model.name=edsr \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="3loss_ablation/A2_RecDC_EDSR" \
    experiment.output_dir="runs_3loss_ablation/A2_RecDC" \
    logging.log_model=true > runs_3loss_ablation/nohup_A2_RecDC_EDSR_local.log 2>&1 &

PID_EDSR=$!
echo "EDSR Rec+DC PID: $PID_EDSR"
sleep 5
if ps -p $PID_EDSR > /dev/null; then
   echo "✅ EDSR process $PID_EDSR is running."
else
   echo "❌ EDSR process $PID_EDSR failed to start. Check runs_3loss_ablation/nohup_A2_RecDC_EDSR_local.log"
   tail -n 20 runs_3loss_ablation/nohup_A2_RecDC_EDSR_local.log
fi
