#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# ==========================================
# 1. UNet: Rec + DC (No Spec)
# ==========================================
echo "Starting UNet Rec+DC (No Spec)..."
mkdir -p runs_3loss_ablation_unet/A2_RecDC

nohup torchrun --nproc_per_node=2 --master_port=29505 tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml \
    model.name=unet \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="3loss_ablation/A2_RecDC" \
    experiment.output_dir="runs_3loss_ablation_unet/A2_RecDC" \
    logging.log_model=true > runs_3loss_ablation_unet/nohup_A2_RecDC_final.log 2>&1 &

PID_UNET=$!
echo "UNet Rec+DC PID: $PID_UNET"
sleep 5
if ps -p $PID_UNET > /dev/null; then
   echo "✅ UNet process $PID_UNET is running."
else
   echo "❌ UNet process $PID_UNET failed to start. Check runs_3loss_ablation_unet/nohup_A2_RecDC_final.log"
   tail -n 20 runs_3loss_ablation_unet/nohup_A2_RecDC_final.log
fi

# ==========================================
# 2. EDSR: Rec + DC (No Spec)
# ==========================================
echo "Starting EDSR Rec+DC (No Spec)..."
mkdir -p runs_3loss_ablation/A2_RecDC

nohup torchrun --nproc_per_node=2 --master_port=29506 tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_ablation_nospec_edsr_sr4.yaml \
    model.name=edsr \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.reconstruction=1.0 \
    training.loss_weights.spectral=0.0 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="3loss_ablation/A2_RecDC_EDSR" \
    experiment.output_dir="runs_3loss_ablation/A2_RecDC" \
    logging.log_model=true > runs_3loss_ablation/nohup_A2_RecDC_EDSR_final.log 2>&1 &

PID_EDSR=$!
echo "EDSR Rec+DC PID: $PID_EDSR"
sleep 5
if ps -p $PID_EDSR > /dev/null; then
   echo "✅ EDSR process $PID_EDSR is running."
else
   echo "❌ EDSR process $PID_EDSR failed to start. Check runs_3loss_ablation/nohup_A2_RecDC_EDSR_final.log"
   tail -n 20 runs_3loss_ablation/nohup_A2_RecDC_EDSR_final.log
fi
