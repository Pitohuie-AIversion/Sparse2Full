#!/bin/bash
export PYTHONUNBUFFERED=1
mkdir -p runs_3loss_ablation_unet/A2_RecSpec
nohup torchrun --nproc_per_node=2 tools/training/train_real_data_ar.py \
    --config thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml \
    model.name=unet \
    training.epochs=50 \
    training.checkpoint.save_every_n_epochs=10 \
    training.loss_weights.spectral=0.05 \
    training.loss_weights.data_consistency=0.0 \
    experiment.name="3loss_ablation/A2_RecSpec" \
    experiment.output_dir="runs_3loss_ablation_unet/A2_RecSpec" \
    logging.log_model=true > runs_3loss_ablation_unet/nohup_A2_final.log 2>&1 &
echo "Process ID: $!"
