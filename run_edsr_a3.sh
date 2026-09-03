#!/bin/bash
python tools/training/train_real_data_ar.py --config thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml \
    model.name=edsr \
    training.epochs=100 \
    training.checkpoint.save_every_n_epochs=20 \
    training.loss_weights.spectral=0.5 \
    training.loss_weights.data_consistency=1.0 \
    experiment.name="Ablation-A3-Full-model_EDSR-s2025" \
    experiment.output_dir="runs_drd_paper/Ablation-A3-Full-model_EDSR-s2025"
