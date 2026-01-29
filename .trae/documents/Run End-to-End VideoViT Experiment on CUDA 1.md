I have prepared the setup for the **Two-Stage Sequential Experiment** (Stage 1: EDSR, Stage 2: VideoViT) as requested.

**Completed Setup:**

1. **Code Modification**: I have updated `models/temporal/components/sequential_spatiotemporal.py` to support `EDSR` as a backbone in the sequential model. This allows extracting features from EDSR to feed into the temporal module.
2. **Configuration**: Created `thesis_paper/configs/ar_paper_sequential_edsr_videoswin_sr4.yaml` with the following settings:

   * **Phase 1 (Spatial)**: EDSR (n\_feats=64, n\_resblocks=32) trained for 100 epochs.

   * **Phase 2 (Temporal)**: VideoSwin (VideoViT) trained for 200 epochs on top of EDSR features.

   * **Total Epochs**: 300.

   * **Dataset**: DR2D (Diffusion-Reaction).

**Next Step:**
I will launch this experiment on **CUDA 1**.

**Command:**

```bash
CUDA_VISIBLE_DEVICES=1 python tools/training/train_real_data_ar.py --config thesis_paper/configs/ar_paper_sequential_edsr_videoswin_sr4.yaml
```

