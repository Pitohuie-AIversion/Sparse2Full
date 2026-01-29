I will launch a "Negative Control" experiment on **CUDA 0** to demonstrate the failure case when low-quality SR inputs are used for temporal prediction.

**Plan:**

1.  **Create Config**:
    *   Duplicate `thesis_paper/configs/temporal/ar_training_config_video_swin.yaml` to `thesis_paper/configs/temporal/ar_training_config_video_swin_lowres.yaml`.
    *   **Simulate Poor SR**: Instead of using a real SR model (which takes time to train), I will re-enable the **`reduce_spatial: avgpool`** mechanism with `reduce_size: [32, 32]`.
    *   *Why*: This mathematically simulates a "bad SR" stage that outputs blurry 32x32 images, which are then upsampled to 128x128. This is exactly what caused the previous failure, and now we will document it as a formal baseline.
    *   Set `experiment.name` to `AR-ShallowWater-VideoSwin-LowRes-s2025`.

2.  **Launch Command**:
    *   Use `CUDA_VISIBLE_DEVICES=0`.
    *   Use `python` (single process).

**Goal**: Obtain a high `Rel-L2` and low `SSIM` to contrast with the `0.008` L2 we just achieved with high-res inputs. This comparison will go straight into your "Ablation Study" section.