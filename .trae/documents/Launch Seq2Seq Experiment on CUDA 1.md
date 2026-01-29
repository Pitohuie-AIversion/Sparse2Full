I will launch a second parallel experiment on **CUDA 1** to train a **Seq2Seq (Many-to-Many)** version of the Video Swin model. This will provide a direct comparison: "Predict Next Frame" (CUDA 0) vs "Predict Future Sequence" (CUDA 1).

**Plan:**

1.  **Create Config**:
    *   Duplicate `thesis_paper/configs/temporal/ar_training_config_video_swin.yaml` to `thesis_paper/configs/temporal/ar_training_config_video_swin_seq2seq.yaml`.
    *   Modify `data.T_out`: Change from `1` to **`10`** (predict next 10 frames at once).
    *   Modify `experiment.name`: Append `_seq2seq` to distinguish logs.
    *   Modify `device.devices`: Set to `1` (we will control ID via env var).

2.  **Verify Model Compatibility**:
    *   The `VideoSwinPredictor` implementation currently has a logic check: `if T_out == 1: ... else: ... repeat(...)`.
    *   **Crucial Step**: I need to ensure the model doesn't just *repeat* the same frame 10 times. The current implementation likely does a naive repeat for `T_out > 1`.
    *   **Refinement**: For true Seq2Seq, the model should output a temporal dimension > 1. I will check `models/temporal/components/video_swin.py` and potentially enhance it to support direct multi-frame output (e.g., using a 3D Conv head that outputs `T_out` channels or depth). *However, given the constraint of not changing the running experiment on CUDA 0, I will check if the current code supports it safely.*
    *   *Self-Correction*: The current `VideoSwinPredictor` output projection is `nn.Conv3d(hidden_dim, out_channels, kernel_size=1)`. This outputs the same temporal depth as the *input* (or whatever the encoder preserves). If we want to predict 10 future frames from 30 past frames, we typically need a **Decoder**.
    *   **Alternative Strategy for Seq2Seq**: Instead of a full architectural change, I will set `T_out=10` but be aware that without a specific decoder, the model might struggle or I might need to adjust the `forward` to output `T_out` frames properly.
    *   *Decision*: To keep it simple and robust for now, I will stick to **Autoregressive (AR) Rollout** training if the architecture is Encoder-only. But you asked for "Seq2Seq".
    *   **Better Approach**: Let's configure it as **AR Rollout Training**.
        *   `T_out: 10`
        *   `training.rollout_training: true`
        *   The system will automatically run the model 10 times recursively during training to compute the loss on the sequence. This IS Seq2Seq training for an AR model.

3.  **Launch Command**:
    *   Use `CUDA_VISIBLE_DEVICES=1`.
    *   Use `python` (single process) to avoid conflict with the running `torchrun`.

**Action**: I will create the config and launch the AR Rollout (Seq2Seq) experiment on GPU 1.