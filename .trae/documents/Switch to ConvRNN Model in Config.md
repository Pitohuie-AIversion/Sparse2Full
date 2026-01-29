Based on the SOTA research and your request, I will modify the configuration file to use the newly implemented **`conv_rnn`** (ConvLSTM) backend. This is the recommended "Tier 1" model for capturing spatial dynamics.

### Plan:
1.  **Read** the current content of `thesis_paper/configs/temporal/ar_training_config_debug_temporal_shallowwater.yaml`.
2.  **Edit** the configuration to:
    *   Change `temporal.backend` to `conv_rnn`.
    *   Set appropriate hyperparameters for ConvLSTM (hidden_dim, num_layers, kernel_size).
    *   Ensure `spatial_feature_dim` aligns with the input.
    *   Update the experiment name to reflect the model change (`AR-ShallowWater-ConvRNN`).
3.  **Verify** the configuration syntax.

I will now execute these steps.