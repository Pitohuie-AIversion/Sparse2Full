"""
Convolutional Recurrent Neural Network (ConvRNN) for Spatiotemporal Prediction.

This module implements a ConvLSTM-based temporal predictor that preserves spatial topology.
Unlike the flattened Linear/Transformer approach, this module treats the input as a 
sequence of 2D feature maps and evolves them using 2D convolutions.

Key Features:
- Preserves spatial structure (no flattening)
- Captures local motion and flow (advection)
- Parameter efficiency (shared weights across space)
- Resolution independent (can handle varying image sizes)
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell as described in "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Convolution for input-to-state and state-to-state transitions
        # We concatenate input and hidden state along the channel dimension
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # 4 gates: input, forget, output, cell
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, C, H, W]
            state: Tuple of (h_prev, c_prev), each [B, hidden_channels, H, W]
        Returns:
            (h_next, c_next)
        """
        if state is None:
            # Initialize states with zeros if not provided
            batch_size, _, height, width = x.size()
            h_prev = torch.zeros(
                batch_size, self.hidden_channels, height, width, device=x.device
            )
            c_prev = torch.zeros(
                batch_size, self.hidden_channels, height, width, device=x.device
            )
        else:
            h_prev, c_prev = state

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # [B, in_c + hidden_c, H, W]

        # Apply convolution
        gates = self.conv(combined)

        # Split into 4 gates: input, forget, output, cell_candidate
        ingate, forgetgate, outgate, cellgate = torch.chunk(gates, 4, dim=1)

        # Apply activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        outgate = torch.sigmoid(outgate)
        cellgate = torch.tanh(cellgate)

        # Update cell state
        c_next = (forgetgate * c_prev) + (ingate * cellgate)

        # Update hidden state
        h_next = outgate * torch.tanh(c_next)

        return h_next, c_next


class ConvTemporalPredictor(nn.Module):
    """
    Multi-layer ConvLSTM for Spatiotemporal Prediction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Build ConvLSTM layers
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = in_channels if i == 0 else hidden_channels
            self.cell_list.append(
                ConvLSTMCell(
                    in_channels=cur_input_dim,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                )
            )

        # Output projection layer (1x1 conv to map hidden state to output)
        self.output_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        T_out: int = 1,
        future_forcing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, T_in, C, H, W]
            T_out: Number of future steps to predict
            future_forcing: Optional forcing terms for future steps (not implemented here but kept for API compat)
        Returns:
            predictions: [B, T_out, out_channels, H, W]
        """
        B, T_in, C, H, W = x.shape

        # Initialize internal states for all layers
        # List of (h, c) tuples, one for each layer
        states = [None] * self.num_layers

        # 1. Encoder Phase: Process input sequence
        # We only care about the final state after seeing the input sequence
        # (Seq2Seq architecture could be used here, but simple AR is fine for now)

        # To save memory, we don't store all intermediate hidden states of the encoder
        # unless we want to do dense prediction (many-to-many).
        # Assuming we want to predict T_out steps AFTER T_in inputs.

        # Feed input sequence
        for t in range(T_in):
            current_input = x[:, t]  # [B, C, H, W]

            for i in range(self.num_layers):
                h, c = self.cell_list[i](current_input, states[i])
                states[i] = (h, c)
                # Next layer input is current layer output
                current_input = self.dropout(h)

        # 2. Decoder/Prediction Phase
        predictions = []

        # The input to the first decoder step is the last frame of the input sequence
        # However, the encoder loop ends with 'current_input' being the hidden state of the LAST layer.
        # But the FIRST layer of the decoder needs an input of 'in_channels'.

        # Correct approach:
        # The last observed frame x[:, -1] should be the input to the first decoder step.
        # (or the last feature map if we are processing features)
        decoder_input = x[:, -1]

        for t in range(T_out):
            # Pass through layers
            current_feature = decoder_input

            # Update states for next step
            next_states = []
            for i in range(self.num_layers):
                h, c = self.cell_list[i](current_feature, states[i])
                next_states.append((h, c))
                current_feature = self.dropout(h)

            states = next_states

            # Project to output space
            pred_frame = self.output_proj(current_feature)  # [B, out_c, H, W]
            predictions.append(pred_frame)

            # Prepare input for next step (Autoregressive)
            if self.in_channels == pred_frame.shape[1]:
                decoder_input = pred_frame
            else:
                # If dimensions mismatch, use zero input as fallback
                # This happens when we feed [SpatialPred, Features] but only output [SpatialPred]
                decoder_input = torch.zeros(B, self.in_channels, H, W, device=x.device)

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [B, T_out, out_c, H, W]

        return predictions
