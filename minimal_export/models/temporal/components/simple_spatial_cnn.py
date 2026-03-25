import torch
import torch.nn as nn


class SimpleSpatialCNN(nn.Module):
    """Simple CNN-based spatial backbone for temporal modeling testing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        padding: int = 1,
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU(inplace=True)

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # Build CNN layers
        layers = []
        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels
            out_ch = hidden_channels

            # Convolution
            conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
            layers.append(conv)

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))

            # Activation
            layers.append(self.activation)

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        self.cnn_layers = nn.ModuleList(layers)

        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C_out, H, W]
        """
        # Input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[SimpleSpatialCNN] Warning: NaN/Inf in input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # Store original shape
        B, C, H, W = x.shape

        # Input projection
        x = self.input_proj(x)

        # Apply CNN layers
        for i, layer in enumerate(self.cnn_layers):
            x_prev = x
            x = layer(x)

            # Add residual connection every 2 layers
            if i % 2 == 1 and x_prev.shape == x.shape:
                x = x + x_prev

            # Check for numerical issues
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[SimpleSpatialCNN] Warning: NaN/Inf after layer {i}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # Output projection
        x = self.output_proj(x)

        # Final numerical check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[SimpleSpatialCNN] Warning: NaN/Inf in final output")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        return x


class SimpleSpatialTemporalCNN(nn.Module):
    """Simple CNN that can handle both spatial and basic temporal modeling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_spatial_layers: int = 3,
        num_temporal_layers: int = 2,
        kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Spatial feature extraction
        self.spatial_cnn = SimpleSpatialCNN(
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_spatial_layers,
            kernel_size=kernel_size,
            **kwargs,
        )

        # Simple temporal modeling (1D conv along time dimension)
        self.temporal_conv = nn.ModuleList(
            [
                nn.Conv3d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0),
                )
                for _ in range(num_temporal_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal modeling.

        Args:
            x: Input tensor [B, T, C, H, W] or [B, C, H, W]

        Returns:
            Output tensor [B, C_out, H, W] or [B, T, C_out, H, W]
        """
        # Handle different input shapes
        if x.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = x.shape

            # Process each timestep
            spatial_features = []
            for t in range(T):
                feat = self.spatial_cnn(x[:, t])  # [B, hidden_channels, H, W]
                spatial_features.append(feat)

            # Stack spatial features [B, T, hidden_channels, H, W]
            spatial_features = torch.stack(spatial_features, dim=1)

            # Apply temporal convolution
            temporal_features = spatial_features.permute(
                0, 2, 1, 3, 4
            )  # [B, hidden_channels, T, H, W]
            for conv in self.temporal_conv:
                temporal_features = self.activation(conv(temporal_features))

            # Take last timestep or average
            temporal_features = temporal_features.permute(
                0, 2, 1, 3, 4
            )  # [B, T, hidden_channels, H, W]
            final_features = temporal_features[:, -1]  # [B, hidden_channels, H, W]

            # Output projection
            output = self.output_proj(final_features)

        else:  # [B, C, H, W]
            # Simple spatial processing
            features = self.spatial_cnn(x)
            output = self.output_proj(features)

        return output


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test SimpleSpatialCNN
    print("Testing SimpleSpatialCNN...")
    model = SimpleSpatialCNN(in_channels=3, out_channels=1, hidden_channels=32).to(
        device
    )
    x = torch.randn(2, 3, 64, 64).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")

    # Test SimpleSpatialTemporalCNN
    print("\nTesting SimpleSpatialTemporalCNN...")
    model_temporal = SimpleSpatialTemporalCNN(
        in_channels=3, out_channels=1, hidden_channels=32
    ).to(device)
    x_temporal = torch.randn(2, 5, 3, 64, 64).to(device)  # [B, T, C, H, W]
    with torch.no_grad():
        output_temporal = model_temporal(x_temporal)
    print(f"Input shape: {x_temporal.shape}, Output shape: {output_temporal.shape}")

    print("All tests passed!")
