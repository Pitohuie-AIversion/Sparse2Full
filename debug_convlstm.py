
import torch
import sys
import os

# Add project root
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from models.temporal.components.conv_temporal import ConvTemporalPredictor
    print("Successfully imported ConvTemporalPredictor")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

try:
    model = ConvTemporalPredictor(in_channels=64, hidden_channels=64, out_channels=1)
    print(f"Model created. in_channels: {model.in_channels}")
except AttributeError as e:
    print(f"Creation failed: {e}")
except Exception as e:
    print(f"Other error: {e}")

x = torch.randn(1, 4, 64, 32, 32)
try:
    y = model(x, T_out=1)
    print(f"Forward pass success. Output shape: {y.shape}")
except Exception as e:
    print(f"Forward pass failed: {e}")
