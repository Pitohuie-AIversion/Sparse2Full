"""
Verification script for ConvTemporalPredictor.
Verifies:
1. Shape correctness (Topology preservation)
2. Gradient flow (Backprop works)
3. Integration with SequentialSpatiotemporalModel
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.temporal.components.conv_temporal import ConvTemporalPredictor
from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporalModel

def verify_conv_temporal():
    print("="*60)
    print("🔍 Verifying ConvTemporalPredictor")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Unit Test: ConvTemporalPredictor
    print("\n[1] Unit Test: ConvTemporalPredictor")
    B, T_in, C, H, W = 2, 5, 4, 32, 32
    out_C = 1
    T_out = 3
    
    model = ConvTemporalPredictor(
        in_channels=C,
        hidden_channels=16,
        out_channels=out_C,
        num_layers=2,
        kernel_size=3
    ).to(device)
    
    x = torch.randn(B, T_in, C, H, W).to(device)
    
    print(f"   Input: {x.shape}")
    try:
        y = model(x, T_out=T_out)
        print(f"   Output: {y.shape}")
        
        expected_shape = (B, T_out, out_C, H, W)
        if y.shape == expected_shape:
            print("   ✅ Shape Correct")
        else:
            print(f"   ❌ Shape Mismatch! Expected {expected_shape}")
            return
            
    except Exception as e:
        print(f"   ❌ Forward failed: {e}")
        return

    # 2. Integration Test: SequentialSpatiotemporalModel
    print("\n[2] Integration Test: SequentialSpatiotemporalModel with backend='conv_rnn'")
    
    spatial_config = {
        'in_channels': 1,
        'out_channels': 1,
        'spatial_feature_dim': 4, # Small feature dim
        'img_size': (32, 32),
        'backbone_type': 'simple_cnn',
        'backbone_config': {'hidden_channels': 8}
    }
    
    temporal_config = {
        'backend': 'conv_rnn', # New backend
        'spatial_feature_dim': 4,
        'temporal_dim': 16, # hidden_channels for ConvLSTM
        'num_layers': 2,
        'out_channels': 1,
        'img_size': (32, 32),
        'kernel_size': 3
    }
    
    data_config = {}
    
    full_model = SequentialSpatiotemporalModel(
        spatial_config, temporal_config, data_config, device=device
    ).to(device)
    
    # Input sequence
    x_seq = torch.randn(2, 5, 1, 32, 32).to(device)
    target_seq = torch.randn(2, 3, 1, 32, 32).to(device) # T_out=3
    
    print("   Running forward pass...")
    try:
        outputs = full_model(x_seq, target_seq)
        final_pred = outputs['final_pred']
        print(f"   Final Prediction: {final_pred.shape}")
        
        if final_pred.shape == (2, 3, 1, 32, 32):
             print("   ✅ Integration Shape Correct")
        else:
             print(f"   ❌ Integration Shape Mismatch! Got {final_pred.shape}")
             
        # Check gradients
        loss = nn.MSELoss()(final_pred, target_seq)
        loss.backward()
        
        grad_sum = 0
        for p in full_model.temporal_module.parameters():
            if p.grad is not None:
                grad_sum += p.grad.abs().sum().item()
        
        print(f"   Gradient Sum: {grad_sum:.6f}")
        if grad_sum > 0:
            print("   ✅ Gradient Flow Correct")
        else:
            print("   ❌ No Gradient Flow!")
            
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_conv_temporal()
