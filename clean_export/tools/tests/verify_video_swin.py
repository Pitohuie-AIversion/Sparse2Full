"""
Verification script for Video Swin Transformer integration.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.temporal.components.video_swin import VideoSwinPredictor
from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporalModel

def verify_video_swin():
    print("="*60)
    print("🔍 Verifying VideoSwinPredictor")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Unit Test
    print("\n[1] Unit Test: VideoSwinPredictor")
    B, T_in, C, H, W = 2, 4, 3, 32, 32
    model = VideoSwinPredictor(
        in_channels=C,
        hidden_dim=24,
        out_channels=1,
        num_layers=2,
        window_size=(2, 8, 8)
    ).to(device)
    
    x = torch.randn(B, T_in, C, H, W).to(device)
    try:
        y = model(x, T_out=2) # Predict 2 steps
        print(f"   Input: {x.shape}")
        print(f"   Output: {y.shape}")
        
        expected = (B, 2, 1, H, W)
        if y.shape == expected:
            print("   ✅ Shape Correct")
        else:
            print(f"   ❌ Shape Mismatch! Expected {expected}")
            return
    except Exception as e:
        print(f"   ❌ Forward failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Integration Test
    print("\n[2] Integration Test: SequentialSpatiotemporalModel")
    spatial_config = {
        'in_channels': 1,
        'out_channels': 1,
        'spatial_feature_dim': 0, # Identity
        'img_size': (32, 32),
        'backbone_type': 'identity'
    }
    temporal_config = {
        'backend': 'video_swin',
        'temporal_dim': 24,
        'num_layers': 2,
        'out_channels': 1,
        'img_size': (32, 32),
        'window_size': (2, 8, 8)
    }
    
    full_model = SequentialSpatiotemporalModel(
        spatial_config, temporal_config, {}, device=device
    ).to(device)
    
    x_seq = torch.randn(2, 4, 1, 32, 32).to(device)
    target = torch.randn(2, 2, 1, 32, 32).to(device)
    
    try:
        out = full_model(x_seq, target)
        pred = out['final_pred']
        print(f"   Integration Output: {pred.shape}")
        
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        
        grad_sum = sum(p.grad.abs().sum() for p in full_model.temporal_module.parameters() if p.grad is not None)
        print(f"   Gradient Sum: {grad_sum:.4f}")
        
        if grad_sum > 0:
            print("   ✅ Gradient Flow Correct")
        else:
            print("   ❌ No Gradient Flow")
            
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_video_swin()
