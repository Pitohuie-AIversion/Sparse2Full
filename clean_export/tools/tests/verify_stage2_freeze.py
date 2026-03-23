#!/usr/bin/env python3
"""
Verification script for Stage 2 (Sequential) Training - Freeze Logic
This script empirically verifies that:
1. The SequentialSpatiotemporalModel can be initialized.
2. When 'freeze_spatial' is simulated, spatial parameters do NOT receive gradients.
3. Temporal parameters DO receive gradients.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

print(f"Project root added to sys.path: {project_root}")
print(f"sys.path[0]: {sys.path[0]}")
try:
    import models
    print(f"Imported models from: {models.__file__}")
except ImportError as e:
    print(f"Failed to import models: {e}")

from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporalModel

def verify_stage2_freeze():
    print("="*60)
    print("🔍 Verifying Stage 2 Freeze Logic")
    print("="*60)

    # 1. Setup Dummy Configuration
    spatial_config = {
        'in_channels': 1,
        'out_channels': 1,
        'spatial_feature_dim': 16,
        'img_size': (32, 32),
        'backbone_type': 'simple_cnn',  # Use simple CNN for testing
        'backbone_config': {'hidden_channels': 8}
    }
    
    temporal_config = {
        'spatial_feature_dim': 16,
        'temporal_dim': 32,
        'num_layers': 2,
        'out_channels': 1,
        'img_size': (32, 32),
        'backend': 'transformer'
    }
    
    data_config = {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 2. Initialize Model
    model = SequentialSpatiotemporalModel(
        spatial_config, temporal_config, data_config, device=device
    ).to(device)
    
    print("\n✅ Model initialized successfully.")
    
    # 3. Simulate Freezing Logic (as in train_real_data_ar.py)
    print("\n❄️  Freezing Spatial Module...")
    if hasattr(model, 'spatial_module'):
        for p in model.spatial_module.parameters():
            p.requires_grad = False
    
    # Verify requires_grad flags
    spatial_trainable = any(p.requires_grad for p in model.spatial_module.parameters())
    temporal_trainable = any(p.requires_grad for p in model.temporal_module.parameters())
    
    print(f"   Spatial parameters trainable? {spatial_trainable} (Expected: False)")
    print(f"   Temporal parameters trainable? {temporal_trainable} (Expected: True)")
    
    if spatial_trainable:
        print("❌ Error: Spatial parameters are still trainable!")
        return
    
    # 4. Forward & Backward Pass
    print("\n🔄 Running Forward & Backward Pass...")
    B, T, C, H, W = 2, 5, 1, 32, 32
    x = torch.randn(B, T, C, H, W).to(device)
    target = torch.randn(B, T, C, H, W).to(device)
    
    # Forward
    outputs = model(x, target)
    pred = outputs['final_pred']
    
    # Loss
    loss = nn.MSELoss()(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # 5. Check Gradients
    print("\ngradient Check:")
    
    # Check Spatial Gradients
    spatial_grads = []
    for name, p in model.spatial_module.named_parameters():
        if p.grad is not None:
            spatial_grads.append(p.grad.abs().sum().item())
            
    spatial_grad_sum = sum(spatial_grads)
    print(f"   Spatial Gradient Sum: {spatial_grad_sum:.6f} (Expected: 0.0 or None)")
    
    if spatial_grad_sum > 1e-6:
        print("❌ FAILED: Spatial module received non-zero gradients!")
    elif len(spatial_grads) > 0 and spatial_grad_sum == 0:
        print("✅ SUCCESS: Spatial module gradients are explicitly zero.")
    else:
        print("✅ SUCCESS: Spatial module gradients are None (not computed).")

    # Check Temporal Gradients
    temporal_grads = []
    for name, p in model.temporal_module.named_parameters():
        if p.grad is not None:
            temporal_grads.append(p.grad.abs().sum().item())
            
    temporal_grad_sum = sum(temporal_grads)
    print(f"   Temporal Gradient Sum: {temporal_grad_sum:.6f} (Expected: > 0)")
    
    if temporal_grad_sum > 0:
        print("✅ SUCCESS: Temporal module received gradients.")
    else:
        print("❌ FAILED: Temporal module did not receive gradients!")

if __name__ == "__main__":
    verify_stage2_freeze()
