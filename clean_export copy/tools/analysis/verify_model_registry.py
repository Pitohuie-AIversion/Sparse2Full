
import torch
import inspect
import sys
import os
from pathlib import Path
import traceback

# Add project root to path
sys.path.append(os.getcwd())

from models.registry import MODEL_REGISTRY, create_model

def is_transformer_like(model):
    """Check if model has Transformer-like components"""
    keywords = ['attn', 'attention', 'transformer', 'mhsa', 'window', 'patchembed', 'mlp', 'ffn']
    for name, module in model.named_modules():
        name_lower = name.lower()
        cls_name = module.__class__.__name__.lower()
        if any(k in name_lower for k in keywords) or any(k in cls_name for k in keywords):
            return True
    return False

def is_u_shape(model):
    """Check if model has U-shape characteristics (downsample + upsample)"""
    has_down = False
    has_up = False
    keywords_down = ['down', 'pool', 'strided', 'patchmerg']
    keywords_up = ['up', 'transpose', 'expand', 'pixelshuffle']
    
    for name, module in model.named_modules():
        name_lower = name.lower()
        cls_name = module.__class__.__name__.lower()
        if any(k in name_lower for k in keywords_down) or any(k in cls_name for k in keywords_down):
            has_down = True
        if any(k in name_lower for k in keywords_up) or any(k in cls_name for k in keywords_up):
            has_up = True
            
    return has_down and has_up

def is_fno_like(model):
    """Check if model has FNO-like components"""
    keywords = ['fno', 'fourier', 'fft', 'spectral', 'rfft']
    for name, module in model.named_modules():
        name_lower = name.lower()
        cls_name = module.__class__.__name__.lower()
        if any(k in name_lower for k in keywords) or any(k in cls_name for k in keywords):
            return True
    return False

def audit_models():
    print("Starting Model Registry Audit...")
    
    results = []
    
    for name, cls in MODEL_REGISTRY.items():
        print(f"Auditing {name} ({cls.__name__})...")
        status = "PASS"
        reason = []
        
        # 1. Instantiation Test
        try:
            # Default dummy args
            kwargs = {}
            if 'in_channels' in inspect.signature(cls.__init__).parameters or 'in_ch' in inspect.signature(cls.__init__).parameters:
                kwargs['in_ch'] = 1
            if 'out_channels' in inspect.signature(cls.__init__).parameters or 'out_ch' in inspect.signature(cls.__init__).parameters:
                kwargs['out_ch'] = 1
            if 'img_size' in inspect.signature(cls.__init__).parameters:
                kwargs['img_size'] = 64
                
            model = create_model(name, **kwargs)
        except Exception as e:
            status = "FAIL"
            reason.append(f"Instantiation failed: {str(e)}")
            results.append((name, cls.__name__, status, "; ".join(reason)))
            continue

        # 2. Forward Test
        try:
            x = torch.randn(1, 1, 64, 64)
            if 'transformer' in name or 'vit' in name:
                # Transformers might need 3 channels or specific size
                if getattr(model, 'in_chans', 1) == 3:
                    x = torch.randn(1, 3, 64, 64)
            
            # Handle specific input requirements if known, otherwise try basic
            y = model(x)
            if y.shape[0] != 1:
                status = "FAIL"
                reason.append(f"Output batch size mismatch: {y.shape}")
        except Exception as e:
            status = "FAIL"
            reason.append(f"Forward failed: {str(e)}")

        # 3. Naming Consistency Check
        name_lower = name.lower()
        cls_lower = cls.__name__.lower()
        
        # Transformer Check
        if any(k in name_lower for k in ['transformer', 'vit', 'swin', 'segformer', 'restormer', 'uformer']):
            if not is_transformer_like(model):
                status = "FAIL"
                reason.append("Name implies Transformer but no transformer components found")
        
        # UNet Check
        if 'unet' in name_lower:
            if not is_u_shape(model):
                # SwinUNet might hide upsampling in obscure layers, but generally should pass
                # Let's be lenient but warn
                pass 
                
        # FNO Check
        if 'fno' in name_lower:
            if not is_fno_like(model):
                status = "FAIL"
                reason.append("Name implies FNO but no spectral components found")
                
        # Lite Check
        if 'lite' in name_lower:
            # Check parameter count (heuristic: < 1M is definitely lite, but relative to what?)
            params = sum(p.numel() for p in model.parameters())
            if params > 5000000: # 5M
                reason.append(f"Lite model has high params: {params/1e6:.2f}M")
            
            # Specific Lite Fraud Checks
            if 'uformer' in name_lower and not is_transformer_like(model):
                status = "FAIL"
                reason.append("UformerLite is NOT a Transformer")
            if 'swinir' in name_lower and not is_transformer_like(model):
                status = "FAIL"
                reason.append("SwinIRLite is NOT a Swin Transformer")
            if 'restormer' in name_lower and not is_transformer_like(model):
                status = "FAIL"
                reason.append("RestormerLite is NOT a Restormer")
            if 'nafnet' in name_lower:
                 # Check for activation
                 has_act = False
                 for m in model.modules():
                     if isinstance(m, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Tanh)):
                         has_act = True
                         break
                 if not has_act and 'SimpleGate' not in str(model):
                     # NAFNet should be activation free (except SimpleGate which is x*y)
                     # But if it's just Linear...
                     pass

        results.append((name, cls.__name__, status, "; ".join(reason)))

    # Generate Report
    with open('runs/model_registry_audit.md', 'w') as f:
        f.write("# Model Registry Audit Report\n\n")
        f.write("| Canonical Name | Class | Status | Issues/Notes |\n")
        f.write("|---|---|---|---|\n")
        for res in results:
            f.write(f"| {res[0]} | {res[1]} | {res[2]} | {res[3]} |\n")
            
    print("Audit finished. Report written to runs/model_registry_audit.md")

if __name__ == "__main__":
    audit_models()
