#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

def load_model_from_path(model_path, model_name=None):
    # Try to import as a module first if inside current directory
    current_dir = Path.cwd()
    abs_model_path = Path(model_path).resolve()
    
    if current_dir in abs_model_path.parents:
        try:
            rel_path = abs_model_path.relative_to(current_dir)
            module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
            print(f"ℹ️  Attempting to import as module: {module_name}")
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            module = importlib.import_module(module_name)
        except (ImportError, ValueError) as e:
            print(f"⚠️  Module import failed ({e}), falling back to file import.")
            module = None
    else:
        module = None

    if module is None:
        # Dynamic import from file path
        spec = importlib.util.spec_from_file_location("dynamic_model", model_path)
        if spec is None or spec.loader is None:
             raise ValueError(f"Could not load spec from {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["dynamic_model"] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            raise ImportError(f"Failed to load file '{model_path}' due to import error: {e}.\nHint: Relative imports require the file to be loaded as a module (use path relative to project root).")

    if model_name:
        if hasattr(module, model_name):
            return getattr(module, model_name)
        else:
            raise ValueError(f"Model class '{model_name}' not found in {model_path}")
    
    # Try to find a nn.Module class
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            # Skip imported modules if possible, simplistic heuristic
            if obj.__module__ == "dynamic_model":
                return obj
    
    # Fallback: return the first found module
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            return obj

    raise ValueError(f"No nn.Module class found in {model_path}")

def check_interface(model_cls):
    print(f"Checking interface for {model_cls.__name__}...")
    import inspect
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters
    
    has_in = 'in_ch' in params or 'in_channels' in params
    has_out = 'out_ch' in params or 'out_channels' in params
    has_img = 'img_size' in params
    
    if not has_in:
        print(f"❌ __init__ missing input channel arg (expected 'in_ch' or 'in_channels')")
    if not has_out:
        print(f"❌ __init__ missing output channel arg (expected 'out_ch' or 'out_channels')")
    if not has_img:
        print(f"❌ __init__ missing 'img_size' arg")
        
    if has_in and has_out and has_img:
        print(f"✅ __init__ signature accepted.")
        return True
    return False

def check_forward_and_resources(model_cls):
    in_ch = 3
    out_ch = 3
    img_size = 64
    batch_size = 1
    
    # Determine correct arguments
    import inspect
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters
    
    kwargs = {'img_size': img_size}
    if 'in_ch' in params:
        kwargs['in_ch'] = in_ch
    elif 'in_channels' in params:
        kwargs['in_channels'] = in_ch
    else:
        # Fallback to in_ch (expecting kwargs support)
        kwargs['in_ch'] = in_ch
        
    if 'out_ch' in params:
        kwargs['out_ch'] = out_ch
    elif 'out_channels' in params:
        kwargs['out_channels'] = out_ch
    else:
        # Fallback
        kwargs['out_ch'] = out_ch
    
    try:
        model = model_cls(**kwargs)
    except Exception as e:
        print(f"❌ Failed to instantiate model: {e}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Check params
    params = sum(p.numel() for p in model.parameters())
    print(f"ℹ️  Parameters: {params / 1e6:.2f}M")
    
    if params > 10 * 1e6:
        print(f"⚠️  Parameters exceed 10M limit!")
    else:
        print(f"✅ Parameters within limit (<=10M).")

    # Check forward
    x = torch.randn(batch_size, in_ch, img_size, img_size).to(device)
    try:
        with torch.no_grad():
            y = model(x)
        print(f"✅ Forward pass successful.")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
        
    # Check output shape
    if y.shape != (batch_size, out_ch, img_size, img_size):
        print(f"⚠️  Output shape mismatch. Expected {(batch_size, out_ch, img_size, img_size)}, got {y.shape}")
    else:
        print(f"✅ Output shape correct.")
        
    # Check FLOPs (using thop if available)
    try:
        from thop import profile
        # Suppress thop printing
        import contextlib
        import io
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            flops, _ = profile(model, inputs=(x,), verbose=False)
        print(f"ℹ️  FLOPs: {flops / 1e9:.2f}G")
    except ImportError:
        print(f"ℹ️  'thop' not installed, skipping FLOPs check.")
    except Exception as e:
        print(f"⚠️  FLOPs calculation failed: {e}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the python file containing the model")
    parser.add_argument("--name", help="Name of the model class (optional)")
    args = parser.parse_args()
    
    try:
        model_cls = load_model_from_path(args.model_path, args.name)
        if check_interface(model_cls):
            check_forward_and_resources(model_cls)
    except Exception as e:
        print(f"❌ Error: {e}")
