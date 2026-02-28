import argparse
import os
import sys
import torch
import subprocess
import shutil
import json
import traceback
from torchinfo import summary
import check_overlaps
import jinja2

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Template Engine Setup ---
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)

def render_template(template_name, context, output_path):
    template = JINJA_ENV.get_template(template_name)
    with open(output_path, 'w') as f:
        f.write(template.render(**context))

# --- Model Builder ---

def build_model(model_name: str, in_ch: int, out_ch: int, img: int, upscale: int):
    # 0. Manual Mapping for tricky names
    model_map = {
        'swin_unet': ('models.spatial.swin_unet', 'SwinUNet'),
        'mlp_mixer': ('models.spatial.mlp_mixer', 'MLPMixer'),
        'ufno': ('models.spatial.ufno_unet_bottleneck', 'UFNOUNet'),
        'fno': ('models.spatial.fno2d', 'FNO2d'),
        'conv_gate_lite': ('models.spatial.conv_gate_lite', 'ConvGateLite'),
        'conv_unet_lite': ('models.spatial.conv_unet_lite', 'ConvUNetLite'),
        'cnn_attn_lite': ('models.spatial.cnn_attn_lite', 'CNNAttnLite'),
        'perceiverio': ('models.spatial.perceiverio', 'PerceiverIO'),
        'uno': ('models.spatial.uno', 'UNO'),
        'nafnet': ('models.spatial.nafnet', 'NAFNet'),
        'restormer': ('models.spatial.restormer', 'Restormer'),
        'vit': ('models.spatial.vit', 'ViT'),
        'swinir': ('models.spatial.swinir', 'SwinIR'),
        'rdn': ('models.spatial.rdn', 'RDN'),
        'rcan': ('models.spatial.rcan', 'RCAN'),
        'unetformer': ('models.spatial.unetformer', 'UNetFormer'),
        'unet_plus_plus': ('models.spatial.unet_plus_plus', 'UNetPlusPlus'),
        'partialconv_unet': ('models.spatial.partialconv_unet', 'PartialConvUNet'),
        'resnet_lite': ('models.spatial.resnet', 'ResNetLite'),
        'sparse_swin_unet': ('models.spatial.sparse_attention_encoder', 'SparseSwinUNet'),
        'swin_t_with_encoder': ('models.spatial.swin_t_with_encoder', 'SwinTWithEncoder'),
        'swin_t': ('models.spatial.swin_t', 'SwinTransformerTiny'),
        # Temporal models
        'convlstm': ('models.temporal.components.conv_temporal', 'ConvTemporalPredictor'),
        'swin_temporal': ('models.temporal.wrappers.swin_temporal', 'SwinTemporal'),
        'videoswin': ('models.temporal.components.video_swin', 'VideoSwinPredictor'),
        'physics_transformer': ('models.temporal.models.physics_transformer', 'PhysicsTransformerTemporal'),
        'physics': ('models.temporal.models.physics_transformer', 'PhysicsTransformerTemporal'),
        'sequential': ('models.temporal.components.sequential_spatiotemporal', 'SequentialSpatiotemporalModel'),
        'deeponet': ('models.spatial.deeponet', 'DeepONet'),
        
        # Aliases
        'u-fno': ('models.spatial.ufno_unet_bottleneck', 'UFNOUNet'),
        'swin': ('models.spatial.swin_t', 'SwinTransformerTiny'),
        'swint': ('models.spatial.swin_t', 'SwinTransformerTiny'),
        'mixer': ('models.spatial.mlp_mixer', 'MLPMixer'),
    }
    
    if model_name.lower() in model_map:
        mod_path, cls_name = model_map[model_name.lower()]
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            
            # Special handling for SwinTemporal which needs nested config
            if cls_name == 'SwinTemporal':
                base_kwargs = {'in_channels': in_ch, 'out_channels': out_ch, 'img_size': img}
                temporal_cfg = {'enabled': True, 'type': 'conv1d'}
                return cls(base_kwargs=base_kwargs, temporal_cfg=temporal_cfg)

            # Special handling for SequentialSpatiotemporalModel
            if cls_name == 'SequentialSpatiotemporalModel':
                 spatial_cfg = {'in_channels': in_ch, 'spatial_feature_dim': 64, 'backbone_type': 'simple', 'img_size': (img, img)}
                 temporal_cfg = {'temporal_dim': 64, 'num_layers': 2, 'out_channels': out_ch, 'img_size': (img, img)}
                 data_cfg = {}
                 return cls(spatial_config=spatial_cfg, temporal_config=temporal_cfg, data_config=data_cfg)
                
            # Generic handling for others
            kwargs = dict(in_channels=in_ch, out_channels=out_ch)
            
            import inspect
            sig = inspect.signature(cls.__init__)
            
            if 'img_size' in sig.parameters:
                kwargs['img_size'] = img
            if 'upscale' in sig.parameters:
                kwargs['upscale'] = upscale
            if 'hidden_channels' in sig.parameters:
                kwargs['hidden_channels'] = 64
            if 'hidden_dim' in sig.parameters:
                kwargs['hidden_dim'] = 64
                
            return cls(**kwargs)
        except Exception as e:
            print(f"Manual loading failed for {model_name}: {e}")
            # Fallthrough to auto discovery

    # 1. Try registry first
    try:
        from models.registry import _model_entrypoints
        for k in _model_entrypoints.keys():
            if k.lower() == model_name.lower():
                create_fn = _model_entrypoints[k]
                kwargs = dict(in_channels=in_ch, out_channels=out_ch, img_size=img)
                kwargs['upscale'] = upscale
                return create_fn(**kwargs)
    except ImportError:
        pass

    # 2. Fallback to models.spatial inspection
    try:
        from models import spatial
    except ImportError:
        raise ValueError("Could not import models.spatial")

    cls = None
    for k, v in spatial.__dict__.items():
        if k.lower() == model_name.lower():
            cls = v
            break
    
    if cls is None:
        try:
            import importlib
            mod = importlib.import_module(f"models.spatial.{model_name.lower()}")
            for k, v in mod.__dict__.items():
                if k.lower() == model_name.lower():
                    cls = v
                    break
            
            if cls is None:
                # Try adding "Model" suffix
                suffix_name = model_name + "Model"
                for k, v in mod.__dict__.items():
                    if k.lower() == suffix_name.lower():
                        cls = v
                        break
        except ImportError:
            pass

    if cls is None:
        raise ValueError(f"Cannot find model '{model_name}'")
    
    if isinstance(cls, type(sys)): # Module check
        if hasattr(cls, model_name):
             cls = getattr(cls, model_name)
        else:
             found = False
             for k, v in cls.__dict__.items():
                 if k.lower() == model_name.lower():
                     cls = v
                     found = True
                     break
             
             if not found:
                 # Try adding "Model" suffix
                 suffix_name = model_name + "Model"
                 for k, v in cls.__dict__.items():
                     if k.lower() == suffix_name.lower():
                         cls = v
                         found = True
                         break
             
             # If still not found, check for temporal models specifically
             if not found and model_name.lower() == 'convlstm':
                 try:
                     from models.temporal.base_temporal import ConvLSTM
                     cls = ConvLSTM
                 except ImportError:
                     pass

    kwargs = dict(in_channels=in_ch, out_channels=out_ch, img_size=img)
    
    # Special handling for temporal models that might need sequence length or other args
    import inspect
    sig = inspect.signature(cls.__init__)
    if 'seq_len' in sig.parameters:
        kwargs['seq_len'] = 10 # Default sequence length
    if 'hidden_dim' in sig.parameters and 'hidden_dim' not in kwargs:
        kwargs['hidden_dim'] = 64
        
    if 'upscale' in kwargs and model_name.lower() == 'unet':
         kwargs['upscale'] = upscale
    else:
         kwargs['upscale'] = upscale

    return cls(**kwargs)

# --- PlotNeuralNet Helpers ---

def setup_pnn_styles(base_dir, target_dir):
    """Copy PlotNeuralNet style files and nn_blocks.tex to the target directory."""
    vendor_layer_dir = os.path.join(base_dir, 'vendor', 'PlotNeuralNet', 'layers')
    styles = ['Ball.sty', 'Box.sty', 'RightBandedBox.sty']
    
    if not os.path.exists(vendor_layer_dir):
        print(f"Warning: PlotNeuralNet vendor directory not found at {vendor_layer_dir}")
        return

    # Copy Styles
    for s in styles:
        src = os.path.join(vendor_layer_dir, s)
        dst = os.path.join(target_dir, s)
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Failed to copy {s}: {e}")
        else:
            print(f"Warning: Style file {src} does not exist.")
            
    # Copy nn_blocks.tex
    src_blocks = os.path.join(base_dir, 'nn_blocks.tex')
    dst_blocks = os.path.join(target_dir, 'nn_blocks.tex')
    if os.path.exists(src_blocks):
        try:
            shutil.copy2(src_blocks, dst_blocks)
        except Exception as e:
             print(f"Failed to copy nn_blocks.tex: {e}")

# --- Registry ---

# Map model keys to (3D_template, 2D_template)
GENERATOR_REGISTRY = {
    'swin_unet': ('swin_unet_3d.tex.j2', 'swin_unet_2d.tex.j2'),
    'swinunet': ('swin_unet_3d.tex.j2', 'swin_unet_2d.tex.j2'),
    
    'edsr': ('edsr.tex.j2', 'generic_2d.tex.j2'),
    'swinir': ('edsr.tex.j2', 'generic_2d.tex.j2'),
    'rdn': ('edsr.tex.j2', 'generic_2d.tex.j2'),
    'rcan': ('edsr.tex.j2', 'generic_2d.tex.j2'),
    
    'unet': ('unet_3d.tex.j2', 'unet_2d.tex.j2'), 
    'nafnet': ('unet_3d.tex.j2', 'unet_2d.tex.j2'),
    'restormer': ('unet_3d.tex.j2', 'unet_2d.tex.j2'),
    'unetformer': ('swin_unet_3d.tex.j2', 'swin_unet_2d.tex.j2'),
    
    'ufno': ('ufno_3d.tex.j2', 'ufno_2d.tex.j2'),
    'u-fno': ('ufno_3d.tex.j2', 'ufno_2d.tex.j2'),
    
    'fno': ('fno_3d.tex.j2', 'edsr.tex.j2'),
    'segformer': ('segformer.tex.j2', 'swin_unet_2d.tex.j2'),
    
    'videoswin': ('videoswin_3d.tex.j2', 'videoswin_2d.tex.j2'),
    'convlstm': ('convlstm_3d.tex.j2', 'convlstm_2d.tex.j2'),
    'physics': ('physics_transformer_3d.tex.j2', 'physics_transformer_2d.tex.j2'),
    'sequential': ('sequential_3d.tex.j2', 'sequential_2d.tex.j2'),
    
    'swin': ('swint.tex.j2', 'edsr.tex.j2'),
    'swint': ('swint.tex.j2', 'edsr.tex.j2'),
    
    'hybrid': ('hybrid.tex.j2', 'generic_2d.tex.j2'),
    'liif': ('liif.tex.j2', 'edsr.tex.j2'),
    
    'mlp': ('mlp_mixer.tex.j2', 'edsr.tex.j2'),
    'mixer': ('mlp_mixer.tex.j2', 'edsr.tex.j2'),
    'mlp_mixer': ('mlp_mixer.tex.j2', 'edsr.tex.j2'),
    
    'deeponet': ('deeponet_3d.tex.j2', 'deeponet_2d.tex.j2'),
    
    # Extra models
    'perceiverio': ('mlp_mixer.tex.j2', 'edsr.tex.j2'),
    'cnn_attn_lite': ('edsr.tex.j2', 'edsr.tex.j2'),
    'conv_gate_lite': ('edsr.tex.j2', 'edsr.tex.j2'),
    'unet_plus_plus': ('unet_3d.tex.j2', 'unet_2d.tex.j2'),
    'partialconv_unet': ('unet_3d.tex.j2', 'unet_2d.tex.j2'),
    
    # Missing models
    'uno': ('ufno_3d.tex.j2', 'ufno_2d.tex.j2'),
    'sparse_swin_unet': ('swin_unet_3d.tex.j2', 'swin_unet_2d.tex.j2'),
    'conv_unet_lite': ('unet_3d.tex.j2', 'unet_2d.tex.j2'),
    'resnet_lite': ('edsr.tex.j2', 'edsr.tex.j2'),
    'swin_temporal': ('videoswin_3d.tex.j2', 'videoswin_2d.tex.j2'),
}

def get_generator(model_key, dimension='3d'):
    model_key = model_key.lower()
    
    # Direct match
    if model_key in GENERATOR_REGISTRY:
        idx = 0 if dimension == '3d' else 1
        return GENERATOR_REGISTRY[model_key][idx]
        
    # Fuzzy match
    for key, (g3, g2) in GENERATOR_REGISTRY.items():
        if key in model_key:
            return g3 if dimension == '3d' else g2
            
    # Fallback
    return 'generic_2d.tex.j2'

def process_model(model_name: str, args, report_data):
    print(f"\\nProcessing model: {model_name}...")
    entry = {
        'status': 'failed',
        'generator': 'generic',
        'paths': {},
        'error': ''
    }
    
    try:
        # Build model
        model = build_model(model_name, args.in_ch, args.out_ch, args.img, args.upscale)
        model.eval()
        
        # Determine paths
        base_dir = os.path.dirname(__file__)
        export_dir = os.path.join(base_dir, 'build_export_j2')
        
        # Create per-model subdirectory
        model_out_dir = os.path.join(export_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        # Setup styles for this model directory
        setup_pnn_styles(base_dir, model_out_dir)
        
        # 1. Summary
        # Determine input shape intelligently
        is_temporal = False
        temporal_keywords = ['convlstm', 'swin_temporal', 'videoswin', 'physics', 'sequential', 'temporal', 'video', 'lstm']
        if any(k in model_name.lower() for k in temporal_keywords):
            is_temporal = True

        try:
            # Try creating input based on heuristic
            if is_temporal:
                T_in = 4
                x = torch.randn(1, T_in, args.in_ch, args.img, args.img)
            else:
                x = torch.randn(1, args.in_ch, args.img, args.img)
            
            # Dry run to verify input shape
            try:
                model(x)
            except RuntimeError as re:
                err_msg = str(re).lower()
                if "dimension" in err_msg or "shape" in err_msg or "channel" in err_msg:
                    print(f"Warning: Initial input shape {x.shape} failed ({re}). Switching mode...")
                    if is_temporal: # Was temporal, try spatial
                        x = torch.randn(1, args.in_ch, args.img, args.img)
                    else: # Was spatial, try temporal
                        x = torch.randn(1, 4, args.in_ch, args.img, args.img)
                    # Try again
                    model(x)
                else:
                    raise re
        except Exception as e:
             print(f"Input shape inference failed: {e}. Falling back to default spatial input.")
             x = torch.randn(1, args.in_ch, args.img, args.img)

        sum_txt = os.path.join(model_out_dir, f"{model_name}_summary.txt")
        try:
            with open(sum_txt, "w") as f:
                f.write(str(summary(model, input_data=x, depth=6)))
        except Exception as e:
            print(f"Summary generation failed for {model_name}: {e}")
            with open(sum_txt, "w") as f:
                f.write(f"Summary generation failed: {e}")
        entry['paths']['summary'] = sum_txt
        
        # 2. ONNX
        onnx_path = os.path.join(model_out_dir, f"{model_name}.onnx")
        try:
            torch.onnx.export(model, x, onnx_path, 
                            input_names=['input'], output_names=['output'], 
                            opset_version=16)
            entry['paths']['onnx'] = onnx_path
        except Exception as e:
            print(f"ONNX export failed: {e}")
            
        # Extract dynamic info
        info = {
            'img': args.img,
            'upscale': args.upscale,
            'model_name_safe': model_name.replace('_', '\\_'),
            # Defaults
            'n_resblocks': getattr(model, 'n_resblocks', 16),
            'n_feats': getattr(model, 'n_feats', 64),
            'embed_dim': getattr(model, 'embed_dim', 96),
            'depths': getattr(model, 'depths', [2, 2, 6, 2]),
            'num_heads': getattr(model, 'num_heads', [3, 6, 12, 24]),
            'window_size': getattr(model, 'window_size', 7),
            'in_ch': args.in_ch,
            'out_ch': args.out_ch,
            'block_type': 'Conv' # Default
        }
        
        # Handle list attributes
        if hasattr(model, 'depths') and isinstance(model.depths, (list, tuple)):
             info['depths'] = model.depths
        if hasattr(model, 'num_heads') and isinstance(model.num_heads, (list, tuple)):
             info['num_heads'] = model.num_heads
        if hasattr(model, 'features') and isinstance(model.features, (list, tuple)):
             info['features'] = model.features
        
        # Auto-generate features if missing (for UNet templates)
        if 'features' not in info:
            base = info.get('embed_dim', info.get('n_feats', 64))
            info['features'] = [base * (2**i) for i in range(5)]
        
        # Specific overrides
        model_key = model_name.lower()
        if 'swinir' in model_key: info['block_type'] = 'RSTB'
        elif 'rdn' in model_key: info['block_type'] = 'RDB'
        elif 'rcan' in model_key: info['block_type'] = 'RCAB'
        elif 'unetformer' in model_key: info['block_type'] = 'Trans. Block'
        elif 'vit' in model_key: info['layer_type'] = 'Trans. Block'

        # 3. TikZ 3D (Iterative)
        tex_path = os.path.join(model_out_dir, f"fig_{model_name.lower()}_auto.tex")
        pdf_path = os.path.join(model_out_dir, f"fig_{model_name.lower()}_auto.pdf")
        gen_3d = get_generator(model_key, '3d')
        entry['generator'] = gen_3d
        entry['paths']['tex'] = tex_path

        # Iterative params for 3D
        base_scale = 1.5
        max_retries_3d = 8 if args.compile else 1
        
        for attempt in range(max_retries_3d):
            current_scale = base_scale + (attempt * 0.4)
            info['dist_scale'] = current_scale
            
            if attempt > 0:
                print(f"Retrying 3D generation with dist_scale={current_scale:.2f}...")

            render_template(gen_3d, info, tex_path)
            
            if args.compile:
                cmd_3d = ['conda', 'run', '-n', args.latex_env, 'tectonic', tex_path]
                print(f"Compiling 3D (attempt {attempt+1}): {' '.join(cmd_3d)}")
                try:
                    subprocess.check_call(cmd_3d)
                except subprocess.CalledProcessError as e:
                    print(f"Compilation failed: {e}")
                    entry['error'] = '3D Compilation failed'
                    break
                
                if os.path.exists(pdf_path):
                    entry['paths']['pdf'] = pdf_path
                    
                    # Check Overlaps
                    img_path = check_overlaps.convert_pdf_to_image(pdf_path)
                    if img_path:
                        count, debug_path = check_overlaps.detect_overlaps(img_path, debug_dir=model_out_dir)
                        if os.path.exists(img_path):
                            os.remove(img_path)
                        
                        if count == 0:
                            print(f"3D Overlap check passed with scale={current_scale:.2f}")
                            entry['status'] = 'success'
                            break 
                        else:
                            print(f"3D Overlap check failed: {count} overlaps (scale={current_scale:.2f}).")
                            if attempt == max_retries_3d - 1:
                                if debug_path:
                                    entry['paths']['debug_overlap_3d'] = debug_path
                    else:
                        break
                else:
                    break
            else:
                entry['status'] = 'success (tex generated)'
                break
        
        # 4. TikZ 2D (Iterative)
        tex_path_2d = os.path.join(model_out_dir, f"fig_{model_name.lower()}_2d_auto.tex")
        pdf_path_2d = os.path.join(model_out_dir, f"fig_{model_name.lower()}_2d_auto.pdf")
        
        gen_2d = get_generator(model_key, '2d')
        
        # Iterative params
        base_dist = 1.5
        max_retries = 5 if args.compile else 1 
        
        for attempt in range(max_retries):
            current_dist = base_dist + (attempt * 0.5)
            info['node_distance'] = f"{current_dist}cm"
            
            if attempt > 0:
                print(f"Retrying 2D generation with node_distance={info['node_distance']}...")

            render_template(gen_2d, info, tex_path_2d)
            
            if args.compile:
                # Compile 2D
                cmd_2d = ['conda', 'run', '-n', args.latex_env, 'tectonic', tex_path_2d]
                try:
                    subprocess.check_call(cmd_2d)
                except subprocess.CalledProcessError as e:
                    print(f"Compilation failed: {e}")
                    entry['error'] = 'Compilation failed'
                    break

                if os.path.exists(pdf_path_2d):
                    entry['paths']['pdf_2d'] = pdf_path_2d
                    
                    # Check Overlaps
                    img_path = check_overlaps.convert_pdf_to_image(pdf_path_2d)
                    if img_path:
                        count, debug_path = check_overlaps.detect_overlaps(img_path, debug_dir=model_out_dir)
                        if os.path.exists(img_path):
                            os.remove(img_path)
                        
                        if count == 0:
                            print(f"Overlap check passed with dist={info['node_distance']}")
                            break # Success!
                        else:
                            print(f"Overlap check failed: {count} overlaps (dist={info['node_distance']}).")
                            if attempt == max_retries - 1:
                                if debug_path:
                                    entry['paths']['debug_overlap'] = debug_path
                    else:
                        break
                else:
                    break
            else:
                break 
        
    except Exception as e:
        print(f"FAILED {model_name}: {e}")
        entry['error'] = str(e)
        traceback.print_exc()
        
    report_data[model_name] = entry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, help="Single model name")
    ap.add_argument("--models", type=str, help="Comma-separated list of models")
    ap.add_argument("--in_ch", type=int, default=1)
    ap.add_argument("--out_ch", type=int, default=1)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--compile", action="store_true", help="Compile generated TeX to PDF")
    ap.add_argument("--latex_env", type=str, default="latex")
    args = ap.parse_args()

    # Ensure style files are present
    base_dir = os.path.dirname(__file__)
    export_dir = os.path.join(base_dir, 'build_export_j2')
    os.makedirs(export_dir, exist_ok=True)

    if args.model and args.models:
        print("Error: Provide either --model or --models, not both.")
        return
    
    if not args.model and not args.models:
        print("Error: Must provide --model or --models.")
        return

    model_list = []
    if args.models:
        model_list = [m.strip() for m in args.models.split(',')]
    else:
        model_list = [args.model]

    report_data = {}
    
    for m in model_list:
        process_model(m, args, report_data)
        
    report_path = os.path.join(base_dir, 'build_export_j2', 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\\nBatch processing complete. Report saved to {report_path}")
    print("\\nSummary:")
    for m, data in report_data.items():
        print(f"{m}: {data['status'].upper()} ({data['generator']})")

if __name__ == "__main__":
    main()
