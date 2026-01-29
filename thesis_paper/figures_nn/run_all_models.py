import subprocess
import sys

models = [
    'unet', 'swin_unet', 'deeponet', 'physics', 'sequential', 'u-fno', 
    'swin', 'swint', 'mixer', 'nafnet', 'restormer', 'unetformer', 
    'edsr', 'rdn', 'rcan', 'segformer', 'videoswin', 'convlstm', 
    'hybrid', 'liif', 'mlp', 'mlp_mixer', 'ufno', 'fno', 'swinir'
]

# swinir was problematic, moved to end

failed_models = []

for model in models:
    print(f"================ Processing {model} ================")
    cmd = [
        sys.executable, 
        'thesis_paper/figures_nn/export_and_gen_tikz.py', 
        '--model', model, 
        '--compile', 
        '--latex_env', 'latex'
    ]
    
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print(f"ERROR: Failed to process {model}")
        failed_models.append(model)
    except KeyboardInterrupt:
        print("Interrupted by user")
        break

if failed_models:
    print(f"\nFailed models: {failed_models}")
else:
    print("\nAll models processed successfully.")
