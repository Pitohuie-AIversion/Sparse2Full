
import json
import os
import glob
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
base_dir = PROJECT_ROOT / "drd_paper_1m"

models_data = []

# List of expected model directories (using the ones we know exist)
# We prioritize the 100ep ones for non-EDSR models as per recent instructions
dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and "tmp_configs" not in d.name])

print(f"Found {len(dirs)} directories.")

for d in dirs:
    model_name = d.name
    
    # Files to read
    test_res_path = d / "test_results.json"
    model_res_path = d / "model_resources.json"
    summary_res_path = d / "resource_summary.json"
    
    if not test_res_path.exists():
        print(f"Skipping {model_name}: No test results")
        continue
        
    try:
        # Read Test Metrics
        with open(test_res_path, 'r') as f:
            test_data = json.load(f)
            metrics = test_data.get("final_test_metrics", {})
            
        # Read Model Resources (Params, Flops)
        model_res = {}
        if model_res_path.exists():
            with open(model_res_path, 'r') as f:
                model_res = json.load(f)
        
        # Read Resource Summary (Memory)
        summary_res = {}
        if summary_res_path.exists():
            with open(summary_res_path, 'r') as f:
                summary_res = json.load(f)
                
        # Extract Key Data
        # Parsing model name from folder name roughly
        # AR-DR2D-edsr-SRx4-1M-300ep -> edsr
        parts = model_name.split('-')
        if len(parts) >= 4:
            short_name = parts[2]
        else:
            short_name = model_name
            
        entry = {
            "Model": short_name,
            "Folder": model_name,
            # Accuracy
            "Rel-L2": metrics.get("rel_l2", 0.0),
            "PSNR": metrics.get("psnr", 0.0),
            "SSIM": metrics.get("ssim", 0.0),
            "MAE": metrics.get("mae", 0.0),
            # Resources
            "Params (M)": model_res.get("params", 0) / 1e6,
            "FLOPs (G)": model_res.get("flops_g", 0.0),
            "Latency (ms)": model_res.get("inference_latency_ms_mean", 0.0),
            "VRAM (GB)": summary_res.get("max_gpu_peak_allocated_gb", 0.0),
            "Training Time (s)": test_data.get("test_time", 0) # This is test time, not training time. 
                                # Training time is usually in resource_summary but let's check keys
        }
        
        models_data.append(entry)
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

# Create DataFrame
df = pd.DataFrame(models_data)

# Sort by Rel-L2 (Accuracy)
if not df.empty:
    df = df.sort_values("Rel-L2")

    # Format for Markdown
    print("\n## 实验结果汇总 (Evaluation Results)\n")
    
    # Define columns to show
    cols = ["Model", "Rel-L2", "PSNR", "SSIM", "Params (M)", "FLOPs (G)", "Latency (ms)", "VRAM (GB)"]
    
    # Print Markdown Table
    print(df[cols].to_markdown(index=False, floatfmt=".4f"))
    
    print("\n\n## 详细数据 (Detailed Data)\n")
    print(df.to_markdown(index=False, floatfmt=".6f"))
else:
    print("No valid data found.")
