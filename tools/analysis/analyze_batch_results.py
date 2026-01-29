import os
import json
import pandas as pd
from pathlib import Path
import glob

def analyze_results(runs_dir="runs", output_file="analysis_report_ar_sw_10m.csv"):
    """
    Scans the runs directory for test_results.json and aggregates metrics into a CSV.
    Targeting AR-SW-10M-* experiments.
    """
    runs_path = Path(runs_dir)
    results_list = []
    
    # Find all test_results.json files
    # Search recursively in runs/
    # Pattern: runs/AR-SW-10M-*/test_results.json
    # Also support AR-ShallowWater-10M-* for backward compatibility if needed
    
    search_patterns = [
        "AR-SW-10M-*/test_results.json",
        "AR-ShallowWater-10M-*/test_results.json",
        "AR-DR2D-FNO2d-SRx4-10M-*/test_results.json" # For some older runs
    ]
    
    files = []
    for pattern in search_patterns:
        files.extend(list(runs_path.glob(pattern)))
        
    print(f"Found {len(files)} result files.")
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract Model Name from Directory
            dir_name = file_path.parent.name
            
            # Try to parse clean model name
            # Format: AR-SW-10M-{model_name} or AR-SW-10M-{model_name}-*
            parts = dir_name.split('-')
            
            # Simple heuristic: find the part after 10M
            model_name = "Unknown"
            if "10M" in parts:
                idx = parts.index("10M")
                if idx + 1 < len(parts):
                    model_name = parts[idx+1]
            
            # Clean up model name (remove random suffix if present)
            # If directory is like AR-SW-10M-SwinUNet-model_SwinUNet-s2025...
            # The simple split above gets 'SwinUNet'. 
            # If directory is AR-SW-10M-SwinUNet, it gets 'SwinUNet'.
            
            # Extract Metrics
            # Adjust keys based on your actual test_results.json structure
            metrics = data.get("final_test_metrics", {})
            
            # Try to get FLOPs and Params more robustly if model_info exists
            model_info = data.get("model_info", {})
            if not model_info:
                # Fallback: check config file in directory if needed, but for now just 0
                pass
            
            # Calculate inference time per sample (ms) if total test time is available
            # Assuming standard test set size or just reporting total test time
            # For simplicity, we report total test time or placeholder
            test_time = data.get("test_time", 0)
            
            row = {
                "Model": model_name,
                "Directory": dir_name,
                "Params (M)": model_info.get("trainable_params", 0) / 1e6,
                "FLOPs (G)": model_info.get("flops", 0) / 1e9, # Assuming FLOPs in bytes/raw count
                "Inference Time (s)": test_time, # Total test time
                "Test Loss": metrics.get("test_loss", 0),
                "Rel L2 (MSE)": metrics.get("rel_l2", 0),
                "MAE": metrics.get("mae", 0),
                "RMSE": metrics.get("rmse", 0),
                "SSIM": metrics.get("ssim", 0),
                "PSNR": metrics.get("psnr", 0),
                "fRMSE (Low)": metrics.get("frmse_low", 0),
                "fRMSE (Mid)": metrics.get("frmse_mid", 0),
                "fRMSE (High)": metrics.get("frmse_high", 0),
                "bRMSE (Boundary)": metrics.get("brmse", 0),
                "Conservation Err": metrics.get("dc_error", 0)
            }
            
            results_list.append(row)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if not results_list:
        print("No results found.")
        return
        
    df = pd.DataFrame(results_list)
    
    # Sort by Test Loss (ascending) - Lower is better
    df = df.sort_values(by="Test Loss", ascending=True)
    
    # Save to CSV
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)
    print(f"Report saved to {output_path.absolute()}")
    
    # Print Summary
    print("\n" + "="*50)
    print("🏆 Top 5 Models by Test Loss (Lower is Better)")
    print("="*50)
    print(df[["Model", "Test Loss", "MAE", "PSNR", "SSIM"]].head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("🚀 Top 5 Models by PSNR (Higher is Better)")
    print("="*50)
    print(df.sort_values(by="PSNR", ascending=False)[["Model", "PSNR", "Test Loss"]].head(5).to_string(index=False))

if __name__ == "__main__":
    analyze_results()
