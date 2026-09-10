#!/usr/bin/env python3
"""
Crop Report Generator
Collects results from crop scan experiments including Params and FLOPs.
"""

import json
from pathlib import Path
import pandas as pd
import sys

# Configuration
SIZES = [48, 32, 16, 8, 4, 1]
RUNS_DIR = Path("runs_drd_paper")

def find_latest_run(size):
    """Find the most recent run directory for a given crop size."""
    pattern = f"AR-DR2D-Crop-Scan-Size{size}-*"
    # Search in runs_drd_paper
    candidates = list(RUNS_DIR.glob(pattern))
    
    if not candidates:
        return None
        
    # Sort by modification time (newest first)
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

def collect_results():
    results = []
    for size in SIZES:
        latest_run = find_latest_run(size)
        if not latest_run:
            print(f"⚠️ No run found for size {size}")
            continue
            
        result_file = latest_run / "test_results.json"
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    metrics = data.get('final_test_metrics', {})
                    
                    # Try to get H_err (might be named 'test_h_err' or similar)
                    h_err = metrics.get('test_h_err_mean', metrics.get('h_err', 0.0))
                    
                    entry = {
                        'Size': size,
                        'Area_Pct': (size*size)/(128*128)*100,
                        'Rel_L2': metrics.get('test_rel_l2_mean', metrics.get('rel_l2', 0.0)),
                        'PSNR': metrics.get('test_psnr_mean', metrics.get('psnr', 0.0)),
                        'SSIM': metrics.get('test_ssim_mean', metrics.get('ssim', 0.0)),
                        'H_Err': h_err,
                        'Path': latest_run.name
                    }
                    
                    # Try to get Params and FLOPs from model_resources.json
                    res_file = latest_run / "model_resources.json"
                    if res_file.exists():
                        try:
                            with open(res_file, 'r') as rf:
                                res_data = json.load(rf)
                                entry['Params(M)'] = res_data.get('params', 0) / 1e6
                                entry['FLOPs(G)'] = res_data.get('flops_g', 0.0)
                        except Exception:
                            entry['Params(M)'] = 0.0
                            entry['FLOPs(G)'] = 0.0
                    else:
                        entry['Params(M)'] = 0.0
                        entry['FLOPs(G)'] = 0.0
                        
                    results.append(entry)
            except Exception as e:
                print(f"⚠️ Failed to parse results for {latest_run.name}: {e}")
        else:
            print(f"⚠️ No results file found for {latest_run.name}")
    
    return results

def main():
    print("\n📊 Generating Summary Report...")
    results = collect_results()
    
    if not results:
        print("No results found.")
        return
        
    df = pd.DataFrame(results)
    # Reorder columns
    cols = ['Size', 'Area_Pct', 'Params(M)', 'FLOPs(G)', 'Rel_L2', 'PSNR', 'SSIM', 'H_Err', 'Path']
    # Filter cols that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    df = df.sort_values('Size', ascending=False)
    
    print("\n=== Crop Capability Scan Results ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))
    
    # Save to file
    report_path = RUNS_DIR / "crop_scan_summary_with_resources.md"
    with open(report_path, "w") as f:
        f.write("# Crop Capability Scan Results (With Resources)\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    
    print(f"\n✅ Report saved to {report_path}")

if __name__ == "__main__":
    main()
