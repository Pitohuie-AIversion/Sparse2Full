
import json
import pandas as pd
from pathlib import Path
import glob

RUNS_DIR = Path("runs_drd_paper")
SIZES = [48, 32, 16, 8, 4, 1]

def collect_results():
    results = []
    print(f"Scanning {RUNS_DIR} for results...")
    
    for size in SIZES:
        # Find the latest run for this size
        pattern = f"AR-DR2D-Crop-Scan-Size{size}-model_EDSR-s2025-*"
        candidates = list(RUNS_DIR.glob(pattern))
        
        if not candidates:
            print(f"⚠️ No run directory found for Size {size}")
            continue
            
        # Sort by modification time (newest first)
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_run = candidates[0]
        
        result_file = latest_run / "test_results.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    metrics = data.get('final_test_metrics', {})
                    
                    entry = {
                        'Size': size,
                        'Area_Pct': (size*size)/(128*128)*100,
                        'Rel_L2': metrics.get('rel_l2', 0.0),
                        'PSNR': metrics.get('psnr', 0.0),
                        'SSIM': metrics.get('ssim', 0.0),
                        'Test_Loss': metrics.get('test_loss', 0.0),
                        'Path': latest_run.name
                    }
                    results.append(entry)
            except Exception as e:
                print(f"⚠️ Failed to parse results for {latest_run.name}: {e}")
        else:
            print(f"⚠️ No test_results.json found in {latest_run.name}")

    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values('Size', ascending=False)
    
    print("\n=== Crop Capability Scan Results ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))
    
    # Save to file
    out_path = RUNS_DIR / "crop_scan_final_summary.md"
    with open(out_path, "w") as f:
        f.write("# Crop Capability Scan Results\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"\n✅ Summary saved to {out_path}")

if __name__ == "__main__":
    collect_results()
