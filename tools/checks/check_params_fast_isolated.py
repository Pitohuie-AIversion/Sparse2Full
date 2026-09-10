import sys
import os
import subprocess
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tools.training.model_loader import list_models, get_model_info

def check_model(model_name):
    cmd = [
        sys.executable,
        "tools/check_single_model_params.py",
        "--model", model_name,
        "--config", "thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30 # 30 seconds per model should be enough without data
        )
        
        # Parse result
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                parts = line.split(":", 2)
                status = parts[1]
                payload = parts[2]
                if status == "SUCCESS":
                    return {"model": model_name, "success": True, "params_m": float(payload)}
                else:
                    return {"model": model_name, "success": False, "error": payload}
                    
        # If no result line found
        return {
            "model": model_name, 
            "success": False, 
            "error": f"No result code. Stderr: {result.stderr[-200:] if result.stderr else 'Empty'}"
        }
        
    except subprocess.TimeoutExpired:
        return {"model": model_name, "success": False, "error": "Timeout"}
    except Exception as e:
        return {"model": model_name, "success": False, "error": str(e)}

def main():
    print("🚀 Starting Fast & Safe 10M Parameter Check...")
    
    # Get spatial models
    all_models = list_models()
    spatial_models = []
    for m in all_models:
        info = get_model_info(m)
        fp = info.get('file_path') if info else None
        if fp and ('/models/spatial/' in fp.replace('\\','/')):
            spatial_models.append(m)
    
    # Filter duplicates or aliases if needed, but list_models usually returns unique keys
    spatial_models = sorted(list(set(spatial_models)))
    
    print(f"📋 Checking {len(spatial_models)} models...")
    
    results = []
    pbar = tqdm(spatial_models)
    
    for m in pbar:
        pbar.set_description(f"Checking {m}")
        res = check_model(m)
        results.append(res)
        
    # Save results
    out_path = project_root / 'runs' / 'fast_10m_check.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n📊 Results saved to {out_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print(f"{'Model':<30} | {'Status':<10} | {'Params (M)':<10}")
    print("-" * 60)
    
    success_count = 0
    for res in results:
        status = "✅ PASS" if res['success'] else "❌ FAIL"
        params = f"{res['params_m']:.2f}" if res.get('params_m') else "-"
        print(f"{res['model']:<30} | {status:<10} | {params:<10}")
        if not res['success']:
            print(f"  └── Error: {res.get('error', 'Unknown')[:100]}...")
        else:
            success_count += 1
            
    print("-" * 60)
    print(f"Summary: {success_count}/{len(results)} passed.")

if __name__ == "__main__":
    main()
