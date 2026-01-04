import sys
import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tools.training.model_loader import list_models, get_model_info

def check_model_safe(model_name):
    """Run model check in a separate process"""
    cmd = [
        sys.executable,
        "tools/training/train_real_data_ar.py",
        "--config", "thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml",
        "--model", model_name,
        "--target-params-m", "10.0",
        "--tolerance-m", "2.0",
        "training.epochs=0",  # Don't run training loop, just setup
        "training.batch_size=1",
        "data.dataloader.batch_size=1",
        "data.sample_limit=4",
        "training.validation.enabled=False",
        "logging.log_model=False",
        "logging.visualization.save_test_visualizations=False",
        "training.checkpoint.save_best=False",
        "training.checkpoint.save_last=False",
        # Force a unique experiment name to avoid conflicts
        f"experiment.name=Check-10M-{model_name}"
    ]
    
    try:
        # Run process and capture output
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout per model
        )
        
        # Parse output for parameter count
        params = None
        success = False
        error_msg = ""
        
        if result.returncode == 0:
            # Look for success message in stdout
            for line in result.stderr.split('\n') + result.stdout.split('\n'):
                if "模型参数量:" in line and "可训练" in line:
                    # Format: ... 模型参数量: 10,936,843 (可训练: ...
                    try:
                        parts = line.split("模型参数量:")[1].split("(")[0].strip().replace(",", "")
                        params = int(parts) / 1e6
                        success = True
                    except:
                        pass
                if "符合预算目标" in line:
                    success = True
        else:
            error_msg = f"Process exited with code {result.returncode}"
            # Try to extract python exception
            for line in result.stderr.split('\n'):
                if "Error" in line or "Exception" in line:
                    error_msg += f" | {line.strip()}"
            
        return {
            "model": model_name,
            "success": success,
            "params_m": params,
            "error": error_msg if not success else None
        }
        
    except subprocess.TimeoutExpired:
        return {
            "model": model_name,
            "success": False,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "error": str(e)
        }

def main():
    print("🔍 Starting Safe 10M Parameter Check (Process Isolated)...")
    
    # Get spatial models
    all_models = list_models()
    spatial_models = []
    for m in all_models:
        info = get_model_info(m)
        fp = info.get('file_path') if info else None
        if fp and ('/models/spatial/' in fp.replace('\\','/')):
            spatial_models.append(m)
            
    print(f"📋 Found {len(spatial_models)} spatial models to check.")
    
    results = []
    
    pbar = tqdm(spatial_models)
    for m in pbar:
        pbar.set_description(f"Checking {m}")
        res = check_model_safe(m)
        results.append(res)
        
        # Immediate feedback
        status = "✅" if res['success'] else "❌"
        param_str = f"{res['params_m']:.2f}M" if res['params_m'] else "N/A"
        # print(f"{status} {m:<20} {param_str} {res['error'] or ''}")
        
    # Save results
    out_dir = project_root / 'runs'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'model_10m_verification.json'
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n📊 Verification Complete. Results saved to {out_path}")
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    print(f"✅ Success: {success_count}/{len(results)}")
    print(f"❌ Failed: {len(results) - success_count}/{len(results)}")

if __name__ == "__main__":
    main()
