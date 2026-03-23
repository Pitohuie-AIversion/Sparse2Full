import sys
import os
import subprocess
import json
import logging
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tools.training.model_loader import list_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_smoke_test(model_name):
    """Run a real training smoke test for a single model."""
    cmd = [
        sys.executable,
        "tools/training/train_real_data_ar.py",
        "--config", "thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml",
        "--model", model_name,
        "training.epochs=1",
        "training.batch_size=4",
        "data.dataloader.batch_size=4",
        "data.sample_limit=8",
        "training.validation.enabled=False",
        "logging.log_model=False",
        "logging.visualization.save_test_visualizations=False",
        "training.checkpoint.save_best=False",
        "training.checkpoint.save_last=False",
        "model_budget.target_params_m=10.0",
        "model_budget.auto_tune=True",
        "model_budget.strict_mode=False",  # Important: Don't fail on parameter count
        f"experiment.name=SmokeTest-{model_name}"
    ]
    
    try:
        # Capture both stdout and stderr
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes max per model
        )
        
        if result.returncode == 0:
            return {"model": model_name, "success": True}
        else:
            # Extract last few lines of error
            error_log = result.stderr[-500:] if result.stderr else result.stdout[-500:]
            return {"model": model_name, "success": False, "error": error_log}
            
    except subprocess.TimeoutExpired:
        return {"model": model_name, "success": False, "error": "Timeout"}
    except Exception as e:
        return {"model": model_name, "success": False, "error": str(e)}

def main():
    logger.info("🚀 Starting Comprehensive Smoke Test for All Spatial Models...")
    
    # Get all valid models (using updated loader with exclusions)
    all_models = list_models()
    
    # Filter for spatial models (by checking file path if possible, or just run all)
    # Since list_models filters components, we can trust it mostly.
    # But let's verify they are spatial models by convention if needed.
    # For now, we assume everything returned by list_models is a candidate.
    
    logger.info(f"📋 Found {len(all_models)} candidate models.")
    
    results = []
    pbar = tqdm(all_models)
    
    for m in pbar:
        pbar.set_description(f"Testing {m}")
        res = run_smoke_test(m)
        results.append(res)
        
        # Log failures immediately
        if not res['success']:
            logger.error(f"❌ {m} Failed: {res.get('error', 'Unknown error')}")
    
    # Save results
    out_path = project_root / 'runs' / 'smoke_test_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"\n📊 Results saved to {out_path}")
    
    # Summary
    passed = [r['model'] for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print("\n" + "="*60)
    print(f"Summary: {len(passed)}/{len(results)} Passed")
    print("="*60)
    
    if failed:
        print("\n❌ Failures:")
        for f in failed:
            print(f"- {f['model']}")
            print(f"  Error: {f['error'].splitlines()[-1] if f['error'] else 'Unknown'}")
    else:
        print("\n✅ All models passed the smoke test!")

if __name__ == "__main__":
    main()
