import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs_drd_paper"

files = [
    "AR-DR2D-E2E-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260116",
    "AR-DR2D-E2E-StrictStride10-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260122",
    "AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116",
    "AR-DR2D-Stage3-VideoSwin-SRx4-JointFineTune-model_unknown-s2025-20260226",
    "AR-DR2D-UNet-SRx4-10M-300ep",
    "AR-DR2D-SwinUNet-E2E-NoPretrain-SRx4-model_SwinUNet-s2025-20260115"
]

for d in files:
    dir_path = RUNS_DIR / d
    print(f"--- {d} ---")
    
    # Read test results
    test_path = os.path.join(dir_path, "test_results.json")
    if os.path.exists(test_path):
        try:
            with open(test_path) as fp:
                data = json.load(fp)
                m = data.get("final_test_metrics", {})
                print(f"  Rel-L2: {m.get('rel_l2', 0):.4f}")
                print(f"  PSNR: {m.get('psnr', 0):.2f}")
                print(f"  SSIM: {m.get('ssim', 0):.4f}")
                print(f"  H_err: {m.get('dc_error', 0):.4f}")
        except Exception as e:
            print(f"  Error reading metrics: {e}")
            
    # Read resource summary
    res_path = os.path.join(dir_path, "resource_summary.json")
    if os.path.exists(res_path):
        try:
            with open(res_path) as fp:
                res = json.load(fp)
                print(f"  Params: {res.get('Params (M)', 0):.2f} M")
                print(f"  FLOPs: {res.get('FLOPs (G)', 0):.2f} G")
                print(f"  Latency: {res.get('Latency (ms)', 0):.2f} ms")
        except Exception as e:
            print(f"  Error reading resources: {e}")
    print()
