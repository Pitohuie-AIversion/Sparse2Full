import json
import os

files = [
    "AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116",
    "AR-DR2D-Stage3-VideoSwin-SRx4-JointFineTune-model_unknown-s2025-20260226",
    "AR-DR2D-SwinUNet-E2E-NoPretrain-SRx4-model_SwinUNet-s2025-20260115"
]

for d in files:
    test_path = f"/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd_paper/{d}/test_results.json"
    if os.path.exists(test_path):
        try:
            with open(test_path) as fp:
                data = json.load(fp)
                print(f"--- {d} ---")
                print("Keys in test_results.json:", list(data.keys()))
                if 'step_metrics' in data:
                    print("Found step_metrics")
        except Exception as e:
            print(f"Error: {e}")
