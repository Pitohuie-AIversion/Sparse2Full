import json
import glob

print("Looking for MSE Only models...")

for path in glob.glob("runs_drd_paper/*/config_merged.yaml"):
    with open(path, 'r') as f:
        content = f.read()
        if "reconstruction: 1.0" in content and "spectral: 0.0" in content and "data_consistency: 0.0" in content:
            print(f"MSE Only config found: {path}")
            test_res = path.replace('config_merged.yaml', 'test_results.json')
            try:
                with open(test_res, 'r') as f2:
                    data = json.load(f2)
                    metrics = data.get('final_test_metrics', {})
                    rel_l2 = metrics.get('rel_l2', 'N/A')
                    print(f"  -> Rel-L2: {rel_l2}")
            except Exception as e:
                print(f"  -> No test results ({e})")
