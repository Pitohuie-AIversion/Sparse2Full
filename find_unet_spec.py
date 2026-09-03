import json
import glob

print("Looking for all UNet configurations in runs_drd_paper...")

for path in glob.glob("runs_drd_paper/*UNet*/config_merged.yaml"):
    with open(path, 'r') as f:
        content = f.read()
        rec = "1.0" if "reconstruction: 1.0" in content else "?"
        spec = "0.5" if "spectral: 0.5" in content else ("0.0" if "spectral: 0.0" in content else "?")
        dc = "1.0" if "data_consistency: 1.0" in content else ("0.0" if "data_consistency: 0.0" in content else "?")
        
        print(f"Path: {path}")
        print(f"  -> Rec: {rec}, Spec: {spec}, DC: {dc}")
