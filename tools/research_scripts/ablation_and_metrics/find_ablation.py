import json
import glob

print("Looking for all EDSR and UNet Ablation configurations...")

for path in glob.glob("runs_drd_paper/*/config_merged.yaml") + glob.glob("runs_3loss_ablation_unet_100ep/*/config_merged.yaml"):
    if "EDSR" in path or "UNet" in path or "unet" in path:
        with open(path, 'r') as f:
            content = f.read()
            # Extract loss weights
            rec = "1.0" if "reconstruction: 1.0" in content else "?"
            spec = "0.5" if "spectral: 0.5" in content else ("0.0" if "spectral: 0.0" in content else "?")
            dc = "1.0" if "data_consistency: 1.0" in content else ("0.0" if "data_consistency: 0.0" in content else ("0.1" if "data_consistency: 0.1" in content else "?"))
            
            # Print if it's an ablation run or a regular run
            if "Ablation" in path or "NoSpec" in path or "Consistent" in path or "RecOnly" in path or "RecSpec" in path or "MSE" in path or "A0" in path or "A2" in path or "A3" in path:
                print(f"Path: {path}")
                print(f"  -> Rec: {rec}, Spec: {spec}, DC: {dc}")
