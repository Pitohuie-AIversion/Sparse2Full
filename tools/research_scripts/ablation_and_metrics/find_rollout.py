import json
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
files = glob.glob(str(PROJECT_ROOT / "runs_drd_paper" / "**" / "*.json"), recursive=True)
for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
            # Find arrays of length 10 or 20 (typical rollout lengths)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list) and len(v) in [10, 20]:
                        print(f"Found array {k} in {f}")
                    elif isinstance(v, dict):
                        for subk, subv in v.items():
                            if isinstance(subv, list) and len(subv) in [10, 20]:
                                print(f"Found array {k}.{subk} in {f}")
    except:
        pass
