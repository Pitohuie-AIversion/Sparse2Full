import os
import json
import glob

dirs = [
    "AR-DR2D-edsr-SRx4-1M-300ep", # Using 300ep since 100ep didn't finish
    "AR-DR2D-ConvUNetLite-SRx4-1M-100ep",
    "AR-DR2D-UNet-SRx4-1M-100ep",
    "AR-DR2D-stablefno2d-SRx4-1M-100ep",
    "AR-DR2D-nafnet-SRx4-1M-100ep"
]

base_path = "./drd_paper_1m"

for d in dirs:
    path = os.path.join(base_path, d)
    res_file = os.path.join(path, "test_results.json")
    res_file2 = os.path.join(path, "eval", "metrics.jsonl")
    res_file3 = os.path.join(path, "model_resources.json")
    
    rel_l2 = None
    params = None
    latency = None
    
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            data = json.load(f)
            rel_l2 = data.get("test_loss", None) or data.get("rel_l2", None) or data.get("test_rel_l2", None)
            if rel_l2 is None:
                # print keys
                pass
    
    if rel_l2 is None and os.path.exists(res_file2):
        with open(res_file2, 'r') as f:
            lines = f.readlines()
            if lines:
                data = json.loads(lines[-1]) # last line or summary
                # wait, let's just print the whole dict or check keys
                rel_l2 = data.get("rel_l2", None)
    
    if os.path.exists(res_file3):
        with open(res_file3, 'r') as f:
            data = json.load(f)
            params = data.get("params", 0) / 1e6
            latency = data.get("inference_latency_ms_mean", 0)
            
    print(f"Model: {d}")
    print(f"  Params: {params:.3f} M") if params else print("  Params: N/A")
    print(f"  Latency: {latency:.3f} ms") if latency else print("  Latency: N/A")
    print(f"  Rel-L2: {rel_l2}")
