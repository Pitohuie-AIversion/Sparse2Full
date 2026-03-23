#!/usr/bin/env python3
"""
Batch Retrain Failed Models (Single GPU Mode)
"""

import os
import sys
import subprocess
import time

# Force single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Optimize CUDA memory allocation to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

FAILED_MODELS = [
    # "hybrid",       # Skipped by user request (Persistent OOM)
    # "swinir",       # Skipped by user request (Persistent OOM)
    "restormer",    # Parameter Budget Exceeded (Too Large)
    "rcan",         # Parameter Budget Exceeded (Too Large)
    "rdn",          # Parameter Budget Exceeded (Too Small)
    "perceiverio"   # Parameter Budget Exceeded (Too Small)
]

def run_retrain():
    print(f"🚀 Starting batch retraining for {len(FAILED_MODELS)} failed models on SINGLE GPU...")
    
    # Ensure PYTHONPATH includes current directory
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = os.getcwd()
    else:
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env["PYTHONPATH"]

    for model in FAILED_MODELS:
        print(f"\n==================================================")
        print(f"🔄 Retrying model: {model}")
        print(f"==================================================")
        
        # Further reduced batch size to 8 to be ultra-safe against OOM
        batch_size = 8
        
        cmd = [
            sys.executable,
            "experiment_scripts/train_real_data_ar.py",
            f"model.name={model}",
            f"experiment.name=AR-SW-10M-{model}-Retry-SingleGPU",
            "model_budget.strict_mode=False",  # Disable budget check
            f"training.batch_size={batch_size}",
            "training.oom_recovery.enabled=True",
            "device.devices=1",               # Explicitly set 1 device
            "device.accelerator=cuda"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"✅ {model} retraining completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ {model} retraining failed with exit code {e.returncode}.")
        except Exception as e:
            print(f"❌ {model} retraining failed with error: {e}")

if __name__ == "__main__":
    run_retrain()
