#!/usr/bin/env python3
"""
PDEBench������װ�ű�
"""

import subprocess
import sys

def install_requirements():
    """��װ������"""
    requirements = [
        "torch>=2.1.0",
        "torchvision",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "hydra-core",
        "omegaconf",
        "tensorboard",
        "tqdm",
        "h5py",
        "netcdf4",
        "xarray"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])

if __name__ == "__main__":
    install_requirements()
    print("Environment setup completed!")
