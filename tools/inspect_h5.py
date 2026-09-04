
import h5py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = PROJECT_ROOT / "data/2D/shallow-water/2D_rdb_NA_NA.h5"
try:
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("Keys:", list(f.keys()))
        for key in list(f.keys())[:3]:
            print(f"Key: {key}, Type: {type(f[key])}")
            if isinstance(f[key], h5py.Dataset):
                print(f"  Shape: {f[key].shape}")
            elif isinstance(f[key], h5py.Group):
                print(f"  Group Keys: {list(f[key].keys())[:3]}")
                for subkey in list(f[key].keys())[:1]:
                    if isinstance(f[key][subkey], h5py.Dataset):
                         print(f"  Subkey: {subkey}, Shape: {f[key][subkey].shape}")
except Exception as e:
    print(f"Error: {e}")
