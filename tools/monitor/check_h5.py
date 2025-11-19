#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="runs/AR-DR2D-Debug-SwinUNet-s42/config_merged.yaml")
    args = ap.parse_args()

    print(f"CFG_EXISTS: {os.path.isfile(args.cfg)}")
    if not os.path.isfile(args.cfg):
        return 1
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dp = cfg.get("data", {}).get("data_path", "")
    print(f"DATA_PATH: {dp}")
    print(f"EXISTS: {os.path.exists(dp)}")
    if not os.path.exists(dp):
        return 2
    try:
        with h5py.File(dp, "r") as f:
            keys = list(f.keys())
            print(f"TOP_KEYS: {keys[:10]}")
            # PDEBench style numerical group keys
            num_keys = [k for k in keys if str(k).isdigit()]
            print(f"NUM_KEY_COUNT: {len(num_keys)}")
            if num_keys:
                k0 = num_keys[0]
                print(f"FIRST_NUM_KEY: {k0}")
                if "data" in f[k0]:
                    print(f"DATA_SHAPE: {f[k0]['data'].shape}")
                else:
                    # alternative layouts
                    subkeys = list(f[k0].keys())
                    print(f"SUBKEYS_IN_FIRST: {subkeys}")
    except Exception as e:
        print(f"H5_ERROR: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())