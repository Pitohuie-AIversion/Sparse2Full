import argparse
from pathlib import Path
import h5py
import numpy as np
from omegaconf import OmegaConf

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--out-dir', type=str, default='splits')
    p.add_argument('--train-count', type=int, default=None)
    p.add_argument('--val-count', type=int, default=None)
    p.add_argument('--test-count', type=int, default=None)
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    data_path = str(getattr(cfg.data, 'data_path'))
    seed = int(getattr(cfg, 'seed', getattr(cfg.experiment, 'seed', 2025)))
    train_ratio = float(getattr(cfg.data, 'train_ratio', 0.8))
    val_ratio = float(getattr(cfg.data, 'val_ratio', 0.15))
    test_ratio = float(getattr(cfg.data, 'test_ratio', 0.05))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, 'r') as f:
        keys = [k for k in f.keys() if k.isdigit()]
    keys = sorted(keys)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(keys))
    total = len(idx)
    if args.train_count or args.val_count or args.test_count:
        trc = int(args.train_count or max(1, int(total * train_ratio)))
        vac = int(args.val_count or max(1, int(total * val_ratio)))
        tec = int(args.test_count or max(1, int(total * test_ratio)))
        end_tr = min(total, trc)
        end_va = min(total, end_tr + vac)
        end_te = min(total, end_va + tec)
        sel = [keys[i] for i in idx[:end_te]]
        train_keys = sel[:end_tr]
        val_keys = sel[end_tr:end_va]
        test_keys = sel[end_va:end_te]
    else:
        tr = int(total * train_ratio)
        va = int(total * val_ratio)
        train_keys = [keys[i] for i in idx[:tr]]
        val_keys = [keys[i] for i in idx[tr:tr+va]]
        test_keys = [keys[i] for i in idx[tr+va:]]
    (out_dir / 'train.txt').write_text('\n'.join(train_keys) + '\n')
    (out_dir / 'val.txt').write_text('\n'.join(val_keys) + '\n')
    (out_dir / 'test.txt').write_text('\n'.join(test_keys) + '\n')
    print(f'Wrote splits to {out_dir} | train={len(train_keys)} val={len(val_keys)} test={len(test_keys)} total={total}')

if __name__ == '__main__':
    main()
