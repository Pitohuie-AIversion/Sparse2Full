#!/usr/bin/env python3
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except Exception:
    print("pyyaml not available")
    sys.exit(1)


def load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def override_batch(cfg: dict, bs: int) -> dict:
    cfg = dict(cfg)
    t = cfg.get('training', {})
    t['batch_size'] = bs
    dl_t = t.get('dataloader', {})
    dl_t['batch_size'] = bs
    dl_t['val_batch_size'] = max(1, bs // 2)
    dl_t['test_batch_size'] = max(1, bs // 2)
    t['dataloader'] = dl_t
    cfg['training'] = t
    d = cfg.get('data', {})
    dl_d = d.get('dataloader', {})
    dl_d['batch_size'] = bs
    dl_d['val_batch_size'] = max(1, bs // 2)
    dl_d['test_batch_size'] = max(1, bs // 2)
    d['dataloader'] = dl_d
    cfg['data'] = d
    return cfg


def run_once(config_path: Path, model: str, seeds: str | None, log_dir: Path, bs: int) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{model}_bs{bs}.log"
    cmd = [
        sys.executable,
        'tools/training/train_real_data_ar.py',
        '--config', str(config_path),
        '--model', model,
    ]
    if seeds:
        cmd += ['--seeds', seeds]
    with log_file.open('w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf)
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-config', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--batches', required=True, help='comma list, e.g. 64,96,128,160')
    p.add_argument('--seeds', default='2025,2026,2027')
    args = p.parse_args()

    base = Path(args.base_config)
    cfg = load_yaml(base)
    batches = [int(x.strip()) for x in args.batches.split(',') if x.strip()]
    batches = sorted(batches, reverse=True)

    logs_root = Path('runs') / 'batch_sweep' / args.model
    best = None
    cfg_dir = logs_root / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for bs in batches:
        curr_bs = bs
        while curr_bs >= 1:
            tmpcfg = cfg_dir / f"cfg_bs{curr_bs}.yaml"
            save_yaml(override_batch(cfg, curr_bs), tmpcfg)
            code = run_once(tmpcfg, args.model, args.seeds, logs_root, curr_bs)
            if code == 0:
                best = curr_bs
                print(f'best_batch_size={best}')
                print(f'summary_log={logs_root}/{args.model}_summary.log')
                sys.exit(0)
            else:
                print(f"failed_batch_size={curr_bs}")
                print(f"see_log={logs_root}/{args.model}_bs{curr_bs}.log")
                next_bs = max(1, curr_bs // 2)
                if next_bs == curr_bs:
                    break
                print(f"retry_batch_size={next_bs}")
                curr_bs = next_bs

    if best is None:
        print('no batch size succeeded')
        sys.exit(1)
    print(f'best_batch_size={best}')
    sys.exit(0)


if __name__ == '__main__':
    main()
