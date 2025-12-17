#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
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


def iter_models(catalog: dict, categories: list[str] | None, models_filter: list[str] | None):
    for item in catalog.get('catalog', []):
        cat = str(item.get('category', '')).strip()
        if categories and cat not in categories:
            continue
        for m in item.get('models', []):
            name = str(m.get('name', '')).strip()
            if not name:
                continue
            if models_filter and name not in models_filter:
                continue
            yield name


def run_sweep_batch(base_config: Path, model: str, batches: str, seeds: str, dry_run: bool, logs_root: Path) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / 'sweep_batch.py'),
        '--base-config', str(base_config),
        '--model', model,
        '--batches', batches,
        '--seeds', seeds,
    ]
    logs_root.mkdir(parents=True, exist_ok=True)
    summary_log = logs_root / f"{model}_summary.log"
    if dry_run:
        with summary_log.open('a', encoding='utf-8') as lf:
            lf.write('DRY_RUN ' + ' '.join(cmd) + '\n')
        return 0, 'DRY_RUN'
    start = time.time()
    with summary_log.open('a', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf)
    dur = time.time() - start
    return proc.returncode, f"{dur:.2f}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', default=str(Path(__file__).resolve().parents[1] / 'models_catalog.yaml'))
    p.add_argument('--base-config', default=str(Path(__file__).resolve().parents[1] / 'configs' / 'ar_paper_aligned.yaml'))
    p.add_argument('--batches', default='64,96,128,160')
    p.add_argument('--seeds', default='2025,2026,2027')
    p.add_argument('--categories', default=None)
    p.add_argument('--models', default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    catalog_path = Path(args.catalog)
    base_config = Path(args.base_config)
    if not catalog_path.exists():
        print('models_catalog.yaml not found')
        sys.exit(1)
    if not base_config.exists():
        print('base config not found')
        sys.exit(1)

    catalog = load_yaml(catalog_path)
    categories = [x.strip() for x in args.categories.split(',')] if args.categories else None
    models_filter = [x.strip() for x in args.models.split(',')] if args.models else None

    logs_root = Path('runs') / 'batch_sweep_catalog'
    results = {}
    for name in iter_models(catalog, categories, models_filter):
        if args.verbose:
            print(f"[START] {name} batches={args.batches} seeds={args.seeds}", flush=True)
        ret, dur = run_sweep_batch(base_config, name, args.batches, args.seeds, args.dry_run, logs_root)
        results[name] = {'returncode': ret, 'duration': dur}
        if args.verbose:
            status = 'OK' if ret == 0 else f'ERR({ret})'
            print(f"[DONE]  {name} status={status} time={dur} log=runs/batch_sweep_catalog/{name}_summary.log", flush=True)

    summary_path = logs_root / 'summary.yaml'
    save_yaml({'batches': args.batches, 'seeds': args.seeds, 'dry_run': bool(args.dry_run), 'results': results}, summary_path)
    print(str(summary_path))


if __name__ == '__main__':
    main()
