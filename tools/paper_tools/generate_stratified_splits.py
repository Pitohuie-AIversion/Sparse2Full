#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def load_all_ids(index_file: Path) -> List[str]:
    ids = []
    with index_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def stratify_by_meta(ids: List[str], meta: Dict[str, Dict]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {}
    for sid in ids:
        m = meta.get(sid, {})
        # 依据可用元信息分桶：变量通道、参数区间、时间段
        ch = m.get('channel', 'u')
        param_bin = m.get('param_bin', 'p0')
        time_bin = m.get('time_bin', 't0')
        key = f"{ch}|{param_bin}|{time_bin}"
        buckets.setdefault(key, []).append(sid)
    return buckets


def split_bucket(bucket_ids: List[str], ratios: Tuple[float, float, float], seed: int) -> Tuple[List[str], List[str], List[str]]:
    r_train, r_val, r_test = ratios
    random.Random(seed).shuffle(bucket_ids)
    n = len(bucket_ids)
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    train = bucket_ids[:n_train]
    val = bucket_ids[n_train:n_train + n_val]
    test = bucket_ids[n_train + n_val:]
    return train, val, test


def write_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for x in items:
            f.write(f"{x}\n")


def main():
    parser = argparse.ArgumentParser(description='Stratified split generator (fixed lists)')
    parser.add_argument('--index', type=str, required=True, help='All sample IDs file (one per line)')
    parser.add_argument('--meta', type=str, required=False, help='JSON meta mapping {id: {channel,param_bin,time_bin}}')
    parser.add_argument('--output', type=str, default='data/splits', help='Output dir for train/val/test.txt')
    parser.add_argument('--ratios', type=str, default='0.8,0.1,0.1', help='Train,Val,Test ratios')
    parser.add_argument('--seed', type=int, default=2025, help='Fixed seed for reproducibility')
    args = parser.parse_args()

    index_file = Path(args.index)
    output_dir = Path(args.output)
    ratios = tuple(float(x) for x in args.ratios.split(','))
    assert abs(sum(ratios) - 1.0) < 1e-6, 'Ratios must sum to 1'

    ids = load_all_ids(index_file)
    meta = {}
    if args.meta:
        meta = json.loads(Path(args.meta).read_text(encoding='utf-8'))

    if meta:
        buckets = stratify_by_meta(ids, meta)
    else:
        buckets = {'default': ids}

    train_all: List[str] = []
    val_all: List[str] = []
    test_all: List[str] = []

    # 每个桶独立按固定种子打乱并切分，最终拼接
    for i, (_, b_ids) in enumerate(buckets.items()):
        t, v, te = split_bucket(b_ids, ratios, seed=args.seed + i)
        train_all.extend(t)
        val_all.extend(v)
        test_all.extend(te)

    # 写出固定清单
    write_list(output_dir / 'train.txt', train_all)
    write_list(output_dir / 'val.txt', val_all)
    write_list(output_dir / 'test.txt', test_all)

    print(f"Written splits to {output_dir}/train.txt, val.txt, test.txt")


if __name__ == '__main__':
    main()

