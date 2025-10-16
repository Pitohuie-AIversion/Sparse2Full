#!/usr/bin/env python3
"""
PDEBenchϡ��۲��ؽ�ʵ�鸴�ֽű�

ʹ�÷���:
    python reproduce_all.py --config config_merged.yaml --seed 42

Ҫ��:
    - Python 3.10+
    - PyTorch >= 2.1
    - �����������Ѱ�װ
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """���������ӡ���"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Reproduce PDEBench experiments')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    
    args = parser.parse_args()
    
    # ���û�������
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # ��Ŀ��Ŀ¼
    project_root = Path(__file__).parent.parent
    
    print("=== PDEBenchʵ�鸴�� ===")
    print(f"�����ļ�: {args.config}")
    print(f"�������: {args.seed}")
    print(f"GPU�豸: {args.gpu}")
    print()
    
    # 1. ����һ���Լ��
    print("1. ��������һ���Լ��...")
    cmd = f"python tools/check_dc_equivalence.py --config-name consistency_check seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("����һ���Լ��ʧ�ܣ�")
        return 1
    
    # 2. ѵ��ģ��
    print("2. ��ʼѵ��...")
    cmd = f"python tools/train.py --config-path ../configs --config-name {Path(args.config).stem} seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("ѵ��ʧ�ܣ�")
        return 1
    
    # 3. ����ģ��
    print("3. ��ʼ����...")
    cmd = f"python tools/eval.py --config-path ../configs --config-name {Path(args.config).stem} seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("����ʧ�ܣ�")
        return 1
    
    print("=== ʵ�鸴����� ===")
    return 0

if __name__ == "__main__":
    exit(main())
