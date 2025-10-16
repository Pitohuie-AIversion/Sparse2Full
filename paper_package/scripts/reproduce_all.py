#!/usr/bin/env python3
"""
PDEBench稀疏观测重建实验复现脚本

使用方法:
    python reproduce_all.py --config config_merged.yaml --seed 42

要求:
    - Python 3.10+
    - PyTorch >= 2.1
    - 所有依赖包已安装
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并打印输出"""
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
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    print("=== PDEBench实验复现 ===")
    print(f"配置文件: {args.config}")
    print(f"随机种子: {args.seed}")
    print(f"GPU设备: {args.gpu}")
    print()
    
    # 1. 数据一致性检查
    print("1. 运行数据一致性检查...")
    cmd = f"python tools/check_dc_equivalence.py --config-name consistency_check seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("数据一致性检查失败！")
        return 1
    
    # 2. 训练模型
    print("2. 开始训练...")
    cmd = f"python tools/train.py --config-path ../configs --config-name {Path(args.config).stem} seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("训练失败！")
        return 1
    
    # 3. 评估模型
    print("3. 开始评估...")
    cmd = f"python tools/eval.py --config-path ../configs --config-name {Path(args.config).stem} seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("评估失败！")
        return 1
    
    print("=== 实验复现完成 ===")
    return 0

if __name__ == "__main__":
    exit(main())
