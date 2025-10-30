#!/usr/bin/env python3
"""
优化版300轮时序NAR训练启动脚本
确保高效GPU利用率和完整300轮训练
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import psutil


def check_gpu_status():
    """检查GPU状态"""
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法使用GPU训练")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return True


def check_system_resources():
    """检查系统资源"""
    # CPU信息
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    
    print(f"💻 系统资源:")
    print(f"   CPU核心数: {cpu_count}")
    print(f"   内存: {memory.total / 1024**3:.1f} GB (可用: {memory.available / 1024**3:.1f} GB)")
    
    # 检查内存是否充足
    if memory.available < 8 * 1024**3:  # 8GB
        print("⚠️  警告: 可用内存不足8GB，可能影响训练性能")
    
    return True


def optimize_environment():
    """优化环境设置"""
    print("🔧 优化环境设置...")
    
    # CUDA优化
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步CUDA操作
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # 启用cuDNN v8 API
    
    # PyTorch优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # CUDA内存管理
    
    # 多进程优化
    os.environ['OMP_NUM_THREADS'] = str(min(8, psutil.cpu_count()))  # OpenMP线程数
    
    print("✅ 环境优化完成")


def create_experiment_config():
    """创建实验配置"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"OptimizedTemporalNAR-300epochs-{timestamp}"
    
    config_overrides = [
        f"experiment.name={experiment_name}",
        f"experiment.output_dir=runs/optimized_300epochs",
        "train.max_epochs=300",
        "experiment.use_amp=true",
        "data.batch_size=12",
        "data.num_workers=12",
        "data.prefetch_factor=6",
        "train.optimizer.lr=3e-3",
        "train.scheduler.T_0=30",
        "train.scheduler.eta_min=1e-7",
        "train.early_stopping.patience=25",
        "experiment.log_every_n_steps=10",
        "experiment.val_check_interval=100",
        "monitoring.gpu_monitoring.enabled=true",
        "monitoring.gpu_monitoring.log_interval=50",
        "monitoring.visualization.enabled=true",
        "monitoring.visualization.save_every_n_epochs=20"
    ]
    
    return experiment_name, config_overrides


def run_training(config_overrides, use_optimized_script=True):
    """运行训练"""
    script_name = "train_temporal_nar_300epochs_optimized.py" if use_optimized_script else "train_temporal_nar_300epochs.py"
    config_name = "temporal_nar_300epochs_optimized"
    
    # 构建命令 - 修复Hydra参数格式
    cmd = [
        sys.executable, script_name,
        "--config-name", config_name
    ]
    
    # 添加配置覆盖 - 使用正确的Hydra语法
    cmd.extend(config_overrides)
    
    print(f"🚀 启动优化训练命令:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # 记录启动信息
    start_time = time.time()
    
    try:
        # 运行训练
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出日志
        for line in process.stdout:
            print(line.rstrip())
        
        # 等待进程完成
        return_code = process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if return_code == 0:
            print(f"✅ 训练成功完成! 耗时: {duration/3600:.2f} 小时")
        else:
            print(f"❌ 训练失败，返回码: {return_code}")
        
        return return_code == 0
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        process.terminate()
        return False
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化版300轮时序NAR训练启动器")
    parser.add_argument("--skip-checks", action="store_true", help="跳过系统检查")
    parser.add_argument("--use-original", action="store_true", help="使用原始训练脚本")
    parser.add_argument("--batch-size", type=int, default=12, help="批处理大小")
    parser.add_argument("--num-workers", type=int, default=12, help="数据加载器工作进程数")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="学习率")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 优化版时序NAR模型300轮训练启动器")
    print("=" * 80)
    print()
    
    # 系统检查
    if not args.skip_checks:
        print("🔍 系统检查...")
        if not check_gpu_status():
            return 1
        
        if not check_system_resources():
            return 1
        
        print()
    
    # 环境优化
    optimize_environment()
    print()
    
    # 创建实验配置
    experiment_name, config_overrides = create_experiment_config()
    
    # 应用命令行参数
    if args.batch_size != 12:
        config_overrides.append(f"data.batch_size={args.batch_size}")
    if args.num_workers != 12:
        config_overrides.append(f"data.num_workers={args.num_workers}")
    if args.learning_rate != 3e-3:
        config_overrides.append(f"train.optimizer.lr={args.learning_rate}")
    
    print(f"📋 实验配置:")
    print(f"   实验名称: {experiment_name}")
    print(f"   训练轮数: 300")
    print(f"   批处理大小: {args.batch_size}")
    print(f"   数据加载器工作进程: {args.num_workers}")
    print(f"   学习率: {args.learning_rate}")
    print(f"   混合精度训练: 启用")
    print(f"   GPU监控: 启用")
    print()
    
    # 确认开始训练
    try:
        response = input("🤔 是否开始训练? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ 训练已取消")
            return 0
    except KeyboardInterrupt:
        print("\n❌ 训练已取消")
        return 0
    
    print()
    print("🎯 开始优化训练...")
    print("=" * 80)
    
    # 运行训练
    success = run_training(config_overrides, use_optimized_script=not args.use_original)
    
    print("=" * 80)
    if success:
        print("🎉 训练任务完成!")
        print(f"📁 结果保存在: runs/optimized_300epochs/{experiment_name}/")
        print("📊 可以使用TensorBoard查看训练过程:")
        print(f"   tensorboard --logdir runs/optimized_300epochs/{experiment_name}/tensorboard")
    else:
        print("💥 训练任务失败!")
        print("🔍 请检查错误日志并重试")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())