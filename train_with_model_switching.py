#!/usr/bin/env python3
"""
简化模型切换训练脚本
专门用于我们已经测试通过的核心模型：UNet 和 SwinUNet
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='简化模型切换训练脚本')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['UNet', 'SwinUNet'],
                       help='要使用的模型（仅支持已测试通过的模型）')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练epoch数（默认：5）')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='批次大小（默认：4）')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率（默认：1e-4）')
    parser.add_argument('--data', type=str, 
                       default='datasets/sample_data.hdf5',
                       help='数据文件路径')
    parser.add_argument('--output', type=str, default='runs',
                       help='输出目录')
    parser.add_argument('--test-first', action='store_true',
                       help='先运行模型测试再训练')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 简化模型切换训练系统")
    print("="*60)
    print(f"📊 选择模型: {args.model}")
    print(f"⏱️  训练epoch: {args.epochs}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📈 学习率: {args.lr}")
    print(f"📁 数据文件: {args.data}")
    print(f"📂 输出目录: {args.output}")
    print()
    
    # 验证数据文件存在，如果不存在则使用默认的测试数据
    if not Path(args.data).exists():
        print(f"⚠️  指定的数据文件不存在: {args.data}")
        # 使用项目中的测试数据
        test_data_path = "test_data.h5"
        if Path(test_data_path).exists():
            args.data = test_data_path
            print(f"✅ 使用测试数据: {args.data}")
        else:
            print("❌ 未找到可用的数据文件")
            return 1
    
    # 模型测试
    if args.test_first:
        print("🧪 运行模型测试...")
        test_cmd = f"""
import torch
from models import {args.model}

# 测试模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'在设备 {{device}} 上测试 {args.model}...')

if '{args.model}' == 'UNet':
    model = {args.model}(in_channels=1, out_channels=1, img_size=128, features=[64, 128, 256, 512], bilinear=True)
elif '{args.model}' == 'SwinUNet':
    model = {args.model}(in_chans=1, num_classes=1, img_size=128, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=8)

model = model.to(device)
model.eval()

# 测试输入
x = torch.randn(1, 1, 128, 128).to(device)
with torch.no_grad():
    output = model(x)

print(f"✅ 模型测试通过！")
print(f"📊 输出形状: {{output.shape}}")
print(f"🔢 参数量: {{sum(p.numel() for p in model.parameters()):,}}")
print(f"💾 内存使用: {{torch.cuda.max_memory_allocated() / 1024**2:.1f}}MB" if torch.cuda.is_available() else "💾 CPU模式")
"""
        
        try:
            result = subprocess.run([sys.executable, '-c', test_cmd], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ 模型测试通过！")
                print(result.stdout)
            else:
                print("❌ 模型测试失败！")
                print(result.stderr)
                return 1
        except Exception as e:
            print(f"❌ 模型测试异常: {e}")
            return 1
    
    # 创建临时配置文件 - 基于最小化调试配置
    config_content = f"""
# 临时训练配置 - {args.model}模型
defaults:
  - _self_

# 最小数据配置
data:
  dataset_name: RealDiffusionReaction
  data_path: "{args.data}"  # 使用实际数据文件
  use_synthetic_data: false  # 使用真实数据
  
  T_in: 1
  T_out: 1
  image_size: 128
  sample_limit: 50
  
  train_ratio: 0.8
  val_ratio: 0.15
  test_ratio: 0.05
  
  normalize: true
  augmentation:
    enabled: false
  
  # 数据加载器配置
  dataloader:
    batch_size: {args.batch_size}
    val_batch_size: {args.batch_size}
    test_batch_size: 1
    num_workers: 0   # 禁用多进程
    pin_memory: false
    persistent_workers: false
    prefetch_factor: 2
    drop_last: true
    shuffle: true

# 模型配置
model:
  name: {args.model.lower()}
  in_channels: 1
  out_channels: 1
  img_size: 128
"""
    
    # 添加模型特定参数
    if args.model == 'SwinUNet':
        config_content += """
  embed_dim: 96
  depths: [2, 2, 6, 2]
  window_size: 8
  patch_size: 4"""
    
    config_content += f"""

# 训练配置
training:
  epochs: {args.epochs}
  batch_size: {args.batch_size}
  torch_compile: false
  
  # 优化器配置
  optimizer:
    name: "AdamW"
    lr: 0.001
    weight_decay: 0.0
    fused: false
    foreach: false
  
  # 调度器配置
  scheduler:
    name: "CosineAnnealingLR"
    T_max: {args.epochs}
    eta_min: 1e-6
  
  # 基本设置
  gradient_clip_val: 0
  
  # AMP配置
  amp:
    enabled: true
    autocast_dtype: bfloat16
  
  # 验证配置
  validation:
    enabled: true
    check_val_every_n_epoch: 1
    log_val_metrics: false
  
  # 检查点配置
  checkpoint:
    save_best: false
    save_last: false
    max_keep: 1
  
  # 禁用早停
  early_stopping:
    enabled: false

# 损失配置
loss:
  reconstruction:
    weight: 1.0
  spectral:
    weight: 0.0
  degradation_consistency:
    weight: 0.0
  gradient_weight: 0.0

# 日志配置
logging:
  experiment_name: "{args.model.lower()}_quick_train"
  log_every_n_steps: 1
  performance_monitoring:
    log_gpu_memory: false
    log_throughput: false
    log_batch_time: false
  tensorboard:
    save_dir: runs/tensorboard
    name: {args.model.lower()}_quick_train
  visualization:
    save_samples_every_n_epochs: 0
    num_samples_to_save: 0
    save_training_curves: false

# 测试配置
testing:
  enabled: false

# 禁用数据增强
data_augmentation:
  enable: false

# 随机种子
seed: 2025

# 实验配置
experiment:
  precision: 16-mixed
  name: {args.model.lower()}_quick_train
  output_dir: {args.output}
  device: cuda:0
  seed: 2025
"""
    
    # 保存临时配置文件
    temp_config = f"temp_{args.model.lower()}_train.yaml"
    with open(temp_config, 'w') as f:
        f.write(config_content)
    
    print(f"💾 配置文件已创建: {temp_config}")
    
    # 开始训练
    print("🚀 开始训练...")
    print("="*60)
    
    cmd = [
        sys.executable, 'train.py',
        '--config-path', '.',
        '--config-name', temp_config.replace('.yaml', '')
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 运行训练
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✅ 训练完成！")
            print(f"📊 结果保存在: {args.output}")
            
            # 清理临时文件
            if Path(temp_config).exists():
                Path(temp_config).unlink()
                print(f"🧹 清理临时配置文件: {temp_config}")
            
            return 0
        else:
            print(f"\n❌ 训练失败，返回码: {result.returncode}")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        # 清理临时文件
        if Path(temp_config).exists():
            Path(temp_config).unlink()
        return 0
    except Exception as e:
        print(f"\n❌ 训练执行失败: {e}")
        # 清理临时文件
        if Path(temp_config).exists():
            Path(temp_config).unlink()
        return 1

if __name__ == "__main__":
    sys.exit(main())