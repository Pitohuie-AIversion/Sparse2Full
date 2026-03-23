#!/usr/bin/env python3
"""
AR训练模型切换工具
用于快速切换和训练不同的模型架构
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def get_available_models():
    """获取可用模型列表"""
    return {
        'swinunet': 'SwinUNet - 基于Swin Transformer的U-Net架构',
        'unet': 'UNet - 经典U-Net架构',
        'fno': 'FNO2d - 傅里叶神经算子',
        'segformer': 'SegFormer - 高效的语义分割Transformer',
        'ufno': 'UFNOUNet - U-FNO混合架构',
        'mlpmixer': 'MLPMixer - 纯MLP架构',
        'hybrid': 'HybridModel - 混合模型',
        'unetplusplus': 'UNetPlusPlus - 嵌套U-Net',
        'vit': 'VisionTransformer - 视觉Transformer',
        'liif': 'LIIFModel - 隐式神经表示'
    }

def get_config_file(model_name):
    """获取模型对应的配置文件"""
    config_files = {
        'swinunet': 'configs/train/ar_training_config debug.yaml',
        'unet': 'configs/train/ar_training_config_unet.yaml',
        'fno': 'configs/train/ar_training_config_fno.yaml',
        'segformer': 'configs/train/ar_training_config_segformer.yaml',
        'ufno': 'configs/train/ar_training_config_ufno.yaml',
        'mlpmixer': 'configs/train/ar_training_config_mlpmixer.yaml',
        'hybrid': 'configs/train/ar_training_config_hybrid.yaml',
        'unetplusplus': 'configs/train/ar_training_config_unetplusplus.yaml',
        'vit': 'configs/train/ar_training_config_vit.yaml',
        'liif': 'configs/train/ar_training_config_liif.yaml'
    }
    return config_files.get(model_name)

def create_config_if_not_exists(model_name):
    """如果配置文件不存在，创建默认配置"""
    config_file = get_config_file(model_name)
    if config_file and not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在，创建默认配置...")
        # 这里可以添加创建默认配置的逻辑
        # 暂时使用基础配置作为模板
        base_config = 'configs/train/ar_training_config debug.yaml'
        if os.path.exists(base_config):
            import shutil
            shutil.copy(base_config, config_file)
            print(f"已创建默认配置文件: {config_file}")
        else:
            print(f"警告: 基础配置文件 {base_config} 不存在")
            return None
    return config_file

def main():
    parser = argparse.ArgumentParser(description='AR训练模型切换工具')
    parser.add_argument('model', type=str, nargs='?', help='模型名称')
    parser.add_argument('--dry-run', action='store_true', help='仅显示命令，不执行')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--config', type=str, help='自定义配置文件')
    parser.add_argument('--list-models', action='store_true', help='列出可用模型')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("可用模型:")
        models = get_available_models()
        for name, desc in models.items():
            print(f"  {name:15} - {desc}")
        return 0
    
    if args.model is None:
        print("错误: 请指定模型名称或使用 --list-models 查看可用模型")
        return 1
    
    model_name = args.model.lower()
    available_models = get_available_models()
    
    if model_name not in available_models:
        print(f"错误: 未知的模型名称 '{model_name}'")
        print("可用模型:")
        for name, desc in available_models.items():
            print(f"  {name:15} - {desc}")
        return 1
    
    # 获取配置文件
    if args.config:
        config_file = args.config
    else:
        config_file = create_config_if_not_exists(model_name)
        if not config_file:
            print(f"错误: 无法获取模型 '{model_name}' 的配置文件")
            return 1
    
    if not os.path.exists(config_file):
        print(f"错误: 配置文件 '{config_file}' 不存在")
        return 1
    
    print(f"模型: {model_name} - {available_models[model_name]}")
    print(f"配置文件: {config_file}")
    
    # 构建训练命令
    train_script = "tools/training/train_real_data_ar.py"
    cmd = [sys.executable, train_script, "--config", config_file]
    
    # 添加额外参数
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    
    print(f"\n执行命令:")
    print(" ".join(cmd))
    
    if args.dry_run:
        print("\n[干运行模式] 命令不会实际执行")
        return 0
    
    # 执行训练
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"训练失败，返回码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return 130

if __name__ == "__main__":
    sys.exit(main())