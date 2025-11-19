#!/usr/bin/env python3
"""
AR训练完整实验启动器
用于启动不同模型的完整训练实验
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime

def get_model_configs():
    """获取模型配置映射"""
    return {
        'swinunet': {
            'name': 'SwinUNet',
            'debug_config': 'configs/train/ar_training_config_debug_swinunet.yaml',
            'full_config': 'configs/train/ar_training_config debug.yaml',
            'description': 'Swin Transformer U-Net - 基于Transformer的U-Net架构'
        },
        'unet': {
            'name': 'UNet',
            'debug_config': 'configs/train/ar_training_config_debug_unet.yaml',
            'full_config': 'configs/train/ar_training_config_unet.yaml',
            'description': '经典U-Net架构 - 卷积神经网络'
        },
        'fno2d': {
            'name': 'FNO2d',
            'debug_config': 'configs/train/ar_training_config_debug_fno2d_single.yaml',
            'full_config': 'configs/train/ar_training_config_fno.yaml',
            'description': '傅里叶神经算子 - 频域建模'
        },
        'segformer': {
            'name': 'SegFormer',
            'debug_config': 'configs/train/ar_training_config_debug_segformer.yaml',
            'full_config': 'configs/train/ar_training_config_segformer.yaml',
            'description': '高效Transformer架构 - 语义分割'
        }
    }

def run_training(model_name, mode='debug', epochs=None, devices=2, dry_run=False):
    """运行训练"""
    configs = get_model_configs()
    
    if model_name not in configs:
        print(f"错误: 未知的模型 '{model_name}'")
        print("可用模型:", list(configs.keys()))
        return False
    
    config = configs[model_name]
    config_file = config['debug_config'] if mode == 'debug' else config['full_config']
    
    if not os.path.exists(config_file):
        print(f"错误: 配置文件 '{config_file}' 不存在")
        return False
    
    print(f"\n{'='*60}")
    print(f"模型: {config['name']}")
    print(f"描述: {config['description']}")
    print(f"模式: {mode}")
    print(f"配置: {config_file}")
    print(f"{'='*60}\n")
    
    # 构建训练命令
    cmd = [
        sys.executable,
        'tools/training/train_real_data_ar.py',
        '--config', config_file
    ]
    
    if devices:
        cmd.extend(['--devices', str(devices)])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    if dry_run:
        print("[干运行模式] 命令不会实际执行")
        return True
    
    # 记录实验信息
    experiment_info = {
        'model': model_name,
        'mode': mode,
        'config': config_file,
        'timestamp': datetime.now().isoformat(),
        'command': ' '.join(cmd)
    }
    
    # 保存实验记录
    log_dir = Path('runs/experiment_logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"{model_name}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    
    print(f"实验记录已保存: {log_file}")
    
    try:
        # 执行训练
        result = subprocess.run(cmd, check=True)
        print(f"✅ {config['name']} 训练完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {config['name']} 训练失败，返回码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {config['name']} 训练被用户中断")
        return False

def run_batch_experiments(models, mode='debug', **kwargs):
    """批量运行实验"""
    results = {}
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"开始训练模型: {model}")
        print(f"{'='*80}")
        
        success = run_training(model, mode=mode, **kwargs)
        results[model] = success
        
        if not success:
            print(f"模型 {model} 训练失败，跳过...")
            continue
    
    # 打印总结
    print(f"\n{'='*80}")
    print("实验总结:")
    print(f"{'='*80}")
    
    for model, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{model:15} - {status}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='AR训练完整实验启动器')
    parser.add_argument('models', nargs='*', help='要训练的模型名称')
    parser.add_argument('--mode', choices=['debug', 'full'], default='debug',
                        help='训练模式: debug(快速调试) 或 full(完整训练)')
    parser.add_argument('--all', action='store_true', help='训练所有可用模型')
    parser.add_argument('--devices', type=int, default=2, help='使用的GPU数量')
    parser.add_argument('--dry-run', action='store_true', help='干运行模式')
    parser.add_argument('--list-models', action='store_true', help='列出可用模型')
    
    args = parser.parse_args()
    
    configs = get_model_configs()
    
    if args.list_models:
        print("可用模型:")
        for name, info in configs.items():
            print(f"  {name:12} - {info['description']}")
        return
    
    # 确定要训练的模型
    if args.all:
        models = list(configs.keys())
    elif args.models:
        models = args.models
    else:
        print("错误: 请指定模型名称或使用 --all 训练所有模型")
        print("使用 --list-models 查看可用模型")
        return
    
    # 验证模型名称
    invalid_models = [m for m in models if m not in configs]
    if invalid_models:
        print(f"错误: 未知的模型: {invalid_models}")
        print("可用模型:", list(configs.keys()))
        return
    
    print(f"\n🚀 开始AR训练实验")
    print(f"模式: {args.mode}")
    print(f"模型: {models}")
    print(f"设备: {args.devices} GPU(s)")
    
    # 运行批量实验
    results = run_batch_experiments(
        models, 
        mode=args.mode,
        devices=args.devices,
        dry_run=args.dry_run
    )
    
    # 统计结果
    successful = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n📊 实验完成统计: {successful}/{total} 模型训练成功")
    
    if successful == total:
        print("🎉 所有模型训练成功！")
    else:
        print("⚠️  部分模型训练失败，请检查日志")

if __name__ == "__main__":
    main()