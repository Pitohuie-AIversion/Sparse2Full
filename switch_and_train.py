#!/usr/bin/env python3
"""
模型更换和训练脚本
支持在测试通过的核心模型之间切换
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def get_available_models():
    """获取可用的核心模型列表"""
    return {
        'UNet': {
            'description': '经典U-Net架构，参数少，推理快',
            'config': {
                'name': 'UNet',
                'features': [64, 128, 256, 512],
                'bilinear': True,
                'dropout': 0.0
            }
        },
        'SwinUNet': {
            'description': '基于Swin Transformer的U-Net，性能强大',
            'config': {
                'name': 'SwinUNet', 
                'patch_size': 4,
                'window_size': 8,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'embed_dim': 96,
                'mlp_ratio': 4.0
            }
        },
        'Hybrid': {
            'description': 'Attention∥FNO∥UNet混合架构（需要配置修正）',
            'config': {
                'name': 'Hybrid',
                'use_attention': True,
                'use_fno': True,
                'use_unet': True,
                'fusion_strategy': 'concat'
            }
        },
        'FNO2d': {
            'description': '2D傅里叶神经算子（需要配置修正）',
            'config': {
                'name': 'FNO2d',
                'modes': 16,
                'width': 32,
                'layers': 4
            }
        }
    }

def update_config_file(model_name, config_path):
    """更新配置文件中的模型设置"""
    
    # 读取当前配置
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # 根据选择的模型更新配置
    available_models = get_available_models()
    
    if model_name not in available_models:
        print(f"❌ 不支持的模型: {model_name}")
        print("可用模型:")
        for name, info in available_models.items():
            print(f"  - {name}: {info['description']}")
        return False
    
    model_info = available_models[model_name]
    
    # 更新模型名称
    config_content = config_content.replace(
        'name: "SwinUNet"',  # 默认模型
        f'name: "{model_name}"'
    )
    
    # 根据模型类型更新kwargs部分
    if model_name == 'UNet':
        kwargs_section = '''      # === UNet 特定参数 ===
      features: [64, 128, 256, 512]
      bilinear: true
      dropout: 0.0'''
    elif model_name == 'SwinUNet':
        kwargs_section = '''      # === SwinUNet 特定参数 ===
      patch_size: 4
      window_size: 8
      depths: [2, 2, 6, 2]
      num_heads: [3, 6, 12, 24]
      embed_dim: 96
      mlp_ratio: 4.0
      drop_rate: 0.0
      attn_drop_rate: 0.0
      drop_path_rate: 0.1'''
    elif model_name == 'Hybrid':
        kwargs_section = '''      # === Hybrid 特定参数 ===
      use_attention: true
      use_fno: true
      use_unet: true
      fusion_strategy: "concat"'''
    elif model_name == 'FNO2d':
        kwargs_section = '''      # === FNO2d 特定参数 ===
      modes: 16
      width: 32
      layers: 4'''
    
    # 替换kwargs部分
    import re
    pattern = r'# === SwinUNet 特定参数 ===.*?# === Hybrid 特定参数 ==='
    replacement = f'{kwargs_section}\n      \n      # === Hybrid 特定参数 ==='
    
    config_content = re.sub(pattern, replacement, config_content, flags=re.DOTALL)
    
    # 写入更新后的配置
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ 已更新配置文件为使用 {model_name} 模型")
    return True

def run_model_test(model_name):
    """运行模型测试验证"""
    print(f"🧪 正在测试 {model_name} 模型配置...")
    
    # 运行我们之前创建的测试
    test_script = """
import torch
from models import {}

# 测试模型创建和基本功能
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根据模型类型设置参数
if '{}' == 'UNet':
    model = {}(in_channels=3, out_channels=3, img_size=128, features=[32, 64, 128, 256], bilinear=True)
elif '{}' == 'SwinUNet':
    model = {}(in_chans=3, num_classes=3, img_size=128, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=8)
elif '{}' == 'Hybrid':
    model = {}(in_ch=3, out_ch=3, img_size=128, backbone='swin', fusion='concat', attention_ch=64, fno_modes=16, fno_width=32)
elif '{}' == 'FNO2d':
    model = {}(modes=16, width=32, layers=4, in_channels=3, out_channels=3, img_size=128)

model = model.to(device)
model.eval()

# 测试输入
x = torch.randn(1, 3, 128, 128).to(device)
with torch.no_grad():
    output = model(x)

print(f"✅ 模型测试通过！输出形状: {{output.shape}}")
print(f"📊 参数量: {{sum(p.numel() for p in model.parameters()):,}}")
""".format(model_name, model_name, model_name, model_name, model_name, model_name, model_name, model_name)
    
    try:
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 模型测试通过！")
            print(result.stdout)
            return True
        else:
            print("❌ 模型测试失败！")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️ 模型测试超时")
        return False
    except Exception as e:
        print(f"❌ 模型测试异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='模型更换和训练脚本')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['UNet', 'SwinUNet', 'Hybrid', 'FNO2d'],
                       help='要使用的模型名称')
    parser.add_argument('--config', type=str, 
                       default='configs/train_model_switching_demo.yaml',
                       help='配置文件路径')
    parser.add_argument('--test-only', action='store_true',
                       help='仅运行测试，不开始训练')
    parser.add_argument('--skip-test', action='store_true',
                       help='跳过模型测试')
    
    args = parser.parse_args()
    
    print("="*60)
    print("模型更换和训练系统")
    print("="*60)
    
    # 获取模型信息
    available_models = get_available_models()
    model_info = available_models[args.model]
    
    print(f"选择的模型: {args.model}")
    print(f"模型描述: {model_info['description']}")
    print()
    
    # 更新配置文件
    if not update_config_file(args.model, args.config):
        print("❌ 配置文件更新失败")
        return 1
    
    # 运行模型测试
    if not args.skip_test:
        if not run_model_test(args.model):
            print("⚠️ 模型测试未通过，但仍可尝试训练")
            response = input("是否继续训练？(y/N): ")
            if response.lower() != 'y':
                return 1
    else:
        print("⏭️  跳过模型测试")
    
    if args.test_only:
        print("✅ 测试完成")
        return 0
    
    # 开始训练
    print("🚀 开始训练...")
    print(f"使用配置: {args.config}")
    
    cmd = [
        sys.executable, 'train.py',
        '--config-path', str(Path(args.config).parent),
        '--config-name', Path(args.config).stem
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        return 0
    except Exception as e:
        print(f"❌ 训练执行失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())