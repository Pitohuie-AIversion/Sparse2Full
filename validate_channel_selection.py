#!/usr/bin/env python3
"""
通道选择验证脚本
帮助你精确控制要保留的通道
"""

import torch
import numpy as np
from pathlib import Path
import h5py
import yaml

def analyze_hdf5_data_structure(data_path):
    """分析HDF5文件的数据结构"""
    print(f"📁 分析数据文件: {data_path}")
    
    try:
        with h5py.File(data_path, 'r') as f:
            print("🔍 文件结构:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"   📊 {name}: 形状{obj.shape}, 类型{obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"   📂 {name}/")
            
            f.visititems(print_structure)
            
            # 检查主要数据键
            if 'data' in f:
                data_shape = f['data'].shape
                print(f"\n📈 主要数据信息:")
                print(f"   - 'data' 形状: {data_shape}")
                print(f"   - 通道数: {data_shape[-1] if len(data_shape) > 3 else '未知'}")
                
                # 读取样本数据
                sample_data = f['data'][0, 0] if len(data_shape) >= 4 else f['data'][0]
                print(f"   - 样本形状: {sample_data.shape}")
                
                return data_shape, sample_data.shape
            else:
                print("⚠️  未找到'data'键")
                return None, None
                
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None, None

def simulate_channel_selection(data_config):
    """模拟通道选择过程"""
    print(f"\n🎯 模拟通道选择...")
    
    keys = data_config.get('keys', ['data'])
    input_channels = data_config.get('input_channels', 1)
    component = data_config.get('component', 'all')
    
    print(f"   - 数据键: {keys}")
    print(f"   - 输入通道: {input_channels}")
    print(f"   - 分量选择: {component}")
    
    # 模拟不同场景
    scenarios = []
    
    if 'u' in keys and 'v' in keys:
        # 分立分量模式
        if component == 'u':
            scenarios.append("选择u分量 (第1通道)")
        elif component == 'v':
            scenarios.append("选择v分量 (第2通道)")
        elif component == 'all':
            scenarios.append("选择u和v分量 (2通道)")
    elif 'data' in keys:
        # 统一数据模式
        if input_channels == 1:
            scenarios.append("从data中选择第1个通道")
        elif input_channels == 2:
            scenarios.append("从data中选择前2个通道")
        else:
            scenarios.append(f"从data中选择前{input_channels}个通道")
    
    # 添加坐标和掩码的影响
    if data_config.get('observation', {}).get('mode') == 'SR':
        scenarios.append("应用SR降采样观测")
    
    print(f"   - 选择逻辑: {' → '.join(scenarios)}")
    
    return scenarios

def create_channel_selection_config(target_channel='u', use_observation=True):
    """创建通道选择配置"""
    print(f"\n⚙️  创建通道选择配置...")
    
    if target_channel in ['u', 'v']:
        # 方法1: 使用component参数
        config1 = {
            'keys': ['u', 'v'],
            'component': target_channel,
            'input_channels': 1,
            'target_channels': 1
        }
        print(f"✅ 方案1 - component模式:")
        print(f"   配置: {config1}")
        print(f"   说明: 直接选择{target_channel}分量")
    
    # 方法2: 使用通道索引
    channel_map = {'u': 0, 'v': 1}
    if target_channel in channel_map:
        config2 = {
            'keys': ['data'],
            'input_channels': 1,
            'target_channels': 1,
            'channel_index': channel_map[target_channel]
        }
        print(f"✅ 方案2 - 通道索引模式:")
        print(f"   配置: {config2}")
        print(f"   说明: 从data中选择第{channel_map[target_channel]+1}个通道")
    
    return config1 if target_channel in ['u', 'v'] else config2

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 通道选择配置验证工具")
    print("=" * 60)
    
    # 当前配置文件路径
    config_path = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/train/ar_training_config debug.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_config = config.get('data', {})
        
        print(f"\n📋 当前配置分析:")
        print(f"   配置文件: {config_path}")
        
        # 分析数据文件结构
        data_path = data_config.get('data_path', '')
        if data_path and Path(data_path).exists():
            analyze_hdf5_data_structure(data_path)
        else:
            print(f"⚠️  数据文件不存在或路径无效: {data_path}")
        
        # 模拟当前通道选择
        simulate_channel_selection(data_config)
        
        # 提供通道选择建议
        print(f"\n💡 通道选择建议:")
        print("1. 使用 'component: u' 选择u分量 (推荐)")
        print("2. 使用 'component: v' 选择v分量")
        print("3. 使用 'input_channels: 1' 从data选择第1通道")
        print("4. 使用 'keys: [\"u\"]' 直接指定u分量数据")
        
        # 创建示例配置
        print(f"\n📝 示例配置:")
        create_channel_selection_config('u')
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")

if __name__ == "__main__":
    main()