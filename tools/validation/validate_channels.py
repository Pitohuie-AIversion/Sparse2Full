#!/usr/bin/env python3
"""
通道配置验证脚本
用于确认模型实际使用的观测通道
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def validate_channel_configuration(config_path):
    """验证通道配置的一致性"""
    print("🔍 开始验证通道配置...")

    # 加载配置
    config = OmegaConf.load(config_path)

    # 1. 检查数据配置
    data_config = config.get('data', {})
    input_channels = data_config.get('input_channels', 1)
    target_channels = data_config.get('target_channels', 1)

    print(f"📊 数据配置:")
    print(f"   - input_channels: {input_channels}")
    print(f"   - target_channels: {target_channels}")

    # 2. 检查模型配置
    model_config = config.get('model', {})
    model_in_channels = model_config.get('in_channels', 1)
    model_out_channels = model_config.get('out_channels', 1)

    print(f"🧠 模型配置:")
    print(f"   - in_channels: {model_in_channels}")
    print(f"   - out_channels: {model_out_channels}")

    # 3. 检查观测配置
    observation_config = data_config.get('observation', {})
    obs_mode = observation_config.get('mode', 'none')

    print(f"👁️ 观测配置:")
    print(f"   - mode: {obs_mode}")
    if obs_mode == 'SR':
        sr_config = observation_config.get('sr', {})
        scale_factor = sr_config.get('scale_factor', 1)
        print(f"   - scale_factor: {scale_factor}")
        print(f"   - blur_sigma: {sr_config.get('blur_sigma', 0)}")

    # 4. 检查分量配置
    component = data_config.get('component', 'all')
    print(f"🎯 分量配置:")
    print(f"   - component: {component}")

    # 5. 一致性检查
    print(f"\n✅ 一致性检查:")

    # 检查输入通道一致性
    if input_channels == model_in_channels:
        print(f"   ✅ 输入通道一致: data({input_channels}) == model({model_in_channels})")
    else:
        print(f"   ⚠️  输入通道不一致: data({input_channels}) != model({model_in_channels})")
        print(f"      系统将以 data.input_channels 为准")

    # 检查输出通道一致性  
    if target_channels == model_out_channels:
        print(f"   ✅ 输出通道一致: data({target_channels}) == model({model_out_channels})")
    else:
        print(f"   ⚠️  输出通道不一致: data({target_channels}) != model({model_out_channels})")

    # 6. 计算理论通道需求
    print(f"\n📐 理论通道分析:")
    base_channels = input_channels  # 基础观测通道
    coord_channels = 2 if model_config.get('use_coords', False) else 0  # 坐标通道
    mask_channels = 1 if obs_mode != 'none' else 0  # 掩码通道

    total_expected = base_channels + coord_channels + mask_channels

    print(f"   - 基础观测通道: {base_channels}")
    print(f"   - 坐标编码通道: {coord_channels}")
    print(f"   - 观测掩码通道: {mask_channels}")
    print(f"   - 预期总通道: {total_expected}")
    print(f"   - 模型配置通道: {model_in_channels}")

    if total_expected == model_in_channels:
        print(f"   ✅ 通道配置正确!")
    else:
        print(f"   ⚠️  通道配置可能有问题!")
        print(f"      建议修改 model.in_channels 为 {total_expected}")

    return {
        'data_input_channels': input_channels,
        'model_in_channels': model_in_channels,
        'expected_total': total_expected,
        'is_consistent': total_expected == model_in_channels
    }


def simulate_data_flow(config):
    """模拟数据流，验证通道处理"""
    print(f"\n🔄 模拟数据流验证...")

    data_config = config.get('data', {})
    model_config = config.get('model', {})

    # 模拟输入数据
    batch_size = 2
    img_size = data_config.get('img_size', 128)
    input_channels = data_config.get('input_channels', 1)

    # 原始高分辨率数据
    hr_data = torch.randn(batch_size, input_channels, img_size, img_size)
    print(f"   - 原始HR数据形状: {hr_data.shape}")

    # 观测处理（降采样）
    observation_config = data_config.get('observation', {})
    if observation_config.get('mode') == 'SR':
        sr_config = observation_config.get('sr', {})
        scale_factor = sr_config.get('scale_factor', 1)

        # 模拟降采样
        lr_size = img_size // scale_factor
        lr_data = torch.nn.functional.interpolate(
            hr_data, size=(lr_size, lr_size), mode='area'
        )
        print(f"   - 观测LR数据形状: {lr_data.shape}")

        # 上采样回原始尺寸（模拟模型输入）
        input_data = torch.nn.functional.interpolate(
            lr_data, size=(img_size, img_size), mode='bilinear'
        )
        print(f"   - 模型输入数据形状: {input_data.shape}")
    else:
        input_data = hr_data
        print(f"   - 无观测处理，直接输入: {input_data.shape}")

    # 坐标编码
    if model_config.get('use_coords', False):
        # 创建坐标网格
        coords = torch.randn(batch_size, 2, img_size, img_size)
        input_data = torch.cat([input_data, coords], dim=1)
        print(f"   - 添加坐标编码后: {input_data.shape}")

    # 观测掩码
    if observation_config.get('mode') != 'none':
        # 创建掩码（标识观测区域）
        mask = torch.ones(batch_size, 1, img_size, img_size)
        input_data = torch.cat([input_data, mask], dim=1)
        print(f"   - 添加观测掩码后: {input_data.shape}")

    print(f"   ✅ 最终模型输入形状: {input_data.shape}")
    print(f"   📊 通道维度: {input_data.shape[1]} (期望: {model_config.get('in_channels', 1)})")

    return input_data.shape


def _resolve_default_config() -> Path:
    """返回仓库内可用的默认配置路径。"""
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / 'configs' / 'train' / 'ar_training_config_debug.yaml',
        project_root / 'configs' / 'train' / 'ar_training_config debug.yaml',
        project_root / 'configs' / 'train' / 'ar_training_config.yaml',
        project_root / 'configs' / 'ar_training_config_debug.yaml',
        project_root / 'configs' / 'ar_training_config.yaml',
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"未找到可用默认配置。请使用 --config 显式指定。已尝试: {[str(path) for path in candidates]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通道配置验证与数据流模拟工具")
    parser.add_argument(
        '--config',
        default=None,
        help='配置文件路径（YAML）。未指定时自动选择仓库内默认配置。',
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser() if args.config else _resolve_default_config()

    print("=" * 60)
    print("通道配置验证工具")
    print("=" * 60)
    print(f"配置路径: {config_path}")

    # 验证配置
    validation_result = validate_channel_configuration(config_path)

    # 加载配置进行数据流模拟
    config = OmegaConf.load(config_path)
    final_shape = simulate_data_flow(config)

    print(f"\n" + "=" * 60)
    if validation_result['is_consistent']:
        print("✅ 通道配置验证通过!")
    else:
        print("⚠️  建议修正通道配置")

    print("=" * 50)
