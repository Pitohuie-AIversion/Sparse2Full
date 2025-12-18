#!/usr/bin/env python3
"""
AR模型测试脚本
用于单独测试已训练的AR模型
支持完整的测试集评估和指标计算
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from train_real_data_ar import RealDataARTrainer
from utils.logger import setup_logger


if __name__ != "__main__":
    import pytest

    pytest.skip(
        "脚本型 AR checkpoint 测试不作为 pytest 单测执行（需要本地配置与权重文件）。",
        allow_module_level=True,
    )


def test_ar_model(config_path: str, checkpoint_path: str, output_dir: str = None):
    """测试AR模型
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
    """
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent / 'test_results'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # 设置日志
    logger = setup_logger('AR_Test', output_dir / 'test.log')
    logger.info("🧪 开始AR模型测试...")
    
    try:
        # 创建训练器（用于加载模型和数据）
        trainer = RealDataARTrainer(config_path)
        
        # 加载检查点
        if not trainer.load_checkpoint(checkpoint_path):
            raise ValueError(f"无法加载检查点: {checkpoint_path}")
        
        logger.info(f"✅ 成功加载模型: {checkpoint_path}")
        
        # 运行测试
        test_metrics = trainer.test_epoch()
        
        # 保存详细测试结果
        test_results = {
            'checkpoint_path': str(checkpoint_path),
            'config_path': str(config_path),
            'test_metrics': test_metrics,
            'model_info': {
                'epoch': trainer.current_epoch,
                'best_val_loss': trainer.best_val_loss,
            },
            'data_info': {
                'train_batches': len(trainer.train_loader),
                'val_batches': len(trainer.val_loader),
                'test_batches': len(trainer.test_loader),
            }
        }
        
        # 保存结果
        results_file = output_dir / 'detailed_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"✅ 测试完成，结果保存到: {results_file}")
        
        # 打印主要指标
        print("\n" + "="*50)
        print("🎯 AR模型测试结果")
        print("="*50)
        print(f"模型: {Path(checkpoint_path).name}")
        print(f"Epoch: {trainer.current_epoch}")
        print(f"验证损失: {trainer.best_val_loss:.6f}")
        print("\n📊 测试集指标:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.6f}")
        print("="*50)
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AR模型测试")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        return
    
    # 运行测试
    test_ar_model(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
