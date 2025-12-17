#!/usr/bin/env python3
"""
测试优化后的AR训练配置
验证双GPU配置、大批次训练和性能优化
"""

import os
import sys
import time
import torch
import psutil
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from tools.training.train_real_data_ar import RealDataARTrainer

def test_hardware_detection():
    """测试硬件检测"""
    print("🔧 硬件资源检测:")
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f} GB")
    else:
        print("未检测到CUDA GPU")
    
    # CPU信息
    cpu_count = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / 1024**3
    
    print(f"CPU: {cpu_count} 物理核心, {cpu_logical} 逻辑核心")
    print(f"系统内存: {memory_gb:.1f} GB")
    print()

def test_trainer_initialization():
    """测试训练器初始化"""
    print("🚀 测试训练器初始化:")
    
    try:
        # 使用优化后的配置文件
        config_path = "configs/ar_training_config.yaml"
        trainer = RealDataARTrainer(config_path)
        
        print("✅ 训练器初始化成功")
        
        # 检查配置
        print(f"📊 配置信息:")
        print(f"  批次大小: {trainer.config.data.dataloader.batch_size}")
        print(f"  Worker数量: {trainer.config.data.dataloader.num_workers}")
        print(f"  训练轮数: {trainer.config.training.epochs}")
        print(f"  学习率: {trainer.config.training.optimizer.lr}")
        print(f"  使用多GPU: {trainer.use_multi_gpu}")
        
        # 检查内存管理配置
        print(f"🧠 内存管理配置:")
        for key, value in trainer.memory_config.items():
            print(f"  {key}: {value}")
        
        # 检查数据加载器
        print(f"📦 数据加载器信息:")
        print(f"  训练批次数: {len(trainer.train_loader)}")
        print(f"  验证批次数: {len(trainer.val_loader)}")
        print(f"  测试批次数: {len(trainer.test_loader)}")
        
        # 检查模型
        model_for_params = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        total_params = sum(p.numel() for p in model_for_params.parameters())
        print(f"🏗️ 模型参数量: {total_params:,}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ 训练器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_usage():
    """测试内存使用情况"""
    print("💾 内存使用测试:")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"  GPU {i}:")
            print(f"    已分配: {allocated:.2f} GB")
            print(f"    已保留: {reserved:.2f} GB")
            print(f"    总显存: {total:.1f} GB")
            print(f"    使用率: {(allocated/total)*100:.1f}%")
    
    # 系统内存
    memory = psutil.virtual_memory()
    print(f"  系统内存使用率: {memory.percent:.1f}%")
    print()

def test_single_batch():
    """测试单个批次的前向传播"""
    print("🔄 测试单批次前向传播:")
    
    trainer = test_trainer_initialization()
    if trainer is None:
        return False
    
    try:
        # 获取一个批次的数据
        batch = next(iter(trainer.train_loader))
        inputs, targets = batch
        
        print(f"📊 批次信息:")
        print(f"  输入形状: {inputs.shape}")
        print(f"  目标形状: {targets.shape}")
        
        # 移动到设备
        inputs = inputs.to(trainer.device)
        targets = targets.to(trainer.device)
        
        # 前向传播
        start_time = time.time()
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = trainer.model(inputs)
        
        forward_time = time.time() - start_time
        
        print(f"  输出形状: {outputs.shape}")
        print(f"  前向传播时间: {forward_time:.3f}s")
        
        # 检查内存使用
        test_memory_usage()
        
        print("✅ 单批次测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 单批次测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 AR训练优化配置测试")
    print("=" * 50)
    
    # 测试硬件检测
    test_hardware_detection()
    
    # 测试训练器初始化
    trainer = test_trainer_initialization()
    if trainer is None:
        print("❌ 训练器初始化失败，退出测试")
        return
    
    print()
    
    # 测试内存使用
    test_memory_usage()
    
    # 测试单批次
    success = test_single_batch()
    
    print("=" * 50)
    if success:
        print("🎉 所有测试通过！配置优化成功")
        print("💡 建议:")
        print("  1. 可以开始正式训练")
        print("  2. 监控GPU内存使用率")
        print("  3. 根据需要调整批次大小")
    else:
        print("⚠️ 测试失败，请检查配置")

if __name__ == "__main__":
    main()