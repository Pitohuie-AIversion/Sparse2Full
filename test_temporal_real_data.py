#!/usr/bin/env python3
"""
基于真实数据的时序训练测试脚本
"""

import os
import sys
import logging
import torch
from pathlib import Path
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入训练器
try:
    from train_temporal import TemporalTrainer
except ImportError:
    # 如果找不到模块，尝试从当前目录导入
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_temporal", 
        "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/train_temporal.py"
    )
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    TemporalTrainer = train_module.TemporalTrainer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_config():
    """加载真实配置文件并适配TemporalTrainer期望的格式"""
    config_path = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/ar_training_config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = OmegaConf.load(config_path)
    logger.info(f"✅ 配置文件加载成功: {config_path}")
    
    # 适配配置结构以匹配TemporalTrainer的期望
    adapted_config = OmegaConf.create({
        'experiment': {
            'name': config.experiment.name,
            'device': config.experiment.device,
            'seed': config.experiment.seed,
            'output_dir': config.experiment.output_dir,
            'use_amp': config.training.amp.enabled,  # 适配AMP配置
        },
        'data': config.data,  # 数据配置保持不变
        'data_path': config.data.data_path,  # 添加顶级data_path
        'keys': ["data"],  # 使用实际的数据键
        'task': config.data.observation.mode,  # 从观测模式获取任务类型
        'model': config.model,  # 模型配置保持不变
        'train': {  # 将training重命名为train
            'optimizer': config.training.optimizer,
            'scheduler': config.training.scheduler,
            'epochs': config.training.epochs,
        },
        'loss': config.loss,  # 损失配置保持不变
        'curriculum': config.training.curriculum,  # 课程学习配置
        'temporal': {  # 添加temporal配置
            'T_in': config.data.T_in,  # 从data配置中获取
            'T_out': config.data.T_out,  # 从data配置中获取
            'dt': 0.1,  # 默认时间步长
            'ar': config.model.ar_config if 'ar_config' in config.model else {}
        },
        'use_official_format': True,  # 对于diff-react数据集，使用官方格式
    })
    
    # 添加任务特定配置
    if adapted_config.task == 'SR':
        adapted_config.scale = config.data.observation.sr.scale_factor
        adapted_config.sigma = config.data.observation.sr.blur_sigma
        adapted_config.blur_kernel = config.data.observation.sr.blur_kernel_size
        adapted_config.boundary = config.data.observation.sr.boundary_mode
    
    logger.info("✅ 配置适配完成")
    return adapted_config

def test_data_path_exists(config):
    """检查数据路径是否存在"""
    data_path = config.data.data_path
    exists = os.path.exists(data_path)
    logger.info(f"数据路径检查: {data_path} - {'存在' if exists else '不存在'}")
    return exists

def test_model_creation():
    """测试模型创建"""
    logger.info("🔧 测试模型创建...")
    
    try:
        config = load_real_config()
        trainer = TemporalTrainer(config)
        
        assert trainer.model is not None
        logger.info(f"✅ 模型创建成功: {type(trainer.model).__name__}")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in trainer.model.parameters())
        logger.info(f"📊 模型参数数量: {total_params:,}")
        return True
    except Exception as e:
        logger.error(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    logger.info("📊 测试数据加载...")
    
    try:
        config = load_real_config()
        
        # 检查数据路径
        if not test_data_path_exists(config):
            logger.warning("⚠️ 数据文件不存在，跳过测试")
            return False
        
        trainer = TemporalTrainer(config)
        
        # 测试训练数据加载
        train_batch = next(iter(trainer.train_loader))
        logger.info(f"✅ 训练批次键: {list(train_batch.keys())}")
        if 'input_sequence' in train_batch:
            logger.info(f"✅ 输入序列形状: {train_batch['input_sequence'].shape}")
        if 'target_sequence' in train_batch:
            logger.info(f"✅ 目标序列形状: {train_batch['target_sequence'].shape}")
        
        # 测试验证数据加载
        val_batch = next(iter(trainer.val_loader))
        logger.info(f"✅ 验证批次键: {list(val_batch.keys())}")
        return True
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """测试前向传播"""
    logger.info("🚀 测试前向传播...")
    
    try:
        config = load_real_config()
        
        # 检查数据路径
        if not test_data_path_exists(config):
            logger.warning("⚠️ 数据文件不存在，跳过测试")
            return False
        
        trainer = TemporalTrainer(config)
        
        # 获取一个批次的数据
        batch = next(iter(trainer.train_loader))
        
        # 从批次中提取输入和目标
        input_seq = batch['input_sequence']  # [B, T_in, C, H, W]
        target_seq = batch['target_sequence']  # [B, T_out, C, H, W]
        
        # 处理输入：使用最后一帧作为模型输入
        if len(input_seq.shape) == 5:  # [B, T, C, H, W]
            model_input = input_seq[:, -1]  # [B, C, H, W]
        else:
            model_input = input_seq
        
        # 移动数据到设备
        device = next(trainer.model.parameters()).device
        model_input = model_input.to(device)
        target_seq = target_seq.to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs = trainer.model(model_input)
        
        logger.info(f"✅ 输入序列形状: {input_seq.shape}")
        logger.info(f"✅ 模型输入形状: {model_input.shape}")
        logger.info(f"✅ 输出形状: {outputs.shape}")
        logger.info(f"✅ 目标序列形状: {target_seq.shape}")
        
        # 检查输出维度是否合理
        assert len(outputs.shape) >= 3, f"输出维度不足: {outputs.shape}"
        logger.info("✅ 前向传播测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_short_training():
    """测试短期训练"""
    logger.info("🏃 测试短期训练...")
    
    try:
        config = load_real_config()
        
        # 检查数据路径
        if not test_data_path_exists(config):
            logger.warning("⚠️ 数据文件不存在，跳过测试")
            return False
        
        # 减少训练轮数用于测试
        config.train.epochs = 1
        
        trainer = TemporalTrainer(config)
        
        # 运行几个训练步骤
        trainer.model.train()
        for i, batch in enumerate(trainer.train_loader):
            if i >= 3:  # 只测试3个批次
                break
            
            # 从批次中提取输入和目标
            input_seq = batch['input_sequence']  # [B, T_in, C, H, W]
            target_seq = batch['target_sequence']  # [B, T_out, C, H, W]
            
            # 处理输入：使用最后一帧作为模型输入
            if len(input_seq.shape) == 5:  # [B, T, C, H, W]
                model_input = input_seq[:, -1]  # [B, C, H, W]
            else:
                model_input = input_seq
            
            # 移动数据到设备并启用梯度
            device = next(trainer.model.parameters()).device
            model_input = model_input.to(device).requires_grad_(True)
            target_seq = target_seq.to(device)
            
            # 前向传播
            outputs = trainer.model(model_input)
            
            # 计算简单的MSE损失用于测试
            if len(outputs.shape) == 4 and len(target_seq.shape) == 5:
                # 如果输出是4D，目标是5D，使用第一个时间步作为目标
                target_for_loss = target_seq[:, 0]  # [B, C, H, W]
            else:
                target_for_loss = target_seq
            
            # 确保输出和目标形状匹配
            if outputs.shape != target_for_loss.shape:
                logger.warning(f"形状不匹配: outputs {outputs.shape} vs target {target_for_loss.shape}")
                # 尝试调整目标形状
                if len(target_for_loss.shape) > len(outputs.shape):
                    target_for_loss = target_for_loss.squeeze()
                elif len(target_for_loss.shape) < len(outputs.shape):
                    target_for_loss = target_for_loss.unsqueeze(1)
            
            loss = torch.nn.functional.mse_loss(outputs, target_for_loss)
            
            # 反向传播
            trainer.optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                trainer.optimizer.step()
            else:
                logger.warning("损失不需要梯度，跳过反向传播")
            
            logger.info(f"✅ 批次 {i+1}: 损失 = {loss.item():.6f}")
        
        logger.info("✅ 短期训练测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 短期训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始基于真实数据的时序训练测试...")
    
    tests = [
        ("模型创建", test_model_creation),
        ("数据加载", test_data_loading),
        ("前向传播", test_forward_pass),
        ("短期训练", test_short_training),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        result = test_func()
        results.append((test_name, result))
        
        if result:
            logger.info(f"✅ {test_name} 测试通过")
        else:
            logger.error(f"❌ {test_name} 测试失败")
    
    # 总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！")
    else:
        logger.warning(f"⚠️ {total - passed} 个测试失败")

if __name__ == "__main__":
    main()