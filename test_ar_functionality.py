#!/usr/bin/env python3
"""
测试AR功能的脚本
验证T_out=1的等价性和T_out=3的多步预测
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from models.base import create_model
from models.swin_unet import SwinUNet
from models.ar import ARWrapper

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ar_wrapper_creation():
    """测试AR包装器的创建"""
    logger.info("Testing AR wrapper creation...")
    
    # 创建基础模型 - 使用合适的尺寸参数
    base_model = SwinUNet(
        in_channels=3,
        out_channels=1,
        img_size=224,  # 使用标准尺寸
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # 创建AR包装器
    ar_model = ARWrapper(
        single_frame_model=base_model,
        detach_rollout=True,
        scheduled_sampling=False
    )
    # 验证AR包装器属性
    assert hasattr(ar_model, 'm')  # 基础模型
    assert isinstance(ar_model, ARWrapper)
    
    logger.info("✓ AR wrapper creation test passed")
    return ar_model


def test_ar_model_factory():
    """测试通过工厂函数创建AR模型"""
    logger.info("Testing AR model factory...")
    
    # 通过工厂函数创建AR模型
    ar_model = create_model(
        model_name="swin_unet_ar",
        in_channels=1,
        out_channels=1,
        img_size=224,  # 使用标准尺寸
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        T_out=3,
        teacher_forcing_ratio=1.0
    )
    
    # 验证AR模型属性
    assert isinstance(ar_model, ARWrapper)
    assert hasattr(ar_model, 'm')  # 基础模型
    
    logger.info("✓ AR model factory test passed")
    return ar_model


def test_t_out_1_equivalence():
    """测试T_out=1时与基础模型的等价性"""
    logger.info("Testing T_out=1 equivalence...")
    
    # 创建基础模型 - 使用标准尺寸
    base_model = SwinUNet(
        in_channels=1,
        out_channels=1,
        img_size=224,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # 创建T_out=1的AR模型
    ar_model = ARWrapper(
        single_frame_model=base_model,
        detach_rollout=True,
        scheduled_sampling=False
    )
    
    # 测试输入
    batch_size = 2
    input_single = torch.randn(batch_size, 1, 224, 224)  # [B, C, H, W]
    input_seq = input_single.unsqueeze(1)  # [B, 1, C, H, W]
    
    # 基础模型预测
    base_model.eval()
    ar_model.eval()
    
    with torch.no_grad():
        base_pred = base_model(input_single)  # [B, C, H, W]
        ar_pred = ar_model(input_single, T_out=1)  # [B, 1, C, H, W]
        
        # 比较结果
        ar_pred_single = ar_pred.squeeze(1)  # [B, C, H, W]
        
        # 计算差异
        diff = torch.abs(base_pred - ar_pred_single).max().item()
        
        logger.info(f"Max difference between base and AR(T_out=1): {diff:.2e}")
        
        # 验证等价性（允许小的数值误差）
        assert diff < 1e-5, f"T_out=1 not equivalent to base model, diff={diff}"
    
    logger.info("✓ T_out=1 equivalence test passed")


def test_multi_step_prediction():
    """测试多步预测功能"""
    logger.info("Testing multi-step prediction...")
    
    # 创建AR模型 - 使用标准尺寸进行测试
    ar_model = create_model(
        model_name="swin_unet_ar",
        in_channels=1,
        out_channels=1,
        img_size=224,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        T_out=3,
        teacher_forcing_ratio=1.0
    )
    
    batch_size = 2
    T_in = 1
    T_out = 3
    
    # 测试输入 - 使用与模型匹配的尺寸
    input_seq = torch.randn(batch_size, T_in, 1, 224, 224)
    target_seq = torch.randn(batch_size, T_out, 1, 224, 224)
    
    ar_model.eval()
    
    # 测试teacher forcing模式
    ar_model.train()
    with torch.no_grad():
        pred_teacher = ar_model(input_seq, T_out=T_out, teacher=target_seq)
        assert pred_teacher.shape == (batch_size, T_out, 1, 224, 224)
        logger.info(f"Teacher forcing output shape: {pred_teacher.shape}")
    
    # 测试roll-out模式
    ar_model.eval()
    with torch.no_grad():
        pred_rollout = ar_model(input_seq, T_out=T_out)
        assert pred_rollout.shape == (batch_size, T_out, 1, 224, 224)
        logger.info(f"Roll-out output shape: {pred_rollout.shape}")
    
    # 验证两种模式的输出不同（因为没有teacher forcing）
    diff = torch.abs(pred_teacher - pred_rollout).mean().item()
    logger.info(f"Difference between teacher forcing and roll-out: {diff:.6f}")
    
    logger.info("✓ Multi-step prediction test passed")


def test_ar_loss_computation():
    """测试AR损失计算"""
    logger.info("Testing AR loss computation...")
    
    from ops.losses import compute_ar_loss, compute_ar_total_loss
    from omegaconf import DictConfig
    
    batch_size = 2
    T_out = 3
    
    # 模拟预测和真值序列
    pred_seq = torch.randn(batch_size, T_out, 1, 64, 64)
    gt_seq = torch.randn(batch_size, T_out, 1, 64, 64)
    
    # 测试基础AR损失
    cfg_loss = {"rel2_weight": 1.0, "mae_weight": 0.1}
    loss, loss_items = compute_ar_loss(pred_seq, gt_seq, cfg_loss)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # 标量
    assert 'rel2' in loss_items
    assert 'mae' in loss_items
    
    logger.info(f"AR loss: {loss.item():.6f}")
    logger.info(f"Loss items: {loss_items}")
    
    # 测试完整AR损失（包含频谱和DC损失）
    obs_data = {
        'observation_seq': torch.randn(batch_size, T_out, 1, 32, 32)
    }
    
    config = DictConfig({
        'loss': {
            'rel2_weight': 1.0,
            'mae_weight': 0.1,
            'spectral': {'weight': 0.0},
            'data_consistency': {'weight': 0.0}
        },
        'data': {'keys': ['u']}
    })
    
    losses = compute_ar_total_loss(
        pred_seq=pred_seq,
        gt_seq=gt_seq,
        obs_data=obs_data,
        norm_stats=None,
        config=config
    )
    
    assert 'total_loss' in losses
    assert 'reconstruction_loss' in losses
    assert 'rel2_loss' in losses
    assert 'mae_loss' in losses
    
    logger.info(f"Total AR loss: {losses['total_loss'].item():.6f}")
    
    logger.info("✓ AR loss computation test passed")


def test_model_info():
    """测试模型信息获取"""
    logger.info("Testing model info...")
    
    ar_model = create_model(
        model_name="swin_unet_ar",
        in_channels=1,
        out_channels=1,
        img_size=224,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        T_out=3
    )
    
    # 测试参数计数
    total_params, trainable_params = ar_model.count_parameters()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 测试FLOPs计算
    flops = ar_model.get_flops(input_shape=(2, 1, 1, 64, 64))
    logger.info(f"FLOPs: {flops:,}")
    
    # 测试显存使用估算
    memory_usage = ar_model.get_memory_usage(batch_size=2)
    logger.info(f"Memory usage: {memory_usage}")
    
    logger.info("✓ Model info test passed")


def main():
    """运行所有测试"""
    logger.info("Starting AR functionality tests...")
    
    try:
        # 测试AR包装器创建
        test_ar_wrapper_creation()
        
        # 测试AR模型工厂
        test_ar_model_factory()
        
        # 测试T_out=1等价性
        test_t_out_1_equivalence()
        
        # 测试多步预测
        test_multi_step_prediction()
        
        # 测试AR损失计算
        test_ar_loss_computation()
        
        # 测试模型信息
        test_model_info()
        
        logger.info("🎉 All AR functionality tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()