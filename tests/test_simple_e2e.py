#!/usr/bin/env python3
"""
简化的端到端测试脚本
测试核心功能的基本工作流程
"""

import os
import sys
import tempfile
import shutil
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_basic_imports():
    """测试基本模块导入"""
    logger.info("Testing basic imports...")
    
    try:
        from models.swin_unet import SwinUNet
        from losses.combined_loss import CombinedLoss
        from utils.metrics import compute_metrics
        from ops.degradation import GaussianBlurDownsample, CenterCrop
        logger.info("✓ Basic imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    logger.info("Testing model creation...")
    
    try:
        from models.swin_unet import SwinUNet
        
        model = SwinUNet(
            in_channels=2,
            out_channels=2,
            img_size=[64, 64],
            patch_size=4,
            window_size=8,
            depths=[2, 2],
            num_heads=[2, 4],
            embed_dim=48
        )
        
        # 测试前向传播
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (1, 2, 64, 64)
        logger.info("✓ Model creation and forward pass successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False


def test_loss_computation():
    """测试损失函数"""
    logger.info("Testing loss computation...")
    
    try:
        from losses.combined_loss import CombinedLoss
        
        loss_fn = CombinedLoss(
            rec_weight=1.0,
            spec_weight=0.5,
            dc_weight=1.0
        )
        
        pred = torch.randn(1, 2, 64, 64)
        target = torch.randn(1, 2, 64, 64)
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        
        logger.info("✓ Loss computation successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Loss computation failed: {e}")
        return False


def test_metrics_computation():
    """测试指标计算"""
    logger.info("Testing metrics computation...")
    
    try:
        from utils.metrics import compute_metrics
        
        pred = torch.randn(1, 2, 64, 64)
        target = torch.randn(1, 2, 64, 64)
        
        metrics = compute_metrics(pred, target)
        
        assert isinstance(metrics, dict)
        assert 'rel_l2' in metrics
        assert 'mae' in metrics
        
        logger.info("✓ Metrics computation successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Metrics computation failed: {e}")
        return False


def test_degradation_operators():
    """测试降质算子"""
    logger.info("Testing degradation operators...")
    
    try:
        from ops.degradation import GaussianBlurDownsample, CenterCrop
        
        # 测试超分辨率降质
        sr_op = GaussianBlurDownsample(scale_factor=2, sigma=1.0, kernel_size=5)
        x = torch.randn(1, 2, 64, 64)
        y_sr = sr_op(x)
        
        assert y_sr.shape == (1, 2, 32, 32)
        
        # 测试裁剪降质
        crop_op = CenterCrop(crop_size=(32, 32), img_size=(64, 64))
        y_crop = crop_op(x)
        
        assert y_crop.shape == (1, 2, 32, 32)
        
        logger.info("✓ Degradation operators successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Degradation operators failed: {e}")
        return False


def test_training_step():
    """测试训练步骤"""
    logger.info("Testing training step...")
    
    try:
        from models.swin_unet import SwinUNet
        from losses.combined_loss import CombinedLoss
        
        # 创建模型和损失函数
        model = SwinUNet(
            in_channels=2,
            out_channels=2,
            img_size=[64, 64],
            patch_size=4,
            window_size=8,
            depths=[2, 2],
            num_heads=[2, 4],
            embed_dim=48
        )
        
        loss_fn = CombinedLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # 模拟训练步骤
        model.train()
        optimizer.zero_grad()
        
        x = torch.randn(2, 2, 64, 64)
        target = torch.randn(2, 2, 64, 64)
        
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        
        # 检查梯度
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters())
        assert has_grad, "No gradients found"
        
        optimizer.step()
        
        logger.info("✓ Training step successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Training step failed: {e}")
        return False


def test_visualization():
    """测试可视化功能"""
    logger.info("Testing visualization...")
    
    try:
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            from utils.visualization import PDEBenchVisualizer
            
            visualizer = PDEBenchVisualizer(save_dir=str(temp_dir))
            
            # 创建测试数据
            import numpy as np
            gt = np.random.rand(64, 64)
            pred = np.random.rand(64, 64)
            
            # 测试基本可视化
            visualizer.plot_field_comparison(gt, pred, save_name="test")
            
            # 检查文件是否生成
            save_path = temp_dir / "test.png"
            assert save_path.exists()
            
            logger.info("✓ Visualization successful")
            return True
            
        except Exception as e:
            logger.warning(f"Visualization test failed: {e}")
            return False
            
        finally:
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        logger.error(f"✗ Visualization failed: {e}")
        return False


def test_checkpoint_operations():
    """测试检查点操作"""
    logger.info("Testing checkpoint operations...")
    
    try:
        from models.swin_unet import SwinUNet
        
        model = SwinUNet(
            in_channels=2,
            out_channels=2,
            img_size=[64, 64],
            patch_size=4,
            window_size=8,
            depths=[2, 2],
            num_heads=[2, 4],
            embed_dim=48
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # 创建临时文件
        temp_dir = Path(tempfile.mkdtemp())
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        
        try:
            # 保存检查点
            checkpoint = {
                'epoch': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0.5
            }
            
            torch.save(checkpoint, checkpoint_path)
            assert checkpoint_path.exists()
            
            # 加载检查点
            loaded = torch.load(checkpoint_path, map_location='cpu')
            
            assert 'epoch' in loaded
            assert 'model_state_dict' in loaded
            assert 'optimizer_state_dict' in loaded
            
            # 加载状态
            model.load_state_dict(loaded['model_state_dict'])
            optimizer.load_state_dict(loaded['optimizer_state_dict'])
            
            logger.info("✓ Checkpoint operations successful")
            return True
            
        finally:
            # 清理
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        logger.error(f"✗ Checkpoint operations failed: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("Running Simple E2E Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Creation", test_model_creation),
        ("Loss Computation", test_loss_computation),
        ("Metrics Computation", test_metrics_computation),
        ("Degradation Operators", test_degradation_operators),
        ("Training Step", test_training_step),
        ("Visualization", test_visualization),
        ("Checkpoint Operations", test_checkpoint_operations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n[{passed + 1}/{total}] {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"Test '{test_name}' failed")
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
    
    logger.info("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)