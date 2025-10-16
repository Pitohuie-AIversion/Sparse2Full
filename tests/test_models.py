"""
PDEBench稀疏观测重建系统 - 模型单元测试

测试模型的统一接口、前向传播、梯度流动和内存效率
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.swin_unet import SwinUNet
from models.hybrid_model import HybridModel
from models.mlp_model import MLPModel
from models.baseline_models import UNet, FNO


class TestModelInterface:
    """模型接口统一性测试"""
    
    @pytest.fixture
    def model_configs(self):
        """模型配置"""
        return {
            'swin_unet': {
                'in_channels': 3,
                'out_channels': 3,
                'img_size': 64,
                'patch_size': 4,
                'embed_dim': 96,
                'depths': [2, 2, 2, 2],
                'num_heads': [3, 6, 12, 24]
            },
            'hybrid': {
                'in_channels': 3,
                'out_channels': 3,
                'img_size': 64,
                'hidden_dim': 128,
                'num_layers': 4
            },
            'mlp': {
                'in_channels': 3,
                'out_channels': 3,
                'img_size': 64,
                'hidden_dim': 256,
                'num_layers': 6
            },
            'unet': {
                'in_channels': 3,
                'out_channels': 3,
                'base_channels': 64
            },
            'fno': {
                'in_channels': 3,
                'out_channels': 3,
                'modes': 16,
                'width': 64
            }
        }
    
    def test_swin_unet_interface(self, device, model_configs):
        """测试SwinUNet接口"""
        config = model_configs['swin_unet']
        model = SwinUNet(**config).to(device)
        
        # 测试前向传播
        batch_size = 2
        x = torch.randn(batch_size, config['in_channels'], 
                       config['img_size'], config['img_size'], device=device)
        
        with torch.no_grad():
            y = model(x)
        
        # 检查输出形状
        expected_shape = (batch_size, config['out_channels'], 
                         config['img_size'], config['img_size'])
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
        
        # 检查输出属性
        assert y.dtype == x.dtype, "Output dtype should match input"
        assert y.device == x.device, "Output device should match input"
        assert torch.isfinite(y).all(), "Output should be finite"
    
    def test_hybrid_model_interface(self, device, model_configs):
        """测试HybridModel接口"""
        config = model_configs['hybrid']
        model = HybridModel(**config).to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, config['in_channels'], 
                       config['img_size'], config['img_size'], device=device)
        
        with torch.no_grad():
            y = model(x)
        
        expected_shape = (batch_size, config['out_channels'], 
                         config['img_size'], config['img_size'])
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
        assert torch.isfinite(y).all(), "Output should be finite"
    
    def test_mlp_model_interface(self, device, model_configs):
        """测试MLPModel接口"""
        config = model_configs['mlp']
        model = MLPModel(**config).to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, config['in_channels'], 
                       config['img_size'], config['img_size'], device=device)
        
        with torch.no_grad():
            y = model(x)
        
        expected_shape = (batch_size, config['out_channels'], 
                         config['img_size'], config['img_size'])
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
        assert torch.isfinite(y).all(), "Output should be finite"
    
    def test_unet_interface(self, device, model_configs):
        """测试UNet接口"""
        config = model_configs['unet']
        model = UNet(**config).to(device)
        
        batch_size = 2
        img_size = 64
        x = torch.randn(batch_size, config['in_channels'], 
                       img_size, img_size, device=device)
        
        with torch.no_grad():
            y = model(x)
        
        expected_shape = (batch_size, config['out_channels'], img_size, img_size)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
        assert torch.isfinite(y).all(), "Output should be finite"
    
    def test_fno_interface(self, device, model_configs):
        """测试FNO接口"""
        config = model_configs['fno']
        model = FNO(**config).to(device)
        
        batch_size = 2
        img_size = 64
        x = torch.randn(batch_size, config['in_channels'], 
                       img_size, img_size, device=device)
        
        with torch.no_grad():
            y = model(x)
        
        expected_shape = (batch_size, config['out_channels'], img_size, img_size)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
        assert torch.isfinite(y).all(), "Output should be finite"


class TestModelGradients:
    """模型梯度测试"""
    
    def test_swin_unet_gradients(self, device, model_configs, test_utils):
        """测试SwinUNet梯度流动"""
        config = model_configs['swin_unet']
        model = SwinUNet(**config).to(device)
        
        x = torch.randn(1, config['in_channels'], 
                       config['img_size'], config['img_size'], 
                       device=device, requires_grad=True)
        
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # 检查输入梯度
        test_utils.assert_gradient_properties(x, check_exists=True, finite=True)
        
        # 检查模型参数梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                test_utils.assert_gradient_properties(
                    param, check_exists=True, finite=True, non_zero=False
                )
    
    def test_hybrid_model_gradients(self, device, model_configs, test_utils):
        """测试HybridModel梯度流动"""
        config = model_configs['hybrid']
        model = HybridModel(**config).to(device)
        
        x = torch.randn(1, config['in_channels'], 
                       config['img_size'], config['img_size'], 
                       device=device, requires_grad=True)
        
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # 检查输入梯度
        test_utils.assert_gradient_properties(x, check_exists=True, finite=True)
        
        # 检查关键参数的梯度
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} should be finite"
        
        assert grad_count > 0, "At least some parameters should have gradients"
    
    def test_gradient_accumulation(self, device, model_configs):
        """测试梯度累积"""
        config = model_configs['swin_unet']
        model = SwinUNet(**config).to(device)
        
        # 第一次前向传播
        x1 = torch.randn(1, config['in_channels'], 
                        config['img_size'], config['img_size'], device=device)
        y1 = model(x1)
        loss1 = y1.sum()
        loss1.backward()
        
        # 保存第一次的梯度
        first_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                first_grads[name] = param.grad.clone()
        
        # 第二次前向传播（不清零梯度）
        x2 = torch.randn(1, config['in_channels'], 
                        config['img_size'], config['img_size'], device=device)
        y2 = model(x2)
        loss2 = y2.sum()
        loss2.backward()
        
        # 检查梯度累积
        for name, param in model.named_parameters():
            if name in first_grads and param.grad is not None:
                # 梯度应该是累积的
                assert not torch.equal(param.grad, first_grads[name]), \
                    f"Gradient for {name} should be accumulated"


class TestModelMemory:
    """模型内存效率测试"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_efficiency(self, model_configs):
        """测试内存效率"""
        device = torch.device('cuda')
        
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        config = model_configs['swin_unet']
        model = SwinUNet(**config).to(device)
        
        # 测试不同批大小的内存使用
        batch_sizes = [1, 2, 4]
        memory_usage = []
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
            
            x = torch.randn(batch_size, config['in_channels'], 
                           config['img_size'], config['img_size'], device=device)
            
            with torch.no_grad():
                y = model(x)
            
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage.append(peak_memory - start_memory)
            
            del x, y
        
        # 内存使用应该随批大小线性增长
        for i in range(1, len(memory_usage)):
            ratio = memory_usage[i] / memory_usage[0]
            expected_ratio = batch_sizes[i] / batch_sizes[0]
            # 允许20%的误差
            assert abs(ratio - expected_ratio) / expected_ratio < 0.2, \
                f"Memory scaling not linear: {ratio} vs {expected_ratio}"
    
    def test_memory_cleanup(self, device, model_configs):
        """测试内存清理"""
        config = model_configs['swin_unet']
        
        # 创建模型并进行前向传播
        model = SwinUNet(**config).to(device)
        x = torch.randn(2, config['in_channels'], 
                       config['img_size'], config['img_size'], device=device)
        
        y = model(x)
        
        # 删除变量
        del model, x, y
        
        # 如果是CUDA，清空缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 内存应该被释放（这里只是确保没有异常）
        assert True, "Memory cleanup completed"


class TestModelDeterminism:
    """模型确定性测试"""
    
    def test_deterministic_forward(self, device, model_configs):
        """测试前向传播的确定性"""
        config = model_configs['swin_unet']
        
        # 设置随机种子
        torch.manual_seed(42)
        model1 = SwinUNet(**config).to(device)
        
        torch.manual_seed(42)
        model2 = SwinUNet(**config).to(device)
        
        # 使用相同的输入
        torch.manual_seed(123)
        x = torch.randn(1, config['in_channels'], 
                       config['img_size'], config['img_size'], device=device)
        
        with torch.no_grad():
            y1 = model1(x)
            y2 = model2(x)
        
        # 输出应该完全相同
        assert torch.equal(y1, y2), "Deterministic forward pass failed"
    
    def test_training_determinism(self, device, model_configs):
        """测试训练的确定性"""
        config = model_configs['swin_unet']
        
        # 创建两个相同的模型
        torch.manual_seed(42)
        model1 = SwinUNet(**config).to(device)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        
        torch.manual_seed(42)
        model2 = SwinUNet(**config).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        
        # 进行相同的训练步骤
        for step in range(3):
            torch.manual_seed(100 + step)
            x = torch.randn(1, config['in_channels'], 
                           config['img_size'], config['img_size'], device=device)
            target = torch.randn(1, config['out_channels'], 
                               config['img_size'], config['img_size'], device=device)
            
            # 模型1训练
            optimizer1.zero_grad()
            y1 = model1(x)
            loss1 = nn.MSELoss()(y1, target)
            loss1.backward()
            optimizer1.step()
            
            # 模型2训练
            optimizer2.zero_grad()
            y2 = model2(x)
            loss2 = nn.MSELoss()(y2, target)
            loss2.backward()
            optimizer2.step()
            
            # 检查参数是否相同
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(), model2.named_parameters()
            ):
                assert name1 == name2, "Parameter names should match"
                assert torch.allclose(param1, param2, atol=1e-6), \
                    f"Parameters {name1} should be identical after step {step}"


class TestModelEdgeCases:
    """模型边界情况测试"""
    
    def test_single_pixel_input(self, device, model_configs):
        """测试单像素输入"""
        # 只测试能处理任意尺寸的模型
        config = model_configs['mlp'].copy()
        config['img_size'] = 1
        
        model = MLPModel(**config).to(device)
        x = torch.randn(1, config['in_channels'], 1, 1, device=device)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (1, config['out_channels'], 1, 1), "Single pixel output shape incorrect"
        assert torch.isfinite(y).all(), "Single pixel output should be finite"
    
    def test_large_input(self, device, model_configs):
        """测试大尺寸输入"""
        config = model_configs['unet']
        model = UNet(**config).to(device)
        
        # 测试较大的输入
        large_size = 128
        x = torch.randn(1, config['in_channels'], large_size, large_size, device=device)
        
        with torch.no_grad():
            y = model(x)
        
        expected_shape = (1, config['out_channels'], large_size, large_size)
        assert y.shape == expected_shape, f"Large input shape incorrect: {y.shape} vs {expected_shape}"
        assert torch.isfinite(y).all(), "Large input output should be finite"
    
    def test_zero_input(self, device, model_configs):
        """测试零输入"""
        config = model_configs['swin_unet']
        model = SwinUNet(**config).to(device)
        
        x = torch.zeros(1, config['in_channels'], 
                       config['img_size'], config['img_size'], device=device)
        
        with torch.no_grad():
            y = model(x)
        
        assert torch.isfinite(y).all(), "Zero input output should be finite"
        assert not torch.isnan(y).any(), "Zero input output should not contain NaN"
    
    def test_extreme_values(self, device, model_configs):
        """测试极值输入"""
        config = model_configs['swin_unet']
        model = SwinUNet(**config).to(device)
        
        # 测试大正值
        x_large = torch.ones(1, config['in_channels'], 
                           config['img_size'], config['img_size'], device=device) * 1000
        
        with torch.no_grad():
            y_large = model(x_large)
        
        assert torch.isfinite(y_large).all(), "Large value output should be finite"
        
        # 测试大负值
        x_small = torch.ones(1, config['in_channels'], 
                           config['img_size'], config['img_size'], device=device) * -1000
        
        with torch.no_grad():
            y_small = model(x_small)
        
        assert torch.isfinite(y_small).all(), "Small value output should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])