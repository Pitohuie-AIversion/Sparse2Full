"""
PDEBench稀疏观测重建系统 - 基础功能测试

测试基本的导入和核心功能
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestBasicImports:
    """基础导入测试"""
    
    def test_torch_import(self):
        """测试PyTorch导入"""
        assert torch.__version__ is not None
        print(f"PyTorch version: {torch.__version__}")
    
    def test_numpy_import(self):
        """测试NumPy导入"""
        assert np.__version__ is not None
        print(f"NumPy version: {np.__version__}")
    
    def test_device_availability(self):
        """测试设备可用性"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 测试基本tensor操作
        x = torch.randn(2, 3, device=device)
        y = x + 1
        assert y.device.type == device.type  # 只比较设备类型，不比较索引
        assert torch.allclose(y, x + 1)


class TestBasicTensorOps:
    """基础tensor操作测试"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_tensor_creation(self, device):
        """测试tensor创建"""
        # 创建不同类型的tensor
        x1 = torch.randn(2, 3, 4, 4, device=device)
        x2 = torch.zeros(2, 3, 4, 4, device=device)
        x3 = torch.ones(2, 3, 4, 4, device=device)
        
        assert x1.shape == (2, 3, 4, 4)
        assert x2.shape == (2, 3, 4, 4)
        assert x3.shape == (2, 3, 4, 4)
        
        assert x1.device.type == device.type
        assert x2.device.type == device.type
        assert x3.device.type == device.type
    
    def test_tensor_operations(self, device):
        """测试tensor基本操作"""
        x = torch.randn(2, 3, 4, 4, device=device)
        y = torch.randn(2, 3, 4, 4, device=device)
        
        # 基本运算
        z1 = x + y
        z2 = x * y
        # 修正矩阵乘法维度
        x_flat = x.view(2, 3, -1)  # [2, 3, 16]
        y_flat = y.view(2, 3, -1)  # [2, 3, 16]
        z3 = torch.bmm(x_flat, y_flat.transpose(-2, -1))  # [2, 3, 16] x [2, 3, 16] -> [2, 3, 3]
        
        assert z1.shape == x.shape
        assert z2.shape == x.shape
        assert z3.shape == (2, 3, 3)  # 修正期望形状
        
        # 检查数值稳定性
        assert torch.isfinite(z1).all()
        assert torch.isfinite(z2).all()
        assert torch.isfinite(z3).all()
    
    def test_fft_operations(self, device):
        """测试FFT操作"""
        x = torch.randn(2, 3, 32, 32, device=device)
        
        # 2D FFT
        x_fft = torch.fft.fft2(x)
        x_ifft = torch.fft.ifft2(x_fft).real
        
        # 检查逆变换精度
        assert torch.allclose(x, x_ifft, atol=1e-6)
        
        # 检查频域属性
        assert x_fft.dtype == torch.complex64 or x_fft.dtype == torch.complex128
        assert x_fft.shape == x.shape


class TestBasicMetrics:
    """基础指标测试"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_mse_calculation(self, device):
        """测试MSE计算"""
        pred = torch.randn(2, 3, 32, 32, device=device)
        target = torch.randn(2, 3, 32, 32, device=device)
        
        # 手动计算MSE
        mse = torch.mean((pred - target) ** 2)
        
        # 使用PyTorch函数
        mse_torch = torch.nn.functional.mse_loss(pred, target)
        
        assert torch.allclose(mse, mse_torch, atol=1e-6)
        assert mse >= 0
        assert torch.isfinite(mse)
    
    def test_l2_norm_calculation(self, device):
        """测试L2范数计算"""
        x = torch.randn(2, 3, 32, 32, device=device)
        
        # 不同维度的L2范数
        norm_all = torch.norm(x)
        norm_spatial = torch.norm(x, dim=(-2, -1))
        norm_channel = torch.norm(x, dim=1)
        
        assert norm_all.dim() == 0  # 标量
        assert norm_spatial.shape == (2, 3)
        assert norm_channel.shape == (2, 32, 32)
        
        assert norm_all >= 0
        assert (norm_spatial >= 0).all()
        assert (norm_channel >= 0).all()
    
    def test_relative_error_calculation(self, device):
        """测试相对误差计算"""
        target = torch.randn(2, 3, 32, 32, device=device) + 1  # 避免零
        pred = target + torch.randn_like(target) * 0.1  # 添加小噪声
        
        # 相对L2误差
        diff = pred - target
        rel_error = torch.norm(diff) / torch.norm(target)
        
        assert rel_error >= 0
        assert torch.isfinite(rel_error)
        assert rel_error < 1.0  # 由于噪声较小，相对误差应该小于1


class TestBasicGradients:
    """基础梯度测试"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_gradient_computation(self, device):
        """测试梯度计算"""
        x = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
        
        # 简单的损失函数
        loss = torch.sum(x ** 2)
        loss.backward()
        
        # 检查梯度
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()
        
        # 梯度应该是 2*x
        expected_grad = 2 * x
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)
    
    def test_gradient_flow(self, device):
        """测试梯度流动"""
        # 创建简单的线性层
        linear = torch.nn.Linear(4, 2).to(device)
        x = torch.randn(3, 4, device=device)
        
        # 前向传播
        y = linear(x)
        loss = torch.sum(y ** 2)
        
        # 反向传播
        loss.backward()
        
        # 检查参数梯度
        for param in linear.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            assert param.grad.shape == param.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])