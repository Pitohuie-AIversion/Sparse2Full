#!/usr/bin/env python3
"""
分布式训练模块测试

测试utils/distributed.py模块的各项功能：
1. 分布式环境初始化
2. 模型包装和数据加载器创建
3. 通信操作（all_reduce, all_gather, broadcast）
4. 检查点管理和性能统计
5. DistributedManager类功能

Author: PDEBench Team
Date: 2025-01-11
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.distributed import DistributedManager


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TestDistributedUtils:
    """分布式工具函数测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_manager_basic_functionality(self):
        """测试DistributedManager基础功能"""
        manager = DistributedManager()
        
        # 测试初始化
        success = manager.setup()
        assert success == True
        
        # 测试基本属性
        assert manager.world_size >= 1
        assert manager.rank >= 0
        assert manager.device is not None
        
        # 清理
        manager.cleanup()


class TestDistributedManager:
    """DistributedManager类测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('torch.distributed.is_available')
    @patch('torch.cuda.is_available')
    def test_manager_init_single_gpu(self, mock_cuda_available, mock_dist_available):
        """测试单GPU环境下的管理器初始化"""
        mock_cuda_available.return_value = True
        mock_dist_available.return_value = True
        
        with patch.dict(os.environ, {}, clear=True):
            manager = DistributedManager()
            
            assert manager.world_size == 1
            assert manager.rank == 0
            assert manager.local_rank == 0
            assert manager.is_distributed == False
            assert manager.device == torch.device('cuda:0')
    
    @patch('torch.distributed.is_available')
    @patch('torch.cuda.is_available')
    def test_manager_init_multi_gpu(self, mock_cuda_available, mock_dist_available):
        """测试多GPU环境下的管理器初始化"""
        mock_cuda_available.return_value = True
        mock_dist_available.return_value = True
        
        with patch.dict(os.environ, {
            'WORLD_SIZE': '4',
            'RANK': '1',
            'LOCAL_RANK': '1'
        }):
            manager = DistributedManager()
            
            assert manager.world_size == 4
            assert manager.rank == 1
            assert manager.local_rank == 1
            assert manager.is_distributed == True
            assert manager.device == torch.device('cuda:1')
    
    def test_manager_basic_functionality(self):
        """测试DistributedManager基本功能"""
        manager = DistributedManager()
        
        # 测试初始化状态
        assert manager.world_size == 1
        assert manager.rank == 0
        assert manager.local_rank == 0
        assert manager.is_distributed == False
        assert manager.device is None  # 在setup之前设备为None
        
        # 测试setup
        success = manager.setup()
        assert success == True
        assert manager.device is not None
        
        # 清理
        manager.cleanup()
    
    def test_manager_wrap_model(self):
        """测试模型包装"""
        model = SimpleModel()
        manager = DistributedManager()
        manager.setup()
        
        # 包装模型
        wrapped_model = manager.wrap_model(model)
        assert wrapped_model is not None
        
        # 清理
        manager.cleanup()
    
    def test_manager_create_dataloader(self):
        """测试数据加载器创建"""
        # 创建虚拟数据集
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 1)
        )
        
        manager = DistributedManager()
        manager.setup()
        
        # 测试数据加载器创建
        dataloader, sampler = manager.create_dataloader(dataset, batch_size=16, shuffle=True)
        
        assert dataloader.batch_size == 16
        assert len(dataloader.dataset) == 100
        
        # 清理
        manager.cleanup()


class TestDistributedDataLoader:
    """分布式数据加载器测试"""
    
    def test_create_distributed_dataloader_single_gpu(self):
        """测试单GPU环境下的数据加载器创建"""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 1)
        )
        
        with patch('utils.distributed.get_world_size', return_value=1):
            dataloader = create_distributed_dataloader(
                dataset, batch_size=16, shuffle=True, num_workers=2
            )
            
            assert dataloader.batch_size == 16
            assert len(dataloader.dataset) == 100
    
    @patch('torch.utils.data.distributed.DistributedSampler')
    def test_create_distributed_dataloader_multi_gpu(self, mock_sampler):
        """测试多GPU环境下的数据加载器创建"""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 1)
        )
        
        with patch('utils.distributed.get_world_size', return_value=4):
            with patch('utils.distributed.get_rank', return_value=1):
                dataloader = create_distributed_dataloader(
                    dataset, batch_size=16, shuffle=True, num_workers=2
                )
                
                # 验证DistributedSampler被创建
                mock_sampler.assert_called_once_with(
                    dataset, num_replicas=4, rank=1, shuffle=True
                )


def test_distributed_training_simulation():
    """模拟分布式训练流程测试"""
    
    # 创建模型和数据
    model = SimpleModel()
    dataset = torch.utils.data.TensorDataset(
        torch.randn(64, 10),
        torch.randn(64, 1)
    )
    
    # 创建分布式管理器
    manager = DistributedManager()
    
    # 包装模型
    model = manager.wrap_model(model)
    
    # 创建数据加载器
    dataloader = manager.create_dataloader(dataset, batch_size=8, shuffle=True)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练步骤
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= 2:  # 只测试几个batch
            break
            
        data, target = data.to(manager.device), target.to(manager.device)
        
        optimizer.zero_grad()
        
        # 记录前向传播时间
        import time
        start_time = time.time()
        output = model(data)
        forward_time = time.time() - start_time
        manager.log_performance('forward_time', forward_time)
        
        # 计算损失
        loss = nn.MSELoss()(output, target)
        
        # 记录反向传播时间
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        manager.log_performance('backward_time', backward_time)
        
        optimizer.step()
        
        # 同步损失值
        loss_tensor = torch.tensor(loss.item())
        avg_loss = manager.all_reduce(loss_tensor)
        
        print(f"Batch {batch_idx}, Loss: {avg_loss.item():.4f}")
    
    # 获取性能统计
    stats = manager.get_performance_stats()
    print("Performance Stats:", stats)
    
    assert len(stats) > 0
    assert 'forward_time' in stats
    assert 'backward_time' in stats


if __name__ == '__main__':
    print("开始分布式训练模块测试...")
    
    # 运行基础功能测试
    print("\n1. 测试分布式工具函数...")
    test_utils = TestDistributedUtils()
    test_utils.setup_method()
    
    try:
        test_utils.test_world_size_and_rank_single_gpu()
        test_utils.test_world_size_and_rank_multi_gpu()
        print("✓ 分布式工具函数测试通过")
    except Exception as e:
        print(f"✗ 分布式工具函数测试失败: {e}")
    finally:
        test_utils.teardown_method()
    
    # 运行管理器测试
    print("\n2. 测试DistributedManager...")
    test_manager = TestDistributedManager()
    test_manager.setup_method()
    
    try:
        test_manager.test_manager_performance_stats()
        test_manager.test_manager_communication_ops()
        print("✓ DistributedManager测试通过")
    except Exception as e:
        print(f"✗ DistributedManager测试失败: {e}")
    finally:
        test_manager.teardown_method()
    
    # 运行训练模拟测试
    print("\n3. 测试分布式训练模拟...")
    try:
        test_distributed_training_simulation()
        print("✓ 分布式训练模拟测试通过")
    except Exception as e:
        print(f"✗ 分布式训练模拟测试失败: {e}")
    
    print("\n分布式训练模块测试完成！")