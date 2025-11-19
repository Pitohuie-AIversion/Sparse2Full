"""
单元测试：硬件分析器
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

# 导入测试模块
from src.optimizers.hardware_profiler import HardwareProfiler, HardwareConfig

class TestHardwareProfiler(unittest.TestCase):
    """硬件分析器单元测试"""
    
    def setUp(self):
        """测试前设置"""
        self.profiler = HardwareProfiler()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.profiler)
        self.assertIsNotNone(self.profiler.cpu_info)
        self.assertIsNotNone(self.profiler.gpu_info)
        self.assertIsNotNone(self.profiler.memory_info)
    
    def test_cpu_detection(self):
        """测试CPU检测"""
        cpu_info = self.profiler.cpu_info
        
        self.assertIn('count', cpu_info)
        self.assertIn('freq', cpu_info)
        self.assertIn('numa_nodes', cpu_info)
        self.assertIn('cache_size_mb', cpu_info)
        
        # 验证数据类型
        self.assertIsInstance(cpu_info['count'], int)
        self.assertIsInstance(cpu_info['numa_nodes'], int)
        self.assertIsInstance(cpu_info['cache_size_mb'], (int, float))
        
        # 验证数值范围
        self.assertGreater(cpu_info['count'], 0)
        self.assertGreater(cpu_info['numa_nodes'], 0)
        self.assertGreater(cpu_info['cache_size_mb'], 0)
    
    def test_gpu_detection(self):
        """测试GPU检测"""
        gpu_info = self.profiler.gpu_info
        
        self.assertIn('count', gpu_info)
        self.assertIn('memory_gb', gpu_info)
        self.assertIn('cuda_cores', gpu_info)
        self.assertIn('compute_capability', gpu_info)
        
        # 验证数据类型
        self.assertIsInstance(gpu_info['count'], int)
        self.assertIsInstance(gpu_info['memory_gb'], (int, float))
        
        # 如果没有GPU，这些值应该为0
        if gpu_info['count'] == 0:
            self.assertEqual(gpu_info['memory_gb'], 0)
            self.assertEqual(gpu_info['cuda_cores'], 0)
    
    def test_memory_detection(self):
        """测试内存检测"""
        memory_info = self.profiler.memory_info
        
        self.assertIn('total_gb', memory_info)
        self.assertIn('available_gb', memory_info)
        self.assertIn('numa_aware', memory_info)
        
        # 验证数据类型
        self.assertIsInstance(memory_info['total_gb'], (int, float))
        self.assertIsInstance(memory_info['available_gb'], (int, float))
        self.assertIsInstance(memory_info['numa_aware'], bool)
        
        # 验证数值关系
        self.assertGreaterEqual(memory_info['total_gb'], memory_info['available_gb'])
        self.assertGreater(memory_info['total_gb'], 0)
    
    def test_optimal_config_calculation(self):
        """测试最优配置计算"""
        config = self.profiler.get_optimal_config()
        
        self.assertIsInstance(config, HardwareConfig)
        
        # 验证关键配置项
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.num_workers, 0)
        self.assertIn(config.mixed_precision, [True, False])
        self.assertIn(config.compile_model, [True, False])
        self.assertGreaterEqual(config.gradient_accumulation_steps, 1)
        self.assertGreaterEqual(config.numa_optimization_level, 0)
    
    def test_batch_size_calculation(self):
        """测试批大小计算"""
        # 测试不同的内存配置
        test_memory_gb = [4, 8, 16, 32]
        
        for memory in test_memory_gb:
            batch_size = self.profiler._calculate_optimal_batch_size(memory)
            
            self.assertIsInstance(batch_size, int)
            self.assertGreater(batch_size, 0)
            self.assertLessEqual(batch_size, 512)  # 合理的上限
    
    def test_num_workers_calculation(self):
        """测试工作进程数计算"""
        cpu_cores = self.profiler.cpu_info['count']
        numa_nodes = self.profiler.cpu_info['numa_nodes']
        
        num_workers = self.profiler._calculate_optimal_num_workers(cpu_cores, numa_nodes)
        
        self.assertIsInstance(num_workers, int)
        self.assertGreater(num_workers, 0)
        self.assertLessEqual(num_workers, cpu_cores)
    
    def test_hardware_info_string(self):
        """测试硬件信息字符串"""
        info_str = self.profiler.get_hardware_info_string()
        
        self.assertIsInstance(info_str, str)
        self.assertIn("CPU", info_str)
        self.assertIn("GPU", info_str)
        self.assertIn("Memory", info_str)
    
    def test_config_serialization(self):
        """测试配置序列化"""
        config = self.profiler.get_optimal_config()
        
        # 转换为字典
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        
        # 验证关键字段存在
        required_fields = [
            'batch_size', 'num_workers', 'mixed_precision',
            'compile_model', 'gradient_accumulation_steps'
        ]
        
        for field in required_fields:
            self.assertIn(field, config_dict)
        
        # 测试JSON序列化
        config_json = json.dumps(config_dict)
        self.assertIsInstance(config_json, str)
        
        # 测试反序列化
        loaded_dict = json.loads(config_json)
        self.assertEqual(config_dict, loaded_dict)
    
    def test_performance_estimation(self):
        """测试性能估算"""
        config = self.profiler.get_optimal_config()
        
        # 估算训练性能
        estimated_performance = self.profiler.estimate_training_performance(config)
        
        self.assertIsInstance(estimated_performance, dict)
        self.assertIn('samples_per_second', estimated_performance)
        self.assertIn('estimated_epoch_time', estimated_performance)
        self.assertIn('memory_usage_gb', estimated_performance)
        
        # 验证数值合理性
        self.assertGreater(estimated_performance['samples_per_second'], 0)
        self.assertGreater(estimated_performance['estimated_epoch_time'], 0)
        self.assertGreater(estimated_performance['memory_usage_gb'], 0)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效输入
        with self.assertRaises(ValueError):
            self.profiler._calculate_optimal_batch_size(-1)
        
        with self.assertRaises(ValueError):
            self.profiler._calculate_optimal_num_workers(0, 1)
        
        with self.assertRaises(ValueError):
            self.profiler._calculate_optimal_num_workers(8, 0)


class TestHardwareConfig(unittest.TestCase):
    """硬件配置类单元测试"""
    
    def test_default_initialization(self):
        """测试默认初始化"""
        config = HardwareConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.mixed_precision, True)
        self.assertEqual(config.compile_model, False)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.numa_optimization_level, 1)
    
    def test_custom_initialization(self):
        """测试自定义初始化"""
        config = HardwareConfig(
            batch_size=64,
            num_workers=8,
            mixed_precision=False,
            compile_model=True,
            gradient_accumulation_steps=2,
            numa_optimization_level=2
        )
        
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.num_workers, 8)
        self.assertEqual(config.mixed_precision, False)
        self.assertEqual(config.compile_model, True)
        self.assertEqual(config.gradient_accumulation_steps, 2)
        self.assertEqual(config.numa_optimization_level, 2)
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = HardwareConfig(
            batch_size=128,
            num_workers=16
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['batch_size'], 128)
        self.assertEqual(config_dict['num_workers'], 16)
    
    def test_from_dict(self):
        """测试从字典创建"""
        config_dict = {
            'batch_size': 256,
            'num_workers': 32,
            'mixed_precision': True,
            'compile_model': True,
            'gradient_accumulation_steps': 4,
            'numa_optimization_level': 3
        }
        
        config = HardwareConfig.from_dict(config_dict)
        
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.num_workers, 32)
        self.assertEqual(config.mixed_precision, True)
        self.assertEqual(config.compile_model, True)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.numa_optimization_level, 3)
    
    def test_immutability(self):
        """测试配置不可变性"""
        config = HardwareConfig()
        
        # 尝试修改应该失败（如果实现了不可变性）
        # 这里假设配置是可变的，测试基本功能
        config.batch_size = 64
        self.assertEqual(config.batch_size, 64)


if __name__ == '__main__':
    # 设置测试日志
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # 运行测试
    unittest.main(verbosity=2)