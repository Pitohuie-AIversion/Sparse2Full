#!/usr/bin/env python3
"""
真实扩散-反应数据AR训练脚本测试套件
测试重构版本的功能、性能和一致性
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional
import warnings

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# 导入要测试的模块
from tools.training.train_real_data_ar_refactored import (
    ConfigManager, DeviceManager, LogManager, DataManager, ModelManager,
    OptimizerManager, LossManager, CurriculumManager, CheckpointManager,
    RealDataARTrainer, convert_numpy_types, seed_worker_fn
)

class TestConfigManager(unittest.TestCase):
    """测试配置管理器"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = Path(self.temp_dir) / "test_config.yaml"
        
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_config_from_file(self):
        """测试从文件加载配置"""
        # 创建测试配置文件
        test_config = {
            'experiment': {'name': 'test_exp', 'seed': 42},
            'training': {'epochs': 10}
        }
        OmegaConf.save(test_config, self.test_config_path)
        
        # 加载配置
        config = ConfigManager.load_config(str(self.test_config_path))
        
        self.assertIsInstance(config, DictConfig)
        self.assertEqual(config.experiment.name, 'test_exp')
        self.assertEqual(config.experiment.seed, 42)
        self.assertEqual(config.training.epochs, 10)
    
    def test_load_default_config(self):
        """测试加载默认配置"""
        config = ConfigManager.load_config()
        
        self.assertIsInstance(config, DictConfig)
        self.assertIn('experiment', config)
        self.assertIn('training', config)
        self.assertIn('data', config)
        self.assertIn('model', config)
    
    def test_validate_config_dataloader_params(self):
        """测试DataLoader参数验证"""
        config = DictConfig({
            'data': {
                'dataloader': {
                    'num_workers': 0,
                    'prefetch_factor': 2,
                    'persistent_workers': True
                }
            }
        })
        
        validated_config = ConfigManager.validate_config(config)
        
        # 当num_workers为0时，应该禁用prefetch和persistent_workers
        self.assertIsNone(validated_config.data.dataloader.prefetch_factor)
        self.assertFalse(validated_config.data.dataloader.persistent_workers)
    
    def test_validate_config_amp_precision(self):
        """测试AMP精度设置验证"""
        config = DictConfig({
            'experiment': {'precision': 'auto'},
            'hardware': {'allow_tf32': True}
        })
        
        validated_config = ConfigManager.validate_config(config)
        
        # 应该根据硬件自动设置精度
        self.assertIn(validated_config.experiment.precision, ['16-mixed', 'bf16-mixed', '32'])
    
    def test_validate_config_observation_params(self):
        """测试观测算子参数验证"""
        config = DictConfig({
            'observation': {
                'kernel_size': 4,  # 偶数，应该被修正
                'blur_sigma': -0.5,  # 负数，应该被修正
                'downsample_interpolation': 'invalid'  # 无效值，应该被修正
            }
        })
        
        validated_config = ConfigManager.validate_config(config)
        
        # 验证参数修正
        self.assertEqual(validated_config.observation.kernel_size, 5)  # 4->5
        self.assertEqual(validated_config.observation.blur_sigma, 0.0)  # -0.5->0.0
        self.assertEqual(validated_config.observation.downsample_interpolation, 'area')  # invalid->area
    
    def test_validate_config_early_stopping(self):
        """测试早停参数验证"""
        config = DictConfig({
            'training': {
                'early_stopping': {
                    'enabled': True,
                    'patience': 10  # 小于20，应该被修正
                }
            }
        })
        
        validated_config = ConfigManager.validate_config(config)
        
        # patience应该被修正为最小值20
        self.assertEqual(validated_config.training.early_stopping.patience, 20)
    
    def test_validate_config_checkpoint(self):
        """测试检查点参数验证"""
        config = DictConfig({
            'training': {
                'checkpoint': {
                    'max_keep': 1,  # 小于2，应该被修正
                    'save_every_n_epochs': -1  # 负数，应该被修正
                }
            }
        })
        
        validated_config = ConfigManager.validate_config(config)
        
        # 验证参数修正
        self.assertEqual(validated_config.training.checkpoint.max_keep, 2)
        self.assertEqual(validated_config.training.checkpoint.save_every_n_epochs, 0)

class TestDeviceManager(unittest.TestCase):
    """测试设备管理器"""
    
    def setUp(self):
        """测试前设置"""
        self.config = DictConfig({
            'experiment': {'device': 'cuda'},
            'hardware': {'allow_tf32': True}
        })
    
    def test_setup_device_cpu(self):
        """测试CPU设备设置"""
        config = DictConfig({
            'experiment': {'device': 'cpu'},
            'hardware': {'allow_tf32': False}
        })
        device_manager = DeviceManager(config)
        device = device_manager.setup_device()
        
        self.assertEqual(device.type, 'cpu')
        self.assertFalse(device_manager.distributed)
        self.assertTrue(device_manager.is_primary)
    
    @patch('torch.cuda.is_available')
    def test_setup_device_cuda(self, mock_cuda_available):
        """测试CUDA设备设置"""
        mock_cuda_available.return_value = True
        
        device_manager = DeviceManager(self.config)
        device = device_manager.setup_device()
        
        self.assertEqual(device.type, 'cuda')
        self.assertFalse(device_manager.distributed)
        self.assertTrue(device_manager.is_primary)
    
    @patch.dict(os.environ, {'RANK': '1', 'WORLD_SIZE': '4', 'LOCAL_RANK': '1'})
    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.init_process_group')
    @patch('torch.cuda.is_available')
    def test_setup_distributed(self, mock_cuda_available, mock_init_process, mock_is_initialized):
        """测试分布式设置"""
        mock_cuda_available.return_value = True
        mock_is_initialized.return_value = False
        
        device_manager = DeviceManager(self.config)
        device = device_manager.setup_device()
        
        self.assertEqual(device.type, 'cuda')
        self.assertTrue(device_manager.distributed)
        self.assertEqual(device_manager.rank, 1)
        self.assertEqual(device_manager.world_size, 4)
        self.assertEqual(device_manager.local_rank, 1)
        self.assertFalse(device_manager.is_primary)

class TestLogManager(unittest.TestCase):
    """测试日志管理器"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.config = DictConfig({
            'experiment': {
                'name': 'test_exp',
                'save_config_snapshot': True
            }
        })
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_setup_logging_primary(self):
        """测试主要进程日志设置"""
        log_manager = LogManager(self.config, self.output_dir, is_primary=True)
        logger = log_manager.setup_logging()
        
        self.assertIsNotNone(logger)
        self.assertTrue((self.output_dir / "training.log").exists())
        self.assertTrue((self.output_dir / "config_merged.yaml").exists())
    
    def test_setup_logging_non_primary(self):
        """测试非主要进程日志设置"""
        with patch.dict(os.environ, {'RANK': '1'}):
            log_manager = LogManager(self.config, self.output_dir, is_primary=False)
            logger = log_manager.setup_logging()
            
            self.assertIsNotNone(logger)
            self.assertTrue((self.output_dir / "training_rank1.log").exists())
    
    def test_log_metrics(self):
        """测试指标记录"""
        log_manager = LogManager(self.config, self.output_dir, is_primary=True)
        logger = log_manager.setup_logging()
        
        # 模拟TensorBoard写入器
        log_manager.writer = Mock()
        
        metrics = {'loss': 0.1, 'accuracy': 0.9}
        log_manager.log_metrics(metrics, step=100)
        
        # 验证TensorBoard写入器被调用
        log_manager.writer.add_scalar.assert_called()

class TestConvertNumpyTypes(unittest.TestCase):
    """测试numpy类型转换函数"""
    
    def test_convert_numpy_integer(self):
        """测试numpy整数转换"""
        result = convert_numpy_types(np.int32(42))
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)
    
    def test_convert_numpy_float(self):
        """测试numpy浮点数转换"""
        result = convert_numpy_types(np.float32(3.14))
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14, places=5)
    
    def test_convert_numpy_array(self):
        """测试numpy数组转换"""
        arr = np.array([1, 2, 3])
        result = convert_numpy_types(arr)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])
    
    def test_convert_nested_dict(self):
        """测试嵌套字典转换"""
        data = {
            'int': np.int64(42),
            'float': np.float32(3.14),
            'array': np.array([1, 2, 3]),
            'nested': {
                'value': np.float64(2.71)
            }
        }
        result = convert_numpy_types(data)
        
        self.assertIsInstance(result['int'], int)
        self.assertIsInstance(result['float'], float)
        self.assertIsInstance(result['array'], list)
        self.assertIsInstance(result['nested']['value'], float)

class TestRealDataARTrainer(unittest.TestCase):
    """测试真实数据AR训练器"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建最小配置用于测试
        self.config = DictConfig({
            'experiment': {
                'name': 'test_trainer',
                'seed': 42,
                'output_dir': self.temp_dir,
                'device': 'cpu',
                'precision': '32',
                'log_every_n_steps': 10,
                'save_config_snapshot': False
            },
            'data': {
                'data_path': 'dummy_path.h5',
                'T_in': 1,
                'T_out': 5,
                'img_size': 64,
                'channels': 2,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'normalize': True,
                'augmentation': {'enabled': False},
                'keys': ['u', 'v'],
                'dataloader': {
                    'batch_size': 2,
                    'val_batch_size': 2,
                    'test_batch_size': 1,
                    'num_workers': 0,
                    'pin_memory': False,
                    'persistent_workers': False,
                    'drop_last': True,
                    'shuffle': True
                }
            },
            'model': {
                'name': 'SwinUNet',
                'in_channels': 2,
                'out_channels': 2,
                'img_size': 64,
                'patch_size': 4,
                'window_size': 8,
                'depths': [2, 2, 2, 2],
                'num_heads': [3, 6, 12, 24],
                'embed_dim': 48,
                'mlp_ratio': 4.0,
                'drop_rate': 0.1,
                'attn_drop_rate': 0.1,
                'drop_path_rate': 0.1
            },
            'training': {
                'epochs': 2,
                'batch_size': 2,
                'accumulate_grad_batches': 1,
                'optimizer': {
                    'name': 'AdamW',
                    'lr': 1e-4,
                    'weight_decay': 1e-4,
                    'betas': [0.9, 0.999]
                },
                'scheduler': {
                    'name': 'CosineAnnealingLR',
                    'T_max': 2,
                    'eta_min': 1e-6,
                    'warmup_epochs': 1
                },
                'gradient_clip_val': 1.0,
                'amp': {'enabled': False},
                'curriculum': {'enabled': False},
                'early_stopping': {'enabled': False},
                'checkpoint': {'save_last': False, 'save_best': False}
            },
            'loss': {
                'reconstruction': {'weight': 1.0},
                'spectral': {'weight': 0.0},
                'data_consistency': {'weight': 1.0},
                'degradation_consistency': {'weight': 0.0},
                'gradient_weight': 0.0
            },
            'observation': {
                'mode': 'identity',
                'scale_factor': 1,
                'blur_sigma': 0.0,
                'kernel_size': 1,
                'boundary': 'mirror',
                'downsample_interpolation': 'area'
            },
            'validation': {
                'check_val_every_n_epoch': 1,
                'val_check_interval': 1.0,
                'metrics': ['mse', 'mae']
            },
            'performance_monitoring': {'enabled': False},
            'hardware': {
                'num_workers': 0,
                'pin_memory': False,
                'persistent_workers': False,
                'allow_tf32': False
            },
            'testing': {'enabled': False},
            'paper_package': {'auto_generate': False}
        })
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        trainer = RealDataARTrainer()
        
        self.assertIsNotNone(trainer.config)
        self.assertIsNotNone(trainer.device_manager)
        self.assertIsNotNone(trainer.log_manager)
        self.assertIsNotNone(trainer.data_manager)
        self.assertIsNotNone(trainer.model_manager)
        self.assertIsNotNone(trainer.optimizer_manager)
        self.assertIsNotNone(trainer.loss_manager)
        self.assertIsNotNone(trainer.curriculum_manager)
        self.assertIsNotNone(trainer.checkpoint_manager)
    
    def test_trainer_setup(self):
        """测试训练器设置"""
        # 模拟数据模块和模型
        with patch('tools.training.train_real_data_ar.RealDiffusionReactionDataModule') as mock_data_module:
            with patch('tools.training.train_real_data_ar.SwinUNet') as mock_model:
                with patch('tools.training.train_real_data_ar.ARWrapper') as mock_ar_wrapper:
                    # 配置mock对象
                    mock_data_instance = Mock()
                    mock_data_instance.train_dataset = Mock()
                    mock_data_instance.train_dataset.__len__ = Mock(return_value=10)
                    
                    mock_mean = Mock()
                    mock_mean.numel.return_value = 2
                    mock_mean.__getitem__ = Mock(return_value=0.0)
                    mock_data_instance.train_dataset.mean = mock_mean
                    
                    mock_std = Mock()
                    mock_std.__getitem__ = Mock(return_value=1.0)
                    mock_data_instance.train_dataset.std = mock_std
                    
                    mock_data_instance.val_dataset = Mock()
                    mock_data_instance.val_dataset.__len__ = Mock(return_value=5)
                    mock_data_instance.test_dataset = Mock()
                    mock_data_instance.test_dataset.__len__ = Mock(return_value=5)
                    mock_data_instance.prepare_data = Mock()
                    mock_data_instance.setup = Mock()
                    mock_data_module.return_value = mock_data_instance
                    
                    mock_model_instance = Mock()
                    mock_model_instance.parameters = Mock(return_value=[])
                    mock_model.return_value = mock_model_instance
                    
                    mock_ar_instance = Mock()
                    mock_ar_instance.to = Mock(return_value=mock_ar_instance)
                    mock_ar_instance.parameters = Mock(return_value=[torch.randn(1, requires_grad=True)])
                    mock_ar_wrapper.return_value = mock_ar_instance
                    
                    trainer = RealDataARTrainer()
                    trainer.config = self.config
                    
                    success = trainer.setup()
                    
                    # 验证设置成功
                    self.assertTrue(success)
                    self.assertIsNotNone(trainer.logger)
    
    def test_curriculum_manager(self):
        """测试课程学习管理器"""
        config = DictConfig({
            'training': {
                'curriculum': {
                    'enabled': True,
                    'stages': [
                        {'epochs': 10, 'T_out': 5, 'description': '阶段1'},
                        {'epochs': 20, 'T_out': 10, 'description': '阶段2'}
                    ]
                }
            },
            'data': {'T_out': 20}
        })
        
        curriculum_manager = CurriculumManager(config, Mock())
        success = curriculum_manager.setup_curriculum()
        
        self.assertTrue(success)
        self.assertTrue(curriculum_manager.enabled)
        self.assertEqual(len(curriculum_manager.stages), 2)
        
        # 测试不同epoch的T_out
        self.assertEqual(curriculum_manager.get_current_T_out(5), 5)
        self.assertEqual(curriculum_manager.get_current_T_out(15), 10)
        self.assertEqual(curriculum_manager.get_current_T_out(35), 10)  # 超过边界

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_training_mock(self):
        """测试端到端训练流程（使用mock）"""
        # 创建最小配置
        config = DictConfig({
            'experiment': {
                'name': 'integration_test',
                'seed': 42,
                'output_dir': self.temp_dir,
                'device': 'cpu',
                'precision': '32',
                'log_every_n_steps': 1,
                'save_config_snapshot': False
            },
            'data': {
                'data_path': 'dummy_path.h5',
                'T_in': 1,
                'T_out': 3,
                'img_size': 32,
                'channels': 1,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'normalize': True,
                'augmentation': {'enabled': False},
                'keys': ['u'],
                'dataloader': {
                    'batch_size': 1,
                    'val_batch_size': 1,
                    'test_batch_size': 1,
                    'num_workers': 0,
                    'pin_memory': False,
                    'persistent_workers': False,
                    'drop_last': True,
                    'shuffle': True
                }
            },
            'model': {
                'name': 'SwinUNet',
                'in_channels': 1,
                'out_channels': 1,
                'img_size': 32,
                'patch_size': 4,
                'window_size': 8,
                'depths': [1, 1, 1, 1],
                'num_heads': [1, 2, 4, 8],
                'embed_dim': 24,
                'mlp_ratio': 2.0,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.0
            },
            'training': {
                'epochs': 1,
                'batch_size': 1,
                'accumulate_grad_batches': 1,
                'optimizer': {
                    'name': 'AdamW',
                    'lr': 1e-3,
                    'weight_decay': 1e-4,
                    'betas': [0.9, 0.999]
                },
                'scheduler': {
                    'name': 'CosineAnnealingLR',
                    'T_max': 1,
                    'eta_min': 1e-6,
                    'warmup_epochs': 0
                },
                'gradient_clip_val': 0.0,
                'amp': {'enabled': False},
                'curriculum': {'enabled': False},
                'early_stopping': {'enabled': False},
                'checkpoint': {'save_last': False, 'save_best': False}
            },
            'loss': {
                'reconstruction': {'weight': 1.0},
                'spectral': {'weight': 0.0},
                'data_consistency': {'weight': 0.0},
                'degradation_consistency': {'weight': 0.0},
                'gradient_weight': 0.0
            },
            'observation': {
                'mode': 'identity',
                'scale_factor': 1,
                'blur_sigma': 0.0,
                'kernel_size': 1,
                'boundary': 'mirror',
                'downsample_interpolation': 'area'
            },
            'validation': {
                'check_val_every_n_epoch': 1,
                'val_check_interval': 1.0,
                'metrics': ['mse']
            },
            'performance_monitoring': {'enabled': False},
            'hardware': {
                'num_workers': 0,
                'pin_memory': False,
                'persistent_workers': False,
                'allow_tf32': False
            },
            'testing': {'enabled': False},
            'paper_package': {'auto_generate': False}
        })
        
        # 模拟所有依赖
        with patch('tools.training.train_real_data_ar.RealDiffusionReactionDataModule') as mock_data_module:
            with patch('tools.training.train_real_data_ar.SwinUNet') as mock_model:
                with patch('tools.training.train_real_data_ar.ARWrapper') as mock_ar_wrapper:
                    with patch('tools.training.train_real_data_ar.compute_ar_total_loss') as mock_loss:
                        with patch('utils.metrics.compute_metrics') as mock_metrics:
                            # 配置mock对象
                            mock_data_instance = Mock()
                            mock_data_instance.train_dataset = Mock()
                            mock_data_instance.train_dataset.__len__ = Mock(return_value=10)
                            mock_mean = Mock()
                            mock_mean.numel.return_value = 2
                            mock_mean.__getitem__ = Mock(return_value=0.0)
                            mock_data_instance.train_dataset.mean = mock_mean
                            
                            mock_std = Mock()
                            mock_std.__getitem__ = Mock(return_value=1.0)
                            mock_data_instance.train_dataset.std = mock_std
                            
                            mock_data_instance.val_dataset = Mock()
                            mock_data_instance.val_dataset.__len__ = Mock(return_value=5)
                            mock_data_instance.test_dataset = Mock()
                            mock_data_instance.test_dataset.__len__ = Mock(return_value=5)
                            mock_data_instance.prepare_data = Mock()
                            mock_data_instance.setup = Mock()
                            mock_data_module.return_value = mock_data_instance
                            
                            mock_model_instance = Mock()
                            mock_model_instance.parameters = Mock(return_value=[Mock()])
                            mock_model_instance.to = Mock(return_value=mock_model_instance)
                            mock_model.return_value = mock_model_instance
                            
                            mock_ar_instance = Mock()
                            mock_ar_instance.to = Mock(return_value=mock_ar_instance)
                            mock_ar_instance.train = Mock()
                            mock_ar_instance.eval = Mock()
                            mock_ar_instance.parameters = Mock(return_value=[torch.randn(1, requires_grad=True)])
                            mock_ar_wrapper.return_value = mock_ar_instance
                            
                            # 模拟损失函数
                            def mock_loss_func(predictions, targets, loss_config, observation, observation_operators):
                                return torch.tensor(0.1), {'reconstruction_loss': 0.1}
                            mock_loss.side_effect = mock_loss_func
                            
                            # 模拟指标计算
                            def mock_metrics_func(predictions, targets, metrics_list):
                                return {'mse': 0.01}
                            mock_metrics.side_effect = mock_metrics_func
                            
                            # 创建训练器
                            trainer = RealDataARTrainer()
                            trainer.config = config
                            
                            # 设置训练环境
                            success = trainer.setup()
                            self.assertTrue(success)
                            
                            # 运行训练（简化版本）
                            # 这里我们只验证设置流程，不实际运行完整训练
                            self.assertIsNotNone(trainer.model_manager.model)
                            self.assertIsNotNone(trainer.optimizer_manager.optimizer)

def create_test_suite():
    """创建测试套件"""
    test_suite = unittest.TestSuite()
    
    # 添加配置管理器测试
    test_suite.addTest(unittest.makeSuite(TestConfigManager))
    
    # 添加设备管理器测试
    test_suite.addTest(unittest.makeSuite(TestDeviceManager))
    
    # 添加日志管理器测试
    test_suite.addTest(unittest.makeSuite(TestLogManager))
    
    # 添加工具函数测试
    test_suite.addTest(unittest.makeSuite(TestConvertNumpyTypes))
    
    # 添加训练器测试
    test_suite.addTest(unittest.makeSuite(TestRealDataARTrainer))
    
    # 添加集成测试
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    return test_suite

if __name__ == '__main__':
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # 输出测试总结
    print(f"\n测试完成:")
    print(f"  运行测试数: {result.testsRun}")
    print(f"  失败测试数: {len(result.failures)}")
    print(f"  错误测试数: {len(result.errors)}")
    print(f"  跳过的测试数: {len(result.skipped)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n出错的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # 返回适当的退出码
    sys.exit(0 if result.wasSuccessful() else 1)
