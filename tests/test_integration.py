"""
集成测试：完整训练流程验证
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
import time
import logging

# 导入项目模块
from src.models.swin_temporal_nar import SwinTemporalNAR, SwinTemporalConfig
from src.data.pdebench_dataset import PDEBenchDataset, PDEBenchDataModule
from src.optimizers.hardware_profiler import HardwareProfiler
from src.optimizers.mixed_precision_trainer import MixedPrecisionTrainer, MixedPrecisionConfig
from src.monitoring.performance_monitor import PerformanceMonitor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logging, get_logger

# 设置测试日志
setup_logging(level=logging.WARNING)
logger = get_logger(__name__)

class TestIntegrationTrainingFlow(unittest.TestCase):
    """集成测试：完整训练流程"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建测试配置
        self.create_test_config()
        self.create_test_data()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config(self):
        """创建测试配置"""
        self.config = {
            'model': {
                'name': 'SwinTemporalNAR',
                'input_channels': 1,
                'hidden_dim': 64,
                'num_layers': 2,
                'num_heads': 4,
                'window_size': 7,
                'time_steps': 5,
                'prediction_steps': 3,
                'spatial_resolution': [32, 32]
            },
            'training': {
                'batch_size': 4,
                'num_epochs': 2,
                'learning_rate': 1e-3,
                'mixed_precision': True,
                'compile_model': False,
                'gradient_accumulation_steps': 1
            },
            'data': {
                'data_dir': self.temp_dir,
                'time_steps': 5,
                'prediction_steps': 3,
                'spatial_resolution': [32, 32],
                'normalize': True,
                'cache_data': False,
                'num_workers': 2
            },
            'hardware': {
                'device': 'auto',
                'num_workers': 2,
                'pin_memory': True
            },
            'monitoring': {
                'enable_monitoring': True,
                'log_dir': str(Path(self.temp_dir) / 'logs'),
                'monitoring_interval': 1.0,
                'enable_tensorboard': False,
                'enable_json_logging': False
            }
        }
        
        # 保存配置文件
        config_file = Path(self.temp_dir) / 'test_config.yaml'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.config_file = config_file
    
    def create_test_data(self):
        """创建测试数据"""
        import numpy as np
        
        # 创建模拟PDE数据
        data_dir = Path(self.temp_dir) / 'data'
        data_dir.mkdir(exist_ok=True)
        
        # 创建训练、验证、测试数据
        for split in ['train', 'validation', 'test']:
            split_dir = data_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # 创建模拟数据文件
            for i in range(2):  # 每个分割创建2个文件
                # 模拟数据: (time, channels, height, width)
                data = np.random.randn(20, 1, 32, 32).astype(np.float32)
                
                # 保存为numpy文件
                np.save(split_dir / f'data_{i}.npy', data)
    
    def test_config_loading(self):
        """测试配置加载"""
        config_loader = ConfigLoader()
        loaded_config = config_loader.load_config(str(self.config_file))
        
        # 验证配置加载
        self.assertIsInstance(loaded_config, dict)
        self.assertIn('model', loaded_config)
        self.assertIn('training', loaded_config)
        self.assertIn('data', loaded_config)
        
        # 验证配置值
        self.assertEqual(loaded_config['model']['name'], 'SwinTemporalNAR')
        self.assertEqual(loaded_config['training']['batch_size'], 4)
    
    def test_model_creation(self):
        """测试模型创建"""
        model_config = self.config['model']
        
        # 创建模型配置
        swin_config = SwinTemporalConfig(
            input_channels=model_config['input_channels'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            window_size=model_config['window_size'],
            time_steps=model_config['time_steps'],
            prediction_steps=model_config['prediction_steps'],
            spatial_resolution=tuple(model_config['spatial_resolution'])
        )
        
        # 创建模型
        model = SwinTemporalNAR(swin_config)
        
        # 验证模型创建
        self.assertIsInstance(model, SwinTemporalNAR)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)
        
        # 测试前向传播
        batch_size = self.config['training']['batch_size']
        time_steps = model_config['time_steps']
        channels = model_config['input_channels']
        height, width = model_config['spatial_resolution']
        
        test_input = torch.randn(batch_size, time_steps, channels, height, width)
        
        with torch.no_grad():
            output = model(test_input)
        
        # 验证输出形状
        expected_shape = (
            batch_size,
            model_config['prediction_steps'],
            channels,
            height,
            width
        )
        self.assertEqual(output.shape, expected_shape)
    
    def test_data_module_creation(self):
        """测试数据模块创建"""
        data_config = self.config['data']
        
        # 创建数据模块
        data_module = PDEBenchDataModule(data_config)
        
        # 设置数据
        data_module.setup()
        
        # 验证数据集创建
        self.assertIsNotNone(data_module.train_dataset)
        self.assertIsNotNone(data_module.val_dataset)
        self.assertIsNotNone(data_module.test_dataset)
        
        # 测试数据加载器
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        # 验证数据加载器
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        
        # 测试数据批次
        for inputs, targets in train_loader:
            self.assertEqual(inputs.shape[0], self.config['training']['batch_size'])
            self.assertEqual(targets.shape[0], self.config['training']['batch_size'])
            break  # 只测试第一个批次
    
    def test_mixed_precision_trainer(self):
        """测试混合精度训练器"""
        model_config = self.config['model']
        
        # 创建模型
        swin_config = SwinTemporalConfig(
            input_channels=model_config['input_channels'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            window_size=model_config['window_size'],
            time_steps=model_config['time_steps'],
            prediction_steps=model_config['prediction_steps'],
            spatial_resolution=tuple(model_config['spatial_resolution'])
        )
        
        model = SwinTemporalNAR(swin_config).to(self.device)
        
        # 创建混合精度配置
        mp_config = MixedPrecisionConfig(
            enabled=self.config['training']['mixed_precision'],
            loss_scale=1024.0,
            init_scale=1024.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000
        )
        
        # 创建混合精度训练器
        mp_trainer = MixedPrecisionTrainer(model, mp_config)
        
        # 验证创建
        self.assertIsInstance(mp_trainer, MixedPrecisionTrainer)
        
        # 测试训练步骤
        batch_size = self.config['training']['batch_size']
        time_steps = model_config['time_steps']
        channels = model_config['input_channels']
        height, width = model_config['spatial_resolution']
        prediction_steps = model_config['prediction_steps']
        
        # 创建测试数据
        inputs = torch.randn(batch_size, time_steps, channels, height, width).to(self.device)
        targets = torch.randn(batch_size, prediction_steps, channels, height, width).to(self.device)
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['training']['learning_rate'])
        criterion = nn.MSELoss()
        
        # 执行训练步骤
        optimizer.zero_grad()
        
        with mp_trainer.autocast_context():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 反向传播
        mp_trainer.scale_loss(loss).backward()
        
        # 优化器步骤
        mp_trainer.step(optimizer)
        
        # 验证训练步骤完成
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)  # MSE损失应该为正
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        monitoring_config = self.config['monitoring']
        
        # 创建性能监控器
        monitor = PerformanceMonitor(
            log_dir=monitoring_config['log_dir'],
            monitoring_interval=monitoring_config['monitoring_interval'],
            enable_tensorboard=monitoring_config['enable_tensorboard'],
            enable_json_logging=monitoring_config['enable_json_logging']
        )
        
        # 启动监控
        monitor.start_monitoring()
        
        # 模拟一些训练步骤
        for epoch in range(2):
            for step in range(5):
                # 模拟训练指标
                monitor.record_training_metrics(
                    epoch=epoch,
                    step=step,
                    loss=np.random.random() * 0.1,
                    learning_rate=self.config['training']['learning_rate'],
                    batch_time=0.1 + np.random.random() * 0.05,
                    data_loading_time=0.01 + np.random.random() * 0.01,
                    model_parameters=1000000,
                    gradient_norm=0.1 + np.random.random() * 0.05,
                    samples_per_second=50 + np.random.random() * 20
                )
                
                time.sleep(0.1)  # 模拟训练时间
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 验证监控数据
        summary = monitor.get_performance_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('total_samples', summary)
        self.assertGreater(summary['total_samples'], 0)
        
        # 清理监控资源
        monitor.cleanup()
    
    def test_complete_training_loop(self):
        """测试完整训练循环"""
        model_config = self.config['model']
        training_config = self.config['training']
        
        # 创建模型
        swin_config = SwinTemporalConfig(
            input_channels=model_config['input_channels'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            window_size=model_config['window_size'],
            time_steps=model_config['time_steps'],
            prediction_steps=model_config['prediction_steps'],
            spatial_resolution=tuple(model_config['spatial_resolution'])
        )
        
        model = SwinTemporalNAR(swin_config).to(self.device)
        
        # 创建数据模块
        data_module = PDEBenchDataModule(self.config['data'])
        data_module.setup()
        
        # 创建优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        criterion = nn.MSELoss()
        
        # 创建混合精度训练器
        mp_config = MixedPrecisionConfig(enabled=training_config['mixed_precision'])
        mp_trainer = MixedPrecisionTrainer(model, mp_config)
        
        # 创建性能监控器
        monitor = PerformanceMonitor(
            log_dir=self.config['monitoring']['log_dir'],
            enable_tensorboard=False,
            enable_json_logging=False
        )
        
        # 启动监控
        monitor.start_monitoring()
        
        try:
            # 训练循环
            train_loader = data_module.train_dataloader()
            
            for epoch in range(training_config['num_epochs']):
                model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    # 数据移动到设备
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 记录批次开始时间
                    batch_start_time = time.time()
                    
                    # 前向传播
                    optimizer.zero_grad()
                    
                    with mp_trainer.autocast_context():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    # 反向传播
                    mp_trainer.scale_loss(loss).backward()
                    mp_trainer.step(optimizer)
                    
                    # 记录指标
                    batch_time = time.time() - batch_start_time
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # 记录到监控器
                    monitor.record_training_metrics(
                        epoch=epoch,
                        step=batch_idx,
                        loss=loss.item(),
                        learning_rate=training_config['learning_rate'],
                        batch_time=batch_time,
                        samples_per_second=len(inputs) / batch_time,
                        model_parameters=sum(p.numel() for p in model.parameters())
                    )
                    
                    # 只测试几个批次
                    if batch_idx >= 3:
                        break
                
                # 计算平均损失
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")
                
                # 验证损失合理
                self.assertGreater(avg_loss, 0)  # MSE损失应该为正
                self.assertLess(avg_loss, 10.0)  # 损失不应该过大
        
        finally:
            # 停止监控
            monitor.stop_monitoring()
            monitor.cleanup()
        
        # 验证训练完成
        summary = monitor.get_performance_summary()
        self.assertGreater(summary['total_samples'], 0)
    
    def test_model_validation(self):
        """测试模型验证"""
        model_config = self.config['model']
        
        # 创建模型
        swin_config = SwinTemporalConfig(
            input_channels=model_config['input_channels'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            window_size=model_config['window_size'],
            time_steps=model_config['time_steps'],
            prediction_steps=model_config['prediction_steps'],
            spatial_resolution=tuple(model_config['spatial_resolution'])
        )
        
        model = SwinTemporalNAR(swin_config).to(self.device)
        
        # 创建数据模块
        data_module = PDEBenchDataModule(self.config['data'])
        data_module.setup()
        
        # 获取验证数据加载器
        val_loader = data_module.val_dataloader()
        
        # 验证模式
        model.eval()
        
        total_loss = 0.0
        num_batches = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 只测试几个批次
                if num_batches >= 3:
                    break
        
        # 计算平均验证损失
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 验证损失合理
        self.assertGreater(avg_val_loss, 0)
        self.assertLess(avg_val_loss, 10.0)

class TestIntegrationSystemComponents(unittest.TestCase):
    """集成测试：系统组件集成"""
    
    def test_hardware_profiler_integration(self):
        """测试硬件分析器集成"""
        profiler = HardwareProfiler()
        
        # 获取硬件配置
        hardware_config = profiler.get_optimal_config()
        
        # 验证配置可用于模型创建
        self.assertIsInstance(hardware_config.batch_size, int)
        self.assertIsInstance(hardware_config.num_workers, int)
        self.assertGreater(hardware_config.batch_size, 0)
        self.assertGreater(hardware_config.num_workers, 0)
        
        # 验证性能估算
        estimated_performance = profiler.estimate_training_performance(hardware_config)
        self.assertIn('samples_per_second', estimated_performance)
        self.assertIn('estimated_epoch_time', estimated_performance)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 创建有效配置
        valid_config = {
            'model': {
                'name': 'SwinTemporalNAR',
                'input_channels': 1,
                'hidden_dim': 64,
                'num_layers': 2,
                'num_heads': 4,
                'window_size': 7,
                'time_steps': 5,
                'prediction_steps': 3,
                'spatial_resolution': [32, 32]
            },
            'training': {
                'batch_size': 4,
                'num_epochs': 2,
                'learning_rate': 1e-3,
                'mixed_precision': True
            },
            'data': {
                'data_dir': '/tmp/test_data',
                'time_steps': 5,
                'prediction_steps': 3,
                'normalize': True
            }
        }
        
        # 验证配置可以成功加载和使用
        config_loader = ConfigLoader()
        
        # 保存并重新加载配置
        config_file = Path(tempfile.mktemp(suffix='.yaml'))
        with open(config_file, 'w') as f:
            json.dump(valid_config, f, indent=2)
        
        loaded_config = config_loader.load_config(str(config_file))
        
        # 验证关键字段存在
        self.assertIn('model', loaded_config)
        self.assertIn('training', loaded_config)
        self.assertIn('data', loaded_config)
        
        # 清理
        config_file.unlink(missing_ok=True)

if __name__ == '__main__':
    # 运行集成测试
    unittest.main(verbosity=2)