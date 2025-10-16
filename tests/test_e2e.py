"""端到端测试脚本

验证PDEBench稀疏观测重建系统的完整训练-评测流程
确保各组件正确协同工作，满足技术架构文档的要求

测试内容：
1. 数据加载和预处理
2. 模型初始化和前向传播
3. 损失函数计算
4. 训练循环
5. 评测指标计算
6. 可视化生成
7. 一致性验证
"""

import os
import sys
import tempfile
import shutil
import torch
import torch.nn as nn
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.darcy_flow_dataset import DarcyFlowDataset
from models.swin_unet import SwinUNet
from models.hybrid import HybridModel
from models.mlp import MLPModel
from ops.loss import TotalLoss as CombinedLoss
from utils.metrics import MetricsCalculator as PDEBenchMetrics
from utils.visualization import PDEBenchVisualizer
from utils.distributed import setup_distributed, cleanup_distributed
from ops.degradation import apply_degradation_operator
from tools.check_dc_equivalence import DataConsistencyChecker
from tools.eval import Evaluator


class E2ETestSuite:
    """端到端测试套件"""
    
    def __init__(self, test_dir: Optional[str] = None):
        """
        Args:
            test_dir: 测试目录，如果为None则使用临时目录
        """
        if test_dir is None:
            self.test_dir = Path(tempfile.mkdtemp(prefix="pde_e2e_test_"))
        else:
            self.test_dir = Path(test_dir)
        
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试子目录
        (self.test_dir / 'data').mkdir(exist_ok=True)
        (self.test_dir / 'configs').mkdir(exist_ok=True)
        (self.test_dir / 'runs').mkdir(exist_ok=True)
        (self.test_dir / 'outputs').mkdir(exist_ok=True)
        
        # 测试配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.img_size = 64  # 小尺寸用于快速测试
        self.num_epochs = 2
        self.num_samples = 4
        
        print(f"E2E测试目录: {self.test_dir}")
        print(f"测试设备: {self.device}")
    
    def setup_test_data(self) -> Dict[str, Any]:
        """设置测试数据"""
        print("设置测试数据...")
        
        # 创建模拟数据
        data_dir = self.test_dir / 'data'
        
        # 生成模拟的Darcy Flow数据
        np.random.seed(42)
        torch.manual_seed(42)
        
        for split in ['train', 'val', 'test']:
            split_dir = data_dir / split
            split_dir.mkdir(exist_ok=True)
            
            num_files = 4 if split == 'train' else 2
            
            for i in range(num_files):
                # 生成模拟的PDE解
                x = np.linspace(0, 1, self.img_size)
                y = np.linspace(0, 1, self.img_size)
                X, Y = np.meshgrid(x, y)
                
                # 模拟Darcy Flow解（简单的高斯函数组合）
                solution = (np.exp(-((X-0.3)**2 + (Y-0.3)**2) / 0.1) + 
                           0.5 * np.exp(-((X-0.7)**2 + (Y-0.7)**2) / 0.05))
                
                # 添加噪声
                solution += 0.01 * np.random.randn(*solution.shape)
                
                # 保存为numpy文件
                np.save(split_dir / f'sample_{i:03d}.npy', solution.astype(np.float32))
        
        # 创建数据集配置
        dataset_config = {
            'data_dir': str(data_dir),
            'img_size': self.img_size,
            'normalize': True,
            'augment': False
        }
        
        return dataset_config
    
    def setup_test_configs(self) -> Dict[str, Dict[str, Any]]:
        """设置测试配置"""
        print("设置测试配置...")
        
        configs_dir = self.test_dir / 'configs'
        
        # 基础配置
        base_config = {
            'experiment': {
                'name': 'e2e_test',
                'seed': 42,
                'device': str(self.device)
            },
            'data': {
                'dataset': 'darcy_flow',
                'batch_size': self.batch_size,
                'num_workers': 0,
                'img_size': self.img_size
            },
            'training': {
                'epochs': self.num_epochs,
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'grad_clip': 1.0,
                'amp': False,  # 关闭AMP以简化测试
                'save_freq': 1
            },
            'loss': {
                'rec_weight': 1.0,
                'spec_weight': 0.5,
                'dc_weight': 1.0
            },
            'eval': {
                'batch_size': self.batch_size,
                'metrics': {
                    'img_size': self.img_size,
                    'boundary_width': 8
                }
            }
        }
        
        # 模型配置
        model_configs = {
            'swin_unet': {
                **base_config,
                'model': {
                    'name': 'swin_unet',
                    'in_channels': 1,
                    'out_channels': 1,
                    'img_size': self.img_size,
                    'patch_size': 4,
                    'window_size': 4,
                    'depths': [2, 2],
                    'num_heads': [2, 4],
                    'embed_dim': 48
                }
            },
            'hybrid': {
                **base_config,
                'model': {
                    'name': 'hybrid',
                    'in_channels': 1,
                    'out_channels': 1,
                    'img_size': self.img_size,
                    'hidden_dim': 64,
                    'num_layers': 2
                }
            },
            'mlp': {
                **base_config,
                'model': {
                    'name': 'mlp',
                    'in_channels': 1,
                    'out_channels': 1,
                    'img_size': self.img_size,
                    'hidden_dim': 128,
                    'num_layers': 3
                }
            }
        }
        
        # 保存配置文件
        for name, config in model_configs.items():
            config_path = configs_dir / f'{name}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        return model_configs
    
    def test_data_loading(self, dataset_config: Dict[str, Any]) -> bool:
        """测试数据加载"""
        print("测试数据加载...")
        
        try:
            # 创建数据集
            train_dataset = DarcyFlowDataset(
                data_dir=dataset_config['data_dir'],
                split='train',
                img_size=dataset_config['img_size'],
                normalize=dataset_config['normalize']
            )
            
            val_dataset = DarcyFlowDataset(
                data_dir=dataset_config['data_dir'],
                split='val',
                img_size=dataset_config['img_size'],
                normalize=dataset_config['normalize']
            )
            
            # 创建数据加载器
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            
            # 测试数据加载
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            
            # 验证数据形状
            assert train_batch['gt'].shape == (self.batch_size, 1, self.img_size, self.img_size)
            assert train_batch['baseline'].shape == (self.batch_size, 1, self.img_size, self.img_size)
            assert val_batch['gt'].shape[0] <= self.batch_size  # 可能小于batch_size
            
            print(f"✓ 数据加载测试通过")
            print(f"  训练集大小: {len(train_dataset)}")
            print(f"  验证集大小: {len(val_dataset)}")
            print(f"  数据形状: {train_batch['gt'].shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ 数据加载测试失败: {e}")
            return False
    
    def test_model_initialization(self, model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """测试模型初始化"""
        print("测试模型初始化...")
        
        results = {}
        
        for model_name, config in model_configs.items():
            try:
                model_config = config['model']
                
                if model_name == 'swin_unet':
                    model = SwinUNet(
                        in_channels=model_config['in_channels'],
                        out_channels=model_config['out_channels'],
                        img_size=model_config['img_size'],
                        patch_size=model_config['patch_size'],
                        window_size=model_config['window_size'],
                        depths=model_config['depths'],
                        num_heads=model_config['num_heads'],
                        embed_dim=model_config['embed_dim']
                    )
                elif model_name == 'hybrid':
                    model = HybridModel(
                        in_channels=model_config['in_channels'],
                        out_channels=model_config['out_channels'],
                        img_size=model_config['img_size'],
                        hidden_dim=model_config['hidden_dim'],
                        num_layers=model_config['num_layers']
                    )
                elif model_name == 'mlp':
                    model = MLPModel(
                        in_channels=model_config['in_channels'],
                        out_channels=model_config['out_channels'],
                        img_size=model_config['img_size'],
                        hidden_dim=model_config['hidden_dim'],
                        num_layers=model_config['num_layers']
                    )
                
                model = model.to(self.device)
                
                # 测试前向传播
                dummy_input = torch.randn(
                    self.batch_size, 
                    model_config['in_channels'], 
                    self.img_size, 
                    self.img_size
                ).to(self.device)
                
                with torch.no_grad():
                    output = model(dummy_input)
                
                expected_shape = (
                    self.batch_size, 
                    model_config['out_channels'], 
                    self.img_size, 
                    self.img_size
                )
                
                assert output.shape == expected_shape, f"输出形状不匹配: {output.shape} vs {expected_shape}"
                
                # 计算参数量
                num_params = sum(p.numel() for p in model.parameters())
                
                print(f"✓ {model_name}模型初始化测试通过")
                print(f"  参数量: {num_params:,}")
                print(f"  输出形状: {output.shape}")
                
                results[model_name] = True
                
            except Exception as e:
                print(f"✗ {model_name}模型初始化测试失败: {e}")
                results[model_name] = False
        
        return results
    
    def test_loss_computation(self, dataset_config: Dict[str, Any]) -> bool:
        """测试损失函数计算"""
        print("测试损失函数计算...")
        
        try:
            # 创建数据集和模型
            dataset = DarcyFlowDataset(
                data_dir=dataset_config['data_dir'],
                split='train',
                img_size=dataset_config['img_size']
            )
            
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False
            )
            
            # 创建简单模型用于测试
            model = nn.Conv2d(1, 1, 3, padding=1).to(self.device)
            
            # 创建损失函数
            criterion = CombinedLoss(
                rec_weight=1.0,
                spec_weight=0.5,
                dc_weight=1.0,
                img_size=self.img_size
            )
            
            # 获取一个batch
            batch = next(iter(dataloader))
            baseline = batch['baseline'].to(self.device)
            gt = batch['gt'].to(self.device)
            
            # 前向传播
            pred = model(baseline)
            
            # 计算损失
            loss_dict = criterion(pred, gt, baseline)
            
            # 验证损失组件
            required_keys = ['total_loss', 'rec_loss', 'spec_loss', 'dc_loss']
            for key in required_keys:
                assert key in loss_dict, f"缺少损失组件: {key}"
                assert isinstance(loss_dict[key], torch.Tensor), f"{key}不是张量"
                assert loss_dict[key].requires_grad, f"{key}不需要梯度"
            
            print(f"✓ 损失函数计算测试通过")
            print(f"  总损失: {loss_dict['total_loss'].item():.6f}")
            print(f"  重建损失: {loss_dict['rec_loss'].item():.6f}")
            print(f"  频谱损失: {loss_dict['spec_loss'].item():.6f}")
            print(f"  数据一致性损失: {loss_dict['dc_loss'].item():.6f}")
            
            return True
            
        except Exception as e:
            print(f"✗ 损失函数计算测试失败: {e}")
            return False
    
    def test_training_loop(self, model_configs: Dict[str, Dict[str, Any]], 
                          dataset_config: Dict[str, Any]) -> Dict[str, bool]:
        """测试训练循环"""
        print("测试训练循环...")
        
        results = {}
        
        for model_name in ['swin_unet']:  # 只测试一个模型以节省时间
            try:
                config = model_configs[model_name]
                
                # 创建数据集
                train_dataset = DarcyFlowDataset(
                    data_dir=dataset_config['data_dir'],
                    split='train',
                    img_size=dataset_config['img_size']
                )
                
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True
                )
                
                # 创建模型
                model_config = config['model']
                model = SwinUNet(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    img_size=model_config['img_size'],
                    patch_size=model_config['patch_size'],
                    window_size=model_config['window_size'],
                    depths=model_config['depths'],
                    num_heads=model_config['num_heads'],
                    embed_dim=model_config['embed_dim']
                ).to(self.device)
                
                # 创建优化器和损失函数
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config['training']['lr'],
                    weight_decay=config['training']['weight_decay']
                )
                
                criterion = CombinedLoss(
                    rec_weight=config['loss']['rec_weight'],
                    spec_weight=config['loss']['spec_weight'],
                    dc_weight=config['loss']['dc_weight'],
                    img_size=self.img_size
                )
                
                # 训练循环
                model.train()
                epoch_losses = []
                
                for epoch in range(self.num_epochs):
                    epoch_loss = 0.0
                    num_batches = 0
                    
                    for batch in train_loader:
                        baseline = batch['baseline'].to(self.device)
                        gt = batch['gt'].to(self.device)
                        
                        # 前向传播
                        pred = model(baseline)
                        
                        # 计算损失
                        loss_dict = criterion(pred, gt, baseline)
                        loss = loss_dict['total_loss']
                        
                        # 反向传播
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            config['training']['grad_clip']
                        )
                        
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                    
                    avg_loss = epoch_loss / num_batches
                    epoch_losses.append(avg_loss)
                    
                    print(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
                
                # 验证损失下降
                if len(epoch_losses) > 1:
                    loss_decreased = epoch_losses[-1] < epoch_losses[0]
                    if not loss_decreased:
                        print(f"  警告: 损失未下降 ({epoch_losses[0]:.6f} -> {epoch_losses[-1]:.6f})")
                
                print(f"✓ {model_name}训练循环测试通过")
                results[model_name] = True
                
            except Exception as e:
                print(f"✗ {model_name}训练循环测试失败: {e}")
                results[model_name] = False
        
        return results
    
    def test_evaluation_metrics(self, dataset_config: Dict[str, Any]) -> bool:
        """测试评测指标计算"""
        print("测试评测指标计算...")
        
        try:
            # 创建数据集
            dataset = DarcyFlowDataset(
                data_dir=dataset_config['data_dir'],
                split='test',
                img_size=dataset_config['img_size']
            )
            
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False
            )
            
            # 创建指标计算器
            metrics_calculator = PDEBenchMetrics(
                img_size=self.img_size,
                boundary_width=8
            )
            
            # 获取一个batch
            batch = next(iter(dataloader))
            baseline = batch['baseline']
            gt = batch['gt']
            
            # 创建模拟预测（添加少量噪声）
            pred = gt + 0.01 * torch.randn_like(gt)
            
            # 计算指标
            metrics = metrics_calculator.compute_all_metrics(pred, gt, baseline)
            
            # 验证指标
            required_metrics = [
                'rel_l2', 'mae', 'psnr', 'ssim', 
                'dc_error', 'frmse_low', 'frmse_mid', 'frmse_high'
            ]
            
            for metric in required_metrics:
                assert metric in metrics, f"缺少指标: {metric}"
                assert isinstance(metrics[metric], (float, torch.Tensor)), f"{metric}类型错误"
                if isinstance(metrics[metric], torch.Tensor):
                    assert not torch.isnan(metrics[metric]), f"{metric}为NaN"
                else:
                    assert not np.isnan(metrics[metric]), f"{metric}为NaN"
            
            print(f"✓ 评测指标计算测试通过")
            for metric, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                print(f"  {metric}: {value:.6f}")
            
            return True
            
        except Exception as e:
            print(f"✗ 评测指标计算测试失败: {e}")
            return False
    
    def test_visualization(self, dataset_config: Dict[str, Any]) -> bool:
        """测试可视化功能"""
        print("测试可视化功能...")
        
        try:
            # 创建数据集
            dataset = DarcyFlowDataset(
                data_dir=dataset_config['data_dir'],
                split='test',
                img_size=dataset_config['img_size']
            )
            
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False
            )
            
            # 创建可视化器
            vis_dir = self.test_dir / 'outputs' / 'visualization'
            visualizer = PDEBenchVisualizer(str(vis_dir))
            
            # 获取一个样本
            batch = next(iter(dataloader))
            baseline = batch['baseline'][0]  # [1, H, W]
            gt = batch['gt'][0]  # [1, H, W]
            
            # 创建模拟预测
            pred = gt + 0.05 * torch.randn_like(gt)
            
            # 测试各种可视化功能
            vis_paths = {}
            
            # 场对比图
            vis_paths['field'] = visualizer.plot_field_comparison(
                gt, pred, baseline, "test_field_comparison"
            )
            
            # 功率谱对比
            vis_paths['spectrum'] = visualizer.plot_power_spectrum_comparison(
                gt, pred, "test_power_spectrum"
            )
            
            # 边界分析
            vis_paths['boundary'] = visualizer.plot_boundary_analysis(
                gt, pred, save_name="test_boundary_analysis"
            )
            
            # 频域分析
            vis_paths['frequency'] = visualizer.plot_frequency_band_analysis(
                gt, pred, save_name="test_frequency_analysis"
            )
            
            # 验证文件生成
            for vis_type, path in vis_paths.items():
                assert os.path.exists(path), f"可视化文件未生成: {path}"
                print(f"  ✓ {vis_type}可视化: {path}")
            
            print(f"✓ 可视化功能测试通过")
            return True
            
        except Exception as e:
            print(f"✗ 可视化功能测试失败: {e}")
            return False
    
    def test_data_consistency(self, dataset_config: Dict[str, Any]) -> bool:
        """测试数据一致性验证"""
        print("测试数据一致性验证...")
        
        try:
            # 创建一致性检查器配置
            checker_config = {
                'dataset': {
                    'name': 'darcy_flow',
                    'data_path': dataset_config['data_dir'],
                    'keys': ['solution'],
                    'normalize': True
                }
            }
            
            checker = DataConsistencyChecker(checker_config)
            
            # 检查一致性
            results = checker.check_multiple_samples(
                num_samples=2,  # 少量样本用于快速测试
                random_seed=42
            )
            
            # 验证结果
            assert 'statistics' in results
            stats = results['statistics']
            
            print(f"✓ 数据一致性验证测试通过")
            print(f"  检查样本数: {stats['total_checked']}")
            print(f"  通过率: {stats['pass_rate']:.2%}")
            print(f"  一致性检查: {'通过' if stats['pass_rate'] >= 0.95 else '失败'}")
            
            return stats['pass_rate'] >= 0.95
            
        except Exception as e:
            print(f"✗ 数据一致性验证测试失败: {e}")
            return False
    
    def test_evaluator_integration(self, model_configs: Dict[str, Dict[str, Any]], 
                                 dataset_config: Dict[str, Any]) -> bool:
        """测试评测器集成"""
        print("测试评测器集成...")
        
        try:
            # 创建配置文件
            eval_config = {
                'data': {
                    'dataset': 'darcy_flow',
                    'data_dir': dataset_config['data_dir'],
                    'split': 'test',
                    'batch_size': self.batch_size,
                    'img_size': self.img_size
                },
                'model': model_configs['swin_unet']['model'],
                'eval': {
                    'batch_size': self.batch_size,
                    'device': str(self.device),
                    'metrics': {
                        'img_size': self.img_size,
                        'boundary_width': 8
                    },
                    'visualization': {
                        'enabled': True,
                        'max_samples': 2
                    },
                    'output': {
                        'save_predictions': True,
                        'save_metrics': True,
                        'formats': ['json', 'csv']
                    }
                }
            }
            
            # 保存配置
            config_path = self.test_dir / 'configs' / 'eval_test.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(eval_config, f)
            
            # 创建评测器
            evaluator = Evaluator(str(config_path))
            
            # 创建虚拟检查点（随机初始化的模型）
            checkpoint_dir = self.test_dir / 'runs' / 'test_model'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            model = SwinUNet(**eval_config['model'])
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 1,
                'config': eval_config
            }
            
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            
            # 运行评测
            results = evaluator.evaluate(str(checkpoint_path))
            
            # 验证结果
            assert 'metrics' in results
            assert 'aggregated' in results['metrics']
            
            required_metrics = ['rel_l2', 'mae', 'psnr', 'ssim']
            for metric in required_metrics:
                assert metric in results['metrics']['aggregated'], f"缺少聚合指标: {metric}"
            
            print(f"✓ 评测器集成测试通过")
            print(f"  评测样本数: {len(results['metrics']['per_sample'])}")
            print(f"  Rel-L2: {results['metrics']['aggregated']['rel_l2']:.6f}")
            
            return True
            
        except Exception as e:
            print(f"✗ 评测器集成测试失败: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        print("=" * 60)
        print("开始PDEBench稀疏观测重建系统端到端测试")
        print("=" * 60)
        
        results = {}
        
        try:
            # 1. 设置测试环境
            dataset_config = self.setup_test_data()
            model_configs = self.setup_test_configs()
            
            # 2. 数据加载测试
            results['data_loading'] = self.test_data_loading(dataset_config)
            
            # 3. 模型初始化测试
            model_init_results = self.test_model_initialization(model_configs)
            results.update({f'model_init_{k}': v for k, v in model_init_results.items()})
            
            # 4. 损失函数测试
            results['loss_computation'] = self.test_loss_computation(dataset_config)
            
            # 5. 训练循环测试
            training_results = self.test_training_loop(model_configs, dataset_config)
            results.update({f'training_{k}': v for k, v in training_results.items()})
            
            # 6. 评测指标测试
            results['evaluation_metrics'] = self.test_evaluation_metrics(dataset_config)
            
            # 7. 可视化测试
            results['visualization'] = self.test_visualization(dataset_config)
            
            # 8. 数据一致性测试
            results['data_consistency'] = self.test_data_consistency(dataset_config)
            
            # 9. 评测器集成测试
            results['evaluator_integration'] = self.test_evaluator_integration(
                model_configs, dataset_config
            )
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            results['overall'] = False
        
        # 汇总结果
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        for test_name, passed in results.items():
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"{test_name:30s} {status}")
        
        print("-" * 60)
        print(f"总计: {passed_tests}/{total_tests} 测试通过")
        
        overall_success = passed_tests == total_tests
        results['overall'] = overall_success
        
        if overall_success:
            print("🎉 所有端到端测试通过！系统运行正常。")
        else:
            print("⚠️  部分测试失败，请检查相关组件。")
        
        return results
    
    def cleanup(self):
        """清理测试环境"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print(f"清理测试目录: {self.test_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDEBench E2E测试')
    parser.add_argument('--test-dir', type=str, help='测试目录')
    parser.add_argument('--keep-files', action='store_true', help='保留测试文件')
    parser.add_argument('--device', type=str, default='auto', help='测试设备')
    
    args = parser.parse_args()
    
    # 创建测试套件
    test_suite = E2ETestSuite(args.test_dir)
    
    # 设置设备
    if args.device != 'auto':
        test_suite.device = torch.device(args.device)
    
    try:
        # 运行测试
        results = test_suite.run_all_tests()
        
        # 保存结果
        results_path = test_suite.test_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n测试结果已保存到: {results_path}")
        
        # 返回退出码
        exit_code = 0 if results.get('overall', False) else 1
        
    finally:
        # 清理
        if not args.keep_files:
            test_suite.cleanup()
    
    return exit_code


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)