#!/usr/bin/env python3
"""PDEBench稀疏观测重建系统 - 简化端到端测试

验证系统核心功能的基础测试脚本，确保主要组件能够正常工作。

使用方法:
    python tests/test_e2e_simple.py
    python tests/test_e2e_simple.py --keep-files
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 基础导入
from models.swin_unet import SwinUNet
from models.hybrid import HybridModel
from models.mlp import MLPModel
from ops.loss import TotalLoss
from utils.metrics import MetricsCalculator as PDEBenchMetrics
from ops.degradation import apply_degradation_operator


class SimpleE2ETest:
    """简化的端到端测试类"""
    
    def __init__(self, test_dir: str = "test_outputs", keep_files: bool = False):
        self.test_dir = Path(test_dir)
        self.keep_files = keep_files
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试参数
        self.img_size = 64  # 小尺寸以加快测试
        self.batch_size = 2
        self.num_epochs = 2
        
        print(f"测试设备: {self.device}")
        print(f"测试目录: {self.test_dir}")
        
        # 创建测试目录
        if not self.keep_files and self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)
        
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / 'data').mkdir(exist_ok=True)
        (self.test_dir / 'configs').mkdir(exist_ok=True)
        (self.test_dir / 'runs').mkdir(exist_ok=True)
    
    def setup_test_data(self) -> Dict[str, Any]:
        """设置测试数据"""
        print("设置测试数据...")
        
        data_dir = self.test_dir / 'data'
        
        # 创建数据集目录结构
        for split in ['train', 'val', 'test']:
            split_dir = data_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # 生成少量测试数据
            num_samples = 3 if split == 'train' else 2
            for i in range(num_samples):
                # 创建简单的2D场数据
                x = np.linspace(0, 1, self.img_size)
                y = np.linspace(0, 1, self.img_size)
                X, Y = np.meshgrid(x, y)
                
                # 模拟Darcy Flow解
                solution = (np.exp(-((X-0.3)**2 + (Y-0.3)**2) / 0.1) + 
                           0.5 * np.exp(-((X-0.7)**2 + (Y-0.7)**2) / 0.05))
                
                # 添加少量噪声
                solution += 0.01 * np.random.randn(*solution.shape)
                
                # 保存为numpy文件
                np.save(split_dir / f'sample_{i:03d}.npy', solution.astype(np.float32))
        
        dataset_config = {
            'data_dir': str(data_dir),
            'img_size': self.img_size,
            'normalize': True,
            'augment': False
        }
        
        return dataset_config
    
    def test_data_loading(self, dataset_config: Dict[str, Any]) -> bool:
        """测试数据加载"""
        print("测试数据加载...")
        
        try:
            # 创建简单的测试数据集类
            class SimpleDarcyDataset(torch.utils.data.Dataset):
                def __init__(self, data_dir, split, img_size, normalize=True):
                    self.data_dir = Path(data_dir)
                    self.split = split
                    self.img_size = img_size
                    self.normalize = normalize
                    
                    # 加载数据文件
                    split_dir = self.data_dir / split
                    self.files = list(split_dir.glob('*.npy'))
                
                def __len__(self):
                    return len(self.files)
                
                def __getitem__(self, idx):
                    # 加载数据
                    data = np.load(self.files[idx])
                    
                    # 调整尺寸
                    if data.shape != (self.img_size, self.img_size):
                        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                        data = F.interpolate(data, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
                        data = data.squeeze().numpy()
                    
                    # 转换为张量
                    gt = torch.from_numpy(data).unsqueeze(0).float()  # [1, H, W]
                    
                    # 创建基线（添加噪声）
                    baseline = gt + 0.1 * torch.randn_like(gt)
                    
                    return {
                        'gt': gt,
                        'baseline': baseline,
                        'coords': torch.zeros(2, self.img_size, self.img_size),  # 占位符
                        'mask': torch.ones(1, self.img_size, self.img_size)  # 占位符
                    }
            
            train_dataset = SimpleDarcyDataset(
                data_dir=dataset_config['data_dir'],
                split='train',
                img_size=dataset_config['img_size'],
                normalize=dataset_config['normalize']
            )
            
            # 创建数据加载器
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            
            # 测试数据加载
            train_batch = next(iter(train_loader))
            
            # 验证数据形状
            assert train_batch['gt'].shape == (self.batch_size, 1, self.img_size, self.img_size)
            assert train_batch['baseline'].shape == (self.batch_size, 1, self.img_size, self.img_size)
            
            print(f"✓ 数据加载测试通过")
            print(f"  训练集大小: {len(train_dataset)}")
            print(f"  数据形状: {train_batch['gt'].shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ 数据加载测试失败: {e}")
            return False
    
    def test_model_initialization(self) -> Dict[str, bool]:
        """测试模型初始化"""
        print("测试模型初始化...")
        
        results = {}
        
        # 测试SwinUNet模型
        try:
            model = SwinUNet(
                in_channels=1,
                out_channels=1,
                img_size=self.img_size,
                patch_size=4,
                window_size=4,
                depths=[2, 2],
                num_heads=[2, 4],
                embed_dim=48
            ).to(self.device)
            
            # 测试前向传播
            dummy_input = torch.randn(self.batch_size, 1, self.img_size, self.img_size).to(self.device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            expected_shape = (self.batch_size, 1, self.img_size, self.img_size)
            assert output.shape == expected_shape, f"输出形状不匹配: {output.shape} vs {expected_shape}"
            
            # 计算参数量
            num_params = sum(p.numel() for p in model.parameters())
            
            print(f"✓ SwinUNet模型初始化测试通过")
            print(f"  参数量: {num_params:,}")
            print(f"  输出形状: {output.shape}")
            
            results['swin_unet'] = True
            
        except Exception as e:
            print(f"✗ SwinUNet模型初始化测试失败: {e}")
            results['swin_unet'] = False
        
        return results
    
    def test_loss_computation(self, dataset_config: Dict[str, Any]) -> bool:
        """测试损失函数计算"""
        print("测试损失函数计算...")
        
        try:
            # 创建简单的测试数据集类
            class SimpleDarcyDataset(torch.utils.data.Dataset):
                def __init__(self, data_dir, split, img_size, normalize=True):
                    self.data_dir = Path(data_dir)
                    self.split = split
                    self.img_size = img_size
                    self.normalize = normalize
                    
                    # 加载数据文件
                    split_dir = self.data_dir / split
                    self.files = list(split_dir.glob('*.npy'))
                
                def __len__(self):
                    return len(self.files)
                
                def __getitem__(self, idx):
                    # 加载数据
                    data = np.load(self.files[idx])
                    
                    # 调整尺寸
                    if data.shape != (self.img_size, self.img_size):
                        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                        data = F.interpolate(data, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
                        data = data.squeeze().numpy()
                    
                    # 转换为张量
                    gt = torch.from_numpy(data).unsqueeze(0).float()  # [1, H, W]
                    
                    # 创建基线（添加噪声）
                    baseline = gt + 0.1 * torch.randn_like(gt)
                    
                    return {
                        'gt': gt,
                        'baseline': baseline,
                        'coords': torch.zeros(2, self.img_size, self.img_size),  # 占位符
                        'mask': torch.ones(1, self.img_size, self.img_size)  # 占位符
                    }
            
            dataset = SimpleDarcyDataset(
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
            criterion = TotalLoss(
                rec_weight=1.0,
                spec_weight=0.5,
                dc_weight=1.0
            )
            
            # 获取一个batch
            batch = next(iter(dataloader))
            baseline = batch['baseline'].to(self.device)
            gt = batch['gt'].to(self.device)
            
            # 前向传播
            pred = model(baseline)
            
            # 计算损失 - 使用简化的损失计算
            task_params = {'task': 'sr', 'scale': 2}
            total_loss, loss_dict = criterion(pred, gt, baseline, task_params)
            
            # 验证损失组件
            required_keys = ['total', 'reconstruction', 'spectral', 'data_consistency']
            for key in required_keys:
                assert key in loss_dict, f"缺少损失组件: {key}"
                assert isinstance(loss_dict[key], torch.Tensor), f"{key}不是张量"
                assert loss_dict[key].requires_grad, f"{key}不需要梯度"
            
            print(f"✓ 损失函数计算测试通过")
            print(f"  总损失: {loss_dict['total'].item():.6f}")
            print(f"  重建损失: {loss_dict['reconstruction'].item():.6f}")
            print(f"  频谱损失: {loss_dict['spectral'].item():.6f}")
            print(f"  数据一致性损失: {loss_dict['data_consistency'].item():.6f}")
            
            return True
            
        except Exception as e:
            print(f"✗ 损失函数计算测试失败: {e}")
            return False
    
    def test_training_loop(self, dataset_config: Dict[str, Any]) -> bool:
        """测试训练循环"""
        print("测试训练循环...")
        
        try:
            # 创建简单的测试数据集类
            class SimpleDarcyDataset(torch.utils.data.Dataset):
                def __init__(self, data_dir, split, img_size, normalize=True):
                    self.data_dir = Path(data_dir)
                    self.split = split
                    self.img_size = img_size
                    self.normalize = normalize
                    
                    # 加载数据文件
                    split_dir = self.data_dir / split
                    self.files = list(split_dir.glob('*.npy'))
                
                def __len__(self):
                    return len(self.files)
                
                def __getitem__(self, idx):
                    # 加载数据
                    data = np.load(self.files[idx])
                    
                    # 调整尺寸
                    if data.shape != (self.img_size, self.img_size):
                        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                        data = F.interpolate(data, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
                        data = data.squeeze().numpy()
                    
                    # 转换为张量
                    gt = torch.from_numpy(data).unsqueeze(0).float()  # [1, H, W]
                    
                    # 创建基线（添加噪声）
                    baseline = gt + 0.1 * torch.randn_like(gt)
                    
                    return {
                        'gt': gt,
                        'baseline': baseline,
                        'coords': torch.zeros(2, self.img_size, self.img_size),  # 占位符
                        'mask': torch.ones(1, self.img_size, self.img_size)  # 占位符
                    }
            
            train_dataset = SimpleDarcyDataset(
                data_dir=dataset_config['data_dir'],
                split='train',
                img_size=dataset_config['img_size']
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            
            # 创建模型
            model = SwinUNet(
                in_channels=1,
                out_channels=1,
                img_size=self.img_size,
                patch_size=4,
                window_size=4,
                depths=[2, 2],
                num_heads=[2, 4],
                embed_dim=48
            ).to(self.device)
            
            # 创建优化器和损失函数
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
            
            criterion = TotalLoss(
                rec_weight=1.0,
                spec_weight=0.5,
                dc_weight=1.0
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
                    task_params = {'task': 'sr', 'scale': 2}
                    loss, loss_dict = criterion(pred, gt, baseline, task_params)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                epoch_losses.append(avg_loss)
                
                print(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
            
            print(f"✓ 训练循环测试通过")
            return True
            
        except Exception as e:
            print(f"✗ 训练循环测试失败: {e}")
            return False
    
    def test_evaluation_metrics(self, dataset_config: Dict[str, Any]) -> bool:
        """测试评测指标计算"""
        print("测试评测指标计算...")
        
        try:
            # 创建指标计算器
            metrics_calc = PDEBenchMetrics(
                image_size=(self.img_size, self.img_size),
                boundary_width=8
            )
            
            # 创建测试数据
            gt = torch.randn(self.batch_size, 1, self.img_size, self.img_size)
            pred = gt + 0.1 * torch.randn_like(gt)  # 添加噪声模拟预测
            
            # 计算指标
            rel_l2 = metrics_calc.compute_rel_l2(pred, gt).mean().item()
            mae = metrics_calc.compute_mae(pred, gt).mean().item()
            psnr = metrics_calc.compute_psnr(pred, gt).mean().item()
            ssim_val = metrics_calc.compute_ssim(pred, gt).mean().item()
            
            metrics = {
                'rel_l2': rel_l2,
                'mae': mae,
                'psnr': psnr,
                'ssim': ssim_val
            }
            
            # 验证指标
            required_metrics = ['rel_l2', 'mae', 'psnr', 'ssim']
            for metric in required_metrics:
                assert metric in metrics, f"缺少指标: {metric}"
                assert isinstance(metrics[metric], (float, torch.Tensor)), f"{metric}类型错误"
            
            print(f"✓ 评测指标计算测试通过")
            print(f"  Rel-L2: {metrics['rel_l2']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  PSNR: {metrics['psnr']:.2f}")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"✗ 评测指标计算测试失败: {e}")
            return False
    
    def test_degradation_operator(self) -> bool:
        """测试退化算子"""
        print("测试退化算子...")
        
        try:
            # 创建测试数据
            test_data = torch.randn(1, 1, self.img_size, self.img_size)
            
            # 测试SR退化
            sr_params = {
                'task': 'sr',
                'scale': 2,
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
            
            sr_result = apply_degradation_operator(test_data, sr_params)
            expected_size = self.img_size // 2
            assert sr_result.shape == (1, 1, expected_size, expected_size), f"SR输出尺寸错误: {sr_result.shape}"
            
            # 测试Crop退化
            crop_params = {
                'task': 'crop',
                'crop_size': (32, 32),
                'boundary': 'mirror'
            }
            
            crop_result = apply_degradation_operator(test_data, crop_params)
            assert crop_result.shape == (1, 1, 32, 32), f"Crop输出尺寸错误: {crop_result.shape}"
            
            print(f"✓ 退化算子测试通过")
            print(f"  SR输出尺寸: {sr_result.shape}")
            print(f"  Crop输出尺寸: {crop_result.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ 退化算子测试失败: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        print("=" * 60)
        print("开始PDEBench稀疏观测重建系统简化端到端测试")
        print("=" * 60)
        
        results = {}
        
        try:
            # 1. 设置测试环境
            dataset_config = self.setup_test_data()
            
            # 2. 数据加载测试
            results['data_loading'] = self.test_data_loading(dataset_config)
            
            # 3. 模型初始化测试
            model_init_results = self.test_model_initialization()
            results.update({f'model_init_{k}': v for k, v in model_init_results.items()})
            
            # 4. 损失函数测试
            results['loss_computation'] = self.test_loss_computation(dataset_config)
            
            # 5. 训练循环测试
            results['training_loop'] = self.test_training_loop(dataset_config)
            
            # 6. 评测指标测试
            results['evaluation_metrics'] = self.test_evaluation_metrics(dataset_config)
            
            # 7. 退化算子测试
            results['degradation_operator'] = self.test_degradation_operator()
            
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
            print("🎉 所有简化端到端测试通过！系统核心功能正常。")
        else:
            print("⚠️  部分测试失败，请检查相关组件。")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='PDEBench稀疏观测重建系统简化端到端测试')
    parser.add_argument('--test-dir', type=str, default='test_outputs', help='测试输出目录')
    parser.add_argument('--keep-files', action='store_true', help='保留测试文件')
    
    args = parser.parse_args()
    
    # 运行测试
    tester = SimpleE2ETest(args.test_dir, args.keep_files)
    results = tester.run_all_tests()
    
    # 返回退出码
    if results.get('overall', False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
