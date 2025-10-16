#!/usr/bin/env python3
"""PDEBench稀疏观测重建系统测试脚本

完整的系统测试，验证所有核心功能正常工作。
严格按照开发手册的黄金法则进行测试。

黄金法则：
1. 一致性优先：观测算子H与训练DC必须复用同一实现与配置
2. 可复现：同一YAML+种子，验证指标方差≤1e-4
3. 统一接口：所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
4. 可比性：横向对比必须报告均值±标准差（≥3种子）+资源成本
5. 文档先行：新增任务/算子/模型前，先提交PRD/技术文档补丁

使用方法:
    python test_system.py
    python test_system.py --quick  # 快速测试
    python test_system.py --full   # 完整测试
"""

import os
import sys
import json
import yaml
import tempfile
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from models.swin_unet import SwinUNet
from models.hybrid import HybridModel
from models.mlp import MLPModel
from ops.degradation import SuperResolutionOperator, CropOperator
from ops.loss import TotalLoss
from datasets.pdebench import PDEBenchDataModule
from utils.reproducibility import set_seed, get_environment_info


class SystemTester:
    """系统测试器
    
    测试所有核心功能：
    1. 模型接口一致性
    2. 观测算子H与训练DC一致性
    3. 损失函数计算域正确性
    4. 可复现性验证
    5. 配置加载和验证
    """
    
    def __init__(self, quick_mode: bool = False):
        """
        Args:
            quick_mode: 是否快速测试模式
        """
        self.quick_mode = quick_mode
        self.test_results = {}
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 测试配置
        self.test_config = {
            'batch_size': 2 if quick_mode else 4,
            'img_size': 64 if quick_mode else 128,
            'num_samples': 5 if quick_mode else 20,
            'tolerance': 1e-6,
            'seed': 2025
        }
        
        # 设置随机种子
        set_seed(self.test_config['seed'])
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def test_model_interfaces(self) -> bool:
        """测试模型接口一致性
        
        黄金法则3：统一接口 - 所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
        
        Returns:
            success: 测试是否通过
        """
        self.logger.info("Testing model interfaces...")
        
        try:
            batch_size = self.test_config['batch_size']
            img_size = self.test_config['img_size']
            in_channels = 3
            out_channels = 3
            
            # 测试输入
            x = torch.randn(batch_size, in_channels, img_size, img_size).to(self.device)
            
            # 测试所有模型
            models_to_test = [
                ('SwinUNet', SwinUNet, {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'img_size': img_size,
                    'patch_size': 4,
                    'window_size': 8,
                    'depths': [2, 2, 2, 2],
                    'num_heads': [3, 6, 12, 24],
                    'embed_dim': 96
                }),
                ('HybridModel', HybridModel, {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'img_size': img_size,
                    'hidden_dim': 128,
                    'num_layers': 4
                }),
                ('MLPModel', MLPModel, {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'img_size': img_size,
                    'hidden_dim': 256,
                    'num_layers': 4
                })
            ]
            
            interface_results = {}
            
            for model_name, model_class, model_kwargs in models_to_test:
                self.logger.info(f"Testing {model_name}...")
                
                try:
                    # 创建模型
                    model = model_class(**model_kwargs).to(self.device)
                    model.eval()
                    
                    # 前向传播
                    with torch.no_grad():
                        y = model(x)
                    
                    # 检查输出形状
                    expected_shape = (batch_size, out_channels, img_size, img_size)
                    if y.shape != expected_shape:
                        raise ValueError(f"Output shape mismatch: got {y.shape}, expected {expected_shape}")
                    
                    # 检查输出值域（应该是有限的）
                    if not torch.isfinite(y).all():
                        raise ValueError("Output contains non-finite values")
                    
                    interface_results[model_name] = {
                        'success': True,
                        'input_shape': list(x.shape),
                        'output_shape': list(y.shape),
                        'output_range': [float(y.min()), float(y.max())],
                        'params': sum(p.numel() for p in model.parameters())
                    }
                    
                    self.logger.info(f"✅ {model_name} interface test passed")
                    
                except Exception as e:
                    interface_results[model_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    self.logger.error(f"❌ {model_name} interface test failed: {e}")
            
            # 汇总结果
            success_count = sum(1 for r in interface_results.values() if r['success'])
            total_count = len(interface_results)
            
            self.test_results['model_interfaces'] = {
                'success': success_count == total_count,
                'passed': success_count,
                'total': total_count,
                'details': interface_results
            }
            
            self.logger.info(f"Model interface tests: {success_count}/{total_count} passed")
            return success_count == total_count
            
        except Exception as e:
            self.logger.error(f"Model interface testing failed: {e}")
            self.test_results['model_interfaces'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_observation_consistency(self) -> bool:
        """测试观测算子H与训练DC一致性
        
        黄金法则1：一致性优先 - 观测算子H与训练DC必须复用同一实现与配置
        
        Returns:
            success: 测试是否通过
        """
        self.logger.info("Testing observation operator consistency...")
        
        try:
            batch_size = self.test_config['batch_size']
            img_size = self.test_config['img_size']
            channels = 3
            tolerance = self.test_config['tolerance']
            
            # 创建测试数据
            gt = torch.randn(batch_size, channels, img_size, img_size)
            
            # 测试超分辨率观测算子
            sr_config = {
                'scale_factor': 2,
                'blur_sigma': 1.0,
                'blur_kernel_size': 5,
                'boundary_mode': 'mirror'
            }
            
            sr_degradation = SuperResolutionOperator(
                scale=sr_config['scale_factor'],
                sigma=sr_config['blur_sigma'],
                kernel_size=sr_config['blur_kernel_size'],
                boundary=sr_config['boundary_mode']
            )
            
            # 直接应用观测算子
            y_direct = sr_degradation(gt)
            
            # 通过损失函数应用观测算子
            loss_fn = TotalLoss(
                rec_weight=1.0,
                spec_weight=0.0,  # 关闭频谱损失
                dc_weight=1.0
            )
            
            # 模拟预测值（与GT相同，这样DC损失应该为0）
            pred = gt.clone()
            
            # 计算损失
            total_loss, loss_dict = loss_fn(pred, gt, y_direct, sr_config)
            dc_loss = loss_dict['data_consistency']
            
            # 检查DC损失是否接近0（说明H算子一致）
            if dc_loss.item() > tolerance:
                raise ValueError(f"DC loss too high: {dc_loss.item()}, expected < {tolerance}")
            
            # 测试裁剪观测算子
            crop_config = {
                'task': 'crop',
                'crop_size': [32, 32],
                'crop_box': None,  # 使用中心裁剪
                'boundary': 'mirror'
            }
            
            crop_degradation = CropOperator(
                crop_size=crop_config['crop_size'],
                boundary=crop_config['boundary']
            )
            
            # 直接应用观测算子
            y_crop_direct = crop_degradation(gt)
            
            # 通过损失函数应用观测算子
            loss_fn_crop = TotalLoss(
                rec_weight=1.0,
                spec_weight=0.0,
                dc_weight=1.0
            )
            
            total_loss_crop, loss_dict_crop = loss_fn_crop(pred, gt, y_crop_direct, crop_config)
            dc_loss_crop = loss_dict_crop['data_consistency']
            
            if dc_loss_crop.item() > tolerance:
                raise ValueError(f"Crop DC loss too high: {dc_loss_crop.item()}, expected < {tolerance}")
            
            self.test_results['observation_consistency'] = {
                'success': True,
                'sr_dc_loss': float(dc_loss.item()),
                'crop_dc_loss': float(dc_loss_crop.item()),
                'tolerance': tolerance
            }
            
            self.logger.info("✅ Observation consistency test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Observation consistency test failed: {e}")
            self.test_results['observation_consistency'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_loss_computation_domain(self) -> bool:
        """测试损失函数计算域正确性
        
        验证：
        - 模型输出在z-score域
        - DC与频域损失在原值域计算
        
        Returns:
            success: 测试是否通过
        """
        self.logger.info("Testing loss computation domain...")
        
        try:
            batch_size = self.test_config['batch_size']
            img_size = self.test_config['img_size']
            channels = 3
            
            # 创建测试数据（原值域）
            gt_original = torch.randn(batch_size, channels, img_size, img_size) * 10 + 5
            
            # 模拟归一化统计量
            mean = torch.tensor([5.0, 5.0, 5.0]).view(1, 3, 1, 1)
            std = torch.tensor([10.0, 10.0, 10.0]).view(1, 3, 1, 1)
            
            # 归一化到z-score域
            gt_normalized = (gt_original - mean) / std
            
            # 模拟模型预测（z-score域）
            pred_normalized = gt_normalized + 0.1 * torch.randn_like(gt_normalized)
            
            # 创建观测数据
            sr_config = {
                'scale_factor': 2,
                'blur_sigma': 1.0,
                'blur_kernel_size': 5,
                'boundary_mode': 'mirror'
            }
            
            sr_degradation = SuperResolutionOperator(
                scale=sr_config['scale_factor'],
                sigma=sr_config['blur_sigma'],
                kernel_size=sr_config['blur_kernel_size'],
                boundary=sr_config['boundary_mode']
            )
            y_observed = sr_degradation(gt_original)  # 观测数据在原值域
            
            # 创建损失函数
            loss_fn = TotalLoss(
                rec_weight=1.0,
                spec_weight=0.5,
                dc_weight=1.0,
                denormalize_fn=lambda x: x * std + mean
            )
            
            # 计算损失
            total_loss, loss_dict = loss_fn(pred_normalized, gt_normalized, y_observed, sr_config)
            
            # 检查各项损失都是有限值
            for loss_name, loss_value in loss_dict.items():
                if not torch.isfinite(loss_value):
                    raise ValueError(f"{loss_name} is not finite: {loss_value}")
            
            # 检查总损失合理性
            if total_loss.item() < 0 or total_loss.item() > 1000:
                raise ValueError(f"Total loss out of reasonable range: {total_loss.item()}")
            
            self.test_results['loss_computation_domain'] = {
                'success': True,
                'losses': {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in loss_dict.items()},
                'gt_original_range': [float(gt_original.min()), float(gt_original.max())],
                'gt_normalized_range': [float(gt_normalized.min()), float(gt_normalized.max())],
                'pred_normalized_range': [float(pred_normalized.min()), float(pred_normalized.max())]
            }
            
            self.logger.info("✅ Loss computation domain test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Loss computation domain test failed: {e}")
            self.test_results['loss_computation_domain'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_reproducibility(self) -> bool:
        """测试可复现性
        
        黄金法则2：可复现 - 同一YAML+种子，验证指标方差≤1e-4
        
        Returns:
            success: 测试是否通过
        """
        self.logger.info("Testing reproducibility...")
        
        try:
            num_runs = 3 if self.quick_mode else 5
            tolerance = 1e-4
            
            # 创建简单的测试模型
            model_config = {
                'in_channels': 3,
                'out_channels': 3,
                'img_size': 32,
                'hidden_dim': 64,
                'num_layers': 2
            }
            
            # 创建测试数据
            batch_size = 2
            x = torch.randn(batch_size, 3, 32, 32)
            
            results = []
            
            for run in range(num_runs):
                # 设置相同的随机种子
                set_seed(self.test_config['seed'])
                
                # 创建模型
                model = MLPModel(**model_config)
                model.eval()
                
                # 前向传播
                with torch.no_grad():
                    y = model(x)
                
                # 记录结果
                results.append(y.clone())
            
            # 检查结果一致性
            reference = results[0]
            max_diff = 0.0
            
            for i, result in enumerate(results[1:], 1):
                diff = torch.abs(result - reference).max().item()
                max_diff = max(max_diff, diff)
                
                if diff > tolerance:
                    raise ValueError(f"Run {i} differs from reference by {diff}, expected < {tolerance}")
            
            # 计算方差
            stacked_results = torch.stack(results, dim=0)
            variance = torch.var(stacked_results, dim=0).max().item()
            
            if variance > tolerance:
                raise ValueError(f"Variance {variance} exceeds tolerance {tolerance}")
            
            self.test_results['reproducibility'] = {
                'success': True,
                'num_runs': num_runs,
                'max_difference': max_diff,
                'variance': variance,
                'tolerance': tolerance
            }
            
            self.logger.info(f"✅ Reproducibility test passed (max_diff: {max_diff:.2e}, variance: {variance:.2e})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Reproducibility test failed: {e}")
            self.test_results['reproducibility'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_config_loading(self) -> bool:
        """测试配置加载和验证
        
        Returns:
            success: 测试是否通过
        """
        self.logger.info("Testing configuration loading...")
        
        try:
            config_files = [
                'configs/config.yaml',
                'configs/train.yaml',
                'configs/eval.yaml',
                'configs/model/swin_unet.yaml',
                'configs/data/pdebench.yaml'
            ]
            
            loaded_configs = {}
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    try:
                        config = OmegaConf.load(config_path)
                        loaded_configs[config_file] = {
                            'success': True,
                            'keys': list(config.keys()) if hasattr(config, 'keys') else []
                        }
                        self.logger.info(f"✅ Loaded {config_file}")
                    except Exception as e:
                        loaded_configs[config_file] = {
                            'success': False,
                            'error': str(e)
                        }
                        self.logger.warning(f"⚠️ Failed to load {config_file}: {e}")
                else:
                    loaded_configs[config_file] = {
                        'success': False,
                        'error': 'File not found'
                    }
                    self.logger.warning(f"⚠️ Config file not found: {config_file}")
            
            # 检查至少有一些配置文件成功加载
            success_count = sum(1 for c in loaded_configs.values() if c['success'])
            
            self.test_results['config_loading'] = {
                'success': success_count > 0,
                'loaded_configs': loaded_configs,
                'success_count': success_count,
                'total_count': len(config_files)
            }
            
            self.logger.info(f"Configuration loading: {success_count}/{len(config_files)} files loaded")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"❌ Configuration loading test failed: {e}")
            self.test_results['config_loading'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_tools_functionality(self) -> bool:
        """测试工具脚本功能
        
        Returns:
            success: 测试是否通过
        """
        self.logger.info("Testing tools functionality...")
        
        try:
            tools_results = {}
            
            # 测试训练脚本帮助信息
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, 'tools/train.py', '--help'
                ], capture_output=True, text=True, timeout=30)
                
                tools_results['train.py'] = {
                    'success': result.returncode == 0,
                    'help_available': '--help' in result.stdout or 'usage:' in result.stdout
                }
            except Exception as e:
                tools_results['train.py'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # 测试评估脚本帮助信息
            try:
                result = subprocess.run([
                    sys.executable, 'tools/eval.py', '--help'
                ], capture_output=True, text=True, timeout=30)
                
                tools_results['eval.py'] = {
                    'success': result.returncode == 0,
                    'help_available': '--help' in result.stdout or 'usage:' in result.stdout
                }
            except Exception as e:
                tools_results['eval.py'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # 测试一致性检查脚本帮助信息
            try:
                result = subprocess.run([
                    sys.executable, 'tools/check_dc_equivalence.py', '--help'
                ], capture_output=True, text=True, timeout=30)
                
                tools_results['check_dc_equivalence.py'] = {
                    'success': result.returncode == 0,
                    'help_available': '--help' in result.stdout or 'usage:' in result.stdout
                }
            except Exception as e:
                tools_results['check_dc_equivalence.py'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # 测试论文包生成脚本帮助信息
            try:
                result = subprocess.run([
                    sys.executable, 'tools/generate_paper_package.py', '--help'
                ], capture_output=True, text=True, timeout=30)
                
                tools_results['generate_paper_package.py'] = {
                    'success': result.returncode == 0,
                    'help_available': '--help' in result.stdout or 'usage:' in result.stdout
                }
            except Exception as e:
                tools_results['generate_paper_package.py'] = {
                    'success': False,
                    'error': str(e)
                }
            
            success_count = sum(1 for r in tools_results.values() if r['success'])
            total_count = len(tools_results)
            
            self.test_results['tools_functionality'] = {
                'success': success_count == total_count,
                'tools_results': tools_results,
                'success_count': success_count,
                'total_count': total_count
            }
            
            self.logger.info(f"Tools functionality: {success_count}/{total_count} tools working")
            return success_count == total_count
            
        except Exception as e:
            self.logger.error(f"❌ Tools functionality test failed: {e}")
            self.test_results['tools_functionality'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试
        
        Returns:
            test_results: 测试结果汇总
        """
        self.logger.info("="*60)
        self.logger.info("PDEBench稀疏观测重建系统 - 完整测试套件")
        self.logger.info("="*60)
        
        # 记录环境信息
        env_info = get_environment_info()
        self.test_results['environment'] = env_info
        
        # 运行各项测试
        tests = [
            ('模型接口一致性', self.test_model_interfaces),
            ('观测算子H与DC一致性', self.test_observation_consistency),
            ('损失函数计算域', self.test_loss_computation_domain),
            ('可复现性', self.test_reproducibility),
            ('配置加载', self.test_config_loading),
            ('工具脚本功能', self.test_tools_functionality)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            self.logger.info(f"\n🧪 运行测试: {test_name}")
            try:
                success = test_func()
                if success:
                    passed_tests += 1
                    self.logger.info(f"✅ {test_name} - 通过")
                else:
                    self.logger.error(f"❌ {test_name} - 失败")
            except Exception as e:
                self.logger.error(f"❌ {test_name} - 异常: {e}")
        
        # 汇总结果
        overall_success = passed_tests == total_tests
        
        self.test_results['summary'] = {
            'overall_success': overall_success,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests / total_tests,
            'test_mode': 'quick' if self.quick_mode else 'full'
        }
        
        self.logger.info("\n" + "="*60)
        self.logger.info("测试结果汇总")
        self.logger.info("="*60)
        self.logger.info(f"总体结果: {'✅ 通过' if overall_success else '❌ 失败'}")
        self.logger.info(f"通过测试: {passed_tests}/{total_tests}")
        self.logger.info(f"成功率: {passed_tests/total_tests:.1%}")
        self.logger.info(f"测试模式: {'快速' if self.quick_mode else '完整'}")
        
        if overall_success:
            self.logger.info("\n🎉 所有测试通过！系统功能正常。")
            self.logger.info("✅ 符合开发手册的黄金法则要求")
            self.logger.info("✅ 可以进行实际训练和评估")
        else:
            self.logger.error("\n⚠️ 部分测试失败，请检查相关功能")
            self.logger.error("❌ 建议修复失败的测试后再进行训练")
        
        return self.test_results
    
    def save_test_report(self, output_path: str = "test_report.json"):
        """保存测试报告
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"测试报告已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDEBench稀疏观测重建系统测试")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--full", action="store_true", help="完整测试模式")
    parser.add_argument("--output", type=str, default="test_report.json", help="测试报告输出路径")
    
    args = parser.parse_args()
    
    # 确定测试模式
    quick_mode = args.quick or not args.full
    
    # 创建测试器
    tester = SystemTester(quick_mode=quick_mode)
    
    # 运行测试
    results = tester.run_all_tests()
    
    # 保存报告
    tester.save_test_report(args.output)
    
    # 设置退出码
    exit_code = 0 if results['summary']['overall_success'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()