"""
测试系统稳定性

验证PDEBench稀疏观测重建系统的稳定性：
1. 可复现性：相同配置多次运行结果一致
2. 错误处理：异常情况下的鲁棒性
3. 断点续训：训练中断后能正确恢复
4. 内存管理：长时间运行不会内存泄漏
5. 数值稳定性：避免梯度爆炸/消失
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json
import time
import random
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import tempfile
import shutil
import warnings

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from utils.metrics import MetricsCalculator, StatisticalAnalyzer
    from test_resource_monitoring import ResourceMonitor
except ImportError as e:
    print(f"警告: 无法导入某些模块: {e}")
    print("将使用简化版本进行测试")


class SimpleTestModel(nn.Module):
    """简单测试模型"""
    
    def __init__(self, in_channels=3, out_channels=3, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv3(x)
        return x


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 10
    save_interval: int = 5
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class SystemStabilityTester:
    """系统稳定性测试器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.temp_dir = None
        
    def setup_temp_dir(self):
        """设置临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="stability_test_")
        return self.temp_dir
    
    def cleanup_temp_dir(self):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def set_seed(self, seed: int):
        """设置随机种子确保可复现性"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 确保确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def test_reproducibility(self) -> Dict[str, Any]:
        """测试可复现性"""
        print("测试系统可复现性...")
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            config = TrainingConfig()
            
            # 创建测试数据
            input_shape = (config.batch_size, 3, 64, 64)
            target_shape = (config.batch_size, 3, 64, 64)
            
            # 多次运行相同配置
            num_runs = 3
            final_losses = []
            final_weights = []
            
            for run in range(num_runs):
                print(f"  运行 {run + 1}/{num_runs}...")
                
                # 设置相同种子
                self.set_seed(config.seed)
                
                # 创建模型和优化器
                model = SimpleTestModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
                criterion = nn.MSELoss()
                
                # 训练几个epoch
                model.train()
                for epoch in range(5):
                    # 生成相同的随机数据（通过种子控制）
                    self.set_seed(config.seed + epoch)
                    x = torch.randn(input_shape).to(self.device)
                    target = torch.randn(target_shape).to(self.device)
                    
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, target)
                    loss.backward()
                    optimizer.step()
                
                # 记录最终损失和权重
                final_losses.append(loss.item())
                
                # 记录第一层权重的哈希值
                first_layer_weights = model.conv1.weight.data.cpu().numpy()
                weight_hash = hash(first_layer_weights.tobytes())
                final_weights.append(weight_hash)
            
            # 检查可复现性
            loss_variance = np.var(final_losses)
            weight_consistency = len(set(final_weights)) == 1
            
            results['details']['loss_variance'] = {
                'variance': loss_variance,
                'losses': final_losses,
                'passed': loss_variance < 1e-6
            }
            
            results['details']['weight_consistency'] = {
                'consistent': weight_consistency,
                'weight_hashes': final_weights,
                'passed': weight_consistency
            }
            
            if loss_variance >= 1e-6:
                results['passed'] = False
                results['errors'].append(f"损失方差过大: {loss_variance:.2e}")
            
            if not weight_consistency:
                results['passed'] = False
                results['errors'].append("权重不一致，可复现性失败")
            
            print(f"✓ 可复现性测试: {'通过' if results['passed'] else '失败'}")
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"可复现性测试异常: {str(e)}")
            print(f"❌ 可复现性测试失败: {e}")
        
        return results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """测试错误处理"""
        print("测试错误处理能力...")
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # 测试1: 输入维度不匹配
            print("  测试输入维度不匹配...")
            model = SimpleTestModel(in_channels=3, out_channels=3).to(self.device)
            
            try:
                wrong_input = torch.randn(2, 5, 64, 64).to(self.device)  # 错误的通道数
                with torch.no_grad():
                    _ = model(wrong_input)
                results['details']['dimension_mismatch'] = {
                    'handled': False,
                    'error': "应该抛出异常但没有"
                }
                results['passed'] = False
                results['errors'].append("维度不匹配未正确处理")
            except RuntimeError as e:
                results['details']['dimension_mismatch'] = {
                    'handled': True,
                    'error_type': type(e).__name__,
                    'error_msg': str(e)
                }
            
            # 测试2: 内存不足处理
            print("  测试内存不足处理...")
            try:
                # 尝试创建过大的张量
                huge_tensor = torch.randn(10000, 10000, 10000).to(self.device)
                results['details']['memory_overflow'] = {
                    'handled': False,
                    'error': "应该抛出内存异常但没有"
                }
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                results['details']['memory_overflow'] = {
                    'handled': True,
                    'error_type': type(e).__name__,
                    'error_msg': str(e)
                }
            except Exception as e:
                results['details']['memory_overflow'] = {
                    'handled': True,
                    'error_type': type(e).__name__,
                    'error_msg': str(e)
                }
            
            # 测试3: 梯度异常处理
            print("  测试梯度异常处理...")
            model = SimpleTestModel().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=1e10)  # 过大的学习率
            criterion = nn.MSELoss()
            
            x = torch.randn(2, 3, 64, 64).to(self.device)
            target = torch.randn(2, 3, 64, 64).to(self.device)
            
            gradient_norms = []
            for i in range(10):
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, target)
                loss.backward()
                
                # 检查梯度范数
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                gradient_norms.append(total_norm)
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    break
            
            has_gradient_explosion = any(norm > 100 for norm in gradient_norms)
            results['details']['gradient_handling'] = {
                'gradient_norms': gradient_norms[:5],  # 只保存前5个
                'has_explosion': has_gradient_explosion,
                'handled': True  # 通过梯度裁剪处理
            }
            
            print(f"✓ 错误处理测试: {'通过' if results['passed'] else '失败'}")
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"错误处理测试异常: {str(e)}")
            print(f"❌ 错误处理测试失败: {e}")
        
        return results
    
    def test_checkpoint_resume(self) -> Dict[str, Any]:
        """测试断点续训"""
        print("测试断点续训功能...")
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            temp_dir = self.setup_temp_dir()
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
            
            config = TrainingConfig(num_epochs=10)
            
            # 第一阶段：训练并保存检查点
            print("  第一阶段：训练并保存检查点...")
            self.set_seed(config.seed)
            
            model1 = SimpleTestModel().to(self.device)
            optimizer1 = optim.Adam(model1.parameters(), lr=config.learning_rate)
            criterion = nn.MSELoss()
            
            # 训练前半部分
            losses_phase1 = []
            for epoch in range(config.num_epochs // 2):
                x = torch.randn(config.batch_size, 3, 64, 64).to(self.device)
                target = torch.randn(config.batch_size, 3, 64, 64).to(self.device)
                
                optimizer1.zero_grad()
                pred = model1(x)
                loss = criterion(pred, target)
                loss.backward()
                optimizer1.step()
                
                losses_phase1.append(loss.item())
            
            # 保存检查点
            checkpoint = {
                'epoch': config.num_epochs // 2,
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'loss': losses_phase1[-1],
                'rng_state': torch.get_rng_state().cpu(),  # 确保在CPU上
                'cuda_rng_state': torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None
            }
            torch.save(checkpoint, checkpoint_path)
            
            # 第二阶段：从检查点恢复并继续训练
            print("  第二阶段：从检查点恢复...")
            
            model2 = SimpleTestModel().to(self.device)
            optimizer2 = optim.Adam(model2.parameters(), lr=config.learning_rate)
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model2.load_state_dict(checkpoint['model_state_dict'])
            optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            
            # 恢复随机数状态
            try:
                if 'rng_state' in checkpoint and checkpoint['rng_state'] is not None:
                    torch.set_rng_state(checkpoint['rng_state'])
            except Exception as e:
                print(f"  警告: 无法恢复RNG状态: {e}")
            
            try:
                if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            except Exception as e:
                print(f"  警告: 无法恢复CUDA RNG状态: {e}")
            
            # 继续训练
            losses_phase2 = []
            for epoch in range(start_epoch, config.num_epochs):
                x = torch.randn(config.batch_size, 3, 64, 64).to(self.device)
                target = torch.randn(config.batch_size, 3, 64, 64).to(self.device)
                
                optimizer2.zero_grad()
                pred = model2(x)
                loss = criterion(pred, target)
                loss.backward()
                optimizer2.step()
                
                losses_phase2.append(loss.item())
            
            # 第三阶段：完整训练作为对比
            print("  第三阶段：完整训练对比...")
            self.set_seed(config.seed)
            
            model3 = SimpleTestModel().to(self.device)
            optimizer3 = optim.Adam(model3.parameters(), lr=config.learning_rate)
            
            losses_complete = []
            for epoch in range(config.num_epochs):
                x = torch.randn(config.batch_size, 3, 64, 64).to(self.device)
                target = torch.randn(config.batch_size, 3, 64, 64).to(self.device)
                
                optimizer3.zero_grad()
                pred = model3(x)
                loss = criterion(pred, target)
                loss.backward()
                optimizer3.step()
                
                losses_complete.append(loss.item())
            
            # 验证断点续训结果
            combined_losses = losses_phase1 + losses_phase2
            
            # 检查损失序列是否一致
            loss_diff = np.mean(np.abs(np.array(combined_losses) - np.array(losses_complete)))
            
            # 检查最终模型权重是否一致
            weights_resumed = model2.conv1.weight.data.cpu().numpy()
            weights_complete = model3.conv1.weight.data.cpu().numpy()
            weight_diff = np.mean(np.abs(weights_resumed - weights_complete))
            
            results['details']['checkpoint_resume'] = {
                'loss_difference': loss_diff,
                'weight_difference': weight_diff,
                'losses_phase1': losses_phase1,
                'losses_phase2': losses_phase2,
                'losses_complete': losses_complete,
                'passed': loss_diff < 1e-6 and weight_diff < 1e-6
            }
            
            if loss_diff >= 1e-6 or weight_diff >= 1e-6:
                results['passed'] = False
                results['errors'].append(f"断点续训不一致: loss_diff={loss_diff:.2e}, weight_diff={weight_diff:.2e}")
            
            print(f"✓ 断点续训测试: {'通过' if results['passed'] else '失败'}")
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"断点续训测试异常: {str(e)}")
            print(f"❌ 断点续训测试失败: {e}")
        finally:
            self.cleanup_temp_dir()
        
        return results
    
    def test_memory_management(self) -> Dict[str, Any]:
        """测试内存管理"""
        print("测试内存管理...")
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # 获取初始内存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                initial_gpu_memory = 0
            
            print(f"  初始内存: CPU={initial_memory:.1f}MB, GPU={initial_gpu_memory:.1f}MB")
            
            # 模拟长时间训练
            model = SimpleTestModel().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            memory_usage = []
            gpu_memory_usage = []
            
            for iteration in range(100):
                # 创建批次数据
                x = torch.randn(8, 3, 128, 128).to(self.device)
                target = torch.randn(8, 3, 128, 128).to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                
                # 每10次迭代记录内存使用
                if iteration % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_usage.append(current_memory)
                    
                    if torch.cuda.is_available():
                        current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                        gpu_memory_usage.append(current_gpu_memory)
                    else:
                        gpu_memory_usage.append(0)
                
                # 清理不必要的变量
                del x, target, pred, loss
                
                # 定期垃圾回收
                if iteration % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 分析内存使用趋势
            memory_growth = memory_usage[-1] - memory_usage[0]
            gpu_memory_growth = gpu_memory_usage[-1] - gpu_memory_usage[0]
            
            # 检查是否有内存泄漏（增长超过50MB认为有问题）
            memory_leak = memory_growth > 50
            gpu_memory_leak = gpu_memory_growth > 50
            
            results['details']['memory_management'] = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': memory_usage[-1],
                'memory_growth_mb': memory_growth,
                'initial_gpu_memory_mb': initial_gpu_memory,
                'final_gpu_memory_mb': gpu_memory_usage[-1],
                'gpu_memory_growth_mb': gpu_memory_growth,
                'memory_leak': memory_leak,
                'gpu_memory_leak': gpu_memory_leak,
                'memory_usage_history': memory_usage,
                'gpu_memory_usage_history': gpu_memory_usage
            }
            
            if memory_leak:
                results['passed'] = False
                results['errors'].append(f"CPU内存泄漏: 增长{memory_growth:.1f}MB")
            
            if gpu_memory_leak:
                results['passed'] = False
                results['errors'].append(f"GPU内存泄漏: 增长{gpu_memory_growth:.1f}MB")
            
            print(f"  最终内存: CPU={memory_usage[-1]:.1f}MB (+{memory_growth:.1f}MB), GPU={gpu_memory_usage[-1]:.1f}MB (+{gpu_memory_growth:.1f}MB)")
            print(f"✓ 内存管理测试: {'通过' if results['passed'] else '失败'}")
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"内存管理测试异常: {str(e)}")
            print(f"❌ 内存管理测试失败: {e}")
        
        return results
    
    def test_numerical_stability(self) -> Dict[str, Any]:
        """测试数值稳定性"""
        print("测试数值稳定性...")
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # 测试不同学习率下的稳定性
            learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
            stability_results = {}
            
            for lr in learning_rates:
                print(f"  测试学习率 {lr}...")
                
                model = SimpleTestModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()
                
                losses = []
                gradient_norms = []
                weight_norms = []
                
                stable = True
                
                for epoch in range(20):
                    x = torch.randn(4, 3, 64, 64).to(self.device)
                    target = torch.randn(4, 3, 64, 64).to(self.device)
                    
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, target)
                    
                    # 检查损失是否为NaN或Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        stable = False
                        break
                    
                    loss.backward()
                    
                    # 计算梯度范数
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    gradient_norms.append(total_norm)
                    
                    # 检查梯度是否爆炸
                    if total_norm > 1000:
                        stable = False
                        break
                    
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    # 计算权重范数
                    weight_norm = sum(p.data.norm(2).item() for p in model.parameters())
                    weight_norms.append(weight_norm)
                    
                    # 检查权重是否爆炸
                    if weight_norm > 1000:
                        stable = False
                        break
                
                stability_results[lr] = {
                    'stable': stable,
                    'final_loss': losses[-1] if losses else float('inf'),
                    'max_gradient_norm': max(gradient_norms) if gradient_norms else 0,
                    'final_weight_norm': weight_norms[-1] if weight_norms else 0,
                    'losses': losses[:10],  # 只保存前10个
                    'gradient_norms': gradient_norms[:10]
                }
            
            # 检查是否有稳定的学习率
            stable_lrs = [lr for lr, result in stability_results.items() if result['stable']]
            
            results['details']['numerical_stability'] = {
                'stability_results': stability_results,
                'stable_learning_rates': stable_lrs,
                'has_stable_lr': len(stable_lrs) > 0
            }
            
            if len(stable_lrs) == 0:
                results['passed'] = False
                results['errors'].append("没有找到数值稳定的学习率")
            
            print(f"  稳定的学习率: {stable_lrs}")
            print(f"✓ 数值稳定性测试: {'通过' if results['passed'] else '失败'}")
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"数值稳定性测试异常: {str(e)}")
            print(f"❌ 数值稳定性测试失败: {e}")
        
        return results
    
    def generate_stability_report(self, results: Dict[str, Dict]) -> str:
        """生成稳定性报告"""
        report = []
        report.append("PDEBench稀疏观测重建系统 - 系统稳定性报告")
        report.append("=" * 60)
        report.append("")
        
        # 总体状态
        all_passed = all(result['passed'] for result in results.values())
        report.append(f"总体状态: {'✓ 通过' if all_passed else '❌ 失败'}")
        report.append("")
        
        # 各测试详情
        test_names = {
            'reproducibility': '1. 可复现性',
            'error_handling': '2. 错误处理',
            'checkpoint_resume': '3. 断点续训',
            'memory_management': '4. 内存管理',
            'numerical_stability': '5. 数值稳定性'
        }
        
        for test_key, test_name in test_names.items():
            if test_key in results:
                result = results[test_key]
                status = '✓ 通过' if result['passed'] else '❌ 失败'
                report.append(f"{test_name}: {status}")
                
                if result['errors']:
                    for error in result['errors']:
                        report.append(f"  - {error}")
                
                report.append("")
        
        # 建议
        report.append("改进建议:")
        if not all_passed:
            for test_key, result in results.items():
                if not result['passed']:
                    test_name = test_names.get(test_key, test_key)
                    report.append(f"- {test_name}: 需要修复上述问题")
        else:
            report.append("- 系统稳定性良好，所有测试均通过")
        
        return "\n".join(report)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有稳定性测试"""
        print("PDEBench稀疏观测重建系统 - 系统稳定性测试")
        print("=" * 60)
        
        results = {}
        
        # 运行各项测试
        results['reproducibility'] = self.test_reproducibility()
        results['error_handling'] = self.test_error_handling()
        results['checkpoint_resume'] = self.test_checkpoint_resume()
        results['memory_management'] = self.test_memory_management()
        results['numerical_stability'] = self.test_numerical_stability()
        
        # 生成报告
        report = self.generate_stability_report(results)
        print("\n" + report)
        
        # 保存结果
        self.results = results
        
        return results


def main():
    """主测试函数"""
    tester = SystemStabilityTester()
    
    try:
        results = tester.run_all_tests()
        
        # 检查总体结果
        all_passed = all(result['passed'] for result in results.values())
        
        if all_passed:
            print("\n🎉 所有系统稳定性测试通过！")
            return 0
        else:
            print("\n⚠️ 部分系统稳定性测试失败，需要改进。")
            return 1
            
    except Exception as e:
        print(f"\n❌ 系统稳定性测试异常: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)