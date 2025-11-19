#!/usr/bin/env python3
"""
综合模型诊断工具 - 全面分析训练停滞问题
分析模型架构、训练过程、数据质量和性能指标
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.swin_unet import SwinUNet
from utils.logger import setup_logger

class ComprehensiveModelDiagnostic:
    """综合模型诊断器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.logger = setup_logger("model_diagnostic")
        self.diagnostics = {}
        
    def analyze_model_architecture(self, model: nn.Module, input_shape: tuple = (1, 2, 128, 128)) -> Dict[str, Any]:
        """分析模型架构问题"""
        self.logger.info("=== 模型架构分析 ===")
        
        diagnostics = {
            'parameter_count': 0,
            'layer_analysis': {},
            'dimension_flow': [],
            'potential_issues': []
        }
        
        try:
            # 参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            diagnostics['parameter_count'] = {
                'total': total_params,
                'trainable': trainable_params,
                'non_trainable': total_params - trainable_params
            }
            
            self.logger.info(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")
            
            # 分析每层的参数分布
            layer_params = {}
            for name, param in model.named_parameters():
                layer_name = name.split('.')[0]
                if layer_name not in layer_params:
                    layer_params[layer_name] = 0
                layer_params[layer_name] += param.numel()
            
            diagnostics['layer_analysis'] = layer_params
            
            # 维度流分析
            if hasattr(model, 'forward'):
                try:
                    with torch.no_grad():
                        x = torch.randn(*input_shape)
                        
                        # 钩子函数收集中间特征
                        activations = {}
                        
                        def hook_fn(name):
                            def hook(module, input, output):
                                if isinstance(output, torch.Tensor):
                                    activations[name] = output.shape
                                elif isinstance(output, (tuple, list)):
                                    activations[name] = [o.shape if isinstance(o, torch.Tensor) else str(type(o)) for o in output]
                            return hook
                        
                        # 注册钩子
                        hooks = []
                        for name, module in model.named_modules():
                            if len(list(module.children())) == 0:  # 叶子模块
                                hooks.append(module.register_forward_hook(hook_fn(name)))
                        
                        # 前向传播
                        _ = model(x)
                        
                        # 移除钩子
                        for hook in hooks:
                            hook.remove()
                        
                        diagnostics['dimension_flow'] = activations
                        
                        # 分析维度变化
                        input_size = input_shape[2] * input_shape[3]
                        for name, shape in activations.items():
                            if isinstance(shape, torch.Size) and len(shape) >= 3:
                                spatial_size = shape[-2] * shape[-1] if len(shape) >= 4 else shape[-1]
                                size_ratio = spatial_size / input_size
                                if size_ratio > 4:  # 异常放大
                                    diagnostics['potential_issues'].append(f"层 {name} 空间尺寸异常放大: {size_ratio:.1f}x")
                                elif size_ratio < 0.1:  # 过度压缩
                                    diagnostics['potential_issues'].append(f"层 {name} 过度压缩: {size_ratio:.2f}x")
                        
                except Exception as e:
                    self.logger.warning(f"维度流分析失败: {e}")
            
            # 检查激活函数配置
            activation_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid)):
                    activation_layers.append(f"{name}: {module.__class__.__name__}")
            
            diagnostics['activation_functions'] = activation_layers
            
            # 检查正则化层
            norm_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d)):
                    norm_layers.append(f"{name}: {module.__class__.__name__}")
            
            diagnostics['normalization_layers'] = norm_layers
            
            # 检查Dropout层
            dropout_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                    dropout_layers.append(f"{name}: p={module.p}")
            
            diagnostics['dropout_layers'] = dropout_layers
            
        except Exception as e:
            self.logger.error(f"模型架构分析失败: {e}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def analyze_training_dynamics(self, log_file_path: str) -> Dict[str, Any]:
        """分析训练动态"""
        self.logger.info("=== 训练动态分析 ===")
        
        diagnostics = {
            'loss_analysis': {},
            'convergence_metrics': {},
            'gradient_analysis': {},
            'training_issues': []
        }
        
        try:
            # 解析训练日志
            train_losses = []
            val_losses = []
            epochs = []
            
            with open(log_file_path, 'r') as f:
                for line in f:
                    if 'Train Loss:' in line and 'Val Loss:' in line:
                        try:
                            # 提取损失值
                            parts = line.split('|')
                            for part in parts:
                                if 'Train Loss:' in part:
                                    train_loss = float(part.split('Train Loss:')[1].split()[0])
                                    train_losses.append(train_loss)
                                if 'Val Loss:' in part:
                                    val_loss = float(part.split('Val Loss:')[1].split()[0])
                                    val_losses.append(val_loss)
                            
                            # 提取epoch
                            if 'Epoch' in line:
                                epoch_str = line.split('Epoch')[1].split('/')[0].strip()
                                epochs.append(int(epoch_str))
                                
                        except (ValueError, IndexError):
                            continue
            
            if not train_losses:
                diagnostics['training_issues'].append("无法解析训练日志中的损失值")
                return diagnostics
            
            # 损失分析
            diagnostics['loss_analysis'] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': epochs,
                'train_loss_mean': np.mean(train_losses),
                'train_loss_std': np.std(train_losses),
                'val_loss_mean': np.mean(val_losses),
                'val_loss_std': np.std(val_losses),
                'loss_ratio': np.mean(val_losses) / np.mean(train_losses) if np.mean(train_losses) > 0 else 0
            }
            
            # 收敛性分析
            if len(train_losses) >= 10:
                recent_losses = train_losses[-10:]
                loss_trend = np.polyfit(range(10), recent_losses, 1)[0]
                
                # 检查是否停滞
                recent_std = np.std(recent_losses)
                is_stagnant = abs(loss_trend) < 1e-4 and recent_std < 0.01
                
                diagnostics['convergence_metrics'] = {
                    'loss_trend': loss_trend,
                    'recent_std': recent_std,
                    'is_stagnant': is_stagnant,
                    'stagnation_severity': 'high' if is_stagnant else 'low'
                }
                
                if is_stagnant:
                    diagnostics['training_issues'].append(f"检测到损失停滞: 趋势={loss_trend:.6f}, 标准差={recent_std:.6f}")
            
            # 过拟合分析
            if len(train_losses) == len(val_losses) and len(train_losses) > 5:
                recent_train = train_losses[-5:]
                recent_val = val_losses[-5:]
                
                train_decrease = recent_train[0] - recent_train[-1]
                val_decrease = recent_val[0] - recent_val[-1]
                
                if train_decrease > 0.01 and val_decrease < 0.001:
                    diagnostics['training_issues'].append("可能过拟合: 训练损失下降但验证损失停滞")
                elif val_decrease < -0.01:
                    diagnostics['training_issues'].append("验证损失上升: 模型性能恶化")
            
            # 损失振荡分析
            if len(train_losses) >= 20:
                recent_oscillation = np.std(train_losses[-20:])
                if recent_oscillation > 0.1:
                    diagnostics['training_issues'].append(f"损失振荡严重: 标准差={recent_oscillation:.4f}")
            
        except Exception as e:
            self.logger.error(f"训练动态分析失败: {e}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def analyze_gradient_health(self, model: nn.Module, sample_data: torch.Tensor, 
                              target: torch.Tensor, loss_fn) -> Dict[str, Any]:
        """分析梯度健康状况"""
        self.logger.info("=== 梯度健康分析 ===")
        
        diagnostics = {
            'gradient_norms': {},
            'gradient_stats': {},
            'vanishing_exploding': {},
            'gradient_issues': []
        }
        
        try:
            model.train()
            
            # 前向传播
            output = model(sample_data)
            loss = loss_fn(output, target)
            
            # 反向传播
            model.zero_grad()
            loss.backward()
            
            # 收集梯度统计
            grad_norms = []
            layer_grad_stats = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    grad_norms.append(grad_norm)
                    
                    # 每层梯度统计
                    grad_mean = param.grad.data.mean().item()
                    grad_std = param.grad.data.std().item()
                    grad_max = param.grad.data.max().item()
                    grad_min = param.grad.data.min().item()
                    
                    layer_grad_stats[name] = {
                        'norm': grad_norm,
                        'mean': grad_mean,
                        'std': grad_std,
                        'max': grad_max,
                        'min': grad_min
                    }
            
            diagnostics['gradient_norms'] = grad_norms
            diagnostics['gradient_stats'] = layer_grad_stats
            
            # 梯度消失/爆炸检测
            if grad_norms:
                total_norm = np.sqrt(sum(g**2 for g in grad_norms))
                
                # 梯度消失检测
                vanishing_threshold = 1e-7
                vanishing_ratio = sum(1 for g in grad_norms if g < vanishing_threshold) / len(grad_norms)
                
                # 梯度爆炸检测
                exploding_threshold = 10.0
                exploding_ratio = sum(1 for g in grad_norms if g > exploding_threshold) / len(grad_norms)
                
                diagnostics['vanishing_exploding'] = {
                    'total_norm': total_norm,
                    'vanishing_ratio': vanishing_ratio,
                    'exploding_ratio': exploding_ratio,
                    'mean_norm': np.mean(grad_norms),
                    'std_norm': np.std(grad_norms)
                }
                
                if vanishing_ratio > 0.5:
                    diagnostics['gradient_issues'].append(f"梯度消失严重: {vanishing_ratio:.1%} 的参数梯度 < {vanishing_threshold}")
                
                if exploding_ratio > 0.1:
                    diagnostics['gradient_issues'].append(f"梯度爆炸风险: {exploding_ratio:.1%} 的参数梯度 > {exploding_threshold}")
                
                if total_norm > 100:
                    diagnostics['gradient_issues'].append(f"总梯度范数过大: {total_norm:.2f}")
                elif total_norm < 1e-6:
                    diagnostics['gradient_issues'].append(f"总梯度范数过小: {total_norm:.2e}")
            
        except Exception as e:
            self.logger.error(f"梯度健康分析失败: {e}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def analyze_data_quality(self, data_loader) -> Dict[str, Any]:
        """分析数据质量"""
        self.logger.info("=== 数据质量分析 ===")
        
        diagnostics = {
            'data_statistics': {},
            'label_distribution': {},
            'data_issues': []
        }
        
        try:
            # 收集样本统计
            sample_stats = []
            label_stats = []
            
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= 10:  # 限制样本数量
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    diagnostics['data_issues'].append("数据格式异常")
                    break
                
                # 输入数据统计
                if isinstance(inputs, torch.Tensor):
                    input_stats = {
                        'mean': inputs.mean().item(),
                        'std': inputs.std().item(),
                        'min': inputs.min().item(),
                        'max': inputs.max().item(),
                        'shape': list(inputs.shape),
                        'nan_ratio': torch.isnan(inputs).float().mean().item(),
                        'inf_ratio': torch.isinf(inputs).float().mean().item()
                    }
                    sample_stats.append(input_stats)
                
                # 目标数据统计
                if isinstance(targets, torch.Tensor):
                    target_stats = {
                        'mean': targets.mean().item(),
                        'std': targets.std().item(),
                        'min': targets.min().item(),
                        'max': targets.max().item(),
                        'nan_ratio': torch.isnan(targets).float().mean().item(),
                        'inf_ratio': torch.isinf(targets).float().mean().item()
                    }
                    label_stats.append(target_stats)
            
            if sample_stats:
                # 聚合统计
                diagnostics['data_statistics'] = {
                    'input_mean': np.mean([s['mean'] for s in sample_stats]),
                    'input_std': np.mean([s['std'] for s in sample_stats]),
                    'input_range': [
                        np.min([s['min'] for s in sample_stats]),
                        np.max([s['max'] for s in sample_stats])
                    ],
                    'nan_ratio': np.mean([s['nan_ratio'] for s in sample_stats]),
                    'inf_ratio': np.mean([s['inf_ratio'] for s in sample_stats])
                }
            
            if label_stats:
                diagnostics['label_distribution'] = {
                    'target_mean': np.mean([s['mean'] for s in label_stats]),
                    'target_std': np.mean([s['std'] for s in label_stats]),
                    'target_range': [
                        np.min([s['min'] for s in label_stats]),
                        np.max([s['max'] for s in label_stats])
                    ]
                }
            
            # 数据质量检查
            if diagnostics['data_statistics'].get('nan_ratio', 0) > 0.01:
                diagnostics['data_issues'].append(f"输入数据NaN比例过高: {diagnostics['data_statistics']['nan_ratio']:.1%}")
            
            if diagnostics['data_statistics'].get('inf_ratio', 0) > 0.001:
                diagnostics['data_issues'].append(f"输入数据Inf比例过高: {diagnostics['data_statistics']['inf_ratio']:.1%}")
            
            # 检查数据范围
            input_range = diagnostics['data_statistics'].get('input_range', [0, 1])
            if abs(input_range[0]) > 1000 or abs(input_range[1]) > 1000:
                diagnostics['data_issues'].append(f"输入数据范围异常: [{input_range[0]:.2f}, {input_range[1]:.2f}]")
            
            # 检查标准化
            input_mean = diagnostics['data_statistics'].get('input_mean', 0)
            input_std = diagnostics['data_statistics'].get('input_std', 1)
            if abs(input_mean) > 2 or input_std > 10:
                diagnostics['data_issues'].append(f"输入数据可能未标准化: mean={input_mean:.2f}, std={input_std:.2f}")
            
        except Exception as e:
            self.logger.error(f"数据质量分析失败: {e}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def generate_architecture_visualization(self, model: nn.Module, save_path: str):
        """生成模型架构可视化"""
        self.logger.info("=== 生成架构可视化 ===")
        
        try:
            # 参数分布图
            plt.figure(figsize=(15, 10))
            
            # 子图1: 参数分布
            plt.subplot(2, 3, 1)
            param_counts = []
            layer_names = []
            
            for name, param in model.named_parameters():
                layer_name = name.split('.')[0]
                if layer_name not in layer_names:
                    layer_names.append(layer_name)
                    param_counts.append(0)
                
                idx = layer_names.index(layer_name)
                param_counts[idx] += param.numel()
            
            plt.bar(range(len(layer_names)), param_counts)
            plt.xticks(range(len(layer_names)), layer_names, rotation=45)
            plt.ylabel('参数数量')
            plt.title('各层参数分布')
            plt.yscale('log')
            
            # 子图2: 激活函数分布
            plt.subplot(2, 3, 2)
            activation_counts = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid)):
                    act_name = module.__class__.__name__
                    activation_counts[act_name] = activation_counts.get(act_name, 0) + 1
            
            if activation_counts:
                plt.pie(activation_counts.values(), labels=activation_counts.keys(), autopct='%1.1f%%')
                plt.title('激活函数分布')
            
            # 子图3: 正则化层分布
            plt.subplot(2, 3, 3)
            norm_counts = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d)):
                    norm_name = module.__class__.__name__
                    norm_counts[norm_name] = norm_counts.get(norm_name, 0) + 1
            
            if norm_counts:
                plt.bar(norm_counts.keys(), norm_counts.values())
                plt.title('正则化层分布')
                plt.xticks(rotation=45)
            
            # 子图4: Dropout分布
            plt.subplot(2, 3, 4)
            dropout_rates = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                    dropout_rates.append(module.p)
            
            if dropout_rates:
                plt.hist(dropout_rates, bins=10, edgecolor='black')
                plt.xlabel('Dropout率')
                plt.ylabel('数量')
                plt.title('Dropout率分布')
            
            # 子图5: 模型复杂度指标
            plt.subplot(2, 3, 5)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            complexity_metrics = ['总参数', '可训练参数', '层数', '激活函数数']
            complexity_values = [
                total_params / 1e6,  # 百万参数
                trainable_params / 1e6,
                len([m for m in model.modules()]),
                len([m for m in model.modules() if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid))])
            ]
            
            plt.bar(complexity_metrics, complexity_values)
            plt.title('模型复杂度指标')
            plt.xticks(rotation=45)
            
            # 子图6: 内存使用估算
            plt.subplot(2, 3, 6)
            # 粗略估算内存使用（参数 + 激活）
            param_memory = total_params * 4 / 1e9  # GB, 假设float32
            
            # 激活内存估算（假设输入为1x2x128x128）
            activation_memory = 0.1  # 粗略估算GB
            
            memory_types = ['参数内存', '激活内存', '总内存']
            memory_values = [param_memory, activation_memory, param_memory + activation_memory]
            
            plt.bar(memory_types, memory_values)
            plt.title('内存使用估算 (GB)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"架构可视化已保存: {save_path}")
            
        except Exception as e:
            self.logger.error(f"架构可视化生成失败: {e}")
    
    def generate_comprehensive_report(self, diagnostics: Dict[str, Any], save_path: str):
        """生成综合诊断报告"""
        self.logger.info("=== 生成综合诊断报告 ===")
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_issues': 0,
                    'critical_issues': [],
                    'warnings': [],
                    'recommendations': []
                },
                'detailed_analysis': diagnostics
            }
            
            # 统计问题
            all_issues = []
            for section, data in diagnostics.items():
                if isinstance(data, dict) and 'issues' in data:
                    all_issues.extend(data['issues'])
                if isinstance(data, dict) and 'training_issues' in data:
                    all_issues.extend(data['training_issues'])
                if isinstance(data, dict) and 'gradient_issues' in data:
                    all_issues.extend(data['gradient_issues'])
                if isinstance(data, dict) and 'data_issues' in data:
                    all_issues.extend(data['data_issues'])
                if isinstance(data, dict) and 'potential_issues' in data:
                    all_issues.extend(data['potential_issues'])
            
            report['summary']['total_issues'] = len(all_issues)
            
            # 分类问题严重程度
            for issue in all_issues:
                if any(keyword in issue.lower() for keyword in ['严重', '爆炸', '消失', '异常', '失败']):
                    report['summary']['critical_issues'].append(issue)
                else:
                    report['summary']['warnings'].append(issue)
            
            # 生成建议
            if 'convergence_metrics' in diagnostics.get('training_dynamics', {}):
                conv_metrics = diagnostics['training_dynamics']['convergence_metrics']
                if conv_metrics.get('is_stagnant', False):
                    report['summary']['recommendations'].append("损失停滞：建议降低学习率或增加正则化")
            
            if 'gradient_health' in diagnostics:
                grad_health = diagnostics['gradient_health']
                if grad_health.get('vanishing_exploding', {}).get('vanishing_ratio', 0) > 0.5:
                    report['summary']['recommendations'].append("梯度消失：建议使用残差连接或调整激活函数")
                
                if grad_health.get('vanishing_exploding', {}).get('exploding_ratio', 0) > 0.1:
                    report['summary']['recommendations'].append("梯度爆炸：建议减小学习率或增加梯度裁剪")
            
            if 'model_architecture' in diagnostics:
                arch = diagnostics['model_architecture']
                total_params = arch.get('parameter_count', {}).get('total', 0)
                if total_params > 50e6:  # 50M参数
                    report['summary']['recommendations'].append("模型过大：考虑减少层数或通道数")
            
            # 保存报告 - 处理numpy类型序列化
            def convert_numpy_types(obj):
                """转换numpy类型为Python原生类型"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # 递归转换所有numpy类型
            def deep_convert(obj):
                if isinstance(obj, dict):
                    return {k: deep_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [deep_convert(item) for item in obj]
                else:
                    return convert_numpy_types(obj)
            
            # 转换后保存
            converted_report = deep_convert(report)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(converted_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"综合诊断报告已保存: {save_path}")
            
            # 打印摘要
            print("\n" + "="*60)
            print("模型诊断摘要")
            print("="*60)
            print(f"总问题数: {report['summary']['total_issues']}")
            print(f"严重问题: {len(report['summary']['critical_issues'])}")
            print(f"警告: {len(report['summary']['warnings'])}")
            
            if report['summary']['critical_issues']:
                print("\n严重问题:")
                for issue in report['summary']['critical_issues']:
                    print(f"  - {issue}")
            
            if report['summary']['recommendations']:
                print("\n建议:")
                for rec in report['summary']['recommendations']:
                    print(f"  - {rec}")
            
            print("="*60)
            
            return report
            
        except Exception as e:
            self.logger.error(f"综合报告生成失败: {e}")
            return None

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="综合模型诊断工具")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--log-file", type=str, required=True, help="训练日志文件路径")
    parser.add_argument("--output-dir", type=str, default="diagnostic_reports", help="输出目录")
    parser.add_argument("--model-config", type=str, help="模型配置文件（用于创建模型）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 初始化诊断器
    diagnostic = ComprehensiveModelDiagnostic(args.config)
    
    # 创建示例模型（基于当前配置）
    model_config = {
        'in_channels': 2,
        'out_channels': 2,
        'img_size': 128,
        'patch_size': 4,
        'embed_dim': 120,
        'depths': [3, 3, 9, 3],
        'num_heads': [3, 6, 12, 24],
        'window_size': 8,
        'mlp_ratio': 4.0,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.1,
        'drop_path_rate': 0.25
    }
    
    print("正在创建模型进行架构分析...")
    model = SwinUNet(**model_config)
    
    # 运行诊断
    diagnostics = {}
    
    # 1. 模型架构分析
    print("1. 分析模型架构...")
    diagnostics['model_architecture'] = diagnostic.analyze_model_architecture(model)
    
    # 2. 训练动态分析
    print("2. 分析训练动态...")
    diagnostics['training_dynamics'] = diagnostic.analyze_training_dynamics(args.log_file)
    
    # 3. 梯度健康分析
    print("3. 分析梯度健康...")
    # 创建示例数据
    sample_input = torch.randn(1, 2, 128, 128)
    sample_target = torch.randn(1, 2, 128, 128)
    
    def simple_loss(pred, target):
        return F.mse_loss(pred, target)
    
    diagnostics['gradient_health'] = diagnostic.analyze_gradient_health(
        model, sample_input, sample_target, simple_loss
    )
    
    # 4. 生成可视化
    print("4. 生成架构可视化...")
    viz_path = output_dir / "model_architecture_analysis.png"
    diagnostic.generate_architecture_visualization(model, str(viz_path))
    
    # 5. 生成综合报告
    print("5. 生成综合诊断报告...")
    report_path = output_dir / "comprehensive_diagnostic_report.json"
    report = diagnostic.generate_comprehensive_report(diagnostics, str(report_path))
    
    print(f"\n诊断完成！结果保存在: {output_dir}")
    print(f"可视化: {viz_path}")
    print(f"详细报告: {report_path}")

if __name__ == "__main__":
    main()