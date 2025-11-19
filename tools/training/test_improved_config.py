#!/usr/bin/env python3
"""
测试脚本 - 验证改进配置的效果
快速测试改进措施是否能解决loss停滞问题
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
training_dir = Path(__file__).resolve().parent
for path in (project_root, training_dir):
    p = str(path)
    if p in sys.path:
        try:
            sys.path.remove(p)
        except Exception:
            pass
    sys.path.insert(0, p)

try:
    from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
except Exception:
    from datasets.real_dr_dataset import RealDiffusionReactionDataModule
from models.swin_unet import SwinUNet
from ops.losses import compute_total_loss
from ops.enhanced_losses import compute_enhanced_total_loss
from ops.enhanced_augmentation import AdvancedDataAugmentation
from utils.metrics import compute_metrics
from utils.logger import setup_logger


class TrainingTester:
    """训练测试器 - 快速验证改进配置的效果"""
    
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置输出目录
        output_dir = self.config.get('output_dir', 'runs')
        exp_name = self.config.get('experiment', {}).get('name', 'test_experiment')
        self.output_dir = Path(output_dir) / exp_name / "test"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        exp_name = self.config.get('experiment', {}).get('name', 'test_experiment')
        self.logger = setup_logger(
            name=f"test_{exp_name}",
            log_file=self.output_dir / "test.log",
            level=logging.INFO
        )
        
        self._setup_data()
        self._setup_models()
        self._setup_components()
        
        self.logger.info(f"Training tester initialized for {self.config.experiment.name}")
    
    def _setup_data(self):
        """设置数据"""
        self.data_module = RealDiffusionReactionDataModule(self.config)
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        # 获取一小批数据进行快速测试
        self.sample_batch = next(iter(self.train_loader))
        self.norm_stats = self.data_module.get_normalization_stats()
        
        self.logger.info(f"Data setup complete - Train samples: {len(self.train_loader.dataset)}")
    
    def _setup_models(self):
        """设置模型 - 原始vs改进版本"""
        model_config = self.config.model
        
        # 原始模型（较大）
        self.original_model = SwinUNet(
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            img_size=model_config.img_size,
            patch_size=model_config.patch_size,
            window_size=model_config.window_size,
            depths=[3, 3, 9, 3],      # 原始深度
            num_heads=[3, 6, 12, 24], # 原始头数
            embed_dim=120,            # 原始嵌入维度
            mlp_ratio=model_config.mlp_ratio,
            drop_rate=0.1,            # 原始dropout
            attn_drop_rate=0.1,
            drop_path_rate=0.25
        ).to(self.device)
        
        # 改进模型（较小，更好的初始化）
        self.improved_model = SwinUNet(
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            img_size=model_config.img_size,
            patch_size=model_config.patch_size,
            window_size=model_config.window_size,
            depths=[2, 2, 6, 2],      # 减小深度
            num_heads=[2, 4, 8, 16],  # 减少头数
            embed_dim=96,             # 减少嵌入维度
            mlp_ratio=model_config.mlp_ratio,
            drop_rate=0.2,            # 增加dropout
            attn_drop_rate=0.2,
            drop_path_rate=0.3
        ).to(self.device)
        
        self.logger.info(f"Models setup complete - "
                        f"Original: {sum(p.numel() for p in self.original_model.parameters()):,} params, "
                        f"Improved: {sum(p.numel() for p in self.improved_model.parameters()):,} params")
    
    def _setup_components(self):
        """设置优化器、损失函数等组件"""
        # 优化器配置
        self.original_lr = 0.001  # 原始学习率
        self.improved_lr = 0.0001   # 改进学习率
        
        self.original_optimizer = torch.optim.AdamW(
            self.original_model.parameters(),
            lr=self.original_lr,
            weight_decay=0.0001
        )
        
        self.improved_optimizer = torch.optim.AdamW(
            self.improved_model.parameters(),
            lr=self.improved_lr,
            weight_decay=0.0005
        )
        
        # 数据增强
        if self.config.data.augmentation.enabled:
            self.augmentation = AdvancedDataAugmentation(self.config.data.augmentation)
        else:
            self.augmentation = None
        
        self.logger.info("Components setup complete")
    
    def test_loss_computation(self, model_name: str, model: nn.Module, 
                            loss_fn, num_steps: int = 50) -> Dict[str, List[float]]:
        """测试损失计算和收敛性"""
        model.train()
        
        losses = []
        gradient_norms = []
        learning_rates = []
        
        # 获取测试数据
        x = self.sample_batch['input'].to(self.device)
        target = self.sample_batch['target'].to(self.device)
        
        # 限制批次大小以加快测试
        if x.size(0) > 8:
            x = x[:8]
            target = target[:8]
        
        # 创建优化器（根据模型类型）
        if model_name == 'original':
            optimizer = self.original_optimizer
            lr = self.original_lr
        else:
            optimizer = self.improved_optimizer
            lr = self.improved_lr
        
        self.logger.info(f"Testing {model_name} model for {num_steps} steps...")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 前向传播
            pred = model(x)
            
            # 准备观测数据
            obs_data = {
                'observation': None,
                'baseline': x,
                'h_params': {'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'}
            }
            
            # 计算损失
            losses_dict = loss_fn(
                pred_z=pred,
                target_z=target,
                obs_data=obs_data,
                norm_stats=self.norm_stats,
                config=self.config
            )
            
            loss = losses_dict['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            
            # 记录指标
            losses.append(loss.item())
            gradient_norms.append(grad_norm.item())
            learning_rates.append(lr)
            
            # 每10步记录一次
            if step % 10 == 0:
                self.logger.info(f"  Step {step}: Loss={loss.item():.4f}, "
                               f"Grad Norm={grad_norm.item():.2f}, "
                               f"LR={lr:.2e}")
        
        return {
            'losses': losses,
            'gradient_norms': gradient_norms,
            'learning_rates': learning_rates
        }
    
    def compare_configurations(self) -> Dict[str, Any]:
        """比较不同配置的效果"""
        results = {}
        
        # 测试1: 原始配置 + 原始损失
        self.logger.info("=== Test 1: Original Configuration ===")
        original_loss_fn = lambda **kwargs: compute_total_loss(**kwargs, loss_weights_override={
            'reconstruction': 1.0, 'spectral': 0.0, 'data_consistency': 0.0, 'gradient': 0.0
        })
        
        results['original'] = self.test_loss_computation(
            'original', self.original_model, original_loss_fn, num_steps=100
        )
        
        # 测试2: 改进配置 + 增强损失
        self.logger.info("=== Test 2: Improved Configuration ===")
        improved_loss_fn = compute_enhanced_total_loss
        
        results['improved'] = self.test_loss_computation(
            'improved', self.improved_model, improved_loss_fn, num_steps=100
        )
        
        # 测试3: 混合配置（原始模型 + 改进损失）
        self.logger.info("=== Test 3: Hybrid Configuration ===")
        results['hybrid'] = self.test_loss_computation(
            'hybrid', self.original_model, improved_loss_fn, num_steps=50
        )
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析测试结果"""
        analysis = {}
        
        for config_name, data in results.items():
            losses = data['losses']
            gradient_norms = data['gradient_norms']
            
            # 基本统计
            initial_loss = losses[0]
            final_loss = losses[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100
            
            # 收敛性分析
            # 计算最后20步的平均变化率
            if len(losses) >= 20:
                recent_losses = losses[-20:]
                loss_trend = np.polyfit(range(20), recent_losses, 1)[0]
                is_converging = loss_trend < -0.001
            else:
                loss_trend = 0
                is_converging = False
            
            # 梯度分析
            avg_grad_norm = np.mean(gradient_norms)
            grad_variation = np.std(gradient_norms)
            
            # 稳定性分析
            loss_variance = np.var(losses[-20:]) if len(losses) >= 20 else np.var(losses)
            is_stable = loss_variance < 0.01
            
            analysis[config_name] = {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'loss_reduction_pct': loss_reduction,
                'loss_trend': loss_trend,
                'is_converging': is_converging,
                'avg_gradient_norm': avg_grad_norm,
                'gradient_variation': grad_variation,
                'loss_variance': loss_variance,
                'is_stable': is_stable,
                'convergence_score': self._calculate_convergence_score(losses)
            }
            
            self.logger.info(f"{config_name} Analysis:")
            self.logger.info(f"  Initial Loss: {initial_loss:.4f}")
            self.logger.info(f"  Final Loss: {final_loss:.4f}")
            self.logger.info(f"  Loss Reduction: {loss_reduction:.1f}%")
            self.logger.info(f"  Converging: {is_converging} (trend: {loss_trend:.4f})")
            self.logger.info(f"  Stable: {is_stable} (variance: {loss_variance:.4f})")
        
        return analysis
    
    def _calculate_convergence_score(self, losses: List[float]) -> float:
        """计算收敛性评分"""
        if len(losses) < 10:
            return 0.0
        
        # 计算损失下降速度
        total_reduction = (losses[0] - losses[-1]) / losses[0]
        
        # 计算稳定性（最后20步的方差）
        recent_variance = np.var(losses[-min(20, len(losses)):])
        
        # 计算趋势一致性
        if len(losses) >= 20:
            recent_trend = np.polyfit(range(20), losses[-20:], 1)[0]
            trend_score = max(0, -recent_trend * 100)  # 负趋势（下降）得分高
        else:
            trend_score = 0
        
        # 综合评分
        convergence_score = (
            total_reduction * 50 +        # 下降幅度权重
            (1 - min(recent_variance, 1.0)) * 30 +  # 稳定性权重
            trend_score * 20                # 趋势权重
        )
        
        return min(convergence_score, 100.0)
    
    def plot_comparison(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Configuration Comparison', fontsize=16)
        
        # 损失曲线对比
        ax = axes[0, 0]
        for config_name, data in results.items():
            losses = data['losses']
            ax.plot(losses, label=f"{config_name} (score: {analysis[config_name]['convergence_score']:.1f})", 
                   alpha=0.8, linewidth=2)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 损失下降率
        ax = axes[0, 1]
        configs = list(analysis.keys())
        reductions = [analysis[c]['loss_reduction_pct'] for c in configs]
        bars = ax.bar(configs, reductions, alpha=0.7)
        ax.set_ylabel('Loss Reduction (%)')
        ax.set_title('Loss Reduction Comparison')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, reduction in zip(bars, reductions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{reduction:.1f}%', ha='center', va='bottom')
        
        # 收敛性评分
        ax = axes[1, 0]
        scores = [analysis[c]['convergence_score'] for c in configs]
        bars = ax.bar(configs, scores, alpha=0.7, color=['red', 'green', 'blue'])
        ax.set_ylabel('Convergence Score')
        ax.set_title('Convergence Score Comparison')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}', ha='center', va='bottom')
        
        # 梯度范数对比
        ax = axes[1, 1]
        for config_name, data in results.items():
            gradient_norms = data['gradient_norms']
            ax.plot(gradient_norms, label=config_name, alpha=0.8, linewidth=2)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.output_dir / "configuration_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison plot saved: {plot_path}")
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 比较不同配置的表现
        best_config = max(analysis.keys(), key=lambda x: analysis[x]['convergence_score'])
        
        recommendations.append(f"最佳配置: {best_config} (收敛评分: {analysis[best_config]['convergence_score']:.1f})")
        
        # 分析具体问题
        for config_name, metrics in analysis.items():
            if not metrics['is_converging']:
                recommendations.append(f"{config_name}: 损失函数未收敛，建议降低学习率或增加正则化")
            
            if not metrics['is_stable']:
                recommendations.append(f"{config_name}: 训练不稳定，建议减小批次大小或增加梯度裁剪")
            
            if metrics['loss_reduction'] < 10:
                recommendations.append(f"{config_name}: 损失下降缓慢，建议检查模型架构或数据预处理")
            
            if metrics['avg_gradient_norm'] > 10:
                recommendations.append(f"{config_name}: 梯度过大，建议降低学习率或增加正则化")
            elif metrics['avg_gradient_norm'] < 0.1:
                recommendations.append(f"{config_name}: 梯度过小，可能存在梯度消失问题")
        
        return recommendations
    
    def run_full_test(self):
        """运行完整测试"""
        self.logger.info("=== Starting Training Configuration Test ===")
        
        # 比较不同配置
        results = self.compare_configurations()
        
        # 分析结果
        analysis = self.analyze_results(results)
        
        # 绘制对比图
        self.plot_comparison(results, analysis)
        
        # 生成建议
        recommendations = self.generate_recommendations(analysis)
        
        # 保存完整报告
        report = {
            'experiment_name': self.config.experiment.name,
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'analysis': analysis,
            'recommendations': recommendations,
            'best_configuration': max(analysis.keys(), key=lambda x: analysis[x]['convergence_score'])
        }
        
        report_path = self.output_dir / "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 输出总结
        self.logger.info("=== Test Summary ===")
        self.logger.info(f"Best configuration: {report['best_configuration']}")
        scores_str = ', '.join([f"{k}: {v['convergence_score']:.1f}" for k, v in analysis.items()])
        self.logger.info(f"Convergence scores: {scores_str}")
        self.logger.info("Recommendations:")
        for rec in recommendations:
            self.logger.info(f"  - {rec}")
        
        self.logger.info(f"Full test report saved: {report_path}")
        
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Training Configuration Test')
    parser.add_argument('--config', type=str, 
                       default='/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/train/ar_training_config_100samples_optimized.yaml',
                       help='改进配置文件路径')
    
    args = parser.parse_args()
    
    # 运行测试
    tester = TrainingTester(args.config)
    report = tester.run_full_test()
    
    print(f"\n=== Test Results ===")
    print(f"Best configuration: {report['best_configuration']}")
    original_score = report['analysis']['original']['convergence_score']
    improved_score = report['analysis']['improved']['convergence_score']
    print(f"Convergence improvement: Original: {original_score:.1f} -> Improved: {improved_score:.1f}")


if __name__ == "__main__":
    main()