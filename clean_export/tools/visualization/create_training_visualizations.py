#!/usr/bin/env python3
"""
创建训练过程的详细可视化
包括损失曲线、指标变化、学习率调度等
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizationGenerator:
    """训练过程可视化生成器"""
    
    def __init__(self, output_dir: str = "runs/training_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 数据存储
        self.training_data = {}
        
        self.logger.info(f"训练可视化生成器初始化完成，输出目录: {self.output_dir}")
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_training_data(self):
        """收集训练数据"""
        self.logger.info("收集训练数据...")
        
        # 查找训练结果目录
        training_dirs = [
            "runs/temporal_nar_300epochs",
            "runs/temporal_nar_100epochs",
            "runs/temporal_nar_optimized"
        ]
        
        for train_dir in training_dirs:
            train_path = Path(train_dir)
            if train_path.exists():
                self._collect_single_training_data(train_path)
    
    def _collect_single_training_data(self, train_path: Path):
        """收集单个训练的数据"""
        try:
            # 查找具体的实验目录
            exp_dirs = [d for d in train_path.iterdir() if d.is_dir()]
            
            for exp_dir in exp_dirs:
                exp_name = exp_dir.name
                self.logger.info(f"处理实验: {exp_name}")
                
                # 读取训练历史
                history_file = exp_dir / "training_history.json"
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    self.training_data[exp_name] = {
                        'history': history,
                        'config_path': exp_dir / "config_snapshot.yaml",
                        'exp_dir': exp_dir
                    }
                    
                    self.logger.info(f"成功收集训练数据: {exp_name}")
                
        except Exception as e:
            self.logger.error(f"收集训练数据失败 {train_path}: {e}")
    
    def create_comprehensive_loss_plot(self):
        """创建综合损失曲线图"""
        self.logger.info("创建综合损失曲线图...")
        
        if not self.training_data:
            self.logger.warning("没有训练数据")
            return
        
        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        
        # 主损失曲线 (占据左上大部分)
        ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
        
        # 子图
        ax_val = plt.subplot2grid((3, 4), (0, 3))
        ax_final = plt.subplot2grid((3, 4), (1, 3))
        ax_convergence = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        ax_stats = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.training_data)))
        
        # 主损失曲线
        for i, (exp_name, data) in enumerate(self.training_data.items()):
            history = data['history']
            color = colors[i]
            
            if 'train_losses' in history:
                epochs = range(1, len(history['train_losses']) + 1)
                train_losses = history['train_losses']
                
                # 平滑处理
                if len(train_losses) > 10:
                    window = min(10, len(train_losses) // 10)
                    train_losses_smooth = pd.Series(train_losses).rolling(window=window, center=True).mean()
                    ax_main.plot(epochs, train_losses_smooth, color=color, linewidth=2, 
                               label=f'{exp_name} (训练)', alpha=0.8)
                    ax_main.plot(epochs, train_losses, color=color, linewidth=0.5, alpha=0.3)
                else:
                    ax_main.plot(epochs, train_losses, color=color, linewidth=2, 
                               label=f'{exp_name} (训练)', alpha=0.8)
        
        ax_main.set_title('训练损失曲线对比', fontsize=16, fontweight='bold')
        ax_main.set_xlabel('Epoch', fontsize=12)
        ax_main.set_ylabel('Training Loss', fontsize=12)
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_yscale('log')
        
        # 验证损失对比
        val_losses = []
        exp_names = []
        for exp_name, data in self.training_data.items():
            history = data['history']
            if 'best_val_loss' in history:
                val_losses.append(history['best_val_loss'])
                exp_names.append(exp_name.split('-')[0])  # 简化名称
        
        if val_losses:
            bars = ax_val.bar(range(len(exp_names)), val_losses, color=colors[:len(val_losses)])
            ax_val.set_title('最佳验证损失', fontsize=12, fontweight='bold')
            ax_val.set_xticks(range(len(exp_names)))
            ax_val.set_xticklabels(exp_names, rotation=45, ha='right')
            ax_val.set_ylabel('Best Val Loss')
            
            # 添加数值标签
            for bar, val in zip(bars, val_losses):
                height = bar.get_height()
                ax_val.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 最终训练损失对比
        final_losses = []
        final_exp_names = []
        for exp_name, data in self.training_data.items():
            history = data['history']
            if 'train_losses' in history and history['train_losses']:
                final_losses.append(history['train_losses'][-1])
                final_exp_names.append(exp_name.split('-')[0])
        
        if final_losses:
            bars = ax_final.bar(range(len(final_exp_names)), final_losses, color=colors[:len(final_losses)])
            ax_final.set_title('最终训练损失', fontsize=12, fontweight='bold')
            ax_final.set_xticks(range(len(final_exp_names)))
            ax_final.set_xticklabels(final_exp_names, rotation=45, ha='right')
            ax_final.set_ylabel('Final Train Loss')
            
            # 添加数值标签
            for bar, val in zip(bars, final_losses):
                height = bar.get_height()
                ax_final.text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 收敛分析
        for i, (exp_name, data) in enumerate(self.training_data.items()):
            history = data['history']
            if 'train_losses' in history and len(history['train_losses']) > 50:
                losses = history['train_losses']
                # 计算收敛速度 (损失下降率)
                convergence_rate = []
                window = 10
                for j in range(window, len(losses)):
                    rate = (losses[j-window] - losses[j]) / losses[j-window]
                    convergence_rate.append(rate)
                
                epochs = range(window+1, len(losses)+1)
                ax_convergence.plot(epochs, convergence_rate, color=colors[i], 
                                  label=f'{exp_name.split("-")[0]}', alpha=0.7)
        
        ax_convergence.set_title('收敛速度分析', fontsize=12, fontweight='bold')
        ax_convergence.set_xlabel('Epoch')
        ax_convergence.set_ylabel('损失下降率')
        ax_convergence.legend()
        ax_convergence.grid(True, alpha=0.3)
        
        # 训练统计
        stats_data = []
        for exp_name, data in self.training_data.items():
            history = data['history']
            if 'train_losses' in history:
                losses = history['train_losses']
                stats_data.append({
                    'Experiment': exp_name.split('-')[0],
                    'Total Epochs': len(losses),
                    'Min Loss': min(losses),
                    'Final Loss': losses[-1],
                    'Std Dev': np.std(losses[-50:]) if len(losses) >= 50 else np.std(losses)
                })
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            ax_stats.axis('tight')
            ax_stats.axis('off')
            table = ax_stats.table(cellText=df.values, colLabels=df.columns,
                                  cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax_stats.set_title('训练统计', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存图片
        img_path = self.output_dir / "comprehensive_training_analysis.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"综合训练分析图已保存: {img_path}")
        return img_path
    
    def create_loss_components_analysis(self):
        """创建损失组件分析图"""
        self.logger.info("创建损失组件分析图...")
        
        # 这里可以添加更详细的损失组件分析
        # 如果训练历史中包含各个损失组件的数据
        pass
    
    def create_training_dynamics_plot(self):
        """创建训练动态图"""
        self.logger.info("创建训练动态图...")
        
        if not self.training_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('训练动态分析', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.training_data)))
        
        for i, (exp_name, data) in enumerate(self.training_data.items()):
            history = data['history']
            color = colors[i]
            
            if 'train_losses' in history:
                losses = history['train_losses']
                epochs = range(1, len(losses) + 1)
                
                # 1. 损失变化率
                if len(losses) > 1:
                    loss_changes = [0] + [losses[j] - losses[j-1] for j in range(1, len(losses))]
                    axes[0, 0].plot(epochs, loss_changes, color=color, alpha=0.7, 
                                   label=f'{exp_name.split("-")[0]}')
                
                # 2. 损失平滑度 (移动标准差)
                if len(losses) > 10:
                    window = 10
                    smoothness = []
                    for j in range(window, len(losses)):
                        std = np.std(losses[j-window:j])
                        smoothness.append(std)
                    
                    smooth_epochs = range(window+1, len(losses)+1)
                    axes[0, 1].plot(smooth_epochs, smoothness, color=color, alpha=0.7,
                                   label=f'{exp_name.split("-")[0]}')
                
                # 3. 累积损失改善
                if len(losses) > 1:
                    cumulative_improvement = []
                    initial_loss = losses[0]
                    for loss in losses:
                        improvement = (initial_loss - loss) / initial_loss
                        cumulative_improvement.append(improvement)
                    
                    axes[1, 0].plot(epochs, cumulative_improvement, color=color, alpha=0.7,
                                   label=f'{exp_name.split("-")[0]}')
                
                # 4. 训练效率 (每个epoch的损失改善)
                if len(losses) > 1:
                    efficiency = []
                    for j in range(1, len(losses)):
                        eff = max(0, losses[j-1] - losses[j]) / losses[j-1]
                        efficiency.append(eff)
                    
                    eff_epochs = range(2, len(losses)+1)
                    axes[1, 1].plot(eff_epochs, efficiency, color=color, alpha=0.7,
                                   label=f'{exp_name.split("-")[0]}')
        
        # 设置子图标题和标签
        axes[0, 0].set_title('损失变化率')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss Change')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('训练平滑度')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Std (10-epoch window)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('累积损失改善')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cumulative Improvement')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('训练效率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Improvement Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        img_path = self.output_dir / "training_dynamics_analysis.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练动态分析图已保存: {img_path}")
        return img_path
    
    def create_epoch_comparison_plot(self):
        """创建不同epoch数实验的对比图"""
        self.logger.info("创建epoch对比图...")
        
        if len(self.training_data) < 2:
            self.logger.warning("实验数量不足，跳过epoch对比")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('不同训练长度实验对比', fontsize=16, fontweight='bold')
        
        # 提取实验信息
        experiments = []
        for exp_name, data in self.training_data.items():
            history = data['history']
            if 'train_losses' in history:
                experiments.append({
                    'name': exp_name,
                    'epochs': len(history['train_losses']),
                    'final_loss': history['train_losses'][-1],
                    'best_val_loss': history.get('best_val_loss', None),
                    'losses': history['train_losses']
                })
        
        # 按epoch数排序
        experiments.sort(key=lambda x: x['epochs'])
        
        # 1. 最终性能对比
        names = [exp['name'].split('-')[0] for exp in experiments]
        epochs = [exp['epochs'] for exp in experiments]
        final_losses = [exp['final_loss'] for exp in experiments]
        
        bars = ax1.bar(names, final_losses, color=plt.cm.viridis(np.linspace(0, 1, len(experiments))))
        ax1.set_title('最终训练损失 vs 训练轮数')
        ax1.set_ylabel('Final Training Loss')
        ax1.set_xlabel('Experiment')
        
        # 添加epoch数标签
        for bar, epoch in zip(bars, epochs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{epoch} epochs', ha='center', va='bottom', fontsize=10)
        
        # 2. 训练效率对比 (损失/epoch)
        efficiency = [exp['final_loss'] / exp['epochs'] for exp in experiments]
        bars = ax2.bar(names, efficiency, color=plt.cm.plasma(np.linspace(0, 1, len(experiments))))
        ax2.set_title('训练效率 (损失改善/epoch)')
        ax2.set_ylabel('Loss Reduction per Epoch')
        ax2.set_xlabel('Experiment')
        
        plt.tight_layout()
        
        # 保存图片
        img_path = self.output_dir / "epoch_comparison_analysis.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Epoch对比分析图已保存: {img_path}")
        return img_path
    
    def run(self):
        """运行完整的可视化生成流程"""
        self.logger.info("开始生成训练可视化...")
        
        # 收集数据
        self.collect_training_data()
        
        if not self.training_data:
            self.logger.warning("没有找到训练数据")
            return
        
        # 生成各种可视化
        visualizations = []
        
        # 综合损失分析
        img_path = self.create_comprehensive_loss_plot()
        if img_path:
            visualizations.append(img_path)
        
        # 训练动态分析
        img_path = self.create_training_dynamics_plot()
        if img_path:
            visualizations.append(img_path)
        
        # Epoch对比分析
        img_path = self.create_epoch_comparison_plot()
        if img_path:
            visualizations.append(img_path)
        
        self.logger.info(f"训练可视化生成完成! 共生成 {len(visualizations)} 个图表")
        return visualizations

def main():
    """主函数"""
    generator = TrainingVisualizationGenerator()
    visualizations = generator.run()
    
    print(f"\n🎨 训练可视化已生成!")
    print(f"📊 生成图表数: {len(visualizations) if visualizations else 0}")
    if visualizations:
        for viz in visualizations:
            print(f"  - {viz}")

if __name__ == "__main__":
    main()