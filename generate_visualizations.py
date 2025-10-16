#!/usr/bin/env python3
"""
可视化生成脚本
按照技术架构文档标准生成完整的模型对比可视化图表
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class VisualizationGenerator:
    """可视化图表生成器"""
    
    def __init__(self, results_dir: str, metrics_dir: str = None):
        self.results_dir = Path(results_dir)
        self.metrics_dir = Path(metrics_dir) if metrics_dir else self.results_dir / "evaluation_metrics"
        
        # 创建可视化输出目录
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # 加载指标数据
        self.metrics_data = self._load_metrics_data()
        
        logger.info(f"初始化可视化生成器，结果目录: {self.results_dir}")
        logger.info(f"可视化输出目录: {self.viz_dir}")
    
    def _load_metrics_data(self) -> Dict[str, Any]:
        """加载指标数据"""
        metrics_file = self.metrics_dir / "all_metrics.json"
        
        if not metrics_file.exists():
            logger.warning(f"指标文件不存在: {metrics_file}")
            return {}
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载 {len(data)} 个模型的指标数据")
            return data
        except Exception as e:
            logger.error(f"加载指标数据失败: {e}")
            return {}
    
    def generate_performance_comparison(self):
        """生成性能对比图表"""
        if not self.metrics_data:
            logger.warning("没有可用的指标数据")
            return
        
        logger.info("生成性能对比图表...")
        
        # 准备数据
        models = list(self.metrics_data.keys())
        rel_l2_values = [self.metrics_data[m]['performance']['rel_l2'] for m in models]
        mae_values = [self.metrics_data[m]['performance']['mae'] for m in models]
        psnr_values = [self.metrics_data[m]['performance']['psnr'] for m in models]
        ssim_values = [self.metrics_data[m]['performance']['ssim'] for m in models]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能指标对比', fontsize=16, fontweight='bold')
        
        # Rel-L2对比
        axes[0, 0].bar(models, rel_l2_values, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Relative L2 Error (越小越好)', fontweight='bold')
        axes[0, 0].set_ylabel('Rel-L2')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE对比
        axes[0, 1].bar(models, mae_values, color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('Mean Absolute Error (越小越好)', fontweight='bold')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # PSNR对比
        axes[1, 0].bar(models, psnr_values, color='lightgreen', alpha=0.8)
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio (越大越好)', fontweight='bold')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # SSIM对比
        axes[1, 1].bar(models, ssim_values, color='gold', alpha=0.8)
        axes[1, 1].set_title('Structural Similarity Index (越大越好)', fontweight='bold')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = self.viz_dir / "performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存性能对比图表: {save_path}")
    
    def generate_resource_comparison(self):
        """生成资源消耗对比图表"""
        if not self.metrics_data:
            return
        
        logger.info("生成资源消耗对比图表...")
        
        # 准备数据
        models = list(self.metrics_data.keys())
        params_values = [self.metrics_data[m]['resources']['params_m'] for m in models]
        flops_values = [self.metrics_data[m]['resources']['flops_g'] for m in models]
        memory_values = [self.metrics_data[m]['resources']['memory_gb'] for m in models]
        latency_values = [self.metrics_data[m]['resources']['latency_ms'] for m in models]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型资源消耗对比', fontsize=16, fontweight='bold')
        
        # 参数量对比
        axes[0, 0].bar(models, params_values, color='steelblue', alpha=0.8)
        axes[0, 0].set_title('模型参数量', fontweight='bold')
        axes[0, 0].set_ylabel('Parameters (M)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # FLOPs对比
        axes[0, 1].bar(models, flops_values, color='darkorange', alpha=0.8)
        axes[0, 1].set_title('计算复杂度 (FLOPs@256²)', fontweight='bold')
        axes[0, 1].set_ylabel('FLOPs (G)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 显存使用对比
        axes[1, 0].bar(models, memory_values, color='mediumseagreen', alpha=0.8)
        axes[1, 0].set_title('显存峰值使用', fontweight='bold')
        axes[1, 0].set_ylabel('Memory (GB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 推理延迟对比
        axes[1, 1].bar(models, latency_values, color='mediumpurple', alpha=0.8)
        axes[1, 1].set_title('推理延迟', fontweight='bold')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = self.viz_dir / "resource_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存资源消耗对比图表: {save_path}")
    
    def generate_radar_chart(self):
        """生成雷达图对比"""
        if not self.metrics_data:
            return
        
        logger.info("生成雷达图对比...")
        
        # 选择前5个模型进行雷达图对比
        models = list(self.metrics_data.keys())[:5]
        
        # 准备雷达图数据（标准化到0-1范围）
        categories = ['Performance', 'Efficiency', 'Speed', 'Memory', 'Accuracy']
        
        fig = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, model in enumerate(models):
            metrics = self.metrics_data[model]
            
            # 计算标准化指标（越大越好）
            performance = 1.0 / (1.0 + metrics['performance']['rel_l2'])  # 性能
            efficiency = 1.0 / (1.0 + metrics['resources']['params_m'] / 50.0)  # 效率
            speed = 1.0 / (1.0 + metrics['resources']['latency_ms'] / 100.0)  # 速度
            memory = 1.0 / (1.0 + metrics['resources']['memory_gb'] / 10.0)  # 内存
            accuracy = metrics['performance']['ssim'] if metrics['performance']['ssim'] > 0 else 0.8  # 准确性
            
            values = [performance, efficiency, speed, memory, accuracy]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model,
                line_color=colors[i],
                fillcolor=colors[i],
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="模型综合性能雷达图对比",
            font=dict(size=12)
        )
        
        # 保存雷达图
        save_path = self.viz_dir / "radar_comparison.html"
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        
        logger.info(f"保存雷达图对比: {save_path}")
    
    def generate_efficiency_scatter(self):
        """生成效率散点图"""
        if not self.metrics_data:
            return
        
        logger.info("生成效率散点图...")
        
        # 准备数据
        models = list(self.metrics_data.keys())
        rel_l2_values = [self.metrics_data[m]['performance']['rel_l2'] for m in models]
        params_values = [self.metrics_data[m]['resources']['params_m'] for m in models]
        latency_values = [self.metrics_data[m]['resources']['latency_ms'] for m in models]
        
        # 创建散点图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 性能 vs 参数量
        scatter1 = ax1.scatter(params_values, rel_l2_values, 
                              s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
        ax1.set_xlabel('Parameters (M)')
        ax1.set_ylabel('Rel-L2 Error')
        ax1.set_title('性能 vs 参数量 (左下角更好)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 添加模型标签
        for i, model in enumerate(models):
            ax1.annotate(model, (params_values[i], rel_l2_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 性能 vs 延迟
        scatter2 = ax2.scatter(latency_values, rel_l2_values, 
                              s=100, alpha=0.7, c=range(len(models)), cmap='plasma')
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Rel-L2 Error')
        ax2.set_title('性能 vs 推理延迟 (左下角更好)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加模型标签
        for i, model in enumerate(models):
            ax2.annotate(model, (latency_values[i], rel_l2_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # 保存散点图
        save_path = self.viz_dir / "efficiency_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存效率散点图: {save_path}")
    
    def generate_frequency_analysis(self):
        """生成频域分析图表"""
        if not self.metrics_data:
            return
        
        logger.info("生成频域分析图表...")
        
        # 准备数据
        models = list(self.metrics_data.keys())
        frmse_low = [self.metrics_data[m]['frequency']['frmse_low'] for m in models]
        frmse_mid = [self.metrics_data[m]['frequency']['frmse_mid'] for m in models]
        frmse_high = [self.metrics_data[m]['frequency']['frmse_high'] for m in models]
        
        # 创建堆叠柱状图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(models))
        width = 0.6
        
        p1 = ax.bar(x, frmse_low, width, label='Low Freq', color='lightblue', alpha=0.8)
        p2 = ax.bar(x, frmse_mid, width, bottom=frmse_low, label='Mid Freq', color='orange', alpha=0.8)
        p3 = ax.bar(x, frmse_high, width, 
                   bottom=np.array(frmse_low) + np.array(frmse_mid), 
                   label='High Freq', color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Frequency RMSE')
        ax.set_title('频域重建误差分析 (按频率分量)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存频域分析图
        save_path = self.viz_dir / "frequency_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存频域分析图表: {save_path}")
    
    def generate_training_curves(self):
        """生成训练曲线对比"""
        logger.info("生成训练曲线对比...")
        
        # 尝试从各个模型目录加载训练日志
        training_data = {}
        
        for model_dir in self.results_dir.iterdir():
            if model_dir.is_dir() and model_dir.name not in ['evaluation_metrics', 'visualizations', 'summary']:
                model_name = model_dir.name
                log_file = model_dir / "train.log"
                
                if log_file.exists():
                    curves = self._parse_training_curves(log_file)
                    if curves:
                        training_data[model_name] = curves
        
        if not training_data:
            logger.warning("没有找到训练曲线数据")
            return
        
        # 绘制训练曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练损失曲线
        for model_name, curves in training_data.items():
            if 'train_loss' in curves and curves['train_loss']:
                epochs = range(len(curves['train_loss']))
                ax1.plot(epochs, curves['train_loss'], label=model_name, alpha=0.8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('训练损失曲线对比', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 验证损失曲线
        for model_name, curves in training_data.items():
            if 'val_loss' in curves and curves['val_loss']:
                epochs = range(len(curves['val_loss']))
                ax2.plot(epochs, curves['val_loss'], label=model_name, alpha=0.8)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('验证损失曲线对比', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存训练曲线
        save_path = self.viz_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存训练曲线对比: {save_path}")
    
    def _parse_training_curves(self, log_file: Path) -> Dict[str, List[float]]:
        """解析训练曲线数据"""
        curves = {
            'train_loss': [],
            'val_loss': [],
            'val_rel_l2': []
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                if "Epoch" in line and "Train Loss" in line and "Val Loss" in line:
                    # 解析epoch结果行
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Loss:" and i > 0 and "Train" in parts[i-1] and i+1 < len(parts):
                            try:
                                train_loss = float(parts[i+1])
                                curves['train_loss'].append(train_loss)
                            except ValueError:
                                pass
                        elif part == "Loss:" and i > 0 and "Val" in parts[i-1] and i+1 < len(parts):
                            try:
                                val_loss = float(parts[i+1])
                                curves['val_loss'].append(val_loss)
                            except ValueError:
                                pass
                        elif part == "Rel-L2:" and i+1 < len(parts):
                            try:
                                rel_l2 = float(parts[i+1])
                                curves['val_rel_l2'].append(rel_l2)
                            except ValueError:
                                pass
        
        except Exception as e:
            logger.warning(f"解析训练曲线失败: {e}")
        
        return curves
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        logger.info("开始生成所有可视化图表...")
        
        try:
            self.generate_performance_comparison()
            self.generate_resource_comparison()
            self.generate_radar_chart()
            self.generate_efficiency_scatter()
            self.generate_frequency_analysis()
            self.generate_training_curves()
            
            logger.info("✓ 所有可视化图表生成完成！")
            logger.info(f"图表保存在: {self.viz_dir}")
            
        except Exception as e:
            logger.error(f"生成可视化图表时出错: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成模型对比可视化图表")
    parser.add_argument("--results_dir", type=str, default="runs/batch_training_results",
                       help="训练结果目录")
    parser.add_argument("--metrics_dir", type=str, default=None,
                       help="指标数据目录")
    
    args = parser.parse_args()
    
    # 创建可视化生成器
    viz_generator = VisualizationGenerator(args.results_dir, args.metrics_dir)
    
    # 生成所有可视化图表
    viz_generator.generate_all_visualizations()


if __name__ == "__main__":
    main()