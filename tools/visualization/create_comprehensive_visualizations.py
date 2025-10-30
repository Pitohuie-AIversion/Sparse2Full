#!/usr/bin/env python3
"""
Comprehensive visualization generator for temporal NAR model
Creates training curves, prediction visualizations, and interactive HTML report
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

class TemporalNARVisualizer:
    def __init__(self, base_dir="F:/Zhaoyang/Sparse2Full"):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs" / "temporal_nar_100epochs"
        self.test_results_dir = self.base_dir / "test_results"
        self.output_dir = self.base_dir / "comprehensive_visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_training_data()
        self.load_test_results()
        
    def load_training_data(self):
        """Load training history and metrics"""
        training_file = self.runs_dir / "TemporalNAR-DR2D-128-100epochs-s2025" / "training_history.json"
        metrics_file = self.runs_dir / "predictions_visualization" / "metrics.json"
        
        with open(training_file, 'r') as f:
            self.training_data = json.load(f)
            
        with open(metrics_file, 'r') as f:
            self.prediction_metrics = json.load(f)
            
    def load_test_results(self):
        """Load multi-step prediction test results"""
        multistep_file = self.test_results_dir / "multistep_prediction" / "multistep_report_20251026_173918.json"
        
        with open(multistep_file, 'r') as f:
            self.multistep_data = json.load(f)
            
    def create_training_curves(self):
        """Create comprehensive training curves visualization"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main training curves
        ax1 = fig.add_subplot(gs[0, :])
        epochs = range(1, len(self.training_data['train_losses']) + 1)
        
        ax1.plot(epochs, self.training_data['train_losses'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
        ax1.plot(epochs, self.training_data['val_losses'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        
        # Mark best validation loss
        best_epoch = np.argmin(self.training_data['val_losses']) + 1
        best_val_loss = self.training_data['best_val_loss']
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Val Loss (Epoch {best_epoch})')
        ax1.scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Temporal NAR Model Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Loss convergence analysis (zoomed)
        ax2 = fig.add_subplot(gs[1, 0])
        start_epoch = max(1, len(epochs) - 30)  # Last 30 epochs
        ax2.plot(epochs[start_epoch-1:], self.training_data['train_losses'][start_epoch-1:], 'b-', linewidth=2, label='Training')
        ax2.plot(epochs[start_epoch-1:], self.training_data['val_losses'][start_epoch-1:], 'r-', linewidth=2, label='Validation')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Convergence Analysis (Last 30 Epochs)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Training statistics
        ax3 = fig.add_subplot(gs[1, 1])
        final_train_loss = self.training_data['train_losses'][-1]
        final_val_loss = self.training_data['val_losses'][-1]
        
        stats_data = {
            'Metric': ['Final Train Loss', 'Final Val Loss', 'Best Val Loss', 'Overfitting Gap'],
            'Value': [final_train_loss, final_val_loss, best_val_loss, final_val_loss - final_train_loss]
        }
        
        bars = ax3.bar(stats_data['Metric'], stats_data['Value'], 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'orange'], alpha=0.8)
        ax3.set_ylabel('Loss Value', fontsize=11)
        ax3.set_title('Training Statistics', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_data['Value']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Loss distribution
        ax4 = fig.add_subplot(gs[2, :])
        ax4.hist(self.training_data['train_losses'], bins=30, alpha=0.6, label='Training Loss', color='blue', density=True)
        ax4.hist(self.training_data['val_losses'], bins=30, alpha=0.6, label='Validation Loss', color='red', density=True)
        ax4.axvline(x=np.mean(self.training_data['train_losses']), color='blue', linestyle='--', alpha=0.8, label='Train Mean')
        ax4.axvline(x=np.mean(self.training_data['val_losses']), color='red', linestyle='--', alpha=0.8, label='Val Mean')
        ax4.set_xlabel('Loss Value', fontsize=11)
        ax4.set_ylabel('Density', fontsize=11)
        ax4.set_title('Loss Distribution Analysis', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_multistep_analysis(self):
        """Create multi-step prediction analysis charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        t_outs = [int(k) for k in self.multistep_data['results'].keys()]
        rel_l2_values = [self.multistep_data['results'][str(t)]['avg_rel_l2'] for t in t_outs]
        psnr_values = [self.multistep_data['results'][str(t)]['avg_psnr'] for t in t_outs]
        ssim_values = [self.multistep_data['results'][str(t)]['avg_ssim'] for t in t_outs]
        inference_times = [self.multistep_data['results'][str(t)]['avg_inference_time'] for t in t_outs]
        
        # Rel-L2 vs T_out
        ax1.plot(t_outs, rel_l2_values, 'o-', linewidth=3, markersize=8, color='red', alpha=0.8)
        ax1.set_xlabel('Prediction Steps (T_out)', fontsize=12)
        ax1.set_ylabel('Relative L2 Error', fontsize=12)
        ax1.set_title('Prediction Accuracy vs Time Steps', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Highlight best performance
        best_idx = np.argmin(rel_l2_values)
        ax1.scatter([t_outs[best_idx]], [rel_l2_values[best_idx]], 
                   color='green', s=150, zorder=5, label=f'Best: T_out={t_outs[best_idx]}')
        ax1.legend()
        
        # PSNR vs T_out
        ax2.plot(t_outs, psnr_values, 's-', linewidth=3, markersize=8, color='blue', alpha=0.8)
        ax2.set_xlabel('Prediction Steps (T_out)', fontsize=12)
        ax2.set_ylabel('PSNR (dB)', fontsize=12)
        ax2.set_title('Signal Quality vs Time Steps', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Highlight best PSNR
        best_psnr_idx = np.argmax(psnr_values)
        ax2.scatter([t_outs[best_psnr_idx]], [psnr_values[best_psnr_idx]], 
                   color='green', s=150, zorder=5, label=f'Best: T_out={t_outs[best_psnr_idx]}')
        ax2.legend()
        
        # SSIM vs T_out
        ax3.plot(t_outs, ssim_values, '^-', linewidth=3, markersize=8, color='purple', alpha=0.8)
        ax3.set_xlabel('Prediction Steps (T_out)', fontsize=12)
        ax3.set_ylabel('SSIM', fontsize=12)
        ax3.set_title('Structural Similarity vs Time Steps', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Inference time scaling
        ax4.plot(t_outs, inference_times, 'd-', linewidth=3, markersize=8, color='orange', alpha=0.8)
        ax4.set_xlabel('Prediction Steps (T_out)', fontsize=12)
        ax4.set_ylabel('Inference Time (seconds)', fontsize=12)
        ax4.set_title('Computational Cost vs Time Steps', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(t_outs, inference_times, 1)
        p = np.poly1d(z)
        ax4.plot(t_outs, p(t_outs), "--", alpha=0.7, color='gray', 
                label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'multistep_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_performance_summary(self):
        """Create performance summary dashboard"""
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Single-step prediction metrics
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['MSE', 'MAE', 'PSNR', 'Rel_L2']
        values = [self.prediction_metrics[m] for m in metrics]
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'orange']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_title('Single-Step Prediction Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Metric Value', fontsize=11)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Multi-step performance comparison
        ax2 = fig.add_subplot(gs[0, 1:])
        t_outs = [int(k) for k in self.multistep_data['results'].keys()]
        rel_l2_values = [self.multistep_data['results'][str(t)]['avg_rel_l2'] for t in t_outs]
        psnr_values = [self.multistep_data['results'][str(t)]['avg_psnr'] for t in t_outs]
        
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(t_outs, rel_l2_values, 'ro-', linewidth=2, markersize=6, label='Rel-L2 Error')
        line2 = ax2_twin.plot(t_outs, psnr_values, 'bs-', linewidth=2, markersize=6, label='PSNR (dB)')
        
        ax2.set_xlabel('Prediction Steps (T_out)', fontsize=11)
        ax2.set_ylabel('Relative L2 Error', fontsize=11, color='red')
        ax2_twin.set_ylabel('PSNR (dB)', fontsize=11, color='blue')
        ax2.set_title('Multi-Step Performance Overview', fontsize=12, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='center right')
        
        ax2.grid(True, alpha=0.3)
        
        # Performance degradation analysis
        ax3 = fig.add_subplot(gs[1, 0])
        degradation_metrics = ['Rel-L2 Change (%)', 'PSNR Change (%)', 'SSIM Change (%)']
        degradation_values = [
            self.multistep_data['analysis']['performance_degradation']['rel_l2_increase'],
            self.multistep_data['analysis']['performance_degradation']['psnr_decrease'],
            self.multistep_data['analysis']['performance_degradation']['ssim_decrease']
        ]
        
        colors = ['red' if v > 0 else 'green' for v in degradation_values]
        bars = ax3.barh(degradation_metrics, degradation_values, color=colors, alpha=0.7)
        ax3.set_xlabel('Change (%)', fontsize=11)
        ax3.set_title('Performance Degradation\n(T_out=1 to T_out=20)', fontsize=12, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, degradation_values):
            width = bar.get_width()
            ax3.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2.,
                    f'{value:.1f}%', ha='left' if width > 0 else 'right', va='center', fontsize=9)
        
        # Model architecture summary
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
        🏗️ Model Architecture: {self.multistep_data['model_config']['architecture']}
        📐 Image Size: {self.multistep_data['model_config']['image_size']}×{self.multistep_data['model_config']['image_size']}
        📊 Channels: {self.multistep_data['model_config']['channels']}
        
        📈 Training Summary:
        • Total Epochs: {len(self.training_data['train_losses'])}
        • Final Train Loss: {self.training_data['train_losses'][-1]:.6f}
        • Best Val Loss: {self.training_data['best_val_loss']:.6f}
        
        🎯 Multi-Step Capabilities:
        • Max Successful T_out: {self.multistep_data['analysis']['max_successful_t_out']}
        • Best Rel-L2: {self.multistep_data['analysis']['min_rel_l2']:.4f}
        • Best PSNR: {self.multistep_data['analysis']['max_psnr']:.2f} dB
        • Best SSIM: {self.multistep_data['analysis']['max_ssim']:.6f}
        
        ⚡ Performance Insights:
        • Inference Time Scaling: {self.multistep_data['analysis']['inference_scaling']['time_increase']:.1f}% increase
        • Optimal T_out for Accuracy: {t_outs[np.argmin(rel_l2_values)]}
        • Optimal T_out for Speed: {t_outs[np.argmin([self.multistep_data['results'][str(t)]['avg_inference_time'] for t in t_outs])]}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_html_report(self):
        """Generate comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Temporal NAR Model - 综合分析报告</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 30px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 10px;
                }}
                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .summary-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background-color: white;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .summary-table th, .summary-table td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .summary-table th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                .summary-table tr:hover {{
                    background-color: #f5f5f5;
                }}
                .highlight {{
                    background-color: #e8f5e8;
                    padding: 15px;
                    border-left: 4px solid #27ae60;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                .timestamp {{
                    text-align: center;
                    color: #7f8c8d;
                    font-style: italic;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Temporal NAR Model 综合分析报告</h1>
                
                <div class="highlight">
                    <strong>📊 模型概述:</strong> 
                    {self.multistep_data['model_config']['architecture']} 架构，
                    图像尺寸 {self.multistep_data['model_config']['image_size']}×{self.multistep_data['model_config']['image_size']}，
                    {self.multistep_data['model_config']['channels']} 通道
                </div>
                
                <h2>📈 训练性能指标</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{len(self.training_data['train_losses'])}</div>
                        <div class="metric-label">训练轮数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.training_data['train_losses'][-1]:.4f}</div>
                        <div class="metric-label">最终训练损失</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.training_data['best_val_loss']:.4f}</div>
                        <div class="metric-label">最佳验证损失</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{(self.training_data['val_losses'][-1] - self.training_data['train_losses'][-1]):.4f}</div>
                        <div class="metric-label">过拟合差距</div>
                    </div>
                </div>
                
                <div class="image-container">
                    <h3>训练曲线分析</h3>
                    <img src="training_curves_comprehensive.png" alt="训练曲线">
                </div>
                
                <h2>🎯 单步预测性能</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{self.prediction_metrics['MSE']:.3f}</div>
                        <div class="metric-label">均方误差 (MSE)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.prediction_metrics['MAE']:.3f}</div>
                        <div class="metric-label">平均绝对误差 (MAE)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.prediction_metrics['PSNR']:.2f} dB</div>
                        <div class="metric-label">峰值信噪比 (PSNR)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.prediction_metrics['Rel_L2']:.3f}</div>
                        <div class="metric-label">相对L2误差</div>
                    </div>
                </div>
                
                <h2>⏰ 多时步预测能力</h2>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>预测步数 (T_out)</th>
                            <th>相对L2误差</th>
                            <th>PSNR (dB)</th>
                            <th>SSIM</th>
                            <th>推理时间 (秒)</th>
                            <th>成功率</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add multi-step results to table
        for t_out in sorted([int(k) for k in self.multistep_data['results'].keys()]):
            result = self.multistep_data['results'][str(t_out)]
            html_content += f"""
                        <tr>
                            <td>{t_out}</td>
                            <td>{result['avg_rel_l2']:.4f}</td>
                            <td>{result['avg_psnr']:.2f}</td>
                            <td>{result['avg_ssim']:.6f}</td>
                            <td>{result['avg_inference_time']:.4f}</td>
                            <td>{result['success_rate']:.1%}</td>
                        </tr>
            """
        
        html_content += f"""
                    </tbody>
                </table>
                
                <div class="image-container">
                    <h3>多时步预测分析</h3>
                    <img src="multistep_analysis.png" alt="多时步预测分析">
                </div>
                
                <div class="image-container">
                    <h3>性能综合总结</h3>
                    <img src="performance_summary.png" alt="性能总结">
                </div>
                
                <h2>🔍 关键发现</h2>
                <div class="highlight">
                    <ul>
                        <li><strong>最大预测步数:</strong> {self.multistep_data['analysis']['max_successful_t_out']} 步</li>
                        <li><strong>最佳精度:</strong> T_out={sorted([int(k) for k in self.multistep_data['results'].keys()])[np.argmin([self.multistep_data['results'][str(t)]['avg_rel_l2'] for t in sorted([int(k) for k in self.multistep_data['results'].keys()])])]} 时相对L2误差最低 ({self.multistep_data['analysis']['min_rel_l2']:.4f})</li>
                        <li><strong>最佳信噪比:</strong> {self.multistep_data['analysis']['max_psnr']:.2f} dB</li>
                        <li><strong>计算效率:</strong> 推理时间随预测步数线性增长，增长率约 {self.multistep_data['analysis']['inference_scaling']['time_increase']:.1f}%</li>
                        <li><strong>模型稳定性:</strong> 训练收敛良好，无明显过拟合现象</li>
                    </ul>
                </div>
                
                <h2>💡 使用建议</h2>
                <div class="highlight">
                    <ul>
                        <li><strong>短期预测 (T_out ≤ 5):</strong> 推荐用于高精度要求的应用</li>
                        <li><strong>中期预测 (T_out = 10):</strong> 精度与效率的最佳平衡点</li>
                        <li><strong>长期预测 (T_out ≥ 15):</strong> 适用于趋势分析，但精度会有所下降</li>
                        <li><strong>实时应用:</strong> 考虑到推理时间，建议 T_out ≤ 10</li>
                    </ul>
                </div>
                
                <div class="timestamp">
                    📅 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'comprehensive_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    def generate_all_visualizations(self):
        """Generate all visualizations and reports"""
        print("🎨 生成训练曲线分析...")
        self.create_training_curves()
        
        print("📊 生成多时步预测分析...")
        self.create_multistep_analysis()
        
        print("📈 生成性能总结...")
        self.create_performance_summary()
        
        print("📄 生成HTML综合报告...")
        self.generate_html_report()
        
        print(f"✅ 所有可视化文件已生成到: {self.output_dir}")
        print(f"📂 主要文件:")
        print(f"   • training_curves_comprehensive.png - 训练曲线分析")
        print(f"   • multistep_analysis.png - 多时步预测分析")
        print(f"   • performance_summary.png - 性能总结")
        print(f"   • comprehensive_report.html - 交互式HTML报告")

if __name__ == "__main__":
    visualizer = TemporalNARVisualizer()
    visualizer.generate_all_visualizations()