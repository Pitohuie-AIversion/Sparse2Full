#!/usr/bin/env python3
"""
AR Training Visualizer (English-only)
Visualizes autoregressive training: curriculum, metrics, sequences, and reports.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch

# Safe English font configuration
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置高质量输出
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class ARTrainingVisualizer:
    """AR training visualizer"""
    
    def __init__(self, output_dir: str):
        """
        Initialize AR training visualizer
        
        Args:
            output_dir: output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_curriculum_learning(self, training_history: Dict) -> bool:
        """
        Visualize curriculum learning progress.
        
        Args:
            training_history: training history dict
            
        Returns:
            bool: success flag
        """
        try:
            if 'curriculum_stages' not in training_history:
                print("Warning: no curriculum stages in history")
                return False
                
            stages = training_history['curriculum_stages']
            epochs = [s['epoch'] for s in stages]
            T_outs = [s['T_out'] for s in stages]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot T_out changes
            ax1.plot(epochs, T_outs, 'b-o', linewidth=2, markersize=4)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Prediction steps (T_out)')
            ax1.set_title('Curriculum: T_out progression')
            ax1.grid(True, alpha=0.3)
            
            # Plot stage distribution
            stage_counts = {}
            for s in stages:
                stage = s['stage']
                if stage not in stage_counts:
                    stage_counts[stage] = 0
                stage_counts[stage] += 1
            
            stages_list = list(stage_counts.keys())
            counts = list(stage_counts.values())
            
            ax2.bar(stages_list, counts, alpha=0.7, color='skyblue')
            ax2.set_xlabel('Curriculum stage')
            ax2.set_ylabel('Epoch count')
            ax2.set_title('Epochs per stage')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            save_path = self.output_dir / "curriculum_learning.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Curriculum visualization saved: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Curriculum visualization failed: {e}")
            return False
    
    def visualize_ar_metrics(self, training_history: Dict) -> bool:
        """
        Visualize AR-specific validation metrics.
        
        Args:
            training_history: training history dict
        Returns:
            bool: success flag
        """
        try:
            if 'val_metrics' not in training_history:
                print("Warning: no validation metrics in history")
                return False
                
            epochs = training_history.get('epochs', [])
            val_metrics = training_history['val_metrics']
            
            # 提取指标
            metrics_data = {}
            for epoch_metrics in val_metrics:
                for key, value in epoch_metrics.items():
                    if key not in metrics_data:
                        metrics_data[key] = []
                    metrics_data[key].append(value)
            
            # 创建子图
            n_metrics = len(metrics_data)
            n_cols = 2
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # 绘制每个指标
            for i, (metric_name, values) in enumerate(metrics_data.items()):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                ax.plot(epochs, values, 'o-', linewidth=2, markersize=3)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name.upper())
                ax.set_title(f'{metric_name.upper()} over epochs')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(n_metrics, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # 保存图片
            save_path = self.output_dir / "ar_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ AR metrics visualization saved: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ AR metrics visualization failed: {e}")
            return False
    
    def visualize_sequence_predictions(self, 
                                    input_seq: torch.Tensor,
                                    target_seq: torch.Tensor, 
                                    pred_seq: torch.Tensor,
                                    timestep: int = 0,
                                    channel: int = 0) -> bool:
        """
        Visualize sequence predictions.
        
        Args:
            input_seq: [B, T_in, C, H, W]
            target_seq: [B, T_out, C, H, W]
            pred_seq: [B, T_out, C, H, W]
            timestep: timestep to visualize
            channel: channel index
        Returns:
            bool: success flag
        """
        try:
            # 转换为numpy
            if isinstance(input_seq, torch.Tensor):
                input_seq = input_seq.detach().cpu().numpy()
            if isinstance(target_seq, torch.Tensor):
                target_seq = target_seq.detach().cpu().numpy()
            if isinstance(pred_seq, torch.Tensor):
                pred_seq = pred_seq.detach().cpu().numpy()
            
            batch_idx = 0  # 使用第一个样本
            
            # 获取数据
            input_frame = input_seq[batch_idx, -1, channel]  # 最后一个输入帧
            target_frame = target_seq[batch_idx, timestep, channel]
            pred_frame = pred_seq[batch_idx, timestep, channel]
            error_frame = np.abs(target_frame - pred_frame)
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 输入帧
            im1 = axes[0, 0].imshow(input_frame, cmap='viridis')
            axes[0, 0].set_title(f'Input Frame (t={-1})')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # 目标帧
            im2 = axes[0, 1].imshow(target_frame, cmap='viridis')
            axes[0, 1].set_title(f'目标帧 (t={timestep})')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # 预测帧
            im3 = axes[1, 0].imshow(pred_frame, cmap='viridis')
            axes[1, 0].set_title(f'Prediction (t={timestep})')
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # 误差帧
            im4 = axes[1, 1].imshow(error_frame, cmap='Reds')
            axes[1, 1].set_title(f'Absolute Error (t={timestep})')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1])
            
            plt.tight_layout()
            
            # 保存图片
            save_path = self.output_dir / f"sequence_prediction_t{timestep}_c{channel}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Sequence prediction visualization saved: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Sequence prediction visualization failed: {e}")
            return False
    
    def visualize_ar_predictions(self,
                               input_seq: torch.Tensor,
                               target_seq: torch.Tensor,
                               pred_seq: torch.Tensor,
                               epoch: int,
                               timestep: int = 0) -> bool:
        """
        可视化AR预测结果
        
        Args:
            input_seq: 输入序列 [B, T_in, C, H, W]
            target_seq: 目标序列 [B, T_out, C, H, W]
            pred_seq: 预测序列 [B, T_out, C, H, W]
            epoch: 当前epoch
            timestep: 时间步
            
        Returns:
            bool: 是否成功
        """
        return self.visualize_sequence_predictions(
            input_seq, target_seq, pred_seq, timestep, channel=0
        )
    
    def create_error_analysis(self,
                            input_seq: torch.Tensor,
                            target_seq: torch.Tensor,
                            pred_seq: torch.Tensor) -> bool:
        """
        Create error analysis.
        
        Args:
            input_seq: input sequence
            target_seq: target sequence
            pred_seq: prediction sequence
        Returns:
            bool: success flag
        """
        try:
            # 转换为numpy
            if isinstance(target_seq, torch.Tensor):
                target_seq = target_seq.detach().cpu().numpy()
            if isinstance(pred_seq, torch.Tensor):
                pred_seq = pred_seq.detach().cpu().numpy()
            
            batch_idx = 0
            channel = 0
            
            # 计算各时间步的误差
            errors = []
            for t in range(target_seq.shape[1]):  # T_out
                target_frame = target_seq[batch_idx, t, channel]
                pred_frame = pred_seq[batch_idx, t, channel]
                error = np.abs(target_frame - pred_frame)
                errors.append(error.mean())
            
            # 创建误差分析图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 时间步误差变化
            ax1.plot(range(len(errors)), errors, 'o-', linewidth=2, markersize=6)
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Mean Absolute Error')
            ax1.set_title('Error over time')
            ax1.grid(True, alpha=0.3)
            
            # 误差分布直方图
            all_errors = []
            for t in range(target_seq.shape[1]):
                target_frame = target_seq[batch_idx, t, channel]
                pred_frame = pred_seq[batch_idx, t, channel]
                error = np.abs(target_frame - pred_frame)
                all_errors.extend(error.flatten())
            
            ax2.hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Absolute Error')
            ax2.set_ylabel('Count')
            ax2.set_title('Error distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            save_path = self.output_dir / "error_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Error analysis saved: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error analysis failed: {e}")
            return False
    
    def create_temporal_analysis(self,
                               pred_seq: torch.Tensor,
                               target_seq: torch.Tensor) -> bool:
        """
        Create temporal analysis plots.
        
        Args:
            pred_seq: prediction sequence
            target_seq: target sequence
        Returns:
            bool: success flag
        """
        try:
            # 转换为numpy
            if isinstance(target_seq, torch.Tensor):
                target_seq = target_seq.detach().cpu().numpy()
            if isinstance(pred_seq, torch.Tensor):
                pred_seq = pred_seq.detach().cpu().numpy()
            
            batch_idx = 0
            channel = 0
            T_out = target_seq.shape[1]
            
            # 计算各种时序指标
            rel_l2_errors = []
            mae_errors = []
            
            for t in range(T_out):
                target_frame = target_seq[batch_idx, t, channel]
                pred_frame = pred_seq[batch_idx, t, channel]
                
                # Relative L2 error
                rel_l2 = np.linalg.norm(pred_frame - target_frame) / np.linalg.norm(target_frame)
                rel_l2_errors.append(rel_l2)
                
                # MAE
                mae = np.mean(np.abs(pred_frame - target_frame))
                mae_errors.append(mae)
            
            # 创建时序分析图
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Rel-L2随时间变化
            axes[0, 0].plot(range(T_out), rel_l2_errors, 'o-', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('Relative L2 Error')
            axes[0, 0].set_title('Rel-L2误差随时间变化')
            axes[0, 0].grid(True, alpha=0.3)
            
            # MAE随时间变化
            axes[0, 1].plot(range(T_out), mae_errors, 'o-', linewidth=2, markersize=6, color='red')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Mean Absolute Error')
            axes[0, 1].set_title('MAE误差随时间变化')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 误差增长率
            rel_l2_growth = [(rel_l2_errors[i] - rel_l2_errors[0]) / rel_l2_errors[0] 
                           for i in range(T_out)]
            axes[1, 0].plot(range(T_out), rel_l2_growth, 'o-', linewidth=2, markersize=6, color='green')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Relative growth rate')
            axes[1, 0].set_title('Rel-L2 error growth')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 预测稳定性分析
            stability = [np.std(pred_seq[batch_idx, t, channel]) for t in range(T_out)]
            axes[1, 1].plot(range(T_out), stability, 'o-', linewidth=2, markersize=6, color='purple')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Std. deviation')
            axes[1, 1].set_title('Prediction stability')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            save_path = self.output_dir / "temporal_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Temporal analysis saved: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Temporal analysis failed: {e}")
            return False

    def create_ar_summary_report(self, 
                               training_history: Dict,
                               sample_data: Optional[Dict] = None) -> bool:
        """
        Create AR training summary report (HTML).
        
        Args:
            training_history: training history dict
            sample_data: optional sample data
        Returns:
            bool: success flag
        """
        try:
            # 创建HTML报告
            html_content = self._generate_ar_html_report(training_history, sample_data)
            
            # 保存HTML文件
            html_path = self.output_dir / "ar_training_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ AR training report saved: {html_path}")
            return True
            
        except Exception as e:
            print(f"❌ AR training report generation failed: {e}")
            return False
    
    def _generate_ar_html_report(self, 
                               training_history: Dict,
                               sample_data: Optional[Dict] = None) -> str:
        """Generate AR training HTML report (English)."""
        
        # 计算统计信息
        final_train_loss = training_history['train_losses'][-1] if training_history['train_losses'] else 0
        final_val_loss = training_history['val_losses'][-1] if training_history['val_losses'] else 0
        best_val_loss = min(training_history['val_losses']) if training_history['val_losses'] else 0
        total_epochs = len(training_history['epochs']) if training_history['epochs'] else 0
        
        # 课程学习统计
        curriculum_info = ""
        if 'curriculum_stages' in training_history:
            stages = training_history['curriculum_stages']
            max_T_out = max([s['T_out'] for s in stages]) if stages else 0
            curriculum_info = f"""
            <p><strong>最大预测时间步长:</strong> {max_T_out}</p>
            <p><strong>课程学习阶段数:</strong> {len(set([s['stage'] for s in stages]))}</p>
            """
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AR Training Report</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .stat-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .image-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .image-card {{
                    text-align: center;
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .image-card img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                }}
                .timestamp {{
                    text-align: right;
                    color: #666;
                    font-size: 0.9em;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 AR Training Report</h1>
                
                <h2>📊 Training Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{total_epochs}</div>
                        <div class="stat-label">Total epochs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{final_train_loss:.4f}</div>
                        <div class="stat-label">Final training loss</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{final_val_loss:.4f}</div>
                        <div class="stat-label">Final validation loss</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{best_val_loss:.4f}</div>
                        <div class="stat-label">Best validation loss</div>
                    </div>
                </div>
                
                <h2>🎯 Curriculum Information</h2>
                {curriculum_info}
                
                <h2>📈 Training Visualizations</h2>
                <div class="image-grid">
                    <div class="image-card">
                        <h3>Curriculum Progress</h3>
                        <img src="curriculum_learning.png" alt="Curriculum visualization">
                    </div>
                    <div class="image-card">
                        <h3>AR Metrics</h3>
                        <img src="ar_metrics.png" alt="AR metrics visualization">
                    </div>
                </div>
                
                <div class="timestamp">
                    Generated at: {np.datetime64('now').astype(str)}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template


def main():
    """测试函数"""
    print("🧪 测试AR可视化器...")
    
    # 创建测试数据
    test_history = {
        'epochs': list(range(10)),
        'train_losses': [2.0 * np.exp(-i/5) + 0.1 * np.random.random() for i in range(10)],
        'val_losses': [2.2 * np.exp(-i/5) + 0.15 * np.random.random() for i in range(10)],
        'val_metrics': [
            {
                'rel_l2': 0.5 - i * 0.03,
                'mae': 0.3 - i * 0.02,
                'psnr': 15 + i * 0.5,
                'ssim': 0.7 + i * 0.02
            } for i in range(10)
        ],
        'curriculum_stages': [
            {'epoch': i, 'T_out': min(5 + i//2, 20), 'stage': i//3}
            for i in range(10)
        ]
    }
    
    # 创建可视化器
    visualizer = ARTrainingVisualizer("test_ar_viz")
    
    # 测试功能
    success1 = visualizer.visualize_curriculum_learning(test_history)
    success2 = visualizer.visualize_ar_metrics(test_history)
    success3 = visualizer.create_ar_summary_report(test_history)
    
    if all([success1, success2, success3]):
        print("✅ AR可视化器测试通过")
    else:
        print("❌ AR可视化器测试失败")


if __name__ == "__main__":
    main()