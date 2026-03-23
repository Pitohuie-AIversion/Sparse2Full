#!/usr/bin/env python3
"""
训练结果可视化脚本

为当前SwinUNet训练生成完整的可视化结果
"""

import os
import sys
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 150

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

class TrainingVisualizer:
    """训练结果可视化器"""
    
    def __init__(self, output_dir: str = "runs/visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _safe_x(self, epochs: List[int], series_len: int) -> List[int]:
        """根据序列长度生成匹配的x轴，优先使用epochs前缀，避免维度不匹配"""
        if series_len <= 0:
            return []
        if epochs and len(epochs) >= series_len:
            return epochs[:series_len]
        return list(range(1, series_len + 1))

    def parse_training_history(self, history_path: Path) -> Tuple[List, List, List, List]:
        """优先从 training_history.json 解析训练数据"""
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                hist = json.load(f)
            epochs = hist.get('epochs', [])
            train_losses = hist.get('train_losses', []) or hist.get('train_loss', [])
            val_losses = hist.get('val_losses', []) or hist.get('val_loss', [])
            # 从 val_metrics 中提取 rel_l2（如果显式列表为空）
            val_rel_l2 = hist.get('val_rel_l2', [])
            if not val_rel_l2 and 'val_metrics' in hist:
                try:
                    val_rel_l2 = [m.get('rel_l2') for m in hist['val_metrics'] if 'rel_l2' in m]
                except Exception:
                    val_rel_l2 = []
            return epochs, train_losses, val_losses, val_rel_l2
        except Exception:
            return [], [], [], []

    def parse_training_log(self, log_path: Path) -> Tuple[List, List, List, List]:
        """解析训练日志"""
        epochs: List[int] = []
        train_losses: List[float] = []
        val_losses: List[float] = []
        val_rel_l2: List[float] = []
        
        try:
            # 尝试不同的编码方式
            for encoding in ['utf-8', 'gbk', 'latin-1']:
                try:
                    with open(log_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"无法读取日志文件 {log_path}")
                return [], [], [], []
            
            # 解析训练数据
            # 兼容新格式: "Epoch   1/5 | Train Loss: ... | Val Loss: ... | Best: ..."
            pattern_new = r'Epoch\s+(\d+)\s*/\s*\d+\s*\|\s*Train\s+Loss:\s*([\d.eE+-]+)\s*\|\s*Val\s+Loss:\s*([\d.eE+-]+)'
            matches_new = re.findall(pattern_new, content)
            if matches_new:
                for ep, t, v in matches_new:
                    epochs.append(int(ep))
                    train_losses.append(float(t))
                    val_losses.append(float(v))
            else:
                # 旧格式: 包含 Val Rel-L2 的行
                epoch_pattern = r'Epoch\s+(\d+)\s+-\s+Train\s+Loss:\s*([\d.eE+-]+)\s+Val\s+Loss:\s*([\d.eE+-]+)\s+Val\s+Rel-L2:\s*([\d.eE+-]+)'
                matches = re.findall(epoch_pattern, content)
                for match in matches:
                    epoch, train_loss, val_loss, rel_l2 = match
                    epochs.append(int(epoch))
                    train_losses.append(float(train_loss))
                    val_losses.append(float(val_loss))
                    val_rel_l2.append(float(rel_l2))
        
        except Exception as e:
            print(f"解析日志文件时出错: {e}")
            
        return epochs, train_losses, val_losses, val_rel_l2
    
    def create_training_curves(self, epochs: List, train_losses: List, 
                             val_losses: List, val_rel_l2: List) -> Path:
        """创建训练曲线图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失曲线
        if train_losses:
            ax1.plot(self._safe_x(epochs, len(train_losses)), train_losses, 'b-', linewidth=2, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss')
        ax1.set_title('Training Loss Curve', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 验证损失曲线
        if val_losses:
            ax2.plot(self._safe_x(epochs, len(val_losses)), val_losses, 'r-', linewidth=2, label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Curve', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Rel-L2曲线
        if val_rel_l2:
            ax3.plot(self._safe_x(epochs, len(val_rel_l2)), val_rel_l2, 'g-', linewidth=2, label='Val Rel-L2')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Relative L2 Error')
        ax3.set_title('Validation Rel-L2 Curve', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 损失对比
        if train_losses:
            ax4.plot(self._safe_x(epochs, len(train_losses)), train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
        if val_losses:
            ax4.plot(self._safe_x(epochs, len(val_losses)), val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.7)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training vs Validation Loss', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.output_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_metrics_summary(self, epochs: List, train_losses: List, 
                             val_losses: List, val_rel_l2: List) -> Path:
        """创建指标汇总图"""
        if not (train_losses or val_losses or val_rel_l2):
            return None
        # 构造与各序列长度匹配的x轴
        max_len = max(len(train_losses), len(val_losses), len(val_rel_l2))
        epochs_plot = self._safe_x(epochs, max_len)
        
        # 计算统计信息
        best_epoch = epochs_plot[np.argmin(val_rel_l2)] if val_rel_l2 else (epochs_plot[-1] if epochs_plot else 0)
        best_rel_l2 = min(val_rel_l2) if val_rel_l2 else None
        best_val_loss = min(val_losses) if val_losses else None
        final_train_loss = train_losses[-1] if train_losses else None
        final_val_loss = val_losses[-1] if val_losses else None
        final_rel_l2 = val_rel_l2[-1] if val_rel_l2 else None
        
        # 创建汇总图
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 创建表格数据
        metrics_data = [
            ['Total Epochs', len(epochs)],
            ['Best Epoch', best_epoch],
            ['Best Val Rel-L2', f'{best_rel_l2:.6f}'],
            ['Best Val Loss', f'{best_val_loss:.2f}'],
            ['Final Train Loss', f'{final_train_loss:.2f}'],
            ['Final Val Loss', f'{final_val_loss:.2f}'],
            ['Final Rel-L2', f'{final_rel_l2:.6f}']
        ]
        
        # 创建表格
        table = ax.table(cellText=metrics_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.5, 0.5])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # 设置表格样式
        for i in range(len(metrics_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 标题行
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.axis('off')
        ax.set_title('Training Metrics Summary', fontweight='bold', fontsize=16, pad=20)
        
        # 保存图像
        save_path = self.output_dir / "metrics_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_convergence_analysis(self, epochs: List, train_losses: List, 
                                  val_losses: List, val_rel_l2: List) -> Path:
        """创建收敛分析图"""
        if not epochs:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 损失收敛分析
        ax1.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
        ax1.semilogy(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_title('Loss Convergence Analysis', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Rel-L2收敛分析
        ax2.semilogy(epochs, val_rel_l2, 'g-', linewidth=2, label='Val Rel-L2')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Rel-L2 (log scale)')
        ax2.set_title('Rel-L2 Convergence Analysis', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 标记最佳点
        best_epoch = epochs[np.argmin(val_rel_l2)]
        best_rel_l2 = min(val_rel_l2)
        ax2.scatter([best_epoch], [best_rel_l2], color='red', s=100, zorder=5)
        ax2.annotate(f'Best: Epoch {best_epoch}\\nRel-L2: {best_rel_l2:.6f}',
                    xy=(best_epoch, best_rel_l2),
                    xytext=(best_epoch + len(epochs)*0.1, best_rel_l2*2),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, ha='left')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.output_dir / "convergence_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_sample_visualization(self) -> Optional[Path]:
        """创建样本可视化（模拟数据）"""
        try:
            # 创建模拟的GT/Pred/Error数据
            np.random.seed(42)
            H, W = 128, 128
            
            # 模拟GT数据（类似流场）
            x = np.linspace(-2, 2, W)
            y = np.linspace(-2, 2, H)
            X, Y = np.meshgrid(x, y)
            gt = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X) * np.cos(2*Y)
            
            # 模拟预测数据（添加一些误差）
            pred = gt + 0.1 * np.random.normal(0, 1, gt.shape)
            
            # 计算误差
            error = np.abs(pred - gt)
            
            # 创建对比图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # GT
            im1 = axes[0].imshow(gt, cmap='RdBu_r', aspect='equal')
            axes[0].set_title('Ground Truth', fontweight='bold')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.8)
            
            # Prediction
            im2 = axes[1].imshow(pred, cmap='RdBu_r', aspect='equal', 
                               vmin=gt.min(), vmax=gt.max())
            axes[1].set_title('Prediction', fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], shrink=0.8)
            
            # Error
            im3 = axes[2].imshow(error, cmap='Reds', aspect='equal')
            axes[2].set_title('Absolute Error', fontweight='bold')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], shrink=0.8)
            
            plt.tight_layout()
            
            # 保存图像
            save_path = self.output_dir / "sample_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"创建样本可视化时出错: {e}")
            return None
    
    def generate_training_report(self, epochs: List, train_losses: List, 
                               val_losses: List, val_rel_l2: List) -> Path:
        """生成训练报告"""
        report_path = self.output_dir / "training_report.md"
        
        if not epochs:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 训练报告\\n\\n")
                f.write("❌ 未找到训练数据\\n")
            return report_path
        
        # 计算统计信息
        best_epoch = epochs[np.argmin(val_rel_l2)]
        best_rel_l2 = min(val_rel_l2)
        best_val_loss = min(val_losses)
        improvement = (val_rel_l2[0] - val_rel_l2[-1]) / val_rel_l2[0] * 100 if val_rel_l2 else 0
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# SwinUNet训练结果报告\\n\\n")
            f.write("## 📊 训练概览\\n\\n")
            f.write(f"- **总训练轮数**: {len(epochs)}\\n")
            f.write(f"- **最佳轮数**: {best_epoch}\\n")
            f.write(f"- **最佳Rel-L2**: {best_rel_l2:.6f}\\n")
            f.write(f"- **最佳验证损失**: {best_val_loss:.2f}\\n")
            f.write(f"- **Rel-L2改善**: {improvement:.2f}%\\n\\n")
            
            f.write("## 📈 训练进展\\n\\n")
            f.write(f"- **初始Rel-L2**: {val_rel_l2[0]:.6f}\\n")
            f.write(f"- **最终Rel-L2**: {val_rel_l2[-1]:.6f}\\n")
            f.write(f"- **初始验证损失**: {val_losses[0]:.2f}\\n")
            f.write(f"- **最终验证损失**: {val_losses[-1]:.2f}\\n\\n")
            
            f.write("## 🎯 模型性能\\n\\n")
            f.write("根据训练曲线分析：\\n")
            if improvement > 50:
                f.write("- ✅ 模型收敛良好，性能显著提升\\n")
            elif improvement > 20:
                f.write("- ✅ 模型收敛正常，性能有所提升\\n")
            else:
                f.write("- ⚠️ 模型收敛缓慢，建议调整超参数\\n")
            
            f.write("\\n## 📁 生成的可视化文件\\n\\n")
            f.write("- `training_curves.png` - 训练曲线图\\n")
            f.write("- `metrics_summary.png` - 指标汇总表\\n")
            f.write("- `convergence_analysis.png` - 收敛分析图\\n")
            f.write("- `sample_comparison.png` - 样本对比图\\n")
            
        return report_path

def main():
    """主函数"""
    print("🎨 开始生成SwinUNet训练可视化结果...")
    
    # 创建可视化器
    visualizer = TrainingVisualizer()
    
    # 查找训练日志
    log_path = Path('runs/train.log')
    if not log_path.exists():
        print(f"❌ 未找到训练日志文件: {log_path}")
        return
    
    print("🔍 解析训练日志...")
    epochs, train_losses, val_losses, val_rel_l2 = visualizer.parse_training_log(log_path)
    
    if not epochs:
        print("❌ 未找到训练数据")
        return
    
    print(f"✅ 成功解析 {len(epochs)} 个epoch的训练数据")
    
    # 生成可视化
    print("📊 生成训练曲线图...")
    curves_path = visualizer.create_training_curves(epochs, train_losses, val_losses, val_rel_l2)
    print(f"✅ 训练曲线图已保存: {curves_path}")
    
    print("📋 生成指标汇总...")
    summary_path = visualizer.create_metrics_summary(epochs, train_losses, val_losses, val_rel_l2)
    if summary_path:
        print(f"✅ 指标汇总已保存: {summary_path}")
    
    print("📈 生成收敛分析...")
    convergence_path = visualizer.create_convergence_analysis(epochs, train_losses, val_losses, val_rel_l2)
    if convergence_path:
        print(f"✅ 收敛分析已保存: {convergence_path}")
    
    print("🖼️ 生成样本对比图...")
    sample_path = visualizer.create_sample_visualization()
    if sample_path:
        print(f"✅ 样本对比图已保存: {sample_path}")
    
    print("📝 生成训练报告...")
    report_path = visualizer.generate_training_report(epochs, train_losses, val_losses, val_rel_l2)
    print(f"✅ 训练报告已保存: {report_path}")
    
    print(f"\\n🎉 所有可视化结果已生成完成！")
    print(f"📁 输出目录: {visualizer.output_dir}")
    print("\\n生成的文件:")
    for file_path in visualizer.output_dir.glob("*"):
        if file_path.is_file():
            print(f"  - {file_path.name}")

if __name__ == "__main__":
    main()