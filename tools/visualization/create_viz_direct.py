#!/usr/bin/env python3
"""
直接创建时序NAR训练结果可视化
使用基础Python库生成训练报告
"""

import json
import os
from pathlib import Path
import numpy as np

def create_visualization():
    """创建可视化报告"""
    # 设置路径
    base_dir = Path("f:/Zhaoyang/Sparse2Full")
    data_path = base_dir / "runs/temporal_nar_100epochs/TemporalNAR-DR2D-128-100epochs-s2025/training_history.json"
    save_dir = base_dir / "runs/temporal_nar_100epochs/visualizations"
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载数据
        print("📊 正在加载训练数据...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_losses = data['train_losses']
        val_losses = data.get('val_losses', [])
        best_val_loss = data.get('best_val_loss', min(val_losses) if val_losses else None)
        
        print(f"✅ 成功加载 {len(train_losses)} 轮训练数据")
        
        # 计算统计信息
        import numpy as np
        
        stats = {
            'total_epochs': len(train_losses),
            'final_train_loss': train_losses[-1],
            'best_train_loss': min(train_losses),
            'train_loss_reduction': (train_losses[0] - train_losses[-1]) / train_losses[0] * 100,
            'best_val_loss': best_val_loss,
            'val_loss_reduction': (val_losses[0] - val_losses[-1]) / val_losses[0] * 100 if val_losses else None,
            'convergence_epoch': None
        }
        
        # 寻找收敛点
        if len(train_losses) > 10:
            for i in range(10, len(train_losses)):
                recent_losses = train_losses[i-10:i]
                if max(recent_losses) - min(recent_losses) < 0.001:
                    stats['convergence_epoch'] = i
                    break
        
        # 生成详细的文本报告
        report_content = f"""时序NAR模型训练可视化报告
{'='*60}

📊 训练配置和结果摘要
{'='*60}

🎯 基本信息
• 总训练轮次: {stats['total_epochs']}
• 最终训练损失: {stats['final_train_loss']:.6f}
• 最佳训练损失: {stats['best_train_loss']:.6f}
• 训练损失降幅: {stats['train_loss_reduction']:.2f}%

📈 验证统计
• 最佳验证损失: {stats['best_val_loss']:.6f if stats['best_val_loss'] else 'N/A'}
• 验证损失降幅: {stats['val_loss_reduction']:.2f}% if stats['val_loss_reduction'] else 'N/A'

🎯 收敛分析
• 收敛轮次: {stats['convergence_epoch'] if stats['convergence_epoch'] else '未检测到明显收敛'}
• 训练状态: {'已收敛' if stats['convergence_epoch'] else '可能需要更多训练'}

📋 模型性能评估
• 过拟合风险: {'低' if val_losses and abs(train_losses[-1] - val_losses[-1]) < 0.01 else '中等' if val_losses else '无法评估'}
• 训练稳定性: {'良好' if len(train_losses) > 50 and np.std(train_losses[-20:]) < 0.01 else '一般'}
• 建议: {'模型训练良好，可以使用' if stats['best_val_loss'] and stats['best_val_loss'] < 1.0 else '考虑更多训练或调整超参数'}

{'='*60}
📊 详细训练数据
{'='*60}

训练损失序列 (前10轮):
{train_losses[:10]}

训练损失序列 (后10轮):
{train_losses[-10:]}

"""
        
        if val_losses:
            report_content += f"""
验证损失序列 (前10轮):
{val_losses[:10]}

验证损失序列 (后10轮):
{val_losses[-10:]}
"""
        
        # 添加损失变化分析
        if len(train_losses) > 1:
            loss_changes = [train_losses[i] - train_losses[i-1] for i in range(1, len(train_losses))]
            avg_improvement = sum(loss_changes) / len(loss_changes)
            
            report_content += f"""
{'='*60}
📈 损失变化分析
{'='*60}

• 平均每轮损失改善: {avg_improvement:.6f}
• 最大单轮改善: {min(loss_changes):.6f}
• 最大单轮恶化: {max(loss_changes):.6f}
• 改善轮次占比: {sum(1 for x in loss_changes if x < 0) / len(loss_changes) * 100:.1f}%

损失变化趋势 (每10轮):
"""
            for i in range(0, len(loss_changes), 10):
                chunk = loss_changes[i:i+10]
                avg_change = sum(chunk) / len(chunk)
                report_content += f"轮次 {i+2}-{i+len(chunk)+1}: 平均变化 {avg_change:.6f}\n"
        
        # 保存报告
        report_path = save_dir / "training_visualization_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 训练可视化报告已保存: {report_path}")
        
        # 尝试生成简单的matplotlib图表
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            
            print("\n🎨 正在生成matplotlib图表...")
            
            # 设置字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            epochs = range(1, len(train_losses) + 1)
            
            # 创建综合训练图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Temporal NAR Model Training Visualization Report', fontsize=16, fontweight='bold')
            
            # 1. 训练损失曲线
            ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            if val_losses:
                val_epochs = range(1, len(val_losses) + 1)
                ax1.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            ax1.set_xlabel('Training Epochs')
            ax1.set_ylabel('Loss Value')
            ax1.set_title('Training Loss Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 对数尺度损失
            ax2.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            if val_losses:
                ax2.semilogy(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            ax2.set_xlabel('Training Epochs')
            ax2.set_ylabel('Loss Value (Log Scale)')
            ax2.set_title('Log Scale Loss Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 损失分布直方图
            ax3.hist(train_losses, bins=30, alpha=0.7, color='blue', label='Training Loss Distribution', density=True)
            if val_losses:
                ax3.hist(val_losses, bins=30, alpha=0.7, color='red', label='Validation Loss Distribution', density=True)
            
            ax3.set_xlabel('Loss Value')
            ax3.set_ylabel('Density')
            ax3.set_title('Loss Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 改善率分析
            if len(train_losses) > 1:
                train_improvement = np.diff(train_losses)
                improvement_epochs = range(2, len(train_losses) + 1)
                ax4.plot(improvement_epochs, train_improvement, 'g-', linewidth=2, alpha=0.8, label='Training Loss Improvement')
                
                if val_losses and len(val_losses) > 1:
                    val_improvement = np.diff(val_losses)
                    val_improvement_epochs = range(2, len(val_losses) + 1)
                    ax4.plot(val_improvement_epochs, val_improvement, 'orange', linewidth=2, alpha=0.8, label='Validation Loss Improvement')
            
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Training Epochs')
            ax4.set_ylabel('Loss Improvement Rate')
            ax4.set_title('Loss Improvement Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = save_dir / "training_curves_comprehensive.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"✅ 综合训练曲线图已保存: {chart_path}")
            
            # 创建收敛分析图
            fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig2.suptitle('Convergence Analysis Report', fontsize=16, fontweight='bold')
            
            # 对数尺度损失趋势
            ax1.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            if val_losses:
                ax1.semilogy(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            # 添加趋势线
            if len(train_losses) > 10:
                z = np.polyfit(epochs, np.log(train_losses), 1)
                trend_line = np.exp(np.poly1d(z)(epochs))
                ax1.semilogy(epochs, trend_line, 'b--', alpha=0.5, label='Training Trend Line')
            
            ax1.set_xlabel('Training Epochs')
            ax1.set_ylabel('Loss Value (Log Scale)')
            ax1.set_title('Log Scale Loss Trend')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 滑动窗口改善率
            window_size = max(5, len(train_losses) // 20)
            if len(train_losses) > window_size:
                moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
                moving_epochs = range(window_size, len(train_losses) + 1)
                if len(moving_avg) > 1:
                    improvement_rate = np.diff(moving_avg) / moving_avg[:-1] * 100
                    
                    ax2.plot(moving_epochs[1:], improvement_rate, 'g-', linewidth=2, alpha=0.8)
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax2.set_xlabel('Training Epochs')
                    ax2.set_ylabel('Improvement Rate (%)')
                    ax2.set_title(f'Moving Window Improvement Rate (Window Size: {window_size})')
                    ax2.grid(True, alpha=0.3)
            
            # 稳定性分析
            if len(train_losses) > 20:
                rolling_std = []
                window = 10
                for i in range(window, len(train_losses)):
                    rolling_std.append(np.std(train_losses[i-window:i]))
                
                std_epochs = range(window + 1, len(train_losses) + 1)
                ax3.plot(std_epochs, rolling_std, 'purple', linewidth=2, alpha=0.8)
                ax3.set_xlabel('Training Epochs')
                ax3.set_ylabel('Rolling Standard Deviation')
                ax3.set_title('Training Stability Analysis')
                ax3.grid(True, alpha=0.3)
            
            # 最佳模型标记
            if val_losses:
                best_epoch = np.argmin(val_losses) + 1
                best_loss = min(val_losses)
                
                ax4.plot(range(1, len(val_losses) + 1), val_losses, 'r-', linewidth=2, alpha=0.8, label='Validation Loss')
                ax4.scatter([best_epoch], [best_loss], color='gold', s=100, zorder=5, label=f'Best Model (Epoch {best_epoch})')
                ax4.axvline(x=best_epoch, color='gold', linestyle='--', alpha=0.5)
                
                ax4.set_xlabel('Training Epochs')
                ax4.set_ylabel('Validation Loss')
                ax4.set_title('Best Model Identification')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存收敛分析图
            convergence_path = save_dir / "convergence_analysis.png"
            plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig2)
            
            print(f"✅ 收敛分析图已保存: {convergence_path}")
            
            print(f"\n🎉 所有可视化图表已成功生成！")
            print(f"📁 保存位置: {save_dir}")
            print("\n生成的文件:")
            print("  - training_visualization_report.txt (详细训练报告)")
            print("  - training_curves_comprehensive.png (综合训练曲线)")
            print("  - convergence_analysis.png (收敛分析)")
            
        except ImportError as e:
            print(f"⚠️  matplotlib不可用: {e}")
            print("✅ 已生成详细的文本报告作为替代")
        
    except FileNotFoundError:
        print(f"❌ 错误: 未找到训练数据文件 {data_path}")
    except Exception as e:
        print(f"❌ 生成可视化时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_visualization()