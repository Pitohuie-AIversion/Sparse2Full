#!/usr/bin/env python3
"""
生成时序训练的最终可视化报告
"""

import os
import sys
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.visualization import TemporalVisualizer, MetricsVisualizer


def load_checkpoint_info(ckpt_path: Path):
    """加载检查点信息"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        return {
            'epoch': ckpt.get('epoch', 'N/A'),
            'global_step': ckpt.get('global_step', 'N/A'),
            'best_val_loss': ckpt.get('best_val_loss', 'N/A'),
            'curriculum_stage': ckpt.get('curriculum_stage', 'N/A'),
            'config': ckpt.get('config', {}),
            'metrics_history': ckpt.get('metrics_history', {})
        }
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return {}


def count_visualization_files(vis_dir: Path):
    """统计可视化文件数量"""
    counts = {}
    
    # 训练可视化
    training_dir = vis_dir / "training"
    if training_dir.exists():
        counts['training_images'] = len(list(training_dir.glob("*.png")))
    else:
        counts['training_images'] = 0
    
    # 结果可视化
    results_dir = vis_dir / "results"
    if results_dir.exists():
        counts['result_images'] = len(list(results_dir.glob("*.png")))
    else:
        counts['result_images'] = 0
    
    # 动画
    animations_dir = vis_dir / "animations"
    if animations_dir.exists():
        counts['animations'] = len(list(animations_dir.glob("*.gif")))
    else:
        counts['animations'] = 0
    
    return counts


def generate_training_summary_plot(exp_dir: Path, ckpt_info: dict):
    """生成训练总结图"""
    metrics_history = ckpt_info.get('metrics_history', {})
    
    if not metrics_history:
        print("没有找到训练历史数据")
        return
    
    # 创建总结图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'训练总结报告 - {exp_dir.name}', fontsize=16)
    
    # 损失曲线
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        train_loss = metrics_history['train_loss']
        val_loss = metrics_history['val_loss']
        if train_loss and val_loss:
            epochs = range(len(train_loss))
            axes[0, 0].plot(epochs, train_loss, label='训练损失', alpha=0.8)
            axes[0, 0].plot(epochs, val_loss, label='验证损失', alpha=0.8)
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
    
    # Rel-L2曲线
    if 'train_rel_l2' in metrics_history and 'val_rel_l2' in metrics_history:
        train_rel_l2 = metrics_history['train_rel_l2']
        val_rel_l2 = metrics_history['val_rel_l2']
        if train_rel_l2 and val_rel_l2:
            epochs = range(len(train_rel_l2))
            axes[0, 1].plot(epochs, train_rel_l2, label='训练Rel-L2', alpha=0.8)
            axes[0, 1].plot(epochs, val_rel_l2, label='验证Rel-L2', alpha=0.8)
            axes[0, 1].set_title('Rel-L2曲线')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Rel-L2')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
    
    # MAE曲线
    if 'train_mae' in metrics_history and 'val_mae' in metrics_history:
        train_mae = metrics_history['train_mae']
        val_mae = metrics_history['val_mae']
        if train_mae and val_mae:
            epochs = range(len(train_mae))
            axes[1, 0].plot(epochs, train_mae, label='训练MAE', alpha=0.8)
            axes[1, 0].plot(epochs, val_mae, label='验证MAE', alpha=0.8)
            axes[1, 0].set_title('MAE曲线')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线
    if 'learning_rate' in metrics_history:
        lr_history = metrics_history['learning_rate']
        if lr_history:
            epochs = range(len(lr_history))
            axes[1, 1].plot(epochs, lr_history, label='学习率', alpha=0.8)
            axes[1, 1].set_title('学习率曲线')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    summary_path = exp_dir / "training_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 训练总结图已保存到: {summary_path}")


def generate_final_report(exp_dir: str):
    """生成最终报告"""
    exp_dir = Path(exp_dir)
    
    if not exp_dir.exists():
        print(f"❌ 实验目录不存在: {exp_dir}")
        return
    
    print(f"🔍 检查实验目录: {exp_dir}")
    
    # 检查检查点文件
    best_ckpt = exp_dir / "best.ckpt"
    last_ckpt = exp_dir / "last.ckpt"
    
    if not best_ckpt.exists():
        print("❌ 未找到best.ckpt文件")
        return
    
    # 加载检查点信息
    print("📊 加载检查点信息...")
    ckpt_info = load_checkpoint_info(best_ckpt)
    
    # 检查可视化目录
    vis_dir = exp_dir / "visualizations"
    if not vis_dir.exists():
        print("❌ 可视化目录不存在")
        return
    
    # 统计可视化文件
    print("📈 统计可视化文件...")
    vis_counts = count_visualization_files(vis_dir)
    
    # 生成训练总结图
    print("🎨 生成训练总结图...")
    generate_training_summary_plot(exp_dir, ckpt_info)
    
    # 生成文本报告
    report_path = exp_dir / "final_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 时序PDE训练最终报告\n\n")
        f.write(f"**实验名称**: {exp_dir.name}\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 训练信息\n\n")
        f.write(f"- **最佳Epoch**: {ckpt_info.get('epoch', 'N/A')}\n")
        f.write(f"- **全局步数**: {ckpt_info.get('global_step', 'N/A')}\n")
        f.write(f"- **最佳验证损失**: {ckpt_info.get('best_val_loss', 'N/A'):.6f}\n")
        f.write(f"- **课程学习阶段**: {ckpt_info.get('curriculum_stage', 'N/A')}\n\n")
        
        f.write("## 文件统计\n\n")
        f.write(f"- **训练可视化图片**: {vis_counts['training_images']} 张\n")
        f.write(f"- **结果分析图片**: {vis_counts['result_images']} 张\n")
        f.write(f"- **动画文件**: {vis_counts['animations']} 个\n\n")
        
        f.write("## 目录结构\n\n")
        f.write("```\n")
        f.write(f"{exp_dir.name}/\n")
        f.write("├── best.ckpt                 # 最佳模型检查点\n")
        f.write("├── last.ckpt                 # 最新模型检查点\n")
        f.write("├── training_summary.png      # 训练总结图\n")
        f.write("├── final_report.md           # 本报告\n")
        f.write("└── visualizations/           # 可视化文件\n")
        f.write("    ├── training/             # 训练过程可视化\n")
        f.write("    ├── results/              # 结果分析\n")
        f.write("    └── animations/           # 动画文件\n")
        f.write("```\n\n")
        
        f.write("## 使用说明\n\n")
        f.write("1. **查看训练过程**: 检查 `visualizations/training/` 目录下的预测图片\n")
        f.write("2. **分析结果**: 查看 `visualizations/results/` 目录下的误差分析图\n")
        f.write("3. **观看动画**: 打开 `visualizations/animations/` 目录下的GIF文件\n")
        f.write("4. **加载模型**: 使用 `best.ckpt` 文件加载训练好的模型\n\n")
        
        f.write("## 训练配置\n\n")
        config = ckpt_info.get('config', {})
        if config:
            f.write("```yaml\n")
            f.write(f"# 主要配置参数\n")
            if hasattr(config, 'model'):
                f.write(f"model: {config.model.get('_target_', 'N/A')}\n")
            if hasattr(config, 'data'):
                f.write(f"dataset: {config.data.get('dataset_name', 'N/A')}\n")
            if hasattr(config, 'training'):
                f.write(f"batch_size: {config.training.get('batch_size', 'N/A')}\n")
                f.write(f"learning_rate: {config.training.get('learning_rate', 'N/A')}\n")
            f.write("```\n\n")
        
        f.write("---\n")
        f.write("*报告由自动化脚本生成*\n")
    
    print(f"✅ 最终报告已生成: {report_path}")
    print(f"📁 实验目录: {exp_dir}")
    print(f"📊 训练可视化: {vis_counts['training_images']} 张图片")
    print(f"📈 结果分析: {vis_counts['result_images']} 张图片")
    print(f"🎬 动画文件: {vis_counts['animations']} 个")
    
    return {
        'report_path': report_path,
        'summary_plot': exp_dir / "training_summary.png",
        'checkpoint_info': ckpt_info,
        'visualization_counts': vis_counts
    }


if __name__ == "__main__":
    # 默认实验目录
    exp_dir = "f:/Zhaoyang/Sparse2Full/runs/Temporal-AR-SwinUNet-T1to3-DR2D-128-s2025"
    
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    
    result = generate_final_report(exp_dir)
    
    if result:
        print("\n🎉 最终报告生成完成!")
        print(f"📄 报告文件: {result['report_path']}")
        print(f"📊 总结图片: {result['summary_plot']}")