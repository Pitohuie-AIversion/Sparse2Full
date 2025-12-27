#!/usr/bin/env python3
"""
补全AR训练运行的缺失可视化文件
针对运行目录: AR-DR2D-Debug-FNO2D-Staged-s2025-model_None_20251120_140708
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_training_history(run_dir):
    """加载训练历史"""
    history_file = run_dir / "training_history.json"
    if not history_file.exists():
        return None
    
    import json
    with open(history_file, 'r') as f:
        return json.load(f)

def create_training_curves_visualization(history, output_dir):
    """创建训练曲线可视化"""
    if not history or 'train_losses' not in history:
        return
    
    plt.figure(figsize=(12, 8))
    
    # 训练损失
    epochs = range(1, len(history['train_losses']) + 1)
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    if 'val_losses' in history and history['val_losses']:
        # 过滤掉inf值
        val_epochs = []
        val_losses = []
        for i, loss in enumerate(history['val_losses']):
            if not np.isinf(loss):
                val_epochs.append(i + 1)
                val_losses.append(loss)
        if val_losses:
            plt.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习率（如果有）
    if 'learning_rates' in history and history['learning_rates']:
        plt.subplot(2, 1, 2)
        plt.plot(epochs[:len(history['learning_rates'])], history['learning_rates'], 'g-', label='Learning Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存: {output_dir / 'training_curves.png'}")

def create_model_architecture_visualization(config, output_dir):
    """创建模型架构可视化"""
    try:
        # 创建简化的模型架构图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 模型架构信息
        spatial_config = config.get('sequential', {}).get('spatial', {})
        temporal_config = config.get('sequential', {}).get('temporal', {})
        
        # 创建架构文本
        architecture_text = f"""
Sequential Spatiotemporal Model
├─ Spatial Feature Extractor (FNO2D)
│  ├─ Input: {spatial_config.get('in_channels', '?')} channels
│  ├─ Output: {spatial_config.get('spatial_feature_dim', '?')} features  
│  ├─ Modes: {spatial_config.get('backbone_config', {}).get('modes1', '?')}×{spatial_config.get('backbone_config', {}).get('modes2', '?')}
│  └─ Layers: {spatial_config.get('backbone_config', {}).get('n_layers', '?')}
└─ Temporal Prediction Module (Transformer)
   ├─ Feature dim: {temporal_config.get('spatial_feature_dim', '?')}
   ├─ Temporal dim: {temporal_config.get('temporal_dim', '?')}
   ├─ Heads: {temporal_config.get('num_heads', '?')}
   └─ Layers: {temporal_config.get('num_layers', '?')}
"""
        
        ax.text(0.1, 0.5, architecture_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Model Architecture Overview', fontsize=16, fontweight='bold')
        
        plt.savefig(output_dir / "model_architecture.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 模型架构图已保存: {output_dir / 'model_architecture.png'}")
        
    except Exception as e:
        print(f"⚠️  模型架构可视化失败: {e}")

def create_resource_usage_visualization(run_dir, output_dir):
    """创建资源使用可视化"""
    try:
        resources_file = run_dir / "resources_epoch.jsonl"
        if not resources_file.exists():
            print("⚠️  资源使用文件未找到")
            return
        
        import json
        import pandas as pd
        
        # 读取资源数据
        resources_data = []
        with open(resources_file, 'r') as f:
            for line in f:
                if line.strip():
                    resources_data.append(json.loads(line))
        
        if not resources_data:
            return
        
        df = pd.DataFrame(resources_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # GPU内存使用
        if 'gpu_memory_mb' in df.columns:
            axes[0, 0].plot(df['epoch'], df['gpu_memory_mb'], 'b-', linewidth=2)
            axes[0, 0].set_title('GPU Memory Usage')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Memory (MB)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 训练时间
        if 'time_per_epoch' in df.columns:
            axes[0, 1].plot(df['epoch'], df['time_per_epoch'], 'g-', linewidth=2)
            axes[0, 1].set_title('Training Time per Epoch')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率
        if 'learning_rate' in df.columns:
            axes[1, 0].plot(df['epoch'], df['learning_rate'], 'r-', linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 损失值
        if 'train_loss' in df.columns:
            axes[1, 1].plot(df['epoch'], df['train_loss'], 'purple', linewidth=2)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "resource_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 资源使用图已保存: {output_dir / 'resource_usage.png'}")
        
    except Exception as e:
        print(f"⚠️  资源使用可视化失败: {e}")

def main():
    """主函数"""
    # 运行目录
    run_name = "AR-DR2D-Debug-FNO2D-Staged-s2025-model_None_20251120_140708"
    run_dir = Path(f"runs/{run_name}")
    
    if not run_dir.exists():
        print(f"❌ 运行目录不存在: {run_dir}")
        return
    
    # 创建可视化输出目录
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    print(f"🎨 开始为 {run_name} 生成缺失的可视化...")
    print(f"📁 输出目录: {viz_dir}")
    
    # 加载配置
    import yaml
    config_file = run_dir / "config_merged.yaml"
    config = None
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    
    # 加载训练历史
    history = load_training_history(run_dir)
    
    # 生成可视化
    if history:
        create_training_curves_visualization(history, viz_dir)
    
    if config:
        create_model_architecture_visualization(config, viz_dir)
    
    create_resource_usage_visualization(run_dir, viz_dir)
    
    # 创建可视化索引文件
    create_visualization_index(viz_dir, run_name, history, config)
    
    print(f"✅ 可视化生成完成!")
    print(f"📊 请在浏览器中查看: {viz_dir / 'index.html'}")

def create_visualization_index(viz_dir, run_name, history, config):
    """创建可视化索引HTML文件"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>训练可视化 - {run_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #4CAF50; margin-top: 30px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .image-item {{ text-align: center; }}
        .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .image-item h3 {{ margin: 10px 0 5px 0; color: #333; }}
        .info-box {{ background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; }}
        .timestamp {{ color: #666; font-size: 0.9em; text-align: center; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AR训练可视化报告</h1>
        <div class="info-box">
            <strong>运行名称:</strong> {run_name}<br>
            <strong>模型类型:</strong> Sequential Spatiotemporal Model<br>
            <strong>架构:</strong> FNO2D + Transformer<br>
            <strong>总轮数:</strong> {len(history.get('train_losses', [])) if history else '未知'}
        </div>
        
        <h2>训练过程可视化</h2>
        <div class="image-grid">
            <div class="image-item">
                <h3>训练曲线</h3>
                <img src="training_curves.png" alt="Training Curves">
            </div>
            <div class="image-item">
                <h3>模型架构</h3>
                <img src="model_architecture.png" alt="Model Architecture">
            </div>
            <div class="image-item">
                <h3>资源使用</h3>
                <img src="resource_usage.png" alt="Resource Usage">
            </div>
        </div>
        
        <div class="timestamp">
            生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    with open(viz_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 可视化索引已保存: {viz_dir / 'index.html'}")

if __name__ == "__main__":
    main()