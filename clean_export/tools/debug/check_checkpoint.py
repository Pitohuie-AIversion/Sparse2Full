#!/usr/bin/env python3
"""检查best.pth检查点文件的详细信息"""

import torch
import os

def check_checkpoint(checkpoint_path):
    """检查检查点文件的详细信息"""
    print(f"检查检查点文件: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        return None
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\n📋 检查点内容:")
        print(f"Keys: {list(checkpoint.keys())}")
        
        # 基本信息
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_loss' in checkpoint:
            print(f"Best Loss: {checkpoint['best_loss']}")
        if 'best_metric' in checkpoint:
            print(f"Best Metric: {checkpoint['best_metric']}")
        if 'config' in checkpoint:
            print(f"Config available: Yes")
        
        # 模型状态字典
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"\n🏗️ 模型参数:")
            print(f"总参数数量: {len(model_state.keys())} keys")
            
            print("\n前10个模型参数:")
            for i, (k, v) in enumerate(list(model_state.items())[:10]):
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
        
        # 优化器状态
        if 'optimizer_state_dict' in checkpoint:
            print(f"\n⚙️ 优化器状态: 可用")
        
        # 调度器状态
        if 'scheduler_state_dict' in checkpoint:
            print(f"📅 调度器状态: 可用")
        
        # 配置信息
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\n⚙️ 配置信息:")
            if 'model' in config:
                print(f"Model: {config['model']}")
            if 'data' in config:
                print(f"Data: {config['data']}")
            if 'training' in config:
                print(f"Training: {config['training']}")
        
        # 验证结果
        if 'best_val_loss' in checkpoint:
            print(f"\n📊 验证结果:")
            print(f"Best Val Loss: {checkpoint['best_val_loss']}")
        if 'val_results' in checkpoint:
            print(f"Val Results: {checkpoint['val_results']}")
            
        return checkpoint
        
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return None

if __name__ == "__main__":
    checkpoint_path = "runs/checkpoints/best.pth"
    checkpoint = check_checkpoint(checkpoint_path)