#!/usr/bin/env python3
"""
模型切换训练对比报告生成器
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def generate_training_report():
    """生成训练对比报告"""
    
    # 训练结果数据（来自演示运行）
    results = {
        "unet": {
            "final_train_loss": 0.000413,
            "final_val_loss": 0.000389,
            "total_time": 0.53,
            "epochs": 5,
            "batch_size": 4,
            "params": "4.3M",  # 来自测试报告
            "inference_time": "1.08ms"  # 来自测试报告
        },
        "swin_unet": {
            "final_train_loss": 0.024444,
            "final_val_loss": 0.023957,
            "total_time": 11.53,
            "epochs": 5,
            "batch_size": 4,
            "params": "41.5M",  # 来自测试报告
            "inference_time": "6.64ms"  # 来自测试报告
        }
    }
    
    # 创建报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "tested_models": list(results.keys()),
            "training_epochs": 5,
            "dataset_size": "100训练样本 + 20验证样本",
            "image_size": "128x128",
            "device": "CUDA"
        },
        "results": results,
        "analysis": {
            "best_performance": "unet",
            "fastest_training": "unet",
            "largest_model": "swin_unet",
            "most_accurate": "unet"
        }
    }
    
    # 保存JSON报告
    os.makedirs("runs/reports", exist_ok=True)
    with open("runs/reports/model_switching_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # 生成可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("模型切换训练性能对比", fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    train_losses = [results[m]["final_train_loss"] for m in models]
    val_losses = [results[m]["final_val_loss"] for m in models]
    training_times = [results[m]["total_time"] for m in models]
    
    # 训练损失对比
    axes[0, 0].bar(models, train_losses, color=['#2E86AB', '#A23B72'])
    axes[0, 0].set_title("最终训练损失")
    axes[0, 0].set_ylabel("MSE损失")
    axes[0, 0].set_yscale('log')
    
    # 验证损失对比
    axes[0, 1].bar(models, val_losses, color=['#2E86AB', '#A23B72'])
    axes[0, 1].set_title("最终验证损失")
    axes[0, 1].set_ylabel("MSE损失")
    axes[0, 1].set_yscale('log')
    
    # 训练时间对比
    axes[1, 0].bar(models, training_times, color=['#2E86AB', '#A23B72'])
    axes[1, 0].set_title("总训练时间")
    axes[1, 0].set_ylabel("时间 (秒)")
    
    # 模型参数对比
    param_values = [4.3, 41.5]  # 单位：百万
    axes[1, 1].bar(models, param_values, color=['#2E86AB', '#A23B72'])
    axes[1, 1].set_title("模型参数量")
    axes[1, 1].set_ylabel("参数量 (百万)")
    
    plt.tight_layout()
    plt.savefig("runs/reports/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成详细报告文档
    report_md = f"""# 模型切换训练对比报告

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 实验概述

本报告展示了在相同数据集和训练条件下，UNet和SwinUNet两种空间预测模型的训练性能对比。

### 实验设置
- **数据集**: 合成PDE数据 (100训练样本 + 20验证样本)
- **图像尺寸**: 128×128
- **训练轮数**: 5 epochs
- **批次大小**: 4
- **优化器**: AdamW + CosineAnnealingLR
- **设备**: CUDA

## 训练结果对比

| 模型 | 训练损失 | 验证损失 | 训练时间(秒) | 参数量 | 推理时间 |
|------|----------|----------|-------------|---------|----------|
| UNet | {results['unet']['final_train_loss']:.6f} | {results['unet']['final_val_loss']:.6f} | {results['unet']['total_time']:.2f} | {results['unet']['params']} | {results['unet']['inference_time']} |
| SwinUNet | {results['swin_unet']['final_train_loss']:.6f} | {results['swin_unet']['final_val_loss']:.6f} | {results['swin_unet']['total_time']:.2f} | {results['swin_unet']['params']} | {results['swin_unet']['inference_time']} |

## 性能分析

### 1. 训练效率
- **UNet**: 训练时间最短 ({results['unet']['total_time']:.2f}秒)
- **SwinUNet**: 训练时间最长 ({results['swin_unet']['total_time']:.2f}秒)

### 2. 模型精度
- **最佳训练损失**: UNet ({results['unet']['final_train_loss']:.6f})
- **最佳验证损失**: UNet ({results['unet']['final_val_loss']:.6f})

### 3. 模型复杂度
- **最小模型**: UNet ({results['unet']['params']} 参数)
- **最大模型**: SwinUNet ({results['swin_unet']['params']} 参数)

## 结论

1. **UNet** 在本次对比中表现最佳，具有最快的训练速度和最低的预测误差
2. **SwinUNet** 模型更大，训练时间更长，但可能具有更好的特征提取能力
3. 对于快速原型设计和资源受限的场景，推荐使用UNet
4. 对于需要更复杂特征提取的任务，可以考虑SwinUNet

## 模型切换建议

基于测试结果，建议：
- **快速实验**: 使用UNet
- **生产部署**: 根据具体任务需求选择
- **资源优化**: 优先考虑UNet
- **精度要求**: 可尝试SwinUNet并进行更长时间的训练

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    with open("runs/reports/model_switching_report.md", "w") as f:
        f.write(report_md)
    
    print("✅ 训练对比报告已生成!")
    print("📊 报告文件:")
    print("  - JSON数据: runs/reports/model_switching_report.json")
    print("  - 可视化图表: runs/reports/model_comparison.png")
    print("  - Markdown报告: runs/reports/model_switching_report.md")
    
    return report

if __name__ == "__main__":
    generate_training_report()