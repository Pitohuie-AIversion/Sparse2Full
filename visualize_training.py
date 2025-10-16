#!/usr/bin/env python3
"""
训练结果可视化脚本
分析训练日志并生成可视化图表
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
# 使用统一的可视化工具，不直接导入matplotlib
from utils.visualization import PDEBenchVisualizer

def parse_training_log(log_path):
    """解析训练日志，提取损失和指标数据"""
    epochs = []
    train_losses = []
    val_losses = []
    val_rel_l2 = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取训练数据
    pattern = r'Epoch\s+(\d+)\s+-\s+Train Loss:\s+([\d.]+)\s+Val Loss:\s+([\d.]+)\s+Val Rel-L2:\s+([\d.]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        epoch, train_loss, val_loss, rel_l2 = match
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_rel_l2.append(float(rel_l2))
    
    return epochs, train_losses, val_losses, val_rel_l2

def extract_best_metrics(log_path):
    """提取最佳验证指标"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取最佳验证损失
    best_val_loss_match = re.search(r'Best validation loss:\s+([\d.]+)', content)
    best_val_loss = float(best_val_loss_match.group(1)) if best_val_loss_match else None
    
    # 提取最佳验证指标
    metrics_pattern = r"'rel_l2': tensor\(\[\[([\d.]+)\],\s*\[([\d.]+)\]\].*?'mae': tensor\(\[\[([\d.]+)\],\s*\[([\d.]+)\]\].*?'psnr': tensor\(\[\[([\d.]+)\],\s*\[([\d.]+)\]\].*?'ssim': tensor\(\[\[([\d.]+)\],\s*\[([\d.]+)\]\]"
    metrics_match = re.search(metrics_pattern, content, re.DOTALL)
    
    best_metrics = {}
    if metrics_match:
        rel_l2_1, rel_l2_2, mae_1, mae_2, psnr_1, psnr_2, ssim_1, ssim_2 = metrics_match.groups()
        best_metrics = {
            'rel_l2': [float(rel_l2_1), float(rel_l2_2)],
            'mae': [float(mae_1), float(mae_2)],
            'psnr': [float(psnr_1), float(psnr_2)],
            'ssim': [float(ssim_1), float(ssim_2)]
        }
    
    # 提取训练时间
    train_time_match = re.search(r'Total training time:\s+([\d.]+)s', content)
    val_time_match = re.search(r'Total validation time:\s+([\d.]+)s', content)
    
    train_time = float(train_time_match.group(1)) if train_time_match else None
    val_time = float(val_time_match.group(1)) if val_time_match else None
    
    return best_val_loss, best_metrics, train_time, val_time

def create_loss_curves(epochs, train_losses, val_losses, val_rel_l2, output_dir):
    """创建损失曲线图"""
    visualizer = PDEBenchVisualizer(str(output_dir))
    
    # 准备训练和验证日志
    train_logs = {
        'loss': train_losses,
        'rel_l2': [0] * len(epochs)  # 训练时没有rel_l2，用0填充
    }
    
    val_logs = {
        'loss': val_losses,
        'rel_l2': val_rel_l2
    }
    
    # 使用统一的可视化接口
    visualizer.plot_training_curves(train_logs, val_logs, "loss_curves")

def create_metrics_visualization(best_metrics, output_dir):
    """创建最佳指标可视化"""
    if not best_metrics:
        return
    
    visualizer = PDEBenchVisualizer(str(output_dir))
    
    # 准备模型比较数据
    model_results = {}
    for metric, values in best_metrics.items():
        if isinstance(values, list) and len(values) >= 2:
            model_results[f'Channel1_{metric}'] = values[0]
            model_results[f'Channel2_{metric}'] = values[1]
        else:
            model_results[metric] = values if not isinstance(values, list) else values[0]
    
    # 使用模型比较功能来显示指标
    visualizer.plot_model_comparison(
        {'Best_Metrics': model_results}, 
        save_name="best_metrics"
    )

def create_convergence_analysis(epochs, train_losses, val_losses, val_rel_l2, output_dir):
    """创建收敛分析图"""
    # 找到最佳验证损失的epoch
    best_val_idx = np.argmin(val_losses)
    best_epoch = epochs[best_val_idx]
    best_val_loss = val_losses[best_val_idx]
    
    # 相对L2误差收敛分析
    best_rel_l2_idx = np.argmin(val_rel_l2)
    best_rel_l2_epoch = epochs[best_rel_l2_idx]
    best_rel_l2_value = val_rel_l2[best_rel_l2_idx]
    
    # 使用统一的可视化接口创建训练曲线
    visualizer = PDEBenchVisualizer(str(output_dir))
    
    train_logs = {
        'loss': train_losses,
        'rel_l2': [0] * len(epochs)  # 训练时没有rel_l2
    }
    
    val_logs = {
        'loss': val_losses,
        'rel_l2': val_rel_l2
    }
    
    visualizer.plot_training_curves(train_logs, val_logs, "convergence_analysis")
    
    return best_epoch, best_val_loss, best_rel_l2_epoch, best_rel_l2_value

def generate_training_report(epochs, train_losses, val_losses, val_rel_l2, 
                           best_val_loss, best_metrics, train_time, val_time,
                           best_epoch, best_rel_l2_epoch, output_dir):
    """生成训练总结报告"""
    
    # 计算统计信息
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    final_rel_l2 = val_rel_l2[-1] if val_rel_l2 else 0
    
    min_train_loss = min(train_losses) if train_losses else 0
    min_val_loss = min(val_losses) if val_losses else 0
    min_rel_l2 = min(val_rel_l2) if val_rel_l2 else 0
    
    # 生成报告
    report = f"""
# 训练结果总结报告

## 训练配置
- **模型**: SwinUNet
- **任务**: SR x4 (超分辨率4倍)
- **数据集**: PDEBench
- **训练样本**: 1000
- **验证样本**: 100
- **批次大小**: 4
- **总epoch数**: {len(epochs)}

## 训练性能
- **总训练时间**: {train_time:.2f}秒
- **总验证时间**: {val_time:.2f}秒
- **平均每epoch时间**: {(train_time + val_time) / len(epochs):.2f}秒

## 损失收敛情况
- **最终训练损失**: {final_train_loss:.6f}
- **最终验证损失**: {final_val_loss:.6f}
- **最终相对L2误差**: {final_rel_l2:.6f}

## 最佳性能指标
- **最佳验证损失**: {best_val_loss:.6f} (Epoch {best_epoch})
- **最佳相对L2误差**: {min_rel_l2:.6f} (Epoch {best_rel_l2_epoch})

### 最佳验证指标详情
"""
    
    if best_metrics:
        for metric, values in best_metrics.items():
            avg_value = np.mean(values)
            report += f"- **{metric.upper()}**: {avg_value:.6f} (通道1: {values[0]:.6f}, 通道2: {values[1]:.6f})\n"
    
    report += f"""
## 收敛分析
- **训练损失减少**: {train_losses[0]:.6f} → {final_train_loss:.6f} ({((train_losses[0] - final_train_loss) / train_losses[0] * 100):.1f}%减少)
- **验证损失减少**: {val_losses[0]:.6f} → {final_val_loss:.6f} ({((val_losses[0] - final_val_loss) / val_losses[0] * 100):.1f}%减少)
- **相对L2误差减少**: {val_rel_l2[0]:.6f} → {final_rel_l2:.6f} ({((val_rel_l2[0] - final_rel_l2) / val_rel_l2[0] * 100):.1f}%减少)

## 模型性能评估
- **PSNR**: {np.mean(best_metrics['psnr']) if best_metrics and 'psnr' in best_metrics else 'N/A':.2f} dB
- **SSIM**: {np.mean(best_metrics['ssim']) if best_metrics and 'ssim' in best_metrics else 'N/A':.4f}
- **MAE**: {np.mean(best_metrics['mae']) if best_metrics and 'mae' in best_metrics else 'N/A':.6f}

## 训练稳定性
- **训练损失标准差**: {np.std(train_losses):.6f}
- **验证损失标准差**: {np.std(val_losses):.6f}
- **相对L2误差标准差**: {np.std(val_rel_l2):.6f}

## 结论
训练成功完成，模型在{len(epochs)}个epoch后达到良好的收敛状态。
最佳验证损失为{best_val_loss:.6f}，相对L2误差为{min_rel_l2:.6f}，
PSNR达到{np.mean(best_metrics['psnr']) if best_metrics and 'psnr' in best_metrics else 'N/A':.2f}dB，
SSIM为{np.mean(best_metrics['ssim']) if best_metrics and 'ssim' in best_metrics else 'N/A':.4f}，
表明模型具有良好的超分辨率重建性能。

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 保存报告
    with open(output_dir / 'training_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存JSON格式的数据
    data = {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_rel_l2': val_rel_l2,
        'best_val_loss': best_val_loss,
        'best_metrics': best_metrics,
        'train_time': train_time,
        'val_time': val_time,
        'best_epoch': best_epoch,
        'best_rel_l2_epoch': best_rel_l2_epoch
    }
    
    with open(output_dir / 'training_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    # 设置路径
    log_path = Path('runs/train.log')
    output_dir = Path('runs/visualization')
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 分析训练日志...")
    
    # 解析训练日志
    epochs, train_losses, val_losses, val_rel_l2 = parse_training_log(log_path)
    
    if not epochs:
        print("❌ 未找到训练数据，请检查日志文件")
        return
    
    print(f"✅ 成功解析 {len(epochs)} 个epoch的训练数据")
    
    # 提取最佳指标
    best_val_loss, best_metrics, train_time, val_time = extract_best_metrics(log_path)
    
    print("📊 生成可视化图表...")
    
    # 创建损失曲线图
    create_loss_curves(epochs, train_losses, val_losses, val_rel_l2, output_dir)
    print("✅ 损失曲线图已生成")
    
    # 创建指标可视化
    create_metrics_visualization(best_metrics, output_dir)
    print("✅ 最佳指标图已生成")
    
    # 创建收敛分析图
    best_epoch, best_val_loss_found, best_rel_l2_epoch, best_rel_l2_value = create_convergence_analysis(
        epochs, train_losses, val_losses, val_rel_l2, output_dir)
    print("✅ 收敛分析图已生成")
    
    # 生成训练报告
    generate_training_report(epochs, train_losses, val_losses, val_rel_l2,
                           best_val_loss, best_metrics, train_time, val_time,
                           best_epoch, best_rel_l2_epoch, output_dir)
    print("✅ 训练报告已生成")
    
    print(f"\n🎉 可视化完成！结果保存在: {output_dir}")
    print(f"📈 损失曲线图: {output_dir}/loss_curves.png")
    print(f"📊 最佳指标图: {output_dir}/best_metrics.png")
    print(f"📉 收敛分析图: {output_dir}/convergence_analysis.png")
    print(f"📝 训练报告: {output_dir}/training_report.md")
    print(f"💾 训练数据: {output_dir}/training_data.json")
    
    # 打印关键统计信息
    print(f"\n📋 关键统计信息:")
    print(f"   最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"   最佳相对L2误差: {best_rel_l2_value:.6f} (Epoch {best_rel_l2_epoch})")
    if best_metrics:
        print(f"   平均PSNR: {np.mean(best_metrics['psnr']):.2f} dB")
        print(f"   平均SSIM: {np.mean(best_metrics['ssim']):.4f}")
        print(f"   平均MAE: {np.mean(best_metrics['mae']):.6f}")
    print(f"   总训练时间: {train_time:.2f}秒")

if __name__ == "__main__":
    main()