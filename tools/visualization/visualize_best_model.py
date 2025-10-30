#!/usr/bin/env python3
"""
可视化best.pth模型的预测结果

使用工作空间现有的PDEBenchVisualizer创建全面的可视化：
- 模型预测结果对比（GT vs Pred vs Error）
- 多个测试样本的可视化展示
- 预测质量分析图
- 功率谱分析
- 误差分布统计图
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.visualization import PDEBenchVisualizer
from models.swin_unet import SwinUNet
from datasets.pdebench import PDEBenchDataModule
from ops.degradation import apply_degradation_operator
from ops.metrics import compute_all_metrics

def load_model_and_config(checkpoint_path):
    """加载模型和配置"""
    print(f"🔄 加载检查点: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # 创建模型
    model_config = config['model']
    model = SwinUNet(
        in_channels=model_config['params']['in_channels'],
        out_channels=model_config['params']['out_channels'],
        img_size=model_config['params']['img_size'],
        **{k: v for k, v in model_config['params'].items() 
           if k not in ['in_channels', 'out_channels', 'img_size']}
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功 - Epoch: {checkpoint['epoch']}")
    print(f"✓ 最佳验证损失: {checkpoint['best_val_loss']:.6f}")
    
    return model, config, checkpoint

def create_data_loader(config):
    """创建数据加载器"""
    print("🔄 创建数据加载器...")
    
    # 创建数据模块
    data_module = PDEBenchDataModule(config['data'])
    data_module.setup()
    
    # 获取测试数据加载器
    test_loader = data_module.test_dataloader()
    
    print(f"✓ 数据加载器创建成功 - 测试样本数: {len(test_loader.dataset)}")
    
    return test_loader, data_module

def generate_predictions(model, test_loader, num_samples=5):
    """生成预测结果"""
    print(f"🔄 生成 {num_samples} 个样本的预测结果...")
    
    results = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # 移动数据到设备
            input_data = batch['observation'].to(device)  # 使用observation作为输入（上采样后的）
            target = batch['target'].to(device)
            
            # 获取真实的低分辨率观测数据
            if 'lr_observation' in batch:
                lr_observation = batch['lr_observation']  # 真实的32x32观测数据
            elif 'original_observation' in batch:
                lr_observation = batch['original_observation']  # 真实的观测数据
            else:
                # 如果没有原始观测数据，使用observation
                lr_observation = batch['observation']
            
            # 模型预测
            pred = model(input_data)
            
            # 计算指标
            metrics = compute_all_metrics(pred, target)
            
            # 转换为numpy用于可视化
            lr_observation_np = lr_observation.cpu().numpy()  # 真实的低分辨率观测
            input_np = input_data.cpu().numpy()  # 上采样后的观测（用于模型输入）
            target_np = target.cpu().numpy()
            pred_np = pred.cpu().numpy()
            
            results.append({
                'lr_observation': lr_observation_np,  # 真实的32x32观测数据
                'observation': input_np,  # 上采样后的观测数据
                'target': target_np,
                'prediction': pred_np,
                'metrics': metrics,
                'sample_id': i + 1
            })
            
            rel_l2_value = metrics['rel_l2'].mean() if hasattr(metrics['rel_l2'], 'mean') else metrics['rel_l2']
            print(f"  样本 {i+1}: Rel-L2 = {rel_l2_value:.4f}")
    
    print(f"✓ 预测结果生成完成")
    return results

def create_comprehensive_visualizations(results, config, checkpoint, output_dir):
    """创建全面的可视化"""
    print("🔄 创建可视化图表...")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化可视化器
    visualizer = PDEBenchVisualizer(str(output_path))
    
    # 1. 为每个样本创建真实观测 vs GT vs Pred vs Error对比图
    print("  📊 创建预测结果对比图（使用真实32x32观测数据）...")
    for result in results:
        sample_id = result['sample_id']
        
        # 取第一个batch和第一个通道
        gt = result['target'][0, 0]  # [H, W] - 128x128真值
        pred = result['prediction'][0, 0]  # [H, W] - 128x128预测
        lr_observed = result['lr_observation'][0, 0]  # 真实的32x32观测数据
        
        # 调试信息
        print(f"DEBUG: gt shape: {gt.shape}")
        print(f"DEBUG: pred shape: {pred.shape}")
        print(f"DEBUG: lr_observed shape: {lr_observed.shape}")
        
        # 创建四联图对比：32x32观测 vs 128x128真值 vs 128x128预测 vs 128x128误差
        rel_l2_value = result['metrics']['rel_l2'].mean() if hasattr(result['metrics']['rel_l2'], 'mean') else result['metrics']['rel_l2']
        visualizer.create_quadruplet_visualization(
            observed=torch.tensor(lr_observed),  # 使用真实的32x32观测数据
            gt=torch.tensor(gt),
            pred=torch.tensor(pred),
            save_name=f"true_observation_comparison_sample_{sample_id}",
            title=f"Sample {sample_id} - True 32x32 Observation - Rel-L2: {rel_l2_value:.4f}"
        )
    
    # 2. 创建功率谱分析图
    print("  📊 创建功率谱分析图...")
    for result in results:
        sample_id = result['sample_id']
        gt = result['target'][0, 0]
        pred = result['prediction'][0, 0]
        
        # GT功率谱
        visualizer.create_power_spectrum_plot(
            field=torch.tensor(gt),
            save_name=f"power_spectrum_gt_sample_{sample_id}"
        )
        
        # 预测功率谱
        visualizer.create_power_spectrum_plot(
            field=torch.tensor(pred),
            save_name=f"power_spectrum_pred_sample_{sample_id}"
        )
    
    # 3. 创建功率谱对比图（跳过，因为方法不存在）
    print("  📊 跳过功率谱对比图（方法不存在）...")
    
    # 4. 跳过边界效应分析（方法不存在）
    print("  📊 跳过边界效应分析（方法不存在）...")
    
    # 5. 创建指标汇总图
    print("  📊 创建指标汇总图...")
    all_metrics = {}
    for result in results:
        for metric, value in result['metrics'].items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(value.mean() if hasattr(value, 'mean') else value)
    
    # 计算统计信息（只保留均值）
    stats = {}
    for metric, values in all_metrics.items():
        stats[metric] = np.mean(values)
    
    visualizer.create_metrics_summary_plot(
        {'Model': stats},
        save_name="metrics_summary"
    )
    
    # 6. 创建边界分析图
    print("  📊 创建边界分析图...")
    for result in results:
        sample_id = result['sample_id']
        gt = result['target'][0, 0]
        pred = result['prediction'][0, 0]
        
        visualizer.create_boundary_analysis(
            gt=torch.tensor(gt),
            pred=torch.tensor(pred),
            boundary_width=16,
            save_name=f"boundary_analysis_sample_{sample_id}"
        )
    
    # 7. 保存详细结果
    print("  💾 保存详细结果...")
    results_summary = {
        'model_info': {
            'epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['best_val_loss'],
            'config': config
        },
        'metrics_summary': {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for key, values in all_metrics.items()
        },
        'sample_metrics': [
            {
                'sample_id': result['sample_id'],
                'metrics': {k: v.mean().item() if isinstance(v, torch.Tensor) else v 
                          for k, v in result['metrics'].items()}
            }
            for result in results
        ]
    }
    
    # 保存JSON结果
    with open(output_path / "results_summary.json", 'w') as f:
        import json
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"✓ 可视化完成，结果保存到: {output_path}")
    return output_path

def main():
    """主函数"""
    checkpoint_path = "f:/Zhaoyang/Sparse2Full/runs/checkpoints/best.pth"
    output_dir = "f:/Zhaoyang/Sparse2Full/runs/true_observation_visualization"
    
    print("🚀 开始可视化best.pth模型...")
    
    try:
        # 1. 加载模型和配置
        model, config, checkpoint = load_model_and_config(checkpoint_path)
        
        # 2. 创建数据加载器
        test_loader, data_module = create_data_loader(config)
        
        # 3. 生成预测结果
        results = generate_predictions(model, test_loader, num_samples=5)
        
        # 4. 创建可视化
        output_path = create_comprehensive_visualizations(
            results, config, checkpoint, output_dir
        )
        
        print("\n🎉 可视化完成!")
        print(f"📁 输出目录: {output_path}")
        print("\n📊 生成的可视化文件:")
        for file_path in sorted(output_path.glob("*.png")):
            print(f"  - {file_path.name}")
        
        # 显示指标汇总
        print("\n📈 指标汇总:")
        with open(output_path / "results_summary.json", 'r') as f:
            import json
            summary = json.load(f)
            
        for metric, value in summary['metrics_summary'].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
            
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()