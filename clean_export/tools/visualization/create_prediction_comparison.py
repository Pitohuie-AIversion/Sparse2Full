#!/usr/bin/env python3
"""
使用工作空间的可视化工具创建预测结果对比图
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.visualization import PDEBenchVisualizer, create_field_comparison
from datasets.pde_bench import PDEBenchDataset
from models.swin_unet import SwinUNet
from ops.degradation import SuperResolutionOperator


def load_model_and_data(run_dir: Path) -> Tuple[torch.nn.Module, torch.utils.data.Dataset, Dict]:
    """加载模型和数据集"""
    print(f"加载模型和数据: {run_dir}")
    
    # 加载配置
    config_path = run_dir / "config_merged.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载最佳模型
    best_model_path = run_dir / "best.pth"
    if not best_model_path.exists():
        # 尝试从checkpoints目录加载
        best_model_path = run_dir / "checkpoints" / "best.pth"
        if not best_model_path.exists():
            raise FileNotFoundError(f"最佳模型不存在: {best_model_path}")
    
    # 创建模型
    model_config = config['model']
    if 'MLP' in str(run_dir):
        # 简单MLP模型
        from models.mlp import MLPModel
        model = MLPModel(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            img_size=model_config.get('img_size', 128),
            hidden_dims=model_config.get('hidden_dims', [128, 256, 128]),
            mode=model_config.get('mode', 'coord'),
            coord_encoding=model_config.get('coord_encoding', 'positional'),
            coord_encoding_dim=model_config.get('coord_encoding_dim', 32),
            max_freq=model_config.get('max_freq', 10.0),
            activation=model_config.get('activation', 'relu'),
            dropout=model_config.get('dropout', 0.1)
        )
    else:
        # SwinUNet模型
        model = SwinUNet(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            img_size=model_config.get('img_size', 128)
        )
    
    # 加载权重
    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # 创建数据集
    data_config = config['data']# 创建数据集
    from datasets.pdebench import PDEBenchSR
    dataset = PDEBenchSR(
        data_path=data_config['data_path'],
        keys=data_config.get('keys', ['tensor']),
        split='test',
        scale=data_config['observation']['sr']['scale_factor'],
        sigma=data_config['observation']['sr']['blur_sigma'],
        blur_kernel=data_config['observation']['sr']['blur_kernel_size'],
        boundary=data_config['observation']['sr']['boundary_mode'],
        normalize=data_config.get('normalize', True),
        image_size=data_config.get('image_size', 128)
    )
    
    return model, dataset, config


def generate_predictions(model: torch.nn.Module, 
                        dataset: torch.utils.data.Dataset,
                        config: Dict,
                        num_samples: int = 5) -> List[Dict[str, np.ndarray]]:
    """生成预测结果"""
    print(f"生成 {num_samples} 个样本的预测结果...")
    
    # 创建降质算子
    obs_config = config['data']['observation']
    if obs_config['mode'] == 'SR':
        sr_config = obs_config['sr']
        degradation_op = SuperResolutionOperator(
            scale=sr_config['scale_factor'],
            sigma=sr_config.get('blur_sigma', 1.0),
            kernel_size=sr_config.get('blur_kernel_size', 5),
            boundary=sr_config.get('boundary_mode', 'mirror')
        )
    else:
        raise NotImplementedError(f"观测模式 {obs_config['mode']} 未实现")
    
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 选择样本索引
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            print(f"  处理样本 {i+1}/{num_samples} (索引 {idx})")
            
            # 获取数据
            data_dict = dataset[idx]
            gt = data_dict['target'].unsqueeze(0).to(device)  # [1, C, H, W]
            
            # 生成观测
            observed = degradation_op(gt)
            
            # 预测
            pred = model(observed)
            
            # 确保pred和gt尺寸一致用于指标计算
        if pred.shape != gt.shape:
            # 使用双线性插值上采样pred
            if pred.dim() == 3:  # [C, H, W]
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(0), 
                    size=gt.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            elif pred.dim() == 4:  # [B, C, H, W]
                pred = torch.nn.functional.interpolate(
                    pred, 
                    size=gt.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
        
        # 计算指标
        rel_l2 = torch.norm(pred - gt) / torch.norm(gt)
        mae = torch.mean(torch.abs(pred - gt))
        
        # 转换为numpy用于可视化
        gt_np = gt.squeeze().cpu().numpy()
        pred_np = pred.squeeze().cpu().numpy()
        obs_np = observed.squeeze().cpu().numpy()
        
        # 计算误差
        error_np = np.abs(gt_np - pred_np)
        
        results.append({
            'observed': obs_np,
            'ground_truth': gt_np,
            'prediction': pred_np,
            'error': error_np,
            'rel_l2': rel_l2.item(),
            'mae': mae.item(),
            'sample_idx': idx
        })
        
        print(f"    Rel-L2: {rel_l2.item():.4f}, MAE: {mae.item():.6f}")
    
    return results


def create_comparison_visualizations(results: List[Dict[str, np.ndarray]], 
                                   output_dir: Path):
    """创建对比可视化"""
    print(f"创建可视化图表，保存到: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化器
    visualizer = PDEBenchVisualizer(str(output_dir))
    
    # 为每个样本创建对比图
    for i, result in enumerate(results):
        sample_idx = result['sample_idx']
        rel_l2 = result['rel_l2']
        mae = result['mae']
        
        # 创建四联图 (Observed, GT, Pred, Error)
        observed_tensor = torch.from_numpy(result['observed']).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(result['ground_truth']).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(result['prediction']).unsqueeze(0).unsqueeze(0)
        
        save_name = f"sample_{sample_idx:03d}_comparison"
        title = f"Sample {sample_idx} - Rel-L2: {rel_l2:.4f}, MAE: {mae:.6f}"
        
        visualizer.create_quadruplet_visualization(
            observed_tensor, gt_tensor, pred_tensor,
            save_name=save_name,
            title=title
        )
        
        print(f"  ✓ 样本 {sample_idx} 对比图已保存")
    
    # 创建汇总统计图
    create_summary_plot(results, output_dir)


def create_summary_plot(results: List[Dict[str, np.ndarray]], output_dir: Path):
    """创建汇总统计图"""
    print("创建汇总统计图...")
    
    # 提取指标
    rel_l2_values = [r['rel_l2'] for r in results]
    mae_values = [r['mae'] for r in results]
    sample_indices = [r['sample_idx'] for r in results]
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Rel-L2 分布
    axes[0].bar(range(len(rel_l2_values)), rel_l2_values, color='steelblue', alpha=0.7)
    axes[0].set_title('Relative L2 Error by Sample', fontweight='bold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Rel-L2')
    axes[0].set_xticks(range(len(sample_indices)))
    axes[0].set_xticklabels([f'{idx}' for idx in sample_indices])
    axes[0].grid(True, alpha=0.3)
    
    # MAE 分布
    axes[1].bar(range(len(mae_values)), mae_values, color='darkorange', alpha=0.7)
    axes[1].set_title('Mean Absolute Error by Sample', fontweight='bold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('MAE')
    axes[1].set_xticks(range(len(sample_indices)))
    axes[1].set_xticklabels([f'{idx}' for idx in sample_indices])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    summary_path = output_dir / "metrics_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print(f"\n📊 预测结果统计:")
    print(f"  平均 Rel-L2: {np.mean(rel_l2_values):.4f} ± {np.std(rel_l2_values):.4f}")
    print(f"  平均 MAE: {np.mean(mae_values):.6f} ± {np.std(mae_values):.6f}")
    print(f"  最佳 Rel-L2: {np.min(rel_l2_values):.4f} (样本 {sample_indices[np.argmin(rel_l2_values)]})")
    print(f"  最差 Rel-L2: {np.max(rel_l2_values):.4f} (样本 {sample_indices[np.argmax(rel_l2_values)]})")
    
    print(f"  ✓ 汇总图已保存: {summary_path}")


def main():
    """主函数"""
    # 指定运行目录
    run_dir = Path("runs/SRx4-DarcyFlow-128-MLP-quick-s2025-20250111")
    
    if not run_dir.exists():
        print(f"❌ 运行目录不存在: {run_dir}")
        return
    
    try:
        # 加载模型和数据
        model, dataset, config = load_model_and_data(run_dir)
        print(f"✅ 模型和数据加载成功")
        
        # 生成预测结果
        results = generate_predictions(model, dataset, config, num_samples=5)
        print(f"✅ 预测结果生成完成")
        
        # 创建可视化
        output_dir = run_dir / "prediction_comparisons"
        create_comparison_visualizations(results, output_dir)
        print(f"✅ 可视化创建完成")
        
        print(f"\n🎉 所有可视化已保存到: {output_dir}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()