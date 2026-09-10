#!/usr/bin/env python3
"""
从best.ckpt加载实际训练模型并生成真实的四联图可视化
Load actual trained model from best.ckpt and generate real four-panel visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
import h5py
from typing import Dict, Any, Tuple, Optional
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 添加项目路径
sys.path.append(str(PROJECT_ROOT))

def load_config_and_model(run_dir: Path, device: str = 'cuda:0') -> Tuple[Dict, nn.Module]:
    """加载配置文件和模型"""
    print(f"🔄 加载配置文件和模型...")
    
    # 加载合并后的配置
    config_path = run_dir / "config_merged.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 配置加载完成: SequentialSpatiotemporalModel")
    
    # 导入模型类 - 根据配置文件结构
    if 'sequential' in config and config['sequential']['enabled']:
        model_type = config['sequential']['model_type']
        if model_type == 'SequentialSpatiotemporalModel':
            # 从配置构建模型参数
            model_config = {
                'spatial_config': config['sequential']['spatial'],
                'temporal_config': config['sequential']['temporal'],
                'consistency_config': config['sequential'].get('consistency', {})
            }
            
            # 使用模型工厂创建
            from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporalModel
            
            # 构建数据配置
            data_config = {
                'dataset_name': 'diffusion-reaction',
                'resolution': 128,
                'in_channels': 1,
                'out_channels': 1
            }
            
            model = SequentialSpatiotemporalModel(
                spatial_config=model_config['spatial_config'],
                temporal_config=model_config['temporal_config'], 
                data_config=data_config,
                device=device
            )
        else:
            raise ValueError(f"不支持的序列模型类型: {model_type}")
    else:
        raise ValueError("配置中未启用序列模型")
    
    # 加载权重
    checkpoint_path = run_dir / "best.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"🔄 加载模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print(f"✅ 模型权重加载完成")
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载完成")
    return config, model

def load_test_data(config: Dict, device: str = 'cuda:0') -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """加载测试数据"""
    print(f"🔄 加载测试数据...")
    
    try:
        # 尝试从数据集配置加载
        data_config = config.get('data', {})
        dataset_name = data_config.get('name', 'RealDiffusionReactionDataset')
        
        if dataset_name == 'RealDiffusionReactionDataset' or 'real' in dataset_name.lower():
            from datasets.real_dr_dataset import RealDiffusionReactionDataset
            
            # 创建测试数据集
            test_dataset = RealDiffusionReactionDataset(
                data_path=data_config.get('data_path', './data/real_diffusion_reaction.h5'),
                T_in=data_config.get('T_in', 10),
                T_out=data_config.get('T_out', 10),
                split='test',
                train_ratio=data_config.get('train_ratio', 0.7),
                val_ratio=data_config.get('val_ratio', 0.15),
                test_ratio=data_config.get('test_ratio', 0.15),
                normalize=True,
                time_step_start=data_config.get('time_step_start', 0),
                time_step_end=data_config.get('time_step_end', 980),
                observation_params=data_config.get('observation_params', None)
            )
            
            # 获取一个样本
            if len(test_dataset) > 0:
                sample = test_dataset[0]
                obs = sample['input_sequence']  # [T_in, C, H, W]
                gt = sample['target_sequence']   # [T_out, C, H, W]
                
                # 添加batch维度
                obs = obs.unsqueeze(0).to(device)  # [1, T_in, C, H, W]
                gt = gt.unsqueeze(0).to(device)   # [1, T_out, C, H, W]
                
                print(f"✅ 测试数据加载完成: obs shape={obs.shape}, gt shape={gt.shape}")
                return obs, gt
            else:
                print("⚠️  测试数据集为空")
                return None
        else:
            print(f"⚠️  不支持的数据集类型: {dataset_name}")
            return None
            
    except Exception as e:
        print(f"⚠️  加载测试数据失败: {e}")
        return None

def generate_predictions(model: nn.Module, observations: torch.Tensor, config: Dict) -> torch.Tensor:
    """生成模型预测"""
    print(f"🔄 生成模型预测...")
    
    with torch.no_grad():
        # 根据模型类型进行预测
        if hasattr(model, 'forward'):
            output = model(observations)
            # SequentialSpatiotemporalModel 返回字典，提取最终预测
            if isinstance(output, dict) and 'final_pred' in output:
                predictions = output['final_pred']
            elif isinstance(output, dict) and 'spatial_pred' in output:
                predictions = output['spatial_pred']
            else:
                predictions = output
        else:
            raise ValueError("模型没有forward方法")
    
    print(f"✅ 预测完成: shape={predictions.shape}")
    return predictions

def create_real_four_panel_viz(
    observations: np.ndarray,
    ground_truth: np.ndarray, 
    predictions: np.ndarray,
    run_name: str,
    output_path: Path,
    sample_idx: int = 0,
    timestep: int = 0,
    channel: int = 0
) -> None:
    """创建真实的四联图可视化"""
    
    print(f"🔄 生成真实四联图可视化...")
    
    # 提取指定样本、时间步和通道的数据
    obs = observations[sample_idx, timestep, channel]  # [H, W]
    gt = ground_truth[sample_idx, timestep, channel]   # [H, W]
    pred = predictions[sample_idx, timestep, channel] # [H, W]
    error = np.abs(pred - gt)
    
    # 创建四联图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'AR模型真实预测结果 - 运行: {run_name}', fontsize=16, fontweight='bold')
    
    # 设置统一的colormap和范围
    vmin, vmax = gt.min(), gt.max()
    error_vmax = error.max()
    
    # 1. 观测数据 (低分辨率输入)
    im1 = axes[0, 0].imshow(obs, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('观测数据 (Observations)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('空间维度 X')
    axes[0, 0].set_ylabel('空间维度 Y')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 2. 真值数据 (高分辨率目标)
    im2 = axes[0, 1].imshow(gt, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('真值数据 (Ground Truth)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('空间维度 X')
    axes[0, 1].set_ylabel('空间维度 Y')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. 预测数据 (模型输出)
    im3 = axes[1, 0].imshow(pred, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('预测数据 (Predictions)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('空间维度 X')
    axes[1, 0].set_ylabel('空间维度 Y')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 4. 误差数据 (绝对误差)
    im4 = axes[1, 1].imshow(error, cmap='Reds', aspect='auto', vmin=0, vmax=error_vmax)
    axes[1, 1].set_title('绝对误差 (Absolute Error)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('空间维度 X')
    axes[1, 1].set_ylabel('空间维度 Y')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 添加统计信息
    mse = np.mean(error**2)
    mae = np.mean(error)
    rel_l2 = np.sqrt(mse) / (np.std(gt) + 1e-8)
    psnr = 20 * np.log10(gt.max() - gt.min()) - 10 * np.log10(mse) if mse > 0 else float('inf')
    
    stats_text = f'MSE: {mse:.6f}\\nMAE: {mae:.6f}\\nRel-L2: {rel_l2:.6f}\\nPSNR: {psnr:.2f} dB'
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 添加样本信息
    info_text = f'Sample: {sample_idx} | Timestep: {timestep} | Channel: {channel}'
    fig.text(0.98, 0.02, info_text, fontsize=10, ha='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 真实四联图已保存: {output_path}")
    print(f"📊 统计信息 - MSE: {mse:.6f}, MAE: {mae:.6f}, Rel-L2: {rel_l2:.6f}, PSNR: {psnr:.2f} dB")

def main():
    """主函数"""
    # 设置运行目录
    run_dir = PROJECT_ROOT / "runs/AR-DR2D-Debug-FNO2D-Staged-s2025-model_None_20251120_140708"
    
    # 确保可视化目录存在
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")
    
    try:
        # 加载配置和模型
        config, model = load_config_and_model(run_dir, device)
        
        # 加载测试数据
        test_data = load_test_data(config, device)
        
        if test_data is not None:
            observations, ground_truth = test_data
            
            # 生成预测
            predictions = generate_predictions(model, observations, config)
            
            # 转换为numpy数组
            obs_np = observations.cpu().numpy()
            gt_np = ground_truth.cpu().numpy()
            pred_np = predictions.cpu().numpy()
            
            # 生成真实四联图
            output_path = viz_dir / "obs_gt_pred_err_real.png"
            create_real_four_panel_viz(
                obs_np, gt_np, pred_np, 
                run_dir.name, output_path,
                sample_idx=0, timestep=0, channel=0
            )
            
            print(f"\\n🎉 真实四联图可视化完成！")
            print(f"📁 输出目录: {viz_dir}")
            print(f"📊 主要文件: {output_path.name}")
            
        else:
            print("⚠️  无法加载测试数据，请检查数据集配置")
            
    except Exception as e:
        print(f"❌ 生成真实四联图失败: {e}")
        print("💡 建议检查模型配置、数据路径或尝试手动运行测试脚本")

if __name__ == "__main__":
    main()
