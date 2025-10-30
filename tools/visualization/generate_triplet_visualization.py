#!/usr/bin/env python3
"""
四联图可视化生成脚本 (重构版本)
使用 utils/visualization.py 中的统一接口

生成包含观测、真值、预测和误差的四联图可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
import yaml
from omegaconf import OmegaConf
import random
from typing import List, Tuple, Dict, Any

# 导入项目模块
import sys
sys.path.append('.')
from models import SwinUNet
from datasets.pdebench import PDEBenchSR
from ops.degradation import SuperResolutionOperator
from utils.config import load_config
from utils.metrics import compute_metrics
from utils.visualization import PDEBenchVisualizer

def load_trained_model(checkpoint_path: str, config: Dict[str, Any], device: str = 'cuda:0') -> nn.Module:
    """加载训练好的模型"""
    print(f"🔄 加载模型从: {checkpoint_path}")
    
    # 创建模型
    model_config = config['model']
    if model_config['name'] == 'SwinUNet':
        model = SwinUNet(**model_config['params'])
    else:
        raise ValueError(f"不支持的模型类型: {model_config['name']}")
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    return model

def load_validation_dataset(config: Dict[str, Any]) -> torch.utils.data.Dataset:
    """加载验证数据集"""
    print("🔄 加载验证数据集...")
    
    # 获取观测配置
    obs_config = config['data']['observation']
    
    # 创建数据集
    dataset = PDEBenchSR(
        data_path=config['data']['data_path'],
        keys=config['data']['keys'],
        split='val',
        splits_dir=None,  # 不使用splits文件
        image_size=config['data']['image_size'],
        normalize=False,  # 暂时不使用归一化避免路径问题
        scale=obs_config['sr']['scale_factor'],
        sigma=obs_config['sr']['blur_sigma'],
        blur_kernel=obs_config['sr']['blur_kernel_size'],
        boundary=obs_config['sr']['boundary_mode']
    )
    
    print(f"✅ 验证数据集加载成功，样本数: {len(dataset)}")
    return dataset

def select_representative_samples(dataset: torch.utils.data.Dataset, 
                                model: nn.Module, 
                                degradation_op: Any,
                                device: str,
                                num_samples: int = 5) -> List[int]:
    """选择代表性样本（包括好的和差的预测结果）"""
    print(f"🔄 从{len(dataset)}个样本中选择{num_samples}个代表性样本...")
    
    # 随机选择一些样本进行评估
    sample_indices = random.sample(range(len(dataset)), min(50, len(dataset)))
    sample_errors = []
    
    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            data_dict = dataset[idx]
            gt = data_dict['target'].unsqueeze(0).to(device)  # [1, C, H, W]
            
            # 生成观测
            observed = degradation_op(gt)
            
            # 将观测数据上采样到模型期望的尺寸
            if observed.shape[-2:] != (128, 128):
                observed = F.interpolate(observed, size=(128, 128), mode='bilinear', align_corners=False)
            
            # 预测
            pred = model(observed)
            
            # 计算误差
            error = torch.mean((pred - gt) ** 2).item()
            sample_errors.append((idx, error))
    
    # 按误差排序
    sample_errors.sort(key=lambda x: x[1])
    
    # 选择代表性样本：最好的2个，最差的2个，中等的1个
    selected_indices = []
    
    # 最好的2个
    selected_indices.extend([sample_errors[i][0] for i in range(min(2, len(sample_errors)))])
    
    # 最差的2个
    selected_indices.extend([sample_errors[i][0] for i in range(max(0, len(sample_errors)-2), len(sample_errors))])
    
    # 中等的1个
    if len(sample_errors) > 4:
        mid_idx = len(sample_errors) // 2
        selected_indices.append(sample_errors[mid_idx][0])
    
    # 确保不重复且数量正确
    selected_indices = list(set(selected_indices))[:num_samples]
    
    print(f"✅ 选择了{len(selected_indices)}个代表性样本")
    return selected_indices

def generate_predictions(model: nn.Module, 
                        dataset: torch.utils.data.Dataset,
                        sample_indices: List[int],
                        degradation_op: Any,
                        device: str) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """生成预测结果，返回观测、真实、预测、误差数据"""
    print("🔄 生成预测结果...")
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for idx in sample_indices:
            # 获取真实数据
            data_dict = dataset[idx]
            gt = data_dict['target'].unsqueeze(0).to(device)  # [1, C, H, W]
            
            # 生成观测
            observed = degradation_op(gt)
            observed_original = observed.clone()  # 保存原始观测数据
            
            # 将观测数据上采样到模型期望的尺寸
            if observed.shape[-2:] != (128, 128):
                observed = F.interpolate(observed, size=(128, 128), mode='bilinear', align_corners=False)
            
            # 预测
            pred = model(observed)
            
            # 转换为numpy数组 (取第一个通道)
            observed_np = observed_original[0, 0].cpu().numpy()  # 原始32x32观测数据
            gt_np = gt[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            error_np = np.abs(gt_np - pred_np)
            
            results.append((observed_np, gt_np, pred_np, error_np))
            
            # 计算指标
            metrics = compute_metrics(pred, gt)
            rel_l2 = metrics['rel_l2'].item()
            psnr = metrics['psnr'].item()
            ssim = metrics['ssim'].item()
            
            print(f"  样本 {idx}: Rel-L2={rel_l2:.4f}, PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
    
    print("✅ 预测结果生成完成")
    return results

def create_individual_quadruplet_visualization(observed_data: np.ndarray,
                                              gt_data: np.ndarray,
                                              pred_data: np.ndarray, 
                                              error_data: np.ndarray,
                                              sample_idx: int,
                                              output_dir: Path,
                                              metrics: Dict[str, float] = None) -> None:
    """创建单个样本的四联图可视化 - 使用utils接口"""
    
    # 使用utils中的统一接口
    save_path = create_quadruplet_visualization(
        observed_data=observed_data,
        gt_data=gt_data,
        pred_data=pred_data,
        error_data=error_data,
        sample_idx=sample_idx,
        metrics=metrics,
        save_dir=str(output_dir),
        output_format='svg'
    )
    
    print(f"✅ 四联图已保存: {save_path}")

def create_combined_quadruplet_visualization_wrapper(results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                                                   sample_indices: List[int],
                                                   output_dir: Path,
                                                   all_metrics: List[Dict[str, float]] = None) -> None:
    """创建组合的四联图可视化 - 使用utils接口"""
    
    # 使用utils中的统一接口
    save_path = create_combined_quadruplet_visualization(
        results=results,
        sample_indices=sample_indices,
        all_metrics=all_metrics,
        save_dir=str(output_dir),
        output_format='svg'
    )
    
    print(f"✅ 组合四联图已保存: {save_path}")

def main():
    """主函数"""
    print("🚀 开始生成四联图可视化...")
    
    # 设置路径
    config_path = Path('configs/train.yaml')
    checkpoint_path = Path('runs/checkpoints/best.pth')
    output_dir = Path('runs/visualization')
    output_dir.mkdir(exist_ok=True)
    
    # 检查文件存在性
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 加载配置
    print("🔄 加载配置文件...")
    config = load_config(str(config_path))
    device = config['experiment']['device']
    
    # 设置随机种子
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    random.seed(config['experiment']['seed'])
    
    # 加载模型
    model = load_trained_model(str(checkpoint_path), config, device)
    
    # 加载验证数据集
    dataset = load_validation_dataset(config)
    
    # 创建退化算子
    print("🔄 创建观测算子...")
    obs_config = config['data']['observation']['sr']
    degradation_op = SuperResolutionOperator(
        scale=obs_config['scale_factor'],
        sigma=obs_config['blur_sigma'],
        kernel_size=obs_config['blur_kernel_size'],
        boundary=obs_config['boundary_mode']
    )
    
    # 选择代表性样本
    sample_indices = select_representative_samples(dataset, model, degradation_op, device, num_samples=5)
    
    # 生成预测结果
    results = generate_predictions(model, dataset, sample_indices, degradation_op, device)
    
    # 计算详细指标
    print("🔄 计算详细指标...")
    all_metrics = []
    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            data_dict = dataset[idx]
            gt = data_dict['target'].unsqueeze(0).to(device)
            observed = degradation_op(gt)
            # 将观测数据上采样到模型期望的尺寸
            if observed.shape[-2:] != (128, 128):
                observed = F.interpolate(observed, size=(128, 128), mode='bilinear', align_corners=False)
            pred = model(observed)
            metrics = compute_metrics(pred, gt)
            all_metrics.append({
                'rel_l2': metrics['rel_l2'].item(),
                'psnr': metrics['psnr'].item(),
                'ssim': metrics['ssim'].item()
            })
    
    # 生成单独的四联图
    print("🔄 生成单独的四联图...")
    for i, (result, sample_idx) in enumerate(zip(results, sample_indices)):
        observed_data, gt_data, pred_data, error_data = result
        metrics = all_metrics[i] if i < len(all_metrics) else None
        create_individual_quadruplet_visualization(observed_data, gt_data, pred_data, error_data, sample_idx, output_dir, metrics)
    
    # 生成组合四联图
    print("🔄 生成组合四联图...")
    create_combined_quadruplet_visualization_wrapper(results, sample_indices, output_dir, all_metrics)
    
    print(f"\n🎉 四联图可视化完成！")
    print(f"📁 结果保存在: {output_dir}")
    print(f"📊 生成了 {len(sample_indices)} 个单独四联图 + 1 个组合图")
    
    # 打印样本统计信息
    print(f"\n📋 样本统计信息:")
    for i, (sample_idx, metrics) in enumerate(zip(sample_indices, all_metrics)):
        print(f"  样本 {sample_idx}: Rel-L2={metrics['rel_l2']:.4f}, PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")

if __name__ == "__main__":
    main()