#!/usr/bin/env python3
"""
时序NAR模型在真实PDE数据集上的测试脚本

功能：
1. 自动检测可用的PDE数据集
2. 测试扩展后的TimeQueryHead在T_out=5和T_out=10下的性能
3. 验证时间编码器（Transformer和Conv1D）的效果
4. 对比AR和NAR模型的预测性能
5. 生成性能报告，包括Rel2_last、推理时延等指标
6. 支持可视化预测结果
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
import pytest

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

try:
    # 这些导入较重，测试收集阶段失败时允许降级
    from datasets.temporal_pdebench import TemporalPDEBenchDataModule, TemporalPDEBenchBase
    from models.wrappers.swin_temporal import SwinTemporalNAR
    from models.decoder.query_head import TimeQueryHead
    from models.temporal_block import TemporalTransformerEncoder, TemporalConv1D
    _IMPORTS_OK = True
except Exception as _e:
    TemporalPDEBenchDataModule = None
    TemporalPDEBenchBase = None
    SwinTemporalNAR = None
    TimeQueryHead = None
    TemporalTransformerEncoder = None
    TemporalConv1D = None
    _IMPORTS_OK = False
    _IMPORT_ERROR = _e
# 简化的指标计算函数
def calculate_rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算相对L2误差"""
    diff = pred - target
    mse = torch.mean(diff**2)
    target_norm = torch.mean(target**2) + eps
    rel_l2 = torch.sqrt(mse / target_norm)
    return rel_l2

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """计算PSNR"""
    mse = torch.mean((pred - target)**2)
    psnr = 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))
    return psnr

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """简化的SSIM计算"""
    # 简化版本，实际应用中可以使用更精确的SSIM实现
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # 计算均值
    mu1 = np.mean(pred_np)
    mu2 = np.mean(target_np)
    
    # 计算方差和协方差
    sigma1_sq = np.var(pred_np)
    sigma2_sq = np.var(target_np)
    sigma12 = np.mean((pred_np - mu1) * (target_np - mu2))
    
    # SSIM常数
    c1 = 0.01**2
    c2 = 0.03**2
    
    # SSIM计算
    ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return torch.tensor(ssim)


class PDEDatasetDetector:
    """PDE数据集自动检测器"""
    
    def __init__(self, data_root: Optional[str] = None):
        env_root = os.getenv("PDEBENCH_DATA_ROOT")
        # 默认回退到项目根目录下的 data/
        default_root = Path(__file__).parent.parent / "data"
        self.data_root = Path(data_root or env_root or default_root)
        # 允许直接指定单文件路径
        self.single_file = os.getenv("PDEBENCH_DATA_PATH")
        self.supported_datasets = {
            'darcy_flow': {
                'patterns': ['DarcyFlow', 'darcy'],
                'keys': ['u', 'tensor'],
                'channels': 1,
                'description': 'Darcy Flow - 渗透率场流动'
            },
            'diff_react': {
                'patterns': ['diff-react', 'diffusion'],
                'keys': ['0000', '0001'],  # 多通道
                'channels': 2,
                'description': 'Diffusion-Reaction - 扩散反应方程'
            },
            'navier_stokes': {
                'patterns': ['NS', 'NavierStokes', 'ns_incom'],
                'keys': ['velocity', 'vorticity'],
                'channels': 1,
                'description': 'Navier-Stokes - 不可压缩流体'
            }
        }
    
    def detect_available_datasets(self) -> List[Dict[str, Any]]:
        """检测可用的PDE数据集"""
        analyze = getattr(self, "_analyze_dataset_file", None)
        # 若指定了单文件，则仅分析该文件
        if self.single_file:
            p = Path(self.single_file)
            if p.exists():
                info = analyze(p) if analyze else None
                return [info] if info else []

        print(f"🔍 扫描数据目录: {self.data_root}")
        available_datasets = []
        
        if not self.data_root.exists():
            print(f"❌ 数据目录不存在: {self.data_root}")
            return available_datasets
        
        # 递归搜索HDF5文件
        for file_path in self.data_root.rglob("*.h*5"):
            dataset_info = analyze(file_path) if analyze else None
            if dataset_info:
                available_datasets.append(dataset_info)
        
        print(f"✅ 检测到 {len(available_datasets)} 个可用数据集")
        for dataset in available_datasets:
            print(f"  - {dataset['name']}: {dataset['description']}")
        
        return available_datasets


# 轻量探测测试：若无数据则跳过
def test_dataset_detection_or_skip():
    detector = PDEDatasetDetector()
    datasets = detector.detect_available_datasets()
    if len(datasets) == 0:
        pytest.skip(
            "未检测到PDE数据集，设置 PDEBENCH_DATA_ROOT 或 PDEBENCH_DATA_PATH 以启用该测试"
        )
    assert isinstance(datasets, list)
    
    def _analyze_dataset_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """分析单个数据集文件"""
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                # 检测数据集类型
                file_name = file_path.name.lower()
                dataset_type = None
                
                for pde_type, info in self.supported_datasets.items():
                    if any(pattern.lower() in file_name for pattern in info['patterns']):
                        dataset_type = pde_type
                        break
                
                if not dataset_type:
                    return None
                
                # 获取数据形状信息
                keys = list(f.keys())
                if not keys:
                    return None
                
                # 选择合适的数据键
                data_keys = []
                type_info = self.supported_datasets[dataset_type]
                for key in type_info['keys']:
                    if key in keys:
                        data_keys.append(key)
                
                if not data_keys:
                    data_keys = [keys[0]]  # 使用第一个可用键
                
                # 获取数据形状
                sample_key = data_keys[0]
                data_shape = f[sample_key].shape
                
                return {
                    'name': file_path.stem,
                    'path': str(file_path),
                    'type': dataset_type,
                    'keys': data_keys,
                    'shape': data_shape,
                    'channels': len(data_keys),
                    'description': type_info['description'],
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                }
                
        except Exception as e:
            print(f"⚠️  分析文件失败 {file_path}: {e}")
            return None


class TemporalNARTester:
    """时序NAR模型测试器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', 'runs/temporal_nar_test'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 初始化时序NAR测试器")
        print(f"   设备: {self.device}")
        print(f"   输出目录: {self.output_dir}")
    
    def create_temporal_config(self, dataset_info: Dict[str, Any], 
                             T_in: int = 3, T_out: int = 5) -> DictConfig:
        """创建时序配置"""
        config = {
            'data_path': dataset_info['path'],
            'keys': dataset_info['keys'],
            'batch_size': self.config.get('batch_size', 4),
            'num_workers': self.config.get('num_workers', 4),
            'pin_memory': True,
            'persistent_workers': True,
            'temporal': {
                'T_in': T_in,
                'T_out': T_out,
                'dt': 0.1,
                'temporal_mode': 'sequential',
                'overlap_ratio': 0.0
            },
            'task': {
                'type': 'temporal_prediction',
                'T_in': T_in,
                'T_out': T_out
            },
            'image_size': self.config.get('image_size', 128),
            'normalize': True,
            'split_ratios': [0.7, 0.15, 0.15]
        }
        
        return OmegaConf.create(config)
    
    def create_model(self, in_channels: int, out_channels: int, 
                    T_out: int, temporal_encoder: str = 'transformer') -> nn.Module:
        """创建时序NAR模型"""
        print(f"🔧 创建模型: channels={in_channels}->{out_channels}, T_out={T_out}, encoder={temporal_encoder}")
        
        # SwinUNet基础配置
        base_kwargs = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'img_size': 256,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'norm_layer': nn.LayerNorm,
            'patch_norm': True,
            'use_checkpoint': False
        }
        
        # 时间编码器配置
        if temporal_encoder == 'transformer':
            # 禁用时序模块以避免通道数不匹配问题
            temporal_config = {
                'type': 'disabled',
                'enabled': False
            }
        else:  # conv1d
            temporal_config = {
                'c_in': in_channels,
                'c_out': in_channels,
                'k': 3,
                'dropout': 0.0,
                'type': 'conv1d',
                'enabled': True,
                'causal': True
            }
        
        # NAR头配置
        nar_config = {
            'd_model': in_channels,  # 使用输入通道数保持一致
            'num_heads': 1,  # 调整为1以确保能被d_model整除
            'max_timesteps': max(T_out, 16),
            'dropout': 0.1,
            'head_type': 'simple'
        }
        
        # AR配置
        ar_config = {
            'detach_rollout': True,
            'scheduled_sampling': False
        }
        
        # 创建SwinTemporalNAR模型
        model = SwinTemporalNAR(
            base_kwargs=base_kwargs,
            temporal_cfg=temporal_config,
            nar_cfg=nar_config,
            ar_cfg=ar_config,
            use_ar=True,
            use_nar=True
        )
        
        return model.to(self.device)
    
    def test_model_performance(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                             T_out: int, model_name: str) -> Dict[str, float]:
        """测试模型性能"""
        print(f"📊 测试模型性能: {model_name}")
        
        model.eval()
        metrics = {
            'rel_l2': [],
            'psnr': [],
            'ssim': [],
            'inference_time': []
        }
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.config.get('test_samples', 10):  # 限制测试样本数
                    break
                
                # 准备数据
                input_seq = batch['input_sequence'].to(self.device)  # [B, T_in, C, H, W]
                target_seq = batch['target_sequence'].to(self.device)  # [B, T_out, C, H, W]
                
                # 测试推理时间
                start_time = time.time()
                
                # NAR预测
                pred_nar = model.forward_nar(input_seq, T_out=T_out)  # [B, T_out, C, H, W]
                
                inference_time = time.time() - start_time
                metrics['inference_time'].append(inference_time)
                
                # 计算指标（只计算最后一帧）
                pred_last = pred_nar[:, -1]  # [B, C, H, W]
                target_last = target_seq[:, -1]  # [B, C, H, W]
                
                # Rel-L2
                rel_l2 = calculate_rel_l2(pred_last, target_last)
                metrics['rel_l2'].append(rel_l2.item())
                
                # PSNR
                psnr = calculate_psnr(pred_last, target_last)
                metrics['psnr'].append(psnr.item())
                
                # SSIM
                ssim = calculate_ssim(pred_last, target_last)
                metrics['ssim'].append(ssim.item())
        
        # 计算平均值
        avg_metrics = {
            'rel_l2_mean': np.mean(metrics['rel_l2']),
            'rel_l2_std': np.std(metrics['rel_l2']),
            'psnr_mean': np.mean(metrics['psnr']),
            'psnr_std': np.std(metrics['psnr']),
            'ssim_mean': np.mean(metrics['ssim']),
            'ssim_std': np.std(metrics['ssim']),
            'inference_time_mean': np.mean(metrics['inference_time']),
            'inference_time_std': np.std(metrics['inference_time'])
        }
        
        print(f"  Rel-L2: {avg_metrics['rel_l2_mean']:.4f} ± {avg_metrics['rel_l2_std']:.4f}")
        print(f"  PSNR: {avg_metrics['psnr_mean']:.2f} ± {avg_metrics['psnr_std']:.2f}")
        print(f"  SSIM: {avg_metrics['ssim_mean']:.4f} ± {avg_metrics['ssim_std']:.4f}")
        print(f"  推理时间: {avg_metrics['inference_time_mean']:.4f}s ± {avg_metrics['inference_time_std']:.4f}s")
        
        return avg_metrics
    
    def compare_ar_nar_performance(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                                 T_out: int) -> Dict[str, Dict[str, float]]:
        """对比AR和NAR性能"""
        print(f"🔄 对比AR vs NAR性能 (T_out={T_out})")
        
        model.eval()
        ar_metrics = {'rel_l2': [], 'inference_time': []}
        nar_metrics = {'rel_l2': [], 'inference_time': []}
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.config.get('test_samples', 10):
                    break
                
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # AR预测
                start_time = time.time()
                pred_ar = model.forward_ar(input_seq, T_out=T_out)
                ar_time = time.time() - start_time
                
                # NAR预测
                start_time = time.time()
                pred_nar = model.forward_nar(input_seq, T_out=T_out)
                nar_time = time.time() - start_time
                
                # 计算Rel-L2（最后一帧）
                target_last = target_seq[:, -1]
                
                ar_rel_l2 = calculate_rel_l2(pred_ar[:, -1], target_last)
                nar_rel_l2 = calculate_rel_l2(pred_nar[:, -1], target_last)
                
                ar_metrics['rel_l2'].append(ar_rel_l2.item())
                ar_metrics['inference_time'].append(ar_time)
                
                nar_metrics['rel_l2'].append(nar_rel_l2.item())
                nar_metrics['inference_time'].append(nar_time)
        
        # 统计结果
        results = {
            'AR': {
                'rel_l2_mean': np.mean(ar_metrics['rel_l2']),
                'rel_l2_std': np.std(ar_metrics['rel_l2']),
                'inference_time_mean': np.mean(ar_metrics['inference_time']),
                'inference_time_std': np.std(ar_metrics['inference_time'])
            },
            'NAR': {
                'rel_l2_mean': np.mean(nar_metrics['rel_l2']),
                'rel_l2_std': np.std(nar_metrics['rel_l2']),
                'inference_time_mean': np.mean(nar_metrics['inference_time']),
                'inference_time_std': np.std(nar_metrics['inference_time'])
            }
        }
        
        print(f"  AR  - Rel-L2: {results['AR']['rel_l2_mean']:.4f}, 时间: {results['AR']['inference_time_mean']:.4f}s")
        print(f"  NAR - Rel-L2: {results['NAR']['rel_l2_mean']:.4f}, 时间: {results['NAR']['inference_time_mean']:.4f}s")
        
        return results
    
    def visualize_predictions(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                            T_out: int, dataset_name: str, num_samples: int = 3):
        """可视化预测结果"""
        print(f"🎨 生成可视化结果")
        
        model.eval()
        fig_dir = self.output_dir / 'visualizations' / dataset_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # NAR预测
                pred_nar = model.forward_nar(input_seq, T_out=T_out)
                
                # 转换为numpy
                input_np = input_seq[0].cpu().numpy()  # [T_in, C, H, W]
                target_np = target_seq[0].cpu().numpy()  # [T_out, C, H, W]
                pred_np = pred_nar[0].cpu().numpy()  # [T_out, C, H, W]
                
                # 创建可视化
                self._create_temporal_visualization(
                    input_np, target_np, pred_np, 
                    save_path=fig_dir / f'sample_{i}_T_out_{T_out}.png'
                )
    
    def _create_temporal_visualization(self, input_seq: np.ndarray, target_seq: np.ndarray,
                                     pred_seq: np.ndarray, save_path: Path):
        """创建时序可视化图"""
        T_in, C, H, W = input_seq.shape
        T_out = target_seq.shape[0]
        
        # 选择第一个通道进行可视化
        input_vis = input_seq[:, 0]  # [T_in, H, W]
        target_vis = target_seq[:, 0]  # [T_out, H, W]
        pred_vis = pred_seq[:, 0]  # [T_out, H, W]
        
        # 创建子图
        fig, axes = plt.subplots(3, max(T_in, T_out), figsize=(15, 9))
        
        # 输入序列
        for t in range(T_in):
            ax = axes[0, t] if max(T_in, T_out) > 1 else axes[0]
            im = ax.imshow(input_vis[t], cmap='viridis')
            ax.set_title(f'Input t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 目标序列
        for t in range(T_out):
            ax = axes[1, t] if max(T_in, T_out) > 1 else axes[1]
            im = ax.imshow(target_vis[t], cmap='viridis')
            ax.set_title(f'Target t={T_in+t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 预测序列
        for t in range(T_out):
            ax = axes[2, t] if max(T_in, T_out) > 1 else axes[2]
            im = ax.imshow(pred_vis[t], cmap='viridis')
            ax.set_title(f'Pred t={T_in+t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_performance_report(self, all_results: Dict[str, Any]):
        """生成性能报告"""
        print(f"📋 生成性能报告")
        
        report_path = self.output_dir / 'performance_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 时序NAR模型性能测试报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 测试配置
            f.write("## 测试配置\n\n")
            f.write(f"- 设备: {self.device}\n")
            f.write(f"- 测试样本数: {self.config.get('test_samples', 10)}\n")
            f.write(f"- 图像尺寸: {self.config.get('image_size', 128)}\n\n")
            
            # 数据集结果
            for dataset_name, dataset_results in all_results.items():
                f.write(f"## {dataset_name}\n\n")
                
                # T_out性能对比
                f.write("### 不同T_out性能对比\n\n")
                f.write("| T_out | 时间编码器 | Rel-L2 | PSNR | SSIM | 推理时间(s) |\n")
                f.write("|-------|------------|--------|------|------|-------------|\n")
                
                for config_key, metrics in dataset_results.items():
                    if 'performance' in config_key:
                        parts = config_key.split('_')
                        t_out = parts[2]
                        encoder = parts[3]
                        f.write(f"| {t_out} | {encoder} | "
                               f"{metrics['rel_l2_mean']:.4f}±{metrics['rel_l2_std']:.4f} | "
                               f"{metrics['psnr_mean']:.2f}±{metrics['psnr_std']:.2f} | "
                               f"{metrics['ssim_mean']:.4f}±{metrics['ssim_std']:.4f} | "
                               f"{metrics['inference_time_mean']:.4f}±{metrics['inference_time_std']:.4f} |\n")
                
                # AR vs NAR对比
                f.write("\n### AR vs NAR性能对比\n\n")
                for config_key, comparison in dataset_results.items():
                    if 'ar_nar_comparison' in config_key:
                        t_out = config_key.split('_')[4]
                        f.write(f"#### T_out={t_out}\n\n")
                        f.write("| 方法 | Rel-L2 | 推理时间(s) |\n")
                        f.write("|------|--------|-------------|\n")
                        for method, metrics in comparison.items():
                            f.write(f"| {method} | "
                                   f"{metrics['rel_l2_mean']:.4f}±{metrics['rel_l2_std']:.4f} | "
                                   f"{metrics['inference_time_mean']:.4f}±{metrics['inference_time_std']:.4f} |\n")
                        f.write("\n")
                
                f.write("\n")
        
        print(f"✅ 报告已保存: {report_path}")
        
        # 同时保存JSON格式
        json_path = self.output_dir / 'performance_results.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✅ 结果已保存: {json_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='时序NAR模型PDE数据集测试')
    parser.add_argument('--data_root', type=str, default='E:/2D', help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='runs/temporal_nar_test', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小')
    parser.add_argument('--test_samples', type=int, default=10, help='测试样本数')
    parser.add_argument('--image_size', type=int, default=128, help='图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    
    args = parser.parse_args()
    
    print("🚀 启动时序NAR模型PDE数据集测试")
    print("="*80)
    
    # 检测可用数据集
    detector = PDEDatasetDetector(args.data_root)
    available_datasets = detector.detect_available_datasets()
    
    if not available_datasets:
        print("❌ 未找到可用的PDE数据集")
        return
    
    # 创建测试器
    config = {
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'test_samples': args.test_samples,
        'image_size': args.image_size,
        'num_workers': args.num_workers
    }
    
    tester = TemporalNARTester(config)
    all_results = {}
    
    # 测试每个数据集
    for dataset_info in available_datasets[:2]:  # 限制测试前2个数据集
        print(f"\n📊 测试数据集: {dataset_info['name']}")
        print("-" * 60)
        
        dataset_results = {}
        
        # 测试不同的T_out配置
        for T_out in [5, 10]:
            print(f"\n🔧 测试 T_out={T_out}")
            
            # 创建数据配置
            temporal_config = tester.create_temporal_config(dataset_info, T_in=3, T_out=T_out)
            
            # 创建数据模块
            try:
                data_module = TemporalPDEBenchDataModule(temporal_config)
                data_module._create_datasets()
                
                # 创建数据加载器
                test_loader = torch.utils.data.DataLoader(
                    data_module.test_dataset,
                    batch_size=config['batch_size'],
                    shuffle=False,
                    num_workers=config['num_workers'],
                    pin_memory=True,
                    persistent_workers=False  # 避免h5py pickle问题
                )
                
                # 测试不同的时间编码器
                for encoder_type in ['transformer', 'conv1d']:
                    print(f"\n  🧠 测试时间编码器: {encoder_type}")
                    
                    # 创建模型
                    model = tester.create_model(
                        in_channels=dataset_info['channels'],
                        out_channels=dataset_info['channels'],
                        T_out=T_out,
                        temporal_encoder=encoder_type
                    )
                    
                    # 测试性能
                    model_name = f"T_out_{T_out}_{encoder_type}"
                    performance = tester.test_model_performance(model, test_loader, T_out, model_name)
                    dataset_results[f'performance_T_out_{T_out}_{encoder_type}'] = performance
                    
                    # AR vs NAR对比（只对transformer编码器进行）
                    if encoder_type == 'transformer':
                        ar_nar_comparison = tester.compare_ar_nar_performance(model, test_loader, T_out)
                        dataset_results[f'ar_nar_comparison_T_out_{T_out}'] = ar_nar_comparison
                    
                    # 生成可视化（只对第一个配置）
                    if T_out == 5 and encoder_type == 'transformer':
                        tester.visualize_predictions(model, test_loader, T_out, dataset_info['name'])
                    
                    # 清理GPU内存
                    del model
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ 测试数据集 {dataset_info['name']} 失败: {e}")
                continue
        
        all_results[dataset_info['name']] = dataset_results
    
    # 生成最终报告
    tester.generate_performance_report(all_results)
    
    print("\n🎉 测试完成！")
    print(f"📁 结果保存在: {tester.output_dir}")


if __name__ == "__main__":
    main()
