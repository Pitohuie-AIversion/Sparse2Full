#!/usr/bin/env python3
"""
模型预测结果可视化脚本
对训练好的.pth模型文件生成的预测结果进行专业可视化

功能特点：
1. 自动搜索并加载runs目录下的.pth模型文件
2. 支持多种模型架构（时序NAR、UNet、SwinUNet等）
3. 生成GT vs 预测结果 vs 误差的对比图
4. 计算并显示详细的性能指标
5. 生成专业的可视化报告

使用方法：
python visualize_model_predictions.py
"""

import os
import sys
import json
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入项目模块
try:
    from datasets.pdebench import PDEBenchBase
    from datasets.pde_bench import PDEBenchDataset
except ImportError:
    from datasets.pdebench import PDEBenchBase as PDEBenchDataset

try:
    from models import *
except ImportError:
    pass

try:
    from ops.metrics import compute_all_metrics
    from ops.degradation import apply_degradation_operator
except ImportError:
    pass

try:
    from utils.config import load_config
    from utils.logger import setup_logger
except ImportError:
    # 简单的日志设置
    import logging
    def setup_logger(log_file):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

class ModelPredictionVisualizer:
    """模型预测结果可视化器"""
    
    def __init__(self, output_dir: str = "model_predictions_visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(str(self.output_dir / "visualization.log"))
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {self.device}")
        
        # 可视化配置
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 16
        
        # 模型架构映射
        self.model_architectures = {
            'swin_unet': 'models.swin_unet.SwinUNet',
            'swin_t': 'models.swin_t.SwinTransformer', 
            'unet': 'models.unet.UNet',
            'fno2d': 'models.fno2d.FNO2D',
            'hybrid': 'models.hybrid.HybridModel',
            'mlp': 'models.mlp.MLPModel',
            'temporal_nar': 'models.ar.wrapper.ARNARWrapper',
            'vit': 'models.vit.ViTModel'
        }
        
    def find_model_files(self, runs_dir: str = "runs") -> List[Dict[str, Any]]:
        """搜索runs目录下的.pth模型文件"""
        self.logger.info(f"搜索模型文件: {runs_dir}")
        
        model_files = []
        runs_path = Path(runs_dir)
        
        if not runs_path.exists():
            self.logger.warning(f"runs目录不存在: {runs_dir}")
            return model_files
            
        # 搜索所有.pth文件
        pth_patterns = [
            "**/*.pth",
            "**/checkpoints/*.pth", 
            "**/best.pth",
            "**/latest.pth"
        ]
        
        for pattern in pth_patterns:
            for pth_file in runs_path.glob(pattern):
                if pth_file.is_file():
                    # 解析模型信息
                    model_info = self._parse_model_info(pth_file)
                    if model_info:
                        model_files.append(model_info)
                        
        self.logger.info(f"找到 {len(model_files)} 个模型文件")
        return model_files
    
    def _parse_model_info(self, pth_file: Path) -> Optional[Dict[str, Any]]:
        """解析模型文件信息"""
        try:
            # 从路径推断模型类型
            path_parts = pth_file.parts
            model_type = None
            experiment_name = None
            
            # 查找实验目录名
            for part in path_parts:
                if any(arch in part.lower() for arch in self.model_architectures.keys()):
                    for arch in self.model_architectures.keys():
                        if arch in part.lower():
                            model_type = arch
                            experiment_name = part
                            break
                    break
            
            if not model_type:
                # 尝试从文件名推断
                filename = pth_file.stem.lower()
                for arch in self.model_architectures.keys():
                    if arch in filename:
                        model_type = arch
                        break
                        
            if not model_type:
                model_type = "unknown"
                
            return {
                'file_path': str(pth_file),
                'model_type': model_type,
                'experiment_name': experiment_name or pth_file.parent.name,
                'file_size': pth_file.stat().st_size,
                'modified_time': datetime.fromtimestamp(pth_file.stat().st_mtime)
            }
            
        except Exception as e:
            self.logger.warning(f"解析模型文件失败 {pth_file}: {e}")
            return None
    
    def load_model(self, model_info: Dict[str, Any]) -> Optional[nn.Module]:
        """加载模型"""
        try:
            self.logger.info(f"加载模型: {model_info['experiment_name']}")
            
            # 修复模型加载问题
            try:
                checkpoint = torch.load(model_info['file_path'], map_location=self.device, weights_only=False)
            except Exception as e:
                self.logger.warning(f"模型加载失败: {e}")
                return None
            
            # 获取模型配置
            model_config = self._get_model_config(checkpoint, model_info)
            
            # 创建模型实例
            model = self._create_model_instance(model_info['model_type'], model_config)
            
            if model is None:
                return None
                
            # 加载权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # 处理键名不匹配问题
            state_dict = self._fix_state_dict_keys(state_dict, model)
            
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"模型加载成功: {model_info['experiment_name']}")
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败 {model_info['experiment_name']}: {e}")
            return None
    
    def _get_model_config(self, checkpoint: Dict, model_info: Dict) -> Dict:
        """获取模型配置"""
        config = {}
        
        # 尝试从检查点获取配置
        if 'config' in checkpoint:
            if isinstance(checkpoint['config'], str):
                # YAML字符串
                try:
                    import yaml
                    config = yaml.safe_load(checkpoint['config'])
                except:
                    pass
            else:
                config = checkpoint['config']
        
        # 默认配置
        default_config = {
            'in_channels': 1,
            'out_channels': 1, 
            'img_size': 128,
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
            'mlp_ratio': 4.0,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1
        }
        
        # 合并配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
                
        return config
    
    def _create_model_instance(self, model_type: str, config: Dict) -> Optional[nn.Module]:
        """创建模型实例"""
        try:
            if model_type == 'swin_unet':
                from models.swin_unet import SwinUNet
                return SwinUNet(
                    in_channels=config.get('in_channels', 1),
                    out_channels=config.get('out_channels', 1),
                    img_size=config.get('img_size', 128)
                )
            elif model_type == 'unet':
                from models.unet import UNet
                return UNet(
                    in_channels=config.get('in_channels', 1),
                    out_channels=config.get('out_channels', 1)
                )
            elif model_type == 'fno2d':
                from models.fno2d import FNO2D
                return FNO2D(
                    in_channels=config.get('in_channels', 1),
                    out_channels=config.get('out_channels', 1),
                    modes1=config.get('modes1', 12),
                    modes2=config.get('modes2', 12),
                    width=config.get('width', 32)
                )
            elif model_type == 'temporal_nar':
                from models.ar.wrapper import ARNARWrapper
                from models.swin_unet import SwinUNet
                base_model = SwinUNet(
                    in_channels=config.get('in_channels', 1),
                    out_channels=config.get('out_channels', 1),
                    img_size=config.get('img_size', 128)
                )
                return ARNARWrapper(
                    base_model=base_model,
                    input_steps=config.get('input_steps', 1),
                    output_steps=config.get('output_steps', 3)
                )
            else:
                self.logger.warning(f"未知模型类型: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"创建模型实例失败 {model_type}: {e}")
            return None
    
    def _fix_state_dict_keys(self, state_dict: Dict, model: nn.Module) -> Dict:
        """修复状态字典键名"""
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        
        # 如果键名完全匹配，直接返回
        if model_keys == state_keys:
            return state_dict
            
        # 尝试移除前缀
        prefixes_to_remove = ['module.', 'model.', 'base_model.']
        
        for prefix in prefixes_to_remove:
            if all(key.startswith(prefix) for key in state_keys):
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key[len(prefix):]
                    new_state_dict[new_key] = value
                return new_state_dict
                
        return state_dict
    
    def load_test_data(self) -> Optional[torch.utils.data.DataLoader]:
        """加载测试数据"""
        try:
            # 查找可用的数据文件
            data_files = [
                "data/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5",
                "data/2D_DarcyFlow_beta1.0_Train.hdf5",
                "/tmp/2D_DarcyFlow_beta1.0_Train.hdf5"
            ]
            
            data_file = None
            for file_path in data_files:
                if os.path.exists(file_path):
                    data_file = file_path
                    break
                    
            if not data_file:
                self.logger.warning("未找到测试数据文件")
                return None
                
            # 创建数据集
            try:
                dataset = PDEBenchDataset(
                    data_path=data_file,
                    keys=["u"],
                    split="test",
                    normalize=True,
                    image_size=128
                )
            except Exception as e:
                self.logger.warning(f"使用PDEBenchDataset失败: {e}")
                # 尝试使用简单的数据加载
                return self._create_dummy_dataloader()
            
            # 创建数据加载器
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            
            self.logger.info(f"测试数据加载成功: {len(dataset)} 个样本")
            return dataloader
            
        except Exception as e:
            self.logger.error(f"测试数据加载失败: {e}")
            return self._create_dummy_dataloader()
    
    def _create_dummy_dataloader(self) -> torch.utils.data.DataLoader:
        """创建虚拟数据加载器用于测试"""
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=10):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # 生成随机测试数据
                observed = torch.randn(1, 32, 32)  # 低分辨率观测
                gt = torch.randn(1, 128, 128)      # 高分辨率真值
                return {'observed': observed, 'gt': gt}
        
        dataset = DummyDataset()
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    def generate_predictions(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                           num_samples: int = 5) -> List[Dict]:
        """生成预测结果"""
        self.logger.info(f"生成预测结果: {num_samples} 个样本")
        
        results = []
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="生成预测")):
                if i >= num_samples:
                    break
                    
                try:
                    # 获取输入和真值
                    if isinstance(batch, dict):
                        if 'observed' in batch and 'gt' in batch:
                            observed = batch['observed'].to(self.device)
                            gt = batch['gt'].to(self.device)
                        elif 'input' in batch and 'target' in batch:
                            observed = batch['input'].to(self.device)
                            gt = batch['target'].to(self.device)
                        else:
                            # 尝试使用第一个和第二个键
                            keys = list(batch.keys())
                            observed = batch[keys[0]].to(self.device)
                            gt = batch[keys[1]].to(self.device)
                    else:
                        # 假设是元组格式
                        observed, gt = batch
                        observed = observed.to(self.device)
                        gt = gt.to(self.device)
                    
                    # 模型预测
                    if hasattr(model, 'predict'):
                        pred = model.predict(observed)
                    else:
                        pred = model(observed)
                    
                    # 确保维度匹配
                    if pred.shape != gt.shape:
                        pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                    
                    # 计算误差
                    error = torch.abs(pred - gt)
                    
                    # 计算指标
                    metrics = self._compute_metrics(pred, gt)
                    
                    # 转换为numpy
                    observed_np = observed.squeeze().cpu().numpy()
                    gt_np = gt.squeeze().cpu().numpy()
                    pred_np = pred.squeeze().cpu().numpy()
                    error_np = error.squeeze().cpu().numpy()
                    
                    results.append({
                        'sample_id': i,
                        'observed': observed_np,
                        'gt': gt_np,
                        'pred': pred_np,
                        'error': error_np,
                        'metrics': metrics
                    })
                    
                except Exception as e:
                    self.logger.warning(f"样本 {i} 预测失败: {e}")
                    continue
                    
        self.logger.info(f"成功生成 {len(results)} 个预测结果")
        return results
    
    def _compute_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
        """计算评估指标"""
        try:
            # MSE
            mse = F.mse_loss(pred, gt).item()
            
            # PSNR
            psnr = 20 * torch.log10(1.0 / torch.sqrt(F.mse_loss(pred, gt))).item()
            
            # MAE
            mae = F.l1_loss(pred, gt).item()
            
            # 相对L2误差
            rel_l2 = torch.norm(pred - gt) / torch.norm(gt)
            rel_l2 = rel_l2.item()
            
            return {
                'mse': mse,
                'psnr': psnr,
                'mae': mae,
                'rel_l2': rel_l2
            }
            
        except Exception as e:
            self.logger.warning(f"指标计算失败: {e}")
            return {'mse': 0.0, 'psnr': 0.0, 'mae': 0.0, 'rel_l2': 0.0}
    
    def create_prediction_visualization(self, results: List[Dict], model_name: str):
        """创建预测结果可视化"""
        self.logger.info(f"创建预测可视化: {model_name}")
        
        # 创建模型专用目录
        model_dir = self.output_dir / model_name.replace('/', '_').replace('\\', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 创建样本对比图
        self._create_sample_comparison(results, model_dir, model_name)
        
        # 2. 创建指标统计图
        self._create_metrics_summary(results, model_dir, model_name)
        
        # 3. 创建误差分析图
        self._create_error_analysis(results, model_dir, model_name)
        
        # 4. 保存预测结果数据
        self._save_prediction_data(results, model_dir, model_name)
        
        self.logger.info(f"可视化完成: {model_dir}")
    
    def _create_sample_comparison(self, results: List[Dict], output_dir: Path, model_name: str):
        """创建样本对比图"""
        n_samples = len(results)
        
        # 创建大图
        fig = plt.figure(figsize=(20, 5 * n_samples))
        gs = GridSpec(n_samples, 4, figure=fig, hspace=0.3, wspace=0.2)
        
        for i, result in enumerate(results):
            # 观测值
            ax1 = fig.add_subplot(gs[i, 0])
            im1 = ax1.imshow(result['observed'], cmap='viridis', aspect='equal')
            ax1.set_title(f'Sample {i}: Observed')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # 真值
            ax2 = fig.add_subplot(gs[i, 1])
            im2 = ax2.imshow(result['gt'], cmap='viridis', aspect='equal')
            ax2.set_title(f'Sample {i}: Ground Truth')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            # 预测值
            ax3 = fig.add_subplot(gs[i, 2])
            im3 = ax3.imshow(result['pred'], cmap='viridis', aspect='equal')
            ax3.set_title(f'Sample {i}: Prediction')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, shrink=0.8)
            
            # 误差
            ax4 = fig.add_subplot(gs[i, 3])
            im4 = ax4.imshow(result['error'], cmap='Reds', aspect='equal')
            ax4.set_title(f'Sample {i}: Error')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, shrink=0.8)
            
            # 添加指标文本
            metrics = result['metrics']
            metrics_text = f"MSE: {metrics['mse']:.6f}\nPSNR: {metrics['psnr']:.2f}\nMAE: {metrics['mae']:.6f}\nRel-L2: {metrics['rel_l2']:.6f}"
            ax4.text(1.1, 0.5, metrics_text, transform=ax4.transAxes, 
                    verticalalignment='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle(f'{model_name} - Prediction Results Comparison', fontsize=16, y=0.98)
        
        # 保存图片
        save_path = output_dir / 'prediction_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"样本对比图已保存: {save_path}")
    
    def _create_metrics_summary(self, results: List[Dict], output_dir: Path, model_name: str):
        """创建指标统计图"""
        # 收集所有指标
        metrics_data = {
            'MSE': [r['metrics']['mse'] for r in results],
            'PSNR': [r['metrics']['psnr'] for r in results],
            'MAE': [r['metrics']['mae'] for r in results],
            'Rel-L2': [r['metrics']['rel_l2'] for r in results]
        }
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[i]
            
            # 柱状图
            sample_ids = [f'Sample {j}' for j in range(len(values))]
            bars = ax.bar(sample_ids, values, alpha=0.7, color=plt.cm.viridis(i/4))
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{metric_name} per Sample')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            
            # 添加平均值线
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_val:.4f}')
            ax.legend()
        
        plt.suptitle(f'{model_name} - Performance Metrics Summary', fontsize=14)
        plt.tight_layout()
        
        # 保存图片
        save_path = output_dir / 'metrics_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"指标统计图已保存: {save_path}")
    
    def _create_error_analysis(self, results: List[Dict], output_dir: Path, model_name: str):
        """创建误差分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 收集误差数据
        all_errors = []
        for result in results:
            all_errors.extend(result['error'].flatten())
        
        # 误差分布直方图
        axes[0, 0].hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 误差统计箱线图
        error_per_sample = [result['error'].flatten() for result in results]
        axes[0, 1].boxplot(error_per_sample, labels=[f'S{i}' for i in range(len(results))])
        axes[0, 1].set_title('Error Distribution per Sample')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 指标相关性
        metrics_matrix = np.array([[r['metrics']['mse'], r['metrics']['psnr'], 
                                  r['metrics']['mae'], r['metrics']['rel_l2']] for r in results])
        
        if len(results) > 1:
            corr_matrix = np.corrcoef(metrics_matrix.T)
            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title('Metrics Correlation')
            metric_names = ['MSE', 'PSNR', 'MAE', 'Rel-L2']
            axes[1, 0].set_xticks(range(len(metric_names)))
            axes[1, 0].set_yticks(range(len(metric_names)))
            axes[1, 0].set_xticklabels(metric_names)
            axes[1, 0].set_yticklabels(metric_names)
            plt.colorbar(im, ax=axes[1, 0])
            
            # 添加相关系数文本
            for i in range(len(metric_names)):
                for j in range(len(metric_names)):
                    text = axes[1, 0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                         ha="center", va="center", color="black")
        
        # 误差热图（平均）
        if results:
            mean_error = np.mean([result['error'] for result in results], axis=0)
            im = axes[1, 1].imshow(mean_error, cmap='Reds', aspect='equal')
            axes[1, 1].set_title('Mean Error Heatmap')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.suptitle(f'{model_name} - Error Analysis', fontsize=14)
        plt.tight_layout()
        
        # 保存图片
        save_path = output_dir / 'error_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"误差分析图已保存: {save_path}")
    
    def _save_prediction_data(self, results: List[Dict], output_dir: Path, model_name: str):
        """保存预测结果数据"""
        # 保存指标数据
        metrics_data = []
        for result in results:
            metrics_data.append({
                'sample_id': result['sample_id'],
                **result['metrics']
            })
        
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # 保存预测结果（numpy格式）
        predictions_file = output_dir / 'predictions.npz'
        np.savez_compressed(
            predictions_file,
            **{f"sample_{r['sample_id']}_observed": r['observed'] for r in results},
            **{f"sample_{r['sample_id']}_gt": r['gt'] for r in results},
            **{f"sample_{r['sample_id']}_pred": r['pred'] for r in results},
            **{f"sample_{r['sample_id']}_error": r['error'] for r in results}
        )
        
        self.logger.info(f"预测数据已保存: {output_dir}")
    
    def create_summary_report(self, all_results: Dict[str, List[Dict]]):
        """创建总结报告"""
        try:
            self.logger.info("创建总结报告")
            
            # 确保输出目录存在
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建HTML报告
            html_content = self._generate_html_report(all_results)
            
            report_file = self.output_dir / 'prediction_visualization_report.html'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # 创建Markdown报告
            md_content = self._generate_markdown_report(all_results)
            
            md_file = self.output_dir / 'prediction_visualization_report.md'
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            self.logger.info(f"报告已生成: {report_file}")
            
        except Exception as e:
            self.logger.error(f"创建总结报告失败: {e}")
    
    def _generate_html_report(self, all_results: Dict[str, List[Dict]]) -> str:
        """生成HTML报告"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>模型预测结果可视化报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .model-section {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .image-item {{ text-align: center; }}
        .image-item img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 模型预测结果可视化报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>总共分析了 {len(all_results)} 个模型</p>
    </div>
"""
        
        for model_name, results in all_results.items():
            if not results:
                continue
                
            # 计算平均指标
            avg_metrics = {}
            for metric in ['mse', 'psnr', 'mae', 'rel_l2']:
                values = [r['metrics'][metric] for r in results]
                avg_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            html += f"""
    <div class="model-section">
        <h2>📊 {model_name}</h2>
        <p>样本数量: {len(results)}</p>
        
        <h3>性能指标统计</h3>
        <table class="metrics-table">
            <tr>
                <th>指标</th>
                <th>平均值</th>
                <th>标准差</th>
                <th>最小值</th>
                <th>最大值</th>
            </tr>
"""
            
            for metric, stats in avg_metrics.items():
                html += f"""
            <tr>
                <td>{metric.upper()}</td>
                <td>{stats['mean']:.6f}</td>
                <td>{stats['std']:.6f}</td>
                <td>{stats['min']:.6f}</td>
                <td>{stats['max']:.6f}</td>
            </tr>
"""
            
            html += """
        </table>
        
        <h3>可视化图表</h3>
        <div class="image-grid">
"""
            
            model_dir_name = model_name.replace('/', '_').replace('\\', '_')
            for img_name, img_desc in [
                ('prediction_comparison.png', '预测结果对比'),
                ('metrics_summary.png', '指标统计'),
                ('error_analysis.png', '误差分析')
            ]:
                html += f"""
            <div class="image-item">
                <img src="{model_dir_name}/{img_name}" alt="{img_desc}">
                <p>{img_desc}</p>
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def _generate_markdown_report(self, all_results: Dict[str, List[Dict]]) -> str:
        """生成Markdown报告"""
        md = f"""# 🎯 模型预测结果可视化报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析模型数量**: {len(all_results)}

## 📋 报告概述

本报告展示了对训练好的.pth模型文件生成的预测结果进行的专业可视化分析。

"""
        
        for model_name, results in all_results.items():
            if not results:
                continue
                
            # 计算平均指标
            avg_metrics = {}
            for metric in ['mse', 'psnr', 'mae', 'rel_l2']:
                values = [r['metrics'][metric] for r in results]
                avg_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            
            md += f"""## 📊 {model_name}

**样本数量**: {len(results)}

### 性能指标

| 指标 | 平均值 | 标准差 |
|------|--------|--------|
"""
            
            for metric, stats in avg_metrics.items():
                md += f"| {metric.upper()} | {stats['mean']:.6f} | {stats['std']:.6f} |\n"
            
            md += f"""
### 可视化结果

- **预测对比图**: `{model_name.replace('/', '_').replace('\\', '_')}/prediction_comparison.png`
- **指标统计图**: `{model_name.replace('/', '_').replace('\\', '_')}/metrics_summary.png`  
- **误差分析图**: `{model_name.replace('/', '_').replace('\\', '_')}/error_analysis.png`

"""
        
        md += """## 🔍 使用说明

1. **查看预测结果**: 打开各模型目录下的 `prediction_comparison.png`
2. **分析性能指标**: 查看 `metrics_summary.png` 了解模型性能
3. **误差分析**: 通过 `error_analysis.png` 分析预测误差分布
4. **数据文件**: `metrics.json` 包含详细指标，`predictions.npz` 包含预测数据

## 📈 指标说明

- **MSE**: 均方误差，越小越好
- **PSNR**: 峰值信噪比，越大越好  
- **MAE**: 平均绝对误差，越小越好
- **Rel-L2**: 相对L2误差，越小越好
"""
        
        return md
    
    def run_visualization(self):
        """运行完整的可视化流程"""
        self.logger.info("🚀 开始模型预测结果可视化")
        
        # 1. 搜索模型文件
        model_files = self.find_model_files()
        
        if not model_files:
            self.logger.error("未找到任何模型文件")
            return
        
        # 2. 加载测试数据
        dataloader = self.load_test_data()
        
        if dataloader is None:
            self.logger.error("测试数据加载失败")
            return
        
        # 3. 处理每个模型
        all_results = {}
        
        for model_info in model_files[:5]:  # 限制处理前5个模型
            self.logger.info(f"处理模型: {model_info['experiment_name']}")
            
            # 加载模型
            model = self.load_model(model_info)
            
            if model is None:
                continue
            
            # 生成预测结果
            results = self.generate_predictions(model, dataloader, num_samples=3)
            
            if results:
                # 创建可视化
                self.create_prediction_visualization(results, model_info['experiment_name'])
                all_results[model_info['experiment_name']] = results
            
            # 清理内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 4. 创建总结报告
        if all_results:
            self.create_summary_report(all_results)
        
        self.logger.info("✅ 可视化完成！")
        print(f"\n🎉 可视化结果已保存到: {self.output_dir}")
        print(f"📊 查看报告: {self.output_dir / 'prediction_visualization_report.html'}")


def main():
    """主函数"""
    print("🎯 模型预测结果可视化工具")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = ModelPredictionVisualizer()
    
    # 运行可视化
    visualizer.run_visualization()


if __name__ == "__main__":
    main()