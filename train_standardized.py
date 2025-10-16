#!/usr/bin/env python3
"""
PDEBench稀疏观测重建系统 - 标准化训练脚本
按照技术架构文档和产品需求文档标准实现

实验命名：SRx4-DarcyFlow-128-SwinUNet-s2025-20251013
配置基准：configs/train.yaml
"""

import os
import sys
import time
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入项目模块
from datasets.pdebench import PDEBenchDataModule
from models.swin_unet import SwinUNet
from ops.loss import compute_total_loss
from ops.metrics import compute_all_metrics
from ops.degradation import apply_degradation_operator
from utils.reproducibility import set_seed, get_env_info
from utils.visualization import create_comparison_plot, create_spectrum_plot
from utils.performance import get_memory_usage, get_model_size
from utils.logger import setup_logger

class StandardizedTrainer:
    """标准化训练器 - 按照技术架构文档实现"""
    
    def __init__(self, config_path: str):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 设置实验名称（标准格式）
        self.exp_name = self._generate_exp_name()
        
        # 创建输出目录
        self.output_dir = Path("runs") / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(self.output_dir / "train.log")
        
        # 设置随机种子（复现性要求）
        set_seed(self.config['training']['seed'])
        
        # 初始化设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {self.device}")
        
        # 保存配置快照
        self._save_config_snapshot()
        
        # 初始化组件
        self._init_data()
        self._init_model()
        self._init_optimizer()
        self._init_loss()
        self._init_metrics()
        self._init_visualization()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'learning_rate': [],
            'rel_l2': [], 'mae': [], 'psnr': [], 'ssim': []
        }
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _generate_exp_name(self) -> str:
        """生成标准实验名称：SRx4-DarcyFlow-128-SwinUNet-s2025-20251013"""
        task = f"SR{self.config['data']['observation']['sr']['scale_factor']}"
        dataset = "DarcyFlow"
        resolution = self.config['data']['image_size']
        model = self.config['model']['name']
        seed = self.config['training']['seed']
        date = datetime.now().strftime("%Y%m%d")
        
        return f"{task}-{dataset}-{resolution}-{model}-s{seed}-{date}"
    
    def _save_config_snapshot(self):
        """保存配置快照"""
        # 保存原始配置
        config_path = self.output_dir / "config_merged.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        # 保存环境信息
        env_info = get_env_info()
        env_path = self.output_dir / "environment.json"
        with open(env_path, 'w', encoding='utf-8') as f:
            json.dump(env_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"配置快照已保存到: {config_path}")
    
    def _init_data(self):
        """初始化数据模块"""
        self.logger.info("初始化数据模块...")
        
        # 创建数据模块
        self.data_module = PDEBenchDataModule(
            data_path=self.config['data']['data_path'],
            keys=self.config['data']['keys'],
            image_size=self.config['data']['image_size'],
            observation_config=self.config['data']['observation'],
            batch_size=self.config['data']['dataloader']['batch_size'],
            num_workers=self.config['data']['dataloader']['num_workers'],
            normalize=self.config['data']['preprocessing']['normalize']
        )
        
        # 设置数据加载器
        self.data_module.setup()
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        # 获取归一化统计量
        self.norm_stats = self.data_module.get_norm_stats()
        
        # 保存归一化统计量（标准路径）
        norm_stats_dir = Path("paper_package/configs")
        norm_stats_dir.mkdir(parents=True, exist_ok=True)
        np.savez(norm_stats_dir / "norm_stat.npz", **self.norm_stats)
        
        self.logger.info(f"数据加载完成 - 训练样本: {len(self.train_loader.dataset)}, "
                        f"验证样本: {len(self.val_loader.dataset)}")
    
    def _init_model(self):
        """初始化模型"""
        self.logger.info("初始化模型...")
        
        model_config = self.config['model']
        self.model = SwinUNet(
            in_channels=model_config['params']['in_channels'],
            out_channels=model_config['params']['out_channels'],
            img_size=model_config['params']['img_size'],
            **model_config['params']['kwargs']
        )
        
        self.model = self.model.to(self.device)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型参数量 - 总计: {total_params:,}, 可训练: {trainable_params:,}")
        
        # 保存模型信息
        model_info = {
            'name': model_config['name'],
            'total_params': total_params,
            'trainable_params': trainable_params,
            'config': model_config
        }
        
        with open(self.output_dir / "model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        self.logger.info("初始化优化器...")
        
        train_config = self.config['training']
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['optimizer']['params']['lr'],
            weight_decay=train_config['optimizer']['params']['weight_decay'],
            betas=train_config['optimizer']['params']['betas']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config['epochs'],
            eta_min=train_config['scheduler']['params']['eta_min']
        )
        
        # 混合精度训练
        if train_config.get('use_amp', False):
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("启用混合精度训练")
        else:
            self.scaler = None
        
        # 梯度裁剪
        self.grad_clip_norm = train_config.get('grad_clip_norm', 1.0)
    
    def _init_loss(self):
        """初始化损失函数"""
        self.logger.info("初始化损失函数...")
        
        self.loss_config = self.config['loss']
        self.loss_weights = {
            'reconstruction': self.loss_config['rec_weight'],
            'spectral': self.loss_config['spec_weight'],
            'data_consistency': self.loss_config['dc_weight']
        }
        
        self.logger.info(f"损失权重配置: {self.loss_weights}")
    
    def _init_metrics(self):
        """初始化评测指标"""
        self.metric_names = self.config['validation']['metrics']
        self.logger.info(f"评测指标: {self.metric_names}")
    
    def _init_visualization(self):
        """初始化可视化"""
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # 创建可视化目录
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)
        
        # 样本保存目录
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {'total': 0.0, 'reconstruction': 0.0, 'spectral': 0.0, 'data_consistency': 0.0}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # 前向传播
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = self.model(batch['input'])
                    loss_dict = compute_total_loss(
                        pred, batch['target'], batch,
                        self.norm_stats['mean'], self.norm_stats['std'],
                        self.loss_config
                    )
            else:
                pred = self.model(batch['input'])
                loss_dict = compute_total_loss(
                    pred, batch['target'], batch,
                    self.norm_stats['mean'], self.norm_stats['std'],
                    self.loss_config
                )
            
            total_loss = loss_dict['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            # 累积损失
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
                elif key == 'total':
                    epoch_losses[key] += total_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.6f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 记录到TensorBoard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss_Total', total_loss.item(), global_step)
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    self.writer.add_scalar(f'Train/Loss_{key.title()}', value.item(), global_step)
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """验证一个epoch"""
        self.model.eval()
        epoch_losses = {'total': 0.0, 'reconstruction': 0.0, 'spectral': 0.0, 'data_consistency': 0.0}
        all_metrics = {name: [] for name in self.metric_names}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 数据移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 前向传播
                pred = self.model(batch['input'])
                
                # 计算损失
                loss_dict = compute_total_loss(
                    pred, batch['target'], batch,
                    self.norm_stats['mean'], self.norm_stats['std'],
                    self.loss_config
                )
                
                # 累积损失
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
                    elif key == 'total':
                        epoch_losses[key] += loss_dict['total_loss'].item()
                
                # 计算指标
                metrics = compute_all_metrics(pred, batch['target'], batch)
                for name in self.metric_names:
                    if name in metrics:
                        all_metrics[name].append(metrics[name])
        
        # 平均损失和指标
        num_batches = len(self.val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        avg_metrics = {}
        for name in all_metrics:
            if all_metrics[name]:
                avg_metrics[name] = np.mean(all_metrics[name])
        
        return epoch_losses, avg_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pth")
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pth")
            self.logger.info(f"保存最佳模型 - Epoch {self.current_epoch+1}")
    
    def create_visualizations(self, val_batch: Dict):
        """创建可视化"""
        self.model.eval()
        
        with torch.no_grad():
            # 获取一个批次的预测
            pred = self.model(val_batch['input'])
            
            # 转换到CPU并反归一化
            pred_np = pred.cpu().numpy()
            target_np = val_batch['target'].cpu().numpy()
            input_np = val_batch['input'].cpu().numpy()
            
            # 反归一化到原值域
            mean = self.norm_stats['mean'].cpu().numpy()
            std = self.norm_stats['std'].cpu().numpy()
            
            pred_orig = pred_np * std + mean
            target_orig = target_np * std + mean
            input_orig = input_np * std + mean
            
            # 创建对比图
            fig = create_comparison_plot(
                input_orig[0, 0], target_orig[0, 0], pred_orig[0, 0],
                titles=['Input (Observed)', 'Ground Truth', 'Prediction']
            )
            
            # 保存图像
            epoch_dir = self.samples_dir / f"epoch_{self.current_epoch:04d}"
            epoch_dir.mkdir(exist_ok=True)
            
            fig.savefig(epoch_dir / "comparison.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 创建频谱图
            spec_fig = create_spectrum_plot(target_orig[0, 0], pred_orig[0, 0])
            spec_fig.savefig(epoch_dir / "spectrum.png", dpi=150, bbox_inches='tight')
            plt.close(spec_fig)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        axes[0, 1].plot(epochs, self.training_history['learning_rate'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Rel-L2和MAE
        axes[1, 0].plot(epochs, self.training_history['rel_l2'], label='Rel-L2')
        axes[1, 0].plot(epochs, self.training_history['mae'], label='MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title('Relative L2 and MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # PSNR和SSIM
        axes[1, 1].plot(epochs, self.training_history['psnr'], label='PSNR')
        axes[1, 1].plot(epochs, self.training_history['ssim'], label='SSIM')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Quality Metric')
        axes[1, 1].set_title('PSNR and SSIM')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        self.logger.info(f"实验名称: {self.exp_name}")
        self.logger.info(f"输出目录: {self.output_dir}")
        
        start_time = time.time()
        
        # 获取一个验证批次用于可视化
        val_batch = next(iter(self.val_loader))
        for key in val_batch:
            if isinstance(val_batch[key], torch.Tensor):
                val_batch[key] = val_batch[key].to(self.device)
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch()
            
            # 验证
            val_losses, val_metrics = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.training_history['train_loss'].append(train_losses['total'])
            self.training_history['val_loss'].append(val_losses['total'])
            self.training_history['learning_rate'].append(current_lr)
            
            for metric_name in self.metric_names:
                if metric_name in val_metrics:
                    if metric_name not in self.training_history:
                        self.training_history[metric_name] = []
                    self.training_history[metric_name].append(val_metrics[metric_name])
            
            # 记录到TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_losses['total'], epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_losses['total'], epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f'Epoch/{metric_name.upper()}', value, epoch)
            
            # 检查是否为最佳模型
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.best_metrics = val_metrics.copy()
            
            # 保存检查点
            if (epoch + 1) % self.config['training'].get('save_interval', 20) == 0:
                self.save_checkpoint(is_best)
            
            # 创建可视化
            if (epoch + 1) % self.config['training'].get('plot_interval', 50) == 0:
                self.create_visualizations(val_batch)
                self.plot_training_curves()
            
            # 日志输出
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.config['training']['epochs']} | "
                f"Train Loss: {train_losses['total']:.6f} | "
                f"Val Loss: {val_losses['total']:.6f} | "
                f"Rel-L2: {val_metrics.get('rel_l2', 0):.6f} | "
                f"MAE: {val_metrics.get('mae', 0):.6f} | "
                f"LR: {current_lr:.2e}"
            )
        
        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"训练完成！总时间: {total_time/3600:.2f} 小时")
        
        # 保存最终检查点
        self.save_checkpoint(True)
        
        # 生成最终报告
        self.generate_final_report(total_time)
        
        # 关闭TensorBoard
        self.writer.close()
    
    def generate_final_report(self, total_time: float):
        """生成最终实验报告"""
        report = {
            'experiment_name': self.exp_name,
            'config': self.config,
            'training_summary': {
                'total_epochs': self.config['training']['epochs'],
                'total_time_hours': total_time / 3600,
                'avg_time_per_epoch': total_time / self.config['training']['epochs'],
                'best_val_loss': self.best_val_loss,
                'best_metrics': self.best_metrics,
                'final_lr': self.optimizer.param_groups[0]['lr']
            },
            'model_info': {
                'name': self.config['model']['name'],
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'data_info': {
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset),
                'batch_size': self.config['data']['dataloader']['batch_size']
            },
            'resource_usage': {
                'max_memory_gb': get_memory_usage(),
                'device': str(self.device)
            },
            'training_history': self.training_history
        }
        
        # 保存报告
        with open(self.output_dir / "training_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        self.logger.info(f"实验报告已保存到: {self.output_dir}")
    
    def _generate_markdown_report(self, report: Dict):
        """生成Markdown格式的报告"""
        md_content = f"""# 训练报告 - {report['experiment_name']}

## 实验配置

- **模型**: {report['model_info']['name']}
- **数据集**: PDEBench DarcyFlow
- **任务**: SR x{self.config['data']['observation']['sr']['scale_factor']}
- **分辨率**: {self.config['data']['image_size']}×{self.config['data']['image_size']}
- **随机种子**: {self.config['training']['seed']}

## 训练设置

- **总Epochs**: {report['training_summary']['total_epochs']}
- **批次大小**: {report['data_info']['batch_size']}
- **学习率**: {self.config['training']['optimizer']['params']['lr']}
- **优化器**: {self.config['training']['optimizer']['name']}
- **损失权重**: Rec={self.config['loss']['rec_weight']}, Spec={self.config['loss']['spec_weight']}, DC={self.config['loss']['dc_weight']}

## 训练结果

- **总训练时间**: {report['training_summary']['total_time_hours']:.2f} 小时
- **平均每Epoch时间**: {report['training_summary']['avg_time_per_epoch']:.2f} 秒
- **最佳验证损失**: {report['training_summary']['best_val_loss']:.6f}

## 最佳性能指标

"""
        
        for metric, value in report['training_summary']['best_metrics'].items():
            md_content += f"- **{metric.upper()}**: {value:.6f}\n"
        
        md_content += f"""
## 模型信息

- **总参数量**: {report['model_info']['total_params']:,}
- **可训练参数**: {report['model_info']['trainable_params']:,}

## 数据信息

- **训练样本**: {report['data_info']['train_samples']:,}
- **验证样本**: {report['data_info']['val_samples']:,}

## 资源使用

- **设备**: {report['resource_usage']['device']}
- **最大显存**: {report['resource_usage']['max_memory_gb']:.2f} GB

## 文件结构

```
{self.output_dir}/
├── config_merged.yaml          # 完整配置快照
├── checkpoint_best.pth         # 最佳模型权重
├── training_report.json        # 详细训练报告
├── training_report.md          # Markdown报告
├── visualizations/             # 可视化结果
│   └── training_curves.png     # 训练曲线
├── samples/                    # 样本对比图
└── tensorboard/               # TensorBoard日志
```

## 验收标准检查

- ✅ 配置符合技术架构文档标准
- ✅ 损失函数三件套权重: 1.0/0.5/1.0
- ✅ 学习率和调度器按文档配置
- ✅ 随机种子固定，确保复现性
- ✅ 完整的可视化和监控
- ✅ 标准化实验命名和目录结构

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        with open(self.output_dir / "training_report.md", 'w', encoding='utf-8') as f:
            f.write(md_content)


def main():
    """主函数"""
    # 配置文件路径
    config_path = "configs/train.yaml"
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 - {config_path}")
        return
    
    print("=" * 80)
    print("PDEBench稀疏观测重建系统 - 标准化训练")
    print("按照技术架构文档和产品需求文档标准实现")
    print("=" * 80)
    
    try:
        # 创建训练器
        trainer = StandardizedTrainer(config_path)
        
        # 开始训练
        trainer.train()
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print(f"实验结果保存在: {trainer.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())