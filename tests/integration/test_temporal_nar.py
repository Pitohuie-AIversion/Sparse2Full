#!/usr/bin/env python3
"""
时序NAR模型真实数据集测试脚本

在真实的时序PDE数据集上测试新开发的时序NAR功能，包括：
- 数据加载和验证
- 时序NAR模型初始化
- 双头训练测试（AR+NAR）
- 单头推理测试（AR、NAR、集成）
- 性能对比分析
- 可视化结果展示
"""

import os
import sys
import time
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
import pytest

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入项目模块（数据模块缺失时跳过整个测试文件）
try:
    from datasets.temporal_pdebench import TemporalPDEBenchDataModule
except Exception:
    pytest.skip("缺少 datasets.temporal_pdebench 模块，跳过时序NAR真实数据测试", allow_module_level=True)

from models.wrappers.ar_nar_wrapper import ARNARWrapper
from utils.config import load_config
from utils.metrics import compute_metrics
from utils.visualization import create_temporal_comparison
from utils.performance import measure_model_performance
from ops.losses import compute_temporal_loss

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_temporal_nar.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TemporalNARTester:
    """时序NAR模型测试器"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.experiment.device)
        self.test_results = {}
        
        # 创建输出目录
        self.output_dir = Path(config.experiment.output_dir) / config.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        self._set_seed(config.experiment.seed)
        
        logger.info(f"初始化时序NAR测试器，输出目录: {self.output_dir}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """设置数据加载器"""
        logger.info("设置数据加载器...")
        
        try:
            # 创建数据模块
            data_module = TemporalPDEBenchDataModule(self.config)
            data_module.setup()
            
            # 创建数据加载器
            train_loader = DataLoader(
                data_module.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                data_module.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            logger.info(f"数据加载完成 - 训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")
            
            # 验证数据格式
            self._validate_data_format(train_loader)
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def _validate_data_format(self, dataloader: DataLoader):
        """验证数据格式"""
        logger.info("验证数据格式...")
        
        try:
            batch = next(iter(dataloader))
            x_seq, y_seq = batch
            
            logger.info(f"输入序列形状: {x_seq.shape}")  # [B, T_in, C, H, W]
            logger.info(f"输出序列形状: {y_seq.shape}")  # [B, T_out, C, H, W]
            
            # 检查维度
            assert len(x_seq.shape) == 5, f"输入序列应为5维，实际: {len(x_seq.shape)}"
            assert len(y_seq.shape) == 5, f"输出序列应为5维，实际: {len(y_seq.shape)}"
            assert x_seq.shape[1] == self.config.temporal.T_in, f"T_in不匹配: {x_seq.shape[1]} vs {self.config.temporal.T_in}"
            assert y_seq.shape[1] == self.config.temporal.T_out, f"T_out不匹配: {y_seq.shape[1]} vs {self.config.temporal.T_out}"
            
            logger.info("✅ 数据格式验证通过")
            
        except Exception as e:
            logger.error(f"❌ 数据格式验证失败: {e}")
            raise
    
    def setup_model(self) -> ARNARWrapper:
        """设置时序NAR模型"""
        logger.info("初始化时序NAR模型...")
        
        try:
            # 创建模型
            model = ARNARWrapper(
                model_config=self.config.model,
                loss_config=self.config.loss,
                training_config=self.config.train
            ).to(self.device)
            
            # 打印模型信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"模型参数总数: {total_params:,}")
            logger.info(f"可训练参数: {trainable_params:,}")
            
            # 测试模型前向传播
            self._test_model_forward(model)
            
            return model
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def _test_model_forward(self, model: ARNARWrapper):
        """测试模型前向传播"""
        logger.info("测试模型前向传播...")
        
        try:
            model.eval()
            with torch.no_grad():
                # 创建测试输入
                B, T_in, C, H, W = 2, self.config.temporal.T_in, 1, 32, 32
                T_out = self.config.temporal.T_out
                
                x_seq = torch.randn(B, T_in, C, H, W).to(self.device)
                y_seq = torch.randn(B, T_out, C, H, W).to(self.device)
                
                # 测试训练模式
                model.train()
                outputs = model(x_seq, y_seq, mode='train')
                
                logger.info(f"训练输出 - AR: {outputs['ar_pred'].shape}, NAR: {outputs['nar_pred'].shape}")
                
                # 测试推理模式
                model.eval()
                for mode in ['ar', 'nar', 'ensemble']:
                    pred = model(x_seq, mode=mode)
                    logger.info(f"{mode.upper()}推理输出: {pred.shape}")
                
                logger.info("✅ 模型前向传播测试通过")
                
        except Exception as e:
            logger.error(f"❌ 模型前向传播测试失败: {e}")
            raise
    
    def test_training(self, model: ARNARWrapper, train_loader: DataLoader, val_loader: DataLoader):
        """测试双头训练"""
        logger.info("开始双头训练测试...")
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.train.optimizer.lr,
            weight_decay=self.config.train.optimizer.weight_decay
        )
        
        # 训练循环
        model.train()
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.train.max_epochs):
            epoch_start = time.time()
            
            # 训练阶段
            train_loss = self._train_epoch(model, train_loader, optimizer, epoch)
            train_losses.append(train_loss)
            
            # 验证阶段
            if epoch % 2 == 0:  # 每2轮验证一次
                val_loss = self._validate_epoch(model, val_loader, epoch)
                val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{self.config.train.max_epochs} - "
                       f"Train Loss: {train_loss:.6f} - Time: {epoch_time:.2f}s")
        
        # 保存训练结果
        self.test_results['training'] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None
        }
        
        logger.info("✅ 双头训练测试完成")
        return model
    
    def _train_epoch(self, model: ARNARWrapper, dataloader: DataLoader, optimizer, epoch: int) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x_seq, y_seq) in enumerate(dataloader):
            x_seq = x_seq.to(self.device)
            y_seq = y_seq.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(x_seq, y_seq, mode='train')
            
            # 计算损失
            loss_dict = model.compute_loss(outputs, y_seq)
            total_loss_val = loss_dict['total_loss']
            
            # 反向传播
            total_loss_val.backward()
            
            # 梯度裁剪
            if hasattr(self.config.train, 'gradient_clip_val'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.gradient_clip_val)
            
            optimizer.step()
            
            total_loss += total_loss_val.item()
            num_batches += 1
            
            # 限制训练批次（测试用）
            if batch_idx >= 10:  # 只训练10个批次
                break
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, model: ARNARWrapper, dataloader: DataLoader, epoch: int) -> float:
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (x_seq, y_seq) in enumerate(dataloader):
                x_seq = x_seq.to(self.device)
                y_seq = y_seq.to(self.device)
                
                # 前向传播
                outputs = model(x_seq, y_seq, mode='train')
                
                # 计算损失
                loss_dict = model.compute_loss(outputs, y_seq)
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
                
                # 限制验证批次
                if batch_idx >= 5:  # 只验证5个批次
                    break
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def test_inference_modes(self, model: ARNARWrapper, val_loader: DataLoader):
        """测试不同推理模式"""
        logger.info("测试推理模式...")
        
        model.eval()
        inference_results = {}
        
        # 获取测试数据
        test_batch = next(iter(val_loader))
        x_seq, y_seq = test_batch
        x_seq = x_seq.to(self.device)
        y_seq = y_seq.to(self.device)
        
        # 限制测试样本数
        x_seq = x_seq[:2]  # 只测试2个样本
        y_seq = y_seq[:2]
        
        with torch.no_grad():
            for mode in ['ar', 'nar', 'ensemble']:
                logger.info(f"测试 {mode.upper()} 推理模式...")
                
                start_time = time.time()
                pred = model(x_seq, mode=mode)
                inference_time = time.time() - start_time
                
                # 计算指标
                metrics = self._compute_inference_metrics(pred, y_seq)
                
                inference_results[mode] = {
                    'prediction_shape': list(pred.shape),
                    'inference_time': inference_time,
                    'metrics': metrics
                }
                
                logger.info(f"{mode.upper()} - 推理时间: {inference_time:.4f}s, Rel-L2: {metrics['rel_l2']:.6f}")
        
        self.test_results['inference'] = inference_results
        logger.info("✅ 推理模式测试完成")
    
    def _compute_inference_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """计算推理指标"""
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # 计算相对L2误差
        rel_l2 = np.mean(np.linalg.norm(pred_np - target_np, axis=(2,3,4)) / 
                        (np.linalg.norm(target_np, axis=(2,3,4)) + 1e-8))
        
        # 计算MAE
        mae = np.mean(np.abs(pred_np - target_np))
        
        # 计算MSE
        mse = np.mean((pred_np - target_np) ** 2)
        
        return {
            'rel_l2': float(rel_l2),
            'mae': float(mae),
            'mse': float(mse)
        }
    
    def test_performance(self, model: ARNARWrapper):
        """测试模型性能"""
        logger.info("测试模型性能...")
        
        performance_results = {}
        
        # 测试参数量和FLOPs
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        performance_results['parameters'] = {
            'total': total_params,
            'trainable': trainable_params
        }
        
        # 测试推理延迟
        model.eval()
        B, T_in, C, H, W = 1, self.config.temporal.T_in, 1, 32, 32
        x_seq = torch.randn(B, T_in, C, H, W).to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(x_seq, mode='nar')
        
        # 测量延迟
        latencies = []
        with torch.no_grad():
            for _ in range(50):
                start_time = time.time()
                _ = model(x_seq, mode='nar')
                torch.cuda.synchronize()
                latencies.append(time.time() - start_time)
        
        performance_results['latency'] = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }
        
        # 测试显存使用
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(x_seq, mode='nar')
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        performance_results['memory'] = {
            'peak_memory_mb': peak_memory
        }
        
        self.test_results['performance'] = performance_results
        
        logger.info(f"性能测试完成:")
        logger.info(f"  参数量: {total_params:,}")
        logger.info(f"  推理延迟: {performance_results['latency']['mean']:.4f}±{performance_results['latency']['std']:.4f}s")
        logger.info(f"  峰值显存: {peak_memory:.2f}MB")
    
    def create_visualizations(self, model: ARNARWrapper, val_loader: DataLoader):
        """创建可视化结果"""
        if not self.config.test.visualization.enabled:
            return
        
        logger.info("创建可视化结果...")
        
        vis_dir = self.output_dir / self.config.test.visualization.save_dir
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        
        # 获取测试数据
        test_batch = next(iter(val_loader))
        x_seq, y_seq = test_batch
        x_seq = x_seq.to(self.device)
        y_seq = y_seq.to(self.device)
        
        # 只可视化第一个样本
        x_seq = x_seq[:1]
        y_seq = y_seq[:1]
        
        with torch.no_grad():
            # 获取不同模式的预测
            ar_pred = model(x_seq, mode='ar')
            nar_pred = model(x_seq, mode='nar')
            ensemble_pred = model(x_seq, mode='ensemble')
        
        # 转换为numpy
        x_np = x_seq.cpu().numpy()[0]  # [T_in, C, H, W]
        y_np = y_seq.cpu().numpy()[0]  # [T_out, C, H, W]
        ar_np = ar_pred.cpu().numpy()[0]
        nar_np = nar_pred.cpu().numpy()[0]
        ensemble_np = ensemble_pred.cpu().numpy()[0]
        
        # 创建对比图
        self._create_comparison_plot(
            x_np, y_np, ar_np, nar_np, ensemble_np, vis_dir
        )
        
        # 创建误差分析图
        self._create_error_analysis(
            y_np, ar_np, nar_np, ensemble_np, vis_dir
        )
        
        logger.info(f"✅ 可视化结果保存至: {vis_dir}")
    
    def _create_comparison_plot(self, x_seq, y_true, ar_pred, nar_pred, ensemble_pred, save_dir):
        """创建预测对比图"""
        T_out = y_true.shape[0]
        
        fig, axes = plt.subplots(5, T_out, figsize=(3*T_out, 15))
        if T_out == 1:
            axes = axes.reshape(-1, 1)
        
        for t in range(T_out):
            # 真实值
            im1 = axes[0, t].imshow(y_true[t, 0], cmap='viridis')
            axes[0, t].set_title(f'Ground Truth (t={t+1})')
            axes[0, t].axis('off')
            plt.colorbar(im1, ax=axes[0, t])
            
            # AR预测
            im2 = axes[1, t].imshow(ar_pred[t, 0], cmap='viridis')
            axes[1, t].set_title(f'AR Prediction (t={t+1})')
            axes[1, t].axis('off')
            plt.colorbar(im2, ax=axes[1, t])
            
            # NAR预测
            im3 = axes[2, t].imshow(nar_pred[t, 0], cmap='viridis')
            axes[2, t].set_title(f'NAR Prediction (t={t+1})')
            axes[2, t].axis('off')
            plt.colorbar(im3, ax=axes[2, t])
            
            # 集成预测
            im4 = axes[3, t].imshow(ensemble_pred[t, 0], cmap='viridis')
            axes[3, t].set_title(f'Ensemble Prediction (t={t+1})')
            axes[3, t].axis('off')
            plt.colorbar(im4, ax=axes[3, t])
            
            # NAR误差
            error = np.abs(nar_pred[t, 0] - y_true[t, 0])
            im5 = axes[4, t].imshow(error, cmap='Reds')
            axes[4, t].set_title(f'NAR Error (t={t+1})')
            axes[4, t].axis('off')
            plt.colorbar(im5, ax=axes[4, t])
        
        plt.tight_layout()
        plt.savefig(save_dir / 'prediction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_error_analysis(self, y_true, ar_pred, nar_pred, ensemble_pred, save_dir):
        """创建误差分析图"""
        T_out = y_true.shape[0]
        
        # 计算各时间步的误差
        ar_errors = []
        nar_errors = []
        ensemble_errors = []
        
        for t in range(T_out):
            ar_err = np.mean((ar_pred[t, 0] - y_true[t, 0]) ** 2)
            nar_err = np.mean((nar_pred[t, 0] - y_true[t, 0]) ** 2)
            ensemble_err = np.mean((ensemble_pred[t, 0] - y_true[t, 0]) ** 2)
            
            ar_errors.append(ar_err)
            nar_errors.append(nar_err)
            ensemble_errors.append(ensemble_err)
        
        # 绘制误差曲线
        plt.figure(figsize=(10, 6))
        timesteps = range(1, T_out + 1)
        
        plt.plot(timesteps, ar_errors, 'o-', label='AR', linewidth=2)
        plt.plot(timesteps, nar_errors, 's-', label='NAR', linewidth=2)
        plt.plot(timesteps, ensemble_errors, '^-', label='Ensemble', linewidth=2)
        
        plt.xlabel('Time Step')
        plt.ylabel('MSE')
        plt.title('Prediction Error vs Time Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """生成测试报告"""
        logger.info("生成测试报告...")
        
        report = {
            'test_config': OmegaConf.to_yaml(self.config),
            'test_timestamp': datetime.now().isoformat(),
            'results': self.test_results
        }
        
        # 保存JSON报告
        report_path = self.output_dir / 'test_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        logger.info(f"✅ 测试报告保存至: {report_path}")
    
    def _generate_markdown_report(self, report: Dict):
        """生成Markdown格式报告"""
        md_content = f"""# 时序NAR模型测试报告

## 测试概览
- **测试时间**: {report['test_timestamp']}
- **实验名称**: {self.config.experiment.name}
- **设备**: {self.config.experiment.device}

## 模型配置
- **输入时间步**: {self.config.temporal.T_in}
- **输出时间步**: {self.config.temporal.T_out}
- **图像尺寸**: {self.config.model.base_kwargs.img_size}
- **嵌入维度**: {self.config.model.base_kwargs.embed_dim}

## 性能结果
"""
        
        if 'performance' in self.test_results:
            perf = self.test_results['performance']
            md_content += f"""
### 模型参数
- **总参数量**: {perf['parameters']['total']:,}
- **可训练参数**: {perf['parameters']['trainable']:,}

### 推理性能
- **平均延迟**: {perf['latency']['mean']:.4f}±{perf['latency']['std']:.4f}s
- **峰值显存**: {perf['memory']['peak_memory_mb']:.2f}MB
"""
        
        if 'inference' in self.test_results:
            md_content += "\n### 推理模式对比\n\n| 模式 | Rel-L2 | MAE | 推理时间(s) |\n|------|--------|-----|-------------|\n"
            
            for mode, results in self.test_results['inference'].items():
                metrics = results['metrics']
                md_content += f"| {mode.upper()} | {metrics['rel_l2']:.6f} | {metrics['mae']:.6f} | {results['inference_time']:.4f} |\n"
        
        if 'training' in self.test_results:
            train_results = self.test_results['training']
            md_content += f"""
### 训练结果
- **最终训练损失**: {train_results['final_train_loss']:.6f}
- **最终验证损失**: {train_results.get('final_val_loss', 'N/A')}
"""
        
        # 保存Markdown报告
        md_path = self.output_dir / 'test_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def run_full_test(self):
        """运行完整测试流程"""
        logger.info("🚀 开始时序NAR模型完整测试")
        
        try:
            # 1. 设置数据
            train_loader, val_loader = self.setup_data()
            
            # 2. 设置模型
            model = self.setup_model()
            
            # 3. 测试训练
            model = self.test_training(model, train_loader, val_loader)
            
            # 4. 测试推理模式
            self.test_inference_modes(model, val_loader)
            
            # 5. 测试性能
            self.test_performance(model)
            
            # 6. 创建可视化
            self.create_visualizations(model, val_loader)
            
            # 7. 生成报告
            self.generate_report()
            
            logger.info("🎉 时序NAR模型测试完成！")
            
        except Exception as e:
            logger.error(f"❌ 测试过程中出现错误: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="时序NAR模型真实数据集测试")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/experiment/test_temporal_nar.yaml",
        help="测试配置文件路径"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="快速测试模式"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="计算设备"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 覆盖配置
    if args.quick:
        config.experiment.quick_test = True
        config.update(config.quick_test_config)
    
    if args.device:
        config.experiment.device = args.device
    
    # 创建测试器并运行
    tester = TemporalNARTester(config)
    tester.run_full_test()

if __name__ == "__main__":
    main()