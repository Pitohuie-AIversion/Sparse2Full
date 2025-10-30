#!/usr/bin/env python3
"""
测试时序NAR模型在不同T_out设置下的多时步预测性能
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
from omegaconf import DictConfig, OmegaConf
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.pdebench import PDEBenchDataModule
from models.wrappers.ar_nar_wrapper import ARNARWrapper
from utils.metrics import MetricsCalculator
from utils.visualization import TemporalVisualizer

class TemporalNARMultiToutTester:
    """时序NAR模型多T_out测试器"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        
        # 加载配置
        self.config = OmegaConf.load(self.config_path)
        
        # 设置输出目录
        self.output_dir = Path("runs/temporal_nar_multi_tout_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {self.device}")
        
        # 测试的T_out值
        self.test_t_outs = [3, 5, 10, 15, 20]
        
        # 结果存储
        self.results = {}
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator(image_size=(128, 128))
        
    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / "multi_tout_test.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self, t_out: int) -> ARNARWrapper:
        """加载模型并调整T_out"""
        self.logger.info(f"加载模型，T_out={t_out}")
        
        # 创建模型配置副本并修改T_out
        model_config = OmegaConf.create(self.config.model)
        model_config.temporal.T_out = t_out
        
        # 初始化模型
        model = ARNARWrapper(
            model_config=model_config,
            loss_config=self.config.loss,
            training_config=self.config.train
        )
        
        # 加载检查点 - 尝试多个可能的检查点文件
        checkpoint_files = [
            self.checkpoint_path,
            self.checkpoint_path.parent / "last.pth",
            self.checkpoint_path.parent / "final.pth",
            self.checkpoint_path.parent / "epoch_299.pth"
        ]
        
        checkpoint_loaded = False
        for ckpt_path in checkpoint_files:
            if ckpt_path.exists():
                try:
                    checkpoint = torch.load(ckpt_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(f"成功加载检查点: {ckpt_path}")
                    checkpoint_loaded = True
                    break
                except Exception as e:
                    self.logger.warning(f"加载检查点失败 {ckpt_path}: {e}")
                    continue
        
        if not checkpoint_loaded:
            self.logger.warning("未找到有效检查点，使用随机初始化的模型")
            
        model.to(self.device)
        model.eval()
        
        return model
        
    def _load_test_data(self, t_out: int) -> torch.utils.data.DataLoader:
        """加载测试数据"""
        # 创建数据配置副本并修改T_out
        data_config = OmegaConf.create(self.config.data)
        data_config.temporal.T_out = t_out
        
        # 创建时序数据模块
        from datasets.temporal_pdebench import TemporalPDEBenchDataModule
        data_module = TemporalPDEBenchDataModule(data_config)
        
        # TemporalPDEBenchDataModule没有setup方法，直接获取测试数据加载器
        return data_module.test_dataloader()
        
    def _evaluate_model(self, model: ARNARWrapper, test_loader: torch.utils.data.DataLoader, t_out: int) -> Dict:
        """评估模型性能"""
        self.logger.info(f"评估模型性能，T_out={t_out}")
        
        metrics = {
            'rel_l2': [],
            'mae': [],
            'psnr': [],
            'ssim': [],
            'ar_rel_l2': [],
            'nar_rel_l2': []
        }
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 5:  # 只测试前5个batch
                    break
                    
                # 数据移到设备
                x = batch['input'].to(self.device)  # [B, T_in, C, H, W]
                y = batch['target'].to(self.device)  # [B, T_out, C, H, W]
                
                # 模型预测
                output = model(x)
                
                # AR和NAR预测
                if isinstance(output, dict):
                    ar_pred = output.get('ar_prediction', None)
                    nar_pred = output.get('nar_prediction', None)
                    # 使用NAR预测作为主要预测（如果可用）
                    pred = nar_pred if nar_pred is not None else ar_pred
                else:
                    # 如果output不是字典，直接使用
                    pred = output
                    ar_pred = None
                    nar_pred = None
                
                if pred is None:
                    self.logger.warning(f"Batch {batch_idx}: 没有有效预测")
                    continue
                
                # 计算指标
                for b in range(pred.shape[0]):
                    pred_b = pred[b]  # [T_out, C, H, W]
                    target_b = y[b]   # [T_out, C, H, W]
                    
                    # 整体指标 - 处理5D张量 [B, T, C, H, W]
                    pred_5d = pred_b.unsqueeze(0).unsqueeze(0)  # [1, 1, T_out, C, H, W]
                    target_5d = target_b.unsqueeze(0).unsqueeze(0)  # [1, 1, T_out, C, H, W]
                    
                    rel_l2 = self.metrics_calculator.compute_rel_l2(pred_5d, target_5d)
                    mae = self.metrics_calculator.compute_mae(pred_5d, target_5d)
                    psnr = self.metrics_calculator.compute_psnr(pred_5d, target_5d)
                    ssim = self.metrics_calculator.compute_ssim(pred_5d, target_5d)
                    
                    metrics['rel_l2'].append(rel_l2.mean().item())
                    metrics['mae'].append(mae.mean().item())
                    metrics['psnr'].append(psnr.mean().item())
                    metrics['ssim'].append(ssim.mean().item())
                    
                    # AR vs NAR对比
                    if ar_pred is not None:
                        ar_5d = ar_pred[b].unsqueeze(0).unsqueeze(0)
                        ar_rel_l2 = self.metrics_calculator.compute_rel_l2(ar_5d, target_5d)
                        metrics['ar_rel_l2'].append(ar_rel_l2.mean().item())
                    
                    if nar_pred is not None:
                        nar_5d = nar_pred[b].unsqueeze(0).unsqueeze(0)
                        nar_rel_l2 = self.metrics_calculator.compute_rel_l2(nar_5d, target_5d)
                        metrics['nar_rel_l2'].append(nar_rel_l2.mean().item())
                    
                    # 保存预测和目标用于可视化
                    predictions.append(pred_b.cpu().numpy())
                    targets.append(target_b.cpu().numpy())
        
        # 计算平均指标
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                avg_metrics[key] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'metrics': avg_metrics,
            'predictions': predictions[:3],  # 保存前3个样本用于可视化
            'targets': targets[:3]
        }
        
    def _create_visualizations(self, results: Dict):
        """创建可视化"""
        self.logger.info("创建可视化...")
        
        # 1. T_out vs 性能指标图
        self._plot_tout_vs_metrics(results)
        
        # 2. AR vs NAR对比图
        self._plot_ar_vs_nar_comparison(results)
        
        # 3. 预测样本可视化
        self._plot_prediction_samples(results)
        
        # 4. 误差演化图
        self._plot_error_evolution(results)
        
    def _plot_tout_vs_metrics(self, results: Dict):
        """绘制T_out vs 性能指标图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('T_out vs Performance Metrics', fontsize=16)
        
        metrics_to_plot = ['rel_l2', 'mae', 'psnr', 'ssim']
        metric_names = ['Relative L2', 'MAE', 'PSNR', 'SSIM']
        
        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            t_outs = []
            means = []
            stds = []
            
            for t_out in self.test_t_outs:
                if t_out in results and metric in results[t_out]['metrics']:
                    t_outs.append(t_out)
                    means.append(results[t_out]['metrics'][metric]['mean'])
                    stds.append(results[t_out]['metrics'][metric]['std'])
            
            if t_outs:
                ax.errorbar(t_outs, means, yerr=stds, marker='o', capsize=5, capthick=2)
                ax.set_xlabel('T_out')
                ax.set_ylabel(name)
                ax.set_title(f'{name} vs T_out')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tout_vs_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_ar_vs_nar_comparison(self, results: Dict):
        """绘制AR vs NAR对比图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        t_outs = []
        ar_means = []
        nar_means = []
        ar_stds = []
        nar_stds = []
        
        for t_out in self.test_t_outs:
            if t_out in results:
                metrics = results[t_out]['metrics']
                if 'ar_rel_l2' in metrics and metrics['ar_rel_l2']['mean'] > 0:
                    t_outs.append(t_out)
                    ar_means.append(metrics['ar_rel_l2']['mean'])
                    ar_stds.append(metrics['ar_rel_l2']['std'])
                    nar_means.append(metrics['nar_rel_l2']['mean'])
                    nar_stds.append(metrics['nar_rel_l2']['std'])
        
        if t_outs:
            x = np.arange(len(t_outs))
            width = 0.35
            
            ax.bar(x - width/2, ar_means, width, yerr=ar_stds, label='AR', alpha=0.8)
            ax.bar(x + width/2, nar_means, width, yerr=nar_stds, label='NAR', alpha=0.8)
            
            ax.set_xlabel('T_out')
            ax.set_ylabel('Relative L2 Error')
            ax.set_title('AR vs NAR Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(t_outs)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ar_vs_nar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_prediction_samples(self, results: Dict):
        """绘制预测样本"""
        for t_out in [5, 10, 20]:  # 选择几个代表性的T_out值
            if t_out not in results:
                continue
                
            predictions = results[t_out]['predictions']
            targets = results[t_out]['targets']
            
            if not predictions:
                continue
                
            # 选择第一个样本
            pred = predictions[0]  # [T_out, C, H, W]
            target = targets[0]
            
            # 选择几个时间步进行可视化
            time_steps = [0, t_out//2, t_out-1] if t_out > 2 else [0, t_out-1]
            
            fig, axes = plt.subplots(3, len(time_steps), figsize=(5*len(time_steps), 12))
            if len(time_steps) == 1:
                axes = axes.reshape(-1, 1)
            
            for t_idx, t in enumerate(time_steps):
                # 选择第一个通道
                pred_t = pred[t, 0]  # [H, W]
                target_t = target[t, 0]
                error_t = np.abs(pred_t - target_t)
                
                # 预测
                im1 = axes[0, t_idx].imshow(pred_t, cmap='viridis')
                axes[0, t_idx].set_title(f'Prediction t={t}')
                axes[0, t_idx].axis('off')
                plt.colorbar(im1, ax=axes[0, t_idx])
                
                # 真实值
                im2 = axes[1, t_idx].imshow(target_t, cmap='viridis')
                axes[1, t_idx].set_title(f'Ground Truth t={t}')
                axes[1, t_idx].axis('off')
                plt.colorbar(im2, ax=axes[1, t_idx])
                
                # 误差
                im3 = axes[2, t_idx].imshow(error_t, cmap='Reds')
                axes[2, t_idx].set_title(f'Error t={t}')
                axes[2, t_idx].axis('off')
                plt.colorbar(im3, ax=axes[2, t_idx])
            
            plt.suptitle(f'Prediction Sample (T_out={t_out})', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'prediction_sample_tout_{t_out}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def _plot_error_evolution(self, results: Dict):
        """绘制误差演化图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for t_out in self.test_t_outs:
            if t_out not in results:
                continue
                
            predictions = results[t_out]['predictions']
            targets = results[t_out]['targets']
            
            if not predictions:
                continue
            
            # 计算每个时间步的平均误差
            time_errors = []
            for t in range(t_out):
                errors = []
                for pred, target in zip(predictions, targets):
                    if t < pred.shape[0]:
                        pred_t = torch.from_numpy(pred[t]).unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
                        target_t = torch.from_numpy(target[t]).unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
                        error = self.metrics_calculator.compute_rel_l2(pred_t, target_t)
                        errors.append(error.mean().item())
                if errors:
                    time_errors.append(np.mean(errors))
                else:
                    time_errors.append(0)
            
            if time_errors:
                ax.plot(range(t_out), time_errors, marker='o', label=f'T_out={t_out}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Relative L2 Error')
        ax.set_title('Error Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_test(self):
        """运行完整测试"""
        self.logger.info("开始多T_out测试...")
        
        for t_out in self.test_t_outs:
            self.logger.info(f"测试T_out={t_out}")
            
            try:
                # 加载模型
                model = self._load_model(t_out)
                
                # 加载测试数据
                test_loader = self._load_test_data(t_out)
                
                # 评估模型
                result = self._evaluate_model(model, test_loader, t_out)
                self.results[t_out] = result
                
                self.logger.info(f"T_out={t_out} 测试完成")
                self.logger.info(f"  Rel L2: {result['metrics']['rel_l2']['mean']:.6f} ± {result['metrics']['rel_l2']['std']:.6f}")
                self.logger.info(f"  MAE: {result['metrics']['mae']['mean']:.6f} ± {result['metrics']['mae']['std']:.6f}")
                
            except Exception as e:
                import traceback
                self.logger.error(f"T_out={t_out} 测试失败: {e}")
                self.logger.error(f"错误详情: {traceback.format_exc()}")
                continue
        
        # 创建可视化
        if self.results:
            self._create_visualizations(self.results)
        
        # 保存结果
        self._save_results()
        
        self.logger.info("多T_out测试完成!")
        
    def _save_results(self):
        """保存测试结果"""
        # 保存详细结果
        results_file = self.output_dir / 'multi_tout_results.json'
        
        # 转换numpy数组为列表以便JSON序列化
        json_results = {}
        for t_out, result in self.results.items():
            json_results[str(t_out)] = {
                'metrics': result['metrics']
                # 不保存predictions和targets，太大了
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 创建汇总表
        summary_file = self.output_dir / 'summary_table.txt'
        with open(summary_file, 'w') as f:
            f.write("T_out Performance Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"{'T_out':<8} {'Rel L2':<15} {'MAE':<15} {'PSNR':<10} {'SSIM':<10}\n")
            f.write("-" * 60 + "\n")
            
            for t_out in self.test_t_outs:
                if t_out in self.results:
                    metrics = self.results[t_out]['metrics']
                    f.write(f"{t_out:<8} ")
                    f.write(f"{metrics['rel_l2']['mean']:.6f}±{metrics['rel_l2']['std']:.4f} ")
                    f.write(f"{metrics['mae']['mean']:.6f}±{metrics['mae']['std']:.4f} ")
                    f.write(f"{metrics['psnr']['mean']:.2f}±{metrics['psnr']['std']:.2f} ")
                    f.write(f"{metrics['ssim']['mean']:.4f}±{metrics['ssim']['std']:.4f}\n")
        
        self.logger.info(f"结果已保存到: {self.output_dir}")

def main():
    """主函数"""
    # 配置文件和检查点路径
    config_path = "configs/experiment/temporal_nar_300epochs.yaml"
    checkpoint_path = "runs/temporal_nar_300epochs/TemporalNAR-DR2D-128-300epochs-s2025/checkpoints/best.pth"
    
    # 检查文件是否存在
    if not Path(config_path).exists():
        print(f"配置文件不存在: {config_path}")
        return
    
    if not Path(checkpoint_path).exists():
        print(f"检查点文件不存在: {checkpoint_path}")
        print("将使用随机初始化的模型进行测试")
    
    # 创建测试器并运行
    tester = TemporalNARMultiToutTester(config_path, checkpoint_path)
    tester.run_test()

if __name__ == "__main__":
    main()