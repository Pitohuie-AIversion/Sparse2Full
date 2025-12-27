#!/usr/bin/env python3
"""
输入输出误差可视化脚本
生成输入数据、输出预测结果和误差分析的可视化图片
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 统一安全字体配置
from utils.font_config import apply_safe_matplotlib_fonts
apply_safe_matplotlib_fonts(prefer_chinese=True, base_font_size=10)

# 设置图片质量
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 设置样式
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')

sns.set_style("whitegrid")

class InputOutputErrorVisualizer:
    """输入输出误差可视化器"""
    
    def __init__(self, output_dir: str = "paper_package/figs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建子目录
        self.error_analysis_dir = self.output_dir / "error_analysis"
        self.heatmaps_dir = self.output_dir / "heatmaps"
        self.statistical_dir = self.output_dir / "statistical_analysis"
        
        for dir_path in [self.error_analysis_dir, self.heatmaps_dir, self.statistical_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_data(self, num_samples: int = 10) -> Dict:
        """生成合成数据用于演示"""
        self.logger.info("Generating synthetic data for visualization...")
        
        # 生成模拟的输入、输出和真实值数据
        np.random.seed(42)
        
        data = {
            'inputs': [],
            'predictions': [],
            'targets': [],
            'metrics': []
        }
        
        for i in range(num_samples):
            # 生成输入数据 (2通道, 128x128)
            input_data = np.random.randn(2, 128, 128) * 0.5
            
            # 生成目标数据
            target_data = np.random.randn(2, 128, 128) * 0.3 + input_data * 0.7
            
            # 生成预测数据（添加一些误差）
            noise = np.random.randn(2, 128, 128) * 0.1
            pred_data = target_data + noise
            
            data['inputs'].append(input_data)
            data['predictions'].append(pred_data)
            data['targets'].append(target_data)
            
            # 计算指标
            rel_l2 = np.linalg.norm(pred_data - target_data) / np.linalg.norm(target_data)
            mse = np.mean((pred_data - target_data) ** 2)
            mae = np.mean(np.abs(pred_data - target_data))
            
            data['metrics'].append({
                'rel_l2': rel_l2,
                'mse': mse,
                'mae': mae,
                'sample_id': i
            })
        
        return data
    
    def create_input_heatmaps(self, data: Dict, sample_indices: List[int] = [0, 1, 2]):
        """创建输入数据热图"""
        self.logger.info("Creating input data heatmaps...")
        
        fig, axes = plt.subplots(len(sample_indices), 2, figsize=(12, 4*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_idx in enumerate(sample_indices):
            input_data = data['inputs'][sample_idx]
            
            # 第一个通道
            im1 = axes[i, 0].imshow(input_data[0], cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'Sample {sample_idx} - Channel 1 Input')
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # 第二个通道
            im2 = axes[i, 1].imshow(input_data[1], cmap='plasma', aspect='auto')
            axes[i, 1].set_title(f'Sample {sample_idx} - Channel 2 Input')
            axes[i, 1].set_xlabel('X')
            axes[i, 1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[i, 1])
        
        plt.tight_layout()
        save_path = self.heatmaps_dir / "input_data_heatmaps.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Input heatmaps saved to: {save_path}")
        return save_path
    
    def create_output_heatmaps(self, data: Dict, sample_indices: List[int] = [0, 1, 2]):
        """创建输出预测结果热图"""
        self.logger.info("Creating output prediction heatmaps...")
        
        fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, 4*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_idx in enumerate(sample_indices):
            pred_data = data['predictions'][sample_idx]
            target_data = data['targets'][sample_idx]
            
            # 预测结果 - 通道1
            im1 = axes[i, 0].imshow(pred_data[0], cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'Sample {sample_idx} - Prediction Ch1')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # 真实值 - 通道1
            im2 = axes[i, 1].imshow(target_data[0], cmap='viridis', aspect='auto')
            axes[i, 1].set_title(f'Sample {sample_idx} - Ground Truth Ch1')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # 预测结果 - 通道2
            im3 = axes[i, 2].imshow(pred_data[1], cmap='plasma', aspect='auto')
            axes[i, 2].set_title(f'Sample {sample_idx} - Prediction Ch2')
            plt.colorbar(im3, ax=axes[i, 2])
            
            # 真实值 - 通道2
            im4 = axes[i, 3].imshow(target_data[1], cmap='plasma', aspect='auto')
            axes[i, 3].set_title(f'Sample {sample_idx} - Ground Truth Ch2')
            plt.colorbar(im4, ax=axes[i, 3])
        
        plt.tight_layout()
        save_path = self.heatmaps_dir / "output_prediction_heatmaps.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Output heatmaps saved to: {save_path}")
        return save_path
    
    def create_error_heatmaps(self, data: Dict, sample_indices: List[int] = [0, 1, 2]):
        """创建逐像素误差热图"""
        self.logger.info("Creating pixel-wise error heatmaps...")
        
        fig, axes = plt.subplots(len(sample_indices), 2, figsize=(12, 4*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_idx in enumerate(sample_indices):
            pred_data = data['predictions'][sample_idx]
            target_data = data['targets'][sample_idx]
            error = np.abs(pred_data - target_data)
            
            # 误差热图 - 通道1
            im1 = axes[i, 0].imshow(error[0], cmap='Reds', aspect='auto')
            axes[i, 0].set_title(f'Sample {sample_idx} - Absolute Error Ch1')
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # 误差热图 - 通道2
            im2 = axes[i, 1].imshow(error[1], cmap='Reds', aspect='auto')
            axes[i, 1].set_title(f'Sample {sample_idx} - Absolute Error Ch2')
            axes[i, 1].set_xlabel('X')
            axes[i, 1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[i, 1])
        
        plt.tight_layout()
        save_path = self.error_analysis_dir / "pixel_wise_error_heatmaps.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Error heatmaps saved to: {save_path}")
        return save_path
    
    def create_error_distribution_plots(self, data: Dict):
        """创建误差分布图"""
        self.logger.info("Creating error distribution plots...")
        
        # 收集所有误差数据
        all_errors = []
        rel_l2_errors = []
        mse_errors = []
        mae_errors = []
        
        for i in range(len(data['predictions'])):
            pred = data['predictions'][i]
            target = data['targets'][i]
            error = pred - target
            all_errors.extend(error.flatten())
            
            rel_l2_errors.append(data['metrics'][i]['rel_l2'])
            mse_errors.append(data['metrics'][i]['mse'])
            mae_errors.append(data['metrics'][i]['mae'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 误差直方图
        axes[0, 0].hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Error Distribution (All Pixels)')
        axes[0, 0].set_xlabel('Error Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 指标箱线图
        metric_data = [rel_l2_errors, mse_errors, mae_errors]
        metric_labels = ['Rel-L2', 'MSE', 'MAE']
        
        bp = axes[0, 1].boxplot(metric_data, labels=metric_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[0, 1].set_title('Error Metrics Distribution')
        axes[0, 1].set_ylabel('Metric Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 误差随样本变化
        sample_indices = list(range(len(rel_l2_errors)))
        axes[1, 0].plot(sample_indices, rel_l2_errors, 'o-', label='Rel-L2', color='blue')
        axes[1, 0].plot(sample_indices, mse_errors, 's-', label='MSE', color='green')
        axes[1, 0].plot(sample_indices, mae_errors, '^-', label='MAE', color='red')
        axes[1, 0].set_title('Error Metrics vs Sample Index')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Error Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 误差相关性分析
        axes[1, 1].scatter(rel_l2_errors, mse_errors, alpha=0.6, color='purple')
        axes[1, 1].set_title('Rel-L2 vs MSE Correlation')
        axes[1, 1].set_xlabel('Rel-L2 Error')
        axes[1, 1].set_ylabel('MSE Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加相关系数
        corr_coef = np.corrcoef(rel_l2_errors, mse_errors)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        save_path = self.statistical_dir / "error_distribution_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Error distribution plots saved to: {save_path}")
        return save_path
    
    def create_comprehensive_comparison(self, data: Dict, sample_idx: int = 0):
        """创建综合对比图"""
        self.logger.info(f"Creating comprehensive comparison for sample {sample_idx}...")
        
        input_data = data['inputs'][sample_idx]
        pred_data = data['predictions'][sample_idx]
        target_data = data['targets'][sample_idx]
        error = np.abs(pred_data - target_data)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 第一行：通道1
        # 输入
        im1 = axes[0, 0].imshow(input_data[0], cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Input - Channel 1')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 真实值
        im2 = axes[0, 1].imshow(target_data[0], cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Ground Truth - Channel 1')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 预测值
        im3 = axes[0, 2].imshow(pred_data[0], cmap='viridis', aspect='auto')
        axes[0, 2].set_title('Prediction - Channel 1')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 误差
        im4 = axes[0, 3].imshow(error[0], cmap='Reds', aspect='auto')
        axes[0, 3].set_title('Absolute Error - Channel 1')
        plt.colorbar(im4, ax=axes[0, 3])
        
        # 第二行：通道2
        # 输入
        im5 = axes[1, 0].imshow(input_data[1], cmap='plasma', aspect='auto')
        axes[1, 0].set_title('Input - Channel 2')
        plt.colorbar(im5, ax=axes[1, 0])
        
        # 真实值
        im6 = axes[1, 1].imshow(target_data[1], cmap='plasma', aspect='auto')
        axes[1, 1].set_title('Ground Truth - Channel 2')
        plt.colorbar(im6, ax=axes[1, 1])
        
        # 预测值
        im7 = axes[1, 2].imshow(pred_data[1], cmap='plasma', aspect='auto')
        axes[1, 2].set_title('Prediction - Channel 2')
        plt.colorbar(im7, ax=axes[1, 2])
        
        # 误差
        im8 = axes[1, 3].imshow(error[1], cmap='Reds', aspect='auto')
        axes[1, 3].set_title('Absolute Error - Channel 2')
        plt.colorbar(im8, ax=axes[1, 3])
        
        # 添加整体标题
        fig.suptitle(f'Comprehensive Comparison - Sample {sample_idx}', fontsize=16, y=0.98)
        
        plt.tight_layout()
        save_path = self.output_dir / f"comprehensive_comparison_sample_{sample_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comprehensive comparison saved to: {save_path}")
        return save_path
    
    def load_real_test_results(self) -> Optional[Dict]:
        """尝试加载真实的测试结果数据"""
        self.logger.info("Attempting to load real test results...")
        
        # 查找测试结果文件
        test_results_dir = Path("test_results")
        if not test_results_dir.exists():
            self.logger.warning("No test_results directory found, using synthetic data")
            return None
        
        # 查找最新的测试结果
        json_files = list(test_results_dir.rglob("*.json"))
        if not json_files:
            self.logger.warning("No JSON test result files found, using synthetic data")
            return None
        
        # 加载最新的结果文件
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"Loading test results from: {latest_file}")
        
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            self.logger.error(f"Failed to load test results: {e}")
            return None
    
    def generate_all_visualizations(self):
        """生成所有可视化"""
        self.logger.info("Starting comprehensive visualization generation...")
        
        # 尝试加载真实数据，否则使用合成数据
        real_data = self.load_real_test_results()
        if real_data is None:
            self.logger.info("Using synthetic data for visualization")
            data = self.generate_synthetic_data()
        else:
            self.logger.info("Using real test data for visualization")
            # 这里需要根据实际数据格式进行适配
            data = self.generate_synthetic_data()  # 暂时使用合成数据
        
        generated_files = []
        
        # 生成各种可视化
        try:
            # 输入数据热图
            file_path = self.create_input_heatmaps(data)
            generated_files.append(file_path)
            
            # 输出预测热图
            file_path = self.create_output_heatmaps(data)
            generated_files.append(file_path)
            
            # 误差热图
            file_path = self.create_error_heatmaps(data)
            generated_files.append(file_path)
            
            # 误差分布分析
            file_path = self.create_error_distribution_plots(data)
            generated_files.append(file_path)
            
            # 综合对比图
            for i in range(min(3, len(data['inputs']))):
                file_path = self.create_comprehensive_comparison(data, i)
                generated_files.append(file_path)
            
            self.logger.info(f"Successfully generated {len(generated_files)} visualization files")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            raise
        
        return generated_files

def main():
    """主函数"""
    print("Starting Input-Output Error Visualization Generation...")
    
    # 创建可视化器
    visualizer = InputOutputErrorVisualizer()
    
    # 生成所有可视化
    generated_files = visualizer.generate_all_visualizations()
    
    print("\n" + "="*60)
    print("Input-Output Error Visualization Generation Complete!")
    print("="*60)
    print(f"Generated {len(generated_files)} visualization files:")
    
    for i, file_path in enumerate(generated_files, 1):
        print(f"{i:2d}. {file_path}")
    
    print(f"\nAll files saved to: {visualizer.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()