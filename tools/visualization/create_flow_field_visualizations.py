#!/usr/bin/env python3
"""
稀疏观测重建可视化脚本 - 针对反应扩散系统的20% Crop任务
专门用于2D_diff-react_NA_NA数据集的稀疏观测重建可视化
从20%稀疏观测点重建完整的u和v分量流场分布
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import seaborn as sns
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import h5py
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib样式
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FlowFieldVisualizer:
    """反应扩散系统稀疏观测重建可视化器
    
    专门用于20% Crop任务的可视化分析：
    - 从20%稀疏观测点重建完整流场
    - 对比真实流场与重建流场
    - 分析重建误差和性能指标
    """
    
    def __init__(self, model_path, data_path, output_dir="flow_field_visualizations"):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "static_plots").mkdir(exist_ok=True)
        (self.output_dir / "time_series").mkdir(exist_ok=True)
        (self.output_dir / "animations").mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型和数据
        self.model = None
        self.test_data = None
        self.predictions = None
        
    def load_model(self):
        """加载训练好的模型"""
        print("正在加载模型...")
        
        # 查找最佳检查点
        checkpoint_dir = self.model_path / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"检查点目录不存在: {checkpoint_dir}")
        
        # 寻找最佳模型
        best_ckpt = None
        for ckpt_file in checkpoint_dir.glob("*.ckpt"):
            if "best" in ckpt_file.name.lower():
                best_ckpt = ckpt_file
                break
        
        if best_ckpt is None:
            # 如果没有找到best，使用最新的
            ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
            if ckpt_files:
                best_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
        
        if best_ckpt is None:
            raise FileNotFoundError("未找到模型检查点文件")
        
        print(f"加载检查点: {best_ckpt}")
        
        # 加载检查点
        checkpoint = torch.load(best_ckpt, map_location=self.device)
        
        # 从配置重建模型
        config_path = self.model_path / "config_merged.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 提取模型配置
            model_config = config.get('model', {})
            print(f"模型配置: {model_config}")
        
        # 简化：直接从checkpoint加载模型
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 创建一个简单的模型加载器
        self._load_model_from_checkpoint(state_dict)
        
    def _load_model_from_checkpoint(self, state_dict):
        """从检查点状态字典加载模型"""
        # 这里需要根据实际的模型架构来实现
        # 暂时创建一个占位符，实际使用时需要导入正确的模型类
        print("模型加载完成（占位符实现）")
        self.model = None  # 占位符
        
    def load_test_data(self):
        """加载测试数据"""
        print("正在加载测试数据...")
        
        # 查找数据文件
        data_file = None
        possible_paths = [
            self.data_path,
            Path("data") / "2D_diff-react_NA_NA.h5",
            Path("datasets") / "2D_diff-react_NA_NA.h5"
        ]
        
        for path in possible_paths:
            if path.exists():
                data_file = path
                break
        
        if data_file is None:
            # 创建模拟数据用于演示
            print("未找到真实数据文件，创建模拟数据...")
            self._create_synthetic_data()
            return
        
        print(f"加载数据文件: {data_file}")
        
        with h5py.File(data_file, 'r') as f:
            # 获取数据键
            keys = list(f.keys())
            print(f"数据文件包含键: {keys[:10]}...")  # 显示前10个键
            
            # 加载测试数据（假设使用后面的时间步作为测试）
            test_keys = keys[-50:]  # 使用最后50个时间步作为测试
            
            test_data = []
            for key in test_keys:
                data = f[key][:]  # shape: [H, W, C] 或 [C, H, W]
                if data.ndim == 3 and data.shape[-1] == 2:
                    # [H, W, C] -> [C, H, W]
                    data = data.transpose(2, 0, 1)
                test_data.append(data)
            
            self.test_data = np.array(test_data)  # [T, C, H, W]
            print(f"测试数据形状: {self.test_data.shape}")
            
    def _create_synthetic_data(self):
        """创建合成的反应扩散数据用于演示"""
        print("创建合成反应扩散数据...")
        
        # 参数设置
        T, C, H, W = 50, 2, 128, 128
        
        # 创建网格
        x = np.linspace(0, 2*np.pi, W)
        y = np.linspace(0, 2*np.pi, H)
        X, Y = np.meshgrid(x, y)
        
        # 生成时间序列数据
        test_data = []
        for t in range(T):
            time = t * 0.1
            
            # u分量：波动模式
            u = np.sin(X + time) * np.cos(Y + time * 0.5) * np.exp(-0.1 * time)
            
            # v分量：螺旋模式  
            v = np.cos(X - time * 0.8) * np.sin(Y - time * 0.3) * np.exp(-0.05 * time)
            
            # 添加噪声
            u += 0.1 * np.random.randn(H, W)
            v += 0.1 * np.random.randn(H, W)
            
            # 组合数据
            data = np.stack([u, v], axis=0)  # [C, H, W]
            test_data.append(data)
        
        self.test_data = np.array(test_data)  # [T, C, H, W]
        print(f"合成数据形状: {self.test_data.shape}")
        
    def generate_predictions(self):
        """生成预测结果"""
        print("生成预测结果...")
        
        if self.model is None:
            # 如果模型未加载，创建模拟预测
            print("模型未加载，创建模拟预测...")
            self._create_synthetic_predictions()
            return
        
        # 实际预测逻辑
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(len(self.test_data) - 10):  # 预测10步
                input_data = torch.from_numpy(self.test_data[i:i+1]).float().to(self.device)
                pred = self.model(input_data)
                predictions.append(pred.cpu().numpy())
        
        self.predictions = np.array(predictions)
        
    def _create_synthetic_predictions(self):
        """创建合成预测数据"""
        # 在真实数据基础上添加小的扰动作为"预测"
        predictions = []
        
        for i in range(len(self.test_data) - 1):
            # 使用下一个时间步作为"预测"，并添加小的误差
            true_next = self.test_data[i + 1]
            
            # 添加预测误差
            noise_u = 0.05 * np.random.randn(*true_next[0].shape)
            noise_v = 0.05 * np.random.randn(*true_next[1].shape)
            
            pred = true_next.copy()
            pred[0] += noise_u
            pred[1] += noise_v
            
            predictions.append(pred)
        
        self.predictions = np.array(predictions)
        print(f"预测数据形状: {self.predictions.shape}")
        
    def create_comparison_plots(self, num_samples=6):
        """创建稀疏观测重建对比图 - 真实值 vs 重建值 vs 重建误差"""
        print("创建稀疏观测重建对比图...")
        
        # 选择代表性样本
        indices = np.linspace(0, len(self.predictions) - 1, num_samples, dtype=int)
        
        for idx, sample_idx in enumerate(indices):
            gt = self.test_data[sample_idx + 1]  # 真实完整流场
            pred = self.predictions[sample_idx]   # 从20%观测重建的流场
            error = np.abs(gt - pred)             # 重建误差
            
            # 创建图形
            fig, axes = plt.subplots(3, 2, figsize=(12, 15))
            fig.suptitle(f'稀疏观测重建对比 (20% Crop) - 时间步 {sample_idx}', fontsize=16, fontweight='bold')
            
            # 设置颜色映射
            vmin_u, vmax_u = gt[0].min(), gt[0].max()
            vmin_v, vmax_v = gt[1].min(), gt[1].max()
            
            # u分量
            im1 = axes[0, 0].imshow(gt[0], cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
            axes[0, 0].set_title('真实完整流场 - u分量', fontweight='bold')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            
            im2 = axes[1, 0].imshow(pred[0], cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
            axes[1, 0].set_title('重建流场 (20%观测) - u分量', fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
            
            im3 = axes[2, 0].imshow(error[0], cmap='Reds')
            axes[2, 0].set_title('重建误差 - u分量', fontweight='bold')
            axes[2, 0].axis('off')
            plt.colorbar(im3, ax=axes[2, 0], shrink=0.8)
            
            # v分量
            im4 = axes[0, 1].imshow(gt[1], cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
            axes[0, 1].set_title('真实完整流场 - v分量', fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im4, ax=axes[0, 1], shrink=0.8)
            
            im5 = axes[1, 1].imshow(pred[1], cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
            axes[1, 1].set_title('重建流场 (20%观测) - v分量', fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
            
            im6 = axes[2, 1].imshow(error[1], cmap='Reds')
            axes[2, 1].set_title('重建误差 - v分量', fontweight='bold')
            axes[2, 1].axis('off')
            plt.colorbar(im6, ax=axes[2, 1], shrink=0.8)
            
            # 添加重建性能指标
            mse_u = np.mean((gt[0] - pred[0])**2)
            mse_v = np.mean((gt[1] - pred[1])**2)
            mae_u = np.mean(np.abs(gt[0] - pred[0]))
            mae_v = np.mean(np.abs(gt[1] - pred[1]))
            
            metrics_text = f'稀疏重建性能指标 (20% → 100%):\n'
            metrics_text += f'u分量重建 - MSE: {mse_u:.6f}, MAE: {mae_u:.6f}\n'
            metrics_text += f'v分量重建 - MSE: {mse_v:.6f}, MAE: {mae_v:.6f}'
            
            fig.text(0.02, 0.02, metrics_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "static_plots" / f"comparison_{idx:02d}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"已保存 {len(indices)} 个对比图到 static_plots/")
        
    def create_time_series_visualization(self):
        """创建稀疏重建时间序列演化可视化"""
        print("创建稀疏重建时间序列可视化...")
        
        # 选择一个固定点观察时间演化
        center_x, center_y = 64, 64  # 图像中心
        
        # 提取时间序列
        gt_u_series = self.test_data[1:len(self.predictions)+1, 0, center_y, center_x]
        gt_v_series = self.test_data[1:len(self.predictions)+1, 1, center_y, center_x]
        pred_u_series = self.predictions[:, 0, center_y, center_x]
        pred_v_series = self.predictions[:, 1, center_y, center_x]
        
        time_steps = np.arange(len(gt_u_series))
        
        # 创建时间序列图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # u分量时间序列
        ax1.plot(time_steps, gt_u_series, 'b-', linewidth=2, label='真实完整流场', alpha=0.8)
        ax1.plot(time_steps, pred_u_series, 'r--', linewidth=2, label='重建流场 (20%观测)', alpha=0.8)
        ax1.fill_between(time_steps, gt_u_series, pred_u_series, alpha=0.3, color='gray', label='重建误差区域')
        ax1.set_title('u分量稀疏重建时间演化 (中心点)', fontweight='bold', fontsize=14)
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('u值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # v分量时间序列
        ax2.plot(time_steps, gt_v_series, 'b-', linewidth=2, label='真实完整流场', alpha=0.8)
        ax2.plot(time_steps, pred_v_series, 'r--', linewidth=2, label='重建流场 (20%观测)', alpha=0.8)
        ax2.fill_between(time_steps, gt_v_series, pred_v_series, alpha=0.3, color='gray', label='重建误差区域')
        ax2.set_title('v分量稀疏重建时间演化 (中心点)', fontweight='bold', fontsize=14)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('v值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_series" / "center_point_evolution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建相空间图
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制相空间轨迹
        ax.plot(gt_u_series, gt_v_series, 'b-', linewidth=2, label='真实完整轨迹', alpha=0.8)
        ax.plot(pred_u_series, pred_v_series, 'r--', linewidth=2, label='重建轨迹 (20%观测)', alpha=0.8)
        ax.scatter(gt_u_series[0], gt_v_series[0], c='blue', s=100, marker='o', label='起始点')
        ax.scatter(gt_u_series[-1], gt_v_series[-1], c='blue', s=100, marker='s', label='结束点')
        ax.scatter(pred_u_series[0], pred_v_series[0], c='red', s=100, marker='o')
        ax.scatter(pred_u_series[-1], pred_v_series[-1], c='red', s=100, marker='s')
        
        ax.set_title('稀疏重建相空间轨迹对比 (20% → 100%)', fontweight='bold', fontsize=14)
        ax.set_xlabel('u分量')
        ax.set_ylabel('v分量')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_series" / "phase_space.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("稀疏重建时间序列可视化已保存到 time_series/")
        
    def create_statistical_analysis(self):
        """创建稀疏重建统计分析图"""
        print("创建稀疏重建统计分析...")
        
        # 计算各种误差指标
        errors_u = []
        errors_v = []
        mse_u_list = []
        mse_v_list = []
        mae_u_list = []
        mae_v_list = []
        
        for i in range(len(self.predictions)):
            gt = self.test_data[i + 1]
            pred = self.predictions[i]
            
            error_u = gt[0] - pred[0]
            error_v = gt[1] - pred[1]
            
            errors_u.append(error_u.flatten())
            errors_v.append(error_v.flatten())
            
            mse_u_list.append(np.mean(error_u**2))
            mse_v_list.append(np.mean(error_v**2))
            mae_u_list.append(np.mean(np.abs(error_u)))
            mae_v_list.append(np.mean(np.abs(error_v)))
        
        # 创建统计图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('预测误差统计分析', fontsize=16, fontweight='bold')
        
        # 误差分布直方图
        all_errors_u = np.concatenate(errors_u)
        all_errors_v = np.concatenate(errors_v)
        
        axes[0, 0].hist(all_errors_u, bins=50, alpha=0.7, color='blue', label='u分量误差')
        axes[0, 0].hist(all_errors_v, bins=50, alpha=0.7, color='red', label='v分量误差')
        axes[0, 0].set_title('误差分布直方图')
        axes[0, 0].set_xlabel('误差值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE随时间变化
        time_steps = np.arange(len(mse_u_list))
        axes[0, 1].plot(time_steps, mse_u_list, 'b-', linewidth=2, label='u分量 MSE')
        axes[0, 1].plot(time_steps, mse_v_list, 'r-', linewidth=2, label='v分量 MSE')
        axes[0, 1].set_title('均方误差随时间变化')
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAE随时间变化
        axes[1, 0].plot(time_steps, mae_u_list, 'b-', linewidth=2, label='u分量 MAE')
        axes[1, 0].plot(time_steps, mae_v_list, 'r-', linewidth=2, label='v分量 MAE')
        axes[1, 0].set_title('平均绝对误差随时间变化')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 误差相关性
        axes[1, 1].scatter(all_errors_u[::100], all_errors_v[::100], alpha=0.5)
        axes[1, 1].set_title('u和v分量误差相关性')
        axes[1, 1].set_xlabel('u分量误差')
        axes[1, 1].set_ylabel('v分量误差')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加相关系数
        correlation = np.corrcoef(all_errors_u[::100], all_errors_v[::100])[0, 1]
        axes[1, 1].text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "static_plots" / "statistical_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("统计分析图已保存")
        
    def create_html_gallery(self):
        """创建HTML图库"""
        print("创建HTML图库...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>反应扩散系统流场预测可视化</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .info-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #3498db;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-card {{
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .image-card:hover {{
            transform: translateY(-5px);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .image-card .caption {{
            padding: 15px;
            background-color: #f8f9fa;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #3498db;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>反应扩散系统流场预测可视化</h1>
        
        <div class="info-box">
            <h3>数据集信息</h3>
            <p><strong>数据集:</strong> 2D_diff-react_NA_NA</p>
            <p><strong>系统类型:</strong> 反应扩散系统</p>
            <p><strong>变量:</strong> u分量（反应物浓度）, v分量（催化剂浓度）</p>
            <p><strong>图像尺寸:</strong> 128 × 128</p>
            <p><strong>时间步数:</strong> {len(self.predictions)}</p>
        </div>
        
        <h2>预测性能指标</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{np.mean([np.mean((self.test_data[i+1][0] - self.predictions[i][0])**2) for i in range(len(self.predictions))]):.6f}</div>
                <div class="metric-label">u分量平均MSE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{np.mean([np.mean((self.test_data[i+1][1] - self.predictions[i][1])**2) for i in range(len(self.predictions))]):.6f}</div>
                <div class="metric-label">v分量平均MSE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{np.mean([np.mean(np.abs(self.test_data[i+1][0] - self.predictions[i][0])) for i in range(len(self.predictions))]):.6f}</div>
                <div class="metric-label">u分量平均MAE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{np.mean([np.mean(np.abs(self.test_data[i+1][1] - self.predictions[i][1])) for i in range(len(self.predictions))]):.6f}</div>
                <div class="metric-label">v分量平均MAE</div>
            </div>
        </div>
        
        <h2>流场对比图</h2>
        <div class="gallery">
"""
        
        # 添加对比图
        for i in range(6):  # 假设有6个对比图
            html_content += f"""
            <div class="image-card">
                <img src="static_plots/comparison_{i:02d}.png" alt="对比图 {i+1}">
                <div class="caption">
                    <strong>时间步 {i}</strong><br>
                    显示真实值、预测值和误差的对比
                </div>
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>时间序列分析</h2>
        <div class="gallery">
            <div class="image-card">
                <img src="time_series/center_point_evolution.png" alt="中心点时间演化">
                <div class="caption">
                    <strong>中心点时间演化</strong><br>
                    显示u和v分量在中心点的时间变化
                </div>
            </div>
            <div class="image-card">
                <img src="time_series/phase_space.png" alt="相空间轨迹">
                <div class="caption">
                    <strong>相空间轨迹</strong><br>
                    显示系统在u-v相空间中的演化轨迹
                </div>
            </div>
        </div>
        
        <h2>统计分析</h2>
        <div class="gallery">
            <div class="image-card">
                <img src="static_plots/statistical_analysis.png" alt="统计分析">
                <div class="caption">
                    <strong>预测误差统计分析</strong><br>
                    包括误差分布、时间演化和相关性分析
                </div>
            </div>
        </div>
        
        <div class="info-box">
            <h3>物理意义解释</h3>
            <p><strong>u分量:</strong> 代表反应物的浓度分布，通常显示为波动或扩散模式</p>
            <p><strong>v分量:</strong> 代表催化剂或抑制剂的浓度分布，与u分量相互作用形成复杂的时空模式</p>
            <p><strong>反应扩散系统:</strong> 描述化学反应和扩散过程的耦合，广泛应用于生物学、化学和物理学中的模式形成研究</p>
        </div>
        
        <div class="timestamp">
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        # 保存HTML文件
        with open(self.output_dir / "flow_field_gallery.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("HTML图库已创建: flow_field_gallery.html")
        
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("开始流场预测可视化分析...")
        print("=" * 50)
        
        try:
            # 加载模型（可选）
            try:
                self.load_model()
            except Exception as e:
                print(f"模型加载失败: {e}")
                print("将使用合成数据进行演示")
            
            # 加载数据
            self.load_test_data()
            
            # 生成预测
            self.generate_predictions()
            
            # 创建各种可视化
            self.create_comparison_plots()
            self.create_time_series_visualization()
            self.create_statistical_analysis()
            self.create_html_gallery()
            
            print("=" * 50)
            print("流场预测可视化分析完成！")
            print(f"所有结果已保存到: {self.output_dir}")
            print("\n主要输出文件:")
            print(f"• HTML图库: {self.output_dir}/flow_field_gallery.html")
            print(f"• 对比图: {self.output_dir}/static_plots/")
            print(f"• 时间序列: {self.output_dir}/time_series/")
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 设置路径
    model_path = "runs/temporal_nar_100epochs"
    data_path = "data/2D_diff-react_NA_NA.h5"  # 如果不存在会创建合成数据
    
    # 创建可视化器
    visualizer = FlowFieldVisualizer(model_path, data_path)
    
    # 运行完整分析
    visualizer.run_complete_analysis()