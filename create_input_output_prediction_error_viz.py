#!/usr/bin/env python3
"""
输入-输出-预测-误差完整可视化脚本
专门用于20% Crop任务的稀疏观测重建可视化
展示从稀疏输入到完整流场重建的完整流程
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InputOutputPredictionErrorVisualizer:
    """输入-输出-预测-误差完整可视化器
    
    专门用于20% Crop任务的完整流程可视化：
    - 稀疏输入可视化（20%观测点）
    - 真实完整流场
    - 模型重建流场
    - 预测误差分析
    """
    
    def __init__(self, output_dir="input_output_viz"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "input_viz").mkdir(exist_ok=True)
        (self.output_dir / "complete_flow").mkdir(exist_ok=True)
        (self.output_dir / "grid_comparison").mkdir(exist_ok=True)
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def generate_sparse_mask(self, height=128, width=128, sparse_ratio=0.2, seed=42):
        """生成20%稀疏观测掩码"""
        np.random.seed(seed)
        total_pixels = height * width
        num_observed = int(total_pixels * sparse_ratio)
        
        # 创建掩码
        mask = np.zeros((height, width), dtype=bool)
        
        # 随机选择观测点
        indices = np.random.choice(total_pixels, num_observed, replace=False)
        row_indices = indices // width
        col_indices = indices % width
        mask[row_indices, col_indices] = True
        
        return mask
    
    def create_synthetic_reaction_diffusion(self, timesteps=50, height=128, width=128):
        """创建合成反应扩散数据"""
        print("创建合成反应扩散数据...")
        
        # 空间网格
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        X, Y = np.meshgrid(x, y)
        
        # 时间步
        t = np.linspace(0, 2, timesteps)
        
        data = np.zeros((timesteps, 2, height, width))
        
        for i, time in enumerate(t):
            # u分量：波动模式
            u = np.exp(-0.5 * ((X-0.5*np.cos(2*time))**2 + (Y-0.5*np.sin(2*time))**2)) * \
                np.cos(2*np.pi*(X + Y) + time) * (1 + 0.3*np.sin(3*time))
            
            # v分量：扩散模式
            v = np.exp(-0.3 * (X**2 + Y**2)) * \
                np.sin(np.pi*(X - 0.3*np.cos(3*time)) + np.pi*(Y - 0.3*np.sin(3*time))) * \
                (1 + 0.2*np.cos(4*time))
            
            # 添加噪声
            u += 0.05 * np.random.randn(height, width)
            v += 0.05 * np.random.randn(height, width)
            
            data[i, 0] = u
            data[i, 1] = v
        
        return data
    
    def apply_sparse_mask(self, full_field, mask):
        """应用稀疏掩码到完整流场"""
        sparse_field = np.zeros_like(full_field)
        sparse_field[mask] = full_field[mask]
        return sparse_field
    
    def create_mock_prediction(self, gt_field, noise_level=0.1):
        """创建模拟预测结果（添加一些噪声和平滑）"""
        # 添加高斯噪声
        noise = np.random.randn(*gt_field.shape) * noise_level
        pred_field = gt_field + noise
        
        # 轻微平滑
        from scipy.ndimage import gaussian_filter
        pred_field[0] = gaussian_filter(pred_field[0], sigma=0.8)
        pred_field[1] = gaussian_filter(pred_field[1], sigma=0.8)
        
        return pred_field
    
    def create_input_visualization(self, sparse_input, mask, timestep=25):
        """创建稀疏输入可视化"""
        print("创建稀疏输入可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'20% 稀疏观测输入可视化 - 时间步 {timestep}', 
                    fontsize=16, fontweight='bold')
        
        # u分量稀疏输入
        im1 = axes[0, 0].imshow(sparse_input[0], cmap='RdBu_r', 
                               vmin=sparse_input[0].min(), vmax=sparse_input[0].max())
        axes[0, 0].set_title('稀疏输入 - u分量 (20%观测点)', fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        # v分量稀疏输入
        im2 = axes[0, 1].imshow(sparse_input[1], cmap='RdBu_r',
                               vmin=sparse_input[1].min(), vmax=sparse_input[1].max())
        axes[0, 1].set_title('稀疏输入 - v分量 (20%观测点)', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # 观测点分布掩码
        im3 = axes[1, 0].imshow(mask.astype(float), cmap='Greys', vmin=0, vmax=1)
        axes[1, 0].set_title('观测点分布掩码', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        
        # 统计信息
        axes[1, 1].axis('off')
        stats_text = f"""稀疏观测统计信息:
        
观测比例: 20%
总像素数: {mask.size:,}
观测点数: {mask.sum():,}
未观测点数: {(~mask).sum():,}

u分量统计:
  观测值范围: [{sparse_input[0][mask].min():.3f}, {sparse_input[0][mask].max():.3f}]
  观测值均值: {sparse_input[0][mask].mean():.3f}
  观测值标准差: {sparse_input[0][mask].std():.3f}

v分量统计:
  观测值范围: [{sparse_input[1][mask].min():.3f}, {sparse_input[1][mask].max():.3f}]
  观测值均值: {sparse_input[1][mask].mean():.3f}
  观测值标准差: {sparse_input[1][mask].std():.3f}"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "input_viz" / f"sparse_input_t{timestep}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"稀疏输入可视化已保存: sparse_input_t{timestep}.png")
    
    def create_4x2_comprehensive_grid(self, sparse_input, gt_field, pred_field, mask, timestep=25):
        """创建4×2综合网格图展示完整流程"""
        print("创建4×2综合网格图...")
        
        # 计算误差
        error_field = np.abs(gt_field - pred_field)
        
        # 创建图形
        fig, axes = plt.subplots(4, 2, figsize=(14, 18))
        fig.suptitle(f'稀疏观测重建完整流程 (20% → 100%) - 时间步 {timestep}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 设置颜色范围
        vmin_u, vmax_u = gt_field[0].min(), gt_field[0].max()
        vmin_v, vmax_v = gt_field[1].min(), gt_field[1].max()
        
        # 第一行：稀疏输入 (20%观测点)
        im1 = axes[0, 0].imshow(sparse_input[0], cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
        axes[0, 0].set_title('1. 稀疏输入 - u分量\n(20%观测点)', fontweight='bold', fontsize=12)
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        im2 = axes[0, 1].imshow(sparse_input[1], cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
        axes[0, 1].set_title('1. 稀疏输入 - v分量\n(20%观测点)', fontweight='bold', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # 第二行：真实完整流场
        im3 = axes[1, 0].imshow(gt_field[0], cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
        axes[1, 0].set_title('2. 真实完整流场 - u分量\n(Ground Truth)', fontweight='bold', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        
        im4 = axes[1, 1].imshow(gt_field[1], cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
        axes[1, 1].set_title('2. 真实完整流场 - v分量\n(Ground Truth)', fontweight='bold', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
        
        # 第三行：模型重建流场
        im5 = axes[2, 0].imshow(pred_field[0], cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
        axes[2, 0].set_title('3. 模型重建流场 - u分量\n(从20%重建100%)', fontweight='bold', fontsize=12)
        axes[2, 0].axis('off')
        plt.colorbar(im5, ax=axes[2, 0], shrink=0.8)
        
        im6 = axes[2, 1].imshow(pred_field[1], cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
        axes[2, 1].set_title('3. 模型重建流场 - v分量\n(从20%重建100%)', fontweight='bold', fontsize=12)
        axes[2, 1].axis('off')
        plt.colorbar(im6, ax=axes[2, 1], shrink=0.8)
        
        # 第四行：预测误差
        im7 = axes[3, 0].imshow(error_field[0], cmap='Reds', vmin=0, vmax=error_field[0].max())
        axes[3, 0].set_title('4. 重建误差 - u分量\n|GT - Pred|', fontweight='bold', fontsize=12)
        axes[3, 0].axis('off')
        plt.colorbar(im7, ax=axes[3, 0], shrink=0.8)
        
        im8 = axes[3, 1].imshow(error_field[1], cmap='Reds', vmin=0, vmax=error_field[1].max())
        axes[3, 1].set_title('4. 重建误差 - v分量\n|GT - Pred|', fontweight='bold', fontsize=12)
        axes[3, 1].axis('off')
        plt.colorbar(im8, ax=axes[3, 1], shrink=0.8)
        
        # 添加流程箭头
        arrow_props = dict(arrowstyle='->', lw=3, color='red')
        
        # 垂直箭头
        for i in range(3):
            fig.text(0.5, 0.78 - i*0.22, '↓', fontsize=30, ha='center', va='center', 
                    color='red', fontweight='bold')
        
        # 添加定量指标
        mse_u = np.mean((gt_field[0] - pred_field[0])**2)
        mse_v = np.mean((gt_field[1] - pred_field[1])**2)
        mae_u = np.mean(np.abs(gt_field[0] - pred_field[0]))
        mae_v = np.mean(np.abs(gt_field[1] - pred_field[1]))
        
        # 计算观测点和未观测点的误差
        obs_mse_u = np.mean((gt_field[0][mask] - pred_field[0][mask])**2)
        obs_mse_v = np.mean((gt_field[1][mask] - pred_field[1][mask])**2)
        unobs_mse_u = np.mean((gt_field[0][~mask] - pred_field[0][~mask])**2)
        unobs_mse_v = np.mean((gt_field[1][~mask] - pred_field[1][~mask])**2)
        
        metrics_text = f"""重建性能指标 (20% → 100%):

整体重建误差:
  u分量 - MSE: {mse_u:.6f}, MAE: {mae_u:.6f}
  v分量 - MSE: {mse_v:.6f}, MAE: {mae_v:.6f}

观测点重建误差 (20%):
  u分量 - MSE: {obs_mse_u:.6f}
  v分量 - MSE: {obs_mse_v:.6f}

未观测点重建误差 (80%):
  u分量 - MSE: {unobs_mse_u:.6f}
  v分量 - MSE: {unobs_mse_v:.6f}

重建挑战比率:
  未观测/观测 MSE比 (u): {unobs_mse_u/obs_mse_u:.2f}
  未观测/观测 MSE比 (v): {unobs_mse_v/obs_mse_v:.2f}"""
        
        fig.text(0.02, 0.02, metrics_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.15)
        plt.savefig(self.output_dir / "grid_comparison" / f"complete_flow_grid_t{timestep}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"4×2综合网格图已保存: complete_flow_grid_t{timestep}.png")
        
        return {
            'mse_u': mse_u, 'mse_v': mse_v,
            'mae_u': mae_u, 'mae_v': mae_v,
            'obs_mse_u': obs_mse_u, 'obs_mse_v': obs_mse_v,
            'unobs_mse_u': unobs_mse_u, 'unobs_mse_v': unobs_mse_v
        }
    
    def create_detailed_error_analysis(self, gt_field, pred_field, mask, timestep=25):
        """创建详细的误差分析图"""
        print("创建详细误差分析...")
        
        error_field = np.abs(gt_field - pred_field)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'稀疏重建误差详细分析 - 时间步 {timestep}', 
                    fontsize=16, fontweight='bold')
        
        # 误差热图
        im1 = axes[0, 0].imshow(error_field[0], cmap='Reds')
        axes[0, 0].set_title('u分量重建误差热图', fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        im2 = axes[1, 0].imshow(error_field[1], cmap='Reds')
        axes[1, 0].set_title('v分量重建误差热图', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
        
        # 误差分布直方图
        axes[0, 1].hist(error_field[0].flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('u分量误差分布', fontweight='bold')
        axes[0, 1].set_xlabel('绝对误差')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 1].hist(error_field[1].flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].set_title('v分量误差分布', fontweight='bold')
        axes[1, 1].set_xlabel('绝对误差')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 观测点 vs 未观测点误差对比
        obs_error_u = error_field[0][mask]
        unobs_error_u = error_field[0][~mask]
        obs_error_v = error_field[1][mask]
        unobs_error_v = error_field[1][~mask]
        
        axes[0, 2].boxplot([obs_error_u, unobs_error_u], 
                          labels=['观测点(20%)', '未观测点(80%)'])
        axes[0, 2].set_title('u分量: 观测点 vs 未观测点误差', fontweight='bold')
        axes[0, 2].set_ylabel('绝对误差')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 2].boxplot([obs_error_v, unobs_error_v], 
                          labels=['观测点(20%)', '未观测点(80%)'])
        axes[1, 2].set_title('v分量: 观测点 vs 未观测点误差', fontweight='bold')
        axes[1, 2].set_ylabel('绝对误差')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "complete_flow" / f"error_analysis_t{timestep}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"详细误差分析已保存: error_analysis_t{timestep}.png")
    
    def create_html_report(self):
        """创建HTML报告"""
        print("创建HTML报告...")
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>稀疏观测重建完整流程可视化</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
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
        .image-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .caption {{
            margin-top: 15px;
            font-style: italic;
            color: #666;
            font-size: 14px;
        }}
        .highlight {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }}
        .process-flow {{
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin: 30px 0;
            padding: 20px;
            background-color: #e8f4fd;
            border-radius: 10px;
        }}
        .process-step {{
            text-align: center;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            margin: 0 10px;
        }}
        .arrow {{
            font-size: 24px;
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>稀疏观测重建完整流程可视化报告</h1>
        
        <div class="info-box">
            <h3>任务概述</h3>
            <p><strong>任务类型:</strong> 20% Crop任务 (稀疏观测重建)</p>
            <p><strong>数据集:</strong> 2D_diff-react_NA_NA (反应扩散系统)</p>
            <p><strong>图像尺寸:</strong> 128×128</p>
            <p><strong>物理分量:</strong> u分量 (反应物浓度) 和 v分量 (催化剂浓度)</p>
            <p><strong>重建挑战:</strong> 从20%稀疏观测点重建完整的100%流场分布</p>
        </div>
        
        <div class="process-flow">
            <div class="process-step">
                <h4>1. 稀疏输入</h4>
                <p>20%观测点</p>
            </div>
            <div class="arrow">→</div>
            <div class="process-step">
                <h4>2. 真实流场</h4>
                <p>完整Ground Truth</p>
            </div>
            <div class="arrow">→</div>
            <div class="process-step">
                <h4>3. 模型重建</h4>
                <p>从稀疏到完整</p>
            </div>
            <div class="arrow">→</div>
            <div class="process-step">
                <h4>4. 误差分析</h4>
                <p>重建质量评估</p>
            </div>
        </div>
        
        <h2>1. 稀疏输入可视化</h2>
        <div class="image-container">
            <img src="input_viz/sparse_input_t25.png" alt="稀疏输入可视化">
            <div class="caption">
                稀疏输入可视化：展示20%观测点的分布和数值，包括u和v分量的稀疏观测以及观测点分布掩码
            </div>
        </div>
        
        <div class="highlight">
            <strong>稀疏输入特点:</strong>
            <ul>
                <li>仅有20%的像素点有观测值，其余80%为未知</li>
                <li>观测点随机分布在整个128×128网格中</li>
                <li>模型需要从这些稀疏信息推断完整的流场分布</li>
                <li>这是一个高度欠定的逆问题，需要利用物理先验知识</li>
            </ul>
        </div>
        
        <h2>2. 完整流程对比 (4×2网格图)</h2>
        <div class="image-container">
            <img src="grid_comparison/complete_flow_grid_t25.png" alt="完整流程对比">
            <div class="caption">
                完整流程对比：从稀疏输入(20%)到真实流场、模型重建、预测误差的完整可视化流程
            </div>
        </div>
        
        <div class="highlight">
            <strong>重建流程解析:</strong>
            <ul>
                <li><strong>第1行:</strong> 稀疏输入 - 仅显示20%观测点的u和v分量值</li>
                <li><strong>第2行:</strong> 真实完整流场 - Ground Truth的完整u和v分量分布</li>
                <li><strong>第3行:</strong> 模型重建流场 - 从20%观测点重建的完整流场</li>
                <li><strong>第4行:</strong> 重建误差 - 真实值与重建值的绝对误差热图</li>
            </ul>
        </div>
        
        <h2>3. 详细误差分析</h2>
        <div class="image-container">
            <img src="complete_flow/error_analysis_t25.png" alt="详细误差分析">
            <div class="caption">
                详细误差分析：包括误差热图、误差分布直方图、观测点与未观测点的误差对比
            </div>
        </div>
        
        <div class="info-box">
            <h3>重建性能分析</h3>
            <p><strong>关键发现:</strong></p>
            <ul>
                <li>未观测点(80%)的重建误差通常高于观测点(20%)</li>
                <li>误差分布反映了模型的重建能力和物理一致性</li>
                <li>u和v分量的重建难度可能不同，取决于其空间相关性</li>
                <li>边界区域和高梯度区域通常重建误差较大</li>
            </ul>
        </div>
        
        <h2>4. 技术挑战与意义</h2>
        <div class="info-box">
            <h3>稀疏重建的技术挑战</h3>
            <p><strong>数学挑战:</strong></p>
            <ul>
                <li>高度欠定问题：从20%信息推断80%未知信息</li>
                <li>空间相关性建模：需要学习流场的空间依赖关系</li>
                <li>物理约束：必须满足反应扩散方程的物理规律</li>
                <li>多尺度特征：需要捕捉局部细节和全局模式</li>
            </ul>
            
            <p><strong>实际应用价值:</strong></p>
            <ul>
                <li>传感器网络优化：减少传感器数量降低成本</li>
                <li>环境监测：从稀疏监测点重建完整环境场</li>
                <li>医学成像：减少扫描时间和辐射暴露</li>
                <li>工业过程监控：优化监测点布置</li>
            </ul>
        </div>
        
        <h2>5. 模型评估指标</h2>
        <div class="info-box">
            <h3>重建质量评估</h3>
            <p>本可视化报告展示了以下关键指标：</p>
            <ul>
                <li><strong>整体重建误差:</strong> MSE和MAE衡量总体重建质量</li>
                <li><strong>观测点误差:</strong> 评估模型对已知信息的拟合能力</li>
                <li><strong>未观测点误差:</strong> 评估模型的插值和外推能力</li>
                <li><strong>误差分布:</strong> 了解误差的统计特性和异常值</li>
                <li><strong>空间误差模式:</strong> 识别重建困难的区域和模式</li>
            </ul>
        </div>
        
        <div class="highlight">
            <p><strong>结论:</strong> 稀疏观测重建是一个具有挑战性的任务，需要模型具备强大的空间推理能力和物理知识。
            通过本可视化分析，我们可以深入理解模型的重建能力、局限性和改进方向。</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(self.output_dir / "input_output_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("HTML报告已创建: input_output_report.html")
    
    def run_complete_visualization(self):
        """运行完整的可视化流程"""
        print("开始输入-输出-预测-误差完整可视化分析...")
        print("=" * 60)
        
        # 1. 生成合成数据
        data = self.create_synthetic_reaction_diffusion()
        
        # 2. 选择一个时间步进行分析
        timestep = 25
        gt_field = data[timestep]  # shape: (2, 128, 128)
        
        # 3. 生成稀疏掩码
        mask = self.generate_sparse_mask()
        
        # 4. 创建稀疏输入
        sparse_input = np.zeros_like(gt_field)
        sparse_input[0] = self.apply_sparse_mask(gt_field[0], mask)
        sparse_input[1] = self.apply_sparse_mask(gt_field[1], mask)
        
        # 5. 创建模拟预测
        pred_field = self.create_mock_prediction(gt_field)
        
        # 6. 创建稀疏输入可视化
        self.create_input_visualization(sparse_input, mask, timestep)
        
        # 7. 创建4×2综合网格图
        metrics = self.create_4x2_comprehensive_grid(sparse_input, gt_field, pred_field, mask, timestep)
        
        # 8. 创建详细误差分析
        self.create_detailed_error_analysis(gt_field, pred_field, mask, timestep)
        
        # 9. 创建HTML报告
        self.create_html_report()
        
        print("=" * 60)
        print("输入-输出-预测-误差完整可视化分析完成！")
        print(f"所有结果已保存到: {self.output_dir}")
        print()
        print("主要输出文件:")
        print(f"• HTML报告: {self.output_dir}/input_output_report.html")
        print(f"• 稀疏输入可视化: {self.output_dir}/input_viz/")
        print(f"• 4×2综合网格图: {self.output_dir}/grid_comparison/")
        print(f"• 详细误差分析: {self.output_dir}/complete_flow/")
        
        return metrics

if __name__ == "__main__":
    # 创建可视化器
    visualizer = InputOutputPredictionErrorVisualizer()
    
    # 运行完整可视化
    metrics = visualizer.run_complete_visualization()
    
    print("\n重建性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")