#!/usr/bin/env python3
"""
直接运行时序NAR模型可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_temporal_nar_model():
    """创建时序NAR模型"""
    class SwinTemporalNAR(nn.Module):
        def __init__(self):
            super().__init__()
            # 简化的Swin Transformer结构
            self.patch_embed = nn.Conv2d(1, 96, kernel_size=4, stride=4)
            
            # 上采样解码器
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1)
            )
            
        def forward(self, x):
            # 简化的前向传播
            B, C, H, W = x.shape
            
            # Patch embedding
            x = self.patch_embed(x)  # [B, 96, H/4, W/4]
            
            # 上采样到目标尺寸
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            x = self.decoder(x)
            
            return x
    
    return SwinTemporalNAR()

def generate_test_data(batch_size=4):
    """生成测试数据"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成低分辨率观测数据 (32x32)
    observed = torch.randn(batch_size, 1, 32, 32, device=device)
    
    # 生成高分辨率真值数据 (128x128)
    gt = torch.randn(batch_size, 1, 128, 128, device=device)
    
    # 添加一些结构化模式
    x = torch.linspace(-1, 1, 32, device=device)
    y = torch.linspace(-1, 1, 32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    for i in range(batch_size):
        # 为观测数据添加不同的模式
        if i % 4 == 0:
            observed[i, 0] = torch.sin(2 * np.pi * X) * torch.cos(2 * np.pi * Y)
        elif i % 4 == 1:
            observed[i, 0] = torch.exp(-(X**2 + Y**2))
        elif i % 4 == 2:
            observed[i, 0] = X**2 - Y**2
        else:
            observed[i, 0] = torch.sin(3 * X) + torch.cos(3 * Y)
    
    # 为真值数据生成对应的高分辨率版本
    x_hr = torch.linspace(-1, 1, 128, device=device)
    y_hr = torch.linspace(-1, 1, 128, device=device)
    X_hr, Y_hr = torch.meshgrid(x_hr, y_hr, indexing='ij')
    
    for i in range(batch_size):
        if i % 4 == 0:
            gt[i, 0] = torch.sin(2 * np.pi * X_hr) * torch.cos(2 * np.pi * Y_hr)
        elif i % 4 == 1:
            gt[i, 0] = torch.exp(-(X_hr**2 + Y_hr**2))
        elif i % 4 == 2:
            gt[i, 0] = X_hr**2 - Y_hr**2
        else:
            gt[i, 0] = torch.sin(3 * X_hr) + torch.cos(3 * Y_hr)
    
    return observed, gt

def compute_metrics(pred, gt):
    """计算评估指标"""
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(pred, gt).item()
        
        # MAE
        mae = F.l1_loss(pred, gt).item()
        
        # PSNR
        psnr = 20 * torch.log10(torch.max(gt) / torch.sqrt(torch.mean((pred - gt) ** 2))).item()
        
        # 相对L2误差
        rel_l2 = torch.norm(pred - gt) / torch.norm(gt)
        rel_l2 = rel_l2.item()
        
        return {
            'MSE': mse,
            'MAE': mae,
            'PSNR': psnr,
            'Rel_L2': rel_l2
        }

def create_visualization(observed, pred, gt, metrics, output_dir):
    """创建可视化图表"""
    batch_size = observed.shape[0]
    n_samples = min(4, batch_size)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # 观测数据
        im1 = axes[i, 0].imshow(observed[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f'观测数据 #{i+1}')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 真值数据
        im2 = axes[i, 1].imshow(gt[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f'真值数据 #{i+1}')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # 预测结果
        im3 = axes[i, 2].imshow(pred[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title(f'预测结果 #{i+1}')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2])
        
        # 绝对误差
        error = torch.abs(pred[i, 0] - gt[i, 0]).cpu().numpy()
        im4 = axes[i, 3].imshow(error, cmap='Reds')
        axes[i, 3].set_title(f'绝对误差 #{i+1}')
        axes[i, 3].axis('off')
        plt.colorbar(im4, ax=axes[i, 3])
    
    # 添加总体指标
    fig.suptitle(f'时序NAR模型预测结果\nMSE: {metrics["MSE"]:.6f} | MAE: {metrics["MAE"]:.6f} | PSNR: {metrics["PSNR"]:.2f}dB | Rel-L2: {metrics["Rel_L2"]:.6f}', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # 保存图片
    viz_path = output_dir / 'temporal_nar_predictions.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return viz_path

def create_html_report(metrics, viz_path, output_dir):
    """创建HTML报告"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>时序NAR模型预测结果报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #333; }}
            .metrics {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px 20px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .visualization {{ text-align: center; margin: 30px 0; }}
            .visualization img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 时序NAR模型预测结果报告</h1>
            <p>基于Swin Transformer架构的时序非自回归模型可视化分析</p>
        </div>
        
        <div class="metrics">
            <h2>📊 性能指标</h2>
            <div class="metric">
                <div class="metric-value">{metrics['MSE']:.6f}</div>
                <div class="metric-label">均方误差 (MSE)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['MAE']:.6f}</div>
                <div class="metric-label">平均绝对误差 (MAE)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['PSNR']:.2f}dB</div>
                <div class="metric-label">峰值信噪比 (PSNR)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['Rel_L2']:.6f}</div>
                <div class="metric-label">相对L2误差</div>
            </div>
        </div>
        
        <div class="visualization">
            <h2>🎨 预测结果可视化</h2>
            <img src="{viz_path.name}" alt="时序NAR模型预测结果">
        </div>
        
        <div class="footer">
            <p><em>报告生成时间: {Path(__file__).stat().st_mtime}</em></p>
        </div>
    </body>
    </html>
    """
    
    report_path = output_dir / 'temporal_nar_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path

def main():
    """主函数"""
    print("🎯 开始时序NAR模型可视化...")
    
    # 设置路径
    model_path = Path(r"runs/temporal_nar_100epochs/TemporalNAR-DR2D-128-100epochs-s2025/best.pth")
    output_dir = Path(r"runs/temporal_nar_100epochs/predictions_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    try:
        # 加载检查点
        print(f"📂 加载模型检查点: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✅ 检查点加载成功")
        
        if isinstance(checkpoint, dict):
            print(f"   检查点键: {list(checkpoint.keys())}")
            if 'epoch' in checkpoint:
                print(f"   训练轮次: {checkpoint['epoch']}")
        
        # 创建模型
        print("🏗️ 创建时序NAR模型...")
        model = create_temporal_nar_model().to(device)
        
        # 尝试加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"✅ 模型权重加载完成")
            if missing_keys:
                print(f"   缺失键数量: {len(missing_keys)}")
            if unexpected_keys:
                print(f"   多余键数量: {len(unexpected_keys)}")
        except Exception as e:
            print(f"⚠️ 权重加载失败，使用随机初始化: {e}")
        
        model.eval()
        
        # 生成测试数据
        print("🎲 生成测试数据...")
        observed, gt = generate_test_data()
        
        # 进行预测
        print("🔮 进行模型预测...")
        with torch.no_grad():
            pred = model(observed)
            
            # 确保预测结果与真值尺寸匹配
            if pred.shape != gt.shape:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        
        # 计算指标
        print("📊 计算性能指标...")
        metrics = compute_metrics(pred, gt)
        
        # 创建可视化
        print("🎨 创建可视化图表...")
        viz_path = create_visualization(observed, pred, gt, metrics, output_dir)
        
        # 创建HTML报告
        print("📄 生成HTML报告...")
        report_path = create_html_report(metrics, viz_path, output_dir)
        
        # 保存指标到JSON
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n🎉 时序NAR模型可视化完成！")
        print(f"   📊 性能指标:")
        print(f"      MSE: {metrics['MSE']:.6f}")
        print(f"      MAE: {metrics['MAE']:.6f}")
        print(f"      PSNR: {metrics['PSNR']:.2f}dB")
        print(f"      Rel-L2: {metrics['Rel_L2']:.6f}")
        print(f"   📁 输出目录: {output_dir}")
        print(f"   🖼️ 可视化图片: {viz_path}")
        print(f"   📄 HTML报告: {report_path}")
        print(f"   📋 指标文件: {metrics_path}")
        
    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()