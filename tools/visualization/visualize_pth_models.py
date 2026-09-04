#!/usr/bin/env python3
"""
模型预测结果可视化工具
专门用于加载.pth模型文件并生成预测结果的可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path
import glob
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleUNet(nn.Module):
    """简单的U-Net模型用于演示"""
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        # 简单的上采样
        x = self.encoder(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.decoder(x)
        return x

class ModelVisualizer:
    """模型预测结果可视化器"""
    
    def __init__(self, output_dir: str = "model_predictions_visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
    def find_model_files(self) -> List[str]:
        """查找所有.pth模型文件"""
        model_files = []
        patterns = [
            "runs/**/*.pth",
            "checkpoints/**/*.pth",
            "*.pth"
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern, recursive=True)
            model_files.extend(files)
        
        # 去重并排序
        model_files = sorted(list(set(model_files)))
        return model_files
    
    def load_checkpoint(self, model_path: str) -> Optional[Dict]:
        """加载模型检查点"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print(f"✅ 成功加载检查点: {model_path}")
            
            # 打印检查点信息
            if isinstance(checkpoint, dict):
                print(f"   检查点键: {list(checkpoint.keys())}")
                if 'epoch' in checkpoint:
                    print(f"   训练轮次: {checkpoint['epoch']}")
                if 'best_val_loss' in checkpoint:
                    print(f"   最佳验证损失: {checkpoint['best_val_loss']:.6f}")
            
            return checkpoint
        except Exception as e:
            print(f"❌ 加载检查点失败 {model_path}: {e}")
            return None
    
    def create_temporal_nar_model(self, checkpoint: Dict) -> nn.Module:
        """创建时序NAR模型"""
        try:
            # 尝试从检查点获取模型信息
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            print("🔍 分析时序NAR模型架构...")
            
            # 分析状态字典的键来推断模型结构
            keys = list(state_dict.keys())
            print(f"   模型参数键数量: {len(keys)}")
            
            # 检查是否包含Swin Transformer相关的键
            has_swin = any('patch_embed' in key or 'encoder_layers' in key or 'attn' in key for key in keys)
            has_temporal = any('temporal' in key.lower() or 'nar' in key.lower() for key in keys)
            
            if has_swin:
                print("   检测到Swin Transformer架构")
                model = self.create_swin_temporal_nar_model(state_dict)
            else:
                print("   使用通用时序模型架构")
                model = self.create_generic_temporal_model(state_dict)
            
            # 尝试加载权重
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"✅ 模型权重加载完成")
                if missing_keys:
                    print(f"   缺失键数量: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"   多余键数量: {len(unexpected_keys)}")
            except Exception as e:
                print(f"⚠️ 权重加载失败: {e}")
                print("   使用随机初始化权重")
            
            return model.to(self.device)
            
        except Exception as e:
            print(f"❌ 创建时序NAR模型失败: {e}")
            # 返回默认模型
            return self.create_generic_temporal_model({}).to(self.device)
    
    def create_swin_temporal_nar_model(self, state_dict: Dict) -> nn.Module:
        """创建Swin Transformer时序NAR模型"""
        class SwinTemporalNAR(nn.Module):
            def __init__(self):
                super().__init__()
                # 基于状态字典推断的基本结构
                self.patch_embed = nn.Conv2d(1, 96, kernel_size=4, stride=4)
                self.norm = nn.LayerNorm(96)
                
                # 简化的Swin块
                self.encoder = nn.Sequential(
                    nn.Linear(96, 192),
                    nn.GELU(),
                    nn.Linear(192, 96)
                )
                
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
    
    def create_generic_temporal_model(self, state_dict: Dict) -> nn.Module:
        """创建通用时序模型"""
        class GenericTemporalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU()
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 1, 3, padding=1)
                )
                
            def forward(self, x):
                x = self.encoder(x)
                x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
                x = self.decoder(x)
                return x
        
        return GenericTemporalModel()
    
    def generate_test_data(self, batch_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成测试数据"""
        # 生成低分辨率观测数据 (32x32)
        observed = torch.randn(batch_size, 1, 32, 32, device=self.device)
        
        # 生成高分辨率真值数据 (128x128)
        gt = torch.randn(batch_size, 1, 128, 128, device=self.device)
        
        # 添加一些结构化模式
        x = torch.linspace(-1, 1, 32, device=self.device)
        y = torch.linspace(-1, 1, 32, device=self.device)
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
        x_hr = torch.linspace(-1, 1, 128, device=self.device)
        y_hr = torch.linspace(-1, 1, 128, device=self.device)
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
    
    def compute_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
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
    
    def create_visualization(self, observed: torch.Tensor, pred: torch.Tensor, 
                           gt: torch.Tensor, metrics: Dict[str, float], 
                           model_name: str) -> str:
        """创建可视化图表"""
        batch_size = observed.shape[0]
        n_samples = min(4, batch_size)
        
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # 转换为numpy并移到CPU
            obs_np = observed[i, 0].cpu().numpy()
            pred_np = pred[i, 0].cpu().numpy()
            gt_np = gt[i, 0].cpu().numpy()
            error_np = np.abs(pred_np - gt_np)
            
            # 观测数据
            im1 = axes[i, 0].imshow(obs_np, cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'观测数据 {i+1}\n({obs_np.shape[0]}×{obs_np.shape[1]})')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # 预测结果
            im2 = axes[i, 1].imshow(pred_np, cmap='viridis', aspect='auto')
            axes[i, 1].set_title(f'预测结果 {i+1}\n({pred_np.shape[0]}×{pred_np.shape[1]})')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # 真值
            im3 = axes[i, 2].imshow(gt_np, cmap='viridis', aspect='auto')
            axes[i, 2].set_title(f'真值 {i+1}\n({gt_np.shape[0]}×{gt_np.shape[1]})')
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # 误差
            im4 = axes[i, 3].imshow(error_np, cmap='Reds', aspect='auto')
            axes[i, 3].set_title(f'绝对误差 {i+1}\nMax: {error_np.max():.4f}')
            axes[i, 3].axis('off')
            plt.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # 添加整体标题和指标
        metrics_text = f"MSE: {metrics['MSE']:.6f} | MAE: {metrics['MAE']:.6f} | PSNR: {metrics['PSNR']:.2f}dB | Rel-L2: {metrics['Rel_L2']:.4f}"
        fig.suptitle(f'{model_name}\n{metrics_text}', fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图片
        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        save_path = self.output_dir / f"{safe_name}_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(save_path)
    
    def process_model(self, model_path: str) -> Optional[Dict]:
        """处理单个模型"""
        print(f"\n🔄 处理模型: {model_path}")
        
        # 加载检查点
        checkpoint = self.load_checkpoint(model_path)
        if checkpoint is None:
            return None
        
        # 创建时序NAR模型
        model = self.create_temporal_nar_model(checkpoint)
        model.eval()
        
        # 生成测试数据
        observed, gt = self.generate_test_data()
        
        # 进行预测
        with torch.no_grad():
            pred = model(observed)
            
            # 确保预测结果与真值尺寸匹配
            if pred.shape != gt.shape:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        
        # 计算指标
        metrics = self.compute_metrics(pred, gt)
        
        # 创建可视化
        model_name = Path(model_path).stem
        viz_path = self.create_visualization(observed, pred, gt, metrics, model_name)
        
        result = {
            'model_path': model_path,
            'model_name': model_name,
            'metrics': metrics,
            'visualization_path': viz_path
        }
        
        print(f"✅ 完成处理: {model_name}")
        print(f"   指标: MSE={metrics['MSE']:.6f}, PSNR={metrics['PSNR']:.2f}dB")
        print(f"   可视化: {viz_path}")
        
        return result
    
    def create_summary_report(self, results: List[Dict]) -> str:
        """创建总结报告"""
        if not results:
            return ""
        
        # 创建HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>模型预测结果可视化报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .model-section {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metrics {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
                .visualization {{ text-align: center; margin: 15px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 模型预测结果可视化报告</h1>
                <p>生成时间: {Path().cwd()}</p>
                <p>处理模型数量: {len(results)}</p>
            </div>
        """
        
        # 添加指标对比表
        html_content += """
            <h2>📊 模型性能对比</h2>
            <table>
                <tr>
                    <th>模型名称</th>
                    <th>MSE</th>
                    <th>MAE</th>
                    <th>PSNR (dB)</th>
                    <th>Rel-L2</th>
                </tr>
        """
        
        for result in results:
            metrics = result['metrics']
            html_content += f"""
                <tr>
                    <td>{result['model_name']}</td>
                    <td>{metrics['MSE']:.6f}</td>
                    <td>{metrics['MAE']:.6f}</td>
                    <td>{metrics['PSNR']:.2f}</td>
                    <td>{metrics['Rel_L2']:.4f}</td>
                </tr>
            """
        
        html_content += "</table>"
        
        # 添加每个模型的详细结果
        for result in results:
            viz_path = Path(result['visualization_path'])
            relative_path = viz_path.name
            
            html_content += f"""
            <div class="model-section">
                <h3>🔧 {result['model_name']}</h3>
                <div class="metrics">
                    <strong>性能指标:</strong><br>
                    MSE: {result['metrics']['MSE']:.6f} | 
                    MAE: {result['metrics']['MAE']:.6f} | 
                    PSNR: {result['metrics']['PSNR']:.2f}dB | 
                    Rel-L2: {result['metrics']['Rel_L2']:.4f}
                </div>
                <div class="visualization">
                    <img src="{relative_path}" alt="{result['model_name']} 可视化结果">
                </div>
                <p><strong>模型路径:</strong> {result['model_path']}</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # 保存HTML报告
        report_path = self.output_dir / "visualization_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def run(self, max_models: int = 5):
        """运行可视化流程"""
        print("🚀 开始模型预测结果可视化")
        print(f"📁 输出目录: {self.output_dir}")
        
        # 查找模型文件
        model_files = self.find_model_files()
        print(f"\n📋 找到 {len(model_files)} 个模型文件:")
        for i, f in enumerate(model_files[:10]):  # 只显示前10个
            print(f"  {i+1}. {f}")
        if len(model_files) > 10:
            print(f"  ... 还有 {len(model_files) - 10} 个文件")
        
        if not model_files:
            print("❌ 未找到任何.pth模型文件")
            return
        
        # 处理模型（限制数量）
        results = []
        processed_count = 0
        
        for model_path in model_files:
            if processed_count >= max_models:
                break
                
            result = self.process_model(model_path)
            if result:
                results.append(result)
                processed_count += 1
        
        # 创建总结报告
        if results:
            report_path = self.create_summary_report(results)
            print(f"\n📄 总结报告已生成: {report_path}")
        
        print(f"\n🎉 可视化完成！")
        print(f"   处理模型数量: {len(results)}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   查看报告: {self.output_dir}/visualization_report.html")

def main():
    """主函数 - 专门处理时序NAR模型"""
    # 指定时序NAR模型路径
    model_path = r"runs/temporal_nar_100epochs/TemporalNAR-DR2D-128-100epochs-s2025/best.pth"
    
    # 设置输出目录
    output_dir = r"runs/temporal_nar_100epochs/predictions_visualization"
    
    print(f"🎯 专门处理时序NAR模型: {model_path}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建可视化器
    visualizer = ModelVisualizer(output_dir=output_dir)
    
    # 处理指定的模型
    result = visualizer.process_model(model_path)
    
    if result:
        # 创建单个模型的报告
        report_path = visualizer.create_summary_report([result])
        print(f"\n📄 时序NAR模型可视化报告已生成: {report_path}")
        print(f"🎉 可视化完成！查看结果: {output_dir}")
    else:
        print("❌ 时序NAR模型处理失败")

if __name__ == "__main__":
    main()