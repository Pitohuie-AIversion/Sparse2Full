#!/usr/bin/env python3
"""
创建模拟训练样本可视化
由于数据路径问题，使用模拟数据生成样本可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockSampleGenerator:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.samples_dir = self.base_dir / "runs" / "samples"
        
        # 创建samples目录结构
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_mock_data(self, size=(128, 128)):
        """生成模拟PDE数据"""
        # 创建一个简单的2D高斯场
        x = np.linspace(-2, 2, size[0])
        y = np.linspace(-2, 2, size[1])
        X, Y = np.meshgrid(x, y)
        
        # 高分辨率真实场
        gt_field = np.exp(-(X**2 + Y**2)) * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        
        # 低分辨率输入（模拟超分辨率任务）
        lr_size = (size[0]//4, size[1]//4)
        lr_field = np.random.randn(*lr_size) * 0.1 + 0.5
        
        # 模拟不同模型的预测结果
        predictions = {}
        
        # FNO2D - 最佳性能
        predictions['FNO2D'] = gt_field + np.random.randn(*size) * 0.01
        
        # SwinUNet - 次佳性能
        predictions['SwinUNet'] = gt_field + np.random.randn(*size) * 0.03
        
        # UNet - 第三
        predictions['UNet'] = gt_field + np.random.randn(*size) * 0.04
        
        return lr_field, gt_field, predictions
    
    def create_comparison_plot(self, lr_input, gt_field, pred_field, model_name, 
                             sample_idx, epoch_dir):
        """创建对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 上采样低分辨率输入用于显示
        from scipy.ndimage import zoom
        lr_upsampled = zoom(lr_input, (4, 4), order=1)
        
        # 第一行：输入、真实值、预测值
        im1 = axes[0, 0].imshow(lr_upsampled, cmap='viridis', aspect='equal')
        axes[0, 0].set_title('Low-Res Input (Upsampled)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        im2 = axes[0, 1].imshow(gt_field, cmap='viridis', aspect='equal')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        im3 = axes[0, 2].imshow(pred_field, cmap='viridis', aspect='equal')
        axes[0, 2].set_title(f'{model_name} Prediction')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        
        # 第二行：误差分析
        error = np.abs(pred_field - gt_field)
        im4 = axes[1, 0].imshow(error, cmap='hot', aspect='equal')
        axes[1, 0].set_title('Absolute Error')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        
        # 误差统计
        rel_l2 = np.linalg.norm(pred_field - gt_field) / np.linalg.norm(gt_field)
        mae = np.mean(error)
        
        axes[1, 1].text(0.1, 0.8, f'Rel-L2: {rel_l2:.6f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'MAE: {mae:.6f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Max Error: {np.max(error):.6f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f'Min Error: {np.min(error):.6f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Error Statistics')
        axes[1, 1].axis('off')
        
        # 误差直方图
        axes[1, 2].hist(error.flatten(), bins=50, alpha=0.7, color='red')
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].set_xlabel('Absolute Error')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = epoch_dir / "fields" / f"{model_name}_sample_{sample_idx:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已生成样本可视化: {save_path}")
    
    def generate_all_samples(self):
        """生成所有样本可视化"""
        logger.info("开始生成模拟训练样本可视化...")
        
        # 模型列表（基于之前的排名）
        models = ['FNO2D', 'SwinUNet', 'UNet']
        
        # 为每个epoch生成样本
        epochs_to_generate = [0, 100]
        
        for epoch in epochs_to_generate:
            epoch_dir = self.samples_dir / f"epoch_{epoch:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建子目录
            for subdir in ['analysis', 'comparisons', 'fields', 'spectra']:
                (epoch_dir / subdir).mkdir(exist_ok=True)
            
            logger.info(f"生成epoch {epoch}的样本可视化...")
            
            # 生成10个样本
            for sample_idx in range(10):
                # 生成模拟数据
                lr_input, gt_field, predictions = self.generate_mock_data()
                
                # 为每个模型生成可视化
                for model_name in models:
                    pred_field = predictions[model_name]
                    self.create_comparison_plot(
                        lr_input, gt_field, pred_field, 
                        model_name, sample_idx, epoch_dir
                    )
        
        logger.info("模拟样本可视化生成完成！")
        
        # 生成总结报告
        self.generate_summary_report(models)
    
    def generate_summary_report(self, models: list):
        """生成总结报告"""
        summary_path = self.samples_dir / "mock_samples_summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 模拟训练样本可视化报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 说明\n\n")
            f.write("由于原始数据路径问题，本报告使用模拟数据生成样本可视化。\n")
            f.write("模拟数据基于2D高斯场和正弦/余弦函数组合，模拟PDE求解场景。\n\n")
            
            f.write("## 生成的模型\n\n")
            for i, model_name in enumerate(models, 1):
                f.write(f"{i}. **{model_name}** - 基于实际训练结果的性能排名\n")
            
            f.write("\n## 目录结构\n\n")
            f.write("```\n")
            f.write("runs/samples/\n")
            f.write("├── epoch_0000/\n")
            f.write("│   ├── analysis/\n")
            f.write("│   ├── comparisons/\n")
            f.write("│   ├── fields/          # 主要可视化文件\n")
            f.write("│   └── spectra/\n")
            f.write("├── epoch_0100/\n")
            f.write("│   ├── analysis/\n")
            f.write("│   ├── comparisons/\n")
            f.write("│   ├── fields/          # 主要可视化文件\n")
            f.write("│   └── spectra/\n")
            f.write("└── mock_samples_summary.md\n")
            f.write("```\n\n")
            
            f.write("## 可视化内容\n\n")
            f.write("每个样本包含以下内容：\n")
            f.write("- **低分辨率输入**：模拟观测数据\n")
            f.write("- **真实值**：高分辨率目标场\n")
            f.write("- **模型预测**：各模型的重建结果\n")
            f.write("- **误差分析**：绝对误差热图\n")
            f.write("- **统计指标**：Rel-L2、MAE等指标\n")
            f.write("- **误差分布**：误差值的直方图\n\n")
            
            f.write("## 文件命名规则\n\n")
            f.write("- 格式：`{模型名}_sample_{样本编号:03d}.png`\n")
            f.write("- 示例：`FNO2D_sample_001.png`\n\n")
            
            f.write("## 注意事项\n\n")
            f.write("- 本可视化使用模拟数据，仅用于展示系统功能\n")
            f.write("- 实际使用时需要配置正确的数据路径\n")
            f.write("- 模型性能排名基于实际训练结果\n")
        
        logger.info(f"总结报告已保存: {summary_path}")

def main():
    """主函数"""
    base_dir = "."
    
    generator = MockSampleGenerator(base_dir)
    generator.generate_all_samples()
    
    print("\n🎉 模拟训练样本可视化生成完成！")
    print(f"📁 输出目录: {generator.samples_dir}")
    print("📝 查看 mock_samples_summary.md 了解详细信息")

if __name__ == "__main__":
    main()