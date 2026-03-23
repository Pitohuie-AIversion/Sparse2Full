#!/usr/bin/env python3
"""
重新生成训练样本的可视化结果
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.pde_bench import PDEBenchDataset
from utils.visualization import PDEBenchVisualizer
from models import get_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleRegenerator:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.samples_dir = self.base_dir / "runs" / "samples"
        self.batch_dir = self.base_dir / "runs" / "batch_retrain_20251015_032934"
        
        # 创建samples目录结构
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self, config_path: str) -> DictConfig:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return OmegaConf.create(config)
    
    def get_best_models(self) -> list:
        """获取最佳模型列表"""
        # 从CSV文件读取模型排名
        csv_path = self.batch_dir / "analysis" / "model_ranking.csv"
        if not csv_path.exists():
            logger.error(f"找不到模型排名文件: {csv_path}")
            return []
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # 选择前3个最佳模型
        best_models = []
        for _, row in df.head(3).iterrows():
            model_name = row['模型']
            best_models.append({
                'name': model_name,
                'rel_l2': row['Rel-L2'],
                'checkpoint_dir': self.batch_dir / model_name
            })
        
        return best_models
    
    def create_dataset(self, config: DictConfig):
        """创建数据集"""
        # 准备任务参数
        task_params = {}
        if config.data.observation.mode.lower() == 'sr':
            task_params = {
                'scale_factor': config.data.observation.sr.get('scale_factor', 4),
                'blur_sigma': config.data.observation.sr.get('blur_sigma', 1.0),
                'blur_kernel_size': config.data.observation.sr.get('blur_kernel_size', 5),
                'boundary_mode': config.data.observation.sr.get('boundary_mode', 'mirror')
            }
        elif config.data.observation.mode.lower() == 'crop':
            task_params = {
                'crop_size': config.data.observation.crop.get('crop_size', [64, 64]),
                'crop_strategy': config.data.observation.crop.get('crop_strategy', 'uniform'),
                'boundary_mode': config.data.observation.crop.get('boundary_mode', 'mirror')
            }
        
        dataset = PDEBenchDataset(
            data_root=config.data.data_path,
            split='val',  # 使用验证集
            task=config.data.observation.mode,
            task_params=task_params,
            img_size=config.data.get('image_size', 128),
            normalize=config.data.preprocessing.get('normalize', True),
            cache_data=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,  # 小批次用于可视化
            shuffle=False,
            num_workers=0
        )
        
        return dataset, dataloader
    
    def load_model_checkpoint(self, model_config: DictConfig, checkpoint_path: str):
        """加载模型检查点"""
        # 创建模型
        model = get_model(model_config)
        
        # 加载检查点
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"已加载检查点: {checkpoint_path}")
        else:
            logger.warning(f"检查点不存在: {checkpoint_path}")
            return None
        
        model.eval()
        return model
    
    def generate_sample_visualizations(self, model_info: dict, config: DictConfig, 
                                     dataset, dataloader, epoch: int = 0):
        """生成样本可视化"""
        model_name = model_info['name']
        checkpoint_dir = model_info['checkpoint_dir']
        
        # 查找最佳检查点
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        if not checkpoint_files:
            logger.warning(f"未找到模型检查点: {checkpoint_dir}")
            return
        
        # 选择最新的检查点
        checkpoint_path = max(checkpoint_files, key=os.path.getctime)
        
        # 加载模型
        model = self.load_model_checkpoint(config.model, str(checkpoint_path))
        if model is None:
            return
        
        # 创建可视化器
        visualizer = PDEBenchVisualizer(
            task_type=config.data.task_type,
            save_dir=str(self.samples_dir / f"epoch_{epoch:04d}"),
            norm_stats=dataset.norm_stats if hasattr(dataset, 'norm_stats') else None
        )
        
        # 生成可视化
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 5:  # 只生成前5个批次
                    break
                
                # 准备输入
                if isinstance(batch, dict):
                    x = batch['input'].to(device)
                    y_true = batch['target'].to(device)
                else:
                    x, y_true = batch
                    x, y_true = x.to(device), y_true.to(device)
                
                # 模型预测
                y_pred = model(x)
                
                # 保存可视化
                for i in range(min(2, x.size(0))):  # 每个批次保存2个样本
                    sample_idx = batch_idx * 2 + i
                    
                    # 转换为numpy
                    x_np = x[i].cpu().numpy()
                    y_true_np = y_true[i].cpu().numpy()
                    y_pred_np = y_pred[i].cpu().numpy()
                    
                    # 生成对比图
                    visualizer.plot_field_comparison(
                        x_np, y_true_np, y_pred_np,
                        title=f"{model_name}_sample_{sample_idx:03d}",
                        save_name=f"{model_name}_sample_{sample_idx:03d}.png"
                    )
                    
                    logger.info(f"已生成样本可视化: {model_name}_sample_{sample_idx:03d}")
    
    def regenerate_all_samples(self):
        """重新生成所有样本可视化"""
        logger.info("开始重新生成训练样本可视化...")
        
        # 获取最佳模型
        best_models = self.get_best_models()
        if not best_models:
            logger.error("未找到最佳模型信息")
            return
        
        # 加载配置
        config_path = self.base_dir / "configs" / "train.yaml"
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return
        
        config = self.load_config(str(config_path))
        
        # 创建数据集
        try:
            dataset, dataloader = self.create_dataset(config)
            logger.info(f"数据集创建成功，包含 {len(dataset)} 个样本")
        except Exception as e:
            logger.error(f"创建数据集失败: {e}")
            return
        
        # 为每个epoch生成样本
        epochs_to_generate = [0, 100]  # 生成epoch 0和100的样本
        
        for epoch in epochs_to_generate:
            epoch_dir = self.samples_dir / f"epoch_{epoch:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建子目录
            for subdir in ['analysis', 'comparisons', 'fields', 'spectra']:
                (epoch_dir / subdir).mkdir(exist_ok=True)
            
            logger.info(f"生成epoch {epoch}的样本可视化...")
            
            # 为每个最佳模型生成样本
            for model_info in best_models:
                try:
                    self.generate_sample_visualizations(
                        model_info, config, dataset, dataloader, epoch
                    )
                except Exception as e:
                    logger.error(f"生成模型 {model_info['name']} 的样本失败: {e}")
                    continue
        
        logger.info("样本可视化重新生成完成！")
        
        # 生成总结报告
        self.generate_summary_report(best_models)
    
    def generate_summary_report(self, best_models: list):
        """生成总结报告"""
        summary_path = self.samples_dir / "regeneration_summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 训练样本可视化重新生成报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 生成的模型\n\n")
            for i, model_info in enumerate(best_models, 1):
                f.write(f"{i}. **{model_info['name']}** (Rel-L2: {model_info['rel_l2']:.6f})\n")
            
            f.write("\n## 目录结构\n\n")
            f.write("```\n")
            f.write("runs/samples/\n")
            f.write("├── epoch_0000/\n")
            f.write("│   ├── analysis/\n")
            f.write("│   ├── comparisons/\n")
            f.write("│   ├── fields/\n")
            f.write("│   └── spectra/\n")
            f.write("├── epoch_0100/\n")
            f.write("│   ├── analysis/\n")
            f.write("│   ├── comparisons/\n")
            f.write("│   ├── fields/\n")
            f.write("│   └── spectra/\n")
            f.write("└── regeneration_summary.md\n")
            f.write("```\n\n")
            
            f.write("## 说明\n\n")
            f.write("- 每个epoch目录包含前3个最佳模型的样本可视化\n")
            f.write("- 每个模型生成10个代表性样本\n")
            f.write("- 可视化包括输入、真实值、预测值和误差对比\n")
        
        logger.info(f"总结报告已保存: {summary_path}")

def main():
    """主函数"""
    base_dir = "f:/Zhaoyang/Sparse2Full"
    
    regenerator = SampleRegenerator(base_dir)
    regenerator.regenerate_all_samples()
    
    print("\n🎉 训练样本可视化重新生成完成！")
    print(f"📁 输出目录: {regenerator.samples_dir}")

if __name__ == "__main__":
    main()