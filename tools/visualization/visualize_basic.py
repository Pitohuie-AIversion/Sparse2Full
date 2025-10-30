#!/usr/bin/env python3
"""
PDEBench稀疏观测重建可视化工具

提供完整的可视化功能：
- 标准图：GT/Pred/Err热图（统一色标）
- 功率谱（log）
- 边界带局部放大
- 失败案例分析

严格遵循黄金法则：
1. 一致性优先：观测算子H与训练DC必须复用同一实现与配置
2. 可复现：同一YAML+种子，验证指标方差≤1e-4
3. 统一接口：所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
4. 可比性：报告均值±标准差（≥3种子）+资源成本
5. 文档先行：新增任务/算子/模型前，先提交PRD/技术文档补丁
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.visualization import PDEBenchVisualizer
from utils.config import get_environment_info
from tools.check_dc_equivalence import DataConsistencyChecker


class VisualizationTool:
    """可视化工具类
    
    提供完整的可视化功能，支持标准图、功率谱、失败案例分析等
    """
    
    def __init__(self, output_dir: Path, config: Optional[DictConfig] = None):
        """
        Args:
            output_dir: 输出目录
            config: 配置对象
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # 初始化统一的可视化器
        self.visualizer = PDEBenchVisualizer(str(self.output_dir))
        
        # 日志设置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_standard_comparison(self, 
                                 pred: torch.Tensor,
                                 target: torch.Tensor,
                                 observation: torch.Tensor,
                                 title: str = "Comparison",
                                 save_path: Optional[Path] = None) -> Path:
        """创建标准对比图：GT/Pred/Err热图
        
        Args:
            pred: 预测值 [C, H, W]
            target: 真实值 [C, H, W]
            observation: 观测值 [C, H_obs, W_obs]
            title: 图标题
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        # 使用统一的可视化接口
        save_name = title.replace(' ', '_') if save_path is None else save_path.stem
        
        return self.visualizer.create_quadruplet_visualization(
            observation, target, pred, 
            save_name=save_name,
            title=title
        )
    
    def create_power_spectrum_plot(self,
                                 pred: torch.Tensor,
                                 target: torch.Tensor,
                                 title: str = "Power Spectrum",
                                 save_path: Optional[Path] = None) -> Path:
        """创建功率谱对比图
        
        Args:
            pred: 预测值 [C, H, W]
            target: 真实值 [C, H, W]
            title: 图标题
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        # 使用统一的可视化接口
        save_name = title.replace(' ', '_') if save_path is None else save_path.stem
        
        return self.visualizer.plot_power_spectrum_comparison(
            target, pred, save_name=save_name
        )
    
    def _radial_average(self, data: np.ndarray) -> np.ndarray:
        """计算径向平均"""
        H, W = data.shape
        center_y, center_x = H // 2, W // 2
        
        # 创建径向距离网格
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # 径向平均
        max_r = min(center_y, center_x)
        radial_avg = np.zeros(max_r)
        
        for i in range(max_r):
            mask = (r == i)
            if np.any(mask):
                radial_avg[i] = np.mean(data[mask])
        
        return radial_avg
    
    def create_boundary_analysis(self,
                               pred: torch.Tensor,
                               target: torch.Tensor,
                               boundary_width: int = 16,
                               title: str = "Boundary Analysis",
                               save_path: Optional[Path] = None) -> Path:
        """创建边界带分析图
        
        Args:
            pred: 预测值 [C, H, W]
            target: 真实值 [C, H, W]
            boundary_width: 边界带宽度
            title: 图标题
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        # 使用统一的可视化接口
        save_name = title.replace(' ', '_') if save_path is None else save_path.stem
        
        return self.visualizer.plot_boundary_analysis(
            target, pred, 
            boundary_width=boundary_width,
            save_name=save_name
        )
    
    def create_failure_case_analysis(self,
                                   cases: List[Dict[str, Any]],
                                   title: str = "Failure Cases",
                                   save_path: Optional[Path] = None) -> Path:
        """创建失败案例分析图
        
        Args:
            cases: 失败案例列表，每个包含pred, target, error_type等
            title: 图标题
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        # 使用统一的可视化接口
        save_name = title.replace(' ', '_') if save_path is None else save_path.stem
        
        return self.visualizer.create_failure_case_analysis(
            cases, save_name=save_name
        )
    
    def create_metrics_summary_plot(self,
                                  metrics_data: Dict[str, Dict[str, float]],
                                  title: str = "Metrics Summary",
                                  save_path: Optional[Path] = None) -> Path:
        """创建指标汇总图
        
        Args:
            metrics_data: {'method_name': {'metric_name': value}}
            title: 图标题
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        # 使用统一的可视化接口
        save_name = title.replace(' ', '_') if save_path is None else save_path.stem
        
        return self.visualizer.create_metrics_summary_plot(
            metrics_data, save_name=save_name
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDEBench可视化工具")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="输出目录")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config:
        config = OmegaConf.load(args.config)
    
    # 创建可视化工具
    viz_tool = VisualizationTool(
        output_dir=Path(args.output_dir),
        config=config
    )
    
    print(f"可视化工具已初始化，输出目录: {args.output_dir}")
    print("使用VisualizationTool类的方法来创建各种可视化图表")


if __name__ == "__main__":
    main()