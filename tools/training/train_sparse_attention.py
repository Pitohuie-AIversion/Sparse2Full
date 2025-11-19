#!/usr/bin/env python3
"""
稀疏注意力模型训练脚本
Senseiver架构 - 专门用于极端稀疏传感器场景

遵循黄金法则，确保H/DC一致性
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.training.train_basic import main as train_basic_main
from tools.training.train_basic import (
    create_model_local, create_optimizer, create_scheduler, 
    train_epoch, validate_epoch, save_checkpoint
)
from datasets.pdebench import PDEBenchDataModule
from models.sparse_attention_encoder import SparseAttentionEncoder, SparseSwinUNet
from ops.total_loss import TotalLoss
from utils.reproducibility import set_seed
from utils.config import get_environment_info


def create_sparse_data_module(cfg):
    """创建支持稀疏观测的数据模块"""
    data_module = PDEBenchDataModule(cfg.data)
    
    # 确保数据模块支持稀疏观测生成
    if not hasattr(data_module, 'sparse_observation'):
        logging.warning("数据模块不支持稀疏观测，将使用默认配置")
    
    return data_module


def validate_sparse_config(cfg):
    """验证稀疏注意力配置"""
    model_name = cfg.model.name.lower()
    
    if model_name in ['sparse_attention_encoder', 'sparse_swin_unet']:
        # 检查必要的稀疏配置
        if 'sparse_encoder_config' not in cfg.model.params:
            raise ValueError(f"模型 {model_name} 需要 sparse_encoder_config 配置")
        
        sparse_config = cfg.model.params.sparse_encoder_config
        required_keys = ['embed_dim', 'num_heads']
        for key in required_keys:
            if key not in sparse_config:
                raise ValueError(f"sparse_encoder_config 缺少必要参数: {key}")
        
        # 验证输入通道数
        if cfg.model.params.in_channels < 3:
            logging.warning(f"稀疏模型建议输入通道数≥3 ([baseline, coords, mask])，当前为 {cfg.model.params.in_channels}")
    
    return True


def main(cfg):
    """稀疏注意力模型训练主函数"""
    
    # 验证配置
    validate_sparse_config(cfg)
    
    # 设置增强的实验名称
    sparse_ratio = cfg.model.params.sparse_encoder_config.get('sparse_ratio', 0.1)
    exp_suffix = f"sparse{sparse_ratio:.2f}"
    
    # 更新配置以包含稀疏标识
    if 'experiment_name' not in cfg:
        cfg.experiment_name = f"{cfg.data.observation.mode}-{cfg.data.get('dataset_name', 'PDEBench')}-" \
                             f"{cfg.data.image_size}-{cfg.model.name}-{exp_suffix}-" \
                             f"s{cfg.training.seed}"
    
    logging.info(f"Starting sparse attention training: {cfg.experiment_name}")
    logging.info(f"Sparse ratio: {sparse_ratio}")
    logging.info(f"Model: {cfg.model.name}")
    
    # 调用基础训练函数
    train_basic_main(cfg)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 使用Hydra配置系统
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(version_base=None, config_path="../configs", config_name="train_sparse_swin_unet")
    def hydra_main(cfg: DictConfig) -> None:
        """Hydra主函数包装器"""
        try:
            main(cfg)
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
    
    hydra_main()