#!/usr/bin/env python3
"""
时空解耦序列化训练管线 (Sequential Spatiotemporal Training Pipeline)

功能：
执行三阶段训练策略，复现论文中的 "Swin-UNet + 序列化训练" 实验结果。
1. Phase 1: Spatial Pre-training (仅训练空间重建)
2. Phase 2: Temporal Pre-training (冻结空间，仅训练时序演化)
3. Phase 3: Joint Fine-tuning (联合微调)

用法：
python tools/training/train_sequential_pipeline.py [hydra options]
"""

import os
import sys
import torch
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from models.temporal.components.sequential_spatiotemporal_trainer import SequentialSpatiotemporalTrainer
from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule

logger = logging.getLogger(__name__)

class KeyMapDataLoader:
    """DataLoader包装器，用于适配 Trainer 的输入 Key 需求"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        
    def __iter__(self):
        for batch in self.dataloader:
            new_batch = {
                'input': batch['input_sequence'],
                'target': batch['target_sequence']
            }
            # 保留其他可能有用的键
            for k, v in batch.items():
                if k not in ['input_sequence', 'target_sequence']:
                    new_batch[k] = v
            yield new_batch
            
    def __len__(self):
        return len(self.dataloader)

def freeze_module(module, freeze=True):
    for param in module.parameters():
        param.requires_grad = not freeze

def run_phase(config, phase_name, pretrained_ckpt=None, freeze_spatial=False, freeze_temporal=False):
    logger.info(f"==========================================================")
    logger.info(f"🚀 Starting Phase: {phase_name}")
    logger.info(f"   Spatial Frozen: {freeze_spatial}, Temporal Frozen: {freeze_temporal}")
    logger.info(f"   Spatial Weight: {config.loss.spatial_weight}, Temporal Weight: {config.loss.temporal_weight}")
    logger.info(f"==========================================================")
    
    # 1. 初始化 DataModule
    dm = RealDiffusionReactionDataModule(config)
    dm.setup()
    
    train_loader = KeyMapDataLoader(dm.train_dataloader())
    val_loader = KeyMapDataLoader(dm.val_dataloader())
    
    # 2. 初始化 Trainer
    # 确保 output_dir 包含阶段名
    base_output_dir = Path(config.output_dir)
    config.output_dir = str(base_output_dir / phase_name)
    
    trainer = SequentialSpatiotemporalTrainer(config)
    
    # 3. 加载预训练权重
    if pretrained_ckpt:
        logger.info(f"📥 Loading pretrained weights from {pretrained_ckpt}")
        trainer.load_checkpoint(pretrained_ckpt)
        
    # 4. 冻结/解冻模块
    if freeze_spatial:
        logger.info("❄️ Freezing Spatial Module")
        freeze_module(trainer.spatial_module, True)
        # 重建优化器以排除冻结参数
        trainer.spatial_optimizer = trainer._build_spatial_optimizer()
    else:
        freeze_module(trainer.spatial_module, False)
        
    if freeze_temporal:
        logger.info("❄️ Freezing Temporal Module")
        freeze_module(trainer.temporal_module, True)
        trainer.temporal_optimizer = trainer._build_temporal_optimizer()
    else:
        freeze_module(trainer.temporal_module, False)
        
    # 5. 执行训练
    num_epochs = config.training.epochs
    history, best_model_path = trainer.train(train_loader, val_loader, num_epochs)
    
    logger.info(f"✅ Phase {phase_name} completed. Best model: {best_model_path}")
    return best_model_path

@hydra.main(config_path="../../configs", config_name="ar_training_config", version_base="1.3")
def main(cfg: DictConfig):
    # 强制设置基础配置以适配 SequentialTrainer
    cfg.logging = OmegaConf.create({"log_interval": 10, "checkpoint_interval": 50})
    if not hasattr(cfg, 'loss'):
        cfg.loss = OmegaConf.create({})
        
    # 基础输出目录
    base_exp_name = cfg.get("experiment", {}).get("name", "sequential_experiment")
    run_dir = Path("runs") / f"{base_exp_name}_{hydra.core.hydra_config.HydraConfig.get().job.name}"
    cfg.output_dir = str(run_dir)
    
    # 全局 Epoch 设置 (为了演示快速运行，可以在命令行 override)
    global_epochs = cfg.training.epochs
    
    # ==========================================
    # Phase 1: Spatial Pre-training
    # ==========================================
    cfg_p1 = deepcopy(cfg)
    cfg_p1.loss.spatial_weight = 1.0
    cfg_p1.loss.temporal_weight = 0.0
    # Phase 1 通常只需要较少的 Epochs 来热身
    cfg_p1.training.epochs = max(1, global_epochs // 2) 
    
    ckpt_p1 = run_phase(
        cfg_p1, 
        "Phase1_Spatial", 
        pretrained_ckpt=None, 
        freeze_spatial=False, 
        freeze_temporal=True  # 实际上权重为0梯度也为0，但显式冻结更安全
    )
    
    # ==========================================
    # Phase 2: Temporal Pre-training
    # ==========================================
    cfg_p2 = deepcopy(cfg)
    cfg_p2.loss.spatial_weight = 0.0
    cfg_p2.loss.temporal_weight = 1.0
    cfg_p2.training.epochs = max(1, global_epochs // 2)
    
    ckpt_p2 = run_phase(
        cfg_p2, 
        "Phase2_Temporal", 
        pretrained_ckpt=ckpt_p1, 
        freeze_spatial=True, 
        freeze_temporal=False
    )
    
    # ==========================================
    # Phase 3: Joint Fine-tuning
    # ==========================================
    cfg_p3 = deepcopy(cfg)
    cfg_p3.loss.spatial_weight = 1.0
    cfg_p3.loss.temporal_weight = 1.0 # 或者 0.5
    cfg_p3.training.epochs = global_epochs
    
    ckpt_p3 = run_phase(
        cfg_p3, 
        "Phase3_Joint", 
        pretrained_ckpt=ckpt_p2, 
        freeze_spatial=False, 
        freeze_temporal=False
    )
    
    logger.info("🎉 All phases completed successfully!")
    logger.info(f"🏆 Final Model: {ckpt_p3}")

if __name__ == "__main__":
    main()
