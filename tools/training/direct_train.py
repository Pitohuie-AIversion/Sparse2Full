#!/usr/bin/env python3
"""
直接训练脚本 - 绕过Hydra配置问题
直接导入训练器类并调用
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import DictConfig, OmegaConf

# 导入训练相关模块
from train import Trainer


def setup_logging():
    """设置日志"""
    log_dir = Path("runs/direct_training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"direct_train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_config(model_name: str, seed: int, epochs: int = 15, batch_size: int = 2) -> DictConfig:
    """创建训练配置"""
    timestamp = datetime.now().strftime("%Y%m%d")
    exp_name = f"SRx4-DarcyFlow-128-{model_name.upper()}-direct-s{seed}-{timestamp}"
    
    config_dict = {
        "data": {
            "name": "pdebench",
            "dataset_name": "DarcyFlow",
            "data_root": "data/pdebench",
            "batch_size": batch_size,
            "num_workers": 4,
            "pin_memory": True,
            "task": "sr",
            "scale_factor": 4,
            "img_size": 128,
            "crop_size": 32,
            "crop_ratio": 0.25,
            "sampling_strategy": "mixed",
            "uniform_ratio": 0.4,
            "boundary_ratio": 0.3,
            "high_gradient_ratio": 0.3,
            "boundary_width": 4,
            "gradient_threshold": 0.1,
            "normalize": True,
            "augmentation": False,
        },
        "model": {
            "name": model_name,
            "in_channels": 1,
            "out_channels": 1,
            "img_size": 128,
        },
        "training": {
            "epochs": epochs,
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "scheduler": {
                "name": "cosine",
                "T_max": epochs,
                "eta_min": 1e-6,
            },
            "loss": {
                "reconstruction_weight": 1.0,
                "spectral_weight": 0.5,
                "degradation_consistency_weight": 1.0,
                "spectral_modes": 16,
            },
            "early_stopping": {
                "enabled": True,
                "patience": 20,
                "min_delta": 1e-6,
                "monitor": "val_loss",
            },
            "gradient_clipping": {
                "enabled": True,
                "max_norm": 1.0,
            },
            "mixed_precision": {
                "enabled": True,
                "dtype": "float16",
            },
        },
        "experiment": {
            "name": exp_name,
            "seed": seed,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "precision": 16,
            "use_amp": True,
            "output_dir": f"runs/{exp_name}",
        },
        "logging": {
            "log_every_n_steps": 10,
            "val_check_interval": 1.0,
            "save_top_k": 3,
            "monitor": "val_loss",
            "mode": "min",
            "use_tensorboard": True,
            "use_wandb": False,
        },
        "output_dir": f"runs/{exp_name}",
        "checkpoint_dir": f"runs/{exp_name}/checkpoints",
        "log_dir": f"runs/{exp_name}/logs",
    }
    
    return OmegaConf.create(config_dict)


def train_single_model(model_name: str, seed: int, epochs: int = 15, batch_size: int = 2) -> Dict:
    """训练单个模型"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练模型: {model_name}, 种子: {seed}")
    
    start_time = time.time()
    
    try:
        # 创建配置
        config = create_config(model_name, seed, epochs, batch_size)
        
        # 创建训练器
        trainer = Trainer(config)
        
        # 开始训练
        trainer.train()
        
        train_time = time.time() - start_time
        
        result = {
            "model": model_name,
            "seed": seed,
            "status": "success",
            "train_time": train_time,
            "exp_name": config.experiment.name,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        
        logger.info(f"模型 {model_name} (种子 {seed}) 训练成功，耗时 {train_time:.2f}s")
        
    except Exception as e:
        train_time = time.time() - start_time
        
        result = {
            "model": model_name,
            "seed": seed,
            "status": "failed",
            "train_time": train_time,
            "error": str(e),
            "exp_name": f"failed-{model_name}-{seed}",
            "epochs": epochs,
            "batch_size": batch_size,
        }
        
        logger.error(f"模型 {model_name} (种子 {seed}) 训练失败: {str(e)}")
    
    return result


def main():
    """主函数"""
    logger = setup_logging()
    
    # 所有模型列表
    models = [
        "unet", "unet_plus_plus", "fno2d", "ufno_unet",
        "segformer_unetformer", "unetformer", "mlp", "mlp_mixer",
        "liif", "hybrid", "segformer", "swin_unet"
    ]
    
    # 模型特定配置
    model_configs = {
        "fno2d": {"batch_size": 4, "epochs": 20},
        "hybrid": {"batch_size": 2, "epochs": 18},
        "segformer": {"batch_size": 2, "epochs": 18},
        "segformer_unetformer": {"batch_size": 2, "epochs": 18},
        "swin_unet": {"batch_size": 2, "epochs": 18},
        "liif": {"batch_size": 4, "epochs": 20},
    }
    
    seeds = [2025, 2026, 2027]
    base_epochs = 15
    base_batch_size = 2
    
    logger.info(f"开始直接训练，模型数量: {len(models)}, 种子数量: {len(seeds)}")
    logger.info(f"总训练任务数: {len(models) * len(seeds)}")
    
    results = []
    start_time = time.time()
    
    for model_name in models:
        # 获取模型特定配置
        config = model_configs.get(model_name, {})
        epochs = config.get("epochs", base_epochs)
        batch_size = config.get("batch_size", base_batch_size)
        
        for seed in seeds:
            result = train_single_model(model_name, seed, epochs, batch_size)
            results.append(result)
    
    total_time = time.time() - start_time
    logger.info(f"直接训练完成，总耗时: {total_time:.2f}s")
    
    # 生成报告
    generate_report(results, total_time, logger)


def generate_report(results: List[Dict], total_time: float, logger):
    """生成训练报告"""
    # 统计结果
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    logger.info(f"直接训练完成统计:")
    logger.info(f"  成功: {len(successful)}/{len(results)}")
    logger.info(f"  失败: {len(failed)}/{len(results)}")
    logger.info(f"  总耗时: {total_time:.2f}s")
    
    # 保存详细结果
    output_dir = Path("runs/direct_training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON结果
    results_file = output_dir / f"direct_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "total_time": total_time,
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed)
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"详细结果已保存到: {results_file}")
    
    # 打印失败的模型
    if failed:
        logger.error("失败的训练任务:")
        for result in failed:
            logger.error(f"  {result['model']} (种子 {result['seed']}): {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()