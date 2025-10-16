#!/usr/bin/env python3
"""
简化的批量训练脚本 - 直接调用训练函数而不是subprocess
避免Hydra配置覆盖问题
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
import hydra
from omegaconf import DictConfig, OmegaConf

# 导入训练相关模块
from train import main as train_main


def setup_logging():
    """设置日志"""
    log_dir = Path("runs/batch_training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simple_batch_train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_single_model(model_name: str, seed: int, epochs: int = 15, batch_size: int = 2) -> Dict:
    """训练单个模型"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练模型: {model_name}, 种子: {seed}")
    
    # 生成实验名称
    timestamp = datetime.now().strftime("%Y%m%d")
    exp_name = f"SRx4-DarcyFlow-128-{model_name.upper()}-batch-s{seed}-{timestamp}"
    
    # 构建配置
    config = {
        "data": "pdebench",
        "model": model_name,
        "train": {
            "epochs": epochs,
        },
        "data": {
            "batch_size": batch_size,
        },
        "experiment": {
            "name": exp_name,
            "seed": seed,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "precision": 16,
            "use_amp": True,
        }
    }
    
    start_time = time.time()
    
    try:
        # 直接调用训练函数
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    f"data=pdebench",
                    f"model={model_name}",
                    f"train.epochs={epochs}",
                    f"+data.batch_size={batch_size}",
                    f"experiment.seed={seed}",
                    f"experiment.name={exp_name}",
                ]
            )
            
            # 调用训练主函数
            train_main(cfg)
            
        train_time = time.time() - start_time
        
        result = {
            "model": model_name,
            "seed": seed,
            "status": "success",
            "train_time": train_time,
            "exp_name": exp_name,
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
            "exp_name": exp_name,
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
    
    logger.info(f"开始批量训练，模型数量: {len(models)}, 种子数量: {len(seeds)}")
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
    logger.info(f"批量训练完成，总耗时: {total_time:.2f}s")
    
    # 生成报告
    generate_report(results, total_time, logger)


def generate_report(results: List[Dict], total_time: float, logger):
    """生成训练报告"""
    # 统计结果
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    logger.info(f"训练完成统计:")
    logger.info(f"  成功: {len(successful)}/{len(results)}")
    logger.info(f"  失败: {len(failed)}/{len(results)}")
    logger.info(f"  总耗时: {total_time:.2f}s")
    
    # 保存详细结果
    output_dir = Path("runs/batch_training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON结果
    results_file = output_dir / f"batch_results_{timestamp}.json"
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