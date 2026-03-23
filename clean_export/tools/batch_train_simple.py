#!/usr/bin/env python3
"""
简化的批量训练脚本 - 使用subprocess调用train.py
通过修改配置文件来训练不同模型
"""

import os
import sys
import time
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml


def setup_logging():
    """设置日志"""
    log_dir = Path("runs/batch_training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_train_simple_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def backup_config():
    """备份原始配置文件"""
    config_path = Path("configs/config.yaml")
    backup_path = Path("configs/config_backup.yaml")
    
    if config_path.exists():
        shutil.copy2(config_path, backup_path)
        return True
    return False


def restore_config():
    """恢复原始配置文件"""
    config_path = Path("configs/config.yaml")
    backup_path = Path("configs/config_backup.yaml")
    
    if backup_path.exists():
        shutil.copy2(backup_path, config_path)
        backup_path.unlink()  # 删除备份文件


def modify_config_for_model(model_name: str, seed: int, epochs: int = 15):
    """修改配置文件以适应特定模型"""
    config_path = Path("configs/config.yaml")
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改配置
    config['defaults'][1] = f'model: {model_name}'  # 修改模型
    
    # 生成实验名称
    timestamp = datetime.now().strftime("%Y%m%d")
    exp_name = f"SRx4-DarcyFlow-128-{model_name.upper()}-batch-s{seed}-{timestamp}"
    config['experiment']['name'] = exp_name
    config['experiment']['seed'] = seed
    
    # 写回配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def train_single_model(model_name: str, seed: int, epochs: int = 15) -> Dict:
    """训练单个模型"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练模型: {model_name}, 种子: {seed}")
    
    start_time = time.time()
    
    try:
        # 修改配置文件
        modify_config_for_model(model_name, seed, epochs)
        
        # 构建训练命令
        cmd = [
            sys.executable, "train.py"
        ]
        
        # 执行训练
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        train_time = time.time() - start_time
        
        if result.returncode == 0:
            # 训练成功
            timestamp = datetime.now().strftime("%Y%m%d")
            exp_name = f"SRx4-DarcyFlow-128-{model_name.upper()}-batch-s{seed}-{timestamp}"
            
            result_dict = {
                "model": model_name,
                "seed": seed,
                "status": "success",
                "train_time": train_time,
                "exp_name": exp_name,
                "epochs": epochs,
                "stdout": result.stdout[-1000:] if result.stdout else "",  # 保留最后1000字符
                "stderr": result.stderr[-1000:] if result.stderr else "",
            }
            
            logger.info(f"模型 {model_name} (种子 {seed}) 训练成功，耗时 {train_time:.2f}s")
        else:
            # 训练失败
            result_dict = {
                "model": model_name,
                "seed": seed,
                "status": "failed",
                "train_time": train_time,
                "error": f"Exit code: {result.returncode}",
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "epochs": epochs,
            }
            
            logger.error(f"模型 {model_name} (种子 {seed}) 训练失败，退出码: {result.returncode}")
            if result.stderr:
                logger.error(f"错误信息: {result.stderr[-500:]}")
        
    except subprocess.TimeoutExpired:
        train_time = time.time() - start_time
        result_dict = {
            "model": model_name,
            "seed": seed,
            "status": "timeout",
            "train_time": train_time,
            "error": "Training timeout (1 hour)",
            "epochs": epochs,
        }
        logger.error(f"模型 {model_name} (种子 {seed}) 训练超时")
        
    except Exception as e:
        train_time = time.time() - start_time
        result_dict = {
            "model": model_name,
            "seed": seed,
            "status": "error",
            "train_time": train_time,
            "error": str(e),
            "epochs": epochs,
        }
        logger.error(f"模型 {model_name} (种子 {seed}) 训练异常: {str(e)}")
    
    return result_dict


def main():
    """主函数"""
    logger = setup_logging()
    
    # 备份原始配置
    if not backup_config():
        logger.error("无法备份配置文件")
        return
    
    try:
        # 所有模型列表
        models = [
            "unet", "unet_plus_plus", "fno2d", "ufno_unet",
            "segformer_unetformer", "unetformer", "mlp", "mlp_mixer",
            "liif", "hybrid", "segformer", "swin_unet"
        ]
        
        # 模型特定配置
        model_configs = {
            "fno2d": {"epochs": 20},
            "hybrid": {"epochs": 18},
            "segformer": {"epochs": 18},
            "segformer_unetformer": {"epochs": 18},
            "swin_unet": {"epochs": 18},
            "liif": {"epochs": 20},
        }
        
        seeds = [2025, 2026, 2027]
        base_epochs = 15
        
        logger.info(f"开始批量训练，模型数量: {len(models)}, 种子数量: {len(seeds)}")
        logger.info(f"总训练任务数: {len(models) * len(seeds)}")
        
        results = []
        start_time = time.time()
        
        for model_name in models:
            # 获取模型特定配置
            config = model_configs.get(model_name, {})
            epochs = config.get("epochs", base_epochs)
            
            for seed in seeds:
                result = train_single_model(model_name, seed, epochs)
                results.append(result)
        
        total_time = time.time() - start_time
        logger.info(f"批量训练完成，总耗时: {total_time:.2f}s")
        
        # 生成报告
        generate_report(results, total_time, logger)
        
    finally:
        # 恢复原始配置
        restore_config()
        logger.info("已恢复原始配置文件")


def generate_report(results: List[Dict], total_time: float, logger):
    """生成训练报告"""
    # 统计结果
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    logger.info(f"批量训练完成统计:")
    logger.info(f"  成功: {len(successful)}/{len(results)}")
    logger.info(f"  失败: {len(failed)}/{len(results)}")
    logger.info(f"  总耗时: {total_time:.2f}s")
    
    # 保存详细结果
    output_dir = Path("runs/batch_training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON结果
    results_file = output_dir / f"simple_batch_results_{timestamp}.json"
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
    
    # 打印成功的模型
    if successful:
        logger.info("成功训练的模型:")
        for result in successful:
            logger.info(f"  {result['model']} (种子 {result['seed']}): {result['train_time']:.2f}s")
    
    # 打印失败的模型
    if failed:
        logger.error("失败的训练任务:")
        for result in failed:
            logger.error(f"  {result['model']} (种子 {result['seed']}): {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()