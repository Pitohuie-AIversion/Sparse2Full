#!/usr/bin/env python3
"""
PDEBench主训练脚本 - 支持多种训练模式
遵循黄金法则，集成测试验证和论文包生成
"""

import os
import sys
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# 添加Sparse2Full根目录到Python路径
sparse2full_root = project_root.parent
sys.path.insert(0, str(sparse2full_root))

from training_system.utils.trainers.trainer import PDEBenchTrainer
from training_system.utils.trainers.curriculum_trainer import CurriculumTrainer
from training_system.utils.trainers.benchmark_trainer import BenchmarkTrainer
from training_system.utils.logging_utils import setup_logging
from training_system.utils.reproducibility import set_seed
from training_system.utils.validation import validate_config

logger = logging.getLogger(__name__)


def run_comprehensive_tests() -> bool:
    """运行综合测试套件"""
    logger.info("运行综合测试套件...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_comprehensive_framework.py", "-v"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            logger.info("✅ 综合测试通过")
            return True
        else:
            logger.error("❌ 综合测试失败")
            logger.error(f"标准输出: {result.stdout}")
            logger.error(f"标准错误: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"测试运行失败: {str(e)}")
        return False


def save_config_snapshot(config: DictConfig, output_dir: Path) -> None:
    """保存配置快照"""
    config_file = output_dir / "config_merged.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(config, f)
    logger.info(f"配置快照已保存: {config_file}")


def compute_resource_usage_summary(model) -> Dict[str, Any]:
    """计算资源使用摘要"""
    from utils.metrics import compute_resource_usage
    
    resource_stats = compute_resource_usage(model)
    
    return {
        'total_parameters': resource_stats['total_params'],
        'trainable_parameters': resource_stats['trainable_params'],
        'flops_g': resource_stats['flops_g'],
        'memory_mb': resource_stats['memory_mb']
    }


def generate_paper_package(training_results: Dict[str, Any], config: DictConfig, 
                          output_dir: Path) -> Optional[Path]:
    """生成论文包"""
    logger.info("生成论文包...")
    
    paper_package_dir = output_dir / "paper_package"
    paper_package_dir.mkdir(exist_ok=True)
    
    # 创建标准目录结构
    (paper_package_dir / "configs").mkdir(exist_ok=True)
    (paper_package_dir / "checkpoints").mkdir(exist_ok=True)
    (paper_package_dir / "metrics").mkdir(exist_ok=True)
    (paper_package_dir / "figs").mkdir(exist_ok=True)
    (paper_package_dir / "data_cards").mkdir(exist_ok=True)
    (paper_package_dir / "scripts").mkdir(exist_ok=True)
    
    # 复制配置文件
    import shutil
    config_file = output_dir / "config_merged.yaml"
    if config_file.exists():
        shutil.copy2(config_file, paper_package_dir / "configs" / "config.yaml")
    
    # 复制检查点
    for ckpt_file in ["best_model.pth", "final_model.pth"]:
        src = output_dir / ckpt_file
        if src.exists():
            shutil.copy2(src, paper_package_dir / "checkpoints" / ckpt_file)
    
    # 生成指标文件
    experiment_name = getattr(config, 'experiment_name', 'default_experiment')
    metrics_data = {
        'experiment_info': {
            'name': experiment_name,
            'timestamp': training_results.get('timestamp', ''),
            'total_epochs': training_results.get('total_epochs', 0),
            'best_val_metric': training_results.get('best_val_metric', 0)
        },
        'model_info': training_results.get('resource_usage', {}),
        'training_history': training_results.get('training_history', {}),
        'validation_metrics': training_results.get('val_metrics', {})
    }
    
    with open(paper_package_dir / "metrics" / "experiment_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2, default=str)
    
    # 生成复现脚本
    reproduce_script = paper_package_dir / "scripts" / "reproduce.sh"
    with open(reproduce_script, "w") as f:
        f.write(f"""#!/bin/bash
# 复现实验脚本
# 实验名称: {experiment_name}
# 时间戳: {training_results.get('timestamp', '')}

echo "开始复现实验: {experiment_name}"

# 设置环境
export PYTHONPATH="{project_root}:$PYTHONPATH"

# 运行训练
python scripts/train.py --config-name=train_real_dr_data_simple

echo "复现实验完成"
""")
    
    reproduce_script.chmod(0o755)
    
    # 生成README
    model_name = getattr(config.model, 'name', 'unknown') if hasattr(config, 'model') else 'unknown'
    readme_content = f"""# PDEBench 实验论文包

## 实验信息
- **实验名称**: {experiment_name}
- **时间戳**: {training_results.get('timestamp', '')}
- **总轮数**: {training_results.get('total_epochs', 0)}
- **最佳验证指标**: {training_results.get('best_val_metric', 0):.6f}

## 模型信息
- **模型类型**: {model_name}
- **总参数**: {training_results.get('resource_usage', {}).get('total_parameters', 0):,}
- **可训练参数**: {training_results.get('resource_usage', {}).get('trainable_parameters', 0):,}
- **FLOPs**: {training_results.get('resource_usage', {}).get('flops_g', 0):.2f}G
- **内存使用**: {training_results.get('resource_usage', {}).get('memory_mb', 0):.1f}MB

## 目录结构
```
paper_package/
├── configs/              # 配置文件
│   └── config.yaml      # 合并后的配置
├── checkpoints/         # 模型检查点
│   ├── best_model.pth   # 最佳模型
│   └── final_model.pth  # 最终模型
├── metrics/             # 指标数据
│   └── experiment_metrics.json
├── figs/                # 可视化图表
├── data_cards/          # 数据卡片
├── scripts/             # 复现脚本
│   └── reproduce.sh     # 一键复现脚本
└── README.md           # 本文件
```

## 复现实验
```bash
# 一键复现实验
./scripts/reproduce.sh

# 或使用原始配置
python scripts/train.py --config-name=train_real_dr_data_simple
```

## 数据一致性
本实验遵循PDEBench数据一致性要求：
- 观测算子与训练DC使用相同实现
- 通过一致性验证脚本检查
- 支持可复现性（种子固定）

## 许可证
代码: MIT/Apache-2.0
模型权重: 按各自许可
数据: 按PDEBench许可
"""
    
    with open(paper_package_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    logger.info(f"论文包生成完成: {paper_package_dir}")
    
    return paper_package_dir


@hydra.main(version_base=None, config_path="../../training_system/configs", config_name="spatial/spatial_sr4_config.yaml")
def main(config: DictConfig) -> None:
    """主训练函数"""
    
    # 设置日志
    log_level = getattr(config, 'logging', {}).get('log_level', 'INFO')
    setup_logging(log_level)
    logger.info("🚀 启动PDEBench训练系统")
    
    # 设置随机种子
    seed = getattr(config, 'experiment', {}).get('seed', 42)
    set_seed(seed)
    
    # 验证配置（简化验证，适配Hydra配置结构）
    try:
        # 将OmegaConf转换为普通dict进行验证
        config_dict = OmegaConf.to_container(config, resolve=True)
        if not validate_config(config_dict):
            logger.error("配置验证失败")
            return 1
    except Exception as e:
        logger.warning(f"配置验证跳过: {e}")
        # 继续训练，不中断
    
    # 创建输出目录
    experiment_name = getattr(config, 'experiment_name', 'default_experiment')
    output_dir = Path('./runs') / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置快照
    save_config_snapshot(config, output_dir)
    
    # 运行综合测试（可选）
    if getattr(config, 'run_tests', True):
        if not run_comprehensive_tests():
            logger.warning("测试失败，但继续训练（可通过配置禁用）")
    
    # 根据训练模式选择训练器
    training_mode = getattr(config, 'training_mode', 'basic')
    
    try:
        if training_mode == 'curriculum':
            logger.info("启动课程学习训练模式")
            trainer = CurriculumTrainer(config)
            
        elif training_mode == 'benchmark':
            logger.info("启动基准测试训练模式")
            trainer = BenchmarkTrainer(config)
            
        else:
            logger.info("启动基础训练模式")
            
            # 确保使用CPU设备（Hydra结构化配置安全设置）
            try:
                config.experiment.device = 'cpu'
            except Exception:
                logger.warning("无法设置config.experiment.device，训练器内部将使用CPU兜底")
            
            trainer = PDEBenchTrainer(config)
        
        # 运行训练
        logger.info("开始训练...")
        start_time = time.time()
        
        training_results = trainer.train()
        
        training_time = time.time() - start_time
        training_results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        training_results['training_time'] = training_time
        
        # 计算资源使用（如果训练器有模型）
        if hasattr(trainer, 'model'):
            resource_usage = compute_resource_usage_summary(trainer.model)
            training_results['resource_usage'] = resource_usage
            logger.info(f"模型参数: {resource_usage['total_parameters']:,}")
        
        logger.info(f"✅ 训练完成！用时: {training_time:.2f}s")
        logger.info(f"最佳验证指标: {training_results.get('best_val_metric', 'N/A')}")
        
        # 保存训练结果
        results_file = output_dir / "training_results.json"
        with open(results_file, "w") as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # 生成论文包（如果启用）
        if getattr(config, 'paper_package', {}).get('enabled', False):
            paper_package_dir = generate_paper_package(training_results, config, output_dir)
            logger.info(f"论文包生成完成: {paper_package_dir}")
        
        logger.info("🎉 训练流程全部完成！")
        return 0
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 保存错误信息
        error_file = output_dir / "error_log.txt"
        with open(error_file, "w") as f:
            f.write(f"错误时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"错误追踪:\n{traceback.format_exc()}\n")
        
        return 1


if __name__ == "__main__":
    # 添加Hydra配置搜索路径
    sys.exit(main())