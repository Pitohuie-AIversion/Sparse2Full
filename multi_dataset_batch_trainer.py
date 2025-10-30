#!/usr/bin/env python3
"""
多数据集批量训练系统
支持在3种不同PDE数据集上训练所有模型，生成完整的训练配置矩阵
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_dataset_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """数据集配置信息"""
    name: str
    path: str
    pde_type: str
    image_size: int
    keys: List[str]
    batch_size: int
    in_channels: int
    out_channels: int

@dataclass
class TaskConfig:
    """任务配置信息"""
    type: str  # 'sr' or 'crop'
    scale_factor: Optional[int] = None  # for SR tasks
    crop_ratio: Optional[float] = None  # for crop tasks
    description: str = ""
    priority: str = "medium"

@dataclass
class TrainingJob:
    """训练任务配置"""
    dataset: DatasetConfig
    task: TaskConfig
    model_name: str
    config_path: str
    output_dir: str
    experiment_name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    best_metrics: Optional[Dict] = None

class MultiDatasetBatchTrainer:
    """多数据集批量训练管理器"""
    
    def __init__(self, base_config_dir: str = "configs/auto_generated"):
        self.base_config_dir = Path(base_config_dir)
        self.output_dir = Path("runs/multi_dataset_batch")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的模型列表
        self.models = [
            "SwinUNet", "Hybrid", "MLP", "UNet", "FNO2D", 
            "UNet++", "UFNO-UNet", "MLP-Mixer", "SegFormer", "UNetFormer"
        ]
        
        # 数据集配置
        self.datasets = self._load_dataset_configs()
        
        # 任务配置
        self.tasks = {
            "sr_x2": TaskConfig("sr", scale_factor=2, description="超分辨率重建 2x", priority="high"),
            "sr_x4": TaskConfig("sr", scale_factor=4, description="超分辨率重建 4x", priority="high"),
            "crop_20": TaskConfig("crop", crop_ratio=0.2, description="稀疏观测重建 20%", priority="medium"),
            "crop_40": TaskConfig("crop", crop_ratio=0.4, description="稀疏观测重建 40%", priority="medium")
        }
        
        # 训练任务队列
        self.training_jobs: List[TrainingJob] = []
        
        # 状态文件
        self.status_file = self.output_dir / "training_status.json"
        self.results_file = self.output_dir / "training_results.json"
        
    def _load_dataset_configs(self) -> Dict[str, DatasetConfig]:
        """从现有配置文件中加载数据集配置"""
        datasets = {}
        
        # 从workflow_results.json加载数据集信息
        workflow_file = self.base_config_dir / "workflow_results.json"
        if workflow_file.exists():
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
                
            for dataset_info in workflow_data.get('selected_datasets', []):
                name = dataset_info['name']
                
                # 从format_info中获取空间维度和通道信息
                format_info = dataset_info.get('format_info', {})
                spatial_dims = format_info.get('spatial_dims', [128, 128])
                channels = format_info.get('channels', 2)
                
                # 如果spatial_dims为空，从现有配置文件中推断
                if not spatial_dims or len(spatial_dims) < 2:
                    spatial_dims = self._infer_spatial_dims_from_configs(name)
                
                # 如果channels为0，从现有配置文件中推断
                if channels == 0:
                    channels = self._infer_channels_from_configs(name)
                
                datasets[name] = DatasetConfig(
                    name=name,
                    path=dataset_info['path'],
                    pde_type=dataset_info['pde_type'],
                    image_size=spatial_dims[0] if spatial_dims else 128,
                    keys=dataset_info['keys'],
                    batch_size=self._get_optimal_batch_size(spatial_dims),
                    in_channels=channels,
                    out_channels=channels
                )
        
        return datasets
    
    def _infer_spatial_dims_from_configs(self, dataset_name: str) -> List[int]:
        """从现有配置文件中推断空间维度"""
        # 查找匹配的配置文件
        pattern = f"*{dataset_name.lower().replace('-', '_').replace(' ', '_')}*.yaml"
        config_files = list(self.base_config_dir.glob(pattern))
        
        if config_files:
            try:
                with open(config_files[0], 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    image_size = config.get('data', {}).get('image_size', 128)
                    return [image_size, image_size]
            except Exception:
                pass
        
        # 默认值
        return [128, 128]
    
    def _infer_channels_from_configs(self, dataset_name: str) -> int:
        """从现有配置文件中推断通道数"""
        # 查找匹配的配置文件
        pattern = f"*{dataset_name.lower().replace('-', '_').replace(' ', '_')}*.yaml"
        config_files = list(self.base_config_dir.glob(pattern))
        
        if config_files:
            try:
                with open(config_files[0], 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    in_channels = config.get('model', {}).get('in_channels', 2)
                    return in_channels
            except Exception:
                pass
        
        # 默认值
        return 2
    
    def _get_optimal_batch_size(self, spatial_dims: List[int]) -> int:
        """根据空间维度确定最优批次大小"""
        if spatial_dims and len(spatial_dims) >= 2:
            h, w = spatial_dims[0], spatial_dims[1]
            if h >= 512 or w >= 512:
                return 1
            elif h >= 256 or w >= 256:
                return 2
            else:
                return 4
        return 2
    
    def generate_training_matrix(self) -> List[TrainingJob]:
        """生成训练配置矩阵"""
        jobs = []
        
        for dataset_name, dataset_config in self.datasets.items():
            for task_name, task_config in self.tasks.items():
                for model_name in self.models:
                    # 生成实验名称
                    experiment_name = f"{dataset_name}_{task_name}_{model_name.lower()}"
                    
                    # 生成配置文件路径
                    config_path = self._generate_config_file(
                        dataset_config, task_config, model_name, experiment_name
                    )
                    
                    # 创建训练任务
                    job = TrainingJob(
                        dataset=dataset_config,
                        task=task_config,
                        model_name=model_name,
                        config_path=str(config_path),
                        output_dir=f"runs/{experiment_name}",
                        experiment_name=experiment_name
                    )
                    
                    jobs.append(job)
        
        self.training_jobs = jobs
        logger.info(f"生成了 {len(jobs)} 个训练任务")
        return jobs
    
    def _generate_config_file(self, dataset: DatasetConfig, task: TaskConfig, 
                            model_name: str, experiment_name: str) -> Path:
        """生成训练配置文件"""
        
        # 生成完整配置
        config = self._generate_config(dataset, task, model_name)
        
        # 更新实验配置
        config['experiment']['name'] = experiment_name
        config['experiment']['output_dir'] = f"runs/{experiment_name}"
        
        # 保存配置文件
        config_dir = self.output_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / f"{experiment_name}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return config_path

    def _generate_config(self, dataset_info, task_info, model_name):
        """生成训练配置"""
        config = {
            'defaults': ['_self_'],
            'data': {
                '_target_': 'datasets.pdebench.PDEBenchDataModule',
                'data_path': dataset_info.path,
                'dataset_name': dataset_info.name,
                'keys': dataset_info.keys,
                'image_size': dataset_info.image_size,
                'batch_size': dataset_info.batch_size,
                'use_official_format': True,
                'num_workers': 4,
                'pin_memory': True,
                'case_ids': [0, 1, 2, 3],  # 限制使用的case数量
                'dataloader': {
                    'batch_size': dataset_info.batch_size,
                    'num_workers': 4,
                    'pin_memory': True
                }
            },
            'model': {
                'name': model_name,
                'in_channels': dataset_info.in_channels,
                'out_channels': dataset_info.out_channels,
                'image_size': dataset_info.image_size
            },
            'task': {
                'type': task_info.type,
                'scale_factor': task_info.scale_factor,
                'crop_ratio': task_info.crop_ratio
            },
            'experiment': {
                'name': f"{dataset_info.name}_{task_info.type}_x{task_info.scale_factor if task_info.scale_factor else 'crop_' + str(int(task_info.crop_ratio * 100))}_{model_name.lower()}",
                'output_dir': f"runs/{dataset_info.name}_{task_info.type}_x{task_info.scale_factor if task_info.scale_factor else 'crop_' + str(int(task_info.crop_ratio * 100))}_{model_name.lower()}",
                'device': 'cuda',
                'seed': 42,
                'tags': [
                    f"{task_info.type}_{task_info.scale_factor if task_info.scale_factor else 'crop_' + str(int(task_info.crop_ratio * 100))}",
                    dataset_info.pde_type,
                    model_name.lower(),
                    'multi_dataset_batch'
                ],
                'notes': f"Auto-optimized config for {dataset_info.name} with {task_info.type}_{task_info.scale_factor if task_info.scale_factor else 'crop_' + str(int(task_info.crop_ratio * 100))} task"
            },
            'training': {
                'max_epochs': self._get_optimal_epochs(dataset_info, task_info),
                'learning_rate': self._get_optimal_lr(dataset_info, task_info),
                'optimizer': 'adamw',
                'weight_decay': 1e-4,
                'scheduler': 'cosine_warmup',
                'warmup_epochs': 5,
                'use_amp': True,
                'gradient_clip_val': 1.0,
                'early_stopping': {
                    'monitor': 'val_rel_l2',
                    'patience': 10,
                    'min_delta': 1e-4
                },
                'reproducibility': {
                    'deterministic': False,
                    'benchmark': True
                }
            },
            'loss': {
                'reconstruction': {
                    'type': 'l1',
                    'weight': 1.0
                },
                'spectral': {
                    'type': 'spectral_l1',
                    'weight': 0.0,
                    'low_freq_modes': 4
                },
                'data_consistency': {
                    'type': 'l2',
                    'weight': 0.0
                }
            },
            'logging': {
                'use_wandb': False,
                'wandb_project': 'sparse2full_auto',
                'wandb_entity': None,
                'monitor': 'val_rel_l2',
                'mode': 'min',
                'save_top_k': 3,
                'log_every_n_steps': 50,
                'val_check_interval': 1.0
            }
        }
        
        # 更新模型特定配置
        config['model'].update(self._get_model_specific_config(model_name, dataset_info))
        
        return config

    def _get_optimal_epochs(self, dataset_info, task_info):
        """获取最优训练轮数"""
        # 根据数据集大小和任务复杂度调整
        if dataset_info.image_size >= 512:
            return 150
        elif dataset_info.image_size >= 256:
            return 200
        else:
            return 250

    def _get_optimal_lr(self, dataset_info, task_info):
        """获取最优学习率"""
        # 根据批次大小调整学习率
        base_lr = 1e-4
        if dataset_info.batch_size == 1:
            return base_lr * 0.5
        elif dataset_info.batch_size >= 4:
            return base_lr * 1.5
        return base_lr

    def _get_model_specific_config(self, model_name: str, dataset: DatasetConfig) -> Dict:
        """获取模型特定配置"""
        # 根据数据集名称和keys数量动态确定通道数
        actual_in_channels = self._get_actual_channels(dataset)
        actual_out_channels = actual_in_channels  # 输出通道数通常与输入相同
        
        base_config = {
            'image_size': dataset.image_size,
            'in_channels': actual_in_channels,
            'out_channels': actual_out_channels,
        }
        
        if model_name == "SwinUNet":
            return {
                **base_config,
                'embed_dim': 128,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8,
                'mlp_ratio': 4.0,
                'drop_rate': 0.0,
                'drop_path_rate': 0.1,
                'use_checkpoint': True
            }
        elif model_name == "UNet":
            return {
                **base_config,
                'features': [64, 128, 256, 512],
                'dropout': 0.1
            }
        elif model_name == "FNO2D":
            return {
                **base_config,
                'modes1': 16,
                'modes2': 16,
                'width': 64,
                'layers': 4
            }
        elif model_name == "Hybrid":
            return {
                **base_config,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'fno_modes': 16,
                'fno_width': 64
            }
        elif model_name == "MLP":
            # MLP模型特定配置，针对不同图像尺寸调整patch_size
            patch_size = min(16, dataset.image_size // 16)  # 确保patch_size合理
            if patch_size < 4:
                patch_size = 4
            return {
                **base_config,
                'mode': 'patch',
                'patch_size': patch_size,
                'hidden_dims': [256, 512, 256],  # 简化网络结构
                'activation': 'relu',
                'dropout': 0.1,
                'use_positional_encoding': True,
                'coord_encoding_dim': 64
            }
        elif model_name == "MLP-Mixer":
            # MLP-Mixer模型特定配置
            patch_size = min(16, dataset.image_size // 16)
            if patch_size < 4:
                patch_size = 4
            return {
                **base_config,
                'patch_size': patch_size,
                'embed_dim': 256,  # 减小embed_dim
                'depth': 6,  # 减少层数
                'mlp_ratio': (0.5, 2.0),  # 减小MLP比例
                'drop_rate': 0.1,
                'drop_path_rate': 0.0
            }
        else:
            return base_config
    
    def _get_actual_channels(self, dataset: DatasetConfig) -> int:
        """根据数据集配置获取实际的通道数"""
        # 检查是否有现有的配置文件可以参考
        pattern = f"*{dataset.name.lower().replace('-', '_').replace(' ', '_')}*.yaml"
        config_files = list(self.base_config_dir.glob(pattern))
        
        if config_files:
            try:
                with open(config_files[0], 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    keys = config.get('data', {}).get('keys', [])
                    if isinstance(keys, list) and len(keys) > 0:
                        # 根据keys的数量确定通道数
                        return len(keys)
            except Exception as e:
                logger.warning(f"Failed to read config file {config_files[0]}: {e}")
        
        # 如果无法从配置文件获取，使用数据集的默认通道数
        return max(1, dataset.in_channels)

    def save_status(self):
        """保存训练状态"""
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'total_jobs': len(self.training_jobs),
            'completed': len([j for j in self.training_jobs if j.status == 'completed']),
            'running': len([j for j in self.training_jobs if j.status == 'running']),
            'failed': len([j for j in self.training_jobs if j.status == 'failed']),
            'pending': len([j for j in self.training_jobs if j.status == 'pending']),
            'jobs': [asdict(job) for job in self.training_jobs]
        }
        
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
    
    def load_status(self) -> bool:
        """加载训练状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                
                # 重建训练任务列表
                self.training_jobs = []
                for job_data in status_data.get('jobs', []):
                    # 重建数据类对象
                    dataset = DatasetConfig(**job_data['dataset'])
                    task = TaskConfig(**job_data['task'])
                    
                    job = TrainingJob(
                        dataset=dataset,
                        task=task,
                        model_name=job_data['model_name'],
                        config_path=job_data['config_path'],
                        output_dir=job_data['output_dir'],
                        experiment_name=job_data['experiment_name'],
                        status=job_data['status'],
                        start_time=job_data.get('start_time'),
                        end_time=job_data.get('end_time'),
                        best_metrics=job_data.get('best_metrics')
                    )
                    self.training_jobs.append(job)
                
                logger.info(f"加载了 {len(self.training_jobs)} 个训练任务状态")
                return True
            except Exception as e:
                logger.error(f"加载状态失败: {e}")
                return False
        return False
    
    def run_training_job(self, job: TrainingJob) -> bool:
        """执行单个训练任务"""
        try:
            logger.info(f"开始训练: {job.experiment_name}")
            job.status = "running"
            job.start_time = datetime.now().isoformat()
            self.save_status()
            
            # 构建训练命令
            cmd = [
                sys.executable, "train.py",
                f"--config-path={os.path.dirname(job.config_path)}",
                f"--config-name={os.path.basename(job.config_path).replace('.yaml', '')}"
            ]
            
            # 执行训练
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                job.status = "completed"
                job.end_time = datetime.now().isoformat()
                
                # 尝试提取最佳指标
                job.best_metrics = self._extract_metrics(job.output_dir)
                
                logger.info(f"训练完成: {job.experiment_name}")
                return True
            else:
                job.status = "failed"
                job.end_time = datetime.now().isoformat()
                logger.error(f"训练失败: {job.experiment_name}")
                logger.error(f"错误信息: {stderr}")
                return False
                
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now().isoformat()
            logger.error(f"训练异常: {job.experiment_name}, 错误: {e}")
            return False
        finally:
            self.save_status()
    
    def _extract_metrics(self, output_dir: str) -> Optional[Dict]:
        """从训练输出中提取最佳指标"""
        try:
            # 查找metrics文件或日志文件
            output_path = Path(output_dir)
            if output_path.exists():
                # 这里可以实现具体的指标提取逻辑
                # 例如从tensorboard日志、CSV文件或其他日志中提取
                pass
        except Exception as e:
            logger.warning(f"提取指标失败: {e}")
        return None
    
    def run_batch_training(self, max_parallel: int = 2, resume: bool = True):
        """执行批量训练"""
        
        # 尝试加载之前的状态
        if resume:
            self.load_status()
        
        # 如果没有任务，生成训练矩阵
        if not self.training_jobs:
            self.generate_training_matrix()
        
        # 获取待执行的任务
        pending_jobs = [job for job in self.training_jobs if job.status == "pending"]
        
        if not pending_jobs:
            logger.info("没有待执行的训练任务")
            return
        
        logger.info(f"开始批量训练，共 {len(pending_jobs)} 个任务，最大并行数: {max_parallel}")
        
        # 使用线程池执行训练
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_job = {
                executor.submit(self.run_training_job, job): job 
                for job in pending_jobs[:max_parallel]
            }
            
            remaining_jobs = pending_jobs[max_parallel:]
            
            while future_to_job:
                # 等待任意一个任务完成
                done_futures = concurrent.futures.as_completed(future_to_job)
                
                for future in done_futures:
                    job = future_to_job[future]
                    success = future.result()
                    
                    logger.info(f"任务 {job.experiment_name} {'成功' if success else '失败'}")
                    
                    # 移除已完成的任务
                    del future_to_job[future]
                    
                    # 如果还有待执行的任务，添加新任务
                    if remaining_jobs:
                        next_job = remaining_jobs.pop(0)
                        new_future = executor.submit(self.run_training_job, next_job)
                        future_to_job[new_future] = next_job
        
        # 生成最终报告
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """生成训练总结报告"""
        completed_jobs = [job for job in self.training_jobs if job.status == "completed"]
        failed_jobs = [job for job in self.training_jobs if job.status == "failed"]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_jobs': len(self.training_jobs),
                'completed': len(completed_jobs),
                'failed': len(failed_jobs),
                'success_rate': len(completed_jobs) / len(self.training_jobs) if self.training_jobs else 0
            },
            'completed_jobs': [
                {
                    'experiment_name': job.experiment_name,
                    'dataset': job.dataset.name,
                    'task': job.task.type,
                    'model': job.model_name,
                    'duration': self._calculate_duration(job.start_time, job.end_time),
                    'best_metrics': job.best_metrics
                }
                for job in completed_jobs
            ],
            'failed_jobs': [
                {
                    'experiment_name': job.experiment_name,
                    'dataset': job.dataset.name,
                    'task': job.task.type,
                    'model': job.model_name
                }
                for job in failed_jobs
            ]
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练总结报告已保存到: {self.results_file}")
        logger.info(f"成功完成 {len(completed_jobs)}/{len(self.training_jobs)} 个训练任务")
    
    def _calculate_duration(self, start_time: str, end_time: str) -> Optional[str]:
        """计算训练持续时间"""
        try:
            if start_time and end_time:
                start = datetime.fromisoformat(start_time)
                end = datetime.fromisoformat(end_time)
                duration = end - start
                return str(duration)
        except Exception:
            pass
        return None

def main():
    parser = argparse.ArgumentParser(description="多数据集批量训练系统")
    parser.add_argument("--config-dir", default="configs/auto_generated", 
                       help="配置文件目录")
    parser.add_argument("--max-parallel", type=int, default=1, 
                       help="最大并行训练数")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="是否恢复之前的训练状态")
    parser.add_argument("--generate-only", action="store_true",
                       help="仅生成配置文件，不执行训练")
    parser.add_argument("--reset-status", action="store_true",
                       help="重置训练状态，重新开始所有任务")
    parser.add_argument("--status", action="store_true",
                       help="查看训练状态")
    
    args = parser.parse_args()
    
    # 创建训练管理器
    trainer = MultiDatasetBatchTrainer(args.config_dir)
    
    if args.status:
        # 查看训练状态
        if trainer.load_status():
            trainer.generate_summary_report()
        else:
            logger.info("没有找到训练状态文件")
        return
    
    if args.reset_status:
        # 重置训练状态
        if trainer.status_file.exists():
            trainer.status_file.unlink()
            logger.info("已重置训练状态")
        else:
            logger.info("训练状态文件不存在，无需重置")
        return
    
    if args.generate_only:
        # 仅生成配置矩阵
        jobs = trainer.generate_training_matrix()
        logger.info(f"生成了 {len(jobs)} 个训练配置")
        trainer.save_status()
    else:
        # 执行批量训练
        trainer.run_batch_training(
            max_parallel=args.max_parallel,
            resume=args.resume
        )

if __name__ == "__main__":
    main()