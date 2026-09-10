"""实验复现脚本

提供完整的实验复现功能，包括：
- 环境检查和依赖安装
- 数据准备和预处理
- 模型训练和评测
- 结果验证和对比
- 自动化批量实验

严格按照技术架构文档要求实现可复现的实验流程。

使用方法：
python scripts/reproduce_experiments.py --config configs/config.yaml --seed 42
python scripts/reproduce_experiments.py --batch_mode --configs_dir configs/ --seeds 42,123,456
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger


class ExperimentReproducer:
    """实验复现器
    
    提供完整的实验复现功能
    """
    
    def __init__(self, output_dir: str = "./runs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger('reproduce', self.output_dir / 'reproduction.log')
        
        # 系统要求
        self.requirements = {
            'python_version': (3, 10),
            'torch_version': '2.1.0',
            'cuda_version': '11.8',
            'memory_gb': 16,
            'storage_gb': 50
        }
        
        # 实验配置
        self.experiment_config = {
            'default_seeds': [42, 123, 456],
            'max_parallel_jobs': 2,
            'checkpoint_interval': 10,
            'early_stopping_patience': 20
        }
    
    def check_environment(self) -> bool:
        """检查环境要求"""
        self.logger.info("Checking environment requirements...")
        
        checks_passed = True
        
        # 检查Python版本
        python_version = sys.version_info[:2]
        if python_version < self.requirements['python_version']:
            self.logger.error(f"Python version {python_version} < required {self.requirements['python_version']}")
            checks_passed = False
        else:
            self.logger.info(f"Python version: {python_version} ✓")
        
        # 检查PyTorch
        try:
            import torch
            torch_version = torch.__version__
            self.logger.info(f"PyTorch version: {torch_version} ✓")
            
            # 检查CUDA
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                self.logger.info(f"CUDA version: {cuda_version} ✓")
                self.logger.info(f"GPU count: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    self.logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.logger.warning("CUDA not available, will use CPU")
        
        except ImportError:
            self.logger.error("PyTorch not installed")
            checks_passed = False
        
        # 检查其他依赖
        required_packages = [
            'numpy', 'matplotlib', 'seaborn', 'pandas', 'scipy',
            'omegaconf', 'tqdm', 'pillow', 'scikit-image', 'h5py'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                self.logger.debug(f"{package} ✓")
            except ImportError:
                missing_packages.append(package)
                self.logger.error(f"{package} not found")
        
        if missing_packages:
            self.logger.error(f"Missing packages: {missing_packages}")
            checks_passed = False
        
        # 检查存储空间
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.output_dir)
            free_gb = free / (1024**3)
            
            if free_gb < self.requirements['storage_gb']:
                self.logger.warning(f"Low disk space: {free_gb:.1f}GB < {self.requirements['storage_gb']}GB")
            else:
                self.logger.info(f"Available storage: {free_gb:.1f}GB ✓")
        
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
        
        return checks_passed
    
    def install_dependencies(self) -> bool:
        """安装依赖包"""
        self.logger.info("Installing dependencies...")
        
        try:
            # 检查requirements.txt
            requirements_file = project_root / 'requirements.txt'
            if requirements_file.exists():
                cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.logger.info("Dependencies installed successfully")
                    return True
                else:
                    self.logger.error(f"Failed to install dependencies: {result.stderr}")
                    return False
            else:
                self.logger.warning("requirements.txt not found, installing core packages...")
                
                # 安装核心包
                core_packages = [
                    'torch>=2.1.0',
                    'torchvision',
                    'numpy',
                    'matplotlib',
                    'seaborn',
                    'pandas',
                    'scipy',
                    'omegaconf',
                    'tqdm',
                    'pillow',
                    'scikit-image',
                    'h5py'
                ]
                
                for package in core_packages:
                    cmd = [sys.executable, '-m', 'pip', 'install', package]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        self.logger.error(f"Failed to install {package}: {result.stderr}")
                        return False
                
                self.logger.info("Core packages installed successfully")
                return True
        
        except Exception as e:
            self.logger.error(f"Error installing dependencies: {e}")
            return False
    
    def prepare_data(self, config: DictConfig) -> bool:
        """准备数据"""
        self.logger.info("Preparing data...")
        
        try:
            # 检查数据目录
            data_dir = Path(config.data.get('data_dir', './data'))
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查数据集是否存在
            dataset_name = config.data.get('dataset', 'unknown')
            dataset_path = data_dir / f"{dataset_name}.h5"
            
            if not dataset_path.exists():
                self.logger.warning(f"Dataset {dataset_path} not found")
                
                # 尝试下载数据集（如果有下载脚本）
                download_script = project_root / 'scripts' / 'download_data.py'
                if download_script.exists():
                    self.logger.info("Attempting to download dataset...")
                    cmd = [sys.executable, str(download_script), '--dataset', dataset_name, '--output_dir', str(data_dir)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.logger.info("Dataset downloaded successfully")
                    else:
                        self.logger.error(f"Failed to download dataset: {result.stderr}")
                        return False
                else:
                    self.logger.error("No download script found. Please manually prepare the data.")
                    return False
            
            # 验证数据完整性
            if dataset_path.exists():
                try:
                    import h5py
                    with h5py.File(dataset_path, 'r') as f:
                        keys = list(f.keys())
                        self.logger.info(f"Dataset keys: {keys}")
                        
                        # 检查数据形状
                        for key in keys[:3]:  # 检查前几个键
                            shape = f[key].shape
                            self.logger.info(f"{key} shape: {shape}")
                    
                    self.logger.info("Data validation passed ✓")
                    return True
                
                except Exception as e:
                    self.logger.error(f"Data validation failed: {e}")
                    return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return False
    
    def run_single_experiment(self, 
                            config_path: str, 
                            seed: int = 42,
                            experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """运行单个实验"""
        
        # 加载配置
        config = OmegaConf.load(config_path)
        
        # 生成实验名称
        if experiment_name is None:
            config_name = Path(config_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{config_name}_s{seed}_{timestamp}"
        
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting experiment: {experiment_name}")
        
        # 设置实验日志
        exp_logger = setup_logger(f'exp_{experiment_name}', exp_dir / 'experiment.log')
        
        try:
            # 1. 保存配置快照
            config_snapshot = exp_dir / 'config_merged.yaml'
            OmegaConf.save(config, config_snapshot)
            
            # 2. 设置随机种子
            self._set_random_seeds(seed)
            
            # 3. 准备数据
            if not self.prepare_data(config):
                raise RuntimeError("Data preparation failed")
            
            # 4. 训练模型
            exp_logger.info("Starting training...")
            train_result = self._run_training(config, exp_dir, seed)
            
            if not train_result['success']:
                raise RuntimeError(f"Training failed: {train_result['error']}")
            
            # 5. 评测模型
            exp_logger.info("Starting evaluation...")
            eval_result = self._run_evaluation(config, exp_dir)
            
            if not eval_result['success']:
                raise RuntimeError(f"Evaluation failed: {eval_result['error']}")
            
            # 6. 生成报告
            exp_logger.info("Generating report...")
            report = self._generate_experiment_report(
                experiment_name, config, train_result, eval_result, exp_dir
            )
            
            # 保存结果
            results = {
                'experiment_name': experiment_name,
                'config_path': str(config_path),
                'seed': seed,
                'success': True,
                'train_result': train_result,
                'eval_result': eval_result,
                'report': report,
                'output_dir': str(exp_dir)
            }
            
            results_file = exp_dir / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            exp_logger.info(f"Experiment {experiment_name} completed successfully")
            return results
        
        except Exception as e:
            exp_logger.error(f"Experiment {experiment_name} failed: {e}")
            
            # 保存失败结果
            results = {
                'experiment_name': experiment_name,
                'config_path': str(config_path),
                'seed': seed,
                'success': False,
                'error': str(e),
                'output_dir': str(exp_dir)
            }
            
            results_file = exp_dir / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
    
    def run_batch_experiments(self, 
                            configs_dir: str,
                            seeds: List[int] = None,
                            config_pattern: str = "*.yaml") -> List[Dict[str, Any]]:
        """批量运行实验"""
        
        if seeds is None:
            seeds = self.experiment_config['default_seeds']
        
        configs_path = Path(configs_dir)
        config_files = list(configs_path.glob(config_pattern))
        
        if not config_files:
            self.logger.error(f"No config files found in {configs_dir} with pattern {config_pattern}")
            return []
        
        self.logger.info(f"Found {len(config_files)} config files")
        self.logger.info(f"Will run {len(seeds)} seeds per config")
        self.logger.info(f"Total experiments: {len(config_files) * len(seeds)}")
        
        all_results = []
        
        # 运行所有实验
        for config_file in config_files:
            for seed in seeds:
                try:
                    result = self.run_single_experiment(str(config_file), seed)
                    all_results.append(result)
                    
                    if result['success']:
                        self.logger.info(f"✓ {result['experiment_name']}")
                    else:
                        self.logger.error(f"✗ {result['experiment_name']}: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    self.logger.error(f"Failed to run experiment {config_file} with seed {seed}: {e}")
                    all_results.append({
                        'experiment_name': f"{config_file.stem}_s{seed}_failed",
                        'config_path': str(config_file),
                        'seed': seed,
                        'success': False,
                        'error': str(e)
                    })
        
        # 生成批量实验报告
        self._generate_batch_report(all_results)
        
        return all_results
    
    def _set_random_seeds(self, seed: int) -> None:
        """设置随机种子"""
        import random
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 确保确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 设置环境变量
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _run_training(self, config: DictConfig, exp_dir: Path, seed: int) -> Dict[str, Any]:
        """运行训练"""
        try:
            # 构建训练命令
            train_script = project_root / 'train.py'
            if not train_script.exists():
                return {'success': False, 'error': 'train.py not found'}
            
            cmd = [
                sys.executable, str(train_script),
                '--config', str(exp_dir / 'config_merged.yaml'),
                '--output_dir', str(exp_dir),
                '--seed', str(seed)
            ]
            
            # 运行训练
            self.logger.info(f"Running training command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=project_root,
                timeout=3600 * 8  # 8小时超时
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'command': ' '.join(cmd)
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'stdout': result.stdout,
                    'command': ' '.join(cmd)
                }
        
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Training timeout (8 hours)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_evaluation(self, config: DictConfig, exp_dir: Path) -> Dict[str, Any]:
        """运行评测"""
        try:
            # 查找最佳检查点
            checkpoints_dir = exp_dir / 'checkpoints'
            if not checkpoints_dir.exists():
                return {'success': False, 'error': 'No checkpoints directory found'}
            
            best_checkpoint = checkpoints_dir / 'best.pth'
            if not best_checkpoint.exists():
                # 查找其他检查点
                checkpoints = list(checkpoints_dir.glob('*.pth'))
                if not checkpoints:
                    return {'success': False, 'error': 'No checkpoints found'}
                best_checkpoint = checkpoints[0]
            
            # 构建评测命令
            eval_script = project_root / 'eval.py'
            if not eval_script.exists():
                return {'success': False, 'error': 'eval.py not found'}
            
            eval_dir = exp_dir / 'evaluation'
            eval_dir.mkdir(exist_ok=True)
            
            cmd = [
                sys.executable, str(eval_script),
                '--config', str(exp_dir / 'config_merged.yaml'),
                '--checkpoint', str(best_checkpoint),
                '--output_dir', str(eval_dir)
            ]
            
            # 运行评测
            self.logger.info(f"Running evaluation command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=3600  # 1小时超时
            )
            
            if result.returncode == 0:
                # 读取评测结果
                results_file = eval_dir / 'results.json'
                eval_results = {}
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        eval_results = json.load(f)
                
                return {
                    'success': True,
                    'results': eval_results,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'command': ' '.join(cmd)
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'stdout': result.stdout,
                    'command': ' '.join(cmd)
                }
        
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Evaluation timeout (1 hour)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_experiment_report(self, 
                                  experiment_name: str,
                                  config: DictConfig,
                                  train_result: Dict[str, Any],
                                  eval_result: Dict[str, Any],
                                  exp_dir: Path) -> Dict[str, Any]:
        """生成实验报告"""
        
        report = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config_summary': {
                'model': config.get('model', {}).get('name', 'unknown'),
                'dataset': config.get('data', {}).get('dataset', 'unknown'),
                'task': config.get('data', {}).get('task', 'unknown'),
                'resolution': config.get('data', {}).get('image_size', 'unknown')
            },
            'training_summary': {
                'success': train_result['success'],
                'duration': 'unknown'  # 可以从日志中解析
            },
            'evaluation_summary': {
                'success': eval_result['success']
            }
        }
        
        # 添加评测指标
        if eval_result['success'] and 'results' in eval_result:
            eval_data = eval_result['results']
            if 'average_metrics' in eval_data:
                metrics = eval_data['average_metrics']
                report['metrics'] = {
                    'rel_l2': metrics.get('rel_l2', {}).get('mean', 'N/A'),
                    'mae': metrics.get('mae', {}).get('mean', 'N/A'),
                    'psnr': metrics.get('psnr', {}).get('mean', 'N/A'),
                    'ssim': metrics.get('ssim', {}).get('mean', 'N/A')
                }
        
        # 保存报告
        report_file = exp_dir / 'experiment_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_batch_report(self, all_results: List[Dict[str, Any]]) -> None:
        """生成批量实验报告"""
        
        # 统计信息
        total_experiments = len(all_results)
        successful_experiments = sum(1 for r in all_results if r['success'])
        failed_experiments = total_experiments - successful_experiments
        
        # 按配置分组
        config_groups = {}
        for result in all_results:
            config_name = Path(result['config_path']).stem
            if config_name not in config_groups:
                config_groups[config_name] = []
            config_groups[config_name].append(result)
        
        # 生成报告
        batch_report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_experiments': total_experiments,
                'successful_experiments': successful_experiments,
                'failed_experiments': failed_experiments,
                'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0
            },
            'config_groups': {}
        }
        
        # 为每个配置组生成统计
        for config_name, results in config_groups.items():
            successful_results = [r for r in results if r['success']]
            
            group_stats = {
                'total_runs': len(results),
                'successful_runs': len(successful_results),
                'failed_runs': len(results) - len(successful_results),
                'success_rate': len(successful_results) / len(results) if results else 0
            }
            
            # 如果有成功的实验，计算指标统计
            if successful_results:
                metrics_data = []
                for result in successful_results:
                    if 'report' in result and 'metrics' in result['report']:
                        metrics_data.append(result['report']['metrics'])
                
                if metrics_data:
                    # 计算指标的均值和标准差
                    metric_names = ['rel_l2', 'mae', 'psnr', 'ssim']
                    group_stats['metrics'] = {}
                    
                    for metric in metric_names:
                        values = [m[metric] for m in metrics_data if metric in m and m[metric] != 'N/A']
                        if values:
                            group_stats['metrics'][metric] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'min': np.min(values),
                                'max': np.max(values)
                            }
            
            batch_report['config_groups'][config_name] = group_stats
        
        # 保存批量报告
        batch_report_file = self.output_dir / 'batch_experiment_report.json'
        with open(batch_report_file, 'w') as f:
            json.dump(batch_report, f, indent=2, default=str)
        
        # 生成Markdown报告
        self._generate_batch_report_markdown(batch_report)
        
        self.logger.info(f"Batch experiment report saved to {batch_report_file}")
    
    def _generate_batch_report_markdown(self, batch_report: Dict[str, Any]) -> None:
        """生成Markdown格式的批量实验报告"""
        
        md_content = f"""# 批量实验报告

**生成时间**: {batch_report['timestamp']}

## 总体统计

- **总实验数**: {batch_report['summary']['total_experiments']}
- **成功实验数**: {batch_report['summary']['successful_experiments']}
- **失败实验数**: {batch_report['summary']['failed_experiments']}
- **成功率**: {batch_report['summary']['success_rate']:.1%}

## 配置组详情

"""
        
        for config_name, stats in batch_report['config_groups'].items():
            md_content += f"""
### {config_name}

- **总运行数**: {stats['total_runs']}
- **成功运行数**: {stats['successful_runs']}
- **失败运行数**: {stats['failed_runs']}
- **成功率**: {stats['success_rate']:.1%}

"""
            
            if 'metrics' in stats:
                md_content += "#### 指标统计\n\n"
                md_content += "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |\n"
                md_content += "|------|------|--------|--------|--------|\n"
                
                for metric, values in stats['metrics'].items():
                    md_content += f"| {metric.upper()} | {values['mean']:.4f} | {values['std']:.4f} | {values['min']:.4f} | {values['max']:.4f} |\n"
                
                md_content += "\n"
        
        # 保存Markdown报告
        md_file = self.output_dir / 'batch_experiment_report.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Experiment Reproducer')
    parser.add_argument('--config', type=str,
                       help='Path to single config file')
    parser.add_argument('--configs_dir', type=str,
                       help='Directory containing config files for batch mode')
    parser.add_argument('--config_pattern', type=str, default='*.yaml',
                       help='Pattern to match config files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for single experiment')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                       help='Comma-separated seeds for batch mode')
    parser.add_argument('--output_dir', type=str, default='./runs',
                       help='Output directory for experiments')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Run in batch mode')
    parser.add_argument('--check_env_only', action='store_true',
                       help='Only check environment and exit')
    parser.add_argument('--install_deps', action='store_true',
                       help='Install dependencies')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建复现器
        reproducer = ExperimentReproducer(args.output_dir)
        
        # 检查环境
        if not reproducer.check_environment():
            if args.install_deps:
                if not reproducer.install_dependencies():
                    print("Failed to install dependencies")
                    return 1
            else:
                print("Environment check failed. Use --install_deps to install dependencies.")
                return 1
        
        if args.check_env_only:
            print("Environment check passed!")
            return 0
        
        # 运行实验
        if args.batch_mode:
            if not args.configs_dir:
                print("--configs_dir required for batch mode")
                return 1
            
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
            results = reproducer.run_batch_experiments(
                args.configs_dir, seeds, args.config_pattern
            )
            
            successful = sum(1 for r in results if r['success'])
            print(f"Batch experiments completed: {successful}/{len(results)} successful")
            
        else:
            if not args.config:
                print("--config required for single experiment mode")
                return 1
            
            result = reproducer.run_single_experiment(args.config, args.seed)
            
            if result['success']:
                print(f"Experiment completed successfully: {result['experiment_name']}")
            else:
                print(f"Experiment failed: {result.get('error', 'Unknown error')}")
                return 1
        
        return 0
        
    except Exception as e:
        logging.error(f"Reproduction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())