#!/usr/bin/env python3
"""
PDEBench稀疏观测重建论文材料生成工具

严格遵循黄金法则：
1. 一致性优先：观测算子H与训练DC必须复用同一实现与配置
2. 可复现：同一YAML+种子，验证指标方差≤1e-4
3. 统一接口：所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
4. 可比性：报告均值±标准差（≥3种子）+资源成本
5. 文档先行：新增任务/算子/模型前，先提交PRD/技术文档补丁

生成内容：
- 数据卡片（来源/许可/切分）
- 配置快照（最终YAML）
- 模型权重（关键ckpt，走LFS）
- 指标汇总（主表/显著性/CSV/每case JSONL）
- 可视化图表（代表图/失败案例/谱图）
- 复现脚本（一键复现与汇总）
- README（环境/命令/结果重现）
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 项目导入
from utils.config import get_environment_info
from utils.reproducibility import set_seed

class PaperPackageGenerator:
    """论文材料包生成器
    
    自动生成完整的论文材料包，包含：
    1. 数据卡片和配置快照
    2. 模型权重和指标汇总
    3. 可视化图表和复现脚本
    4. README和环境信息
    """
    
    def __init__(self, config: DictConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.package_config = config.get('paper_package', {})
        
        # 创建输出目录结构
        self.create_directory_structure()
        
        # 设置日志
        self.setup_logging()
        
        # 收集环境信息
        self.env_info = get_environment_info()
        
    def create_directory_structure(self):
        """创建论文材料包目录结构"""
        self.dirs = {
            'root': self.output_dir,
            'data_cards': self.output_dir / 'data_cards',
            'configs': self.output_dir / 'configs',
            'checkpoints': self.output_dir / 'checkpoints',
            'metrics': self.output_dir / 'metrics',
            'figs': self.output_dir / 'figs',
            'scripts': self.output_dir / 'scripts',
            'logs': self.output_dir / 'logs'
        }
        
        # 创建所有目录
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logging.info(f"Created paper package directory structure at {self.output_dir}")
    
    def setup_logging(self):
        """设置日志系统"""
        log_file = self.dirs['logs'] / 'package_generation.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Paper package generation started")
    
    def generate_data_cards(self):
        """生成数据卡片"""
        self.logger.info("Generating data cards...")
        
        data_config = self.config.get('data', {})
        
        # 基本数据卡片模板
        data_card = {
            'dataset_name': data_config.get('dataset_name', 'PDEBench'),
            'version': data_config.get('version', '1.0'),
            'source': {
                'url': data_config.get('source_url', 'https://github.com/pdebench/PDEBench'),
                'license': data_config.get('license', 'MIT'),
                'citation': data_config.get('citation', 'PDEBench: An Extensive Benchmark for Scientific Machine Learning')
            },
            'description': {
                'task_type': data_config.get('task_type', 'sparse_observation_reconstruction'),
                'variables': data_config.get('variables', []),
                'spatial_resolution': data_config.get('spatial_resolution', [256, 256]),
                'temporal_steps': data_config.get('temporal_steps', 1),
                'boundary_conditions': data_config.get('boundary_conditions', 'periodic')
            },
            'splits': {
                'train': data_config.get('train_split', 0.7),
                'val': data_config.get('val_split', 0.15),
                'test': data_config.get('test_split', 0.15),
                'split_method': data_config.get('split_method', 'random'),
                'split_seed': data_config.get('split_seed', 42)
            },
            'preprocessing': {
                'normalization': data_config.get('normalization', 'z_score'),
                'observation_operator': data_config.get('observation_operator', {}),
                'degradation_params': data_config.get('degradation_params', {})
            },
            'statistics': self.compute_data_statistics_from_config(data_config),
            'meta': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0',
                'environment': self.env_info
            }
        }
        
        # 保存数据卡片 - 转换OmegaConf对象为普通字典
        def convert_to_serializable(obj):
            """递归转换OmegaConf对象为可序列化的普通对象"""
            if hasattr(obj, '_content'):  # OmegaConf对象
                return OmegaConf.to_container(obj, resolve=True)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_data_card = convert_to_serializable(data_card)
        
        data_card_path = self.dirs['data_cards'] / 'dataset_card.json'
        with open(data_card_path, 'w') as f:
            json.dump(serializable_data_card, f, indent=2)
        
        # 生成Markdown版本
        self.generate_data_card_markdown(data_card)
        
        self.logger.info(f"Data cards saved to {self.dirs['data_cards']}")
    
    def compute_data_statistics_from_config(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """从配置中计算数据统计量"""
        try:
            # 尝试从配置中获取数据路径
            data_path = data_config.get('data_path', data_config.get('path', ''))
            
            if data_path and Path(data_path).exists():
                return self.compute_data_statistics(Path(data_path))
            else:
                # 返回模板统计量
                return {
                    'total_samples': data_config.get('total_samples', 10000),
                    'train_samples': int(data_config.get('total_samples', 10000) * data_config.get('train_split', 0.7)),
                    'val_samples': int(data_config.get('total_samples', 10000) * data_config.get('val_split', 0.15)),
                    'test_samples': int(data_config.get('total_samples', 10000) * data_config.get('test_split', 0.15)),
                    'data_shape': data_config.get('spatial_resolution', [256, 256]),
                    'channels': data_config.get('channels', 1),
                    'data_range': data_config.get('data_range', [-3.0, 3.0]),
                    'mean': data_config.get('mean', 0.0),
                    'std': data_config.get('std', 1.0),
                    'note': 'Statistics computed from configuration (actual data not found)'
                }
        except Exception as e:
            self.logger.warning(f"Could not compute data statistics: {e}")
            return {'error': str(e)}

    def compute_data_statistics(self, data_path: Path) -> Dict[str, Any]:
        """计算数据统计量
        
        Args:
            data_path: 数据路径
            
        Returns:
            statistics: 数据统计信息
        """
        try:
            import h5py
            import numpy as np
            
            stats = {}
            
            if data_path.suffix == '.h5' or data_path.suffix == '.hdf5':
                with h5py.File(data_path, 'r') as f:
                    # 获取数据集信息
                    datasets = list(f.keys())
                    stats['datasets'] = datasets
                    
                    if datasets:
                        # 分析第一个数据集
                        first_dataset = f[datasets[0]]
                        stats['data_shape'] = list(first_dataset.shape)
                        stats['data_dtype'] = str(first_dataset.dtype)
                        
                        # 采样计算统计量（避免内存溢出）
                        if first_dataset.size > 1000000:  # 如果数据太大，采样
                            sample_indices = np.random.choice(
                                first_dataset.size, 
                                size=min(100000, first_dataset.size), 
                                replace=False
                            )
                            sample_data = first_dataset.flat[sample_indices]
                        else:
                            sample_data = first_dataset[:]
                        
                        stats['data_range'] = [float(np.min(sample_data)), float(np.max(sample_data))]
                        stats['data_mean'] = float(np.mean(sample_data))
                        stats['data_std'] = float(np.std(sample_data))
                        stats['total_samples'] = first_dataset.shape[0] if len(first_dataset.shape) > 0 else 1
                        
                        if len(first_dataset.shape) >= 3:
                            stats['channels'] = first_dataset.shape[1] if len(first_dataset.shape) == 4 else 1
                        else:
                            stats['channels'] = 1
            
            elif data_path.suffix == '.nc':
                import xarray as xr
                
                ds = xr.open_dataset(data_path)
                data_vars = list(ds.data_vars.keys())
                stats['data_vars'] = data_vars
                
                if data_vars:
                    first_var = ds[data_vars[0]]
                    stats['data_shape'] = list(first_var.shape)
                    stats['data_dtype'] = str(first_var.dtype)
                    
                    # 计算统计量
                    sample_data = first_var.values.flatten()
                    if len(sample_data) > 100000:
                        sample_data = np.random.choice(sample_data, 100000, replace=False)
                    
                    stats['data_range'] = [float(np.min(sample_data)), float(np.max(sample_data))]
                    stats['data_mean'] = float(np.mean(sample_data))
                    stats['data_std'] = float(np.std(sample_data))
                    stats['total_samples'] = first_var.shape[0] if len(first_var.shape) > 0 else 1
                    stats['channels'] = first_var.shape[1] if len(first_var.shape) >= 3 else 1
                
                ds.close()
            
            else:
                # 其他格式的基本信息
                stats['file_size_mb'] = data_path.stat().st_size / (1024 * 1024)
                stats['file_format'] = data_path.suffix
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Could not compute statistics for {data_path}: {e}")
            return {'error': str(e)}
    
    def generate_data_card_markdown(self, data_card: Dict[str, Any]):
        """生成数据卡片Markdown版本"""
        md_path = self.dirs['data_cards'] / 'dataset_card.md'
        
        with open(md_path, 'w') as f:
            f.write("# 数据集卡片\n\n")
            
            # 基本信息
            f.write("## 基本信息\n\n")
            f.write(f"- **数据集名称**: {data_card['dataset_name']}\n")
            f.write(f"- **版本**: {data_card['version']}\n")
            f.write(f"- **许可证**: {data_card['source']['license']}\n")
            f.write(f"- **来源**: {data_card['source']['url']}\n\n")
            
            # 任务描述
            desc = data_card['description']
            f.write("## 任务描述\n\n")
            f.write(f"- **任务类型**: {desc['task_type']}\n")
            f.write(f"- **空间分辨率**: {desc['spatial_resolution']}\n")
            f.write(f"- **边界条件**: {desc['boundary_conditions']}\n\n")
            
            # 数据切分
            splits = data_card['splits']
            f.write("## 数据切分\n\n")
            f.write(f"- **训练集**: {splits['train']:.1%}\n")
            f.write(f"- **验证集**: {splits['val']:.1%}\n")
            f.write(f"- **测试集**: {splits['test']:.1%}\n")
            f.write(f"- **切分方法**: {splits['split_method']}\n")
            f.write(f"- **随机种子**: {splits['split_seed']}\n\n")
            
            # 统计信息
            if 'statistics' in data_card and data_card['statistics']:
                stats = data_card['statistics']
                f.write("## 统计信息\n\n")
                f.write(f"- **总样本数**: {stats.get('total_samples', 'N/A')}\n")
                f.write(f"- **数据形状**: {stats.get('data_shape', 'N/A')}\n")
                f.write(f"- **通道数**: {stats.get('channels', 'N/A')}\n")
                f.write(f"- **数据范围**: {stats.get('data_range', 'N/A')}\n\n")
    
    def collect_config_snapshots(self):
        """收集配置快照"""
        self.logger.info("Collecting configuration snapshots...")
        
        # 保存当前完整配置
        config_snapshot = {
            'config': OmegaConf.to_container(self.config, resolve=True),
            'timestamp': datetime.now().isoformat(),
            'git_info': self.get_git_info(),
            'environment': self.env_info
        }
        
        # 保存JSON格式
        config_json_path = self.dirs['configs'] / 'config_merged.json'
        with open(config_json_path, 'w') as f:
            json.dump(config_snapshot, f, indent=2)
        
        # 保存YAML格式
        config_yaml_path = self.dirs['configs'] / 'config_merged.yaml'
        with open(config_yaml_path, 'w') as f:
            yaml.dump(config_snapshot, f, default_flow_style=False)
        
        self.logger.info(f"Configuration snapshots saved to {self.dirs['configs']}")
    
    def get_git_info(self) -> Dict[str, Any]:
        """获取Git信息"""
        try:
            import subprocess
            
            # 获取当前commit hash
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=Path.cwd(),
                text=True
            ).strip()
            
            # 获取当前分支
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=Path.cwd(),
                text=True
            ).strip()
            
            # 检查是否有未提交的更改
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=Path.cwd(),
                text=True
            ).strip()
            
            return {
                'commit_hash': commit_hash,
                'branch': branch,
                'has_uncommitted_changes': bool(status),
                'status': status
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get git info: {e}")
            return {'error': str(e)}
    
    def collect_checkpoints(self):
        """收集模型权重"""
        self.logger.info("Collecting model checkpoints...")
        
        # 查找runs目录中的checkpoints
        runs_dir = Path('runs')
        if not runs_dir.exists():
            self.logger.warning("No runs directory found")
            return
        
        checkpoint_info = []
        
        # 遍历所有实验目录
        for exp_dir in runs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            # 查找checkpoints
            ckpt_dir = exp_dir / 'checkpoints'
            if ckpt_dir.exists():
                for ckpt_file in ckpt_dir.glob('*.ckpt'):
                    # 复制重要的checkpoints
                    if any(keyword in ckpt_file.name.lower() 
                          for keyword in ['best', 'final', 'epoch']):
                        
                        dest_path = self.dirs['checkpoints'] / f"{exp_dir.name}_{ckpt_file.name}"
                        shutil.copy2(ckpt_file, dest_path)
                        
                        checkpoint_info.append({
                            'experiment': exp_dir.name,
                            'checkpoint': ckpt_file.name,
                            'path': str(dest_path.relative_to(self.output_dir)),
                            'size_mb': ckpt_file.stat().st_size / (1024 * 1024),
                            'modified': datetime.fromtimestamp(ckpt_file.stat().st_mtime).isoformat()
                        })
        
        # 保存checkpoint信息
        ckpt_info_path = self.dirs['checkpoints'] / 'checkpoint_info.json'
        with open(ckpt_info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        self.logger.info(f"Collected {len(checkpoint_info)} checkpoints")
    
    def collect_metrics(self):
        """收集指标汇总"""
        self.logger.info("Collecting metrics...")
        
        # 查找runs目录中的指标文件
        runs_dir = Path('runs')
        if not runs_dir.exists():
            self.logger.warning("No runs directory found")
            return
        
        all_metrics = []
        experiment_summaries = {}
        
        # 遍历所有实验目录
        for exp_dir in runs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            exp_name = exp_dir.name
            exp_metrics = {
                'experiment': exp_name,
                'metrics': {},
                'config': {},
                'resources': {}
            }
            
            # 收集指标文件
            metrics_files = list(exp_dir.glob('**/metrics*.json*'))
            for metrics_file in metrics_files:
                try:
                    if metrics_file.suffix == '.jsonl':
                        # JSONL格式
                        with open(metrics_file, 'r') as f:
                            for line in f:
                                metric_data = json.loads(line)
                                all_metrics.append({
                                    'experiment': exp_name,
                                    **metric_data
                                })
                    else:
                        # JSON格式
                        with open(metrics_file, 'r') as f:
                            metric_data = json.load(f)
                            exp_metrics['metrics'].update(metric_data)
                            
                except Exception as e:
                    self.logger.warning(f"Could not read metrics file {metrics_file}: {e}")
            
            # 收集配置信息
            config_files = list(exp_dir.glob('**/config*.yaml')) + list(exp_dir.glob('**/config*.json'))
            for config_file in config_files:
                try:
                    if config_file.suffix == '.yaml':
                        with open(config_file, 'r') as f:
                            config_data = yaml.safe_load(f)
                    else:
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                    
                    exp_metrics['config'].update(config_data)
                    
                except Exception as e:
                    self.logger.warning(f"Could not read config file {config_file}: {e}")
            
            experiment_summaries[exp_name] = exp_metrics
        
        # 保存所有指标
        if all_metrics:
            all_metrics_path = self.dirs['metrics'] / 'all_metrics.jsonl'
            with open(all_metrics_path, 'w') as f:
                for metric in all_metrics:
                    f.write(json.dumps(metric) + '\n')
        
        # 保存实验汇总
        if experiment_summaries:
            summary_path = self.dirs['metrics'] / 'experiment_summaries.json'
            with open(summary_path, 'w') as f:
                json.dump(experiment_summaries, f, indent=2)
        
        # 生成主表
        self.generate_main_results_table(experiment_summaries)
        
        self.logger.info(f"Collected metrics from {len(experiment_summaries)} experiments")
    
    def generate_main_results_table(self, experiment_summaries: Dict[str, Any]):
        """生成主结果表格"""
        if not experiment_summaries:
            return
        
        # 生成CSV格式主表
        csv_path = self.dirs['metrics'] / 'main_results.csv'
        
        # 提取关键指标
        key_metrics = ['rel_l2', 'mae', 'psnr', 'ssim', 'params_m', 'flops_g']
        
        with open(csv_path, 'w') as f:
            # 写入表头
            f.write('Experiment,' + ','.join(key_metrics) + '\n')
            
            # 写入每个实验的结果
            for exp_name, exp_data in experiment_summaries.items():
                metrics = exp_data.get('metrics', {})
                row = [exp_name]
                
                for metric in key_metrics:
                    value = metrics.get(metric, 'N/A')
                    if isinstance(value, (int, float)):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                
                f.write(','.join(row) + '\n')
        
        # 生成Markdown格式主表
        self.generate_main_results_markdown(experiment_summaries, key_metrics)
    
    def generate_main_results_markdown(self, experiment_summaries: Dict[str, Any], key_metrics: List[str]):
        """生成Markdown格式主结果表"""
        md_path = self.dirs['metrics'] / 'main_results.md'
        
        with open(md_path, 'w') as f:
            f.write("# 主要实验结果\n\n")
            
            # 表格标题
            f.write("| 实验 | " + " | ".join(key_metrics) + " |\n")
            f.write("|" + "---|" * (len(key_metrics) + 1) + "\n")
            
            # 表格内容
            for exp_name, exp_data in experiment_summaries.items():
                metrics = exp_data.get('metrics', {})
                row = [exp_name]
                
                for metric in key_metrics:
                    value = metrics.get(metric, 'N/A')
                    if isinstance(value, (int, float)):
                        if metric in ['rel_l2', 'mae']:
                            row.append(f"{value:.4f}")
                        elif metric in ['psnr', 'ssim']:
                            row.append(f"{value:.2f}")
                        elif metric in ['params_m', 'flops_g']:
                            row.append(f"{value:.1f}")
                        else:
                            row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                
                f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n## 指标说明\n\n")
            f.write("- **rel_l2**: 相对L2误差\n")
            f.write("- **mae**: 平均绝对误差\n")
            f.write("- **psnr**: 峰值信噪比\n")
            f.write("- **ssim**: 结构相似性指数\n")
            f.write("- **params_m**: 参数量（百万）\n")
            f.write("- **flops_g**: 计算量（十亿FLOPs）\n")
    
    def collect_figures(self):
        """收集可视化图表"""
        self.logger.info("Collecting figures...")
        
        # 查找runs目录中的图表文件
        runs_dir = Path('runs')
        if not runs_dir.exists():
            self.logger.warning("No runs directory found")
            return
        
        figure_info = []
        
        # 遍历所有实验目录
        for exp_dir in runs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # 查找图表文件
            fig_patterns = ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.svg']
            for pattern in fig_patterns:
                for fig_file in exp_dir.rglob(pattern):
                    # 复制图表文件
                    dest_path = self.dirs['figs'] / f"{exp_dir.name}_{fig_file.name}"
                    shutil.copy2(fig_file, dest_path)
                    
                    figure_info.append({
                        'experiment': exp_dir.name,
                        'figure': fig_file.name,
                        'path': str(dest_path.relative_to(self.output_dir)),
                        'type': fig_file.suffix[1:],
                        'size_kb': fig_file.stat().st_size / 1024
                    })
        
        # 保存图表信息
        if figure_info:
            fig_info_path = self.dirs['figs'] / 'figure_info.json'
            with open(fig_info_path, 'w') as f:
                json.dump(figure_info, f, indent=2)
        
        self.logger.info(f"Collected {len(figure_info)} figures")
    
    def generate_reproduction_scripts(self):
        """生成复现脚本"""
        self.logger.info("Generating reproduction scripts...")
        
        # 生成主复现脚本
        main_script_path = self.dirs['scripts'] / 'reproduce_all.py'
        
        with open(main_script_path, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
PDEBench稀疏观测重建实验复现脚本

使用方法:
    python reproduce_all.py --config config_merged.yaml --seed 42

要求:
    - Python 3.10+
    - PyTorch >= 2.1
    - 所有依赖包已安装
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并打印输出"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Reproduce PDEBench experiments')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    print("=== PDEBench实验复现 ===")
    print(f"配置文件: {args.config}")
    print(f"随机种子: {args.seed}")
    print(f"GPU设备: {args.gpu}")
    print()
    
    # 1. 数据一致性检查
    print("1. 运行数据一致性检查...")
    cmd = f"python tools/check_dc_equivalence.py --config-name consistency_check seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("数据一致性检查失败！")
        return 1
    
    # 2. 训练模型
    print("2. 开始训练...")
    cmd = f"python tools/train.py --config-path ../configs --config-name {Path(args.config).stem} seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("训练失败！")
        return 1
    
    # 3. 评估模型
    print("3. 开始评估...")
    cmd = f"python tools/eval.py --config-path ../configs --config-name {Path(args.config).stem} seed={args.seed}"
    if not run_command(cmd, cwd=project_root):
        print("评估失败！")
        return 1
    
    print("=== 实验复现完成 ===")
    return 0

if __name__ == "__main__":
    exit(main())
''')
        
        # 生成环境安装脚本
        env_script_path = self.dirs['scripts'] / 'setup_environment.py'
        
        with open(env_script_path, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
PDEBench环境安装脚本
"""

import subprocess
import sys

def install_requirements():
    """安装依赖包"""
    requirements = [
        "torch>=2.1.0",
        "torchvision",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "hydra-core",
        "omegaconf",
        "tensorboard",
        "tqdm",
        "h5py",
        "netcdf4",
        "xarray"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])

if __name__ == "__main__":
    install_requirements()
    print("Environment setup completed!")
''')
        
        # 设置脚本可执行权限
        main_script_path.chmod(0o755)
        env_script_path.chmod(0o755)
        
        self.logger.info(f"Reproduction scripts saved to {self.dirs['scripts']}")
    
    def generate_readme(self):
        """生成README文件"""
        self.logger.info("Generating README...")
        
        readme_path = self.output_dir / 'README.md'
        
        with open(readme_path, 'w') as f:
            f.write(f"""# PDEBench稀疏观测重建论文材料包

本材料包包含了PDEBench稀疏观测重建实验的完整材料，支持论文审阅和结果复现。

## 目录结构

```
paper_package/
├── data_cards/          # 数据卡片（来源/许可/切分）
├── configs/            # 配置快照（最终YAML）
├── checkpoints/        # 模型权重（关键ckpt）
├── metrics/           # 指标汇总（主表/显著性/CSV）
├── figs/              # 可视化图表（代表图/失败案例/谱图）
├── scripts/           # 复现脚本（一键复现与汇总）
├── logs/              # 生成日志
└── README.md          # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装Python依赖
python scripts/setup_environment.py

# 或手动安装
pip install torch>=2.1.0 torchvision numpy scipy matplotlib seaborn pandas scikit-learn hydra-core omegaconf tensorboard tqdm h5py netcdf4 xarray
```

### 2. 数据准备

请从以下来源下载PDEBench数据集：
- 官方网站: https://github.com/pdebench/PDEBench
- 数据详情: 参见 `data_cards/dataset_card.md`

### 3. 复现实验

```bash
# 一键复现所有实验
python scripts/reproduce_all.py --config configs/config_merged.yaml --seed 42

# 或分步执行
python tools/check_dc_equivalence.py  # 数据一致性检查
python tools/train.py                 # 训练模型
python tools/eval.py                  # 评估模型
```

## 主要结果

详细结果请参见：
- `metrics/main_results.md` - 主要实验结果表格
- `metrics/main_results.csv` - CSV格式结果
- `figs/` - 可视化图表

## 黄金法则验证

本实验严格遵循以下黄金法则：

1. **一致性优先**: 观测算子H与训练DC复用同一实现与配置
2. **可复现**: 同一YAML+种子，验证指标方差≤1e-4
3. **统一接口**: 所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
4. **可比性**: 报告均值±标准差（≥3种子）+资源成本
5. **文档先行**: 完整的文档和测试覆盖

## 验证清单

- [ ] 数据一致性检查通过（MSE < 1e-8）
- [ ] 训练脚本运行成功
- [ ] 评估指标计算正确
- [ ] 可视化图表生成完整
- [ ] 复现脚本可执行

## 技术支持

如有问题，请检查：
1. Python版本 >= 3.10
2. PyTorch版本 >= 2.1
3. CUDA环境配置正确
4. 数据路径设置正确

## 许可证

本材料包遵循MIT许可证，数据集请遵循各自的许可证要求。

## 引用

如果使用本材料包，请引用：

```bibtex
@article{{pdebench_sparse_reconstruction,
  title={{PDEBench稀疏观测重建：深度学习方法的系统性评估}},
  author={{作者姓名}},
  journal={{期刊名称}},
  year={{2024}}
}}
```

---

生成时间: {datetime.now().isoformat()}
生成器版本: 1.0
""")
        
        self.logger.info(f"README saved to {readme_path}")
    
    def generate_package(self):
        """生成完整的论文材料包"""
        self.logger.info("Starting paper package generation...")
        
        try:
            # 1. 生成数据卡片
            self.generate_data_cards()
            
            # 2. 收集配置快照
            self.collect_config_snapshots()
            
            # 3. 收集模型权重
            self.collect_checkpoints()
            
            # 4. 收集指标汇总
            self.collect_metrics()
            
            # 5. 收集可视化图表
            self.collect_figures()
            
            # 6. 生成复现脚本
            self.generate_reproduction_scripts()
            
            # 7. 生成README
            self.generate_readme()
            
            # 8. 生成元信息
            self.generate_meta_info()
            
            self.logger.info("Paper package generation completed successfully!")
            self.logger.info(f"Package saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Paper package generation failed: {e}")
            return False
    
    def generate_meta_info(self):
        """生成元信息文件"""
        meta_info = {
            'package_info': {
                'name': 'PDEBench稀疏观测重建论文材料包',
                'version': '1.0',
                'generated_at': datetime.now().isoformat(),
                'generator': 'PaperPackageGenerator',
                'total_size_mb': self.calculate_package_size()
            },
            'contents': {
                'data_cards': len(list(self.dirs['data_cards'].glob('*'))),
                'configs': len(list(self.dirs['configs'].glob('*'))),
                'checkpoints': len(list(self.dirs['checkpoints'].glob('*.ckpt'))),
                'metrics': len(list(self.dirs['metrics'].glob('*'))),
                'figures': len(list(self.dirs['figs'].glob('*'))),
                'scripts': len(list(self.dirs['scripts'].glob('*')))
            },
            'environment': self.env_info,
            'git_info': self.get_git_info()
        }
        
        meta_path = self.output_dir / 'package_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)
    
    def calculate_package_size(self) -> float:
        """计算材料包总大小（MB）"""
        total_size = 0
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # 转换为MB


@hydra.main(version_base=None, config_path="../configs", config_name="paper_package")
def main(cfg: DictConfig) -> None:
    """主生成函数
    
    Args:
        cfg: 配置对象
    """
    # 设置输出目录
    output_dir = Path(cfg.get('output_dir', 'paper_package'))
    
    # 创建生成器
    generator = PaperPackageGenerator(cfg, output_dir)
    
    # 生成材料包
    success = generator.generate_package()
    
    if success:
        print(f"\n✅ 论文材料包生成成功！")
        print(f"📁 输出目录: {output_dir.absolute()}")
        print(f"📊 包大小: {generator.calculate_package_size():.1f} MB")
        print(f"\n📋 内容清单:")
        for name, path in generator.dirs.items():
            if name != 'root':
                count = len(list(path.glob('*')))
                print(f"   - {name}: {count} 个文件")
        
        print(f"\n🚀 使用方法:")
        print(f"   cd {output_dir}")
        print(f"   python scripts/reproduce_all.py --config configs/config_merged.yaml")
        
        return 0
    else:
        print("❌ 论文材料包生成失败！")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)