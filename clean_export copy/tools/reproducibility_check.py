#!/usr/bin/env python3
"""
PDEBench稀疏观测重建可复现性验证脚本

严格遵循黄金法则：
2. 可复现：同一YAML+种子，验证指标方差≤1e-4

验证目标：
- 同一配置文件+随机种子，多次运行的指标方差≤1e-4
- 验证训练过程的确定性
- 验证评估指标的一致性
- 生成可复现性报告
"""

import os
import sys
import json
import logging
import argparse
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 项目导入
try:
    from utils.reproducibility import set_seed, get_deterministic_config
    from utils.config import get_environment_info
    from utils.metrics import compute_metrics
except ImportError as e:
    logging.warning(f"Import error: {e}. Using fallback implementations.")
    
    def set_seed(seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 设置确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    def get_deterministic_config():
        """获取确定性配置"""
        return {
            'torch.backends.cudnn.deterministic': torch.backends.cudnn.deterministic,
            'torch.backends.cudnn.benchmark': torch.backends.cudnn.benchmark,
            'torch.are_deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled()
        }
    
    def get_environment_info():
        """获取环境信息"""
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'torch_version': torch.__version__,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    
    def compute_metrics(pred, target):
        """计算基本指标"""
        mse = torch.mean((pred - target) ** 2).item()
        mae = torch.mean(torch.abs(pred - target)).item()
        
        # 相对L2误差
        rel_l2 = torch.norm(pred - target, p=2) / torch.norm(target, p=2)
        rel_l2 = rel_l2.item()
        
        return {
            'mse': mse,
            'mae': mae,
            'rel_l2': rel_l2
        }


class SimpleModel(nn.Module):
    """简单的测试模型"""
    
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class SyntheticDataset(Dataset):
    """合成测试数据集"""
    
    def __init__(self, num_samples: int = 100, image_size: int = 64, seed: int = 42):
        self.num_samples = num_samples
        self.image_size = image_size
        
        # 使用固定种子生成数据
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            # 生成输入和目标
            x = torch.randn(1, image_size, image_size)
            y = torch.randn(1, image_size, image_size)
            
            self.data.append({'input': x, 'target': y})
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class ReproducibilityChecker:
    """可复现性检查器
    
    验证同一配置+种子下的结果一致性：
    1. 模型初始化一致性
    2. 训练过程一致性
    3. 评估指标一致性
    4. 方差≤1e-4验证
    """
    
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # 检查配置
        self.repro_config = config.get('reproducibility_check', {})
        self.num_runs = self.repro_config.get('num_runs', 3)
        self.num_epochs = self.repro_config.get('num_epochs', 5)
        self.tolerance = self.repro_config.get('variance_tolerance', 1e-4)
        self.base_seed = self.repro_config.get('base_seed', 42)
        
        # 结果存储
        self.run_results = []
        self.variance_results = {}
        
        logging.info(f"Reproducibility check config: {self.num_runs} runs, {self.num_epochs} epochs, tolerance={self.tolerance}")
    
    def create_model(self, seed: int) -> nn.Module:
        """创建模型"""
        set_seed(seed)
        model = SimpleModel()
        return model.to(self.device)
    
    def create_dataloader(self, seed: int) -> DataLoader:
        """创建数据加载器"""
        dataset = SyntheticDataset(num_samples=50, image_size=64, seed=seed)
        return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    def train_model(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        seed: int
    ) -> Dict[str, List[float]]:
        """训练模型并记录指标"""
        set_seed(seed)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        epoch_metrics = {
            'train_loss': [],
            'train_mse': [],
            'train_mae': [],
            'train_rel_l2': []
        }
        
        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_mae = 0.0
            epoch_rel_l2 = 0.0
            num_batches = 0
            
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # 计算指标
                with torch.no_grad():
                    metrics = compute_metrics(outputs, targets)
                    epoch_loss += loss.item()
                    epoch_mse += metrics['mse']
                    epoch_mae += metrics['mae']
                    epoch_rel_l2 += metrics['rel_l2']
                    num_batches += 1
            
            # 记录epoch平均指标
            epoch_metrics['train_loss'].append(epoch_loss / num_batches)
            epoch_metrics['train_mse'].append(epoch_mse / num_batches)
            epoch_metrics['train_mae'].append(epoch_mae / num_batches)
            epoch_metrics['train_rel_l2'].append(epoch_rel_l2 / num_batches)
        
        return epoch_metrics
    
    def evaluate_model(
        self, 
        model: nn.Module, 
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        total_rel_l2 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = model(inputs)
                metrics = compute_metrics(outputs, targets)
                
                total_mse += metrics['mse']
                total_mae += metrics['mae']
                total_rel_l2 += metrics['rel_l2']
                num_batches += 1
        
        return {
            'eval_mse': total_mse / num_batches,
            'eval_mae': total_mae / num_batches,
            'eval_rel_l2': total_rel_l2 / num_batches
        }
    
    def run_single_experiment(self, run_id: int) -> Dict[str, Any]:
        """运行单次实验"""
        logging.info(f"Running experiment {run_id + 1}/{self.num_runs}")
        
        # 使用相同的种子
        seed = self.base_seed
        set_seed(seed)
        
        # 记录确定性配置
        deterministic_config = get_deterministic_config()
        
        # 创建模型和数据
        model = self.create_model(seed)
        train_dataloader = self.create_dataloader(seed)
        eval_dataloader = self.create_dataloader(seed)  # 使用相同种子确保数据一致
        
        # 记录初始模型参数
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.data.clone().cpu().numpy()
        
        # 训练模型
        train_metrics = self.train_model(model, train_dataloader, seed)
        
        # 评估模型
        eval_metrics = self.evaluate_model(model, eval_dataloader)
        
        # 记录最终模型参数
        final_params = {}
        for name, param in model.named_parameters():
            final_params[name] = param.data.clone().cpu().numpy()
        
        result = {
            'run_id': run_id,
            'seed': seed,
            'deterministic_config': deterministic_config,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'initial_params_hash': self._compute_params_hash(initial_params),
            'final_params_hash': self._compute_params_hash(final_params),
            'environment': get_environment_info()
        }
        
        return result
    
    def _compute_params_hash(self, params: Dict[str, np.ndarray]) -> str:
        """计算参数哈希值"""
        import hashlib
        
        # 将所有参数连接成一个字符串
        param_str = ""
        for name in sorted(params.keys()):
            param_str += f"{name}:{params[name].tobytes().hex()}"
        
        # 计算MD5哈希
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def compute_variance_statistics(self) -> Dict[str, Any]:
        """计算方差统计量"""
        if len(self.run_results) < 2:
            return {'error': 'Need at least 2 runs for variance calculation'}
        
        # 收集所有指标
        metrics_collection = defaultdict(list)
        
        for result in self.run_results:
            # 训练指标（取最后一个epoch）
            train_metrics = result['train_metrics']
            for metric_name, values in train_metrics.items():
                if values:  # 确保有值
                    metrics_collection[f'final_{metric_name}'].append(values[-1])
            
            # 评估指标
            eval_metrics = result['eval_metrics']
            for metric_name, value in eval_metrics.items():
                metrics_collection[metric_name].append(value)
        
        # 计算统计量
        variance_stats = {}
        for metric_name, values in metrics_collection.items():
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)  # 样本标准差
                var_val = np.var(values, ddof=1)  # 样本方差
                min_val = np.min(values)
                max_val = np.max(values)
                
                # 检查是否满足容忍度
                passes_tolerance = bool(var_val <= self.tolerance)
                
                variance_stats[metric_name] = {
                    'values': [float(v) for v in values],
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'variance': float(var_val),
                    'min': float(min_val),
                    'max': float(max_val),
                    'range': float(max_val - min_val),
                    'passes_tolerance': passes_tolerance,
                    'tolerance': float(self.tolerance)
                }
        
        # 检查参数一致性
        param_hashes = {
            'initial': [r['initial_params_hash'] for r in self.run_results],
            'final': [r['final_params_hash'] for r in self.run_results]
        }
        
        param_consistency = {
            'initial_params_identical': bool(len(set(param_hashes['initial'])) == 1),
            'final_params_identical': bool(len(set(param_hashes['final'])) == 1),
            'initial_hashes': param_hashes['initial'],
            'final_hashes': param_hashes['final']
        }
        
        # 总体通过状态
        all_metrics_pass = all(
            stats['passes_tolerance'] 
            for stats in variance_stats.values()
        )
        
        overall_pass = bool(
            all_metrics_pass and 
            param_consistency['initial_params_identical']
        )
        
        return {
            'variance_statistics': variance_stats,
            'parameter_consistency': param_consistency,
            'overall_pass': overall_pass,
            'num_runs': int(len(self.run_results)),
            'tolerance': float(self.tolerance),
            'base_seed': int(self.base_seed)
        }
    
    def run_reproducibility_check(self, output_dir: Path) -> Dict[str, Any]:
        """运行完整的可复现性检查"""
        logging.info(f"Starting reproducibility check with {self.num_runs} runs...")
        
        self.run_results = []
        
        # 运行多次实验
        for run_id in range(self.num_runs):
            try:
                result = self.run_single_experiment(run_id)
                self.run_results.append(result)
                
                logging.info(f"Completed run {run_id + 1}/{self.num_runs}")
                
            except Exception as e:
                logging.error(f"Error in run {run_id}: {e}")
                continue
        
        # 计算方差统计
        variance_results = self.compute_variance_statistics()
        
        # 保存结果
        self.save_results(output_dir, variance_results)
        
        return variance_results
    
    def save_results(self, output_dir: Path, variance_results: Dict[str, Any]):
        """保存检查结果"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        detailed_results_path = output_dir / 'reproducibility_detailed.json'
        with open(detailed_results_path, 'w') as f:
            json.dump(self.run_results, f, indent=2)
        
        # 保存方差统计
        variance_path = output_dir / 'reproducibility_variance.json'
        with open(variance_path, 'w') as f:
            json.dump(variance_results, f, indent=2)
        
        # 生成报告
        self.generate_report(output_dir, variance_results)
        
        logging.info(f"Reproducibility check results saved to {output_dir}")
    
    def generate_report(self, output_dir: Path, variance_results: Dict[str, Any]):
        """生成可复现性检查报告"""
        report_path = output_dir / 'reproducibility_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# PDEBench可复现性检查报告\n\n")
            
            # 基本信息
            f.write("## 基本信息\n\n")
            f.write(f"- 实验运行次数: {variance_results['num_runs']}\n")
            f.write(f"- 基础随机种子: {variance_results['base_seed']}\n")
            f.write(f"- 方差容忍度: {variance_results['tolerance']:.2e}\n")
            f.write(f"- 训练轮数: {self.num_epochs}\n\n")
            
            # 总体结果
            f.write("## 总体结果\n\n")
            f.write(f"- **可复现性验证**: {'✅ 通过' if variance_results['overall_pass'] else '❌ 失败'}\n")
            
            # 参数一致性
            param_consistency = variance_results['parameter_consistency']
            f.write(f"- **初始参数一致性**: {'✅ 一致' if param_consistency['initial_params_identical'] else '❌ 不一致'}\n")
            f.write(f"- **最终参数一致性**: {'✅ 一致' if param_consistency['final_params_identical'] else '❌ 不一致'}\n\n")
            
            # 方差统计
            f.write("## 指标方差统计\n\n")
            f.write("| 指标 | 均值 | 标准差 | 方差 | 范围 | 通过容忍度 |\n")
            f.write("|------|------|--------|------|------|------------|\n")
            
            variance_stats = variance_results['variance_statistics']
            for metric_name, stats in variance_stats.items():
                pass_icon = "✅" if stats['passes_tolerance'] else "❌"
                f.write(f"| {metric_name} | {stats['mean']:.6f} | {stats['std']:.2e} | "
                       f"{stats['variance']:.2e} | {stats['range']:.2e} | {pass_icon} |\n")
            
            # 详细数值
            f.write("\n## 详细数值\n\n")
            for metric_name, stats in variance_stats.items():
                f.write(f"### {metric_name}\n\n")
                f.write(f"- **所有运行的值**: {stats['values']}\n")
                f.write(f"- **均值**: {stats['mean']:.6f}\n")
                f.write(f"- **标准差**: {stats['std']:.2e}\n")
                f.write(f"- **方差**: {stats['variance']:.2e}\n")
                f.write(f"- **最小值**: {stats['min']:.6f}\n")
                f.write(f"- **最大值**: {stats['max']:.6f}\n")
                f.write(f"- **范围**: {stats['range']:.2e}\n")
                f.write(f"- **通过容忍度**: {'是' if stats['passes_tolerance'] else '否'}\n\n")
            
            # 结论
            f.write("## 结论\n\n")
            if variance_results['overall_pass']:
                f.write("✅ **可复现性验证通过**\n\n")
                f.write("- 所有指标的方差均≤容忍度\n")
                f.write("- 模型初始化参数完全一致\n")
                f.write("- 系统满足黄金法则的可复现性要求\n")
            else:
                f.write("❌ **可复现性验证失败**\n\n")
                f.write("存在以下问题：\n")
                
                failed_metrics = [
                    name for name, stats in variance_stats.items() 
                    if not stats['passes_tolerance']
                ]
                
                if failed_metrics:
                    f.write(f"- 以下指标方差超过容忍度: {', '.join(failed_metrics)}\n")
                
                if not param_consistency['initial_params_identical']:
                    f.write("- 模型初始化参数不一致\n")
                
                f.write("\n请检查：\n")
                f.write("1. 随机种子设置是否正确\n")
                f.write("2. 确定性算法是否启用\n")
                f.write("3. 环境配置是否一致\n")
                f.write("4. 数据加载是否确定性\n")
        
        logging.info(f"Reproducibility report saved to {report_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="reproducibility_check")
def main(cfg: DictConfig) -> None:
    """主检查函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(cfg.get('output_dir', 'reproducibility_check_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建可复现性检查器
    checker = ReproducibilityChecker(cfg, device)
    
    # 运行检查
    logger.info("Starting reproducibility check...")
    results = checker.run_reproducibility_check(output_dir)
    
    # 打印结果
    logger.info("Reproducibility check completed!")
    logger.info(f"Overall pass: {'YES' if results['overall_pass'] else 'NO'}")
    logger.info(f"Number of runs: {results['num_runs']}")
    logger.info(f"Tolerance: {results['tolerance']:.2e}")
    
    # 统计通过的指标数量
    variance_stats = results['variance_statistics']
    passed_metrics = sum(1 for stats in variance_stats.values() if stats['passes_tolerance'])
    total_metrics = len(variance_stats)
    
    logger.info(f"Metrics passing tolerance: {passed_metrics}/{total_metrics}")
    
    if not results['overall_pass']:
        failed_metrics = [
            name for name, stats in variance_stats.items() 
            if not stats['passes_tolerance']
        ]
        logger.warning(f"Failed metrics: {failed_metrics}")
    
    logger.info(f"Detailed results saved to: {output_dir}")
    
    # 返回退出码
    return 0 if results['overall_pass'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)