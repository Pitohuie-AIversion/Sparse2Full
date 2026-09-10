#!/usr/bin/env python3
"""
批量训练脚本 - PDEBench稀疏观测重建系统
支持自动化训练所有已验证的模型

遵循黄金法则：
1. 一致性优先：所有模型使用相同的数据配置和训练参数
2. 可复现：固定随机种子，确保结果可重现
3. 统一接口：所有模型遵循统一的训练接口
4. 可比性：记录详细的性能指标和资源消耗
5. 文档先行：生成完整的训练报告和日志
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# 设置UTF-8编码，解决Windows下的编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    # Windows系统下设置控制台编码
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass  # 如果都失败，继续执行

import yaml
import torch
# import pandas as pd  # 移除pandas依赖

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

class BatchTrainer:
    """批量训练管理器"""
    
    def __init__(self, 
                 data_config: str = "pdebench",
                 base_epochs: int = 15,
                 base_batch_size: int = 2,
                 seeds: List[int] = [2025, 2026, 2027],
                 output_dir: str = "runs"):
        """
        初始化批量训练器
        
        Args:
            data_config: 数据配置名称
            base_epochs: 基础训练轮数
            base_batch_size: 基础批次大小
            seeds: 随机种子列表
            output_dir: 输出目录
        """
        self.data_config = data_config
        self.base_epochs = base_epochs
        self.base_batch_size = base_batch_size
        self.seeds = seeds
        self.output_dir = Path(output_dir)
        
        # 所有已验证的模型列表
        self.models = [
            "unet", "unet_plus_plus", "fno2d", "ufno_unet",
            "segformer_unetformer", "unetformer", "mlp", "mlp_mixer",
            "liif", "hybrid", "segformer", "swin_unet"
        ]
        
        # 模型特定配置
        self.model_configs = {
            "fno2d": {"batch_size": 4, "epochs": 20},
            "hybrid": {"batch_size": 2, "epochs": 18},
            "segformer": {"batch_size": 2, "epochs": 18},
            "segformer_unetformer": {"batch_size": 2, "epochs": 18},
            "swin_unet": {"batch_size": 2, "epochs": 18},
            "liif": {"batch_size": 4, "epochs": 20},
        }
        
        # 设置日志
        self.setup_logging()
        
        # 训练结果存储
        self.results = []
        
    def setup_logging(self):
        """设置日志系统"""
        log_dir = self.output_dir / "batch_training_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"batch_train_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_model_config(self, model_name: str) -> Dict:
        """获取模型特定配置"""
        config = self.model_configs.get(model_name, {})
        return {
            "batch_size": config.get("batch_size", self.base_batch_size),
            "epochs": config.get("epochs", self.base_epochs)
        }
        
    def generate_experiment_name(self, model_name: str, seed: int) -> str:
        """生成实验名称"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"SRx4-DarcyFlow-128-{model_name.upper()}-batch-s{seed}-{timestamp}"
        
    def train_single_model(self, model_name: str, seed: int) -> Dict:
        """训练单个模型"""
        self.logger.info(f"开始训练模型: {model_name}, 种子: {seed}")
        
        # 获取模型配置
        model_config = self.get_model_config(model_name)
        exp_name = self.generate_experiment_name(model_name, seed)
        
        # 构建训练命令 - 修复Hydra配置覆盖问题
        cmd = [
            sys.executable, "train.py",
            f"data={self.data_config}",  # 直接覆盖默认配置
            f"model={model_name}",  # 直接覆盖默认配置
            f"train.epochs={model_config['epochs']}",  # 使用train.epochs而不是trainer.max_epochs
            f"data.batch_size={model_config['batch_size']}",
            f"experiment.seed={seed}",  # 使用experiment.seed
            f"experiment.name={exp_name}",  # 使用experiment.name
        ]
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 执行训练
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            # 计算训练时间
            train_time = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"模型 {model_name} (种子 {seed}) 训练成功，耗时 {train_time:.2f}s")
                
                # 解析训练结果
                metrics = self.parse_training_results(exp_name)
                
                return {
                    "model": model_name,
                    "seed": seed,
                    "exp_name": exp_name,
                    "status": "success",
                    "train_time": train_time,
                    "epochs": model_config['epochs'],
                    "batch_size": model_config['batch_size'],
                    **metrics
                }
            else:
                self.logger.error(f"模型 {model_name} (种子 {seed}) 训练失败")
                self.logger.error(f"错误输出: {result.stderr}")
                
                return {
                    "model": model_name,
                    "seed": seed,
                    "exp_name": exp_name,
                    "status": "failed",
                    "train_time": train_time,
                    "error": result.stderr[:500]  # 截取前500字符
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"模型 {model_name} (种子 {seed}) 训练超时")
            return {
                "model": model_name,
                "seed": seed,
                "exp_name": exp_name,
                "status": "timeout",
                "train_time": 3600
            }
        except Exception as e:
            self.logger.error(f"模型 {model_name} (种子 {seed}) 训练异常: {str(e)}")
            return {
                "model": model_name,
                "seed": seed,
                "exp_name": exp_name,
                "status": "error",
                "error": str(e)
            }
            
    def parse_training_results(self, exp_name: str) -> Dict:
        """解析训练结果"""
        exp_dir = self.output_dir / exp_name
        
        # 默认指标
        metrics = {
            "rel_l2": None,
            "mae": None,
            "psnr": None,
            "ssim": None,
            "params": None,
            "flops": None,
            "memory": None
        }
        
        try:
            # 尝试读取指标文件
            metrics_file = exp_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    metrics.update(data)
                    
            # 尝试读取日志文件获取更多信息
            log_file = exp_dir / "train.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    
                # 解析参数量和FLOPs
                if "Total params:" in log_content:
                    import re
                    params_match = re.search(r"Total params: ([\d,]+)", log_content)
                    if params_match:
                        metrics["params"] = int(params_match.group(1).replace(',', ''))
                        
        except Exception as e:
            self.logger.warning(f"解析实验 {exp_name} 结果时出错: {str(e)}")
            
        return metrics
        
    def run_batch_training(self, models: Optional[List[str]] = None):
        """执行批量训练"""
        if models is None:
            models = self.models
            
        self.logger.info(f"开始批量训练，模型数量: {len(models)}, 种子数量: {len(self.seeds)}")
        self.logger.info(f"总训练任务数: {len(models) * len(self.seeds)}")
        
        total_start_time = time.time()
        
        for model_name in models:
            for seed in self.seeds:
                result = self.train_single_model(model_name, seed)
                self.results.append(result)
                
                # 保存中间结果
                self.save_intermediate_results()
                
        total_time = time.time() - total_start_time
        self.logger.info(f"批量训练完成，总耗时: {total_time:.2f}s")
        
        # 生成最终报告
        self.generate_final_report()
        
    def save_intermediate_results(self):
        """保存中间结果"""
        results_file = self.output_dir / "batch_training_logs" / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def generate_final_report(self):
        """生成最终报告"""
        self.logger.info("生成最终训练报告...")
        
        # 保存详细结果为JSON格式
        results_file = self.output_dir / "batch_training_logs" / "batch_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 生成汇总统计
        summary = self.generate_summary_statistics(self.results)
        
        # 保存汇总报告
        summary_file = self.output_dir / "batch_training_logs" / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # 生成Markdown报告
        self.generate_markdown_report(self.results, summary)
        
        self.logger.info(f"训练报告已保存到: {self.output_dir / 'batch_training_logs'}")
        
    def generate_summary_statistics(self, results: List[Dict]) -> Dict:
        """生成汇总统计"""
        total_experiments = len(results)
        successful_experiments = len([r for r in results if r['status'] == 'success'])
        failed_experiments = len([r for r in results if r['status'] == 'failed'])
        timeout_experiments = len([r for r in results if r['status'] == 'timeout'])
        
        summary = {
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "failed_experiments": failed_experiments,
            "timeout_experiments": timeout_experiments,
            "success_rate": successful_experiments / total_experiments * 100 if total_experiments > 0 else 0,
            "total_training_time": sum(r.get('train_time', 0) for r in results),
            "average_training_time": sum(r.get('train_time', 0) for r in results) / total_experiments if total_experiments > 0 else 0,
            "models_trained": len(set(r['model'] for r in results)),
            "seeds_used": len(set(r['seed'] for r in results))
        }
        
        # 按模型统计
        model_stats = {}
        models = set(r['model'] for r in results)
        
        for model in models:
            model_results = [r for r in results if r['model'] == model]
            successful_model_results = [r for r in model_results if r['status'] == 'success']
            
            model_stats[model] = {
                "total_runs": len(model_results),
                "successful_runs": len(successful_model_results),
                "success_rate": len(successful_model_results) / len(model_results) * 100 if len(model_results) > 0 else 0,
                "avg_train_time": sum(r.get('train_time', 0) for r in model_results) / len(model_results) if len(model_results) > 0 else 0,
                "avg_rel_l2": sum(r.get('rel_l2', 0) for r in successful_model_results if r.get('rel_l2') is not None) / len(successful_model_results) if len(successful_model_results) > 0 else None,
                "avg_psnr": sum(r.get('psnr', 0) for r in successful_model_results if r.get('psnr') is not None) / len(successful_model_results) if len(successful_model_results) > 0 else None
            }
            
        summary["model_statistics"] = model_stats
        
        return summary
        
    def generate_markdown_report(self, results: List[Dict], summary: Dict):
        """生成Markdown格式的报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# PDEBench批量训练报告

## 基本信息

- **生成时间**: {timestamp}
- **数据配置**: {self.data_config}
- **训练轮数**: {self.base_epochs}
- **随机种子**: {self.seeds}

## 训练概览

- **总实验数**: {summary['total_experiments']}
- **成功实验数**: {summary['successful_experiments']}
- **失败实验数**: {summary['failed_experiments']}
- **超时实验数**: {summary['timeout_experiments']}
- **成功率**: {summary['success_rate']:.1f}%
- **总训练时间**: {summary['total_training_time']:.2f}s ({summary['total_training_time']/3600:.2f}h)
- **平均训练时间**: {summary['average_training_time']:.2f}s

## 模型性能统计

| 模型 | 运行次数 | 成功次数 | 成功率 | 平均训练时间(s) | 平均Rel-L2 | 平均PSNR |
|------|----------|----------|--------|----------------|------------|----------|
"""
        
        for model, stats in summary['model_statistics'].items():
            rel_l2 = f"{stats['avg_rel_l2']:.4f}" if stats['avg_rel_l2'] else "N/A"
            psnr = f"{stats['avg_psnr']:.2f}" if stats['avg_psnr'] else "N/A"
            
            report += f"| {model} | {stats['total_runs']} | {stats['successful_runs']} | {stats['success_rate']:.1f}% | {stats['avg_train_time']:.2f} | {rel_l2} | {psnr} |\n"
            
        report += f"""

## 详细结果

详细的训练结果请查看: `batch_training_results.json`

## 失败分析

"""
        
        # 添加失败分析
        failed_results = [r for r in results if r['status'] != 'success']
        if len(failed_results) > 0:
            report += "### 失败实验列表\n\n"
            for result in failed_results:
                report += f"- **{result['model']}** (种子 {result['seed']}): {result['status']}\n"
                if 'error' in result and result['error']:
                    report += f"  - 错误信息: {result['error'][:100]}...\n"
        else:
            report += "🎉 所有实验都成功完成！\n"
            
        # 保存报告
        report_file = self.output_dir / "batch_training_logs" / "training_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDEBench批量训练脚本")
    parser.add_argument("--data", default="pdebench", help="数据配置名称")
    parser.add_argument("--epochs", type=int, default=15, help="基础训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="基础批次大小")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027], help="随机种子列表")
    parser.add_argument("--models", nargs="+", help="指定要训练的模型列表")
    parser.add_argument("--output-dir", default="runs", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建批量训练器
    trainer = BatchTrainer(
        data_config=args.data,
        base_epochs=args.epochs,
        base_batch_size=args.batch_size,
        seeds=args.seeds,
        output_dir=args.output_dir
    )
    
    # 执行批量训练
    trainer.run_batch_training(models=args.models)


if __name__ == "__main__":
    main()