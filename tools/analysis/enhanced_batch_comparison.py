#!/usr/bin/env python3
"""
Enhanced Batch Comparison Tool with Mixed Channel Support
支持不同输入通道模型的批量对比实验
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.training.model_loader import list_models
from tools.training.model_loader_improved import list_improved_models
from tools.training.model_loader_enhanced import list_enhanced_models


class EnhancedBatchComparison:
    """增强版批量对比工具，支持混合通道配置"""
    
    def __init__(self, config_path: Path, output_dir: Path, seeds: List[int]):
        """
        初始化批量对比工具
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            seeds: 随机种子列表
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.seeds = seeds
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        
        # 加载配置
        self.base_config = self._load_config()
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("EnhancedBatchComparison")
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(self.output_dir / "enhanced_batch_comparison.log")
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """加载基础配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _get_model_channel_config(self, model_name: str) -> int:
        """获取模型所需的输入通道数"""
        # 从配置中获取模型特定通道配置
        model_channel_configs = self.base_config.get('comparison', {}).get('model_channel_configs', {})
        
        # 返回模型特定配置，默认使用基础配置的in_channels
        return model_channel_configs.get(model_name, self.base_config.get('model', {}).get('in_channels', 4))
    
    def _create_model_specific_config(self, model_name: str, seed: int) -> Path:
        """
        为特定模型创建配置文件
        
        Args:
            model_name: 模型名称
            seed: 随机种子
            
        Returns:
            配置文件路径
        """
        # 获取模型所需的通道数
        required_channels = self._get_model_channel_config(model_name)
        
        # 创建模型特定配置
        model_config = self.base_config.copy()
        
        # 更新模型配置
        if 'model' not in model_config:
            model_config['model'] = {}
        
        model_config['model']['in_channels'] = required_channels
        
        # 更新实验名称
        if 'experiment_name' not in model_config:
            model_config['experiment_name'] = f"{model_name}_s{seed}"
        
        # 保存到临时文件
        config_file = self.output_dir / f"temp_config_{model_name}_s{seed}.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"为模型 {model_name} 创建配置文件 (输入通道: {required_channels}): {config_file}")
        
        return config_file
    
    def run_single_experiment(self, model_name: str, seed: int) -> Dict[str, Any]:
        """
        运行单个实验
        
        Args:
            model_name: 模型名称
            seed: 随机种子
            
        Returns:
            实验结果
        """
        experiment_name = f"{model_name}_s{seed}"
        self.logger.info(f"\n实验进度: [{self.current_experiment}/{self.total_experiments}] 模型: {model_name}, 种子: {seed}")
        self.logger.info(f"开始实验: {experiment_name}")
        
        start_time = time.time()
        
        try:
            # 为特定模型创建配置文件
            model_config_file = self._create_model_specific_config(model_name, seed)
            
            # 构建命令
            cmd = [
                sys.executable, "tools/training/train_real_data_ar.py",
                "--config", str(model_config_file),
                "--model", model_name,
                "--seeds", str(seed)
            ]
            
            self.logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 运行训练
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            elapsed_time = time.time() - start_time
            
            # 检查运行结果
            success = result.returncode == 0
            
            # 查找实验目录
            runs_dir = project_root / "runs"
            exp_dirs = list(runs_dir.glob(f"*{model_name}*s{seed}*"))
            exp_dir = exp_dirs[0] if exp_dirs else None
            
            experiment_info = {
                'model': model_name,
                'seed': seed,
                'success': success,
                'return_code': result.returncode,
                'elapsed_time': elapsed_time,
                'experiment_dir': str(exp_dir) if exp_dir else None,
                'input_channels': self._get_model_channel_config(model_name),
                'stdout': result.stdout[-2000:],  # 保存最后2000字符
                'stderr': result.stderr[-1000:],  # 保存最后1000字符
                'timestamp': datetime.now().isoformat(),
                'config_file': str(model_config_file)
            }
            
            if success:
                self.logger.info(f"✅ 实验 {experiment_name} 成功完成 (用时: {elapsed_time:.1f}s)")
                self.successful_experiments += 1
            else:
                self.logger.error(f"❌ 实验 {experiment_name} 失败 (返回码: {result.returncode})")
                self.logger.error(f"错误信息: {result.stderr[-500:]}")
                self.failed_experiments += 1
            
            # 清理临时配置文件
            try:
                model_config_file.unlink()
            except:
                pass
            
            return experiment_info
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"实验 {experiment_name} 异常失败: {e}")
            self.failed_experiments += 1
            
            return {
                'model': model_name,
                'seed': seed,
                'success': False,
                'return_code': -1,
                'elapsed_time': elapsed_time,
                'experiment_dir': None,
                'input_channels': self._get_model_channel_config(model_name),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_experiments(self, models: List[str]) -> Dict[str, Any]:
        """
        运行所有实验
        
        Args:
            models: 模型列表
            
        Returns:
            所有实验结果
        """
        self.total_experiments = len(models) * len(self.seeds)
        self.current_experiment = 0
        self.successful_experiments = 0
        self.failed_experiments = 0
        
        self.logger.info(f"开始增强批量对比实验: {len(models)} 个模型 × {len(self.seeds)} 个种子 = {self.total_experiments} 个实验")
        self.logger.info(f"使用配置文件: {self.config_path}")
        self.logger.info(f"输出目录: {self.output_dir}")
        
        all_results = {
            'config': {
                'models': models,
                'seeds': self.seeds,
                'total_experiments': self.total_experiments,
                'config_file': str(self.config_path),
                'start_time': datetime.now().isoformat(),
                'base_config': self.base_config
            },
            'experiments': [],
            'summary': {
                'total': self.total_experiments,
                'successful': 0,
                'failed': 0,
                'total_time': 0,
                'models_tested': len(models),
                'channel_configurations': {}
            }
        }
        
        # 按模型分组运行实验
        for model_idx, model_name in enumerate(models, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"模型进度: [{model_idx}/{len(models)}] {model_name}")
            self.logger.info(f"{'='*60}")
            
            # 获取模型通道配置
            required_channels = self._get_model_channel_config(model_name)
            self.logger.info(f"模型 {model_name} 使用输入通道: {required_channels}")
            
            # 记录通道配置统计
            if required_channels not in all_results['summary']['channel_configurations']:
                all_results['summary']['channel_configurations'][required_channels] = []
            all_results['summary']['channel_configurations'][required_channels].append(model_name)
            
            for seed in self.seeds:
                self.current_experiment += 1
                result = self.run_single_experiment(model_name, seed)
                all_results['experiments'].append(result)
                
                # 保存中间结果
                self._save_intermediate_results(all_results)
        
        # 更新最终统计
        all_results['summary']['successful'] = self.successful_experiments
        all_results['summary']['failed'] = self.failed_experiments
        all_results['summary']['end_time'] = datetime.now().isoformat()
        
        # 保存最终结果
        self._save_final_results(all_results)
        
        return all_results
    
    def _save_intermediate_results(self, results: Dict[str, Any]):
        """保存中间结果"""
        try:
            with open(self.output_dir / "enhanced_batch_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"保存中间结果失败: {e}")
    
    def _save_final_results(self, results: Dict[str, Any]):
        """保存最终结果"""
        try:
            # 保存详细结果
            with open(self.output_dir / "enhanced_batch_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 保存摘要
            summary = {
                'total_experiments': results['summary']['total'],
                'successful': results['summary']['successful'],
                'failed': results['summary']['failed'],
                'success_rate': results['summary']['successful'] / results['summary']['total'] if results['summary']['total'] > 0 else 0,
                'models_tested': results['summary']['models_tested'],
                'channel_configurations': results['summary']['channel_configurations'],
                'start_time': results['config']['start_time'],
                'end_time': results['summary']['end_time']
            }
            
            with open(self.output_dir / "experiment_summary.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info("实验完成摘要:")
            self.logger.info(f"总实验数: {summary['total_experiments']}")
            self.logger.info(f"成功: {summary['successful']} ({summary['success_rate']:.1%})")
            self.logger.info(f"失败: {summary['failed']}")
            self.logger.info(f"通道配置: {summary['channel_configurations']}")
            self.logger.info(f"结果保存至: {self.output_dir}")
            self.logger.info(f"{'='*60}")
            
        except Exception as e:
            self.logger.error(f"保存最终结果失败: {e}")


def get_available_models() -> List[str]:
    """获取所有可用模型"""
    try:
        # 合并所有加载器的模型
        all_models = set()
        
        # 原始加载器
        try:
            original_models = list_models()
            all_models.update(original_models)
        except:
            pass
        
        # 改进加载器
        try:
            improved_models = list_improved_models()
            all_models.update(improved_models)
        except:
            pass
        
        # 增强加载器
        try:
            enhanced_models = list_enhanced_models()
            all_models.update(enhanced_models)
        except:
            pass
        
        # 返回排序后的列表
        return sorted(list(all_models))
        
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        # 返回常用模型作为备选
        return ['swin_unet', 'unet', 'fno2d', 'segformer', 'hybrid', 'vit', 'mlp_mixer']


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强版批量模型对比工具")
    parser.add_argument("--config", type=Path, required=True, help="配置文件路径")
    parser.add_argument("--models", nargs="+", required=True, help="模型列表")
    parser.add_argument("--seeds", type=lambda x: [int(s) for s in x.split(",")], 
                       default=[42, 123, 456], help="随机种子，逗号分隔")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    
    args = parser.parse_args()
    
    # 创建批量对比工具
    comparator = EnhancedBatchComparison(
        config_path=args.config,
        output_dir=args.output,
        seeds=args.seeds
    )
    
    # 运行实验
    results = comparator.run_all_experiments(args.models)
    
    print(f"\n实验完成!")
    print(f"总实验数: {results['summary']['total']}")
    print(f"成功: {results['summary']['successful']}")
    print(f"失败: {results['summary']['failed']}")
    print(f"结果保存至: {args.output}")


if __name__ == "__main__":
    import time
    main()