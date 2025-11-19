#!/usr/bin/env python3
"""
批量模型横向对比训练脚本

支持多模型、多种子的自动化横向对比实验，生成论文级别的对比报告。

使用方法:
    python tools/batch_comparison.py --config configs/train/ar_training_config\ debug.yaml --models swin_unet unet fno2d --seeds 42,123,456 --output paper_package/comparison/
    python tools/batch_comparison.py --config configs/train/ar_training_config\ debug.yaml --all-models --seeds 5 --output paper_package/full_comparison/
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 延迟导入，避免循环依赖问题
def get_available_models():
    """获取可用模型列表"""
    try:
        from tools.training.model_loader import list_models
        return list_models()
    except ImportError:
        # 如果导入失败，返回默认模型列表
        return ['swin_unet', 'unet', 'fno2d', 'segformer', 'unet_plus_plus', 'hybrid']


class BatchComparison:
    """批量横向对比实验管理器"""
    
    def __init__(self, config_path: str, output_dir: str, seeds: List[int]):
        """
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            seeds: 随机种子列表
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.seeds = seeds
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # 实验记录
        self.experiments = []
        self.results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('BatchComparison')
        logger.setLevel(logging.INFO)
        
        # 创建处理器
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.output_dir / 'batch_comparison.log')
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def get_available_models(self) -> List[str]:
        """获取所有可用模型"""
        try:
            available_models = get_available_models()
            self.logger.info(f"发现 {len(available_models)} 个可用模型: {available_models}")
            return available_models
        except Exception as e:
            self.logger.error(f"获取模型列表失败: {e}")
            # 返回常用模型作为备选
            return ['swin_unet', 'unet', 'fno2d', 'segformer', 'unet_plus_plus']
    
    def run_single_experiment(self, model_name: str, seed: int) -> Dict:
        """
        运行单个实验
        
        Args:
            model_name: 模型名称
            seed: 随机种子
            
        Returns:
            实验结果信息
        """
        experiment_name = f"{model_name}_s{seed}"
        self.logger.info(f"开始实验: {experiment_name}")
        
        start_time = time.time()
        
        try:
            # 构建命令
            cmd = [
                sys.executable, "tools/training/train_real_data_ar.py",
                "--config", str(self.config_path),
                "--model", model_name,
                "--seeds", str(seed)
            ]
            
            self.logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 运行训练
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                check=False
            )
            
            elapsed_time = time.time() - start_time
            
            # 检查运行结果
            success = result.returncode == 0
            
            # 查找实验目录
            runs_dir = Path(__file__).parent.parent / "runs"
            exp_dirs = list(runs_dir.glob(f"*{model_name}*s{seed}*"))
            exp_dir = exp_dirs[0] if exp_dirs else None
            
            experiment_info = {
                'model': model_name,
                'seed': seed,
                'success': success,
                'return_code': result.returncode,
                'elapsed_time': elapsed_time,
                'experiment_dir': str(exp_dir) if exp_dir else None,
                'stdout': result.stdout[-2000:],  # 保存最后2000字符
                'stderr': result.stderr[-1000:],  # 保存最后1000字符
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                self.logger.info(f"✅ 实验 {experiment_name} 成功完成 (用时: {elapsed_time:.1f}s)")
            else:
                self.logger.error(f"❌ 实验 {experiment_name} 失败 (返回码: {result.returncode})")
                self.logger.error(f"错误信息: {result.stderr[-500:]}")
            
            return experiment_info
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"实验 {experiment_name} 异常失败: {e}")
            
            return {
                'model': model_name,
                'seed': seed,
                'success': False,
                'return_code': -1,
                'elapsed_time': elapsed_time,
                'experiment_dir': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_experiments(self, models: List[str]) -> Dict:
        """
        运行所有实验
        
        Args:
            models: 模型列表
            
        Returns:
            所有实验结果
        """
        total_experiments = len(models) * len(self.seeds)
        self.logger.info(f"开始批量对比实验: {len(models)} 个模型 × {len(self.seeds)} 个种子 = {total_experiments} 个实验")
        
        all_results = {
            'config': {
                'models': models,
                'seeds': self.seeds,
                'total_experiments': total_experiments,
                'config_file': str(self.config_path),
                'start_time': datetime.now().isoformat()
            },
            'experiments': [],
            'summary': {
                'total': total_experiments,
                'successful': 0,
                'failed': 0,
                'total_time': 0
            }
        }
        
        start_time = time.time()
        
        # 顺序运行实验（避免资源冲突）
        for i, model in enumerate(models, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"模型进度: [{i}/{len(models)}] {model}")
            self.logger.info(f"{'='*60}")
            
            for j, seed in enumerate(self.seeds, 1):
                experiment_num = (i-1) * len(self.seeds) + j
                self.logger.info(f"\n实验进度: [{experiment_num}/{total_experiments}] 模型: {model}, 种子: {seed}")
                
                # 运行单个实验
                result = self.run_single_experiment(model, seed)
                all_results['experiments'].append(result)
                
                # 更新统计
                if result['success']:
                    all_results['summary']['successful'] += 1
                else:
                    all_results['summary']['failed'] += 1
                
                # 保存中间结果
                self._save_intermediate_results(all_results)
                
                # 短暂休息，避免系统过载
                time.sleep(2)
        
        # 最终统计
        total_time = time.time() - start_time
        all_results['summary']['total_time'] = total_time
        all_results['config']['end_time'] = datetime.now().isoformat()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"批量实验完成!")
        self.logger.info(f"总计: {all_results['summary']['total']} 个实验")
        self.logger.info(f"成功: {all_results['summary']['successful']} 个")
        self.logger.info(f"失败: {all_results['summary']['failed']} 个")
        self.logger.info(f"总用时: {total_time/60:.1f} 分钟")
        self.logger.info(f"{'='*60}")
        
        return all_results
    
    def _save_intermediate_results(self, results: Dict):
        """保存中间结果"""
        try:
            with open(self.output_dir / 'batch_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"保存中间结果失败: {e}")
    
    def generate_comparison_report(self, results: Dict):
        """生成对比报告"""
        self.logger.info("生成对比报告...")
        
        try:
            # 运行汇总脚本
            cmd = [
                sys.executable, "tools/summarize_runs.py",
                "--runs_dir", "runs/",
                "--output", str(self.output_dir / "comparison_results"),
                "--baseline_method", "unet"  # 默认以UNet为基线
            ]
            
            self.logger.info(f"执行汇总命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                self.logger.info("✅ 对比报告生成成功")
                self.logger.info(f"报告位置: {self.output_dir}/comparison_results/")
            else:
                self.logger.error(f"❌ 对比报告生成失败: {result.stderr}")
            
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"生成对比报告异常: {e}")
            return False
    
    def run(self, models: List[str]) -> bool:
        """运行完整的批量对比实验"""
        try:
            # 运行所有实验
            results = self.run_all_experiments(models)
            
            # 保存最终结果
            self._save_intermediate_results(results)
            
            # 生成对比报告
            if results['summary']['successful'] > 0:
                self.generate_comparison_report(results)
            else:
                self.logger.warning("没有成功的实验，跳过报告生成")
            
            return results['summary']['successful'] > 0
            
        except Exception as e:
            self.logger.error(f"批量对比实验失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量模型横向对比训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 对比指定模型
    python tools/batch_comparison.py --config configs/train/ar_training_config\\ debug.yaml --models swin_unet unet fno2d --seeds 42,123,456
    
    # 对比所有可用模型，每个模型5个种子
    python tools/batch_comparison.py --config configs/train/ar_training_config\\ debug.yaml --all-models --seeds 5
    
    # 快速对比3个模型，2个种子
    python tools/batch_comparison.py --config configs/train/ar_training_config\\ debug.yaml --models swin_unet unet --seeds 42,123 --output results/quick_comparison/
        """
    )
    
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="要对比的模型列表 (如: swin_unet unet fno2d)")
    parser.add_argument("--all-models", action="store_true",
                       help="使用所有可用模型")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                       help="随机种子，可以是数字列表或数字个数 (如: '42,123,456' 或 '5')")
    parser.add_argument("--output", type=str, default="paper_package/batch_comparison/",
                       help="输出目录")
    parser.add_argument("--list-models", action="store_true",
                       help="列出所有可用模型并退出")
    
    args = parser.parse_args()
    
    # 如果请求列出模型
    if args.list_models:
        print("\n可用模型:")
        available_models = get_available_models()
        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}. {model}")
        print(f"\n总计: {len(available_models)} 个模型\n")
        return
    
    # 解析种子
    try:
        if args.seeds.isdigit():
            # 如果是一个数字，生成连续的多个种子
            num_seeds = int(args.seeds)
            seeds = list(range(42, 42 + num_seeds))
        else:
            # 解析逗号分隔的种子列表
            seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    except Exception as e:
        print(f"❌ 种子解析失败: {e}")
        print("请使用格式如: '42,123,456' 或 '5'")
        return
    
    if not seeds:
        print("❌ 没有有效的种子")
        return
    
    # 获取模型列表
    if args.all_models:
        # 获取所有可用模型
        batch_comparison = BatchComparison(args.config, args.output, seeds)
        models = batch_comparison.get_available_models()
        print(f"将使用所有 {len(models)} 个可用模型")
    elif args.models:
        models = args.models
        print(f"将使用指定的 {len(models)} 个模型: {models}")
    else:
        print("❌ 必须指定 --models 或 --all-models")
        return
    
    print(f"\n🚀 开始批量对比实验")
    print(f"配置文件: {args.config}")
    print(f"模型数量: {len(models)}")
    print(f"种子数量: {len(seeds)}: {seeds}")
    print(f"总实验数: {len(models) * len(seeds)}")
    print(f"输出目录: {args.output}")
    print("="*60 + "\n")
    
    # 运行批量对比
    batch_comparison = BatchComparison(args.config, args.output, seeds)
    success = batch_comparison.run(models)
    
    if success:
        print(f"\n✅ 批量对比实验完成！")
        print(f"📊 结果保存在: {args.output}")
    else:
        print(f"\n❌ 批量对比实验失败，请查看日志: {args.output}/batch_comparison.log")


if __name__ == "__main__":
    main()