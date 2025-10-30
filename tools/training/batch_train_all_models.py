#!/usr/bin/env python3
"""
批量训练所有模型脚本
基于 train.yaml 配置文件运行所有12个可用模型的完整训练和评估

作者: PDEBench稀疏观测重建系统
日期: 2025-01-13
"""

import os
import sys
import yaml
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchModelTrainer:
    """批量模型训练器"""
    
    def __init__(self, base_config_path: str = "configs/train.yaml"):
        """初始化批量训练器
        
        Args:
            base_config_path: 基础配置文件路径
        """
        self.base_config_path = Path(base_config_path)
        self.project_root = Path(__file__).parent
        self.model_configs_dir = self.project_root / "configs" / "model"
        self.results_dir = self.project_root / "runs" / "batch_training_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载基础配置
        self.base_config = self.load_base_config()
        
        # 定义所有可用模型
        self.available_models = [
            "unet",
            "unet_plus_plus", 
            "fno2d",
            "ufno_unet",
            "segformer",
            "unetformer",
            "segformer_unetformer",
            "mlp",
            "mlp_mixer",
            "liif",
            "swin_unet",
            "hybrid",
            "transformer"  # 新增经典Transformer模型
        ]
        
        # 训练结果存储
        self.training_results = {}
        self.resource_stats = {}
        
        # 设置logger
        self.logger = logger
        
    def load_base_config(self) -> Dict[str, Any]:
        """加载基础配置文件"""
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载基础配置: {self.base_config_path}")
            return config
        except Exception as e:
            logger.error(f"加载基础配置失败: {e}")
            raise
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """加载模型特定配置"""
        model_config_path = self.model_configs_dir / f"{model_name}.yaml"
        
        if not model_config_path.exists():
            logger.warning(f"模型配置文件不存在: {model_config_path}")
            return {}
            
        try:
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
            return model_config
        except Exception as e:
            logger.error(f"加载模型配置失败 {model_name}: {e}")
            return {}
    
    def create_training_config(self, model_name: str) -> Dict[str, Any]:
        """为特定模型创建训练配置"""
        # 复制基础配置
        config = self.base_config.copy()
        
        # 加载模型特定配置
        model_config = self.load_model_config(model_name)
        
        # 更新模型配置
        if model_config:
            config['model'] = model_config
        else:
            # 使用默认模型配置
            config['model'] = {
                'name': model_name,
                'params': {
                    'in_channels': 1,
                    'out_channels': 1,
                    'img_size': 128
                }
            }
        
        # 确保使用标准化训练参数（基于train.yaml）
        config['training']['epochs'] = 200
        config['training']['seed'] = 2025
        config['training']['optimizer']['params']['lr'] = 1.0e-3
        config['training']['optimizer']['params']['weight_decay'] = 1.0e-4
        config['training']['scheduler']['params']['warmup_steps'] = 1000
        
        # 确保损失函数三件套标准配置
        config['loss']['rec_weight'] = 1.0
        config['loss']['spec_weight'] = 0.5
        config['loss']['dc_weight'] = 1.0
        config['loss']['low_freq_modes'] = 16
        config['loss']['mirror_padding'] = True
        
        # 为FNO相关模型禁用AMP
        if model_name in ['fno2d', 'hybrid', 'ufno_unet']:
            config['training']['use_amp'] = False
            self.logger.info(f"Disabled AMP for {model_name} (complex operations compatibility)")
        
        # 修复UNetFormer的num_heads配置问题
        if model_name == 'unetformer' and 'model' in config and 'params' in config['model']:
            if 'num_heads' in config['model']['params']:
                # 确保num_heads是整数而不是ListConfig
                num_heads = config['model']['params']['num_heads']
                if isinstance(num_heads, (list, tuple)):
                    config['model']['params']['num_heads'] = num_heads[0] if num_heads else 8
                elif not isinstance(num_heads, int):
                    config['model']['params']['num_heads'] = 8
                self.logger.info(f"Fixed num_heads config for {model_name}: {config['model']['params']['num_heads']}")
        
        # 更新实验名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment']['name'] = f"batch_{model_name}_sr_x4_{timestamp}"
        
        return config
    
    def save_training_config(self, model_name: str, config: Dict[str, Any]) -> Path:
        """保存训练配置到文件"""
        config_path = self.results_dir / f"config_{model_name}.yaml"
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"保存训练配置: {config_path}")
            return config_path
        except Exception as e:
            self.logger.error(f"保存配置失败 {model_name}: {e}")
            raise
    
    def run_single_model_training(self, model_name: str) -> Dict[str, Any]:
        """运行单个模型的训练"""
        self.logger.info(f"开始训练模型: {model_name}")
        start_time = time.time()
        
        try:
            # 创建模型特定的输出目录
            model_output_dir = self.results_dir / model_name
            model_output_dir.mkdir(exist_ok=True)
            
            # 检查是否应该跳过
            if self.should_skip_model(model_name, model_output_dir):
                return {
                    'model_name': model_name,
                    'status': 'skipped',
                    'training_time': 0
                }
            
            # 创建训练配置
            config = self.create_training_config(model_name)
            config_path = self.save_training_config(model_name, config)
            
            # 构建训练命令 - 使用正确的Hydra格式
            train_script = self.project_root / "train.py"
            cmd = [
                sys.executable, str(train_script),
                f"--config-path={config_path.parent}",
                f"--config-name={config_path.name}",
                f"hydra.run.dir={model_output_dir}",
                f"experiment.output_dir={model_output_dir}"
            ]
            
            # 执行训练
            self.logger.info(f"执行训练命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            training_time = time.time() - start_time
            
            # 检查训练结果
            if result.returncode == 0:
                logger.info(f"模型 {model_name} 训练成功，耗时: {training_time:.2f}秒")
                
                # 创建成功标志文件
                success_file = model_output_dir / "training_completed.txt"
                with open(success_file, 'w') as f:
                    f.write(f"Training completed at {datetime.now()}\n")
                    f.write(f"Training time: {training_time:.2f} seconds\n")
                
                # 解析训练结果
                training_result = self.parse_training_result(model_name, model_output_dir)
                training_result['training_time'] = training_time
                training_result['status'] = 'success'
                
                return training_result
            else:
                logger.error(f"模型 {model_name} 训练失败:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                
                # 保存错误信息到文件
                error_file = model_output_dir / "training_error.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Training failed at {datetime.now()}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"STDOUT:\n{result.stdout}\n")
                    f.write(f"STDERR:\n{result.stderr}\n")
                
                return {
                    'model_name': model_name,
                    'status': 'failed',
                    'error': result.stderr,
                    'training_time': training_time
                }
                
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"模型 {model_name} 训练异常: {e}")
            return {
                'model_name': model_name,
                'status': 'error',
                'error': str(e),
                'training_time': training_time
            }
    
    def should_skip_model(self, model_name: str, output_dir: Path) -> bool:
        """检查是否应该跳过已完成的模型训练"""
        if not output_dir.exists():
            return False
        
        # 检查是否存在成功完成的标志
        success_file = output_dir / "training_completed.txt"
        if success_file.exists():
            logger.info(f"模型 {model_name} 已完成训练，跳过")
            return True
        
        # 检查是否存在最佳检查点
        best_checkpoint = output_dir / "best.pth"
        if best_checkpoint.exists():
            logger.info(f"模型 {model_name} 存在最佳检查点，跳过")
            return True
        
        return False
    
    def parse_training_result(self, model_name: str, output_dir: Path) -> Dict[str, Any]:
        """解析训练结果"""
        result = {'model_name': model_name}
        
        try:
            # 查找训练日志文件
            log_files = list(output_dir.glob("*.log"))
            if log_files:
                log_file = log_files[0]
                result['log_file'] = str(log_file)
                
                # 解析日志中的指标
                metrics = self.parse_metrics_from_log(log_file)
                result.update(metrics)
            
            # 查找检查点文件
            checkpoint_files = list(output_dir.glob("*.pth"))
            if checkpoint_files:
                result['checkpoint'] = str(checkpoint_files[0])
            
            # 查找配置文件
            config_files = list(output_dir.glob("config_*.yaml"))
            if config_files:
                result['config_file'] = str(config_files[0])
                
        except Exception as e:
            logger.warning(f"解析训练结果失败 {model_name}: {e}")
            
        return result
    
    def parse_metrics_from_log(self, log_file: Path) -> Dict[str, float]:
        """从日志文件解析指标"""
        metrics = {}
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 查找最终验证指标
            for line in reversed(lines):
                if 'Final validation metrics' in line or 'Best validation' in line:
                    # 解析指标行
                    if 'rel_l2' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'rel_l2:' and i + 1 < len(parts):
                                metrics['rel_l2'] = float(parts[i + 1].rstrip(','))
                            elif part == 'mae:' and i + 1 < len(parts):
                                metrics['mae'] = float(parts[i + 1].rstrip(','))
                            elif part == 'psnr:' and i + 1 < len(parts):
                                metrics['psnr'] = float(parts[i + 1].rstrip(','))
                            elif part == 'ssim:' and i + 1 < len(parts):
                                metrics['ssim'] = float(parts[i + 1].rstrip(','))
                    break
                    
        except Exception as e:
            logger.warning(f"解析指标失败: {e}")
            
        return metrics
    
    def run_batch_training(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """运行批量训练"""
        if models is None:
            models = self.available_models
        
        logger.info(f"开始批量训练 {len(models)} 个模型: {models}")
        batch_start_time = time.time()
        
        results = {}
        
        for i, model_name in enumerate(models, 1):
            logger.info(f"进度: {i}/{len(models)} - 训练模型: {model_name}")
            
            try:
                result = self.run_single_model_training(model_name)
                results[model_name] = result
                
                # 保存中间结果
                self.save_intermediate_results(results)
                
            except Exception as e:
                logger.error(f"模型 {model_name} 训练失败: {e}")
                results[model_name] = {
                    'model_name': model_name,
                    'status': 'error',
                    'error': str(e)
                }
        
        batch_time = time.time() - batch_start_time
        logger.info(f"批量训练完成，总耗时: {batch_time:.2f}秒")
        
        # 保存最终结果
        self.training_results = results
        self.save_final_results(results, batch_time)
        
        return results
    
    def save_intermediate_results(self, results: Dict[str, Any]):
        """保存中间结果"""
        results_file = self.results_dir / "intermediate_results.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"保存中间结果失败: {e}")
    
    def save_final_results(self, results: Dict[str, Any], batch_time: float):
        """保存最终结果"""
        # 保存JSON格式结果
        results_file = self.results_dir / "final_results.json"
        final_results = {
            'batch_training_time': batch_time,
            'timestamp': datetime.now().isoformat(),
            'total_models': len(results),
            'successful_models': len([r for r in results.values() if r.get('status') == 'success']),
            'failed_models': len([r for r in results.values() if r.get('status') != 'success']),
            'results': results
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"保存最终结果: {results_file}")
        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")
        
        # 生成汇总报告
        self.generate_summary_report(results, batch_time)
    
    def generate_summary_report(self, results: Dict[str, Any], batch_time: float):
        """生成汇总报告"""
        report_file = self.results_dir / "batch_training_summary.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# 批量模型训练汇总报告\n\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**总训练时间**: {batch_time:.2f}秒 ({batch_time/60:.1f}分钟)\n")
                f.write(f"**训练模型数量**: {len(results)}\n\n")
                
                # 成功/失败统计
                successful = [r for r in results.values() if r.get('status') == 'success']
                failed = [r for r in results.values() if r.get('status') != 'success']
                
                f.write("## 训练状态统计\n\n")
                f.write(f"- ✅ **成功**: {len(successful)} 个模型\n")
                f.write(f"- ❌ **失败**: {len(failed)} 个模型\n")
                f.write(f"- 📊 **成功率**: {len(successful)/len(results)*100:.1f}%\n\n")
                
                # 成功模型详情
                if successful:
                    f.write("## 成功训练的模型\n\n")
                    f.write("| 模型名称 | 训练时间(s) | Rel-L2 | MAE | PSNR | SSIM |\n")
                    f.write("|---------|------------|--------|-----|------|------|\n")
                    
                    for result in successful:
                        model_name = result['model_name']
                        training_time = result.get('training_time', 0)
                        rel_l2 = result.get('rel_l2', 'N/A')
                        mae = result.get('mae', 'N/A')
                        psnr = result.get('psnr', 'N/A')
                        ssim = result.get('ssim', 'N/A')
                        
                        f.write(f"| {model_name} | {training_time:.1f} | {rel_l2} | {mae} | {psnr} | {ssim} |\n")
                    f.write("\n")
                
                # 失败模型详情
                if failed:
                    f.write("## 失败的模型\n\n")
                    for result in failed:
                        model_name = result['model_name']
                        error = result.get('error', 'Unknown error')
                        f.write(f"### {model_name}\n")
                        f.write(f"**错误信息**: {error}\n\n")
                
                # 配置信息
                f.write("## 训练配置\n\n")
                f.write("基于 `configs/train.yaml` 的标准化配置:\n\n")
                f.write("- **Epochs**: 200\n")
                f.write("- **优化器**: AdamW (lr=1e-3, wd=1e-4)\n")
                f.write("- **调度器**: Cosine + 1k warmup\n")
                f.write("- **损失函数**: 三件套 (rec=1.0, spec=0.5, dc=1.0)\n")
                f.write("- **批大小**: 4\n")
                f.write("- **随机种子**: 2025\n")
                f.write("- **混合精度**: 启用\n\n")
                
            logger.info(f"生成汇总报告: {report_file}")
            
        except Exception as e:
            logger.error(f"生成汇总报告失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量训练所有模型")
    parser.add_argument(
        "--config", 
        default="configs/train.yaml",
        help="基础配置文件路径"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="指定要训练的模型列表"
    )
    parser.add_argument(
        "--output_dir",
        default="runs/batch_training_results",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建批量训练器
    trainer = BatchModelTrainer(args.config)
    
    # 如果指定了输出目录，更新结果目录
    if args.output_dir:
        trainer.results_dir = Path(args.output_dir)
        trainer.results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 运行批量训练
        results = trainer.run_batch_training(args.models)
        
        # 输出结果摘要
        successful = len([r for r in results.values() if r.get('status') == 'success'])
        total = len(results)
        
        print(f"\n🎉 批量训练完成!")
        print(f"📊 成功: {successful}/{total} 个模型")
        print(f"📁 结果保存在: {trainer.results_dir}")
        print(f"📋 查看详细报告: {trainer.results_dir / 'batch_training_summary.md'}")
        
        if successful < total:
            print(f"⚠️  有 {total - successful} 个模型训练失败，请查看日志")
            sys.exit(1)
        else:
            print("✅ 所有模型训练成功!")
            
    except Exception as e:
        logger.error(f"批量训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()