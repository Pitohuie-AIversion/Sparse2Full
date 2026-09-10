#!/usr/bin/env python3
"""
增强版多模型扫描工具
使用真实PDEBench数据进行完整的7模型对比
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_model_scan.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('MultiModelScan')

class MultiModelScanner:
    """多模型扫描器 - 使用真实数据进行完整对比"""
    
    def __init__(self, config_path: str, output_dir: str):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义所有7个模型及其配置
        # 基于实际PDEBench数据：data_shape=(101,128,128,2)，有2个物理通道
        self.all_models = {
            # 基准组 - 2通道输入（匹配真实数据）
            'swin_unet': {
                'channels': 2,
                'description': 'Swin Transformer U-Net',
                'group': 'baseline'
            },
            'unet': {
                'channels': 2,
                'description': '经典U-Net',
                'group': 'baseline'
            },
            'fno2d': {
                'channels': 2,
                'description': '2D傅里叶神经算子',
                'group': 'baseline'
            },
            'segformer': {
                'channels': 2,
                'description': 'SegFormer分割模型',
                'group': 'baseline'
            },
            # 高级组 - 2通道输入（匹配真实数据）
            'hybrid': {
                'channels': 2,
                'description': '混合注意力模型',
                'group': 'advanced'
            },
            'vit': {
                'channels': 2,
                'description': 'Vision Transformer',
                'group': 'advanced'
            },
            'mlp_mixer': {
                'channels': 2,
                'description': 'MLP-Mixer',
                'group': 'advanced'
            }
        }
        
        self.seeds = [42, 123, 456]  # 三种子确保统计显著性
        
        # 结果收集
        self.results = {
            'experiments': [],
            'summary': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def create_model_config(self, model_name: str, seed: int) -> Dict:
        """为特定模型创建配置"""
        model_info = self.all_models[model_name]
        
        # 基础配置
        config = {
            'experiment': {
                'name': f'AR-DR2D-{model_name}-s{seed}',
                'description': f'{model_info["description"]} - 种子{seed}',
                'seed': seed
            },
            'model': {
                'name': model_name,
                'in_channels': model_info['channels'],  # 关键：根据模型设置输入通道
                'out_channels': 1,
                'img_size': 128
            },
            'data': {
                'input_channels': model_info['channels']  # 数据输入通道与模型匹配
            }
        }
        
        return config
    
    def run_single_experiment(self, model_name: str, seed: int) -> Dict:
        """运行单个实验"""
        logger.info(f"开始实验: {model_name}_s{seed}")
        
        # 创建临时配置文件
        temp_config = self.output_dir / f'temp_config_{model_name}_s{seed}.yaml'
        
        # 生成模型特定配置
        model_config = self.create_model_config(model_name, seed)
        
        # 写入临时配置文件
        with open(temp_config, 'w', encoding='utf-8') as f:
            # 写入基础配置头
            f.write(f"# 临时配置 - {model_name} 种子{seed}\n")
            f.write(f"# 基于: {self.config_path}\n\n")
            
            # 实验配置
            f.write("experiment:\n")
            f.write(f"  name: \"{model_config['experiment']['name']}\"\n")
            f.write(f"  description: \"{model_config['experiment']['description']}\"\n")
            f.write(f"  seed: {seed}\n")
            f.write(f"  output_dir: \"{self.output_dir / f'{model_name}_s{seed}'}\"\n")
            f.write(f"  device: cuda\n")
            f.write(f"  log_every_n_steps: 10\n\n")
            
            # 模型配置
            f.write("model:\n")
            f.write(f"  name: \"{model_name}\"\n")
            f.write(f"  in_channels: {model_config['model']['in_channels']}\n")
            f.write(f"  out_channels: 1\n")
            f.write(f"  img_size: 128\n")
            
            # 添加模型特定参数
            if model_name == 'swin_unet':
                f.write(f"  patch_size: 4\n")
                f.write(f"  window_size: 8\n")
                f.write(f"  depths: [2, 2, 6, 2]\n")
                f.write(f"  num_heads: [3, 6, 12, 24]\n")
                f.write(f"  embed_dim: 96\n")
            elif model_name == 'unet':
                f.write(f"  base_channels: 64\n")
                f.write(f"  channel_multiplier: [1, 2, 4, 8]\n")
                f.write(f"  num_blocks: [2, 2, 2, 2]\n")
            elif model_name == 'fno2d':
                f.write(f"  modes: 16\n")
                f.write(f"  width: 32\n")
                f.write(f"  layers: 4\n")
            elif model_name == 'segformer':
                f.write(f"  encoder_name: \"mit_b0\"\n")
                f.write(f"  decoder_channels: 256\n")
            elif model_name == 'hybrid':
                f.write(f"  hidden_dim: 256\n")
                f.write(f"  num_heads: 8\n")
                f.write(f"  num_layers: 6\n")
            elif model_name == 'vit':
                f.write(f"  patch_size: 16\n")
                f.write(f"  hidden_dim: 768\n")
                f.write(f"  num_heads: 12\n")
                f.write(f"  num_layers: 12\n")
            elif model_name == 'mlp_mixer':
                f.write(f"  patch_size: 16\n")
                f.write(f"  hidden_dim: 256\n")
                f.write(f"  num_blocks: 8\n")
                f.write(f"  tokens_mlp_dim: 256\n")
                f.write(f"  channels_mlp_dim: 1024\n")
            
            f.write("\n")
            
            # 数据配置（关键：确保使用真实数据）
            f.write("data:\n")
            f.write(f"  data_path: \"data/DR2D/2D_diff-react_NA_NA.h5\"\n")
            f.write(f"  dataset_name: RealDiffusionReaction\n")
            f.write(f"  input_channels: {model_config['data']['input_channels']}\n")
            f.write(f"  target_channels: {model_config['data']['input_channels']}\n")  # 输入输出通道一致
            f.write(f"  img_size: 128\n")
            f.write(f"  normalize: true\n")
            f.write(f"  use_synthetic_data: false\n")  # 关键：禁用合成数据
            f.write(f"  splits_dir: splits\n")
            f.write(f"  train_ratio: 0.7\n")
            f.write(f"  val_ratio: 0.15\n")
            f.write(f"  test_ratio: 0.15\n")
            
            # 简化的训练配置
            f.write("\ntraining:\n")
            f.write(f"  epochs: 50\n")  # 适中训练轮数
            f.write(f"  batch_size: 256\n")
            f.write(f"  validation:\n")
            f.write(f"    enabled: true\n")
            f.write(f"    check_val_every_n_epoch: 5\n")
            
            # 启用测试
            f.write("\ntesting:\n")
            f.write(f"  enabled: true\n")
            f.write(f"  run_final_test: true\n")
            f.write(f"  compute_detailed_metrics: true\n")
            f.write(f"  batch_size: 128\n")
        
        # 执行训练命令
        cmd = [
            sys.executable, 'tools/training/train_real_data_ar.py',
            '--config', str(temp_config),
            '--model', model_name,
            '--seeds', str(seed)
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ 实验 {model_name}_s{seed} 成功完成 (用时: {elapsed_time:.1f}s)")
                
                # 解析结果
                experiment_result = {
                    'model': model_name,
                    'seed': seed,
                    'success': True,
                    'duration': elapsed_time,
                    'group': self.all_models[model_name]['group'],
                    'channels': self.all_models[model_name]['channels'],
                    'output_dir': str(self.output_dir / f'{model_name}_s{seed}')
                }
                
                return experiment_result
            else:
                logger.error(f"❌ 实验 {model_name}_s{seed} 失败 (用时: {elapsed_time:.1f}s)")
                logger.error(f"错误信息: {result.stderr}")
                
                return {
                    'model': model_name,
                    'seed': seed,
                    'success': False,
                    'duration': elapsed_time,
                    'error': result.stderr,
                    'group': self.all_models[model_name]['group'],
                    'channels': self.all_models[model_name]['channels']
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ 实验 {model_name}_s{seed} 超时 (30分钟)")
            return {
                'model': model_name,
                'seed': seed,
                'success': False,
                'error': '超时 (30分钟)',
                'group': self.all_models[model_name]['group'],
                'channels': self.all_models[model_name]['channels']
            }
        
        except Exception as e:
            logger.error(f"💥 实验 {model_name}_s{seed} 异常: {str(e)}")
            return {
                'model': model_name,
                'seed': seed,
                'success': False,
                'error': str(e),
                'group': self.all_models[model_name]['group'],
                'channels': self.all_models[model_name]['channels']
            }
    
    def run_all_experiments(self):
        """运行所有实验"""
        logger.info("=" * 60)
        logger.info("开始多模型完整扫描实验")
        logger.info(f"总计: {len(self.all_models)} 个模型 × {len(self.seeds)} 个种子 = {len(self.all_models) * len(self.seeds)} 个实验")
        logger.info("=" * 60)
        
        total_experiments = len(self.all_models) * len(self.seeds)
        completed = 0
        successful = 0
        
        # 按组运行实验
        for group_name in ['baseline', 'advanced']:
            group_models = [name for name, info in self.all_models.items() if info['group'] == group_name]
            
            logger.info(f"\n🔍 开始{group_name}组实验 ({len(group_models)} 个模型)")
            logger.info("-" * 50)
            
            for model_name in group_models:
                for seed in self.seeds:
                    completed += 1
                    logger.info(f"\n实验进度: [{completed}/{total_experiments}] 模型: {model_name}, 种子: {seed}")
                    
                    result = self.run_single_experiment(model_name, seed)
                    self.results['experiments'].append(result)
                    
                    if result['success']:
                        successful += 1
                    
                    # 保存中间结果
                    self.save_results()
                    
                    # 短暂休息避免GPU过热
                    time.sleep(5)
        
        # 生成最终总结
        self.generate_summary()
        self.save_results()
        
        logger.info("\n" + "=" * 60)
        logger.info("实验扫描完成!")
        logger.info(f"总实验数: {total_experiments}")
        logger.info(f"成功: {successful} ({successful/total_experiments*100:.1f}%)")
        logger.info(f"失败: {total_experiments - successful}")
        logger.info(f"结果保存至: {self.output_dir}")
        logger.info("=" * 60)
    
    def generate_summary(self):
        """生成实验总结"""
        summary = {
            'total_experiments': len(self.results['experiments']),
            'successful_experiments': sum(1 for exp in self.results['experiments'] if exp['success']),
            'failed_experiments': sum(1 for exp in self.results['experiments'] if not exp['success']),
            'success_rate': sum(1 for exp in self.results['experiments'] if exp['success']) / len(self.results['experiments']) * 100,
            'by_model': {},
            'by_group': {},
            'by_channels': {}
        }
        
        # 按模型统计
        for model_name in self.all_models:
            model_exps = [exp for exp in self.results['experiments'] if exp['model'] == model_name]
            if model_exps:
                summary['by_model'][model_name] = {
                    'total': len(model_exps),
                    'successful': sum(1 for exp in model_exps if exp['success']),
                    'success_rate': sum(1 for exp in model_exps if exp['success']) / len(model_exps) * 100,
                    'avg_duration': sum(exp.get('duration', 0) for exp in model_exps) / len(model_exps)
                }
        
        # 按组统计
        for group_name in ['baseline', 'advanced']:
            group_exps = [exp for exp in self.results['experiments'] if exp['group'] == group_name]
            if group_exps:
                summary['by_group'][group_name] = {
                    'total': len(group_exps),
                    'successful': sum(1 for exp in group_exps if exp['success']),
                    'success_rate': sum(1 for exp in group_exps if exp['success']) / len(group_exps) * 100
                }
        
        # 按通道数统计
        for channels in [1, 4]:
            channel_exps = [exp for exp in self.results['experiments'] if exp['channels'] == channels]
            if channel_exps:
                summary['by_channels'][channels] = {
                    'total': len(channel_exps),
                    'successful': sum(1 for exp in channel_exps if exp['success']),
                    'success_rate': sum(1 for exp in channel_exps if exp['success']) / len(channel_exps) * 100
                }
        
        self.results['summary'] = summary
    
    def save_results(self):
        """保存结果"""
        results_file = self.output_dir / 'multi_model_scan_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 也保存CSV格式便于分析
        csv_file = self.output_dir / 'multi_model_scan_summary.csv'
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("模型,组,输入通道,种子,成功,用时(秒),输出目录\n")
            for exp in self.results['experiments']:
                f.write(f"{exp['model']},{exp['group']},{exp['channels']},{exp['seed']},"
                       f"{'是' if exp['success'] else '否'},{exp.get('duration', 0):.1f},"
                       f"{exp.get('output_dir', '')}\n")

def main():
    """主函数"""
    # 输出目录
    output_dir = Path('paper_package/multi_model_scan_real_data')
    
    # 基础配置文件
    base_config = 'configs/train/ar_multi_model_scan.yaml'
    
    # 创建扫描器
    scanner = MultiModelScanner(base_config, str(output_dir))
    
    # 运行所有实验
    scanner.run_all_experiments()

if __name__ == '__main__':
    main()