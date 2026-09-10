#!/usr/bin/env python3
"""
简化版多模型扫描工具 - 单次运行模式
避免seeds分支的配置覆盖问题，使用真实PDEBench数据进行完整的7模型对比
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
        logging.FileHandler('multi_model_scan_single.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('MultiModelScanSingle')

class MultiModelScannerSingle:
    """多模型扫描器 - 单次运行模式避免配置覆盖"""
    
    def __init__(self, output_dir: str):
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
    
    def create_model_config(self, model_name: str, seed: int) -> str:
        """为特定模型创建配置并返回配置文件路径"""
        model_info = self.all_models[model_name]
        
        # 创建临时配置文件
        temp_config = self.output_dir / f'config_{model_name}_s{seed}.yaml'
        
        with open(temp_config, 'w', encoding='utf-8') as f:
            # 实验配置
            f.write(f"# 配置 - {model_name} 种子{seed}\n")
            f.write(f"# 基于真实PDEBench数据，2通道输入输出\n\n")
            
            f.write("experiment:\n")
            f.write(f"  name: \"AR-DR2D-{model_name}-s{seed}-RealData\"\n")
            f.write(f"  description: \"{model_info['description']} - 真实数据评估\"\n")
            f.write(f"  seed: {seed}\n")
            f.write(f"  output_dir: \"{self.output_dir / f'{model_name}_s{seed}'}\"\n")
            f.write(f"  device: cuda\n")
            f.write(f"  log_every_n_steps: 10\n\n")
            
            # 设备配置
            f.write("device:\n")
            f.write(f"  accelerator: cuda\n")
            f.write(f"  devices: 1\n")
            f.write(f"  strategy: auto\n")
            f.write(f"  precision: bf16-mixed\n\n")
            
            # AR配置
            f.write("ar:\n")
            f.write(f"  enabled: false\n\n")
            
            f.write("sequential:\n")
            f.write(f"  enabled: false\n\n")
            
            # 模型配置 - 关键：2通道输入输出
            f.write("model:\n")
            f.write(f"  name: \"{model_name}\"\n")
            f.write(f"  in_channels: 2\n")  # 关键：匹配真实数据
            f.write(f"  out_channels: 2\n")  # 关键：匹配真实数据
            f.write(f"  img_size: 128\n")
            
            # 模型特定参数
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
            
            # 数据配置 - 关键：使用真实PDEBench数据
            f.write("data:\n")
            f.write(f"  data_path: \"data/DR2D/2D_diff-react_NA_NA.h5\"\n")
            f.write(f"  dataset_name: RealDiffusionReaction\n")
            f.write(f"  input_channels: 2\n")  # 关键：匹配真实数据
            f.write(f"  target_channels: 2\n")  # 关键：匹配真实数据
            f.write(f"  img_size: 128\n")
            f.write(f"  normalize: true\n")
            f.write(f"  use_synthetic_data: false\n")  # 关键：禁用合成数据
            f.write(f"  splits_dir: splits\n")
            f.write(f"  train_ratio: 0.7\n")
            f.write(f"  val_ratio: 0.15\n")
            f.write(f"  test_ratio: 0.15\n")
            f.write(f"  time_step_start: 0\n")
            f.write(f"  time_step_end: 50\n")  # 适中时间步数
            f.write(f"  time_step_stride: 1\n")
            
            # 观测配置
            f.write(f"  observation:\n")
            f.write(f"    mode: SR\n")
            f.write(f"    sr:\n")
            f.write(f"      scale_factor: 2\n")
            f.write(f"      blur_sigma: 1.0\n")
            f.write(f"      blur_kernel_size: 5\n")
            f.write(f"      boundary_mode: mirror\n")
            f.write(f"      downsample_mode: area\n")
            f.write(f"      align_corners: false\n")
            f.write(f"      antialias: true\n")
            
            # 数据加载器
            f.write(f"  dataloader:\n")
            f.write(f"    batch_size: 128\n")
            f.write(f"    val_batch_size: 128\n")
            f.write(f"    test_batch_size: 128\n")
            f.write(f"    num_workers: 8\n")
            f.write(f"    pin_memory: true\n")
            f.write(f"    persistent_workers: true\n")
            f.write(f"    prefetch_factor: 4\n")
            f.write(f"    drop_last: true\n")
            f.write(f"    shuffle: true\n")
            f.write(f"    timeout: 60\n")
            
            f.write("\n")
            
            # 训练配置
            f.write("training:\n")
            f.write(f"  epochs: 30\n")  # 适中训练轮数
            f.write(f"  batch_size: 128\n")
            f.write(f"  gradient_accumulation_steps: 1\n")
            f.write(f"  torch_compile: false\n")
            f.write(f"  channels_last: true\n")
            
            # 优化器
            f.write(f"  optimizer:\n")
            f.write(f"    name: \"AdamW\"\n")
            f.write(f"    lr: 0.001\n")
            f.write(f"    weight_decay: 0.0001\n")
            f.write(f"    betas: [0.9, 0.999]\n")
            f.write(f"    eps: 1e-8\n")
            f.write(f"    fused: false\n")
            f.write(f"    foreach: false\n")
            
            # 调度器
            f.write(f"  scheduler:\n")
            f.write(f"    name: \"CosineAnnealingLR\"\n")
            f.write(f"    T_max: 30\n")
            f.write(f"    eta_min: 1e-6\n")
            f.write(f"    warmup_epochs: 3\n")
            
            f.write(f"  gradient_clip_val: 1.0\n")
            f.write(f"  gradient_clip_algorithm: norm\n")
            
            # AMP
            f.write(f"  amp:\n")
            f.write(f"    enabled: true\n")
            f.write(f"    autocast_dtype: bfloat16\n")
            
            # 验证
            f.write(f"  validation:\n")
            f.write(f"    enabled: true\n")
            f.write(f"    check_val_every_n_epoch: 5\n")
            f.write(f"    save_val_batch_for_viz: true\n")
            f.write(f"    log_val_metrics: true\n")
            
            # 检查点
            f.write(f"  checkpoint:\n")
            f.write(f"    save_best: true\n")
            f.write(f"    save_last: true\n")
            f.write(f"    max_keep: 2\n")
            f.write(f"    monitor: val_loss\n")
            f.write(f"    mode: min\n")
            f.write(f"    save_every_n_epochs: 10\n")
            
            # 早停
            f.write(f"  early_stopping:\n")
            f.write(f"    enabled: true\n")
            f.write(f"    patience: 10\n")
            f.write(f"    monitor: val_loss\n")
            f.write(f"    mode: min\n")
            f.write(f"    min_delta: 1e-4\n")
            
            f.write("\n")
            
            # 损失配置
            f.write("loss:\n")
            f.write(f"  reconstruction:\n")
            f.write(f"    weight: 1.0\n")
            f.write(f"  spectral:\n")
            f.write(f"    weight: 0.5\n")
            f.write(f"  degradation_consistency:\n")
            f.write(f"    weight: 1.0\n")
            f.write(f"  gradient_weight: 0.0\n")
            f.write(f"  ar_loss:\n")
            f.write(f"    weight: 0.0\n")
            f.write(f"    reduction: mean\n")
            
            f.write("\n")
            
            # 验证配置
            f.write("validation:\n")
            f.write(f"  check_val_every_n_epoch: 5\n")
            f.write(f"  use_observation: true\n")
            f.write(f"  metrics:\n")
            f.write(f"    - rel_l2\n")
            f.write(f"    - mae\n")
            f.write(f"    - psnr\n")
            f.write(f"    - ssim\n")
            f.write(f"  rollout_steps: [1]\n")
            f.write(f"  convergence_criteria:\n")
            f.write(f"    target_rel_l2: 0.5\n")
            f.write(f"    patience_for_convergence: 5\n")
            f.write(f"    min_improvement: 1e-3\n")
            
            f.write("\n")
            
            # 测试配置 - 关键：启用测试
            f.write("testing:\n")
            f.write(f"  enabled: true\n")
            f.write(f"  run_final_test: true\n")
            f.write(f"  save_predictions: true\n")
            f.write(f"  compute_detailed_metrics: true\n")
            f.write(f"  batch_size: 128\n")
            f.write(f"  save_visualizations: true\n")
            f.write(f"  num_visualization_samples: 3\n")
            f.write(f"  fast_mode: false\n")
            f.write(f"  skip_complex_metrics: false\n")
            f.write(f"  minimal_logging: false\n")
            
            f.write("\n")
            
            # 日志配置
            f.write("logging:\n")
            f.write(f"  experiment_name: \"AR-DR2D-{model_name}-s{seed}-RealData\"\n")
            f.write(f"  version: null\n")
            f.write(f"  default_hp_metric: false\n")
            f.write(f"  log_model: false\n")
            f.write(f"  performance_monitoring:\n")
            f.write(f"    log_gpu_memory: true\n")
            f.write(f"    log_throughput: true\n")
            f.write(f"    log_batch_time: true\n")
            f.write(f"  tensorboard:\n")
            f.write(f"    save_dir: runs/tensorboard\n")
            f.write(f"    name: multi_model_scan\n")
            f.write(f"    version: null\n")
            f.write(f"  visualization:\n")
            f.write(f"    save_samples_every_n_epochs: 10\n")
            f.write(f"    num_samples_to_save: 3\n")
            f.write(f"    save_training_curves: true\n")
            f.write(f"    save_rollout_visualization: false\n")
            
            f.write("\n")
            f.write("seed: 2025\n")
        
        return str(temp_config)
    
    def run_single_experiment(self, model_name: str, seed: int) -> Dict:
        """运行单个实验 - 使用单次运行模式避免配置覆盖"""
        logger.info(f"开始实验: {model_name}_s{seed}")
        
        # 创建配置文件
        config_path = self.create_model_config(model_name, seed)
        
        # 执行训练命令 - 关键：不使用--seeds参数，避免配置覆盖
        cmd = [
            sys.executable, 'tools/training/train_real_data_ar.py',
            '--config', config_path,
            '--model', model_name
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
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
                logger.error(f"错误信息: {result.stderr[:500]}...")  # 限制错误信息长度
                
                return {
                    'model': model_name,
                    'seed': seed,
                    'success': False,
                    'duration': elapsed_time,
                    'error': result.stderr[:500],  # 限制错误信息长度
                    'group': self.all_models[model_name]['group'],
                    'channels': self.all_models[model_name]['channels']
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ 实验 {model_name}_s{seed} 超时 (1小时)")
            return {
                'model': model_name,
                'seed': seed,
                'success': False,
                'error': '超时 (1小时)',
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
        logger.info("开始多模型完整扫描实验 - 单次运行模式")
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
                    time.sleep(10)
        
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
        for channels in [2]:
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
        results_file = self.output_dir / 'multi_model_scan_single_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 也保存CSV格式便于分析
        csv_file = self.output_dir / 'multi_model_scan_single_summary.csv'
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("模型,组,输入通道,种子,成功,用时(秒),输出目录\n")
            for exp in self.results['experiments']:
                f.write(f"{exp['model']},{exp['group']},{exp['channels']},{exp['seed']},"
                       f"{'是' if exp['success'] else '否'},{exp.get('duration', 0):.1f},"
                       f"{exp.get('output_dir', '')}\n")

def main():
    """主函数"""
    # 输出目录
    output_dir = Path('paper_package/multi_model_scan_real_data_single')
    
    # 创建扫描器
    scanner = MultiModelScannerSingle(str(output_dir))
    
    # 运行所有实验
    scanner.run_all_experiments()

if __name__ == '__main__':
    main()