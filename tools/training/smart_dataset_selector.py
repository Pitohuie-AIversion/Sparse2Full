#!/usr/bin/env python3
"""
智能数据集选择和配置工具
基于dataset_manager.py，实现随机选择3种不同类型的PDE数据集
并为每种数据集自动生成最优训练配置
"""

import os
import random
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import argparse

from dataset_manager import DatasetManager, PDEDatasetAnalyzer, DatasetConfigGenerator


class SmartDatasetSelector:
    """智能数据集选择器"""
    
    def __init__(self, data_root: str = "data/2D", seed: int = 42):
        self.data_root = data_root
        self.seed = seed
        self.manager = DatasetManager(data_root)
        self.config_generator = DatasetConfigGenerator()
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 扩展任务配置
        self.task_configs = {
            'sr_x2': {
                'type': 'sr',
                'scale_factor': 2,
                'sigma': 1.0,
                'description': '超分辨率重建 2x',
                'priority': 'high'
            },
            'sr_x4': {
                'type': 'sr', 
                'scale_factor': 4,
                'sigma': 1.0,
                'description': '超分辨率重建 4x',
                'priority': 'high'
            },
            'crop_20': {
                'type': 'crop',
                'crop_ratio': 0.2,
                'description': '稀疏观测重建 20%',
                'priority': 'medium'
            },
            'crop_40': {
                'type': 'crop',
                'crop_ratio': 0.4,
                'description': '稀疏观测重建 40%',
                'priority': 'medium'
            }
        }
        
        # PDE类型优先级和特性
        self.pde_characteristics = {
            'darcy': {
                'complexity': 'medium',
                'recommended_tasks': ['sr_x4', 'crop_20'],
                'optimal_lr': 1e-3,
                'optimal_batch_size': 4,
                'typical_channels': 1,
                'description': 'Darcy流方程 - 多孔介质流动'
            },
            'diffusion_reaction': {
                'complexity': 'high',
                'recommended_tasks': ['sr_x2', 'crop_40'],
                'optimal_lr': 5e-4,
                'optimal_batch_size': 2,
                'typical_channels': 2,
                'description': '扩散反应方程 - 化学反应扩散'
            },
            'navier_stokes': {
                'complexity': 'very_high',
                'recommended_tasks': ['sr_x2', 'crop_20'],
                'optimal_lr': 1e-4,
                'optimal_batch_size': 2,
                'typical_channels': 3,
                'description': 'Navier-Stokes方程 - 不可压缩流体'
            },
            'burgers': {
                'complexity': 'low',
                'recommended_tasks': ['sr_x4', 'crop_40'],
                'optimal_lr': 2e-3,
                'optimal_batch_size': 8,
                'typical_channels': 1,
                'description': 'Burgers方程 - 非线性波动'
            },
            'wave': {
                'complexity': 'medium',
                'recommended_tasks': ['sr_x2', 'crop_20'],
                'optimal_lr': 1e-3,
                'optimal_batch_size': 4,
                'typical_channels': 1,
                'description': '波动方程 - 波传播现象'
            },
            'heat': {
                'complexity': 'low',
                'recommended_tasks': ['sr_x4', 'crop_40'],
                'optimal_lr': 1e-3,
                'optimal_batch_size': 6,
                'typical_channels': 1,
                'description': '热传导方程 - 热扩散'
            }
        }
    
    def select_diverse_datasets(self, num_datasets: int = 3) -> List[Dict[str, Any]]:
        """智能选择多样化的数据集"""
        print(f"🎯 开始智能选择 {num_datasets} 种不同类型的PDE数据集...")
        
        # 获取所有数据集
        analysis = self.manager.list_datasets()
        all_datasets = list(analysis['datasets'].values())
        
        if len(all_datasets) == 0:
            raise ValueError("未找到任何数据集")
        
        # 按PDE类型分组
        pde_groups = {}
        for dataset in all_datasets:
            pde_type = dataset['pde_type']
            if pde_type not in pde_groups:
                pde_groups[pde_type] = []
            pde_groups[pde_type].append(dataset)
        
        print(f"📊 发现 {len(pde_groups)} 种PDE类型: {list(pde_groups.keys())}")
        
        # 智能选择策略
        selected_datasets = []
        selected_types = set()
        
        # 优先选择不同类型
        available_types = list(pde_groups.keys())
        random.shuffle(available_types)
        
        for pde_type in available_types:
            if len(selected_datasets) >= num_datasets:
                break
                
            # 从该类型中选择最佳数据集
            candidates = pde_groups[pde_type]
            
            # 评分选择最佳候选
            best_dataset = self._score_and_select_dataset(candidates, pde_type)
            if best_dataset:
                selected_datasets.append(best_dataset)
                selected_types.add(pde_type)
                print(f"✅ 选择 {pde_type}: {best_dataset['name']}")
        
        # 如果还需要更多数据集，从已选类型中选择
        if len(selected_datasets) < num_datasets:
            remaining_datasets = [d for d in all_datasets 
                                if d not in selected_datasets]
            
            while len(selected_datasets) < num_datasets and remaining_datasets:
                dataset = random.choice(remaining_datasets)
                selected_datasets.append(dataset)
                remaining_datasets.remove(dataset)
                print(f"✅ 补充选择: {dataset['name']} ({dataset['pde_type']})")
        
        print(f"\n🎉 成功选择 {len(selected_datasets)} 个数据集:")
        for i, dataset in enumerate(selected_datasets, 1):
            print(f"  {i}. {dataset['name']} ({dataset['pde_type']})")
        
        return selected_datasets
    
    def _score_and_select_dataset(self, candidates: List[Dict[str, Any]], 
                                 pde_type: str) -> Optional[Dict[str, Any]]:
        """为候选数据集评分并选择最佳"""
        if not candidates:
            return None
        
        scored_candidates = []
        
        for dataset in candidates:
            score = 0
            
            # 大小评分 (适中的大小更好)
            size_mb = dataset['size_mb']
            if 100 <= size_mb <= 2000:  # 100MB - 2GB
                score += 10
            elif 50 <= size_mb <= 5000:  # 50MB - 5GB
                score += 5
            
            # 格式评分
            format_info = dataset.get('format_info', {})
            if format_info.get('is_official_format'):
                score += 15
            
            # 空间分辨率评分
            spatial_dims = format_info.get('spatial_dims', [])
            if spatial_dims:
                resolution = max(spatial_dims)
                if 64 <= resolution <= 256:  # 适中分辨率
                    score += 10
                elif 32 <= resolution <= 512:
                    score += 5
            
            # 时间步数评分
            time_steps = format_info.get('time_steps', 0)
            if 100 <= time_steps <= 10000:
                score += 8
            elif 50 <= time_steps <= 20000:
                score += 4
            
            # 数据完整性评分
            if dataset.get('keys') and len(dataset['keys']) > 0:
                score += 5
            
            scored_candidates.append((score, dataset))
        
        # 选择得分最高的
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]
    
    def generate_optimal_configs(self, selected_datasets: List[Dict[str, Any]], 
                               output_dir: str = "configs/auto_generated") -> List[Dict[str, Any]]:
        """为选中的数据集生成最优配置"""
        print(f"\n🔧 为 {len(selected_datasets)} 个数据集生成最优训练配置...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_configs = []
        
        for i, dataset in enumerate(selected_datasets, 1):
            print(f"\n📝 处理数据集 {i}: {dataset['name']}")
            
            pde_type = dataset['pde_type']
            pde_chars = self.pde_characteristics.get(pde_type, {})
            
            # 为每个数据集生成多种任务配置
            recommended_tasks = pde_chars.get('recommended_tasks', ['sr_x4', 'crop_20'])
            
            for task_type in recommended_tasks:
                config_info = self._generate_optimized_config(dataset, task_type, pde_chars)
                
                # 保存配置文件
                safe_name = dataset['name'].replace(' ', '_').replace('-', '_').lower()
                config_filename = f"{safe_name}_{task_type}_optimized.yaml"
                config_path = output_dir / config_filename
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_info['config'], f, default_flow_style=False, allow_unicode=True)
                
                config_info['config_path'] = str(config_path)
                generated_configs.append(config_info)
                
                print(f"  ✅ {task_type}: {config_path}")
        
        print(f"\n🎉 成功生成 {len(generated_configs)} 个优化配置文件")
        return generated_configs
    
    def _generate_optimized_config(self, dataset: Dict[str, Any], task_type: str, 
                                  pde_chars: Dict[str, Any]) -> Dict[str, Any]:
        """生成单个优化配置"""
        format_info = dataset.get('format_info', {})
        spatial_dims = format_info.get('spatial_dims', [128, 128])
        channels = format_info.get('channels', pde_chars.get('typical_channels', 1))
        image_size = max(spatial_dims) if spatial_dims else 128
        
        # 优化的训练参数
        optimal_lr = pde_chars.get('optimal_lr', 1e-3)
        optimal_batch_size = pde_chars.get('optimal_batch_size', 4)
        
        # 根据图像大小调整批处理大小
        if image_size > 256:
            optimal_batch_size = max(1, optimal_batch_size // 2)
        elif image_size < 128:
            optimal_batch_size = min(8, optimal_batch_size * 2)
        
        # 根据任务类型调整参数
        task_config = self.task_configs[task_type]
        if task_config['type'] == 'sr':
            epochs = 100 if task_config['scale_factor'] >= 4 else 80
            lr_factor = 1.0
        else:  # crop
            epochs = 60 if task_config['crop_ratio'] <= 0.2 else 50
            lr_factor = 0.8  # crop任务使用稍低的学习率
        
        # 生成完整配置
        config = {
            'defaults': ['_self_'],
            'experiment': {
                'name': f"{dataset['name']}_{task_type}_optimized",
                'device': 'cuda',
                'seed': self.seed,
                'output_dir': f"runs/{dataset['name']}_{task_type}_optimized",
                'tags': [task_type, dataset['pde_type'], 'auto_optimized'],
                'notes': f"Auto-optimized config for {dataset['name']} with {task_type} task"
            },
            'data': {
                '_target_': 'datasets.pdebench.PDEBenchDataModule',
                'data_path': dataset.get('path', ''),
                'dataset_name': dataset['name'],
                'batch_size': optimal_batch_size,
                'num_workers': 4,
                'pin_memory': True,
                'image_size': image_size,
                'keys': dataset.get('keys', []),
                'use_official_format': format_info.get('is_official_format', False)
            },
            'model': {
                'name': 'SwinUNet',  # 默认使用最佳模型
                'in_channels': channels,
                'out_channels': channels,
                'image_size': image_size,
                'embed_dim': 96 if image_size <= 256 else 128,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8 if image_size >= 128 else 4,
                'mlp_ratio': 4.0,
                'drop_rate': 0.0,
                'drop_path_rate': 0.1,
                'use_checkpoint': image_size > 256
            },
            'task': task_config,
            'training': {
                'max_epochs': epochs,
                'learning_rate': optimal_lr * lr_factor,
                'weight_decay': 1e-4,
                'optimizer': 'adamw',
                'scheduler': 'cosine_warmup',
                'warmup_epochs': max(5, epochs // 20),
                'gradient_clip_val': 1.0,
                'use_amp': True,
                'early_stopping': {
                    'patience': max(10, epochs // 10),
                    'min_delta': 1e-4,
                    'monitor': 'val_rel_l2'
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
                    'weight': 0.0,  # 使用稳定配置
                    'low_freq_modes': 4
                },
                'data_consistency': {
                    'type': 'l2',
                    'weight': 0.0  # 使用稳定配置
                }
            },
            'logging': {
                'log_every_n_steps': 50,
                'val_check_interval': 1.0,
                'save_top_k': 3,
                'monitor': 'val_rel_l2',
                'mode': 'min',
                'use_wandb': False,
                'wandb_project': 'sparse2full_auto',
                'wandb_entity': None
            }
        }
        
        return {
            'dataset': dataset,
            'task_type': task_type,
            'config': config,
            'optimization_info': {
                'pde_type': dataset['pde_type'],
                'complexity': pde_chars.get('complexity', 'medium'),
                'optimal_lr': optimal_lr * lr_factor,
                'optimal_batch_size': optimal_batch_size,
                'epochs': epochs,
                'image_size': image_size,
                'channels': channels
            }
        }
    
    def create_batch_training_script(self, config_infos: List[Dict[str, Any]], 
                                   output_path: str = "batch_train_selected_datasets.py") -> str:
        """创建批量训练脚本"""
        print(f"\n🚀 创建批量训练脚本: {output_path}")
        
        script_content = f'''#!/usr/bin/env python3
"""
自动生成的批量训练脚本
训练选中的 {len(config_infos)} 个数据集配置
生成时间: {datetime.now().isoformat()}
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_training(config_path, experiment_name):
    """运行单个训练任务"""
    print(f"\\n{'='*60}")
    print(f"🚀 开始训练: {{experiment_name}}")
    print(f"📁 配置文件: {{config_path}}")
    print(f"🕐 开始时间: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"{'='*60}")
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        f"--config-path={{Path(config_path).parent}}",
        f"--config-name={{Path(config_path).stem}}"
    ]
    
    try:
        # 执行训练
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\\n✅ 训练完成: {{experiment_name}}")
        print(f"⏱️  训练时长: {{duration/3600:.2f}} 小时")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ 训练失败: {{experiment_name}}")
        print(f"错误代码: {{e.returncode}}")
        return False, 0
    except KeyboardInterrupt:
        print(f"\\n⚠️  训练被用户中断: {{experiment_name}}")
        return False, 0

def main():
    """主函数"""
    print("🎯 开始批量训练选中的数据集")
    print(f"📊 总计 {len(config_infos)} 个训练任务")
    
    # 训练配置列表
    training_configs = [
'''
        
        # 添加配置信息
        for config_info in config_infos:
            dataset_name = config_info['dataset']['name']
            task_type = config_info['task_type']
            config_path = config_info['config_path']
            pde_type = config_info['dataset']['pde_type']
            
            script_content += f'''        {{
            "name": "{dataset_name}_{task_type}",
            "config_path": "{config_path}",
            "dataset": "{dataset_name}",
            "task_type": "{task_type}",
            "pde_type": "{pde_type}"
        }},
'''
        
        script_content += f'''    ]
    
    # 执行训练
    successful_runs = 0
    failed_runs = 0
    total_time = 0
    
    for i, config in enumerate(training_configs, 1):
        print(f"\\n📋 进度: {{i}}/{len(config_infos)}")
        print(f"🔬 PDE类型: {{config['pde_type']}}")
        print(f"📊 数据集: {{config['dataset']}}")
        print(f"🎯 任务: {{config['task_type']}}")
        
        success, duration = run_training(config['config_path'], config['name'])
        
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
        
        total_time += duration
        
        # 显示进度统计
        print(f"\\n📈 当前统计:")
        print(f"  ✅ 成功: {{successful_runs}}")
        print(f"  ❌ 失败: {{failed_runs}}")
        print(f"  ⏱️  总时长: {{total_time/3600:.2f}} 小时")
    
    # 最终统计
    print(f"\\n{'='*80}")
    print("🎉 批量训练完成!")
    print(f"📊 最终统计:")
    print(f"  ✅ 成功训练: {{successful_runs}}/{len(config_infos)}")
    print(f"  ❌ 失败训练: {{failed_runs}}/{len(config_infos)}")
    print(f"  ⏱️  总训练时长: {{total_time/3600:.2f}} 小时")
    print(f"  📅 完成时间: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
'''
        
        # 保存脚本
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 设置执行权限 (Unix系统)
        try:
            os.chmod(output_path, 0o755)
        except:
            pass
        
        print(f"✅ 批量训练脚本已生成: {output_path}")
        return output_path
    
    def generate_comparison_report(self, selected_datasets: List[Dict[str, Any]], 
                                 config_infos: List[Dict[str, Any]], 
                                 output_path: str = "dataset_comparison_report.md") -> str:
        """生成数据集对比分析报告"""
        print(f"\n📊 生成数据集对比分析报告: {output_path}")
        
        report_content = f"""# 智能数据集选择与配置报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**随机种子**: {self.seed}  
**数据根目录**: {self.data_root}

## 📋 执行摘要

本报告展示了智能数据集选择器从 `{self.data_root}` 目录中自动选择的 {len(selected_datasets)} 种不同类型的PDE数据集，并为每个数据集生成了 {len(config_infos)} 个优化的训练配置。

## 🎯 选中的数据集

"""
        
        # 数据集详情表格
        report_content += "| 序号 | 数据集名称 | PDE类型 | 大小(MB) | 分辨率 | 时间步 | 通道数 | 复杂度 |\n"
        report_content += "|------|------------|---------|----------|--------|--------|--------|--------|\n"
        
        for i, dataset in enumerate(selected_datasets, 1):
            format_info = dataset.get('format_info', {})
            pde_type = dataset['pde_type']
            pde_chars = self.pde_characteristics.get(pde_type, {})
            
            spatial_dims = format_info.get('spatial_dims', [])
            resolution = f"{spatial_dims[0]}×{spatial_dims[1]}" if len(spatial_dims) >= 2 else "未知"
            
            report_content += f"| {i} | {dataset['name']} | {pde_type} | {dataset['size_mb']:.1f} | {resolution} | {format_info.get('time_steps', 0)} | {format_info.get('channels', 0)} | {pde_chars.get('complexity', '未知')} |\n"
        
        # 配置详情
        report_content += f"\n## ⚙️ 生成的训练配置\n\n"
        report_content += f"总计生成了 **{len(config_infos)}** 个优化配置文件：\n\n"
        
        # 按数据集分组显示配置
        dataset_configs = {}
        for config_info in config_infos:
            dataset_name = config_info['dataset']['name']
            if dataset_name not in dataset_configs:
                dataset_configs[dataset_name] = []
            dataset_configs[dataset_name].append(config_info)
        
        for dataset_name, configs in dataset_configs.items():
            report_content += f"### 📊 {dataset_name}\n\n"
            
            for config_info in configs:
                opt_info = config_info['optimization_info']
                task_type = config_info['task_type']
                task_desc = self.task_configs[task_type]['description']
                
                report_content += f"**{task_type}** ({task_desc}):\n"
                report_content += f"- 学习率: {opt_info['optimal_lr']:.2e}\n"
                report_content += f"- 批处理大小: {opt_info['optimal_batch_size']}\n"
                report_content += f"- 训练轮数: {opt_info['epochs']}\n"
                report_content += f"- 图像尺寸: {opt_info['image_size']}×{opt_info['image_size']}\n"
                report_content += f"- 配置文件: `{config_info['config_path']}`\n\n"
        
        # PDE类型分析
        report_content += "## 🔬 PDE类型分析\n\n"
        
        pde_types = set(d['pde_type'] for d in selected_datasets)
        for pde_type in pde_types:
            pde_chars = self.pde_characteristics.get(pde_type, {})
            datasets_of_type = [d for d in selected_datasets if d['pde_type'] == pde_type]
            
            report_content += f"### {pde_type.upper()}\n\n"
            report_content += f"**描述**: {pde_chars.get('description', '未知')}\n"
            report_content += f"**复杂度**: {pde_chars.get('complexity', '未知')}\n"
            report_content += f"**推荐任务**: {', '.join(pde_chars.get('recommended_tasks', []))}\n"
            report_content += f"**数据集数量**: {len(datasets_of_type)}\n\n"
        
        # 优化策略说明
        report_content += """## 🧠 优化策略

### 数据集选择策略
1. **多样性优先**: 确保选择不同类型的PDE方程
2. **质量评分**: 基于数据大小、格式、分辨率等因素评分
3. **平衡选择**: 兼顾计算复杂度和训练效果

### 配置优化策略
1. **PDE特性适配**: 根据不同PDE类型的特性调整参数
2. **任务类型优化**: SR和Crop任务使用不同的优化策略
3. **资源平衡**: 根据图像尺寸动态调整批处理大小
4. **稳定配置**: 使用经过验证的稳定损失函数权重

## 🚀 使用方法

### 1. 执行批量训练
```bash
python batch_train_selected_datasets.py
```

### 2. 单独训练某个配置
```bash
python train.py --config-path configs/auto_generated --config-name [配置文件名]
```

### 3. 监控训练进度
训练日志将保存在各自的 `runs/` 目录下。

## 📈 预期结果

基于优化策略，预期各数据集的训练效果：
- **Darcy流**: 适合SR任务，预期Rel-L2 < 0.1
- **扩散反应**: 复杂度高，需要更多训练轮数
- **Navier-Stokes**: 最具挑战性，需要小学习率和长时间训练

## 📝 注意事项

1. 训练时间可能较长，建议在GPU环境下运行
2. 监控显存使用，必要时调整批处理大小
3. 可根据实际效果进一步调整超参数
4. 建议保存训练日志用于后续分析

---
*本报告由智能数据集选择器自动生成*
"""
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 对比分析报告已生成: {output_path}")
        return output_path
    
    def run_complete_workflow(self, num_datasets: int = 3, 
                            output_dir: str = "configs/auto_generated") -> Dict[str, Any]:
        """运行完整的工作流程"""
        print("🎯 开始智能数据集选择和配置工作流程")
        print("=" * 80)
        
        workflow_results = {
            'start_time': datetime.now().isoformat(),
            'seed': self.seed,
            'num_datasets': num_datasets,
            'output_dir': output_dir
        }
        
        try:
            # 1. 选择数据集
            selected_datasets = self.select_diverse_datasets(num_datasets)
            workflow_results['selected_datasets'] = selected_datasets
            
            # 2. 生成配置
            config_infos = self.generate_optimal_configs(selected_datasets, output_dir)
            workflow_results['generated_configs'] = config_infos
            
            # 3. 创建批量训练脚本
            batch_script_path = self.create_batch_training_script(config_infos)
            workflow_results['batch_script_path'] = batch_script_path
            
            # 4. 生成对比报告
            report_path = self.generate_comparison_report(selected_datasets, config_infos)
            workflow_results['report_path'] = report_path
            
            workflow_results['success'] = True
            workflow_results['end_time'] = datetime.now().isoformat()
            
            # 保存工作流程结果
            results_path = f"{output_dir}/workflow_results.json"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n🎉 工作流程完成!")
            print(f"📊 选择了 {len(selected_datasets)} 个数据集")
            print(f"⚙️  生成了 {len(config_infos)} 个配置")
            print(f"🚀 批量训练脚本: {batch_script_path}")
            print(f"📋 对比报告: {report_path}")
            print(f"💾 结果摘要: {results_path}")
            
            return workflow_results
            
        except Exception as e:
            workflow_results['success'] = False
            workflow_results['error'] = str(e)
            workflow_results['end_time'] = datetime.now().isoformat()
            
            print(f"\n❌ 工作流程失败: {e}")
            return workflow_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能数据集选择和配置工具')
    parser.add_argument('--data-root', default='data/2D', help='数据根目录')
    parser.add_argument('--num-datasets', type=int, default=3, help='选择的数据集数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', default='configs/auto_generated', help='输出目录')
    parser.add_argument('--select-only', action='store_true', help='仅选择数据集，不生成配置')
    parser.add_argument('--generate-configs', action='store_true', help='仅生成配置，不创建训练脚本')
    
    args = parser.parse_args()
    
    # 创建选择器
    selector = SmartDatasetSelector(args.data_root, args.seed)
    
    if args.select_only:
        # 仅选择数据集
        selected_datasets = selector.select_diverse_datasets(args.num_datasets)
        print(f"\n✅ 已选择 {len(selected_datasets)} 个数据集")
        
    elif args.generate_configs:
        # 选择数据集并生成配置
        selected_datasets = selector.select_diverse_datasets(args.num_datasets)
        config_infos = selector.generate_optimal_configs(selected_datasets, args.output_dir)
        print(f"\n✅ 已生成 {len(config_infos)} 个配置文件")
        
    else:
        # 运行完整工作流程
        results = selector.run_complete_workflow(args.num_datasets, args.output_dir)
        
        if results['success']:
            print(f"\n🎯 下一步操作:")
            print(f"1. 查看对比报告: {results['report_path']}")
            print(f"2. 执行批量训练: python {results['batch_script_path']}")
            print(f"3. 监控训练进度: 查看各自的 runs/ 目录")
        else:
            print(f"\n❌ 工作流程失败，请检查错误信息")


if __name__ == "__main__":
    main()