#!/usr/bin/env python3
"""
PDEBench数据集管理器
支持扫描、分析、切换多种PDE数据集
为用户提供友好的数据集管理界面
"""

import os
import h5py
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import argparse


class PDEDatasetAnalyzer:
    """PDE数据集分析器"""
    
    def __init__(self, data_root: str = "E:/2D"):
        self.data_root = Path(data_root)
        self.supported_formats = ['.hdf5', '.h5']
        self.pde_types = {
            'darcy': ['DarcyFlow', 'darcy', 'Darcy'],
            'diffusion_reaction': ['diff-react', 'diffusion', 'reaction'],
            'navier_stokes': ['NS', 'NavierStokes', 'incomp', 'CFD'],
            'burgers': ['Burgers', 'burgers'],
            'wave': ['Wave', 'wave'],
            'heat': ['Heat', 'heat', 'thermal'],
            'advection': ['Advection', 'advection'],
            'shallow_water': ['SW', 'shallow', 'water']
        }
    
    def scan_datasets(self) -> Dict[str, Any]:
        """扫描数据根目录下的所有数据集"""
        print(f"🔍 扫描数据目录: {self.data_root}")
        
        if not self.data_root.exists():
            print(f"❌ 数据目录不存在: {self.data_root}")
            return self._create_mock_analysis()
        
        datasets = {}
        total_size = 0
        
        try:
            # 递归扫描所有HDF5文件
            for file_path in self.data_root.rglob("*"):
                if file_path.suffix.lower() in self.supported_formats:
                    try:
                        dataset_info = self._analyze_dataset(file_path)
                        if dataset_info:
                            datasets[str(file_path)] = dataset_info
                            total_size += dataset_info['size_mb']
                            print(f"✅ 分析完成: {file_path.name}")
                    except Exception as e:
                        print(f"⚠️  跳过文件 {file_path.name}: {e}")
            
            return {
                'datasets': datasets,
                'total_files': len(datasets),
                'total_size_gb': total_size / 1024,
                'scan_time': datetime.now().isoformat(),
                'data_root': str(self.data_root)
            }
            
        except Exception as e:
            print(f"❌ 扫描失败: {e}")
            return self._create_mock_analysis()
    
    def _analyze_dataset(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """分析单个数据集文件"""
        try:
            with h5py.File(file_path, 'r') as f:
                # 基本信息
                info = {
                    'path': str(file_path),
                    'name': file_path.stem,
                    'size_mb': os.path.getsize(file_path) / (1024 * 1024),
                    'keys': list(f.keys()),
                    'pde_type': self._identify_pde_type(file_path, f),
                    'datasets': {},
                    'metadata': {}
                }
                
                # 分析数据集结构
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        dataset = f[key]
                        info['datasets'][key] = {
                            'shape': list(dataset.shape),
                            'dtype': str(dataset.dtype),
                            'size_mb': dataset.nbytes / (1024 * 1024),
                            'ndim': dataset.ndim
                        }
                        
                        # 计算统计信息（小数据集）
                        if dataset.nbytes < 50 * 1024 * 1024:  # 小于50MB
                            try:
                                data = dataset[:]
                                if np.issubdtype(data.dtype, np.number):
                                    info['datasets'][key].update({
                                        'min': float(np.min(data)),
                                        'max': float(np.max(data)),
                                        'mean': float(np.mean(data)),
                                        'std': float(np.std(data))
                                    })
                            except:
                                pass
                    
                    # 提取元数据
                    elif hasattr(f[key], 'attrs'):
                        info['metadata'][key] = dict(f[key].attrs)
                
                # 推断数据格式
                info['format_info'] = self._infer_data_format(info['datasets'])
                
                return info
                
        except Exception as e:
            print(f"分析文件 {file_path} 时出错: {e}")
            return None
    
    def _identify_pde_type(self, file_path: Path, h5_file) -> str:
        """识别PDE类型"""
        path_str = str(file_path).lower()
        
        for pde_type, keywords in self.pde_types.items():
            for keyword in keywords:
                if keyword.lower() in path_str:
                    return pde_type
        
        # 检查HDF5文件内的属性
        try:
            if 'equation' in h5_file.attrs:
                equation = str(h5_file.attrs['equation']).lower()
                for pde_type, keywords in self.pde_types.items():
                    for keyword in keywords:
                        if keyword.lower() in equation:
                            return pde_type
        except:
            pass
        
        return 'unknown'
    
    def _infer_data_format(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """推断数据格式"""
        format_info = {
            'is_official_format': False,
            'time_steps': 0,
            'spatial_dims': [],
            'channels': 0,
            'recommended_keys': []
        }
        
        if not datasets:
            return format_info
        
        # 查找主数据键
        main_key = None
        if 'data' in datasets:
            main_key = 'data'
        elif 'tensor' in datasets:
            main_key = 'tensor'
        else:
            # 选择最大的数据集
            main_key = max(datasets.keys(), key=lambda k: datasets[k]['size_mb'])
        
        if main_key:
            shape = datasets[main_key]['shape']
            ndim = datasets[main_key]['ndim']
            
            if ndim == 5:  # [batch, time, height, width, channels]
                format_info['is_official_format'] = True
                format_info['time_steps'] = shape[1]
                format_info['spatial_dims'] = shape[2:4]
                format_info['channels'] = shape[4]
            elif ndim == 4:  # [time, channels, height, width]
                format_info['time_steps'] = shape[0]
                format_info['channels'] = shape[1]
                format_info['spatial_dims'] = shape[2:4]
            elif ndim == 3:  # [time, height, width]
                format_info['time_steps'] = shape[0]
                format_info['channels'] = 1
                format_info['spatial_dims'] = shape[1:3]
            
            format_info['recommended_keys'] = [main_key]
        
        return format_info
    
    def _create_mock_analysis(self) -> Dict[str, Any]:
        """创建模拟分析结果（当无法访问真实数据时）"""
        print("📝 创建模拟数据集分析结果...")
        
        mock_datasets = {
            "E:/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5": {
                "name": "2D_DarcyFlow_beta1.0_Train",
                "pde_type": "darcy",
                "size_mb": 1024.5,
                "keys": ["tensor"],
                "datasets": {
                    "tensor": {
                        "shape": [10000, 1, 128, 128],
                        "dtype": "float32",
                        "size_mb": 1024.0,
                        "ndim": 4
                    }
                },
                "format_info": {
                    "is_official_format": False,
                    "time_steps": 10000,
                    "spatial_dims": [128, 128],
                    "channels": 1,
                    "recommended_keys": ["tensor"]
                }
            },
            "E:/2D/DiffusionReaction/2D_diff-react_NA_NA.h5": {
                "name": "2D_diff-react_NA_NA",
                "pde_type": "diffusion_reaction",
                "size_mb": 512.3,
                "keys": ["data"],
                "datasets": {
                    "data": {
                        "shape": [1, 20, 64, 64, 1],
                        "dtype": "float64",
                        "size_mb": 512.0,
                        "ndim": 5
                    }
                },
                "format_info": {
                    "is_official_format": True,
                    "time_steps": 20,
                    "spatial_dims": [64, 64],
                    "channels": 1,
                    "recommended_keys": ["data"]
                }
            },
            "E:/2D/NavierStokes/2D_incompNS_Re1000_Train.hdf5": {
                "name": "2D_incompNS_Re1000_Train",
                "pde_type": "navier_stokes",
                "size_mb": 2048.7,
                "keys": ["u", "v", "p"],
                "datasets": {
                    "u": {
                        "shape": [5000, 128, 128],
                        "dtype": "float32",
                        "size_mb": 682.0,
                        "ndim": 3
                    },
                    "v": {
                        "shape": [5000, 128, 128],
                        "dtype": "float32",
                        "size_mb": 682.0,
                        "ndim": 3
                    },
                    "p": {
                        "shape": [5000, 128, 128],
                        "dtype": "float32",
                        "size_mb": 682.0,
                        "ndim": 3
                    }
                },
                "format_info": {
                    "is_official_format": False,
                    "time_steps": 5000,
                    "spatial_dims": [128, 128],
                    "channels": 3,
                    "recommended_keys": ["u", "v", "p"]
                }
            }
        }
        
        return {
            'datasets': mock_datasets,
            'total_files': len(mock_datasets),
            'total_size_gb': sum(d['size_mb'] for d in mock_datasets.values()) / 1024,
            'scan_time': datetime.now().isoformat(),
            'data_root': str(self.data_root),
            'is_mock': True
        }


class DatasetConfigGenerator:
    """数据集配置生成器"""
    
    def __init__(self):
        self.task_configs = {
            'sr_x2': {
                'type': 'sr',
                'scale_factor': 2,
                'sigma': 1.0
            },
            'sr_x4': {
                'type': 'sr',
                'scale_factor': 4,
                'sigma': 1.0
            },
            'crop_20': {
                'type': 'crop',
                'crop_ratio': 0.2
            },
            'crop_40': {
                'type': 'crop',
                'crop_ratio': 0.4
            }
        }
    
    def generate_config(self, dataset_info: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """生成完整的训练配置"""
        if task_type not in self.task_configs:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        task_config = self.task_configs[task_type]
        
        # 确定图像尺寸
        spatial_dims = dataset_info.get('spatial_dims', [])
        if not spatial_dims:
            spatial_dims = [128, 128]  # 默认值
        image_size = max(spatial_dims) if spatial_dims else 128
        
        # 确定通道数
        channels = dataset_info.get('channels', 0)
        if channels == 0:
            # 根据数据集类型推断通道数
            if 'darcy' in dataset_info['name'].lower():
                channels = 1
            elif 'diff' in dataset_info['name'].lower() or 'react' in dataset_info['name'].lower():
                channels = 2
            elif 'ns' in dataset_info['name'].lower() or 'navier' in dataset_info['name'].lower():
                channels = 1
            else:
                channels = 1
        
        # 生成完整配置
        config = {
            'defaults': [
                '_self_'
            ],
            'experiment': {
                'name': f"{dataset_info['name']}_{task_type}",
                'device': 'cuda',
                'seed': 42,
                'output_dir': f"runs/{dataset_info['name']}_{task_type}",
                'tags': [task_type, dataset_info['name']],
                'notes': f"Auto-generated config for {dataset_info['name']} with {task_type} task"
            },
            'data': {
                '_target_': 'datasets.pdebench.PDEBenchDataModule',
                'data_path': dataset_info['path'],
                'dataset_name': dataset_info['name'],
                'batch_size': 4 if image_size <= 256 else 2,
                'num_workers': 4,
                'pin_memory': True,
                'image_size': image_size,
                'keys': dataset_info.get('keys', []),
                'use_official_format': dataset_info.get('use_official_format', False)
            },
            'model': {
                'in_channels': channels,
                'out_channels': channels,
                'image_size': image_size,
                'embed_dim': 96 if image_size <= 256 else 128,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8,
                'mlp_ratio': 4.0,
                'drop_rate': 0.0,
                'drop_path_rate': 0.1,
                'use_checkpoint': False
            },
            'task': task_config,
            'training': {
                'max_epochs': 50 if task_type.startswith('sr') else 30,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'warmup_epochs': 5,
                'gradient_clip_val': 1.0,
                'use_amp': True,
                'early_stopping': {
                    'patience': 10,
                    'min_delta': 1e-4,
                    'monitor': 'val_loss'
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
                    'weight': 0.5,
                    'low_freq_modes': 16
                },
                'data_consistency': {
                    'type': 'l2',
                    'weight': 1.0
                }
            },
            'logging': {
                'log_every_n_steps': 50,
                'val_check_interval': 1.0,
                'save_top_k': 3,
                'monitor': 'val_loss',
                'mode': 'min',
                'use_wandb': False,
                'wandb_project': 'sparse2full',
                'wandb_entity': None
            }
        }
        
        return config
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """保存配置到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 配置已保存到: {output_path}")


class DatasetManager:
    """数据集管理器主类"""
    
    def __init__(self, data_root: str = "E:/2D"):
        self.analyzer = PDEDatasetAnalyzer(data_root)
        self.config_generator = DatasetConfigGenerator()
        self.analysis_cache = None
    
    def list_datasets(self) -> Dict[str, Any]:
        """列出所有可用数据集"""
        if self.analysis_cache is None:
            self.analysis_cache = self.analyzer.scan_datasets()
        
        return self.analysis_cache
    
    def show_dataset_summary(self):
        """显示数据集摘要"""
        analysis = self.list_datasets()
        
        print("\n" + "=" * 80)
        print("📊 PDEBench 数据集摘要")
        print("=" * 80)
        
        if analysis.get('is_mock'):
            print("⚠️  注意: 由于无法访问数据目录，显示的是模拟数据")
        
        print(f"📁 数据根目录: {analysis['data_root']}")
        print(f"📄 总文件数: {analysis['total_files']}")
        print(f"💾 总大小: {analysis['total_size_gb']:.2f} GB")
        print(f"🕐 扫描时间: {analysis['scan_time']}")
        
        print("\n" + "=" * 60)
        print("📋 数据集详情")
        print("=" * 60)
        
        # 按PDE类型分组显示
        pde_groups = {}
        for path, dataset in analysis['datasets'].items():
            pde_type = dataset['pde_type']
            if pde_type not in pde_groups:
                pde_groups[pde_type] = []
            pde_groups[pde_type].append(dataset)
        
        for pde_type, datasets in pde_groups.items():
            print(f"\n🔬 {pde_type.upper()} ({len(datasets)} 个数据集):")
            for dataset in datasets:
                print(f"  📄 {dataset['name']}")
                print(f"     大小: {dataset['size_mb']:.1f} MB")
                print(f"     键名: {dataset['keys']}")
                format_info = dataset['format_info']
                print(f"     格式: {format_info['spatial_dims']} @ {format_info['time_steps']} 时间步")
                print(f"     通道: {format_info['channels']}")
    
    def generate_dataset_config(self, dataset_name: str, task_type: str = 'sr_x4', 
                              output_dir: str = 'configs/datasets') -> str:
        """为指定数据集生成配置文件"""
        analysis = self.list_datasets()
        
        # 查找数据集
        target_dataset = None
        for path, dataset in analysis['datasets'].items():
            if dataset_name.lower() in dataset['name'].lower():
                target_dataset = dataset
                break
        
        if target_dataset is None:
            available = [d['name'] for d in analysis['datasets'].values()]
            raise ValueError(f"数据集 '{dataset_name}' 未找到。可用数据集: {available}")
        
        # 生成配置
        config = self.config_generator.generate_config(target_dataset, task_type)
        
        # 保存配置
        safe_name = dataset_name.replace(' ', '_').replace('-', '_').lower()
        output_path = f"{output_dir}/{safe_name}_{task_type}.yaml"
        self.config_generator.save_config(config, output_path)
        
        return output_path
    
    def interactive_mode(self):
        """交互式模式"""
        print("\n🎯 PDEBench 数据集管理器 - 交互模式")
        print("=" * 50)
        
        while True:
            print("\n请选择操作:")
            print("1. 显示数据集摘要")
            print("2. 生成数据集配置")
            print("3. 列出所有数据集")
            print("4. 退出")
            
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == '1':
                self.show_dataset_summary()
            
            elif choice == '2':
                analysis = self.list_datasets()
                datasets = list(analysis['datasets'].values())
                
                print("\n可用数据集:")
                for i, dataset in enumerate(datasets, 1):
                    print(f"{i}. {dataset['name']} ({dataset['pde_type']})")
                
                try:
                    idx = int(input("\n选择数据集编号: ")) - 1
                    if 0 <= idx < len(datasets):
                        dataset = datasets[idx]
                        
                        print("\n可用任务类型:")
                        tasks = ['sr_x2', 'sr_x4', 'crop_20', 'crop_40']
                        for i, task in enumerate(tasks, 1):
                            print(f"{i}. {task}")
                        
                        task_idx = int(input("\n选择任务类型编号: ")) - 1
                        if 0 <= task_idx < len(tasks):
                            task_type = tasks[task_idx]
                            
                            try:
                                config_path = self.generate_dataset_config(
                                    dataset['name'], task_type
                                )
                                print(f"\n✅ 配置文件已生成: {config_path}")
                            except Exception as e:
                                print(f"\n❌ 生成配置失败: {e}")
                        else:
                            print("❌ 无效的任务类型编号")
                    else:
                        print("❌ 无效的数据集编号")
                except ValueError:
                    print("❌ 请输入有效的数字")
            
            elif choice == '3':
                analysis = self.list_datasets()
                print(f"\n📋 找到 {len(analysis['datasets'])} 个数据集:")
                for i, (path, dataset) in enumerate(analysis['datasets'].items(), 1):
                    print(f"{i:2d}. {dataset['name']}")
                    print(f"     类型: {dataset['pde_type']}")
                    print(f"     大小: {dataset['size_mb']:.1f} MB")
                    print(f"     路径: {path}")
            
            elif choice == '4':
                print("\n👋 再见!")
                break
            
            else:
                print("❌ 无效选择，请重试")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDEBench数据集管理器')
    parser.add_argument('--data-root', default='E:/2D', help='数据根目录')
    parser.add_argument('--list', action='store_true', help='列出所有数据集')
    parser.add_argument('--summary', action='store_true', help='显示数据集摘要')
    parser.add_argument('--generate-config', help='为指定数据集生成配置')
    parser.add_argument('--task-type', default='sr_x4', help='任务类型')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    manager = DatasetManager(args.data_root)
    
    if args.list:
        analysis = manager.list_datasets()
        for path, dataset in analysis['datasets'].items():
            print(f"{dataset['name']} ({dataset['pde_type']}) - {dataset['size_mb']:.1f} MB")
    
    elif args.summary:
        manager.show_dataset_summary()
    
    elif args.generate_config:
        try:
            config_path = manager.generate_dataset_config(args.generate_config, args.task_type)
            print(f"配置文件已生成: {config_path}")
        except Exception as e:
            print(f"生成配置失败: {e}")
    
    elif args.interactive:
        manager.interactive_mode()
    
    else:
        # 默认显示摘要
        manager.show_dataset_summary()


if __name__ == "__main__":
    main()