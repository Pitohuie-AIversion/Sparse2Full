"""环境设置脚本

自动化环境配置和依赖安装，确保实验环境的一致性和可复现性。

功能：
- 检查系统要求
- 安装Python依赖
- 配置CUDA环境
- 下载预训练模型
- 验证环境配置

使用方法：
python scripts/setup_environment.py --install_all
python scripts/setup_environment.py --check_only
python scripts/setup_environment.py --download_models
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.request import urlretrieve
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EnvironmentSetup:
    """环境设置器"""
    
    def __init__(self):
        self.project_root = project_root
        self.system_info = self._get_system_info()
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 系统要求
        self.requirements = {
            'python_version': (3, 10),
            'memory_gb': 16,
            'storage_gb': 50,
            'cuda_version': '11.8'
        }
        
        # 依赖包配置
        self.dependencies = {
            'core': [
                'torch>=2.1.0',
                'torchvision>=0.16.0',
                'numpy>=1.24.0',
                'scipy>=1.10.0',
                'matplotlib>=3.6.0',
                'seaborn>=0.12.0',
                'pandas>=1.5.0',
                'pillow>=9.0.0',
                'tqdm>=4.64.0',
                'omegaconf>=2.3.0',
                'hydra-core>=1.3.0'
            ],
            'scientific': [
                'scikit-image>=0.20.0',
                'scikit-learn>=1.2.0',
                'h5py>=3.8.0',
                'netcdf4>=1.6.0',
                'xarray>=2023.1.0'
            ],
            'visualization': [
                'plotly>=5.13.0',
                'ipywidgets>=8.0.0',
                'jupyter>=1.0.0'
            ],
            'development': [
                'pytest>=7.2.0',
                'black>=23.0.0',
                'isort>=5.12.0',
                'flake8>=6.0.0',
                'mypy>=1.0.0'
            ]
        }
        
        # 预训练模型配置
        self.pretrained_models = {
            'swin_unet_base': {
                'url': 'https://github.com/microsoft/Swin-Transformer/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
                'path': 'pretrained/swin_base_patch4_window7_224.pth',
                'description': 'Swin Transformer Base model'
            },
            'fno_weights': {
                'url': 'https://example.com/fno_pretrained.pth',  # 替换为实际URL
                'path': 'pretrained/fno_pretrained.pth',
                'description': 'Fourier Neural Operator pretrained weights'
            }
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'platform': platform.platform(),
            'python_version': sys.version_info[:3],
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'machine': platform.machine()
        }
        
        # 获取内存信息
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['memory_total_gb'] = memory.total / (1024**3)
            info['memory_available_gb'] = memory.available / (1024**3)
        except ImportError:
            info['memory_total_gb'] = 'unknown'
            info['memory_available_gb'] = 'unknown'
        
        # 获取GPU信息
        try:
            import torch
            if torch.cuda.is_available():
                info['cuda_available'] = True
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                info['gpu_memory'] = [torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                                    for i in range(torch.cuda.device_count())]
            else:
                info['cuda_available'] = False
        except ImportError:
            info['cuda_available'] = 'unknown'
        
        return info
    
    def check_system_requirements(self) -> bool:
        """检查系统要求"""
        self.logger.info("Checking system requirements...")
        
        checks_passed = True
        
        # 检查Python版本
        python_version = self.system_info['python_version'][:2]
        if python_version < self.requirements['python_version']:
            self.logger.error(f"Python version {python_version} < required {self.requirements['python_version']}")
            checks_passed = False
        else:
            self.logger.info(f"✓ Python version: {python_version}")
        
        # 检查内存
        if isinstance(self.system_info['memory_total_gb'], (int, float)):
            if self.system_info['memory_total_gb'] < self.requirements['memory_gb']:
                self.logger.warning(f"Low memory: {self.system_info['memory_total_gb']:.1f}GB < {self.requirements['memory_gb']}GB")
            else:
                self.logger.info(f"✓ Memory: {self.system_info['memory_total_gb']:.1f}GB")
        
        # 检查存储空间
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free / (1024**3)
            
            if free_gb < self.requirements['storage_gb']:
                self.logger.warning(f"Low disk space: {free_gb:.1f}GB < {self.requirements['storage_gb']}GB")
            else:
                self.logger.info(f"✓ Available storage: {free_gb:.1f}GB")
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
        
        # 检查CUDA
        if self.system_info.get('cuda_available'):
            self.logger.info(f"✓ CUDA available: {self.system_info['cuda_version']}")
            self.logger.info(f"✓ GPU count: {self.system_info['gpu_count']}")
            for i, (name, memory) in enumerate(zip(self.system_info['gpu_names'], self.system_info['gpu_memory'])):
                self.logger.info(f"  GPU {i}: {name} ({memory:.1f}GB)")
        else:
            self.logger.warning("CUDA not available, will use CPU")
        
        return checks_passed
    
    def install_dependencies(self, 
                           categories: List[str] = None,
                           upgrade: bool = False,
                           user_install: bool = False) -> bool:
        """安装依赖包"""
        
        if categories is None:
            categories = ['core', 'scientific']
        
        self.logger.info(f"Installing dependencies: {categories}")
        
        # 收集所有包
        packages = []
        for category in categories:
            if category in self.dependencies:
                packages.extend(self.dependencies[category])
            else:
                self.logger.warning(f"Unknown category: {category}")
        
        if not packages:
            self.logger.error("No packages to install")
            return False
        
        # 构建pip命令
        cmd = [sys.executable, '-m', 'pip', 'install']
        
        if upgrade:
            cmd.append('--upgrade')
        
        if user_install:
            cmd.append('--user')
        
        # 分批安装以避免依赖冲突
        batch_size = 5
        for i in range(0, len(packages), batch_size):
            batch = packages[i:i+batch_size]
            batch_cmd = cmd + batch
            
            self.logger.info(f"Installing batch {i//batch_size + 1}: {batch}")
            
            try:
                result = subprocess.run(
                    batch_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10分钟超时
                )
                
                if result.returncode == 0:
                    self.logger.info(f"✓ Batch {i//batch_size + 1} installed successfully")
                else:
                    self.logger.error(f"✗ Batch {i//batch_size + 1} failed:")
                    self.logger.error(result.stderr)
                    return False
            
            except subprocess.TimeoutExpired:
                self.logger.error(f"✗ Batch {i//batch_size + 1} timed out")
                return False
            except Exception as e:
                self.logger.error(f"✗ Batch {i//batch_size + 1} error: {e}")
                return False
        
        self.logger.info("All dependencies installed successfully")
        return True
    
    def setup_cuda_environment(self) -> bool:
        """设置CUDA环境"""
        self.logger.info("Setting up CUDA environment...")
        
        if not self.system_info.get('cuda_available'):
            self.logger.warning("CUDA not available, skipping CUDA setup")
            return True
        
        try:
            # 检查PyTorch CUDA支持
            import torch
            if torch.cuda.is_available():
                self.logger.info(f"✓ PyTorch CUDA support: {torch.version.cuda}")
                
                # 测试CUDA操作
                device = torch.device('cuda:0')
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.matmul(test_tensor, test_tensor.T)
                
                self.logger.info("✓ CUDA operations test passed")
                return True
            else:
                self.logger.error("PyTorch CUDA support not available")
                return False
        
        except Exception as e:
            self.logger.error(f"CUDA setup failed: {e}")
            return False
    
    def download_pretrained_models(self, models: List[str] = None) -> bool:
        """下载预训练模型"""
        
        if models is None:
            models = list(self.pretrained_models.keys())
        
        self.logger.info(f"Downloading pretrained models: {models}")
        
        # 创建预训练模型目录
        pretrained_dir = self.project_root / 'pretrained'
        pretrained_dir.mkdir(exist_ok=True)
        
        success_count = 0
        
        for model_name in models:
            if model_name not in self.pretrained_models:
                self.logger.warning(f"Unknown model: {model_name}")
                continue
            
            model_config = self.pretrained_models[model_name]
            model_path = pretrained_dir / Path(model_config['path']).name
            
            # 检查是否已存在
            if model_path.exists():
                self.logger.info(f"✓ {model_name} already exists: {model_path}")
                success_count += 1
                continue
            
            try:
                self.logger.info(f"Downloading {model_name}...")
                self.logger.info(f"  URL: {model_config['url']}")
                self.logger.info(f"  Path: {model_path}")
                
                # 下载文件
                def progress_hook(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min(100, (block_num * block_size * 100) // total_size)
                        if block_num % 100 == 0:  # 每100个块打印一次
                            print(f"\r  Progress: {percent}%", end='', flush=True)
                
                urlretrieve(model_config['url'], model_path, progress_hook)
                print()  # 换行
                
                # 验证下载
                if model_path.exists() and model_path.stat().st_size > 0:
                    self.logger.info(f"✓ {model_name} downloaded successfully")
                    success_count += 1
                else:
                    self.logger.error(f"✗ {model_name} download failed (empty file)")
                    if model_path.exists():
                        model_path.unlink()
            
            except Exception as e:
                self.logger.error(f"✗ {model_name} download failed: {e}")
                if model_path.exists():
                    model_path.unlink()
        
        self.logger.info(f"Downloaded {success_count}/{len(models)} models successfully")
        return success_count == len(models)
    
    def create_directory_structure(self) -> bool:
        """创建项目目录结构"""
        self.logger.info("Creating directory structure...")
        
        directories = [
            'data',
            'configs',
            'models',
            'datasets',
            'ops',
            'utils',
            'tools',
            'scripts',
            'runs',
            'paper_package',
            'paper_package/data_cards',
            'paper_package/configs',
            'paper_package/checkpoints',
            'paper_package/metrics',
            'paper_package/figs',
            'paper_package/scripts',
            'pretrained',
            'logs'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"✓ Created directory: {dir_path}")
        
        # 创建__init__.py文件
        init_files = [
            'models/__init__.py',
            'datasets/__init__.py',
            'ops/__init__.py',
            'utils/__init__.py'
        ]
        
        for init_file in init_files:
            init_path = self.project_root / init_file
            if not init_path.exists():
                init_path.touch()
                self.logger.debug(f"✓ Created __init__.py: {init_path}")
        
        self.logger.info("Directory structure created successfully")
        return True
    
    def create_config_templates(self) -> bool:
        """创建配置模板"""
        self.logger.info("Creating configuration templates...")
        
        configs_dir = self.project_root / 'configs'
        
        # 基础配置模板
        base_config = {
            'model': {
                'name': 'swin_unet',
                'in_channels': 1,
                'out_channels': 1,
                'img_size': 256,
                'patch_size': 4,
                'window_size': 8,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'mlp_ratio': 4.0
            },
            'data': {
                'dataset': 'pde_bench',
                'data_dir': './data',
                'task': 'sr',
                'scale_factor': 4,
                'image_size': 256,
                'batch_size': 8,
                'num_workers': 4,
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1
            },
            'training': {
                'epochs': 100,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'warmup_epochs': 10,
                'gradient_clip': 1.0,
                'amp': True,
                'checkpoint_interval': 10,
                'early_stopping_patience': 20
            },
            'loss': {
                'reconstruction_weight': 1.0,
                'spectral_weight': 0.5,
                'data_consistency_weight': 1.0,
                'spectral_modes': 16
            },
            'evaluation': {
                'metrics': ['rel_l2', 'mae', 'psnr', 'ssim', 'frmse', 'brmse', 'crmse'],
                'boundary_width': 16,
                'frequency_bands': [0.25, 0.5]
            }
        }
        
        # 保存基础配置
        base_config_path = configs_dir / 'base_config.yaml'
        with open(base_config_path, 'w') as f:
            import yaml
            yaml.dump(base_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"✓ Created base config: {base_config_path}")
        
        # 创建任务特定配置
        tasks = {
            'sr_x2': {'data': {'task': 'sr', 'scale_factor': 2}},
            'sr_x4': {'data': {'task': 'sr', 'scale_factor': 4}},
            'crop_40': {'data': {'task': 'crop', 'crop_ratio': 0.4}},
            'crop_20': {'data': {'task': 'crop', 'crop_ratio': 0.2}}
        }
        
        for task_name, task_config in tasks.items():
            # 合并配置
            merged_config = base_config.copy()
            for key, value in task_config.items():
                if key in merged_config:
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            
            # 保存任务配置
            task_config_path = configs_dir / f'{task_name}_config.yaml'
            with open(task_config_path, 'w') as f:
                import yaml
                yaml.dump(merged_config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"✓ Created task config: {task_config_path}")
        
        return True
    
    def validate_installation(self) -> bool:
        """验证安装"""
        self.logger.info("Validating installation...")
        
        validation_passed = True
        
        # 检查核心包
        core_packages = ['torch', 'numpy', 'matplotlib', 'omegaconf']
        for package in core_packages:
            try:
                __import__(package)
                self.logger.info(f"✓ {package}")
            except ImportError:
                self.logger.error(f"✗ {package} not found")
                validation_passed = False
        
        # 检查CUDA（如果可用）
        if self.system_info.get('cuda_available'):
            try:
                import torch
                if torch.cuda.is_available():
                    # 简单的CUDA测试
                    device = torch.device('cuda:0')
                    x = torch.randn(10, 10, device=device)
                    y = x @ x.T
                    self.logger.info("✓ CUDA operations")
                else:
                    self.logger.error("✗ CUDA not available in PyTorch")
                    validation_passed = False
            except Exception as e:
                self.logger.error(f"✗ CUDA test failed: {e}")
                validation_passed = False
        
        # 检查目录结构
        required_dirs = ['data', 'configs', 'models', 'runs']
        for directory in required_dirs:
            dir_path = self.project_root / directory
            if dir_path.exists():
                self.logger.info(f"✓ Directory: {directory}")
            else:
                self.logger.error(f"✗ Directory missing: {directory}")
                validation_passed = False
        
        # 检查配置文件
        base_config_path = self.project_root / 'configs' / 'base_config.yaml'
        if base_config_path.exists():
            self.logger.info("✓ Base configuration")
        else:
            self.logger.error("✗ Base configuration missing")
            validation_passed = False
        
        return validation_passed
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """生成设置报告"""
        
        report = {
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now()),
            'system_info': self.system_info,
            'requirements_check': self.check_system_requirements(),
            'project_root': str(self.project_root),
            'python_executable': sys.executable,
            'environment_variables': {
                'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                'TORCH_HOME': os.environ.get('TORCH_HOME', '')
            }
        }
        
        # 保存报告
        report_path = self.project_root / 'setup_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Setup report saved to: {report_path}")
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Environment Setup')
    parser.add_argument('--check_only', action='store_true',
                       help='Only check requirements without installing')
    parser.add_argument('--install_all', action='store_true',
                       help='Install all dependencies')
    parser.add_argument('--install_categories', type=str, nargs='+',
                       choices=['core', 'scientific', 'visualization', 'development'],
                       help='Install specific categories of dependencies')
    parser.add_argument('--download_models', action='store_true',
                       help='Download pretrained models')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Specific models to download')
    parser.add_argument('--create_structure', action='store_true',
                       help='Create directory structure and config templates')
    parser.add_argument('--validate', action='store_true',
                       help='Validate installation')
    parser.add_argument('--upgrade', action='store_true',
                       help='Upgrade existing packages')
    parser.add_argument('--user_install', action='store_true',
                       help='Install packages for current user only')
    
    args = parser.parse_args()
    
    try:
        # 创建设置器
        setup = EnvironmentSetup()
        
        # 检查系统要求
        if not setup.check_system_requirements():
            if args.check_only:
                print("System requirements check failed!")
                return 1
            else:
                print("Warning: System requirements not fully met, continuing anyway...")
        
        if args.check_only:
            print("System requirements check passed!")
            return 0
        
        # 创建目录结构
        if args.create_structure or args.install_all:
            if not setup.create_directory_structure():
                print("Failed to create directory structure")
                return 1
            
            if not setup.create_config_templates():
                print("Failed to create config templates")
                return 1
        
        # 安装依赖
        if args.install_all:
            categories = ['core', 'scientific']
            if not setup.install_dependencies(categories, args.upgrade, args.user_install):
                print("Failed to install dependencies")
                return 1
        elif args.install_categories:
            if not setup.install_dependencies(args.install_categories, args.upgrade, args.user_install):
                print("Failed to install dependencies")
                return 1
        
        # 设置CUDA环境
        if args.install_all or args.install_categories:
            if not setup.setup_cuda_environment():
                print("Warning: CUDA setup failed, continuing anyway...")
        
        # 下载预训练模型
        if args.download_models:
            models = args.models if args.models else None
            if not setup.download_pretrained_models(models):
                print("Warning: Some models failed to download")
        
        # 验证安装
        if args.validate or args.install_all:
            if not setup.validate_installation():
                print("Installation validation failed!")
                return 1
            else:
                print("Installation validation passed!")
        
        # 生成报告
        setup.generate_setup_report()
        
        print("Environment setup completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())