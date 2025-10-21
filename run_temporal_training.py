#!/usr/bin/env python3
"""
时序PDE训练一键运行脚本

使用方法:
    python run_temporal_training.py --config configs/experiment/temporal_training.yaml
    python run_temporal_training.py --config configs/experiment/temporal_training.yaml --data_path /path/to/your/data
    python run_temporal_training.py --help
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import yaml
import torch


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查PyTorch
    try:
        print(f"✅ PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查必要的包
    required_packages = [
        ('hydra-core', 'hydra'), 
        ('omegaconf', 'omegaconf'), 
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'), 
        ('tensorboard', 'tensorboard'), 
        ('h5py', 'h5py'), 
        ('opencv-python', 'cv2')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True


def check_data_path(data_path: str) -> bool:
    """检查数据路径"""
    print(f"🔍 检查数据路径: {data_path}")
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"❌ 数据路径不存在: {data_path}")
        return False
    
    # 检查是否有HDF5文件
    h5_files = list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5"))
    if not h5_files:
        print(f"❌ 在 {data_path} 中未找到HDF5文件")
        return False
    
    print(f"✅ 找到 {len(h5_files)} 个HDF5文件")
    for f in h5_files[:3]:  # 显示前3个文件
        print(f"   - {f.name}")
    if len(h5_files) > 3:
        print(f"   ... 还有 {len(h5_files) - 3} 个文件")
    
    return True


def update_config(config_path: str, data_path: Optional[str] = None) -> str:
    """更新配置文件"""
    print(f"🔧 更新配置文件: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试其他编码
        with open(config_path, 'r', encoding='gbk') as f:
            config = yaml.safe_load(f)
    
    # 更新数据路径
    if data_path:
        config['data']['data_path'] = str(Path(data_path).absolute())
        print(f"   更新数据路径: {data_path}")
    
    # 确保输出目录存在
    output_dir = Path(config.get('experiment', {}).get('output_dir', 'runs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   输出目录: {output_dir}")
    
    # 保存更新后的配置
    temp_config_path = config_path.replace('.yaml', '_temp.yaml')
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"   临时配置文件: {temp_config_path}")
    
    return temp_config_path


def run_training(config_path: str, resume: bool = False):
    """运行训练"""
    print(f"🚀 开始训练...")
    print(f"   配置文件: {config_path}")
    
    # 构建命令
    cmd = [sys.executable, "train_temporal.py", f"--config-path={Path(config_path).parent}", 
           f"--config-name={Path(config_path).stem}"]
    
    if resume:
        cmd.append("resume=true")
    
    print(f"   执行命令: {' '.join(cmd)}")
    
    try:
        # 运行训练
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print("✅ 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return False


def show_results(output_dir: str):
    """显示结果"""
    print(f"📊 查看训练结果...")
    
    output_path = Path(output_dir)
    
    # 检查日志文件
    log_files = list(output_path.glob("*.log"))
    if log_files:
        print(f"✅ 日志文件: {log_files[0]}")
    
    # 检查检查点
    ckpt_files = list(output_path.glob("*.ckpt"))
    if ckpt_files:
        print(f"✅ 检查点文件: {len(ckpt_files)} 个")
    
    # 检查可视化结果
    vis_dir = output_path / "visualizations"
    if vis_dir.exists():
        print(f"✅ 可视化结果: {vis_dir}")
        
        # 统计文件数量
        training_plots = len(list((vis_dir / "training").glob("*.png")))
        result_plots = len(list((vis_dir / "results").glob("*.png")))
        animations = len(list((vis_dir / "animations").glob("*.gif")))
        
        print(f"   - 训练图表: {training_plots} 个")
        print(f"   - 结果图表: {result_plots} 个")
        print(f"   - 动画文件: {animations} 个")
    
    # 检查TensorBoard日志
    tb_dir = output_path / "tensorboard"
    if tb_dir.exists():
        print(f"✅ TensorBoard日志: {tb_dir}")
        print(f"   运行命令查看: tensorboard --logdir {tb_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="时序PDE训练一键运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置训练
  python run_temporal_training.py
  
  # 指定配置文件
  python run_temporal_training.py --config configs/experiment/temporal_training.yaml
  
  # 指定数据路径
  python run_temporal_training.py --data_path /path/to/your/pde/data
  
  # 从检查点恢复训练
  python run_temporal_training.py --resume
  
  # 只检查环境，不运行训练
  python run_temporal_training.py --check_only
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/experiment/temporal_training.yaml',
        help='配置文件路径 (默认: configs/experiment/temporal_training.yaml)'
    )
    
    parser.add_argument(
        '--data_path', 
        type=str,
        help='PDE数据集路径 (可选，会覆盖配置文件中的路径)'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='从最新检查点恢复训练'
    )
    
    parser.add_argument(
        '--check_only', 
        action='store_true',
        help='只检查环境和数据，不运行训练'
    )
    
    parser.add_argument(
        '--skip_env_check', 
        action='store_true',
        help='跳过环境检查'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌊 时序PDE训练系统")
    print("=" * 60)
    
    # 检查环境
    if not args.skip_env_check:
        if not check_environment():
            print("\n❌ 环境检查失败，请解决上述问题后重试")
            return 1
        print()
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 1
    
    # 检查数据路径
    if args.data_path:
        if not check_data_path(args.data_path):
            return 1
        print()
    
    # 如果只检查环境，则退出
    if args.check_only:
        print("✅ 环境检查完成")
        return 0
    
    # 更新配置
    try:
        temp_config_path = update_config(str(config_path), args.data_path)
        print()
        
        # 运行训练
        success = run_training(temp_config_path, args.resume)
        
        if success:
            # 显示结果
            with open(temp_config_path, 'r') as f:
                config = yaml.safe_load(f)
            show_results(config['output_dir'])
            
            print("\n" + "=" * 60)
            print("🎉 训练完成!")
            print("=" * 60)
        
        # 清理临时配置文件
        if Path(temp_config_path).exists():
            Path(temp_config_path).unlink()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())