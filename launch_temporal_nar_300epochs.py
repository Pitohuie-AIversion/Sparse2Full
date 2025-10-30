#!/usr/bin/env python3
"""
时序NAR模型300轮训练启动脚本
提供便捷的训练启动、监控和管理功能
"""

import os
import sys
import argparse
import subprocess
import time
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import psutil
import torch


class TemporalNARLauncher:
    """时序NAR训练启动器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_path = "configs/experiment/temporal_nar_300epochs.yaml"
        self.training_script = "train_temporal_nar_300epochs.py"
        self.process = None
        
    def check_environment(self) -> Dict[str, Any]:
        """检查训练环境"""
        env_info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
        }
        
        # 修复Windows下的磁盘使用率检查
        try:
            disk_usage = psutil.disk_usage(os.getcwd())
            env_info['disk_free'] = disk_usage.free / (1024**3)  # GB
        except Exception as e:
            # 如果获取磁盘使用率失败，设置默认值
            env_info['disk_free'] = 100.0  # 假设有100GB可用空间
        
        if torch.cuda.is_available():
            env_info['gpu_info'] = []
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                env_info['gpu_info'].append({
                    'name': gpu_props.name,
                    'memory_total': gpu_props.total_memory / (1024**3),  # GB
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                })
        
        return env_info
    
    def print_environment_info(self):
        """打印环境信息"""
        env_info = self.check_environment()
        
        print("=" * 60)
        print("🚀 时序NAR模型300轮训练启动器")
        print("=" * 60)
        print(f"📍 项目路径: {self.project_root}")
        print(f"🐍 Python版本: {env_info['python_version'].split()[0]}")
        print(f"🔥 PyTorch版本: {env_info['pytorch_version']}")
        
        if env_info['cuda_available']:
            print(f"⚡ CUDA版本: {env_info['cuda_version']}")
            print(f"🎮 GPU数量: {env_info['gpu_count']}")
            for i, gpu in enumerate(env_info['gpu_info']):
                print(f"   GPU {i}: {gpu['name']} ({gpu['memory_total']:.1f}GB)")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")
        
        print(f"💾 系统内存: {env_info['memory_available']:.1f}GB / {env_info['memory_total']:.1f}GB")
        print(f"💿 磁盘空间: {env_info['disk_free']:.1f}GB")
        print("=" * 60)
    
    def check_prerequisites(self) -> bool:
        """检查训练前置条件"""
        print("🔍 检查训练前置条件...")
        
        # 检查配置文件
        config_file = self.project_root / self.config_path
        if not config_file.exists():
            print(f"❌ 配置文件不存在: {config_file}")
            return False
        print(f"✅ 配置文件: {config_file}")
        
        # 检查训练脚本
        script_file = self.project_root / self.training_script
        if not script_file.exists():
            print(f"❌ 训练脚本不存在: {script_file}")
            return False
        print(f"✅ 训练脚本: {script_file}")
        
        # 检查数据路径（从配置文件读取）
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            data_path = config.get('data', {}).get('data_path')
            if data_path and not Path(data_path).exists():
                print(f"⚠️  数据文件不存在: {data_path}")
                print("   请确保数据文件路径正确")
        except Exception as e:
            print(f"⚠️  无法验证数据路径: {e}")
        
        # 检查GPU内存
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                # 尝试分配一些内存来检查GPU状态
                test_tensor = torch.randn(1000, 1000, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                print("✅ GPU内存检查通过")
            except Exception as e:
                print(f"⚠️  GPU内存检查失败: {e}")
        
        # 检查磁盘空间（至少需要10GB）
        env_info = self.check_environment()
        if env_info['disk_free'] < 10:
            print(f"⚠️  磁盘空间不足: {env_info['disk_free']:.1f}GB < 10GB")
            print("   建议清理磁盘空间或更改输出目录")
        else:
            print(f"✅ 磁盘空间充足: {env_info['disk_free']:.1f}GB")
        
        print("✅ 前置条件检查完成")
        return True
    
    def create_launch_command(self, **kwargs) -> list:
        """创建启动命令"""
        cmd = [sys.executable, self.training_script]
        
        # 添加Hydra覆盖参数
        if kwargs.get('resume'):
            cmd.extend(['resume.auto_resume=true'])
        
        if kwargs.get('debug'):
            cmd.extend(['debug.enabled=true'])
        
        if kwargs.get('quick_test'):
            cmd.extend(['experiment.quick_test=true'])
        
        if kwargs.get('device'):
            cmd.extend([f'experiment.device={kwargs["device"]}'])
        
        if kwargs.get('batch_size'):
            cmd.extend([f'data.batch_size={kwargs["batch_size"]}'])
        
        if kwargs.get('lr'):
            cmd.extend([f'train.optimizer.lr={kwargs["lr"]}'])
        
        if kwargs.get('max_epochs'):
            cmd.extend([f'train.max_epochs={kwargs["max_epochs"]}'])
        
        return cmd
    
    def start_training(self, **kwargs):
        """启动训练"""
        print("🚀 启动训练...")
        
        # 创建启动命令
        cmd = self.create_launch_command(**kwargs)
        print(f"📝 执行命令: {' '.join(cmd)}")
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root)
        
        # 启动训练进程
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"✅ 训练进程已启动 (PID: {self.process.pid})")
            print("📊 实时日志输出:")
            print("-" * 60)
            
            # 实时输出日志
            for line in iter(self.process.stdout.readline, ''):
                print(line.rstrip())
                
                # 检查进程是否结束
                if self.process.poll() is not None:
                    break
            
            # 等待进程结束
            return_code = self.process.wait()
            
            if return_code == 0:
                print("✅ 训练成功完成!")
            else:
                print(f"❌ 训练异常结束 (返回码: {return_code})")
            
            return return_code
            
        except KeyboardInterrupt:
            print("\n⚠️  收到中断信号，正在停止训练...")
            self.stop_training()
            return -1
        except Exception as e:
            print(f"❌ 启动训练失败: {e}")
            return -1
    
    def stop_training(self):
        """停止训练"""
        if self.process and self.process.poll() is None:
            print("🛑 正在停止训练进程...")
            
            try:
                # 发送SIGTERM信号
                self.process.terminate()
                
                # 等待进程结束（最多10秒）
                try:
                    self.process.wait(timeout=10)
                    print("✅ 训练进程已正常停止")
                except subprocess.TimeoutExpired:
                    # 强制杀死进程
                    print("⚠️  进程未响应，强制终止...")
                    self.process.kill()
                    self.process.wait()
                    print("✅ 训练进程已强制终止")
                    
            except Exception as e:
                print(f"❌ 停止训练进程失败: {e}")
    
    def monitor_training(self, output_dir: str):
        """监控训练进度"""
        output_path = Path(output_dir)
        
        print(f"📊 监控训练进度: {output_path}")
        
        # 监控文件
        log_file = output_path / "train.log"
        history_file = output_path / "training_history.json"
        
        last_log_size = 0
        last_history_time = 0
        
        try:
            while True:
                # 监控日志文件
                if log_file.exists():
                    current_size = log_file.stat().st_size
                    if current_size > last_log_size:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            f.seek(last_log_size)
                            new_content = f.read()
                            if new_content.strip():
                                print("📝 新日志:")
                                print(new_content.rstrip())
                        last_log_size = current_size
                
                # 监控训练历史
                if history_file.exists():
                    current_time = history_file.stat().st_mtime
                    if current_time > last_history_time:
                        try:
                            with open(history_file, 'r') as f:
                                history = json.load(f)
                            
                            if history.get('train_losses'):
                                latest_loss = history['train_losses'][-1]
                                epoch = len(history['train_losses'])
                                print(f"📈 Epoch {epoch}: Loss = {latest_loss:.6f}")
                            
                            last_history_time = current_time
                        except Exception:
                            pass
                
                time.sleep(5)  # 每5秒检查一次
                
        except KeyboardInterrupt:
            print("\n⚠️  停止监控")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="时序NAR模型300轮训练启动器")
    
    # 基本参数
    parser.add_argument('--action', choices=['train', 'monitor', 'stop'], default='train',
                       help='执行动作: train(训练), monitor(监控), stop(停止)')
    
    # 训练参数
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--quick-test', action='store_true', help='快速测试')
    parser.add_argument('--device', default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--max-epochs', type=int, help='最大训练轮数')
    
    # 监控参数
    parser.add_argument('--output-dir', default='runs/temporal_nar_300epochs',
                       help='输出目录 (用于监控)')
    
    args = parser.parse_args()
    
    # 创建启动器
    launcher = TemporalNARLauncher()
    
    if args.action == 'train':
        # 打印环境信息
        launcher.print_environment_info()
        
        # 检查前置条件
        if not launcher.check_prerequisites():
            print("❌ 前置条件检查失败，请解决问题后重试")
            return 1
        
        # 确认启动
        if not args.quick_test:
            response = input("\n🤔 确认启动300轮训练? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ 用户取消训练")
                return 0
        
        # 启动训练
        kwargs = {
            'resume': args.resume,
            'debug': args.debug,
            'quick_test': args.quick_test,
            'device': args.device,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'max_epochs': args.max_epochs
        }
        
        return_code = launcher.start_training(**kwargs)
        return return_code
        
    elif args.action == 'monitor':
        # 监控训练
        launcher.monitor_training(args.output_dir)
        return 0
        
    elif args.action == 'stop':
        # 停止训练
        print("🛑 尝试停止训练进程...")
        
        # 查找训练进程
        found_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'train_temporal_nar_300epochs.py' in ' '.join(proc.info['cmdline']):
                    found_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not found_processes:
            print("❌ 未找到运行中的训练进程")
            return 1
        
        for proc in found_processes:
            try:
                print(f"🛑 停止进程 {proc.pid}: {proc.info['name']}")
                proc.terminate()
                proc.wait(timeout=10)
                print(f"✅ 进程 {proc.pid} 已停止")
            except psutil.TimeoutExpired:
                print(f"⚠️  进程 {proc.pid} 未响应，强制终止")
                proc.kill()
            except Exception as e:
                print(f"❌ 停止进程 {proc.pid} 失败: {e}")
        
        return 0


if __name__ == "__main__":
    # 设置信号处理
    def signal_handler(signum, frame):
        print("\n⚠️  收到中断信号，正在退出...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行主函数
    exit_code = main()
    sys.exit(exit_code)