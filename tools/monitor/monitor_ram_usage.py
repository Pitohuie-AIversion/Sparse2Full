#!/usr/bin/env python3
"""
🚀 超极限RAM使用监控脚本
实时监控训练过程中的RAM使用情况
"""

import psutil
import time
import os
import subprocess
from datetime import datetime

def get_memory_info():
    """获取详细的内存信息"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_gb': mem.total / (1024**3),
        'used_gb': mem.used / (1024**3),
        'available_gb': mem.available / (1024**3),
        'percent': mem.percent,
        'cached_gb': getattr(mem, 'cached', 0) / (1024**3),
        'buffers_gb': getattr(mem, 'buffers', 0) / (1024**3),
        'swap_used_gb': swap.used / (1024**3),
        'swap_total_gb': swap.total / (1024**3)
    }

def get_gpu_memory():
    """获取GPU内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                used, total = map(int, line.split(', '))
                gpu_info.append({
                    'gpu_id': i,
                    'used_mb': used,
                    'total_mb': total,
                    'used_gb': used / 1024,
                    'total_gb': total / 1024,
                    'percent': (used / total) * 100
                })
            return gpu_info
    except:
        pass
    return []

def get_process_memory(process_name="python"):
    """获取特定进程的内存使用"""
    total_memory = 0
    process_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                total_memory += proc.info['memory_info'].rss
                process_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return {
        'total_gb': total_memory / (1024**3),
        'process_count': process_count
    }

def monitor_ram(interval=5, duration=300):
    """
    监控RAM使用情况
    
    Args:
        interval: 监控间隔（秒）
        duration: 监控持续时间（秒），0表示无限监控
    """
    print("🚀 超极限RAM使用监控启动")
    print("=" * 80)
    print(f"监控间隔: {interval}秒")
    print(f"监控时长: {'无限' if duration == 0 else f'{duration}秒'}")
    print("=" * 80)
    
    start_time = time.time()
    max_ram_usage = 0
    max_ram_time = None
    
    try:
        while True:
            current_time = time.time()
            if duration > 0 and (current_time - start_time) > duration:
                break
                
            # 获取系统信息
            mem_info = get_memory_info()
            gpu_info = get_gpu_memory()
            python_mem = get_process_memory("python")
            
            # 更新最大RAM使用记录
            if mem_info['percent'] > max_ram_usage:
                max_ram_usage = mem_info['percent']
                max_ram_time = datetime.now().strftime("%H:%M:%S")
            
            # 显示信息
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n⏰ {timestamp}")
            print(f"🧠 RAM: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB ({mem_info['percent']:.1f}%)")
            print(f"💾 可用: {mem_info['available_gb']:.1f}GB | 缓存: {mem_info['cached_gb']:.1f}GB")
            print(f"🐍 Python进程: {python_mem['total_gb']:.1f}GB ({python_mem['process_count']}个进程)")
            
            if gpu_info:
                for gpu in gpu_info:
                    print(f"🎮 GPU{gpu['gpu_id']}: {gpu['used_gb']:.1f}GB / {gpu['total_gb']:.1f}GB ({gpu['percent']:.1f}%)")
            
            print(f"📈 最大RAM使用: {max_ram_usage:.1f}% (时间: {max_ram_time or 'N/A'})")
            
            # RAM使用率状态指示
            if mem_info['percent'] < 50:
                status = "🟢 低使用率 - 可以进一步优化"
            elif mem_info['percent'] < 80:
                status = "🟡 中等使用率 - 优化效果显现"
            elif mem_info['percent'] < 95:
                status = "🟠 高使用率 - 优化目标达成"
            else:
                status = "🔴 极高使用率 - 注意监控稳定性"
            
            print(f"📊 状态: {status}")
            print("-" * 80)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 监控已停止")
        print(f"📈 最大RAM使用率: {max_ram_usage:.1f}% (时间: {max_ram_time or 'N/A'})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="🚀 超极限RAM使用监控")
    parser.add_argument("--interval", "-i", type=int, default=5, 
                       help="监控间隔（秒），默认5秒")
    parser.add_argument("--duration", "-d", type=int, default=0, 
                       help="监控持续时间（秒），0表示无限监控，默认0")
    
    args = parser.parse_args()
    
    monitor_ram(args.interval, args.duration)