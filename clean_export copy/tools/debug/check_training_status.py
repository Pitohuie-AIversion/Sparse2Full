import psutil
import os
import glob
from datetime import datetime, timedelta

# 检查进程19728的详细信息
try:
    p = psutil.Process(19728)
    print(f'进程名: {p.name()}')
    print(f'命令行: {" ".join(p.cmdline())}')
    print(f'工作目录: {p.cwd()}')
    print(f'CPU使用率: {p.cpu_percent()}%')
    print(f'内存使用: {p.memory_info().rss / 1024 / 1024:.1f} MB')
    print(f'运行时间: {datetime.fromtimestamp(p.create_time())}')
    print(f'状态: {p.status()}')
    
    # 检查是否有打开的文件
    try:
        open_files = p.open_files()
        print(f'\n打开的文件数量: {len(open_files)}')
        
        # 查找可能的日志文件
        log_files = [f for f in open_files if 'log' in f.path.lower() or 'runs' in f.path.lower()]
        if log_files:
            print('\n相关日志文件:')
            for f in log_files[:5]:
                print(f'  {f.path}')
    except psutil.AccessDenied:
        print('\n无法访问进程打开的文件列表')
    
except psutil.NoSuchProcess:
    print('进程19728不存在')
except Exception as e:
    print(f'错误: {e}')

# 查找最近的运行目录
print('\n' + '='*50)
print('查找最近的训练运行:')

runs_dir = 'F:/Zhaoyang/Sparse2Full/runs'
now = datetime.now()
recent_cutoff = now - timedelta(hours=12)

recent_dirs = []
for item in os.listdir(runs_dir):
    item_path = os.path.join(runs_dir, item)
    if os.path.isdir(item_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
        if mtime > recent_cutoff:
            recent_dirs.append((item, mtime, item_path))

recent_dirs.sort(key=lambda x: x[1], reverse=True)

print(f'最近12小时内修改的运行目录 ({len(recent_dirs)} 个):')
for name, mtime, path in recent_dirs[:10]:
    print(f'{name}: {mtime}')
    
    # 检查是否有训练日志
    log_path = os.path.join(path, 'train.log')
    if os.path.exists(log_path):
        log_size = os.path.getsize(log_path)
        log_mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
        print(f'  -> train.log: {log_size} bytes, 修改时间: {log_mtime}')
    
    # 检查checkpoints目录
    ckpt_dir = os.path.join(path, 'checkpoints')
    if os.path.exists(ckpt_dir):
        ckpts = os.listdir(ckpt_dir)
        print(f'  -> checkpoints: {len(ckpts)} 个文件')