#!/usr/bin/env python3
"""监控批量训练状态"""

import os
import json
from pathlib import Path
from datetime import datetime

def monitor_training():
    """监控训练状态"""
    results_dir = Path('runs/batch_training_results')
    models = ['fno2d', 'hybrid', 'ufno_unet', 'unet', 'unet_plus_plus', 
              'segformer_unetformer', 'mlp', 'mlp_mixer', 'swin_unet', 
              'segformer', 'liif', 'unetformer']

    print(f'当前训练状态监控 ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
    print('=' * 60)

    for model in models:
        model_dir = results_dir / model
        log_file = model_dir / 'train.log'
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                if lines:
                    last_line = lines[-1].strip()
                    if 'Epoch' in last_line and 'Train Loss' in last_line:
                        epoch_info = last_line.split("Epoch")[1].split("-")[0].strip()
                        print(f'{model:20} - 训练中: Epoch {epoch_info}')
                    elif 'Training completed' in last_line:
                        print(f'{model:20} - 已完成')
                    elif 'ERROR' in last_line or 'failed' in last_line.lower():
                        error_msg = last_line[-50:] if len(last_line) > 50 else last_line
                        print(f'{model:20} - 失败: {error_msg}')
                    else:
                        status_msg = last_line[-30:] if len(last_line) > 30 else last_line
                        print(f'{model:20} - 状态未知: {status_msg}')
                else:
                    print(f'{model:20} - 日志为空')
            except Exception as e:
                print(f'{model:20} - 读取日志失败: {str(e)}')
        else:
            print(f'{model:20} - 无日志文件')

if __name__ == "__main__":
    monitor_training()