#!/usr/bin/env python3
"""
修复AR可视化器中的图像形状问题
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_ar_visualizer():
    """修复ar_visualizer.py中的图像形状问题"""
    
    file_path = PROJECT_ROOT / "utils/ar_visualizer.py"
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: error_map处理
    old_pattern1 = r'error_map = errors\[-1, 0\]  # 获取最后时间步的第一个通道'
    new_pattern1 = '''error_map = errors[-1]  # 获取最后时间步 [C, H, W]
            # 确保取第一个通道
            if error_map.ndim == 3:  # [C, H, W]
                error_map = error_map[0]  # 取第一个通道 [H, W]'''
    
    content = re.sub(old_pattern1, new_pattern1, content)
    
    # 修复2: spatial_error处理
    old_pattern2 = r'spatial_error = np\.mean\(errors, axis=0\)\[0\]  # 平均所有时间步，取第一个通道'
    new_pattern2 = '''spatial_error = np.mean(errors, axis=0)  # 平均所有时间步 [C, H, W]
            # 确保取第一个通道
            if spatial_error.ndim == 3:  # [C, H, W]
                spatial_error = spatial_error[0]  # 取第一个通道 [H, W]'''
    
    content = re.sub(old_pattern2, new_pattern2, content)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ AR可视化器修复完成")

if __name__ == "__main__":
    fix_ar_visualizer()