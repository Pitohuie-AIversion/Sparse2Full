#!/usr/bin/env python3
"""
彻底修复AR可视化器中的图像形状问题
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_error_map():
    """修复error_map处理逻辑"""
    
    file_path = PROJECT_ROOT / "utils/ar_visualizer.py"
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换error_map处理逻辑
    old_code = '''            error_map = errors[-1]  # 获取最后时间步 [C, H, W]
            # 确保取第一个通道
            if error_map.ndim == 3:  # [C, H, W]
                error_map = error_map[0]  # 取第一个通道 [H, W]
            # 确保是2D数组
            if error_map.ndim > 2:
                error_map = error_map.squeeze()
            elif error_map.ndim == 1:'''
    
    new_code = '''            # errors 形状可能是 [T, C, H, W] 或 [C, H, W]
            if errors.ndim == 4:  # [T, C, H, W]
                error_map = errors[-1, 0]  # 获取最后时间步的第一个通道 [H, W]
            elif errors.ndim == 3:  # [C, H, W]
                error_map = errors[0]  # 获取第一个通道 [H, W]
            else:  # [H, W]
                error_map = errors
            
            # 确保是2D数组
            while error_map.ndim > 2:
                error_map = error_map[0]  # 继续取第一个元素直到2D
            
            if error_map.ndim == 1:'''
    
    # 替换内容
    if old_code in content:
        content = content.replace(old_code, new_code)
        print("✅ 找到并替换了error_map处理逻辑")
    else:
        print("❌ 未找到目标代码段")
        return False
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ AR可视化器修复完成")
    return True

if __name__ == "__main__":
    fix_error_map()
