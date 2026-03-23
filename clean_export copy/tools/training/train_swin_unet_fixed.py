#!/usr/bin/env python3
"""SwinUNet修复后的训练脚本"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train import main

if __name__ == "__main__":
    # 使用修复后的配置文件
    config_path = "configs/fixed/swin_unet_fixed.yaml"
    
    # 设置命令行参数 - 使用正确的Hydra格式
    sys.argv = [
        "train_swin_unet_fixed.py",
        "--config-name", "swin_unet_fixed",
        "--config-path", "configs/fixed",
        "device=cuda:0"
    ]
    
    print(f"开始训练SwinUNet（修复版本）")
    print(f"配置文件: {config_path}")
    print("=" * 60)
    
    # 启动训练
    main()
