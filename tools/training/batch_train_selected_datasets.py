#!/usr/bin/env python3
"""
自动生成的批量训练脚本
训练选中的 6 个数据集配置
生成时间: 2025-10-17T02:44:54.491838
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_training(config_path, experiment_name):
    """运行单个训练任务"""
    print(f"\n============================================================")
    print(f"🚀 开始训练: {experiment_name}")
    print(f"📁 配置文件: {config_path}")
    print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"============================================================")
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        f"--config-path={Path(config_path).parent}",
        f"--config-name={Path(config_path).stem}"
    ]
    
    try:
        # 执行训练
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\n✅ 训练完成: {experiment_name}")
        print(f"⏱️  训练时长: {duration/3600:.2f} 小时")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {experiment_name}")
        print(f"错误代码: {e.returncode}")
        return False, 0
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断: {experiment_name}")
        return False, 0

def main():
    """主函数"""
    print("🎯 开始批量训练选中的数据集")
    print(f"📊 总计 6 个训练任务")
    
    # 训练配置列表
    training_configs = [
        {
            "name": "ns_incom_inhom_2d_512-0_sr_x2",
            "config_path": "configs/auto_generated/ns_incom_inhom_2d_512_0_sr_x2_optimized.yaml",
            "dataset": "ns_incom_inhom_2d_512-0",
            "task_type": "sr_x2",
            "pde_type": "navier_stokes"
        },
        {
            "name": "ns_incom_inhom_2d_512-0_crop_20",
            "config_path": "configs/auto_generated/ns_incom_inhom_2d_512_0_crop_20_optimized.yaml",
            "dataset": "ns_incom_inhom_2d_512-0",
            "task_type": "crop_20",
            "pde_type": "navier_stokes"
        },
        {
            "name": "2D_diff-react_NA_NA_sr_x2",
            "config_path": "configs/auto_generated/2d_diff_react_na_na_sr_x2_optimized.yaml",
            "dataset": "2D_diff-react_NA_NA",
            "task_type": "sr_x2",
            "pde_type": "diffusion_reaction"
        },
        {
            "name": "2D_diff-react_NA_NA_crop_40",
            "config_path": "configs/auto_generated/2d_diff_react_na_na_crop_40_optimized.yaml",
            "dataset": "2D_diff-react_NA_NA",
            "task_type": "crop_40",
            "pde_type": "diffusion_reaction"
        },
        {
            "name": "2D_rdb_NA_NA_sr_x4",
            "config_path": "configs/auto_generated/2d_rdb_na_na_sr_x4_optimized.yaml",
            "dataset": "2D_rdb_NA_NA",
            "task_type": "sr_x4",
            "pde_type": "shallow_water"
        },
        {
            "name": "2D_rdb_NA_NA_crop_20",
            "config_path": "configs/auto_generated/2d_rdb_na_na_crop_20_optimized.yaml",
            "dataset": "2D_rdb_NA_NA",
            "task_type": "crop_20",
            "pde_type": "shallow_water"
        },
    ]
    
    # 执行训练
    successful_runs = 0
    failed_runs = 0
    total_time = 0
    
    for i, config in enumerate(training_configs, 1):
        print(f"\n📋 进度: {i}/6")
        print(f"🔬 PDE类型: {config['pde_type']}")
        print(f"📊 数据集: {config['dataset']}")
        print(f"🎯 任务: {config['task_type']}")
        
        success, duration = run_training(config['config_path'], config['name'])
        
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
        
        total_time += duration
        
        # 显示进度统计
        print(f"\n📈 当前统计:")
        print(f"  ✅ 成功: {successful_runs}")
        print(f"  ❌ 失败: {failed_runs}")
        print(f"  ⏱️  总时长: {total_time/3600:.2f} 小时")
    
    # 最终统计
    print(f"\n================================================================================")
    print("🎉 批量训练完成!")
    print(f"📊 最终统计:")
    print(f"  ✅ 成功训练: {successful_runs}/6")
    print(f"  ❌ 失败训练: {failed_runs}/6")
    print(f"  ⏱️  总训练时长: {total_time/3600:.2f} 小时")
    print(f"  📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"================================================================================")

if __name__ == "__main__":
    main()
