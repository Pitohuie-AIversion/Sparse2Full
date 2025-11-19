#!/usr/bin/env python3
"""
横向对比系统快速测试脚本

测试批量对比系统的基本功能，使用少量模型和种子进行快速验证。
"""

import subprocess
import sys
import time
from pathlib import Path

def run_quick_test():
    """运行快速测试"""
    print("🚀 开始横向对比系统快速测试")
    print("="*60)
    
    # 测试配置
    config_path = "configs/train/ar_training_config debug.yaml"
    models = ["swin_unet", "unet", "fno2d"]  # 选择3个代表性模型
    seeds = ["42,123"]  # 使用2个种子进行快速测试
    output_dir = "paper_package/quick_test"
    
    print(f"配置文件: {config_path}")
    print(f"测试模型: {models}")
    print(f"随机种子: {seeds}")
    print(f"输出目录: {output_dir}")
    print()
    
    try:
        # 1. 运行批量对比实验
        print("步骤1: 运行批量对比实验...")
        cmd = [
            sys.executable, "tools/batch_comparison.py",
            "--config", config_path,
            "--models"] + models + [
            "--seeds", seeds[0],
            "--output", output_dir
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode != 0:
            print(f"❌ 批量对比实验失败:")
            print(f"返回码: {result.returncode}")
            print(f"错误输出: {result.stderr[-500:]}")
            return False
        
        print("✅ 批量对比实验完成")
        print()
        
        # 2. 等待实验结果生成
        print("步骤2: 等待实验结果生成...")
        time.sleep(5)  # 给系统一些时间生成结果
        
        # 3. 运行增强版汇总
        print("步骤3: 运行增强版汇总...")
        summarize_cmd = [
            sys.executable, "tools/enhanced_summarize.py",
            "--runs_dir", "runs/",
            "--output", f"{output_dir}/summary",
            "--baseline_method", "unet",
            "--verbose"
        ]
        
        print(f"执行命令: {' '.join(summarize_cmd)}")
        summarize_result = subprocess.run(summarize_cmd, capture_output=True, text=True)
        
        if summarize_result.returncode != 0:
            print(f"⚠️  增强版汇总失败，尝试标准汇总:")
            standard_summarize_cmd = [
                sys.executable, "tools/summarize_runs.py",
                "--runs_dir", "runs/",
                "--output", f"{output_dir}/summary",
                "--baseline_method", "unet"
            ]
            
            standard_result = subprocess.run(standard_summarize_cmd, capture_output=True, text=True)
            if standard_result.returncode != 0:
                print(f"❌ 汇总也失败: {standard_result.stderr[-300:]}")
                return False
            else:
                print("✅ 标准汇总完成")
        else:
            print("✅ 增强版汇总完成")
        
        print()
        
        # 4. 检查输出文件
        print("步骤4: 检查输出文件...")
        output_path = Path(output_dir)
        expected_files = [
            "batch_results.json",
            "batch_comparison.log"
        ]
        
        summary_path = output_path / "summary"
        expected_summary_files = [
            "comparison_report.md",
            "comparison_results.csv",
            "main_table.tex"
        ]
        
        all_files_exist = True
        
        for file in expected_files:
            file_path = output_path / file
            if file_path.exists():
                print(f"✅ {file}")
            else:
                print(f"❌ {file} (不存在)")
                all_files_exist = False
        
        for file in expected_summary_files:
            file_path = summary_path / file
            if file_path.exists():
                print(f"✅ summary/{file}")
            else:
                print(f"❌ summary/{file} (不存在)")
                all_files_exist = False
        
        print()
        
        # 5. 显示关键结果
        print("步骤5: 显示关键结果...")
        try:
            # 读取批量实验结果
            batch_results_file = output_path / "batch_results.json"
            if batch_results_file.exists():
                import json
                with open(batch_results_file, 'r') as f:
                    results = json.load(f)
                
                summary = results.get('summary', {})
                print(f"实验统计:")
                print(f"  总实验数: {summary.get('total', 0)}")
                print(f"  成功实验: {summary.get('successful', 0)}")
                print(f"  失败实验: {summary.get('failed', 0)}")
                print(f"  总用时: {summary.get('total_time', 0)/60:.1f} 分钟")
                
                # 显示实验详情
                experiments = results.get('experiments', [])
                if experiments:
                    print(f"\n实验详情:")
                    for exp in experiments:
                        status = "✅" if exp.get('success') else "❌"
                        print(f"  {status} {exp.get('model', 'unknown')}_s{exp.get('seed', 'unknown')} "
                              f"({exp.get('elapsed_time', 0):.1f}s)")
        except Exception as e:
            print(f"⚠️  读取结果失败: {e}")
        
        print()
        print("="*60)
        
        if all_files_exist and summary.get('successful', 0) > 0:
            print("✅ 横向对比系统测试通过!")
            print(f"📊 结果保存在: {output_dir}/")
            return True
        else:
            print("❌ 横向对比系统测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程异常: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)