#!/usr/bin/env python3
"""
训练产物验证脚本
验证训练产物是否符合技术架构文档要求
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any

def validate_training_output(output_dir: Path) -> Dict[str, Any]:
    """验证训练产物是否符合规范"""
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": []
    }
    
    # 1. 检查基本目录结构
    required_dirs = ["checkpoints", "logs", "metrics", "visualizations"]
    for dir_name in required_dirs:
        dir_path = output_dir / dir_name
        if not dir_path.exists():
            results["errors"].append(f"缺少必要目录: {dir_name}")
            results["valid"] = False
        else:
            results["info"].append(f"✓ 目录存在: {dir_name}")
    
    # 2. 检查配置文件
    config_file = output_dir / "config_merged.yaml"
    if not config_file.exists():
        results["errors"].append("缺少配置文件: config_merged.yaml")
        results["valid"] = False
    else:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # 检查必要配置项
            required_config_keys = ["experiment", "model", "training"]
            for key in required_config_keys:
                if key not in config:
                    results["errors"].append(f"配置缺少必要键: {key}")
                    results["valid"] = False
                else:
                    results["info"].append(f"✓ 配置键存在: {key}")
                    
        except Exception as e:
            results["errors"].append(f"配置文件解析错误: {e}")
            results["valid"] = False
    
    # 3. 检查检查点文件
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        if not checkpoint_files:
            results["warnings"].append("检查点目录为空")
        else:
            for ckpt_file in checkpoint_files:
                results["info"].append(f"✓ 检查点文件: {ckpt_file.name}")
    
    # 4. 检查指标文件
    metrics_dir = output_dir / "metrics"
    if metrics_dir.exists():
        metrics_files = list(metrics_dir.glob("*.json"))
        if not metrics_files:
            results["warnings"].append("指标目录为空")
        else:
            for metrics_file in metrics_files:
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # 检查必要指标
                    required_metrics = ["epoch", "train_loss", "val_loss"]
                    for metric in required_metrics:
                        if metric not in metrics:
                            results["warnings"].append(f"指标文件缺少: {metric}")
                        else:
                            results["info"].append(f"✓ 指标存在: {metric}")
                            
                except Exception as e:
                    results["errors"].append(f"指标文件解析错误: {e}")
                    results["valid"] = False
    
    # 5. 检查日志文件
    logs_dir = output_dir / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if not log_files:
            results["warnings"].append("日志目录为空")
        else:
            for log_file in log_files:
                results["info"].append(f"✓ 日志文件: {log_file.name}")
    
    # 6. 检查可视化文件
    viz_dir = output_dir / "visualizations"
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*"))
        if not viz_files:
            results["warnings"].append("可视化目录为空")
        else:
            for viz_file in viz_files:
                results["info"].append(f"✓ 可视化文件: {viz_file.name}")
    
    return results

def validate_paper_package(paper_dir: Path) -> Dict[str, Any]:
    """验证paper包结构"""
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": []
    }
    
    # 检查paper包目录结构
    required_paper_dirs = ["configs", "checkpoints", "metrics", "figs", "data_cards"]
    for dir_name in required_paper_dirs:
        dir_path = paper_dir / dir_name
        if not dir_path.exists():
            results["errors"].append(f"Paper包缺少目录: {dir_name}")
            results["valid"] = False
        else:
            results["info"].append(f"✓ Paper目录存在: {dir_name}")
    
    # 检查元数据文件
    meta_file = paper_dir / "package_meta.json"
    if not meta_file.exists():
        results["warnings"].append("缺少paper包元数据文件")
    else:
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            required_meta_keys = ["experiment_name", "model_type", "dataset", "task"]
            for key in required_meta_keys:
                if key not in meta:
                    results["warnings"].append(f"元数据缺少键: {key}")
                else:
                    results["info"].append(f"✓ 元数据键存在: {key}")
                    
        except Exception as e:
            results["errors"].append(f"元数据文件解析错误: {e}")
            results["valid"] = False
    
    return results

def main():
    """主函数"""
    
    print("=" * 70)
    print("训练产物验证报告")
    print("=" * 70)
    
    # 找到最新的训练输出目录
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("❌ 找不到runs目录")
        return 1
    
    # 获取最新的训练目录
    train_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not train_dirs:
        print("❌ 找不到训练输出目录")
        return 1
    
    latest_train_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 验证目录: {latest_train_dir}")
    
    # 验证训练输出
    print("\n" + "-" * 50)
    print("1. 训练输出验证")
    print("-" * 50)
    
    train_results = validate_training_output(latest_train_dir)
    
    if train_results["info"]:
        print("✅ 成功项:")
        for info in train_results["info"]:
            print(f"  {info}")
    
    if train_results["warnings"]:
        print("⚠️  警告项:")
        for warning in train_results["warnings"]:
            print(f"  {warning}")
    
    if train_results["errors"]:
        print("❌ 错误项:")
        for error in train_results["errors"]:
            print(f"  {error}")
    
    # 验证paper包
    print("\n" + "-" * 50)
    print("2. Paper包验证")
    print("-" * 50)
    
    paper_dir = Path("paper_package")
    if paper_dir.exists():
        paper_results = validate_paper_package(paper_dir)
        
        if paper_results["info"]:
            print("✅ Paper包成功项:")
            for info in paper_results["info"]:
                print(f"  {info}")
        
        if paper_results["warnings"]:
            print("⚠️  Paper包警告项:")
            for warning in paper_results["warnings"]:
                print(f"  {warning}")
        
        if paper_results["errors"]:
            print("❌ Paper包错误项:")
            for error in paper_results["errors"]:
                print(f"  {error}")
    else:
        print("⚠️  Paper包目录不存在")
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 验证总结")
    print("=" * 70)
    
    if train_results["valid"] and not train_results["errors"]:
        print("✅ 训练输出验证通过!")
    else:
        print("❌ 训练输出验证失败!")
    
    if paper_dir.exists() and paper_results["valid"]:
        print("✅ Paper包验证通过!")
    elif paper_dir.exists():
        print("⚠️  Paper包存在一些问题")
    
    print(f"\n📋 技术架构符合性:")
    print(f"  - 配置文件快照: {'✅' if (latest_train_dir / 'config_merged.yaml').exists() else '❌'}")
    print(f"  - 检查点管理: {'✅' if (latest_train_dir / 'checkpoints').exists() else '❌'}")
    print(f"  - 指标记录: {'✅' if (latest_train_dir / 'metrics').exists() else '❌'}")
    print(f"  - 日志系统: {'✅' if (latest_train_dir / 'logs').exists() else '❌'}")
    print(f"  - 可视化输出: {'✅' if (latest_train_dir / 'visualizations').exists() else '❌'}")
    print(f"  - Paper包结构: {'✅' if paper_dir.exists() else '❌'}")
    
    return 0 if train_results["valid"] else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())