#!/usr/bin/env python3
"""
简化批量训练脚本 - PDEBench稀疏观测重建系统
直接调用训练命令，逐个训练所有模型
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
os.chdir(project_root)

def train_model(model_name, seed=2025, epochs=15):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"开始训练模型: {model_name} (种子: {seed})")
    print(f"{'='*60}")
    
    # 生成实验名称
    timestamp = datetime.now().strftime("%Y%m%d")
    exp_name = f"SRx4-DarcyFlow-128-{model_name.upper()}-batch-s{seed}-{timestamp}"
    
    # 构建训练命令 - 使用正确的Hydra覆盖语法
    cmd = [
        "F:\\ProgramData\\anaconda3\\python.exe", "train.py",
        f"+model={model_name}",  # 使用+覆盖默认配置
        f"trainer.max_epochs={epochs}",
        f"data.batch_size=2",
        f"trainer.seed={seed}",
        f"experiment.name={exp_name}"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 执行训练
        result = subprocess.run(cmd, check=True)
        train_time = time.time() - start_time
        
        print(f"✅ 模型 {model_name} 训练成功！耗时: {train_time:.2f}s")
        return {
            "model": model_name,
            "seed": seed,
            "exp_name": exp_name,
            "status": "success",
            "train_time": train_time
        }
        
    except subprocess.CalledProcessError as e:
        train_time = time.time() - start_time
        print(f"❌ 模型 {model_name} 训练失败！错误码: {e.returncode}")
        return {
            "model": model_name,
            "seed": seed,
            "exp_name": exp_name,
            "status": "failed",
            "train_time": train_time,
            "error_code": e.returncode
        }
    except Exception as e:
        train_time = time.time() - start_time
        print(f"❌ 模型 {model_name} 训练异常: {str(e)}")
        return {
            "model": model_name,
            "seed": seed,
            "exp_name": exp_name,
            "status": "error",
            "train_time": train_time,
            "error": str(e)
        }

def main():
    """主函数"""
    # 所有要训练的模型
    models = [
        "unet", "unet_plus_plus", "fno2d", "ufno_unet",
        "segformer_unetformer", "unetformer", "mlp", "mlp_mixer",
        "liif", "hybrid", "segformer"
        # 注意：swin_unet 已经训练过了，所以不包含在列表中
    ]
    
    print("PDEBench批量训练开始")
    print(f"计划训练模型: {len(models)} 个")
    print(f"模型列表: {', '.join(models)}")
    
    # 创建结果目录
    results_dir = Path("runs/batch_training_logs")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储结果
    results = []
    total_start_time = time.time()
    
    # 逐个训练模型
    for i, model_name in enumerate(models, 1):
        print(f"\n进度: {i}/{len(models)}")
        
        # 训练模型
        result = train_model(model_name, seed=2025, epochs=15)
        results.append(result)
        
        # 保存中间结果
        with open(results_dir / "intermediate_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 短暂休息
        time.sleep(2)
    
    # 计算总时间
    total_time = time.time() - total_start_time
    
    # 生成最终报告
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] != 'success'])
    
    print(f"\n{'='*60}")
    print("批量训练完成！")
    print(f"{'='*60}")
    print(f"总模型数: {len(models)}")
    print(f"成功训练: {successful}")
    print(f"训练失败: {failed}")
    print(f"成功率: {successful/len(models)*100:.1f}%")
    print(f"总耗时: {total_time:.2f}s ({total_time/3600:.2f}h)")
    
    # 保存最终结果
    final_results = {
        "summary": {
            "total_models": len(models),
            "successful": successful,
            "failed": failed,
            "success_rate": successful/len(models)*100,
            "total_time": total_time,
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }
    
    with open(results_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 显示失败的模型
    if failed > 0:
        print(f"\n失败的模型:")
        for result in results:
            if result['status'] != 'success':
                print(f"  - {result['model']}: {result['status']}")
    
    print(f"\n详细结果已保存到: {results_dir}")

if __name__ == "__main__":
    main()