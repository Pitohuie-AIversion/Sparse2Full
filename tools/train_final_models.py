#!/usr/bin/env python3
"""
最终批量训练脚本 - 修复所有已知问题后的版本

修复的问题：
1. FNO2d: 禁用AMP以避免复数操作问题
2. SegFormer/UNetFormer: 修复BaseModel继承问题
3. SegFormerUNetFormer: 修复注意力头数问题
4. MLPMixer: 修复mlp_ratio类型问题
5. LIIF: 创建缺失的模块
6. Hybrid: 修复输入通道问题
7. MLP: 跳过有patch处理问题的模型
"""

import subprocess
import sys
import time
import json
import os
from datetime import datetime
from pathlib import Path
import re


def parse_training_output(output_lines):
    """解析训练输出，提取关键指标"""
    metrics = {
        'training_time': None,
        'final_train_loss': None,
        'final_val_loss': None,
        'final_rel_l2': None,
        'epochs_completed': None
    }
    
    for line in output_lines:
        # 提取训练时间
        if 'Training completed in' in line:
            time_match = re.search(r'Training completed in ([\d.]+) seconds', line)
            if time_match:
                metrics['training_time'] = float(time_match.group(1))
        
        # 提取最终损失
        if 'Final metrics:' in line:
            # 下一行通常包含指标
            continue
        
        # 提取Rel-L2
        if 'Rel-L2:' in line:
            rel_l2_match = re.search(r'Rel-L2:\s*([\d.]+)', line)
            if rel_l2_match:
                metrics['final_rel_l2'] = float(rel_l2_match.group(1))
        
        # 提取epoch信息
        if 'Epoch' in line and '/' in line:
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                metrics['epochs_completed'] = int(epoch_match.group(1))
    
    return metrics


def run_training(model_name, epochs=15, batch_size=2, seed=2025):
    """运行单个模型的训练"""
    print(f"\n{'='*60}")
    print(f"开始训练模型: {model_name.upper()}")
    print(f"{'='*60}")
    
    # 构建训练命令
    cmd = [
        r"F:\ProgramData\anaconda3\python.exe", "train.py",
        f"+model={model_name}",
        f"+train.epochs={epochs}",
        f"+dataloader.batch_size={batch_size}",
        f"experiment.seed={seed}",
        f"experiment.name=SRx4-DarcyFlow-128-{model_name.upper()}-final-s{seed}-{datetime.now().strftime('%Y%m%d')}"
    ]
    
    # 对于FNO2d模型，禁用AMP
    if model_name == 'fno2d':
        cmd.append("+train.use_amp=false")
        print(f"禁用AMP for {model_name} (复数操作兼容性)")
    
    print(f"执行命令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # 运行训练
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30分钟超时
            cwd=Path(__file__).parent.parent
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✓ {model_name} 训练成功! 用时: {training_time:.1f}秒")
            
            # 解析输出获取指标
            output_lines = result.stdout.split('\n') + result.stderr.split('\n')
            metrics = parse_training_output(output_lines)
            metrics['training_time'] = training_time
            metrics['status'] = 'success'
            
            return True, metrics, None
        else:
            print(f"✗ {model_name} 训练失败!")
            print("错误输出:")
            print(result.stderr)
            
            return False, {'status': 'failed', 'training_time': training_time}, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"✗ {model_name} 训练超时 (30分钟)")
        return False, {'status': 'timeout', 'training_time': 1800}, "Training timeout"
    except Exception as e:
        print(f"✗ {model_name} 训练出现异常: {str(e)}")
        return False, {'status': 'error', 'training_time': 0}, str(e)


def save_results(results, output_dir):
    """保存训练结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON结果
    json_file = output_dir / f"training_results_final_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    md_file = output_dir / f"training_summary_final_{timestamp}.md"
    
    successful_models = [r for r in results['models'] if r['success']]
    failed_models = [r for r in results['models'] if not r['success']]
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 最终批量训练结果报告\n\n")
        f.write(f"**生成时间**: {results['timestamp']}\n\n")
        
        f.write("## 训练概况\n\n")
        f.write(f"- 总模型数: {results['total_models']}\n")
        f.write(f"- 成功训练: {results['successful_models']}\n")
        f.write(f"- 失败训练: {results['failed_models']}\n")
        f.write(f"- 成功率: {results['success_rate']:.1f}%\n\n")
        
        if successful_models:
            f.write("## 成功训练的模型\n\n")
            f.write("| 模型 | 训练时间(秒) | 最终训练损失 | 最终验证损失 | 最终Rel-L2 |\n")
            f.write("|------|-------------|-------------|-------------|------------|\n")
            
            for model in successful_models:
                metrics = model['metrics']
                f.write(f"| {model['name']} | {metrics.get('training_time', 'N/A'):.1f} | "
                       f"{metrics.get('final_train_loss', 'N/A')} | "
                       f"{metrics.get('final_val_loss', 'N/A')} | "
                       f"{metrics.get('final_rel_l2', 'N/A')} |\n")
        
        if failed_models:
            f.write("\n## 失败的模型\n\n")
            for model in failed_models:
                f.write(f"- **{model['name']}**: {model['error']}\n\n")
    
    print(f"\n结果已保存到:")
    print(f"- JSON: {json_file}")
    print(f"- Markdown: {md_file}")
    
    return json_file, md_file


def main():
    """主函数"""
    print("开始最终批量训练 - 所有已知问题已修复")
    print("="*80)
    
    # 要训练的模型列表（排除有问题的模型）
    models_to_train = [
        'unet',
        'unet_plus_plus', 
        'fno2d',           # 已修复：禁用AMP
        'segformer',       # 已修复：BaseModel继承
        'unetformer',      # 已修复：BaseModel继承
        'segformer_unetformer',  # 已修复：注意力头数
        # 'mlp',           # 跳过：patch处理问题复杂
        'mlp_mixer',       # 已修复：mlp_ratio类型
        'liif',            # 已修复：创建缺失模块
        'hybrid',          # 已修复：输入通道
    ]
    
    # 跳过的模型及原因
    skipped_models = {
        'mlp': 'patch处理逻辑复杂，需要深度重构',
        'ufno_unet': '模型架构问题，需要进一步调试'
    }
    
    print(f"计划训练 {len(models_to_train)} 个模型:")
    for i, model in enumerate(models_to_train, 1):
        print(f"  {i}. {model}")
    
    if skipped_models:
        print(f"\n跳过 {len(skipped_models)} 个模型:")
        for model, reason in skipped_models.items():
            print(f"  - {model}: {reason}")
    
    print(f"\n训练配置:")
    print(f"  - Epochs: 15")
    print(f"  - Batch Size: 2") 
    print(f"  - Seed: 2025")
    print()
    
    # 创建输出目录
    output_dir = Path("runs/batch_training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(models_to_train),
        'successful_models': 0,
        'failed_models': 0,
        'success_rate': 0.0,
        'models': []
    }
    
    # 开始训练
    start_time = time.time()
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n进度: {i}/{len(models_to_train)}")
        
        success, metrics, error = run_training(model_name)
        
        model_result = {
            'name': model_name,
            'success': success,
            'metrics': metrics,
            'error': error
        }
        
        results['models'].append(model_result)
        
        if success:
            results['successful_models'] += 1
        else:
            results['failed_models'] += 1
    
    # 计算总体统计
    total_time = time.time() - start_time
    results['success_rate'] = (results['successful_models'] / results['total_models']) * 100
    results['total_training_time'] = total_time
    
    print(f"\n{'='*80}")
    print("批量训练完成!")
    print(f"总用时: {total_time:.1f} 秒")
    print(f"成功: {results['successful_models']}/{results['total_models']} "
          f"({results['success_rate']:.1f}%)")
    print(f"{'='*80}")
    
    # 保存结果
    json_file, md_file = save_results(results, output_dir)
    
    return results


if __name__ == "__main__":
    main()