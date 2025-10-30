import json
import numpy as np
from collections import defaultdict
import re

# 读取批量训练结果
with open('runs/batch_training_results/simple_batch_results_20251013_052249.json', 'r') as f:
    data = json.load(f)

# 提取每个模型的性能指标
model_metrics = defaultdict(list)

for result in data['results']:
    model = result['model']
    stdout = result['stdout']
    
    # 从stdout中提取指标
    # 提取rel_l2
    rel_l2_match = re.search(r"'rel_l2': tensor\(\[\[([0-9.]+)\],\s*\[([0-9.]+)\]\]", stdout)
    if rel_l2_match:
        rel_l2_values = [float(rel_l2_match.group(1)), float(rel_l2_match.group(2))]
        model_metrics[model].extend([{'metric': 'rel_l2', 'values': rel_l2_values}])
    
    # 提取mae
    mae_match = re.search(r"'mae': tensor\(\[\[([0-9.]+)\],\s*\[([0-9.]+)\]\]", stdout)
    if mae_match:
        mae_values = [float(mae_match.group(1)), float(mae_match.group(2))]
        model_metrics[model].extend([{'metric': 'mae', 'values': mae_values}])
    
    # 提取psnr
    psnr_match = re.search(r"'psnr': tensor\(\[\[([0-9.]+)\],\s*\[([0-9.]+)\]\]", stdout)
    if psnr_match:
        psnr_values = [float(psnr_match.group(1)), float(psnr_match.group(2))]
        model_metrics[model].extend([{'metric': 'psnr', 'values': psnr_values}])
    
    # 提取ssim
    ssim_match = re.search(r"'ssim': tensor\(\[\[([0-9.]+)\],\s*\[([0-9.]+)\]\]", stdout)
    if ssim_match:
        ssim_values = [float(ssim_match.group(1)), float(ssim_match.group(2))]
        model_metrics[model].extend([{'metric': 'ssim', 'values': ssim_values}])

# 计算统计数据
model_stats = {}
for model in model_metrics:
    stats = {}
    
    # 按指标分组
    metrics_by_type = defaultdict(list)
    for entry in model_metrics[model]:
        metrics_by_type[entry['metric']].extend(entry['values'])
    
    # 计算均值和标准差
    for metric_type, values in metrics_by_type.items():
        if values:
            stats[metric_type] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0.0
            }
    
    model_stats[model] = stats

# 输出结果
print('模型性能统计结果:')
for model, stats in model_stats.items():
    print(f'\n{model.upper()}:')
    for metric, stat in stats.items():
        print(f'  {metric}: {stat["mean"]:.4f} ± {stat["std"]:.4f}')

# 保存结果到文件
with open('runs/batch_training_results/model_performance_stats.json', 'w') as f:
    json.dump(model_stats, f, indent=2)

print('\n结果已保存到: runs/batch_training_results/model_performance_stats.json')