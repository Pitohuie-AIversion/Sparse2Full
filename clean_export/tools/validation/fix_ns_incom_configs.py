#!/usr/bin/env python3
"""
修复ns_incom数据集配置文件中的样本数量问题
"""
import yaml
import os
import h5py

def check_dataset_samples(data_path):
    """检查数据集中每个键的样本数量"""
    try:
        with h5py.File(data_path, 'r') as f:
            min_samples = float('inf')
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    samples = f[key].shape[0]
                    min_samples = min(min_samples, samples)
                    print(f"  {key}: {samples} samples")
            return min_samples if min_samples != float('inf') else 0
    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return 0

def fix_ns_incom_configs():
    """修复ns_incom相关的配置文件"""
    config_dir = "configs/auto_generated"
    ns_incom_configs = [
        "ns_incom_inhom_2d_512_0_sr_x2_optimized.yaml",
        "ns_incom_inhom_2d_512_0_crop_20_optimized.yaml"
    ]
    
    # 检查数据集实际样本数量
    data_path = "data/2D/NS_incom/ns_incom_inhom_2d_512-0.h5"
    print(f"检查数据集: {data_path}")
    min_samples = check_dataset_samples(data_path)
    print(f"最小样本数: {min_samples}")
    
    if min_samples == 0:
        print("❌ 无法读取数据集，跳过修复")
        return
    
    # 修复配置文件
    for config_file in ns_incom_configs:
        config_path = os.path.join(config_dir, config_file)
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件不存在: {config_path}")
            continue
            
        print(f"\n修复配置文件: {config_file}")
        
        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 添加case_ids限制
        if 'data' not in config:
            config['data'] = {}
        
        # 为ns_incom数据集添加case_ids，限制在实际样本数量内
        config['data']['case_ids'] = list(range(min_samples))
        
        # 调整批量大小，确保不超过样本数量
        if config['data'].get('batch_size', 1) > min_samples:
            config['data']['batch_size'] = min_samples
            print(f"  调整batch_size为: {min_samples}")
        
        # 保存修复后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"  ✅ 已添加case_ids: {config['data']['case_ids']}")
        print(f"  ✅ batch_size: {config['data']['batch_size']}")

def main():
    print("🔧 修复ns_incom数据集配置文件")
    fix_ns_incom_configs()
    print("\n✅ 修复完成!")

if __name__ == "__main__":
    main()