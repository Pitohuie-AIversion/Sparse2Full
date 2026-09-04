import h5py
import os

# 检查不同数据集的结构
datasets = [
    'data/2D/NS_incom/ns_incom_inhom_2d_512-0.h5',
    'data/DR2D/2D_diff-react_NA_NA.h5',
    'data/2D/shallow-water/2D_rdb_NA_NA.h5'
]

for dataset_path in datasets:
    if os.path.exists(dataset_path):
        print(f'\n=== {os.path.basename(dataset_path)} ===')
        try:
            with h5py.File(dataset_path, 'r') as f:
                keys = list(f.keys())
                print(f'总键数: {len(keys)}')
                
                # 检查前几个键的结构
                for i, key in enumerate(keys[:3]):
                    item = f[key]
                    if isinstance(item, h5py.Group):
                        print(f'{key} (组): {list(item.keys())}')
                        if 'data' in item:
                            print(f'  data形状: {item["data"].shape}')
                    else:
                        print(f'{key} (数据集): {item.shape}')
                    
                    if i >= 2:  # 只检查前3个
                        break
        except Exception as e:
            print(f'读取失败: {e}')
    else:
        print(f'文件不存在: {dataset_path}')