import sys
sys.path.append('F:/Zhaoyang/Sparse2Full')
from datasets.temporal_pdebench import TemporalPDEBenchBase
import numpy as np

# 测试数据集加载
dataset = TemporalPDEBenchBase(
    data_path='e:/2d/diffusion-reaction/2D_diff-react_NA_NA.h5',
    keys=['data'],
    T_in=10,
    T_out=10,
    dt=1,
    normalize=True,
    use_official_format=True,
    image_size=(128, 128)
)

print(f'数据集长度: {len(dataset)}')
sample = dataset[0]
print(f'样本键: {list(sample.keys())}')
print(f'输入序列形状: {sample["input_sequence"].shape}')
print(f'目标序列形状: {sample["target_sequence"].shape}')
print('数据集加载成功！')