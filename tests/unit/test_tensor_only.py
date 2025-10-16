#!/usr/bin/env python3
"""
测试只使用tensor数据的PDEBench数据集配置
验证修改后的数据读取逻辑是否正确
更新：专门测试tensor数据读取，不使用nu参数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import h5py
from datasets.pdebench import PDEBenchSR
from omegaconf import OmegaConf

def test_tensor_only_dataset():
    """测试只使用tensor数据的数据集"""
    print("=== 测试只使用tensor数据的PDEBench数据集 ===")
    
    # 加载配置
    config_path = "configs/data/pdebench.yaml"
    config = OmegaConf.load(config_path)
    
    print(f"配置文件keys: {config.keys}")
    print(f"数据路径: {config.data_path}")
    
    # 检查HDF5文件内容
    hdf5_file = "E:/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5"
    print(f"\n检查HDF5文件: {hdf5_file}")
    
    with h5py.File(hdf5_file, 'r') as f:
        print(f"文件中的键: {list(f.keys())}")
        if 'tensor' in f:
            tensor_shape = f['tensor'].shape
            print(f"tensor形状: {tensor_shape}")
            print(f"tensor数据类型: {f['tensor'].dtype}")
        
    # 创建数据集实例 - 直接使用具体的HDF5文件路径
    try:
        dataset = PDEBenchSR(
            data_path="E:/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5",  # 直接指定HDF5文件
            keys=config.keys,  # 只包含["tensor"]
            scale=4,
            sigma=1.0,
            image_size=config.image_size,
            normalize=config.normalize,
            split="train"
        )
        
        print(f"\n数据集创建成功!")
        print(f"数据集长度: {len(dataset)}")
        print(f"使用的keys: {dataset.keys}")
        
        # 测试读取第一个样本
        print("\n测试读取第一个样本...")
        sample = dataset[0]
        
        print(f"样本键: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: 形状={value.shape}, 数据类型={value.dtype}, 设备={value.device}")
                print(f"  - 最小值: {value.min():.6f}")
                print(f"  - 最大值: {value.max():.6f}")
                print(f"  - 均值: {value.mean():.6f}")
                print(f"  - 标准差: {value.std():.6f}")
                print(f"  - NaN数量: {torch.isnan(value).sum()}")
                print(f"  - Inf数量: {torch.isinf(value).sum()}")
            else:
                print(f"{key}: {value}")
        
        # 测试读取多个样本
        print("\n测试读取前3个样本...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            target = sample['target']
            observation = sample['observation']
            print(f"样本{i}: target形状={target.shape}, observation形状={observation.shape}")
            
        print("\n✅ 数据集测试成功!")
        return True
        
    except Exception as e:
        print(f"\n❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_hdf5_access():
    """直接测试HDF5文件访问"""
    print("\n=== 直接测试HDF5文件访问 ===")
    
    hdf5_file = "E:/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5"
    
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # 读取tensor数据的前3个样本
            tensor_data = f['tensor']
            print(f"tensor完整形状: {tensor_data.shape}")
            
            for i in range(3):
                sample = tensor_data[i]  # 形状应该是 (1, 128, 128)
                print(f"样本{i}: 形状={sample.shape}, 数据类型={sample.dtype}")
                print(f"  - 最小值: {sample.min():.6f}")
                print(f"  - 最大值: {sample.max():.6f}")
                print(f"  - 均值: {sample.mean():.6f}")
                print(f"  - 标准差: {sample.std():.6f}")
                
                # 转换为torch tensor
                tensor_torch = torch.tensor(sample, dtype=torch.float32)
                print(f"  - torch tensor形状: {tensor_torch.shape}")
                
        print("✅ 直接HDF5访问测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 直接HDF5访问测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试tensor数据读取...")
    
    # 测试直接HDF5访问
    hdf5_success = test_direct_hdf5_access()
    
    # 测试数据集
    dataset_success = test_tensor_only_dataset()
    
    print(f"\n=== 测试结果 ===")
    print(f"直接HDF5访问: {'✅ 成功' if hdf5_success else '❌ 失败'}")
    print(f"数据集测试: {'✅ 成功' if dataset_success else '❌ 失败'}")
    
    if hdf5_success and dataset_success:
        print("\n🎉 所有测试通过! 数据集配置正确，只使用tensor数据。")
    else:
        print("\n⚠️  部分测试失败，需要进一步调试。")