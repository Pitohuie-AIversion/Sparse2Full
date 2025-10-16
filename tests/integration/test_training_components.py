"""训练组件测试脚本

验证数据加载、模型初始化、损失计算等核心组件
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_data_loading():
    """测试数据加载"""
    print("=" * 50)
    print("测试数据加载...")
    
    try:
        from datasets import PDEBenchDataModule
        
        # 加载配置
        config_path = "configs/data/pdebench.yaml"
        data_config = OmegaConf.load(config_path)
        
        # 创建数据模块
        data_module = PDEBenchDataModule(data_config)
        
        # 准备数据
        data_module.setup()
        
        # 获取数据加载器
        train_loader = data_module.train_dataloader()
        
        # 测试一个批次
        batch = next(iter(train_loader))
        
        print(f"✓ 数据加载成功")
        print(f"  - 批次大小: {len(batch)}")
        if isinstance(batch, (list, tuple)):
            for i, item in enumerate(batch):
                if torch.is_tensor(item):
                    print(f"  - 张量 {i}: {item.shape}, dtype: {item.dtype}")
        elif torch.is_tensor(batch):
            print(f"  - 张量形状: {batch.shape}, dtype: {batch.dtype}")
        
        return True, batch
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False, None

def test_model_creation():
    """测试模型创建"""
    print("=" * 50)
    print("测试模型创建...")
    
    try:
        from models.base import create_model
        from omegaconf import OmegaConf
        
        # 加载模型配置
        model_config = OmegaConf.load("configs/model/swin_unet.yaml")
        
        # 修改配置以匹配PDEBench数据
        model_config.params.in_channels = 1  # PDEBench单通道
        model_config.params.out_channels = 1
        model_config.params.img_size = 128   # PDEBench分辨率
        
        # 创建模型
        model = create_model(model_config)
        
        # 测试前向传播
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 创建测试输入
        test_input = torch.randn(2, 1, 128, 128).to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ 模型创建成功")
        print(f"  - 模型类型: {type(model).__name__}")
        print(f"  - 输入形状: {test_input.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - 设备: {device}")
        
        return True, model
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_loss_computation():
    """测试损失计算"""
    print("=" * 50)
    print("测试损失计算...")
    
    try:
        from ops.losses import compute_total_loss
        
        # 创建测试数据
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = torch.randn(2, 1, 128, 128).to(device)
        target = torch.randn(2, 1, 128, 128).to(device)
        
        # 模拟观测数据
        observed = torch.randn(2, 1, 32, 32).to(device)  # SR x4下采样
        
        # 模拟观测数据字典
        obs_data = {
            'baseline': observed,
            'mask': torch.ones_like(observed),
            'coords': torch.randn(2, 2, 32, 32).to(device),
            'h_params': {'task': 'SR', 'scale': 4},
            'observation': observed
        }
        
        # 模拟配置
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            'train': {
                'loss_weights': {
                    'reconstruction': 1.0,
                    'spectral': 0.5,
                    'data_consistency': 1.0
                }
            }
        })
        
        # 计算损失
        loss_dict = compute_total_loss(
            pred_z=pred,
            target_z=target,
            obs_data=obs_data,
            norm_stats=None,
            config=config
        )
        
        print(f"✓ 损失计算成功")
        for loss_name, loss_value in loss_dict.items():
            print(f"  - {loss_name}: {loss_value:.6f}")
        
        return True, loss_dict
        
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_gpu_memory():
    """测试GPU内存使用"""
    print("=" * 50)
    print("测试GPU内存...")
    
    if not torch.cuda.is_available():
        print("CPU模式，跳过GPU内存测试")
        return True
    
    try:
        device = torch.device("cuda")
        
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 获取初始内存
        initial_memory = torch.cuda.memory_allocated(device)
        max_memory = torch.cuda.max_memory_allocated(device)
        
        print(f"✓ GPU内存状态")
        print(f"  - 设备: {torch.cuda.get_device_name(device)}")
        print(f"  - 总内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"  - 已分配: {initial_memory / 1024**3:.3f} GB")
        print(f"  - 峰值使用: {max_memory / 1024**3:.3f} GB")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU内存检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("PDEBench训练组件测试")
    print("=" * 50)
    
    # 测试结果
    results = {}
    
    # 1. 测试数据加载
    results['data'], batch = test_data_loading()
    
    # 2. 测试模型创建
    results['model'], model = test_model_creation()
    
    # 3. 测试损失计算
    results['loss'], loss_dict = test_loss_computation()
    
    # 4. 测试GPU内存
    results['gpu'] = test_gpu_memory()
    
    # 汇总结果
    print("=" * 50)
    print("测试结果汇总:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  - {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 所有测试通过！可以开始训练。")
        return True
    else:
        print("❌ 部分测试失败，请检查配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)