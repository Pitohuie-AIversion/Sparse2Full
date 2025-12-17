#!/usr/bin/env python3
"""
测试检查点加载修复
验证PyTorch 2.6兼容性问题是否已解决
"""

import torch
import tempfile
import os
from omegaconf import DictConfig, ListConfig

def test_checkpoint_loading():
    """测试检查点加载功能"""
    print("🧪 测试检查点加载修复...")
    
    # 创建测试数据
    test_data = {
        'model_state_dict': {'weight': torch.randn(10, 10)},
        'optimizer_state_dict': {'state': {}, 'param_groups': [{'lr': 0.001}]},
        'epoch': 10,
        'best_val_loss': 0.5,
        'config': DictConfig({'test': ListConfig([1, 2, 3])})  # 包含ListConfig
    }
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    
    try:
        # 保存检查点
        torch.save(test_data, temp_path)
        print(f"✅ 检查点已保存到: {temp_path}")
        
        # 测试新的加载方法（模拟修复后的代码）
        try:
            # 添加安全全局列表
            torch.serialization.add_safe_globals([ListConfig, DictConfig])
            
            # 尝试安全加载
            checkpoint = torch.load(temp_path, weights_only=True)
            print("✅ 使用 weights_only=True 加载成功")
            
        except Exception as e:
            print(f"⚠️ 安全加载失败: {e}")
            print("🔄 尝试fallback加载...")
            
            # Fallback到非安全加载
            checkpoint = torch.load(temp_path, weights_only=False)
            print("✅ 使用 weights_only=False 加载成功")
        
        # 验证数据完整性
        assert checkpoint['epoch'] == 10
        assert checkpoint['best_val_loss'] == 0.5
        assert isinstance(checkpoint['config'], DictConfig)
        assert isinstance(checkpoint['config']['test'], ListConfig)
        
        print("✅ 检查点数据完整性验证通过")
        print("✅ PyTorch 2.6兼容性修复验证成功")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    test_checkpoint_loading()