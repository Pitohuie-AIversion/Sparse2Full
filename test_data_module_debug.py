#!/usr/bin/env python3
"""测试数据模块调试脚本"""

import sys
sys.path.append('.')

from datasets.real_dr_dataset import RealDiffusionReactionDataModule

def test_data_module():
    """测试数据模块"""
    print('🧪 测试数据模块...')
    
    try:
        # 导入测试
        print('✅ 数据模块导入成功')
        
        # 初始化测试
        data_module = RealDiffusionReactionDataModule(
            data_path='E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
            T_in=5,
            T_out=20,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )
        print('✅ 数据模块初始化成功')
        
        # Setup测试
        data_module.setup()
        print('✅ 数据模块setup成功')
        
        # 数据加载器测试
        train_loader = data_module.train_dataloader()
        print(f'✅ 训练数据加载器创建成功，批次数: {len(train_loader)}')
        
        val_loader = data_module.val_dataloader()
        print(f'✅ 验证数据加载器创建成功，批次数: {len(val_loader)}')
        
        # 批次数据测试
        batch = next(iter(train_loader))
        print('✅ 成功获取批次数据')
        print(f'  - 输入序列形状: {batch["input_sequence"].shape}')
        print(f'  - 目标序列形状: {batch["target_sequence"].shape}')
        print(f'  - 样本索引: {batch["sample_idx"]}')
        print(f'  - 起始时间: {batch["start_time"]}')
        
        # 数据统计
        input_seq = batch["input_sequence"]
        target_seq = batch["target_sequence"]
        print(f'  - 输入数据范围: [{input_seq.min():.4f}, {input_seq.max():.4f}]')
        print(f'  - 目标数据范围: [{target_seq.min():.4f}, {target_seq.max():.4f}]')
        print(f'  - 输入数据均值: {input_seq.mean():.4f}')
        print(f'  - 目标数据均值: {target_seq.mean():.4f}')
        
        print('🎉 数据模块测试完全成功！')
        return True
        
    except Exception as e:
        print(f'❌ 数据模块测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_module()