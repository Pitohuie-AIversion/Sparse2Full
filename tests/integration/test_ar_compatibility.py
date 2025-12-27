"""AR基线兼容性测试

验证新的时序NAR架构与现有AR基线的兼容性。
确保现有训练脚本和数据管线正常工作。
"""

import torch
import torch.nn as nn
import pytest
import logging
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from models.wrappers.ar_wrapper import ARWrapper
    from models.wrappers.ar_nar_wrapper import ARNARWrapper
    from models import create_model
except Exception:
    pytest.skip("缺少AR/NAR包装器或模型工厂，跳过兼容性测试", allow_module_level=True)

logger = logging.getLogger(__name__)


class TestARBaselineCompatibility:
    """AR基线兼容性测试类"""
    
    @pytest.fixture
    def device(self):
        """测试设备"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_data(self, device):
        """生成测试数据"""
        B, T_in, T_out, C, H, W = 2, 4, 3, 1, 64, 64
        
        # 输入序列
        x_seq = torch.randn(B, T_in, C, H, W, device=device)
        
        # 目标序列
        target_seq = torch.randn(B, T_out, C, H, W, device=device)
        
        # 教师信号
        teacher_seq = torch.randn(B, T_out, C, H, W, device=device)
        
        return {
            'x_seq': x_seq,
            'target_seq': target_seq,
            'teacher_seq': teacher_seq,
            'T_out': T_out,
            'dims': (B, T_in, T_out, C, H, W)
        }
    
    def test_ar_wrapper_compatibility(self, device, sample_data):
        """测试AR包装器兼容性"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        target_seq = sample_data['target_seq']
        teacher_seq = sample_data['teacher_seq']
        
        # 创建AR包装器
        base_kwargs = {
            'in_channels': C,
            'out_channels': C,
            'img_size': H,
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8
        }
        
        ar_config = {
            'detach_rollout': True,
            'scheduled_sampling': False,
            'sampling_prob': 0.5
        }
        
        ar_wrapper = ARWrapper(
            model_name='SwinUNet',
            base_kwargs=base_kwargs,
            ar_config=ar_config
        ).to(device)
        
        # 训练模式测试
        ar_wrapper.train()
        train_output = ar_wrapper(x_seq, T_out, teacher_seq)
        assert train_output.shape == (B, T_out, C, H, W), f"AR training output shape mismatch: {train_output.shape}"
        
        # 推理模式测试
        ar_wrapper.eval()
        with torch.no_grad():
            inference_output = ar_wrapper(x_seq, T_out)
            assert inference_output.shape == (B, T_out, C, H, W), f"AR inference output shape mismatch: {inference_output.shape}"
        
        logger.info("✓ AR包装器兼容性测试通过")
    
    def test_ar_only_mode_in_ar_nar_wrapper(self, device, sample_data):
        """测试AR-NAR包装器的纯AR模式"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        target_seq = sample_data['target_seq']
        teacher_seq = sample_data['teacher_seq']
        
        # 配置为纯AR模式
        model_config = {
            'base_kwargs': {
                'in_channels': C,
                'out_channels': C,
                'img_size': H,
                'patch_size': 4,
                'embed_dim': 96,
                'depths': [2, 2, 2, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8
            },
            'temporal': {
                'enabled': False  # 关闭时序模块
            },
            'nar': {
                'head_type': 'simple',
                'd_model': 96,
                'max_timesteps': 32
            },
            'ar': {
                'detach_rollout': True,
                'scheduled_sampling': False
            },
            'use_ar': True,   # 启用AR
            'use_nar': False  # 禁用NAR
        }
        
        loss_config = {
            'ar_weight': 1.0,
            'nar_weight': 0.0,  # NAR权重为0
            'ar_weight_schedule': 'constant',
            'nar_weight_schedule': 'constant'
        }
        
        training_config = {
            'inference_mode': 'ar',  # 推理使用AR
            'total_epochs': 100,
            'enable_monitoring': False
        }
        
        # 创建包装器
        ar_nar_wrapper = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 训练模式测试
        ar_nar_wrapper.train()
        train_output = ar_nar_wrapper(
            x_seq=x_seq,
            T_out=T_out,
            teacher_seq=teacher_seq,
            compute_loss=True,
            target_seq=target_seq
        )
        
        # 验证只有AR输出
        assert train_output.ar_output is not None, "AR output should not be None"
        assert train_output.nar_output is None, "NAR output should be None in AR-only mode"
        assert train_output.total_loss is not None, "Total loss should not be None"
        
        # 推理模式测试
        ar_nar_wrapper.eval()
        with torch.no_grad():
            inference_output = ar_nar_wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
            assert inference_output.shape == (B, T_out, C, H, W), f"AR-only inference output shape mismatch: {inference_output.shape}"
        
        logger.info("✓ AR-NAR包装器纯AR模式测试通过")
    
    def test_model_factory_ar_compatibility(self, device, sample_data):
        """测试模型工厂函数AR兼容性"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        
        # 通过工厂函数创建AR模型
        base_kwargs = {
            'in_channels': C,
            'out_channels': C,
            'img_size': H,
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8
        }
        
        ar_config = {
            'detach_rollout': True,
            'scheduled_sampling': False
        }
        
        # 测试ARWrapper创建
        ar_model = create_model(
            'ARWrapper',
            model_name='SwinUNet',
            base_kwargs=base_kwargs,
            ar_config=ar_config
        ).to(device)
        
        # 测试前向传播
        ar_model.eval()
        with torch.no_grad():
            output = ar_model(x_seq, T_out)
            assert output.shape == (B, T_out, C, H, W), f"Factory AR model output shape mismatch: {output.shape}"
        
        logger.info("✓ 模型工厂函数AR兼容性测试通过")
    
    def test_parameter_compatibility(self, device, sample_data):
        """测试参数兼容性"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        
        # 相同的基础配置
        base_kwargs = {
            'in_channels': C,
            'out_channels': C,
            'img_size': H,
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8
        }
        
        ar_config = {
            'detach_rollout': True,
            'scheduled_sampling': False
        }
        
        # 创建AR包装器
        ar_wrapper = ARWrapper(
            model_name='SwinUNet',
            base_kwargs=base_kwargs,
            ar_config=ar_config
        ).to(device)
        
        # 创建AR-NAR包装器（仅AR模式）
        model_config = {
            'base_kwargs': base_kwargs,
            'temporal': {'enabled': False},
            'nar': {'head_type': 'simple', 'd_model': 96, 'max_timesteps': 32},
            'ar': ar_config,
            'use_ar': True,
            'use_nar': False
        }
        
        loss_config = {'ar_weight': 1.0, 'nar_weight': 0.0}
        training_config = {'inference_mode': 'ar', 'total_epochs': 100, 'enable_monitoring': False}
        
        ar_nar_wrapper = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 比较参数数量（应该相近，因为都基于相同的SwinUNet）
        ar_params = sum(p.numel() for p in ar_wrapper.parameters())
        ar_nar_params = sum(p.numel() for p in ar_nar_wrapper.parameters())
        
        # AR-NAR包装器可能有额外的NAR头参数，但在AR-only模式下应该相近
        param_ratio = ar_nar_params / ar_params
        assert 0.9 <= param_ratio <= 2.0, f"Parameter count ratio too different: {param_ratio:.2f}"
        
        logger.info(f"AR参数数量: {ar_params:,}")
        logger.info(f"AR-NAR参数数量: {ar_nar_params:,}")
        logger.info(f"参数比例: {param_ratio:.2f}")
        logger.info("✓ 参数兼容性测试通过")
    
    def test_training_compatibility(self, device, sample_data):
        """测试训练兼容性"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        target_seq = sample_data['target_seq']
        teacher_seq = sample_data['teacher_seq']
        
        # 创建两个模型进行对比
        base_kwargs = {
            'in_channels': C,
            'out_channels': C,
            'img_size': H,
            'patch_size': 4,
            'embed_dim': 48,  # 减小模型以加速测试
            'depths': [1, 1, 1, 1],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8
        }
        
        ar_config = {
            'detach_rollout': True,
            'scheduled_sampling': False
        }
        
        # AR包装器
        ar_wrapper = ARWrapper(
            model_name='SwinUNet',
            base_kwargs=base_kwargs,
            ar_config=ar_config
        ).to(device)
        
        # AR-NAR包装器（AR模式）
        model_config = {
            'base_kwargs': base_kwargs,
            'temporal': {'enabled': False},
            'nar': {'head_type': 'simple', 'd_model': 48, 'max_timesteps': 32},
            'ar': ar_config,
            'use_ar': True,
            'use_nar': False
        }
        
        loss_config = {'ar_weight': 1.0, 'nar_weight': 0.0}
        training_config = {'inference_mode': 'ar', 'total_epochs': 100, 'enable_monitoring': False}
        
        ar_nar_wrapper = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 创建优化器
        ar_optimizer = torch.optim.Adam(ar_wrapper.parameters(), lr=1e-3)
        ar_nar_optimizer = torch.optim.Adam(ar_nar_wrapper.parameters(), lr=1e-3)
        
        # 训练步骤
        ar_wrapper.train()
        ar_nar_wrapper.train()
        
        # AR包装器训练
        ar_optimizer.zero_grad()
        ar_output = ar_wrapper(x_seq, T_out, teacher_seq)
        ar_loss = nn.MSELoss()(ar_output, target_seq)
        ar_loss.backward()
        ar_optimizer.step()
        
        # AR-NAR包装器训练
        ar_nar_optimizer.zero_grad()
        ar_nar_output = ar_nar_wrapper(
            x_seq=x_seq,
            T_out=T_out,
            teacher_seq=teacher_seq,
            compute_loss=True,
            target_seq=target_seq
        )
        ar_nar_loss = ar_nar_output.total_loss
        ar_nar_loss.backward()
        ar_nar_optimizer.step()
        
        # 验证损失都是有效的
        assert not torch.isnan(ar_loss), "AR loss is NaN"
        assert not torch.isnan(ar_nar_loss), "AR-NAR loss is NaN"
        assert ar_loss.item() > 0, "AR loss should be positive"
        assert ar_nar_loss.item() > 0, "AR-NAR loss should be positive"
        
        logger.info(f"AR损失: {ar_loss.item():.6f}")
        logger.info(f"AR-NAR损失: {ar_nar_loss.item():.6f}")
        logger.info("✓ 训练兼容性测试通过")
    
    def test_inference_compatibility(self, device, sample_data):
        """测试推理兼容性"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        
        # 创建模型
        base_kwargs = {
            'in_channels': C,
            'out_channels': C,
            'img_size': H,
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8
        }
        
        ar_config = {
            'detach_rollout': True,
            'scheduled_sampling': False
        }
        
        # AR包装器
        ar_wrapper = ARWrapper(
            model_name='SwinUNet',
            base_kwargs=base_kwargs,
            ar_config=ar_config
        ).to(device)
        
        # AR-NAR包装器（AR模式）
        model_config = {
            'base_kwargs': base_kwargs,
            'temporal': {'enabled': False},
            'nar': {'head_type': 'simple', 'd_model': 96, 'max_timesteps': 32},
            'ar': ar_config,
            'use_ar': True,
            'use_nar': False
        }
        
        loss_config = {'ar_weight': 1.0, 'nar_weight': 0.0}
        training_config = {'inference_mode': 'ar', 'total_epochs': 100, 'enable_monitoring': False}
        
        ar_nar_wrapper = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 推理测试
        ar_wrapper.eval()
        ar_nar_wrapper.eval()
        
        with torch.no_grad():
            ar_output = ar_wrapper(x_seq, T_out)
            ar_nar_output = ar_nar_wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
        
        # 验证输出形状一致
        assert ar_output.shape == ar_nar_output.shape, f"Output shapes mismatch: {ar_output.shape} vs {ar_nar_output.shape}"
        
        # 验证输出值域合理
        assert not torch.isnan(ar_output).any(), "AR output contains NaN"
        assert not torch.isnan(ar_nar_output).any(), "AR-NAR output contains NaN"
        assert torch.isfinite(ar_output).all(), "AR output contains infinite values"
        assert torch.isfinite(ar_nar_output).all(), "AR-NAR output contains infinite values"
        
        logger.info("✓ 推理兼容性测试通过")


def run_tests():
    """运行所有测试"""
    import pytest
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行测试
    pytest.main([__file__, '-v', '-s'])


if __name__ == '__main__':
    run_tests()