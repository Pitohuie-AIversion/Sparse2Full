"""时序NAR管线端到端测试

测试整个时序NAR架构的数据流、训练和推理完整性。
验证所有组件的集成和兼容性。
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

from models.temporal_block import TemporalConv1D, FiLMTemporalBlock, create_temporal_module
from models.decoder.query_head import TimeQueryHead, CrossAttentionQueryHead, create_query_head
from models.wrappers.swin_temporal import SwinTemporal, SwinTemporalNAR
from models.wrappers.ar_nar_wrapper import ARNARWrapper, ARNAROutput
from models import create_model

logger = logging.getLogger(__name__)


class TestTemporalNARPipeline:
    """时序NAR管线测试类"""
    
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
    
    def test_temporal_modules(self, device, sample_data):
        """测试时序模块"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        
        # 测试TemporalConv1D
        temporal_conv = TemporalConv1D(c_in=C, c_out=C, k=3, causal=True).to(device)
        out_conv = temporal_conv(x_seq)
        assert out_conv.shape == (B, C, H, W), f"TemporalConv1D output shape mismatch: {out_conv.shape}"
        
        # 测试FiLMTemporalBlock
        temporal_film = FiLMTemporalBlock(c_in=C, c_out=C).to(device)
        out_film = temporal_film(x_seq)
        assert out_film.shape == (B, C, H, W), f"FiLMTemporalBlock output shape mismatch: {out_film.shape}"
        
        # 测试工厂函数
        temporal_factory = create_temporal_module('conv1d', c_in=C, c_out=C, k=3).to(device)
        out_factory = temporal_factory(x_seq)
        assert out_factory.shape == (B, C, H, W), f"Factory temporal output shape mismatch: {out_factory.shape}"
        
        logger.info("✓ 时序模块测试通过")
    
    def test_query_heads(self, device, sample_data):
        """测试NAR查询头"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        d_model = 96
        
        # 生成memory特征
        memory = torch.randn(B, d_model, H//4, W//4, device=device)  # 下采样特征
        
        # 测试TimeQueryHead
        time_head = TimeQueryHead(d_model=d_model, c_out=C, max_timesteps=32).to(device)
        out_time = time_head(memory, T_out)
        assert out_time.shape == (B, T_out, C, H//4, W//4), f"TimeQueryHead output shape mismatch: {out_time.shape}"
        
        # 测试CrossAttentionQueryHead
        cross_head = CrossAttentionQueryHead(d_model=d_model, c_out=C, num_heads=8).to(device)
        out_cross = cross_head(memory, T_out)
        assert out_cross.shape == (B, T_out, C, H//4, W//4), f"CrossAttentionQueryHead output shape mismatch: {out_cross.shape}"
        
        # 测试工厂函数
        head_factory = create_query_head('simple', d_model=d_model, c_out=C).to(device)
        out_factory = head_factory(memory, T_out)
        assert out_factory.shape == (B, T_out, C, H//4, W//4), f"Factory query head output shape mismatch: {out_factory.shape}"
        
        logger.info("✓ NAR查询头测试通过")
    
    def test_swin_temporal(self, device, sample_data):
        """测试Swin时序包装器"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        
        # SwinUNet基础配置
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
        
        # 时序配置
        temporal_cfg = {
            'enabled': True,
            'type': 'conv1d',
            'c_out': C,
            'k': 3,
            'causal': True
        }
        
        # 测试SwinTemporal
        swin_temporal = SwinTemporal(base_kwargs, temporal_cfg).to(device)
        
        # 单帧输入
        x_single = x_seq[:, -1]  # (B, C, H, W)
        out_single = swin_temporal(x_single)
        assert out_single.shape == (B, C, H, W), f"SwinTemporal single frame output shape mismatch: {out_single.shape}"
        
        # 多帧输入
        out_multi = swin_temporal(x_seq)
        assert out_multi.shape == (B, C, H, W), f"SwinTemporal multi frame output shape mismatch: {out_multi.shape}"
        
        # 测试特征提取
        out_with_features, features = swin_temporal(x_seq, return_features=True)
        assert out_with_features.shape == (B, C, H, W), f"SwinTemporal output with features shape mismatch: {out_with_features.shape}"
        assert features is not None, "Features should not be None"
        
        logger.info("✓ Swin时序包装器测试通过")
    
    def test_swin_temporal_nar(self, device, sample_data):
        """测试Swin时序NAR双头模块"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        teacher_seq = sample_data['teacher_seq']
        
        # 配置
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
        
        temporal_cfg = {
            'enabled': True,
            'type': 'conv1d',
            'c_out': C,
            'k': 3,
            'causal': True
        }
        
        nar_cfg = {
            'head_type': 'simple',
            'd_model': 96,
            'max_timesteps': 32,
            'dropout': 0.1
        }
        
        ar_cfg = {
            'detach_rollout': True,
            'scheduled_sampling': False
        }
        
        # 测试双头模式
        swin_nar = SwinTemporalNAR(
            base_kwargs=base_kwargs,
            temporal_cfg=temporal_cfg,
            nar_cfg=nar_cfg,
            ar_cfg=ar_cfg,
            use_ar=True,
            use_nar=True
        ).to(device)
        
        # 训练模式
        swin_nar.train()
        ar_out, nar_out = swin_nar(x_seq, T_out, teacher_seq, train_mode=True, return_both=True)
        
        if ar_out is not None:
            assert ar_out.shape == (B, T_out, C, H, W), f"AR output shape mismatch: {ar_out.shape}"
        if nar_out is not None:
            assert nar_out.shape == (B, T_out, C, H, W), f"NAR output shape mismatch: {nar_out.shape}"
        
        # 推理模式
        swin_nar.eval()
        with torch.no_grad():
            inference_out = swin_nar(x_seq, T_out, return_both=False)
            assert inference_out.shape == (B, T_out, C, H, W), f"Inference output shape mismatch: {inference_out.shape}"
        
        logger.info("✓ Swin时序NAR双头模块测试通过")
    
    def test_ar_nar_wrapper(self, device, sample_data):
        """测试AR-NAR双头包装器"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        target_seq = sample_data['target_seq']
        teacher_seq = sample_data['teacher_seq']
        
        # 配置
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
                'enabled': True,
                'type': 'conv1d',
                'c_out': C,
                'k': 3,
                'causal': True
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
            'use_ar': True,
            'use_nar': True
        }
        
        loss_config = {
            'ar_weight': 1.0,
            'nar_weight': 1.0,
            'ar_weight_schedule': 'constant',
            'nar_weight_schedule': 'constant'
        }
        
        training_config = {
            'inference_mode': 'nar',
            'total_epochs': 100,
            'enable_monitoring': False  # 测试时关闭监控
        }
        
        # 创建包装器
        wrapper = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 训练前向传播
        wrapper.train()
        train_output = wrapper(
            x_seq=x_seq,
            T_out=T_out,
            teacher_seq=teacher_seq,
            compute_loss=True,
            target_seq=target_seq
        )
        
        assert isinstance(train_output, ARNAROutput), f"Training output should be ARNAROutput, got {type(train_output)}"
        assert train_output.total_loss is not None, "Total loss should not be None"
        assert train_output.metrics is not None, "Metrics should not be None"
        
        # 推理前向传播
        wrapper.eval()
        with torch.no_grad():
            inference_output = wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
            assert inference_output.shape == (B, T_out, C, H, W), f"Inference output shape mismatch: {inference_output.shape}"
        
        # 测试推理模式切换
        wrapper.set_inference_mode('ar')
        with torch.no_grad():
            ar_inference = wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
            assert ar_inference.shape == (B, T_out, C, H, W), f"AR inference output shape mismatch: {ar_inference.shape}"
        
        logger.info("✓ AR-NAR双头包装器测试通过")
    
    def test_model_factory_integration(self, device, sample_data):
        """测试模型工厂函数集成"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        
        # 测试通过工厂函数创建模型
        model_config = {
            'model_config': {
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
                    'enabled': True,
                    'type': 'conv1d',
                    'c_out': C,
                    'k': 3,
                    'causal': True
                },
                'nar': {
                    'head_type': 'simple',
                    'd_model': 96,
                    'max_timesteps': 32
                },
                'ar': {
                    'detach_rollout': True
                },
                'use_ar': True,
                'use_nar': True
            },
            'loss_config': {
                'ar_weight': 1.0,
                'nar_weight': 1.0
            },
            'training_config': {
                'inference_mode': 'nar',
                'total_epochs': 100,
                'enable_monitoring': False
            }
        }
        
        # 通过工厂函数创建
        model = create_model('ARNARWrapper', **model_config).to(device)
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            output = model(x_seq=x_seq, T_out=T_out, compute_loss=False)
            assert output.shape == (B, T_out, C, H, W), f"Factory model output shape mismatch: {output.shape}"
        
        logger.info("✓ 模型工厂函数集成测试通过")
    
    def test_gradient_flow(self, device, sample_data):
        """测试梯度流"""
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        target_seq = sample_data['target_seq']
        teacher_seq = sample_data['teacher_seq']
        
        # 创建简化模型
        model_config = {
            'base_kwargs': {
                'in_channels': C,
                'out_channels': C,
                'img_size': H,
                'patch_size': 4,
                'embed_dim': 48,  # 减小模型
                'depths': [1, 1, 1, 1],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8
            },
            'temporal': {
                'enabled': True,
                'type': 'conv1d',
                'c_out': C,
                'k': 3,
                'causal': True
            },
            'nar': {
                'head_type': 'simple',
                'd_model': 48,
                'max_timesteps': 32
            },
            'ar': {
                'detach_rollout': True
            },
            'use_ar': True,
            'use_nar': True
        }
        
        loss_config = {
            'ar_weight': 1.0,
            'nar_weight': 1.0
        }
        
        training_config = {
            'inference_mode': 'nar',
            'total_epochs': 100,
            'enable_monitoring': False
        }
        
        model = ARNARWrapper(model_config, loss_config, training_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        output = model(
            x_seq=x_seq,
            T_out=T_out,
            teacher_seq=teacher_seq,
            compute_loss=True,
            target_seq=target_seq
        )
        
        loss = output.total_loss
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in model parameters"
        
        optimizer.step()
        
        logger.info("✓ 梯度流测试通过")
    
    def test_memory_efficiency(self, device, sample_data):
        """测试显存效率"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping memory test")
        
        B, T_in, T_out, C, H, W = sample_data['dims']
        x_seq = sample_data['x_seq']
        
        # 清空显存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # 创建模型
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
                'enabled': True,
                'type': 'conv1d',
                'c_out': C,
                'k': 3,
                'causal': True
            },
            'nar': {
                'head_type': 'simple',
                'd_model': 96,
                'max_timesteps': 32
            },
            'ar': {
                'detach_rollout': True
            },
            'use_ar': False,  # 只测试NAR以节省显存
            'use_nar': True
        }
        
        loss_config = {'ar_weight': 0.0, 'nar_weight': 1.0}
        training_config = {'inference_mode': 'nar', 'total_epochs': 100, 'enable_monitoring': False}
        
        model = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(x_seq=x_seq, T_out=T_out, compute_loss=False)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1024**2  # MB
        
        logger.info(f"显存使用: {memory_used:.2f} MB")
        
        # 简单的显存检查（不应该超过1GB）
        assert memory_used < 1024, f"Memory usage too high: {memory_used:.2f} MB"
        
        logger.info("✓ 显存效率测试通过")


def run_tests():
    """运行所有测试"""
    import pytest
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行测试
    pytest.main([__file__, '-v', '-s'])


if __name__ == '__main__':
    run_tests()