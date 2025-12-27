"""测试ARWrapper统一接口

验证ARWrapper在接口统一后的功能正确性，包括：
1. 统一接口规范符合性
2. 单帧预测功能
3. 时间序列预测功能（通过外部函数）
4. 输入打包格式支持
5. 向后兼容性
"""

import pytest
import torch
import torch.nn as nn
from models.ar import ARWrapper, autoregressive_predict


class MockSingleFrameModel(nn.Module):
    """模拟单帧模型"""
    def __init__(self, in_ch=4, out_ch=4, img_size=64):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.img_size = img_size
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        """统一接口：x [B, C_in, H, W] -> y [B, C_out, H, W]"""
        return self.conv(x)


class TestARWrapperUnifiedInterface:
    """测试ARWrapper统一接口"""
    
    def test_unified_interface_signature(self):
        """测试统一接口签名规范"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model)
        
        # 验证forward方法签名
        import inspect
        sig = inspect.signature(ar_model.forward)
        params = list(sig.parameters.keys())
        
        # 统一接口应该只有 'x' 参数（除了self）
        assert params == ['x'], f"Expected params ['x'], got {params}"
    
    def test_unified_interface_dimensions(self):
        """测试统一接口维度规范"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model)
        
        B, C, H, W = 2, 4, 64, 64
        x = torch.randn(B, C, H, W)
        
        # 测试4D输入
        y = ar_model(x)
        assert y.dim() == 4, f"Output should be 4D, got {y.dim()}D"
        assert y.shape[0] == B, f"Batch size mismatch: {y.shape[0]} != {B}"
        assert y.shape[2:] == (H, W), f"Spatial size mismatch: {y.shape[2:]} != {(H, W)}"
        
        # 测试5D输入应该报错
        x_5d = torch.randn(B, 3, C, H, W)
        with pytest.raises(ValueError, match="Unified interface requires 4D input"):
            ar_model(x_5d)
    
    def test_single_frame_prediction(self):
        """测试单帧预测功能"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model)
        
        B, C, H, W = 2, 4, 64, 64
        x = torch.randn(B, C, H, W)
        
        # 单帧预测
        y = ar_model(x)
        
        assert y.shape == (B, base_model.out_channels, H, W)
        assert torch.is_tensor(y)
        assert not torch.isnan(y).any()
    
    def test_packed_input_format(self):
        """测试打包输入格式 [baseline, coords, mask]"""
        base_model = MockSingleFrameModel(in_ch=4)
        ar_model = ARWrapper(base_model)
        
        B, H, W = 2, 64, 64
        # 模拟打包输入：12通道（4+4+4）
        packed_input = torch.randn(B, 12, H, W)
        
        # 应该能正确处理打包输入
        y = ar_model(packed_input)
        
        assert y.shape == (B, base_model.out_channels, H, W)
        assert not torch.isnan(y).any()
    
    def test_temporal_prediction_external(self):
        """测试外部时间序列预测函数"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model)
        
        B, T_in, T_out, C, H, W = 2, 5, 10, 4, 64, 64
        
        # 输入序列
        x_seq = torch.randn(B, T_in, C, H, W)
        teacher = torch.randn(B, T_out, C, H, W)
        
        # 使用外部函数进行时间序列预测
        y_seq = autoregressive_predict(
            model=ar_model,
            x_seq=x_seq,
            T_out=T_out,
            teacher=teacher,
            train_mode=True
        )
        
        assert y_seq.shape == (B, T_out, C, H, W)
        assert not torch.isnan(y_seq).any()
    
    def test_backward_compatibility_method(self):
        """测试向后兼容方法"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model)
        
        B, T_in, T_out, C, H, W = 2, 5, 10, 4, 64, 64
        
        x_seq = torch.randn(B, T_in, C, H, W)
        teacher = torch.randn(B, T_out, C, H, W)
        
        # 使用向后兼容的autoregressive_predict方法
        y_seq = ar_model.autoregressive_predict(
            x_seq=x_seq,
            T_out=T_out,
            teacher=teacher,
            train_mode=True
        )
        
        assert y_seq.shape == (B, T_out, C, H, W)
        assert not torch.isnan(y_seq).any()
    
    def test_model_info_unified_interface(self):
        """测试模型信息中的统一接口标识"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model)
        
        info = ar_model.get_model_info()
        
        assert info['unified_interface'] == True
        assert info['interface_version'] == 'unified_v1.0'
        assert info['model_type'] == 'AR_Wrapper'
    
    def test_inference_mode(self):
        """测试推理模式"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model)
        ar_model.eval()
        
        B, T_in, T_out, C, H, W = 2, 5, 10, 4, 64, 64
        
        x_seq = torch.randn(B, T_in, C, H, W)
        
        # 推理模式（无教师强制）
        y_seq = autoregressive_predict(
            model=ar_model,
            x_seq=x_seq,
            T_out=T_out,
            train_mode=False  # 推理模式
        )
        
        assert y_seq.shape == (B, T_out, C, H, W)
        assert not torch.isnan(y_seq).any()
    
    def test_scheduled_sampling(self):
        """测试scheduled sampling功能"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model, scheduled_sampling=True)
        
        B, T_in, T_out, C, H, W = 2, 5, 10, 4, 64, 64
        
        x_seq = torch.randn(B, T_in, C, H, W)
        teacher = torch.randn(B, T_out, C, H, W)
        
        # 设置epoch以启用scheduled sampling
        ar_model.set_epoch(epoch=50, total_epochs=100)
        
        # 使用scheduled sampling
        y_seq = autoregressive_predict(
            model=ar_model,
            x_seq=x_seq,
            T_out=T_out,
            teacher=teacher,
            train_mode=True,
            scheduled_sampling_prob=ar_model.get_sampling_prob()
        )
        
        assert y_seq.shape == (B, T_out, C, H, W)
        assert not torch.isnan(y_seq).any()


class TestARWrapperIntegration:
    """集成测试：验证与训练脚本的兼容性"""
    
    def test_training_loop_compatibility(self):
        """测试与训练循环的兼容性"""
        base_model = MockSingleFrameModel()
        ar_model = ARWrapper(base_model)
        
        B, T_in, T_out, C, H, W = 2, 5, 3, 4, 64, 64
        
        # 模拟训练批次
        batch = {
            'input': torch.randn(B, T_in, C, H, W),
            'target': torch.randn(B, T_out, C, H, W)
        }
        
        # 使用新的时间序列预测方式
        predictions = autoregressive_predict(
            model=ar_model,
            x_seq=batch['input'],
            T_out=T_out,
            teacher=batch['target'],
            train_mode=True
        )
        
        assert predictions.shape == (B, T_out, C, H, W)
        
        # 验证可以计算损失
        loss = nn.MSELoss()(predictions, batch['target'])
        assert torch.is_tensor(loss)
        assert not torch.isnan(loss)
        
        # 验证可以反向传播
        loss.backward()
        
        # 验证梯度存在
        for param in ar_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])