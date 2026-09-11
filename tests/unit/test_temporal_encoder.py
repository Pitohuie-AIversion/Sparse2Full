import pytest
import torch
from models.temporal.components.temporal_encoder import (
    CausalConv1D,
    TemporalConv1D,
    PositionalEncoding,
    TemporalEncoder,
    create_temporal_encoder,
)


class TestTemporalEncoderComponents:
    """测试时序编码器核心组件"""

    def test_causal_conv_1d_causality(self):
        """验证因果卷积的严格因果性：改变未来时间步不应影响当前及过去的输出"""
        in_ch, out_ch, k = 16, 16, 3
        conv = CausalConv1D(in_channels=in_ch, out_channels=out_ch, kernel_size=k)
        conv.eval()

        x1 = torch.randn(2, in_ch, 8)
        x2 = x1.clone()
        # 修改未来时间步 (t >= 4)
        x2[:, :, 4:] += torch.randn_like(x2[:, :, 4:])

        with torch.no_grad():
            out1 = conv(x1)
            out2 = conv(x2)

        # 时间步 0 到 3 的输出必须严格一致
        assert torch.allclose(out1[:, :, :4], out2[:, :, :4], atol=1e-6)

    def test_temporal_conv_1d_forward_and_residual(self):
        """测试 1D 时序卷积残差网络的前向传播"""
        conv = TemporalConv1D(in_channels=32, hidden_channels=64, num_layers=3)
        x = torch.randn(2, 32, 10)
        out = conv(x)
        assert out.shape == x.shape
        assert out.requires_grad

    def test_positional_encoding(self):
        """测试位置编码"""
        pe = PositionalEncoding(d_model=32, max_len=50)
        x = torch.randn(2, 10, 32)
        out = pe(x)
        assert out.shape == x.shape

    def test_temporal_encoder_3d(self):
        """测试 3D 序列 [B, T, C] 输入"""
        encoder = create_temporal_encoder(input_dim=48)
        x = torch.randn(2, 12, 48)
        res = encoder(x)
        assert "encoded_sequence" in res
        assert res["encoded_sequence"].shape == (2, 12, 48)
        assert res["sequence_length"] == 12
        assert res["batch_size"] == 2

    def test_temporal_encoder_5d_channel_mode(self):
        """测试 5D 视频序列 [B, T, C, H, W] 逐像素通道模式 (C == input_dim)"""
        encoder = create_temporal_encoder(input_dim=4)
        x = torch.randn(2, 8, 4, 16, 16)
        res = encoder(x)
        assert res["encoded_sequence"].shape == (2, 8, 4, 16, 16)

    def test_temporal_encoder_5d_flattened_mode(self):
        """测试 5D 视频序列 [B, T, C, H, W] 全图展平模式 (C*H*W == input_dim)"""
        encoder = create_temporal_encoder(input_dim=2 * 8 * 8)
        x = torch.randn(2, 6, 2, 8, 8)
        res = encoder(x)
        assert res["encoded_sequence"].shape == (2, 6, 2, 8, 8)

    def test_temporal_encoder_gradient_flow(self):
        """验证端到端梯度回传"""
        encoder = create_temporal_encoder(input_dim=8)
        x = torch.randn(2, 5, 8, requires_grad=True)
        res = encoder(x)
        loss = res["encoded_sequence"].sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_receptive_field_computation(self):
        """验证感受野计算"""
        encoder = create_temporal_encoder(input_dim=16, config={"num_conv_layers": 4, "kernel_size": 3, "dilation_base": 2})
        rf = encoder.get_receptive_field()
        # 4 层 dilation=1,2,4,8, kernel_size=3 => 1 + 2*(1+2+4+8) = 1 + 30 = 31
        assert rf == 31
