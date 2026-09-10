"""
SwinFluidSR 单元测试

覆盖：
- 模型实例化与参数校验
- 前向传播形状一致性（低分辨率输入 → 高分辨率输出）
- 全局物理残差学习（Bicubic Residual）的开关行为
- 梯度反传稳定性
- 统一注册表集成
"""

import pytest
import torch
import torch.nn as nn


class TestSwinFluidSR:
    """SwinFluidSR 模型核心功能测试"""

    @pytest.fixture(autouse=True)
    def setup(self, device):
        self.device = device

    def _create_model(self, **kwargs):
        from models.spatial.swin_fluid_sr import SwinFluidSR

        defaults = dict(
            in_channels=1,
            out_channels=1,
            img_size=16,
            upscale_factor=4,
            embed_dim=32,
            depths=[2, 2],
            num_heads=[2, 2],
            window_size=8,
            mlp_ratio=2.0,
            drop_path_rate=0.0,
            global_residual=True,
        )
        defaults.update(kwargs)
        return SwinFluidSR(**defaults).to(self.device)

    def test_instantiation(self):
        """模型正常实例化，参数成员属性设置正确"""
        model = self._create_model()
        assert model.in_channels == 1
        assert model.out_channels == 1
        assert model.upscale_factor == 4
        assert model.embed_dim == 32
        assert len(model.layers) == 2

    def test_forward_shape_4x(self):
        """4x 超分辨率：16x16 → 64x64"""
        model = self._create_model(img_size=16, upscale_factor=4)
        x = torch.randn(2, 1, 16, 16, device=self.device)
        y = model(x)
        assert y.shape == (2, 1, 64, 64), f"Expected (2,1,64,64) but got {y.shape}"

    def test_forward_shape_2x(self):
        """2x 超分辨率：16x16 → 32x32"""
        model = self._create_model(img_size=16, upscale_factor=2)
        x = torch.randn(2, 1, 16, 16, device=self.device)
        y = model(x)
        assert y.shape == (2, 1, 32, 32), f"Expected (2,1,32,32) but got {y.shape}"

    def test_forward_shape_8x_16_to_128(self):
        """8x 超分辨率：16x16 输入 → 128x128 高清流场重建"""
        model = self._create_model(img_size=16, upscale_factor=8, window_size=8)
        x = torch.randn(2, 1, 16, 16, device=self.device)
        y = model(x)
        assert y.shape == (2, 1, 128, 128), f"Expected (2,1,128,128) but got {y.shape}"

    def test_multichannel_forward(self):
        """多通道输入输出（如速度场 u,v 两分量）"""
        model = self._create_model(in_channels=2, out_channels=2, img_size=16, upscale_factor=4)
        x = torch.randn(2, 2, 16, 16, device=self.device)
        y = model(x)
        assert y.shape == (2, 2, 64, 64)

    def test_global_residual_enabled(self):
        """全局残差开启时：输出 = 高频特征 + Bicubic 基底"""
        model = self._create_model(global_residual=True)
        x = torch.randn(1, 1, 16, 16, device=self.device)
        y_on = model(x)

        # 全局残差关闭
        model_off = self._create_model(global_residual=False)
        # 复制权重
        model_off.load_state_dict(model.state_dict(), strict=False)
        y_off = model_off(x)

        # 两者应该产生不同输出（因为残差项被加/减了）
        assert not torch.allclose(y_on, y_off, atol=1e-6), \
            "全局残差开关应导致输出差异"

    def test_gradient_flow(self):
        """梯度反传：损失对所有参数可导且无 NaN"""
        model = self._create_model()
        x = torch.randn(2, 1, 16, 16, device=self.device, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()

        # 输入梯度
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # 模型参数梯度
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}"

    def test_model_info(self):
        """get_model_info() 返回完整的架构元信息"""
        model = self._create_model()
        info = model.get_model_info()
        assert info['arch'] == 'SwinFluidSR'
        assert info['upscale_factor'] == 4
        assert info['embed_dim'] == 32
        assert info['global_residual'] is True
        assert 'num_params' in info
        assert info['num_params'] > 0

    def test_registry_integration(self):
        """通过统一注册表 create_model 正确创建 SwinFluidSR"""
        from models.registry import create_model

        model = create_model(
            'SwinFluidSR',
            in_channels=1,
            out_channels=1,
            img_size=16,
            upscale_factor=4,
            embed_dim=32,
            depths=[2, 2],
            num_heads=[2, 2],
            window_size=8,
        )
        assert model.__class__.__name__ == 'SwinFluidSR'

        # 同名别名也应该有效
        model2 = create_model(
            'swin_fluid_sr',
            in_channels=1,
            out_channels=1,
            img_size=16,
            upscale_factor=4,
            embed_dim=32,
            depths=[2, 2],
            num_heads=[2, 2],
            window_size=8,
        )
        assert model2.__class__.__name__ == 'SwinFluidSR'

    def test_eval_mode_deterministic(self):
        """eval 模式下同一输入两次前向得到相同输出"""
        model = self._create_model(drop_path_rate=0.1)
        model.eval()
        x = torch.randn(1, 1, 16, 16, device=self.device)
        y1 = model(x)
        y2 = model(x)
        assert torch.allclose(y1, y2, atol=1e-6), \
            "eval 模式下应具有确定性输出"
