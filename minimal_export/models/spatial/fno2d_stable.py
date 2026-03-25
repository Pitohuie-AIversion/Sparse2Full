"""
Stabilized FNO 2D 模型（工程增强版）

核心架构来源（FNO 原型）：
- Zongyi Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", arXiv:2010.08895
  https://arxiv.org/abs/2010.08895

说明：
- 本文件中的 "stable"（NaN/Inf 检测、fallback、禁用 AMP 干扰、谱归一化、初始化、归一化策略等）
  属于工程增强，并非逐行复刻某一篇“官方稳定版”论文实现。建议在论文/报告中明确写为：
  "FNO-style spectral operator with stability-oriented engineering safeguards."
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class StableSpectralConv2d(nn.Module):
    """数值稳定的 2D 频谱卷积层（FNO-style spectral conv）"""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 用 Python float，避免 device/dtype 隐患
        scale = math.sqrt(1.0 / (in_channels * out_channels))

        # 复数权重参数（与常见 FNO 实现一致：complex weights）
        # 这里用 complex(real, imag) 显式构造，避免 view_as_complex 的形状/类型坑
        w1r = scale * torch.randn(in_channels, out_channels, modes1, modes2)
        w1i = scale * torch.randn(in_channels, out_channels, modes1, modes2)
        w2r = scale * torch.randn(in_channels, out_channels, modes1, modes2)
        w2i = scale * torch.randn(in_channels, out_channels, modes1, modes2)

        self.weights1 = nn.Parameter(torch.complex(w1r, w1i))
        self.weights2 = nn.Parameter(torch.complex(w2r, w2i))

    @staticmethod
    def _isfinite(x: torch.Tensor) -> bool:
        # 对 complex 也适用：会检查实部/虚部
        return torch.isfinite(x).all().item()

    def stable_compl_mul2d(
        self, input: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """数值稳定的复数乘法：einsum 实现"""
        if not self._isfinite(input):
            input = torch.nan_to_num(input, nan=0.0, posinf=1e6, neginf=-1e6)
        if not self._isfinite(weights):
            weights = torch.nan_to_num(weights, nan=0.0, posinf=1e6, neginf=-1e6)

        # [B, in, Hf, Wf] x [in, out, Hf, Wf] -> [B, out, Hf, Wf]
        out = torch.einsum("bixy,ioxy->boxy", input, weights)

        if not self._isfinite(out):
            out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, H, W]
        return: [B, out_channels, H, W]
        """
        B, _, H, W = x.shape

        # 兼容 CPU：CUDA autocast 仅在 cuda 上用；这里是“禁用”AMP 的上下文
        amp_ctx = torch.cuda.amp.autocast(enabled=False) if x.is_cuda else nullcontext()
        with amp_ctx:
            # 如果你确实需要 fp64 稳定性，可以保留 double；否则建议改为 float() + complex64
            x_work = x.double()

            if not self._isfinite(x_work):
                x_work = torch.nan_to_num(x_work, nan=0.0, posinf=1e6, neginf=-1e6)

            # FFT（norm='ortho'：减少尺度爆炸风险；norm 语义见 PyTorch torch.fft 文档体系）
            try:
                x_ft = torch.fft.rfft2(x_work, norm="ortho")
            except Exception:
                return torch.zeros(
                    B, self.out_channels, H, W, dtype=torch.float32, device=x.device
                )

            if not self._isfinite(x_ft):
                x_ft = torch.nan_to_num(x_ft, nan=0.0, posinf=1e6, neginf=-1e6)

            out_ft = torch.zeros(
                B,
                self.out_channels,
                H,
                W // 2 + 1,
                dtype=torch.complex128,
                device=x.device,
            )

            m1 = min(self.modes1, H)
            m2 = min(self.modes2, W // 2 + 1)

            try:
                x1 = x_ft[:, :, :m1, :m2].to(torch.complex128)
                w1 = self.weights1[:, :, :m1, :m2].to(torch.complex128)
                out_ft[:, :, :m1, :m2] = self.stable_compl_mul2d(x1, w1)

                x2 = x_ft[:, :, -m1:, :m2].to(torch.complex128)
                w2 = self.weights2[:, :, :m1, :m2].to(torch.complex128)
                out_ft[:, :, -m1:, :m2] = self.stable_compl_mul2d(x2, w2)
            except Exception:
                # 保底：保持 out_ft 为 0，继续 IFFT 输出 0（形状正确）
                pass

            if not self._isfinite(out_ft):
                out_ft = torch.nan_to_num(out_ft, nan=0.0, posinf=1e6, neginf=-1e6)

            try:
                y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
                y = y.float()
            except Exception:
                return torch.zeros(
                    B, self.out_channels, H, W, dtype=torch.float32, device=x.device
                )

        return y


class StableFNO2d(BaseModel):
    """数值稳定的 FNO2d（FNO-style spectral operator with safeguards）"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_layers: int = 4,
        activation: str = "gelu",
        spectral_norm: bool = True,
        gradient_clip: float = 1.0,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm
        self.gradient_clip = gradient_clip  # 建议放在 trainer 里真正执行 clip

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.fc0 = nn.Linear(in_channels + 2, self.width)
        self.input_norm = nn.LayerNorm(self.width)

        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(self.n_layers):
            self.conv_layers.append(
                StableSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            )
            conv1x1 = nn.Conv2d(self.width, self.width, 1)
            if self.spectral_norm:
                conv1x1 = nn.utils.spectral_norm(conv1x1)
            self.w_layers.append(conv1x1)
            self.layer_norms.append(
                nn.GroupNorm(num_groups=self.width, num_channels=self.width)
            )

        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, out_channels)
        self.dropout = nn.Dropout(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu", a=0.1
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_grid(
        self, shape: tuple[int, int, int, int], device: torch.device
    ) -> torch.Tensor:
        B, _, H, W = shape
        gridx = (
            torch.linspace(-1, 1, H, device=device, dtype=torch.float32)
            .view(1, H, 1, 1)
            .repeat(B, 1, W, 1)
        )
        gridy = (
            torch.linspace(-1, 1, W, device=device, dtype=torch.float32)
            .view(1, 1, W, 1)
            .repeat(B, H, 1, 1)
        )
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B = x.shape[0]
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        grid = self.get_grid(x.shape, x.device)

        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        x = torch.cat((x, grid), dim=-1)  # [B,H,W,C+2]

        x = self.fc0(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)  # [B,width,H,W]

        x_residual = x  # 不要 clone；节省显存/时间

        for i in range(self.n_layers):
            try:
                x1 = self.conv_layers[i](x)
                x2 = self.w_layers[i](x)
                x = x1 + x2
                x = self.layer_norms[i](x)

                if i < self.n_layers - 1:
                    x = self.activation(x)
                    x = self.dropout(x)

                if not torch.isfinite(x).all():
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

            except Exception:
                # 关键修复：不要 return（会绕过 fc1/fc2 且通道不对）
                # 回退到 residual 并跳出循环，确保最终输出通道正确
                x = x_residual
                break

        x = x + 0.1 * x_residual

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)

        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        return x


StableFNOModel = StableFNO2d
