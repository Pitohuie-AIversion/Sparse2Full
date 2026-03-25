"""
基础时序模型类

提供统一的时序模型接口，遵循黄金法则：
1. 统一接口：forward(x[B,T,C,H,W]) -> y[B,T,C,H,W]
2. 一致性：观测算子H与训练DC完全一致
3. 可复现：相同配置+种子=相同结果
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseTemporalModel(nn.Module, ABC):
    """基础时序模型类

    所有时序模型必须继承此类，实现统一接口。
    支持AR（自回归）和NAR（非自回归）两种模式。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 图像尺寸
        T_in: 输入时间步数
        T_out: 输出时间步数
        mode: 模式 ('ar', 'nar', 'both')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        T_in: int = 1,
        T_out: int = 1,
        mode: str = "ar",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.T_in = T_in
        self.T_out = T_out
        self.mode = mode

        # 验证模式
        if mode not in ["ar", "nar", "both"]:
            raise ValueError(
                f"Unsupported mode: {mode}. Must be 'ar', 'nar', or 'both'"
            )

        logger.info(
            f"BaseTemporalModel: {in_channels}->{out_channels}, T_in={T_in}, T_out={T_out}, mode={mode}"
        )

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        T_out: int | None = None,
        teacher_forcing: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """统一前向传播接口

        Args:
            x: 输入张量 [B, T_in, C, H, W]
            T_out: 输出时间步数（如果为None，使用self.T_out）
            teacher_forcing: 教师信号 [B, T_out, C, H, W]
            return_dict: 是否返回字典格式结果

        Returns:
            如果return_dict=False: 输出张量 [B, T_out, C, H, W]
            如果return_dict=True: 字典包含输出和其他信息
        """
        pass

    def get_model_info(self) -> dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "T_in": self.T_in,
            "T_out": self.T_out,
            "mode": self.mode,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def compute_flops(self, input_shape: tuple[int, ...]) -> dict[str, int]:
        """计算FLOPs（子类可重写）"""
        # 基础FLOPs估算
        B, T, C, H, W = input_shape
        total_flops = B * T * C * H * W * self.total_parameters

        return {
            "total_flops": total_flops,
            "model_flops": total_flops,
            "temporal_flops": total_flops // T,
        }

    def get_memory_usage(
        self, batch_size: int = 1, T_out: int | None = None
    ) -> dict[str, float]:
        """估算显存使用量（MB）"""
        if T_out is None:
            T_out = self.T_out

        # 模型参数显存
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / (
            1024**2
        )

        # 激活值显存估算
        activation_memory = 0
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 粗略估算：每层输出大小 * 4字节 * 2（前向+反向）
                if hasattr(module, "out_features"):
                    activation_memory += module.out_features * 4 * 2 / (1024**2)
                elif hasattr(module, "out_channels"):
                    activation_memory += (
                        module.out_channels * self.img_size**2 * 4 * 2 / (1024**2)
                    )

        # 时序相关显存
        temporal_memory = T_out * self.in_channels * self.img_size**2 * 4 / (1024**2)

        return {
            "parameters_mb": param_memory,
            "activations_mb": activation_memory,
            "temporal_mb": temporal_memory,
            "total_mb": param_memory + activation_memory + temporal_memory,
        }

    def validate_input(self, x: torch.Tensor) -> None:
        """验证输入格式"""
        if x.dim() != 5:
            raise ValueError(f"Input must be 5D tensor [B, T, C, H, W], got {x.dim()}D")

        B, T, C, H, W = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, got {C}"
            )

        if T != self.T_in:
            logger.warning(f"Input time steps mismatch: expected {self.T_in}, got {T}")

        if H != self.img_size or W != self.img_size:
            logger.warning(
                f"Input spatial size mismatch: expected {self.img_size}x{self.img_size}, got {H}x{W}"
            )

    def set_mode(self, mode: str) -> None:
        """设置运行模式"""
        if mode not in ["ar", "nar", "both"]:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        logger.info(f"Model mode set to: {mode}")

    def get_receptive_field(self) -> int:
        """获取时序感受野大小（子类可重写）"""
        return self.T_in


class ARMixin:
    """AR（自回归）模式混入类"""

    def ar_forward(
        self, x: torch.Tensor, T_out: int, teacher_forcing: torch.Tensor | None = None
    ) -> torch.Tensor:
        """AR模式前向传播"""
        raise NotImplementedError("AR forward must be implemented by subclass")

    def set_teacher_forcing_ratio(self, ratio: float) -> None:
        """设置teacher forcing比例"""
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Teacher forcing ratio must be in [0, 1], got {ratio}")

        if hasattr(self, "teacher_forcing_ratio"):
            self.teacher_forcing_ratio = ratio
        else:
            logger.warning("Model does not support teacher forcing ratio")


class NARMixin:
    """NAR（非自回归）模式混入类"""

    def nar_forward(self, x: torch.Tensor, T_out: int) -> torch.Tensor:
        """NAR模式前向传播"""
        raise NotImplementedError("NAR forward must be implemented by subclass")


class TemporalConsistencyMixin:
    """时序一致性检查混入类"""

    def check_temporal_consistency(
        self, pred_seq: torch.Tensor, gt_seq: torch.Tensor | None = None
    ) -> dict[str, float]:
        """检查时序一致性"""
        if pred_seq.dim() != 5:
            raise ValueError(
                f"Predicted sequence must be 5D [B, T, C, H, W], got {pred_seq.dim()}D"
            )

        B, T, C, H, W = pred_seq.shape

        # 计算时序差分
        diff = pred_seq[:, 1:] - pred_seq[:, :-1]  # [B, T-1, C, H, W]

        # 时序平滑性指标
        temporal_smoothness = torch.mean(torch.abs(diff)).item()

        # 时序方差
        temporal_variance = torch.var(pred_seq, dim=1).mean().item()

        consistency_info = {
            "temporal_smoothness": temporal_smoothness,
            "temporal_variance": temporal_variance,
            "sequence_length": T,
        }

        # 如果有真实值，计算额外指标
        if gt_seq is not None:
            if gt_seq.shape != pred_seq.shape:
                logger.warning(
                    f"Shape mismatch: pred {pred_seq.shape} vs gt {gt_seq.shape}"
                )
            else:
                # 计算时序一致性误差
                temporal_error = torch.mean(torch.abs(pred_seq - gt_seq)).item()
                consistency_info["temporal_error"] = temporal_error

        return consistency_info


# 辅助函数
def create_temporal_model(
    model_type: str,
    in_channels: int,
    out_channels: int,
    img_size: int,
    T_in: int = 1,
    T_out: int = 1,
    mode: str = "ar",
    **kwargs,
) -> BaseTemporalModel:
    """工厂函数：创建时序模型"""

    from models.temporal.factory import create_model

    # 基础配置
    base_config = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "img_size": img_size,
        "T_in": T_in,
        "T_out": T_out,
        "mode": mode,
    }

    # 合并额外配置
    base_config.update(kwargs)

    try:
        model = create_model(model_type, **base_config)
        logger.info(f"Created temporal model: {model_type}")
        return model
    except Exception as e:
        logger.error(f"Failed to create temporal model {model_type}: {e}")
        raise


def validate_temporal_model_consistency(
    spatial_model: nn.Module,
    temporal_model: BaseTemporalModel,
    test_input: torch.Tensor,
) -> dict[str, bool]:
    """验证时序模型与空间模型的一致性"""

    consistency_results = {}

    try:
        # 检查输入格式兼容性
        if test_input.dim() == 4:  # 空间模型输入 [B, C, H, W]
            # 为时序模型添加时间维度
            temporal_input = test_input.unsqueeze(1)  # [B, 1, C, H, W]
        elif test_input.dim() == 5:  # 时序模型输入 [B, T, C, H, W]
            temporal_input = test_input
            # 为空间模型移除时间维度
            test_input = test_input[:, -1]  # 使用最后一帧
        else:
            raise ValueError(f"Unsupported input dimension: {test_input.dim()}")

        # 空间模型推理（单帧）
        with torch.no_grad():
            spatial_output = spatial_model(test_input)

        # 时序模型推理（单步）
        with torch.no_grad():
            temporal_output = temporal_model(temporal_input, T_out=1)
            if temporal_output.dim() == 5:
                temporal_output = temporal_output[:, -1]  # 取最后一帧

        # 检查输出形状一致性
        consistency_results["shape_consistency"] = (
            spatial_output.shape == temporal_output.shape
        )

        # 检查输出值范围一致性
        spatial_range = torch.max(spatial_output) - torch.min(spatial_output)
        temporal_range = torch.max(temporal_output) - torch.min(temporal_output)
        range_ratio = abs(spatial_range.item() - temporal_range.item()) / max(
            spatial_range.item(), 1e-8
        )
        consistency_results["range_consistency"] = range_ratio < 0.1  # 10%误差容忍

        # 检查统计特性一致性
        spatial_mean = torch.mean(spatial_output).item()
        temporal_mean = torch.mean(temporal_output).item()
        mean_diff_ratio = abs(spatial_mean - temporal_mean) / max(
            abs(spatial_mean), 1e-8
        )
        consistency_results["mean_consistency"] = mean_diff_ratio < 0.1

        logger.info(f"Temporal model consistency check: {consistency_results}")

    except Exception as e:
        logger.error(f"Consistency check failed: {e}")
        consistency_results["error"] = str(e)

    return consistency_results


# 测试用具体实现类
class TestTemporalModel(BaseTemporalModel, TemporalConsistencyMixin):
    """测试用的具体时序模型实现"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 简单的测试网络
        self.temporal_conv = nn.Conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(self.T_in, 3, 3),
            padding=(0, 1, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        T_out: int | None = None,
        teacher_forcing: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """测试前向传播"""
        # 验证输入
        self.validate_input(x)

        # 简单的时序卷积
        B, T, C, H, W = x.shape

        # 调整维度: [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        # 应用3D卷积
        output = self.temporal_conv(x)  # [B, C_out, 1, H, W]

        # 移除时间维度并复制到多个时间步
        output = output.squeeze(2)  # [B, C_out, H, W]

        if T_out is None:
            T_out = self.T_out

        # 复制到多个时间步
        output = output.unsqueeze(1).repeat(
            1, T_out, 1, 1, 1
        )  # [B, T_out, C_out, H, W]

        if return_dict:
            return {
                "output": output,
                "temporal_features": output,
                "model_info": self.get_model_info(),
            }
        else:
            return output


if __name__ == "__main__":
    # 基础测试
    import logging

    logging.basicConfig(level=logging.INFO)

    print("🧪 测试基础时序模型类...")

    # 创建测试配置
    config = {
        "in_channels": 2,
        "out_channels": 2,
        "img_size": 128,
        "T_in": 1,
        "T_out": 3,
        "mode": "ar",
    }

    # 测试模型信息
    print(f"Model config: {config}")

    # 测试内存使用估算
    base_model = TestTemporalModel(**config)
    memory_info = base_model.get_memory_usage(batch_size=2)
    print(f"Memory usage: {memory_info}")

    # 测试模型信息
    model_info = base_model.get_model_info()
    print(f"Model info: {model_info}")

    # 测试前向传播
    test_input = torch.randn(2, 1, 2, 128, 128)  # [B, T, C, H, W]
    output = base_model(test_input, T_out=3)
    print(f"Input shape: {test_input.shape}, Output shape: {output.shape}")

    # 测试字典返回格式
    output_dict = base_model(test_input, T_out=3, return_dict=True)
    print(f"Dict output keys: {list(output_dict.keys())}")

    # 测试一致性检查
    consistency_info = base_model.check_temporal_consistency(output)
    print(f"Temporal consistency: {consistency_info}")

    print("✅ 基础时序模型类测试完成！")
