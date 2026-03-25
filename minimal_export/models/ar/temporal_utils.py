"""时间序列处理工具模块

提供外部队列式时间序列预测功能，支持教师强制和自回归模式。
与统一接口的ARWrapper配合使用。
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def autoregressive_predict(
    model: nn.Module,
    x_seq: torch.Tensor,
    T_out: int,
    teacher: torch.Tensor | None = None,
    train_mode: bool = True,
    scheduled_sampling_prob: float = 0.0,
    detach_rollout: bool = True,
) -> torch.Tensor:
    """外部队列式时间序列预测 - 与统一接口ARWrapper配合使用

    Args:
        model: ARWrapper实例（统一接口）
        x_seq: 输入序列 [B, T_in, C, H, W]
        T_out: 输出时间步数
        teacher: 教师强制序列 [B, T_out, C, H, W]
        train_mode: 训练模式（True）或推理模式（False）
        scheduled_sampling_prob: scheduled sampling概率
        detach_rollout: 推理时是否断开梯度

    Returns:
        预测序列 [B, T_out, C, H, W]

    Example:
        >>> # 创建统一接口模型
        >>> model = ARWrapper(base_model)  # 只有统一forward(x)->y接口
        >>> # 时间序列预测
        >>> output = autoregressive_predict(model, x_seq, T_out=10, teacher=teacher_seq)
    """
    B, T_in, C, H, W = x_seq.shape
    device = x_seq.device

    # 初始化输出序列
    y_seq = torch.zeros(B, T_out, C, H, W, device=device)

    # 当前输入帧（使用输入序列的最后一帧）
    x_current = x_seq[:, -1, :, :, :]  # [B, C, H, W]

    # 推理模式或没有teacher信号时使用纯自回归
    if (not train_mode) or (teacher is None):
        # 纯自回归模式
        for t in range(T_out):
            # 使用统一接口进行单帧预测
            y_current = model(x_current)  # [B, C, H, W]
            y_seq[:, t, :, :, :] = y_current

            # 准备下一帧输入（使用当前预测）
            x_current = y_current.detach() if detach_rollout else y_current

    else:
        # 训练模式：教师强制 + scheduled sampling
        for t in range(T_out):
            # 使用统一接口进行单帧预测
            y_current = model(x_current)  # [B, C, H, W]
            y_seq[:, t, :, :, :] = y_current

            # 准备下一帧输入
            if t < T_out - 1:  # 不是最后一步
                if (
                    scheduled_sampling_prob > 0
                    and torch.rand(1).item() < scheduled_sampling_prob
                ):
                    # scheduled sampling：使用模型预测
                    x_current = y_current.detach()
                else:
                    # 教师强制：使用真值
                    x_current = teacher[:, t, :, :, :]

    return y_seq  # [B, T_out, C, H, W]


def validate_temporal_inputs(
    x_seq: torch.Tensor, T_out: int, teacher: torch.Tensor | None = None
) -> None:
    """验证时间序列输入的有效性

    Args:
        x_seq: 输入序列
        T_out: 输出时间步数
        teacher: 教师序列

    Raises:
        ValueError: 输入格式无效时
    """
    if x_seq.dim() != 5:
        raise ValueError(f"Input sequence must be 5D [B,T,C,H,W], got {x_seq.dim()}D")

    if T_out <= 0:
        raise ValueError(f"T_out must be positive, got {T_out}")

    if teacher is not None:
        if teacher.dim() != 5:
            raise ValueError(
                f"Teacher sequence must be 5D [B,T,C,H,W], got {teacher.dim()}D"
            )

        if teacher.size(0) != x_seq.size(0):
            raise ValueError(
                f"Batch size mismatch: input {x_seq.size(0)} vs teacher {teacher.size(0)}"
            )

        if teacher.size(1) < T_out:
            raise ValueError(f"Teacher sequence too short: {teacher.size(1)} < {T_out}")


def create_temporal_model_wrapper(model: nn.Module, **kwargs) -> nn.Module:
    """创建支持时间序列处理的模型包装器

    Args:
        model: 基础模型（统一接口）
        **kwargs: 额外参数

    Returns:
        包装后的模型
    """
    # 如果模型已经是ARWrapper，直接返回
    if hasattr(model, "autoregressive_predict"):
        return model

    # 为模型添加时间序列预测方法
    def temporal_predict(
        self, x_seq, T_out, teacher=None, train_mode=True, **pred_kwargs
    ):
        return autoregressive_predict(
            self, x_seq, T_out, teacher, train_mode, **pred_kwargs
        )

    # 绑定方法到模型
    model.autoregressive_predict = temporal_predict.__get__(model)

    return model
