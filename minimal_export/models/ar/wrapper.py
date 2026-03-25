"""AR包装器模块

将单帧模型包装成多步自回归预测模型，支持：

* 训练：teacher forcing（每步用真值作为下一步输入）

* 推理：roll-out（每步用上一步预测作为下一步输入）

兼容现有的baseline/target标准化域处理流程。
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ARWrapper(nn.Module):
    """自回归包装器

    将单帧模型包装成多步自回归预测：
    - 训练：teacher forcing（每步用真值作为下一步输入）
    - 推理：roll-out（每步用上一步预测作为下一步输入）

    Args:
        single_frame_model: 单帧预测模型
        detach_rollout: 推理/评估阶段是否断开梯度（避免梯度累积）
        scheduled_sampling: 是否启用scheduled sampling
        sampling_schedule: scheduled sampling的调度参数
    """

    def __init__(
        self,
        single_frame_model: nn.Module | None = None,
        detach_rollout: bool = True,
        scheduled_sampling: bool = False,
        sampling_schedule: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__()
        # 兼容不同构造方式：
        # 1) 直接传入单帧模型：single_frame_model
        # 2) 使用参数名 model 传入单帧模型（兼容旧测试）
        # 3) 通过工厂方法创建：model_name + base_kwargs
        base_model = single_frame_model
        if (
            base_model is None
            and "model" in kwargs
            and isinstance(kwargs["model"], nn.Module)
        ):
            base_model = kwargs["model"]
        if base_model is None and ("model_name" in kwargs or "base_kwargs" in kwargs):
            try:
                from models import create_model as _create_model

                model_name = kwargs.get("model_name", "SwinUNet")
                base_kwargs = kwargs.get("base_kwargs", {})
                base_model = _create_model(model_name, **base_kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to create base model for ARWrapper: {e}")
        if base_model is None:
            raise ValueError(
                "ARWrapper requires a base single-frame model via 'single_frame_model' or 'model'."
            )

        self.m = base_model
        self.detach_rollout = detach_rollout
        self.scheduled_sampling = scheduled_sampling

        # 统一接口标识
        self._unified_interface = True
        # 兼容 teacher_forcing_ratio（旧接口）：将其映射为scheduled sampling的常量概率
        tfr = kwargs.get("teacher_forcing_ratio", None)
        if tfr is not None:
            try:
                tfr = float(tfr)
            except Exception:
                tfr = 0.5
            # 使用常量采样概率（使用预测的概率 = 1 - teacher_forcing_ratio）
            if sampling_schedule is None:
                sampling_schedule = {
                    "start_prob": max(0.0, min(1.0, 1.0 - tfr)),
                    "end_prob": max(0.0, min(1.0, 1.0 - tfr)),
                    "schedule_type": "constant",
                }
            self.scheduled_sampling = True
        # 兼容 T_in/T_out（旧接口），仅存储不强制使用
        self.T_in = kwargs.get("T_in", None)
        self.T_out = kwargs.get("T_out", None)

        # Scheduled sampling参数
        if sampling_schedule is None:
            sampling_schedule = {
                "start_prob": 0.0,
                "end_prob": 0.5,
                "schedule_type": "linear",
            }
        self.sampling_schedule = sampling_schedule
        self.current_epoch = 0
        self.total_epochs = 100  # 默认值，会在训练时更新

        # 继承基础模型的属性
        if hasattr(single_frame_model, "in_channels"):
            self.in_channels = single_frame_model.in_channels
        if hasattr(single_frame_model, "out_channels"):
            self.out_channels = single_frame_model.out_channels
        if hasattr(single_frame_model, "img_size"):
            self.img_size = single_frame_model.img_size

    def set_epoch(self, epoch: int, total_epochs: int = None):
        """设置当前epoch，用于scheduled sampling"""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs

    def get_sampling_prob(self) -> float:
        """获取当前的sampling概率"""
        if not self.scheduled_sampling:
            return 0.0

        progress = self.current_epoch / self.total_epochs
        start_prob = self.sampling_schedule["start_prob"]
        end_prob = self.sampling_schedule["end_prob"]

        if self.sampling_schedule["schedule_type"] == "linear":
            return start_prob + (end_prob - start_prob) * progress
        elif self.sampling_schedule["schedule_type"] == "exponential":
            # 指数调度
            return start_prob * (end_prob / start_prob) ** progress
        else:
            return start_prob

    @torch.no_grad()
    def _rollout(self, x0: torch.Tensor, T_out: int) -> torch.Tensor:
        """推理：以x0作为第1步输入，串行滚动输出T_out帧

        Args:
            x0: 初始输入 (B,C,H,W)
            T_out: 输出时间步数

        Returns:
            预测序列 (B,T_out,C,H,W)
        """
        self.m.eval()
        last = x0
        outs = []

        for t in range(T_out):
            y = self.m(last)  # (B,C,H,W)
            outs.append(y.unsqueeze(1))  # (B,1,C,H,W)

            # 准备下一步输入
            last = y if not self.detach_rollout else y.detach()

        return torch.cat(outs, dim=1)  # (B,T_out,C,H,W)

    def _unpack_input(self, x_packed: torch.Tensor) -> torch.Tensor:
        """解包 [baseline, coords, mask] 格式输入

        Args:
            x_packed: 打包输入 [B, C_packed, H, W]

        Returns:
            解包后的输入 [B, C_model, H, W]
        """
        # 获取模型期望的输入通道数
        model_in_ch = getattr(self.m, "in_channels", None) or getattr(
            self.m, "in_ch", None
        )
        if model_in_ch is None:
            # 如果无法获取，使用打包输入的前半部分
            model_in_ch = x_packed.size(1) // 2

        # 截取前 model_in_ch 个通道作为模型输入
        return x_packed[:, :model_in_ch, :, :]

    def __call__(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        # 智能参数解析：处理位置参数中的 teacher
        teacher = None
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            # 假设第一个位置参数是 teacher/target
            teacher = args[0]
            # 从 args 中移除 teacher，避免重复传递
            args = args[1:]

        # 同样检查 kwargs 中的 teacher
        if "teacher" in kwargs:
            teacher = kwargs.pop("teacher")

        # 检查是否需要进行时序预测 (如果有 teacher 或 explicit T_out)
        if teacher is not None or any(k in kwargs for k in ("T_out", "train_mode")):
            T_out = kwargs.pop("T_out", None)
            train_mode = kwargs.pop("train_mode", self.training)

            x_seq = x
            if isinstance(x, torch.Tensor) and x.dim() == 4:
                x_seq = x.unsqueeze(1)

            if T_out is None:
                if teacher is not None and isinstance(teacher, torch.Tensor):
                    # 如果 teacher 是 5D [B, T, C, H, W]，推断 T_out
                    if teacher.dim() == 5:
                        T_out = int(teacher.size(1))
                    # 如果 teacher 是 4D，可能只是单步 target，T_out=1
                    elif teacher.dim() == 4:
                        T_out = 1
                elif self.T_out is not None:
                    T_out = int(self.T_out)

            # 如果推断出 T_out，且处于训练模式或有明确指令，则进行 AR 预测
            if T_out is not None:
                return self.autoregressive_predict(
                    x_seq=x_seq,
                    T_out=int(T_out),
                    teacher=teacher,
                    train_mode=bool(train_mode),
                    **kwargs,
                )

        # Fallback: 如果没有识别出时序意图，且没有剩余 args，则调用单帧 forward
        if not args:
            return super().__call__(x)

        # 如果还有其他未处理的位置参数，只能尝试透传（可能会报错，但这是用户用法的责任）
        return super().__call__(x, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Unified interface requires 4D input")

        model_in_ch = getattr(self.m, "in_channels", None) or getattr(
            self.m, "in_ch", None
        )
        if model_in_ch is not None and x.size(1) > model_in_ch:
            x = self._unpack_input(x)
        return self.m(x)

    def autoregressive_predict(
        self,
        x_seq: torch.Tensor,
        T_out: int,
        teacher: torch.Tensor | None = None,
        train_mode: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """时间序列预测 - 兼容旧接口

        Args:
            x_seq: 输入序列 [B, T_in, C, H, W]
            T_out: 输出时间步数
            teacher: 教师强制序列 [B, T_out, C, H, W]
            train_mode: 训练模式
            **kwargs: 额外参数（scheduled_sampling_prob, detach_rollout等）

        Returns:
            预测序列 [B, T_out, C, H, W]

        Note:
            此方法提供向后兼容性，推荐使用外部的autoregressive_predict函数
        """
        from .temporal_utils import autoregressive_predict as external_predict

        return external_predict(
            model=self,
            x_seq=x_seq,
            T_out=T_out,
            teacher=teacher,
            train_mode=train_mode,
            scheduled_sampling_prob=(
                self.get_sampling_prob() if self.scheduled_sampling else 0.0
            ),
            detach_rollout=self.detach_rollout,
            **kwargs,
        )

    def get_model_info(self) -> dict[str, Any]:
        """获取模型信息"""
        base_info = {}
        if hasattr(self.m, "get_model_info"):
            base_info = self.m.get_model_info()

        # 添加AR包装器信息
        ar_info = {
            "model_type": "AR_Wrapper",
            "base_model": base_info.get("model_type", type(self.m).__name__),
            "detach_rollout": self.detach_rollout,
            "scheduled_sampling": self.scheduled_sampling,
            "unified_interface": getattr(self, "_unified_interface", False),
            "interface_version": "unified_v1.0",
        }

        # 合并信息
        base_info.update(ar_info)
        return base_info

    def compute_flops(self, input_shape: tuple = None) -> int:
        """计算FLOPs（粗略估计）

        Args:
            input_shape: 输入形状 (B,C,H,W)

        Returns:
            FLOPs数量
        """
        if hasattr(self.m, "compute_flops"):
            base_flops = self.m.compute_flops(input_shape)
        else:
            # 简单估计
            if input_shape is None:
                input_shape = (
                    1,
                    getattr(self, "in_channels", 4),
                    getattr(self, "img_size", 256),
                    getattr(self, "img_size", 256),
                )

            # 估计基础模型的FLOPs
            param_count = sum(p.numel() for p in self.m.parameters())
            base_flops = param_count * input_shape[0] * input_shape[2] * input_shape[3]

        # AR包装器的FLOPs是基础模型的T_out倍（串行执行）
        # 这里使用默认的T_out=3进行估计
        return base_flops * 3

    def get_memory_usage(self, batch_size: int = 1, T_out: int = 3) -> dict[str, float]:
        """估算显存使用量

        Args:
            batch_size: 批次大小
            T_out: 输出时间步数

        Returns:
            显存使用量信息（MB）
        """
        if hasattr(self.m, "get_memory_usage"):
            base_memory = self.m.get_memory_usage(batch_size)
        else:
            # 简单估计
            param_memory = (
                sum(p.numel() * p.element_size() for p in self.m.parameters()) / 1024**2
            )
            activation_memory = (
                batch_size
                * getattr(self, "in_channels", 4)
                * getattr(self, "img_size", 256) ** 2
                * 4
                / 1024**2
            )
            base_memory = {
                "parameters_MB": param_memory,
                "activations_MB": activation_memory,
                "gradients_MB": param_memory,
                "total_MB": param_memory * 2 + activation_memory,
            }

        # AR包装器需要额外的序列存储空间
        sequence_memory = T_out * base_memory["activations_MB"]

        return {
            "parameters_MB": base_memory["parameters_MB"],
            "activations_MB": base_memory["activations_MB"] + sequence_memory,
            "gradients_MB": base_memory["gradients_MB"],
            "sequence_MB": sequence_memory,
            "total_MB": base_memory["total_MB"] + sequence_memory,
        }

    def load_pretrained(self, checkpoint_path: str, strict: bool = True) -> None:
        """加载预训练权重（仅加载基础模型）"""
        if hasattr(self.m, "load_pretrained"):
            self.m.load_pretrained(checkpoint_path, strict)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # 只加载基础模型的权重
            self.m.load_state_dict(state_dict, strict=strict)
            logger.info(
                f"Loaded pretrained weights for base model from {checkpoint_path}"
            )

    def freeze_encoder(self) -> None:
        """冻结编码器参数（如果基础模型支持）"""
        if hasattr(self.m, "freeze_encoder"):
            self.m.freeze_encoder()

    def unfreeze_all(self) -> None:
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self) -> tuple:
        """统计模型参数

        Returns:
            (总参数量, 可训练参数量)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def get_flops(self, input_shape: tuple = None) -> int:
        """计算FLOPs（兼容接口）

        Args:
            input_shape: 输入形状 (B,T_in,C,H,W) 或 (B,C,H,W)

        Returns:
            FLOPs数量
        """
        if input_shape is not None and len(input_shape) == 5:
            # 如果是5维输入，转换为4维给基础模型
            base_input_shape = (
                input_shape[0],
                input_shape[2],
                input_shape[3],
                input_shape[4],
            )
        else:
            base_input_shape = input_shape

        return self.compute_flops(base_input_shape)
