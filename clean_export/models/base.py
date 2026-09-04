"""
基础模型接口（BaseModel）

定义统一的模型接口，确保所有模型都遵循相同的签名：
forward(x[B,C_in,H,W]) → y[B,C_out,H,W]

重要约定：
- 所有模型应继承 BaseModel
- 推荐使用 registry.py 的注册机制创建模型
- 这里提供一个向后兼容的 create_model（委托 registry.create_model）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """统一模型接口基类

    所有模型必须继承此类并实现 forward 方法，确保接口一致性。

    接口规范：
    - __init__(in_channels, out_channels, img_size, **kwargs)
    - forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
    - 允许通过 **kwargs 接收可选输入（coords, mask, fourier_pe 等），但不强制使用
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        **kwargs
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.img_size = int(img_size)

        # 存储模型配置（不做强约束）
        self.config: Dict[str, Any] = dict(kwargs)

        # 性能统计缓存
        self._param_count: Optional[int] = None
        self._flops: Optional[int] = None

    @property
    def in_ch(self) -> int:
        return self.in_channels

    @property
    def out_ch(self) -> int:
        return self.out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（coords, mask, fourier_pe 等）

        Returns:
            输出张量 [B, C_out, H, W]
        """
        raise NotImplementedError

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息（参数量、FLOPs等）"""
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters())

        info: Dict[str, Any] = {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "parameters": self._param_count,
            "parameters_M": self._param_count / 1e6,
        }

        if self._flops is not None:
            info["flops"] = self._flops
            info["flops_G"] = self._flops / 1e9

        return info

    def compute_flops(self, input_shape: Tuple[int, ...] | None = None) -> int:
        """计算 FLOPs（默认粗估；子类可重写以提供更精确实现）"""
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)

        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters())

        # 粗估：参数量 * batch * H * W
        b = int(input_shape[0])
        h = int(input_shape[2])
        w = int(input_shape[3])
        self._flops = int(self._param_count * b * h * w)
        return self._flops

    def get_memory_usage(self, batch_size: int = 1) -> Dict[str, float]:
        """估算显存使用量（MB，粗略）"""
        # 参数显存
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        # 激活显存（粗略估计：输入激活）
        activation_memory = batch_size * self.in_channels * (self.img_size**2) * 4 / 1024**2  # float32
        # 梯度显存（近似等于参数显存）
        grad_memory = param_memory
        total_memory = param_memory + activation_memory + grad_memory

        return {
            "parameters_MB": float(param_memory),
            "activations_MB": float(activation_memory),
            "gradients_MB": float(grad_memory),
            "total_MB": float(total_memory),
        }

    def load_pretrained(self, checkpoint_path: str, strict: bool = True) -> None:
        """加载预训练权重"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not strict:
            model_keys = set(self.state_dict().keys())
            ckpt_keys = set(state_dict.keys())

            extra_keys = ckpt_keys - model_keys
            for k in extra_keys:
                state_dict.pop(k, None)

            missing_keys = model_keys - ckpt_keys
            if missing_keys:
                print(f"[load_pretrained] Missing keys in checkpoint: {sorted(list(missing_keys))[:20]} ...")

        self.load_state_dict(state_dict, strict=strict)
        print(f"[load_pretrained] Loaded pretrained weights from: {checkpoint_path}")

    def freeze_encoder(self) -> None:
        """冻结编码器参数（如适用，子类可重写）"""
        return

    def unfreeze_all(self) -> None:
        """解冻所有参数"""
        for p in self.parameters():
            p.requires_grad = True


# -------------------------
# Backward-compatible factory
# -------------------------
def create_model(model_name_or_config: Union[str, Any] = None, **kwargs) -> nn.Module:
    if model_name_or_config is None:
        model_name_or_config = kwargs.pop("model_name", None)
    if model_name_or_config is None:
        raise TypeError("model_name_or_config or model_name is required")

    if isinstance(model_name_or_config, str):
        model_name = model_name_or_config
        model_params: Dict[str, Any] = dict(kwargs)
    else:
        config = model_name_or_config
        model_name = str(getattr(config, "name", "")).strip()
        if not model_name:
            raise ValueError("config.name is empty")

        model_params = {}
        if hasattr(config, "params"):
            model_params.update(dict(config.params))
        else:
            try:
                model_params.update({k: v for k, v in config.items() if k != "name"})
            except Exception:
                model_params.update(dict(config))

        if "kwargs" in model_params and isinstance(model_params["kwargs"], (dict,)):
            extra = dict(model_params["kwargs"])
            model_params.pop("kwargs", None)
            extra.update(model_params)
            model_params = extra

        if kwargs:
            model_params.update(kwargs)

    name = str(model_name).strip()
    if not name:
        raise ValueError("model_name is empty")

    from importlib import import_module

    import_module("models.spatial")
    import_module("models.temporal")

    lower = name.lower()

    if lower in {"arwrapper", "ar_wrapper"}:
        from .temporal.factory import create_model as temporal_create_model
        wrapper_keys = {
            "single_frame_model",
            "model",
            "model_name",
            "base_kwargs",
            "detach_rollout",
            "scheduled_sampling",
            "sampling_schedule",
            "teacher_forcing_ratio",
            "T_in",
            "T_out",
            "t_in",
            "t_out",
            "t_out_steps",
        }

        ar_config = model_params.pop("ar_config", None)
        if isinstance(ar_config, dict):
            for k, v in ar_config.items():
                model_params.setdefault(k, v)

        base_model = model_params.pop("base_model", None)
        if isinstance(base_model, str):
            model_params.setdefault("model_name", base_model)
        elif isinstance(base_model, dict):
            base_name = base_model.get("name", None)
            if base_name is not None:
                model_params.setdefault("model_name", base_name)
            merged_base_kwargs = dict(base_model)
            merged_base_kwargs.pop("name", None)
            existing_base_kwargs = model_params.pop("base_kwargs", None)
            if isinstance(existing_base_kwargs, dict):
                merged_base_kwargs.update(existing_base_kwargs)
            model_params["base_kwargs"] = merged_base_kwargs

        wrapper_kwargs: Dict[str, Any] = {}
        base_kwargs: Dict[str, Any] = {}
        for k, v in model_params.items():
            if k in wrapper_keys:
                wrapper_kwargs[k] = v
            else:
                base_kwargs[k] = v

        if "model_name" not in wrapper_kwargs:
            wrapper_kwargs["model_name"] = "SwinUNet"

        merged_base_kwargs = {}
        if isinstance(wrapper_kwargs.get("base_kwargs"), dict):
            merged_base_kwargs.update(wrapper_kwargs["base_kwargs"])
        merged_base_kwargs.update(base_kwargs)
        wrapper_kwargs["base_kwargs"] = merged_base_kwargs

        return temporal_create_model("ARWrapper", **wrapper_kwargs)

    temporal_names = {
        "swintemporal",
        "swin_temporal",
        "swintemporalnar",
        "swin_temporal_nar",
        "arnarwrapper",
        "ar_nar_wrapper",
        "physicstransformer",
        "physics_transformer",
        "temporalencoder",
        "temporalblock",
        "narpredictionhead",
        "sequentialspatiotemporal",
        "sequentialtrainer",
        "sequentialdcconsistency",
    }
    if lower in temporal_names:
        from .temporal.factory import create_model as temporal_create_model
        return temporal_create_model(name, **model_params)

    is_ar = lower.endswith("_ar") or lower.startswith("ar_")
    if is_ar:
        base_name = lower[:-3] if lower.endswith("_ar") else lower[3:]
        ar_keys = {
            "scheduled_sampling",
            "sampling_schedule",
            "detach_rollout",
            "teacher_forcing_ratio",
            "T_in",
            "T_out",
            "t_in",
            "t_out",
            "t_out_steps",
        }
        drop_keys = {
            "use_ar",
            "use_nar",
            "nar_cfg",
            "ar_cfg",
        }

        ar_kwargs: Dict[str, Any] = {}
        ar_config = model_params.pop("ar_config", None)
        if isinstance(ar_config, dict):
            for k, v in ar_config.items():
                if k in ar_keys and k not in ar_kwargs and k not in model_params:
                    ar_kwargs[k] = v

        for k in list(model_params.keys()):
            if k in ar_keys:
                ar_kwargs[k] = model_params.pop(k)

        for k in drop_keys:
            model_params.pop(k, None)

        from .registry import create_model as registry_create_model

        base_model = registry_create_model(base_name, **model_params)
        from .ar.wrapper import ARWrapper

        if "t_out" in ar_kwargs and "T_out" not in ar_kwargs:
            ar_kwargs["T_out"] = ar_kwargs.pop("t_out")
        if "t_in" in ar_kwargs and "T_in" not in ar_kwargs:
            ar_kwargs["T_in"] = ar_kwargs.pop("t_in")
        if "t_out_steps" in ar_kwargs and "T_out" not in ar_kwargs:
            ar_kwargs["T_out"] = ar_kwargs.pop("t_out_steps")

        return ARWrapper(single_frame_model=base_model, **ar_kwargs)

    from .registry import create_model as registry_create_model

    return registry_create_model(name, **model_params)


# -------------------------
# Utility functions (optional)
# -------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数（总参数量, 可训练参数量）"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total_params), int(trainable_params)


def get_model_size(model: nn.Module) -> float:
    """获取模型大小（MB）"""
    param_size = 0
    buffer_size = 0

    for p in model.parameters():
        param_size += p.numel() * p.element_size()

    for b in model.buffers():
        buffer_size += b.numel() * b.element_size()

    return float((param_size + buffer_size) / 1024**2)
