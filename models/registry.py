from typing import Dict, Any, Type, Optional, List, Union
import inspect
import logging
import torch.nn as nn
from .base import BaseModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[Any]] = {}
MODEL_ALIASES: Dict[str, str] = {}
DEPRECATED_ALIASES: Dict[str, str] = {
    "restormerlite": "cnn_attn_lite",
}


def register_model(name: str = None, aliases: List[str] = None):
    """模型注册装饰器，支持继承自 nn.Module 的所有空间与时序模型"""
    def decorator(cls):
        if not issubclass(cls, nn.Module):
            raise TypeError(f"Registered model must inherit nn.Module, got {cls}")

        model_name = (name if name is not None else cls.__name__).strip()
        if not model_name:
            raise ValueError("Model name is empty")

        if model_name in MODEL_REGISTRY:
            # 允许重复注册同一类（如多次导入同一模块）
            if MODEL_REGISTRY[model_name] is cls:
                return cls
            raise ValueError(f"Model {model_name} is already registered to a different class!")
        if model_name in MODEL_ALIASES:
            raise ValueError(f"Model name {model_name} conflicts with an existing alias -> {MODEL_ALIASES[model_name]}")

        MODEL_REGISTRY[model_name] = cls

        if aliases:
            for alias in aliases:
                alias = str(alias).strip()
                if not alias:
                    raise ValueError(f"Empty alias for model {model_name}")

                if alias in MODEL_REGISTRY and MODEL_REGISTRY[alias] is not cls:
                    raise ValueError(f"Alias {alias} conflicts with an existing canonical model name.")
                if alias in MODEL_ALIASES and MODEL_ALIASES[alias] != model_name:
                    raise ValueError(f"Alias {alias} is already used by {MODEL_ALIASES[alias]}")

                MODEL_ALIASES[alias] = model_name

        return cls
    return decorator


def resolve_model_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("Model name is empty")

    if raw in MODEL_REGISTRY:
        return raw
    if raw in MODEL_ALIASES:
        return MODEL_ALIASES[raw]

    lowered = raw.lower()
    if lowered in DEPRECATED_ALIASES:
        return DEPRECATED_ALIASES[lowered]

    if lowered in MODEL_REGISTRY:
        return lowered
    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]

    for k in MODEL_REGISTRY.keys():
        if k.lower() == lowered:
            return k
    for k, v in MODEL_ALIASES.items():
        if k.lower() == lowered:
            return v

    raise ValueError(f"Model '{raw}' not found in registry. Registered models: {list(MODEL_REGISTRY.keys())[:15]}...")


def create_model(
    model_name_or_config: Union[str, Any] = None,
    strict: bool = False,
    verbose: bool = True,
    **kwargs
) -> nn.Module:
    """统一模型创建工厂函数
    
    支持以下调用形态：
    1. 字符串名称 + 参数：create_model('SwinUNet', in_channels=1, out_channels=1)
    2. 配置字典/DictConfig：create_model(config.model)
    3. AR 包装器与时序模型透传
    """
    if model_name_or_config is None:
        model_name_or_config = kwargs.pop("model_name", None)
    if model_name_or_config is None:
        raise TypeError("model_name_or_config or model_name is required")

    if isinstance(model_name_or_config, str):
        raw_name = model_name_or_config.strip()
        model_params: Dict[str, Any] = dict(kwargs)
    else:
        config = model_name_or_config
        if hasattr(config, "model"):
            config = getattr(config, "model")

        raw_name = ""
        if hasattr(config, "name"):
            raw_name = str(config.name).strip()
        elif isinstance(config, dict) and "name" in config:
            raw_name = str(config["name"]).strip()

        if not raw_name:
            raise ValueError("config.name is empty")

        model_params = {}
        if hasattr(config, "params"):
            model_params.update(dict(config.params))
        elif isinstance(config, dict) and "params" in config:
            model_params.update(dict(config["params"]))
        else:
            try:
                model_params.update({k: v for k, v in config.items() if k != "name"})
            except Exception:
                model_params.update(dict(config))

        if "kwargs" in model_params and isinstance(model_params["kwargs"], dict):
            extra = dict(model_params["kwargs"])
            model_params.pop("kwargs", None)
            extra.update(model_params)
            model_params = extra

        if kwargs:
            model_params.update(kwargs)

    lower = raw_name.lower()

    # 1. 尝试惰性加载空间和时序模块以触发注册
    from importlib import import_module
    candidate_attrs = {raw_name, raw_name.replace("_", ""), "".join(w.capitalize() for w in raw_name.split("_"))}
    try:
        spatial_mod = import_module("models.spatial")
        for attr in candidate_attrs:
            if hasattr(spatial_mod, attr):
                getattr(spatial_mod, attr)
                break
    except Exception:
        pass
    try:
        temporal_mod = import_module("models.temporal")
        for attr in candidate_attrs:
            if hasattr(temporal_mod, attr):
                getattr(temporal_mod, attr)
                break
    except Exception:
        pass


    # 2. 特殊时序包装处理：ARWrapper
    if lower in {"arwrapper", "ar_wrapper"}:
        from models.ar.wrapper import ARWrapper
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

        if "single_frame_model" not in wrapper_kwargs and "model" not in wrapper_kwargs:
            sub_model_name = wrapper_kwargs.pop("model_name", "SwinUNet")
            merged_base_kwargs = {}
            if isinstance(wrapper_kwargs.get("base_kwargs"), dict):
                merged_base_kwargs.update(wrapper_kwargs["base_kwargs"])
            merged_base_kwargs.update(base_kwargs)
            wrapper_kwargs.pop("base_kwargs", None)
            single_frame_model = create_model(sub_model_name, strict=strict, verbose=verbose, **merged_base_kwargs)
            wrapper_kwargs["single_frame_model"] = single_frame_model

        return ARWrapper(**wrapper_kwargs)

    # 3. 后缀 _ar / 前缀 ar_ 自动包装
    is_ar = lower.endswith("_ar") or lower.startswith("ar_")
    if is_ar and lower not in MODEL_REGISTRY and lower not in MODEL_ALIASES:
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
        drop_keys = {"use_ar", "use_nar", "nar_cfg", "ar_cfg"}
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

        base_model = create_model(base_name, strict=strict, verbose=verbose, **model_params)
        from models.ar.wrapper import ARWrapper
        if "t_out" in ar_kwargs and "T_out" not in ar_kwargs:
            ar_kwargs["T_out"] = ar_kwargs.pop("t_out")
        if "t_in" in ar_kwargs and "T_in" not in ar_kwargs:
            ar_kwargs["T_in"] = ar_kwargs.pop("t_in")
        if "t_out_steps" in ar_kwargs and "T_out" not in ar_kwargs:
            ar_kwargs["T_out"] = ar_kwargs.pop("t_out_steps")
        return ARWrapper(single_frame_model=base_model, **ar_kwargs)

    # 4. 解析注册表规范名称
    canonical_name = resolve_model_name(raw_name)
    model_cls = MODEL_REGISTRY[canonical_name]

    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # 常用参数别名映射
    arg_mapping = {
        "in_ch": "in_channels",
        "out_ch": "out_channels",
        "in_chans": "in_channels",
        "num_classes": "out_channels",
        "input_channels": "in_channels",
        "input_ch": "in_channels",
        "output_channels": "out_channels",
        "output_ch": "out_channels",
        "image_size": "img_size",
        "input_size": "img_size",
    }

    processed = dict(model_params)
    for alias, target in arg_mapping.items():
        if alias in processed and target not in processed:
            if (target in params) or has_var_keyword:
                processed[target] = processed[alias]
        if alias in processed and (alias not in params):
            processed.pop(alias, None)

    valid_kwargs = {}
    filtered = []

    for k, v in processed.items():
        if k in params:
            valid_kwargs[k] = v
        elif has_var_keyword:
            valid_kwargs[k] = v
        else:
            filtered.append(k)

    if filtered:
        msg = f"Filtered kwargs for model {canonical_name}: {filtered}"
        if strict:
            raise TypeError(msg)
        if verbose:
            logger.warning(msg)

    return model_cls(**valid_kwargs)


def list_models() -> List[str]:
    return sorted(list(MODEL_REGISTRY.keys()))

