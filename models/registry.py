from typing import Dict, Any, Type, Optional, List
import inspect
import logging
from .base import BaseModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}
MODEL_ALIASES: Dict[str, str] = {}
DEPRECATED_ALIASES: Dict[str, str] = {
    "restormerlite": "cnn_attn_lite",
}

def register_model(name: str = None, aliases: List[str] = None):
    def decorator(cls):
        if not issubclass(cls, BaseModel):
            raise TypeError(f"Registered model must inherit BaseModel, got {cls}")

        model_name = (name if name is not None else cls.__name__).strip()
        if not model_name:
            raise ValueError("Model name is empty")

        if model_name in MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} is already registered!")
        if model_name in MODEL_ALIASES:
            raise ValueError(f"Model name {model_name} conflicts with an existing alias -> {MODEL_ALIASES[model_name]}")

        MODEL_REGISTRY[model_name] = cls

        if aliases:
            for alias in aliases:
                alias = str(alias).strip()
                if not alias:
                    raise ValueError(f"Empty alias for model {model_name}")

                # 关键：alias 不能和 canonical 冲突
                if alias in MODEL_REGISTRY:
                    raise ValueError(f"Alias {alias} conflicts with an existing canonical model name.")
                if alias in MODEL_ALIASES:
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

    raise ValueError(f"Model {raw} not found in registry.")

def create_model(model_name: str, strict: bool = False, verbose: bool = True, **kwargs) -> BaseModel:
    from importlib import import_module

    import_module("models.spatial")

    canonical_name = resolve_model_name(model_name)
    model_cls = MODEL_REGISTRY[canonical_name]

    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # 建议只保留“非常确定”的通用映射
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

    processed = dict(kwargs)

    # 映射：写入 canonical，并删除 alias，避免重复传参
    for alias, target in arg_mapping.items():
        if alias in processed and target not in processed:
            if (target in params) or has_var_keyword:
                processed[target] = processed[alias]
        if alias in processed and (alias not in params):  # alias 不是显式参数时，直接移除
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
