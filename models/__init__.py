"""模型模块 - 分类组织的空间和时间预测模型"""

import importlib


def __getattr__(name: str):
    if name in ("base", "spatial", "temporal"):
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    elif name in ("BaseModel", "create_model", "get_model"):
        base = importlib.import_module(".base", __name__)
        return getattr(base, name)
    
    # Try importing from spatial
    try:
        spatial = importlib.import_module(".spatial", __name__)
        if hasattr(spatial, name):
            return getattr(spatial, name)
    except Exception:
        pass

    # Try importing from temporal
    try:
        temporal = importlib.import_module(".temporal", __name__)
        if hasattr(temporal, name):
            return getattr(temporal, name)
    except Exception:
        pass

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def create_model(model_name_or_config=None, **kwargs):
    from .base import create_model as base_create_model
    return base_create_model(model_name_or_config, **kwargs)


def get_model(model_name, **kwargs):
    return create_model(model_name, **kwargs)


__all__ = [
    "spatial",
    "temporal",
    "BaseModel",
    "create_model",
    "get_model",
]
