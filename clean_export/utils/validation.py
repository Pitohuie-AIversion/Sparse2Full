"""配置验证模块（测试兼容版）

提供 `validate_config(config) -> bool` 接口，满足集成测试对布尔返回值的期望。
优先复用 `training_system.utils.validation.validate_config`，并将异常转为 False。
"""

from typing import Any, Dict, List, Optional


def validate_config(config: Dict[str, Any], required_keys: Optional[List[str]] = None) -> bool:
    """验证配置有效性并返回布尔值。

    - 若可用，复用 `training_system.utils.validation.validate_config` 并捕获异常；
    - 否则使用简化的兜底校验，只检查必要键存在。
    """
    # 尝试复用训练系统的严格校验器
    try:
        from training_system.utils.validation import validate_config as strict_validate  # type: ignore
    except Exception:
        strict_validate = None

    if strict_validate is not None:
        try:
            strict_validate(config, required_keys)
            return True
        except Exception:
            return False

    # 兜底：最简校验逻辑（仅键存在性）
    try:
        if not isinstance(config, dict):
            return False

        if required_keys is None:
            required_keys = ['data', 'model', 'training']

        for key in required_keys:
            if key not in config:
                return False

        return True
    except Exception:
        return False


__all__ = ["validate_config"]