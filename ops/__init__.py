"""核心操作模块

包含损失函数、指标计算、H算子等核心操作。
"""


def __getattr__(name: str):
    if name == "apply_degradation_operator":
        from .degradation import apply_degradation_operator
        return apply_degradation_operator
    elif name in ("compute_total_loss", "compute_temporal_loss"):
        from .losses import compute_total_loss
        return compute_total_loss
    elif name == "compute_all_metrics":
        from .metrics import compute_all_metrics
        return compute_all_metrics
    elif name in ("compute_fluid_physics_loss", "compute_vorticity", "compute_divergence"):
        from .loss import compute_fluid_physics_loss, compute_vorticity, compute_divergence
        if name == "compute_fluid_physics_loss":
            return compute_fluid_physics_loss
        elif name == "compute_vorticity":
            return compute_vorticity
        else:
            return compute_divergence
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "apply_degradation_operator",
    "compute_total_loss",
    "compute_temporal_loss",
    "compute_all_metrics",
    "compute_fluid_physics_loss",
    "compute_vorticity",
    "compute_divergence",
]