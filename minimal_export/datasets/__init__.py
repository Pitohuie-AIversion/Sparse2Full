"""数据集包初始化

尽量采用惰性/安全导入，避免无关模块的语法错误阻断整个包的初始化。
仅在需要时导入具体数据集实现。
"""

try:
    from .pdebench_dataset import PDEBenchDataset, create_dataloader  # type: ignore
except Exception:
    PDEBenchDataset = None  # type: ignore
    create_dataloader = None  # type: ignore

try:
    from .pdebench import PDEBenchDataModule  # type: ignore
except Exception:
    PDEBenchDataModule = None  # type: ignore

try:
    from .temporal_pdebench import TemporalPDEBenchDataModule  # type: ignore
except Exception:
    TemporalPDEBenchDataModule = None  # type: ignore

# 安全导入：Darcy数据集
try:
    from .darcy_flow_dataset import DarcyFlowDataset  # type: ignore
except Exception:
    DarcyFlowDataset = None  # type: ignore


def get_dataset(dataset_name, **kwargs):
    """获取数据集实例（惰性判断可用性）"""
    if dataset_name == "darcy_flow":
        if DarcyFlowDataset is None:
            raise ImportError("DarcyFlowDataset 未可用：模块导入失败或缺失")
        return DarcyFlowDataset(**kwargs)
    elif dataset_name == "pde_bench":
        if PDEBenchDataset is None:
            raise ImportError("PDEBenchDataset 未可用：模块导入失败或缺失")
        return PDEBenchDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# 仅导出可用符号
__all__ = [
    name
    for name in (
        "PDEBenchDataset",
        "create_dataloader",
        "PDEBenchDataModule",
        "TemporalPDEBenchDataModule",
        "DarcyFlowDataset",
        "get_dataset",
    )
    if globals().get(name) is not None
]
