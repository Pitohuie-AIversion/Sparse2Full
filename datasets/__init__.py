"""数据集包初始化

采用惰性导入，避免包初始化时的重载与框架冲突。
"""


def __getattr__(name: str):
    if name in ("PDEBenchDataset", "create_dataloader"):
        from . import pdebench_dataset
        return getattr(pdebench_dataset, name)
    elif name in ("PDEBenchDataModule", "TemporalPDEBenchDataModule"):
        from . import pdebench
        return getattr(pdebench, "PDEBenchDataModule")
    elif name == "DarcyFlowDataset":
        from . import darcy_flow_dataset
        return getattr(darcy_flow_dataset, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_dataset(dataset_name: str, **kwargs):
    """获取数据集实例（惰性判断可用性）"""
    if dataset_name == 'darcy_flow':
        from .darcy_flow_dataset import DarcyFlowDataset
        return DarcyFlowDataset(**kwargs)
    elif dataset_name == 'pde_bench':
        from .pdebench_dataset import PDEBenchDataset
        return PDEBenchDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = [
    'PDEBenchDataset',
    'create_dataloader',
    'PDEBenchDataModule',
    'TemporalPDEBenchDataModule',
    'DarcyFlowDataset',
    'get_dataset',
]
