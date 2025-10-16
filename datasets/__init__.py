"""数据集模块

提供PDEBench数据集的加载和处理功能。
"""

from .pde_bench import PDEBenchDataset, create_dataloader
from .pdebench import PDEBenchDataModule
from .darcy_flow_dataset import DarcyFlowDataset

def get_dataset(dataset_name, **kwargs):
    """获取数据集实例"""
    if dataset_name == 'darcy_flow':
        return DarcyFlowDataset(**kwargs)
    elif dataset_name == 'pde_bench':
        return PDEBenchDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

__all__ = ['PDEBenchDataset', 'create_dataloader', 'PDEBenchDataModule', 'DarcyFlowDataset', 'get_dataset']