import torch


def cpu_burn_collate(batch):
    # 在聚合批次前执行一次纯 CPU 矩阵乘法以提升 CPU 占用
    _ = torch.randn(4096, 4096, dtype=torch.float32).mm(torch.randn(4096, 4096))
    # 使用 PyTorch 默认的 collate 行为
    return torch.utils.data.dataloader.default_collate(batch)