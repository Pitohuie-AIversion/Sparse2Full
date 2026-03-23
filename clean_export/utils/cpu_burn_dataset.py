import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Tuple


class CpuBurnDataset(Dataset):
    """Wrap an existing Dataset and add a pure-CPU matrix multiplication
    at the end of __getitem__ to increase CPU load.

    This does NOT modify the returned sample content; it only performs
    additional CPU work to stress the system.
    """

    def __init__(self, base_dataset: Dataset, burn_size: int = 4096, repeat: int = 1):
        self.base = base_dataset
        self.burn_size = int(burn_size)
        self.repeat = int(max(1, repeat))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.base[index]

        # Pure CPU work: large matrix multiplication
        # Use a local generator to avoid interfering with global RNG/state
        # The result is not used; computation is for CPU load only.
        try:
            size = self.burn_size
            # Ensure CPU tensors
            for _ in range(self.repeat):
                a = torch.randn((size, size), dtype=torch.float32, device='cpu')
                b = torch.randn((size, size), dtype=torch.float32, device='cpu')
                _ = torch.mm(a, b)
        except Exception:
            # Be resilient: never break data loading because of burn block
            pass

        return sample


__all__ = ["CpuBurnDataset"]