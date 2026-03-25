"""
自定义数据批处理函数，用于处理None值
"""

from typing import Any

import torch


def filter_none_collate_fn(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """
    自定义collate函数，过滤掉None值

    Args:
        batch: 批次数据列表，可能包含None值

    Returns:
        过滤None后的批次数据，如果全部为None则返回None
    """
    # 过滤掉None值
    filtered_batch = [item for item in batch if item is not None]

    # 如果过滤后没有有效数据，返回None
    if not filtered_batch:
        return None

    # 使用PyTorch默认的collate函数处理有效数据
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def safe_collate_fn(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """
    安全的collate函数，处理None值和异常情况

    Args:
        batch: 批次数据列表

    Returns:
        处理后的批次数据
    """
    try:
        # 过滤None值
        valid_batch = []
        for item in batch:
            if item is not None:
                valid_batch.append(item)

        # 如果没有有效数据，返回None
        if not valid_batch:
            print("Warning: All items in batch are None")
            return None

        # 如果有效数据数量少于原始数量，发出警告
        if len(valid_batch) < len(batch):
            print(
                f"Warning: Filtered {len(batch) - len(valid_batch)} None items from batch"
            )

        # 使用默认collate函数
        return torch.utils.data.dataloader.default_collate(valid_batch)

    except Exception as e:
        print(f"Error in collate function: {e}")
        return None


def fast_collate_fn(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """
    快速专用的collate函数：
    - 过滤None项
    - 直接对关键张量键进行torch.stack，减少默认collate的递归与类型检查开销
    - 其余非关键键以轻量方式聚合，降低GIL压力

    预期输入项包含以下键：
    - 'input_sequence': Tensor [T_in, C, H, W]
    - 'target_sequence': Tensor [T_out, C, H, W]
    - 可选：'sample_idx' (int)、'start_time' (int)、'sample_key' (str)、'time_indices' (List[int])
    """
    # 过滤 None
    items = [it for it in batch if it is not None]
    if not items:
        return None
    try:
        # 关键张量直接堆叠，避免默认collate的深层递归
        input_seq = torch.stack([it["input_sequence"] for it in items], dim=0)
        target_seq = torch.stack([it["target_sequence"] for it in items], dim=0)

        # 轻量聚合元数据，避免默认collate的开销
        sample_idx = torch.tensor(
            [int(it.get("sample_idx", -1)) for it in items], dtype=torch.int64
        )
        start_time = torch.tensor(
            [int(it.get("start_time", 0)) for it in items], dtype=torch.int64
        )
        sample_key = [it.get("sample_key") for it in items]
        time_indices = [it.get("time_indices") for it in items]

        return {
            "input_sequence": input_seq,
            "target_sequence": target_seq,
            "sample_idx": sample_idx,
            "start_time": start_time,
            "sample_key": sample_key,
            "time_indices": time_indices,
        }
    except Exception:
        # 回退到默认collate，确保鲁棒性
        return torch.utils.data.dataloader.default_collate(items)
