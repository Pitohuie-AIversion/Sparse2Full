"""
自定义数据批处理函数，用于处理None值
"""
import torch
from typing import List, Dict, Any, Optional


def filter_none_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
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


def safe_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
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
            print(f"Warning: Filtered {len(batch) - len(valid_batch)} None items from batch")
        
        # 使用默认collate函数
        return torch.utils.data.dataloader.default_collate(valid_batch)
        
    except Exception as e:
        print(f"Error in collate function: {e}")
        return None