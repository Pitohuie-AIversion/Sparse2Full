"""PDEBench数据集

提供PDEBench数据集的加载和处理功能。
支持SR和Crop任务的数据生成，确保与观测算子H的一致性。
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
import h5py
from omegaconf import DictConfig

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.degradation import apply_degradation_operator


class PDEBenchDataset(Dataset):
    """PDEBench数据集类
    
    支持超分辨率(SR)和裁剪(Crop)任务的数据加载。
    严格按照开发手册要求，确保观测生成H与训练DC复用同一实现。
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        task: str = "SR",
        task_params: Optional[Dict] = None,
        img_size: int = 256,
        normalize: bool = True,
        cache_data: bool = False
    ):
        """初始化数据集
        
        Args:
            data_root: 数据根目录
            split: 数据切分 ('train', 'val', 'test')
            task: 任务类型 ('SR', 'Crop')
            task_params: 任务参数
            img_size: 图像尺寸
            normalize: 是否标准化
            cache_data: 是否缓存数据
        """
        self.data_root = data_root
        self.split = split
        self.task = task
        self.task_params = task_params or {}
        self.img_size = img_size
        self.normalize = normalize
        self.cache_data = cache_data
        
        # 数据缓存
        self._data_cache = {} if cache_data else None
        self._norm_stats = None
        
        # 加载数据文件列表
        self.data_files = self._load_data_files()
        
        # 加载或计算标准化统计
        if normalize:
            self._load_normalization_stats()
    
    def _load_data_files(self) -> List[str]:
        """加载数据文件列表
        
        Returns:
            数据文件路径列表
        """
        split_file = os.path.join(self.data_root, "splits", f"{self.split}.txt")
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                files = [line.strip() for line in f.readlines()]
        else:
            # 如果没有切分文件，使用所有数据文件
            data_dir = os.path.join(self.data_root, "data")
            files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
            
            # 简单切分
            if self.split == "train":
                files = files[:int(0.8 * len(files))]
            elif self.split == "val":
                files = files[int(0.8 * len(files)):int(0.9 * len(files))]
            else:  # test
                files = files[int(0.9 * len(files)):]
        
        return [os.path.join(self.data_root, "data", f) for f in files]
    
    def _load_normalization_stats(self):
        """加载或计算标准化统计"""
        stats_file = os.path.join(self.data_root, "norm_stats.npz")
        
        if os.path.exists(stats_file):
            stats = np.load(stats_file)
            self._norm_stats = {
                'mean': torch.from_numpy(stats['mean']).float(),
                'std': torch.from_numpy(stats['std']).float()
            }
        else:
            # 计算统计量（这里使用简单的默认值）
            self._norm_stats = {
                'mean': torch.zeros(3),
                'std': torch.ones(3)
            }
    
    def __len__(self) -> int:
        """数据集长度"""
        # 每个文件包含多个样本，这里简化为文件数 * 每文件样本数
        # 实际应该根据文件内容动态计算
        return len(self.data_files) * 20  # 假设每个文件20个样本
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            数据字典，包含target, baseline, observation等
        """
        # 检查缓存
        if self.cache_data and idx in self._data_cache:
            return self._data_cache[idx]
        
        # 加载原始数据
        file_idx = idx // 20  # 文件索引
        sample_idx = idx % 20  # 文件内样本索引
        data_file = self.data_files[file_idx]
        target = self._load_data_from_file(data_file, sample_idx)
        
        # 确保数据尺寸正确
        target = self._resize_data(target, self.img_size)
        
        # 标准化
        if self.normalize:
            target = self._normalize_data(target)
        
        # 生成观测数据（应用退化算子H）
        # 创建包含task信息的参数字典
        task_params_with_task = dict(self.task_params)
        task_params_with_task['task'] = self.task
        
        # 为SR任务添加必要的参数映射
        if self.task == "SR":
            task_params_with_task['scale'] = task_params_with_task.get('scale_factor', 4)
            task_params_with_task['sigma'] = task_params_with_task.get('blur_sigma', 1.0)
            task_params_with_task['kernel_size'] = task_params_with_task.get('blur_kernel_size', 5)
            task_params_with_task['boundary'] = task_params_with_task.get('boundary_mode', 'mirror')
        
        observation = apply_degradation_operator(target.unsqueeze(0), task_params_with_task)
        observation = observation.squeeze(0)
        
        # 保存原始观测数据用于损失计算
        original_observation = observation.clone()
        
        # 对于SR任务，需要将观测数据上采样到目标尺寸以匹配模型输入
        if self.task == "SR":
            import torch.nn.functional as F
            observation = observation.unsqueeze(0)  # [1, C, H, W]
            observation = F.interpolate(
                observation, 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # [C, H, W]
        
        # 构建数据项
        data_item = {
            'target': target,
            'baseline': target,  # 对于重建任务，baseline就是target
            'observation': observation,
            'original_observation': original_observation,  # 原始观测数据用于损失计算
            'task_params': task_params_with_task,  # 使用包含完整参数的字典
            'file_path': data_file,
            'index': idx
        }
        
        # 缓存数据
        if self.cache_data:
            self._data_cache[idx] = data_item
        
        return data_item
    
    def _load_data_from_file(self, file_path: str, sample_idx: int = 0) -> torch.Tensor:
        """从文件加载数据
        
        Args:
            file_path: 文件路径
            sample_idx: 样本索引
            
        Returns:
            数据张量 [C, H, W]
        """
        if file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                # 尝试不同的键名
                if 'data' in f:
                    data = f['data'][:]
                elif 'u' in f:
                    data = f['u'][:]
                else:
                    # 使用第一个可用的数据集
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][:]
                    else:
                        raise ValueError(f"No data found in {file_path}")
                
                # 转换为torch张量
                data = torch.from_numpy(data).float()
                
                # 处理多样本数据，选择指定样本
                if len(data.shape) == 4:  # [N, C, H, W]
                    sample_idx = min(sample_idx, data.shape[0] - 1)
                    data = data[sample_idx]  # [C, H, W]
                elif len(data.shape) == 3:
                    # 判断是 [N, H, W] 还是 [C, H, W]
                    if data.shape[0] > 10:  # 假设样本数大于10，通道数小于10
                        sample_idx = min(sample_idx, data.shape[0] - 1)
                        data = data[sample_idx].unsqueeze(0)  # [1, H, W] -> [C, H, W]
                    elif data.shape[0] > data.shape[2]:  # HWC -> CHW
                        data = data.permute(2, 0, 1)
                elif len(data.shape) == 2:
                    data = data.unsqueeze(0)  # 添加通道维度 [H, W] -> [C, H, W]
                
                return data
        else:
            # 生成合成数据用于测试
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> torch.Tensor:
        """生成合成测试数据
        
        Returns:
            合成数据张量 [C, H, W]
        """
        # 创建具有不同频率成分的合成数据
        H, W = self.img_size, self.img_size
        C = 3
        
        x = torch.linspace(0, 2*np.pi, W).unsqueeze(0).repeat(H, 1)
        y = torch.linspace(0, 2*np.pi, H).unsqueeze(1).repeat(1, W)
        
        data_list = []
        for c in range(C):
            # 不同通道使用不同的频率组合
            freq_x = 1.0 + c * 0.5
            freq_y = 1.0 + c * 0.3
            phase = c * np.pi / 3
            
            # 组合正弦波
            signal = torch.sin(freq_x * x + phase) * torch.cos(freq_y * y)
            noise = torch.randn_like(signal) * 0.1
            channel_data = signal + noise
            
            data_list.append(channel_data)
        
        data = torch.stack(data_list, dim=0)  # [C, H, W]
        return data
    
    def _resize_data(self, data: torch.Tensor, target_size: int) -> torch.Tensor:
        """调整数据尺寸
        
        Args:
            data: 输入数据 [C, H, W]
            target_size: 目标尺寸
            
        Returns:
            调整后的数据
        """
        if data.shape[-1] != target_size or data.shape[-2] != target_size:
            import torch.nn.functional as F
            # 确保数据有正确的维度 [N, C, H, W]
            if len(data.shape) == 3:  # [C, H, W]
                data = data.unsqueeze(0)  # [1, C, H, W]
                data = F.interpolate(
                    data, 
                    size=(target_size, target_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)  # [C, H, W]
            elif len(data.shape) == 4:  # [T, C, H, W] 或 [N, C, H, W]
                # 取第一个时间步或批次
                data = data[0] if data.shape[0] > 1 else data.squeeze(0)
                if len(data.shape) == 3:  # 现在是 [C, H, W]
                    data = data.unsqueeze(0)  # [1, C, H, W]
                    data = F.interpolate(
                        data, 
                        size=(target_size, target_size), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)  # [C, H, W]
        
        return data
    
    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """标准化数据
        
        Args:
            data: 输入数据 [C, H, W]
            
        Returns:
            标准化后的数据
        """
        if self._norm_stats is None:
            return data
        
        mean = self._norm_stats['mean'].view(-1, 1, 1)
        std = self._norm_stats['std'].view(-1, 1, 1)
        
        return (data - mean) / std
    
    def denormalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """反标准化数据
        
        Args:
            data: 标准化的数据 [C, H, W] 或 [B, C, H, W]
            
        Returns:
            原始尺度的数据
        """
        if self._norm_stats is None:
            return data
        
        mean = self._norm_stats['mean']
        std = self._norm_stats['std']
        
        if len(data.shape) == 4:  # [B, C, H, W]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:  # [C, H, W]
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        
        return data * std + mean
    
    @property
    def normalization_stats(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取标准化统计量"""
        return self._norm_stats


def create_dataloader(
    config: DictConfig,
    split: str = "train",
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """创建数据加载器
    
    Args:
        config: 配置对象
        split: 数据切分
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        
    Returns:
        数据加载器
    """
    # 默认参数
    if batch_size is None:
        batch_size = config.training.batch_size
    if shuffle is None:
        shuffle = (split == "train")
    
    # 创建数据集
    dataset = PDEBenchDataset(
        data_root=config.data.data_root,
        split=split,
        task=config.data.task,
        task_params=config.data.task_params,
        img_size=config.data.img_size,
        normalize=config.data.get('normalize', True),
        cache_data=config.data.get('cache_data', False)
    )
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )
    
    return dataloader