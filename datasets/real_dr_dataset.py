"""
真实扩散-反应数据集加载器
专门处理真实扩散-反应数据集的时序结构
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RealDiffusionReactionDataset(Dataset):
    """真实扩散-反应数据集
    
    数据结构: 每个时间步为一个组，包含'data'和'grid'键
    - 时间步: '0000', '0001', ..., '0999'
    - 数据形状: (101, 128, 128, 2) - 101个样本，128x128分辨率，2个通道(u,v)
    """
    
    def __init__(
        self,
        data_path: str,
        T_in: int = 1,
        T_out: int = 20,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        time_step_start: int = 0,
        time_step_end: int = 980,  # 确保有足够的时间步用于T_out=20
        augmentation: bool = False,
        sample_limit: Optional[int] = None  # 新增：限制样本数量
    ):
        """初始化数据集
        
        Args:
            data_path: HDF5文件路径
            T_in: 输入时间步数
            T_out: 输出时间步数
            split: 数据分割 ('train', 'val', 'test')
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            normalize: 是否归一化
            time_step_start: 起始时间步
            time_step_end: 结束时间步
            augmentation: 是否数据增强
            sample_limit: 限制使用的样本数量，None表示使用所有样本
        """
        self.data_path = data_path
        self.T_in = T_in
        self.T_out = T_out
        self.split = split
        self.normalize = normalize
        self.time_step_start = time_step_start
        self.time_step_end = time_step_end
        self.augmentation = augmentation
        self.sample_limit = sample_limit
        
        # 验证文件存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 加载数据信息
        self._load_data_info()
        
        # 创建样本索引
        self._create_sample_indices(train_ratio, val_ratio, test_ratio)
        
        # 计算归一化统计
        if self.normalize:
            self._compute_normalization_stats()
        
        logger.info(f"创建{split}数据集: {len(self.sample_indices)}个样本")
    
    def _load_data_info(self):
        """加载数据信息"""
        with h5py.File(self.data_path, 'r') as f:
            # 获取时间步列表
            self.time_steps = sorted([k for k in f.keys() if k.isdigit()])
            
            # 获取第一个时间步的数据形状
            first_timestep = self.time_steps[0]
            first_data = f[first_timestep]['data']
            self.data_shape = first_data.shape  # (101, 128, 128, 2)
            
            self.num_samples = self.data_shape[0]  # 101
            self.height = self.data_shape[1]       # 128
            self.width = self.data_shape[2]        # 128
            self.channels = self.data_shape[3]     # 2
            
            # 应用样本限制
            if self.sample_limit is not None:
                self.num_samples = min(self.num_samples, self.sample_limit)
                logger.info(f"应用样本限制: {self.sample_limit}，实际使用: {self.num_samples}")
            
            logger.info(f"数据集信息:")
            logger.info(f"  时间步数: {len(self.time_steps)}")
            logger.info(f"  数据形状: {self.data_shape}")
            logger.info(f"  样本数: {self.num_samples}")
            logger.info(f"  分辨率: {self.height}x{self.width}")
            logger.info(f"  通道数: {self.channels}")
    
    def _create_sample_indices(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """创建样本索引"""
        # 计算可用的时间窗口
        max_time_idx = min(len(self.time_steps) - self.T_in - self.T_out + 1, 
                          self.time_step_end - self.time_step_start + 1 - self.T_in - self.T_out + 1)
        
        # 为每个样本和每个时间窗口创建索引
        all_indices = []
        for sample_idx in range(self.num_samples):  # 使用限制后的样本数
            for time_start in range(self.time_step_start, 
                                  self.time_step_start + max_time_idx):
                all_indices.append((sample_idx, time_start))
        
        # 分割数据集
        np.random.seed(42)  # 固定随机种子确保可复现
        np.random.shuffle(all_indices)
        
        total_samples = len(all_indices)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        if self.split == 'train':
            self.sample_indices = all_indices[:train_end]
        elif self.split == 'val':
            self.sample_indices = all_indices[train_end:val_end]
        elif self.split == 'test':
            self.sample_indices = all_indices[val_end:]
        else:
            raise ValueError(f"未知的数据分割: {self.split}")
    
    def _compute_normalization_stats(self):
        """计算归一化统计"""
        logger.info("计算归一化统计...")
        
        # 采样部分数据计算统计
        sample_size = min(1000, len(self.sample_indices))
        sample_indices = np.random.choice(len(self.sample_indices), sample_size, replace=False)
        
        all_data = []
        with h5py.File(self.data_path, 'r') as f:
            for idx in sample_indices:
                sample_idx, time_start = self.sample_indices[idx]
                
                # 加载输入和目标序列
                for t in range(self.T_in + self.T_out):
                    time_step = f"{time_start + t:04d}"
                    data = f[time_step]['data'][sample_idx]  # (128, 128, 2)
                    all_data.append(data)
        
        all_data = np.stack(all_data, axis=0)  # (N, 128, 128, 2)
        
        # 计算每个通道的均值和标准差
        self.mean = np.mean(all_data, axis=(0, 1, 2))  # (2,)
        self.std = np.std(all_data, axis=(0, 1, 2))    # (2,)
        
        # 避免除零
        self.std = np.maximum(self.std, 1e-8)
        
        logger.info(f"归一化统计:")
        logger.info(f"  均值: {self.mean}")
        logger.info(f"  标准差: {self.std}")
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化数据"""
        if self.normalize:
            return (data - self.mean) / self.std
        return data
    
    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """反归一化数据"""
        if self.normalize:
            return data * self.std + self.mean
        return data
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本"""
        sample_idx, time_start = self.sample_indices[idx]
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                # 加载输入序列
                input_sequence = []
                for t in range(self.T_in):
                    time_step = f"{time_start + t:04d}"
                    data = f[time_step]['data'][sample_idx]  # 可能是 (128, 128, 2) 或 (1, 128, 128, 2)
                    # 处理可能的额外维度
                    if data.ndim == 4 and data.shape[0] == 1:
                        data = data.squeeze(0)  # 移除第一个维度 (1, 128, 128, 2) -> (128, 128, 2)
                    data = self._normalize(data)
                    # 确保数据形状正确
                    if data.shape != (128, 128, 2):
                        logger.warning(f"数据形状异常: {data.shape}, 期望 (128, 128, 2)")
                        data = np.zeros((128, 128, 2), dtype=np.float32)
                    # 转换为 (C, H, W) 格式
                    data = data.transpose(2, 0, 1)  # (2, 128, 128)
                    input_sequence.append(data)
                
                # 加载目标序列
                target_sequence = []
                for t in range(self.T_out):
                    time_step = f"{time_start + self.T_in + t:04d}"
                    data = f[time_step]['data'][sample_idx]  # 可能是 (128, 128, 2) 或 (1, 128, 128, 2)
                    # 处理可能的额外维度
                    if data.ndim == 4 and data.shape[0] == 1:
                        data = data.squeeze(0)  # 移除第一个维度 (1, 128, 128, 2) -> (128, 128, 2)
                    data = self._normalize(data)
                    # 确保数据形状正确
                    if data.shape != (128, 128, 2):
                        logger.warning(f"数据形状异常: {data.shape}, 期望 (128, 128, 2)")
                        data = np.zeros((128, 128, 2), dtype=np.float32)
                    # 转换为 (C, H, W) 格式
                    data = data.transpose(2, 0, 1)  # (2, 128, 128)
                    target_sequence.append(data)
                
                # 转换为张量
                input_sequence = torch.from_numpy(np.stack(input_sequence, axis=0)).float()   # (T_in, C, H, W)
                target_sequence = torch.from_numpy(np.stack(target_sequence, axis=0)).float() # (T_out, C, H, W)
                
                # 数据增强
                if self.augmentation and self.split == 'train':
                    input_sequence, target_sequence = self._apply_augmentation(input_sequence, target_sequence)
                
                return {
                    'input_sequence': input_sequence,
                    'target_sequence': target_sequence,
                    'sample_idx': sample_idx,
                    'time_start': time_start,
                    'metadata': {
                        'T_in': self.T_in,
                        'T_out': self.T_out,
                        'channels': self.channels,
                        'height': self.height,
                        'width': self.width
                    }
                }
        
        except Exception as e:
            logger.error(f"加载样本失败 idx={idx}, sample_idx={sample_idx}, time_start={time_start}: {e}")
            # 返回零张量作为备用
            input_sequence = torch.zeros(self.T_in, self.channels, self.height, self.width)
            target_sequence = torch.zeros(self.T_out, self.channels, self.height, self.width)
            return {
                'input_sequence': input_sequence,
                'target_sequence': target_sequence,
                'sample_idx': sample_idx,
                'time_start': time_start,
                'metadata': {
                    'T_in': self.T_in,
                    'T_out': self.T_out,
                    'channels': self.channels,
                    'height': self.height,
                    'width': self.width
                }
            }
    
    def _apply_augmentation(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用数据增强"""
        # 随机水平翻转
        if torch.rand(1) < 0.5:
            input_seq = torch.flip(input_seq, dims=[-1])
            target_seq = torch.flip(target_seq, dims=[-1])
        
        # 随机垂直翻转
        if torch.rand(1) < 0.5:
            input_seq = torch.flip(input_seq, dims=[-2])
            target_seq = torch.flip(target_seq, dims=[-2])
        
        # 随机旋转90度
        if torch.rand(1) < 0.3:
            k = torch.randint(1, 4, (1,)).item()
            input_seq = torch.rot90(input_seq, k, dims=[-2, -1])
            target_seq = torch.rot90(target_seq, k, dims=[-2, -1])
        
        # 添加噪声
        if torch.rand(1) < 0.3:
            noise_std = 0.01
            input_seq += torch.randn_like(input_seq) * noise_std
            target_seq += torch.randn_like(target_seq) * noise_std
        
        return input_seq, target_seq


class RealDiffusionReactionDataModule:
    """真实扩散-反应数据模块"""
    
    def __init__(
        self,
        data_path: str,
        T_in: int = 1,
        T_out: int = 20,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        augmentation: bool = True,
        time_step_start: int = 0,
        time_step_end: int = 980,
        sample_limit: Optional[int] = None  # 新增：限制样本数量
    ):
        """初始化数据模块"""
        self.data_path = data_path
        self.T_in = T_in
        self.T_out = T_out
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalize = normalize
        self.augmentation = augmentation
        self.time_step_start = time_step_start
        self.time_step_end = time_step_end
        self.sample_limit = sample_limit
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """设置数据集"""
        logger.info("设置真实扩散-反应数据集...")
        
        # 创建数据集
        self.train_dataset = RealDiffusionReactionDataset(
            data_path=self.data_path,
            T_in=self.T_in,
            T_out=self.T_out,
            split='train',
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            normalize=self.normalize,
            time_step_start=self.time_step_start,
            time_step_end=self.time_step_end,
            augmentation=self.augmentation,
            sample_limit=self.sample_limit
        )
        
        self.val_dataset = RealDiffusionReactionDataset(
            data_path=self.data_path,
            T_in=self.T_in,
            T_out=self.T_out,
            split='val',
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            normalize=self.normalize,
            time_step_start=self.time_step_start,
            time_step_end=self.time_step_end,
            augmentation=False,  # 验证集不使用数据增强
            sample_limit=self.sample_limit
        )
        
        self.test_dataset = RealDiffusionReactionDataset(
            data_path=self.data_path,
            T_in=self.T_in,
            T_out=self.T_out,
            split='test',
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            normalize=self.normalize,
            time_step_start=self.time_step_start,
            time_step_end=self.time_step_end,
            augmentation=False,  # 测试集不使用数据增强
            sample_limit=self.sample_limit
        )
        
        logger.info("数据集设置完成")
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False
        )