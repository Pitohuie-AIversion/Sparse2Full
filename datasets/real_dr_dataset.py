"""
真实扩散-反应数据集加载器
专门处理真实扩散-反应数据集的时序结构
"""

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging

# 导入降采样操作
try:
    from ops.degradation import apply_degradation_operator
except ImportError:
    # 如果无法导入，定义一个简单的降采样函数
    def apply_degradation_operator(data, params):
        """简单的降采样实现"""
        if params.get('task') == 'SR':
            scale = params.get('scale', 2)
            # 简单的平均池化降采样
            return F.avg_pool2d(data, kernel_size=scale, stride=scale)
        return data

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
        sample_limit: Optional[int] = None,  # 新增：限制样本数量
        observation_params: Optional[Dict] = None  # 新增：观测参数
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
        
        # 处理观测参数
        self.observation_params = observation_params
        self.use_observation = bool(observation_params)
        
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
            # 获取样本ID列表 (0000-0999 是1000个独立的仿真样本)
            self.sample_ids = sorted([k for k in f.keys() if k.isdigit()])
            
            # 获取第一个样本的数据形状
            first_sample_id = self.sample_ids[0]
            first_data = f[first_sample_id]['data']
            self.data_shape = first_data.shape  # (101, 128, 128, 2)
            
            # 正确的数据结构理解：
            # - HDF5文件有1000个样本ID (0000-0999)
            # - 每个样本包含101个时间步的数据，形状为 (101, 128, 128, 2)
            self.num_samples = len(self.sample_ids)     # 1000个独立仿真样本
            self.num_timesteps = self.data_shape[0]     # 101个时间步
            self.height = self.data_shape[1]            # 128
            self.width = self.data_shape[2]             # 128
            self.channels = 1  # 只使用第一个通道（u通道）
            
            # 应用样本限制
            if self.sample_limit is not None:
                self.num_samples = min(self.num_samples, self.sample_limit)
                logger.info(f"应用样本限制: {self.sample_limit}，实际使用: {self.num_samples}")
            
            logger.info(f"数据集信息:")
            logger.info(f"  仿真样本数: {len(self.sample_ids)}")
            logger.info(f"  每个样本的时间步数: {self.num_timesteps}")
            logger.info(f"  数据形状 (每个样本): {self.data_shape}")
            logger.info(f"  实际使用样本数: {self.num_samples}")
            logger.info(f"  分辨率: {self.height}x{self.width}")
            logger.info(f"  通道数: {self.channels}")
    
    def _create_sample_indices(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """创建样本索引 - 按样本ID划分避免数据泄露"""
        # 计算可用的时间窗口 (基于每个样本的时间步数)
        max_time_idx = min(self.num_timesteps - self.T_in - self.T_out + 1, 
                          self.time_step_end - self.time_step_start + 1 - self.T_in - self.T_out + 1)
        
        # 🔥 关键修复：按样本ID进行划分，避免数据泄露
        np.random.seed(42)  # 固定随机种子确保可复现
        sample_ids = np.random.permutation(self.num_samples)
        
        # 计算每个集合的样本数
        n_train = int(self.num_samples * train_ratio)
        n_val = int(self.num_samples * val_ratio)
        
        # 分配样本ID到不同集合
        if self.split == 'train':
            assigned_samples = set(sample_ids[:n_train])
        elif self.split == 'val':
            assigned_samples = set(sample_ids[n_train:n_train + n_val])
        elif self.split == 'test':
            assigned_samples = set(sample_ids[n_train + n_val:])
        else:
            raise ValueError(f"未知的数据分割: {self.split}")
        
        # 为分配的样本生成所有时间窗口
        self.sample_indices = []
        for sample_idx in assigned_samples:
            for time_start in range(self.time_step_start, 
                                  self.time_step_start + max_time_idx):
                self.sample_indices.append((sample_idx, time_start))
        
        logger.info(f"{self.split} 集: {len(assigned_samples)} 个样本, {len(self.sample_indices)} 个序列")
    
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
                
                # 获取样本数据
                sample_id = f"{sample_idx:04d}"
                sample_data = f[sample_id]['data']  # (101, 128, 128, 2)
                
                # 加载输入和目标序列
                for t in range(self.T_in + self.T_out):
                    time_idx = time_start + t
                    data = sample_data[time_idx]  # (128, 128, 2)
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
            # 使用惰性打开的只读 HDF5 句柄（每个 worker 一个）
            f = self._ensure_h5_open()
            # 获取样本数据 (101, 128, 128, 2)
            sample_id = f"{sample_idx:04d}"
            sample_data = f[sample_id]['data']  # (101, 128, 128, 2)
                
            # 加载输入序列
            input_sequence = []
            for t in range(self.T_in):
                time_idx = time_start + t
                data = sample_data[time_idx]  # (128, 128, 2)
                data = self._normalize(data)
                # 确保数据形状正确
                if data.shape != (128, 128, 2):
                    logger.warning(f"数据形状异常: {data.shape}, 期望 (128, 128, 2)")
                    data = np.zeros((128, 128, 2), dtype=np.float32)
                # 只使用第一个通道（u通道）
                data = data[:, :, 0:1]  # (128, 128, 1)
                # 转换为 (C, H, W) 格式
                data = data.transpose(2, 0, 1)  # (1, 128, 128)
                input_sequence.append(data)
                
            # 加载目标序列
            target_sequence = []
            for t in range(self.T_out):
                time_idx = time_start + self.T_in + t
                data = sample_data[time_idx]  # (128, 128, 2)
                data = self._normalize(data)
                # 确保数据形状正确
                if data.shape != (128, 128, 2):
                    logger.warning(f"数据形状异常: {data.shape}, 期望 (128, 128, 2)")
                    data = np.zeros((128, 128, 2), dtype=np.float32)
                # 只使用第一个通道（u通道）
                data = data[:, :, 0:1]  # (128, 128, 1)
                # 转换为 (C, H, W) 格式
                data = data.transpose(2, 0, 1)  # (1, 128, 128)
                target_sequence.append(data)
                
            # 转换为张量
            input_sequence = torch.from_numpy(np.stack(input_sequence, axis=0)).float()   # (T_in, C, H, W)
            target_sequence = torch.from_numpy(np.stack(target_sequence, axis=0)).float() # (T_out, C, H, W)
            
            # 数据增强
            if self.augmentation and self.split == 'train':
                input_sequence, target_sequence = self._apply_augmentation(input_sequence, target_sequence)
            
            # 应用观测配置（降采样）
            if self.use_observation and self.observation_params:
                # 对输入序列应用降采样
                degraded_input = []
                for t in range(input_sequence.shape[0]):
                    frame = input_sequence[t:t+1]  # [1, C, H, W]
                    degraded_frame = apply_degradation_operator(frame, self.observation_params)
                    # 上采样回原尺寸
                    if degraded_frame.shape[-2:] != frame.shape[-2:]:
                        degraded_frame = F.interpolate(
                            degraded_frame,
                            size=frame.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    degraded_input.append(degraded_frame.squeeze(0))
                input_sequence = torch.stack(degraded_input, dim=0)
            
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
                    'width': self.width,
                    'observation_params': self.observation_params or {}  # 避免None值
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
                    'width': self.width,
                    'observation_params': {}  # 避免None值
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

    # ---------------- HDF5 句柄惰性管理（每个 worker 独立只读句柄） ----------------
    def _ensure_h5_open(self):
        if getattr(self, '_h5_file', None) is None:
            self._h5_file = h5py.File(self.data_path, 'r')
        return self._h5_file

    def __getstate__(self):
        state = self.__dict__.copy()
        # h5py.File 不可序列化，DataLoader 需要移除句柄
        state['_h5_file'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5_file = None


# DataLoader worker 初始化：在子进程惰性（或提前）打开只读 HDF5 句柄
def realdr_worker_init_fn(worker_id: int):
    try:
        info = torch.utils.data.get_worker_info()
        ds = getattr(info, 'dataset', None)
        if ds is not None and hasattr(ds, '_ensure_h5_open'):
            ds._ensure_h5_open()
            logger.info(f"RealDR worker {worker_id}: HDF5 handle ready")
    except Exception as e:
        logger.warning(f"RealDR worker {worker_id}: init failed: {e}")


class RealDiffusionReactionDataModule:
    """真实扩散-反应数据模块"""
    
    def __init__(
        self,
        data_path: str,
        T_in: int = 1,
        T_out: int = 20,
        batch_size: int = 8,
        val_batch_size: Optional[int] = None,  # 新增：验证批次大小
        test_batch_size: int = 1,  # 新增：测试批次大小
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        multiprocessing_context: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        augmentation: bool = True,
        time_step_start: int = 0,
        time_step_end: int = 980,
        sample_limit: Optional[int] = None,  # 新增：限制样本数量
        observation: Optional[Dict] = None  # 新增：观测配置
    ):
        """初始化数据模块"""
        self.data_path = data_path
        self.T_in = T_in
        self.T_out = T_out
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size  # 验证批次大小，默认与训练批次相同
        self.test_batch_size = test_batch_size  # 保存测试批次大小
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.multiprocessing_context = multiprocessing_context
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalize = normalize
        self.augmentation = augmentation
        self.time_step_start = time_step_start
        self.time_step_end = time_step_end
        self.sample_limit = sample_limit
        
        # 处理观测配置
        self.observation = observation or {}
        self.use_observation = bool(observation)
        if self.use_observation:
            # 解析观测配置
            self.observation_mode = self.observation.get('mode', 'SR')
            if self.observation_mode.lower() == 'sr':
                self.observation_params = {
                    'task': 'SR',
                    'scale': self.observation.get('scale_factor', 2),
                    'sigma': self.observation.get('blur_sigma', 1.0),
                    'kernel_size': self.observation.get('blur_kernel_size', 5),
                    'boundary': self.observation.get('boundary_mode', 'mirror'),
                    'downsample_mode': self.observation.get('downsample_mode', 'area'),
                    'align_corners': self.observation.get('align_corners', False),
                    'antialias': self.observation.get('antialias', True)
                }
            else:
                raise ValueError(f"Unsupported observation mode: {self.observation_mode}")
        else:
            self.observation_params = None
        
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
            sample_limit=self.sample_limit,
            observation_params=self.observation_params
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
            sample_limit=self.sample_limit,
            observation_params=self.observation_params
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
            sample_limit=self.sample_limit,
            observation_params=self.observation_params
        )
        
        logger.info("数据集设置完成")
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'drop_last': True,
            'worker_init_fn': realdr_worker_init_fn,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            kwargs['prefetch_factor'] = self.prefetch_factor
        if self.multiprocessing_context is not None:
            kwargs['multiprocessing_context'] = self.multiprocessing_context
        return DataLoader(self.train_dataset, **kwargs)
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        kwargs = {
            'batch_size': self.val_batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'drop_last': False,
            'worker_init_fn': realdr_worker_init_fn,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            kwargs['prefetch_factor'] = self.prefetch_factor
        if self.multiprocessing_context is not None:
            kwargs['multiprocessing_context'] = self.multiprocessing_context
        return DataLoader(self.val_dataset, **kwargs)
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        kwargs = {
            'batch_size': self.test_batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'drop_last': False,
            'worker_init_fn': realdr_worker_init_fn,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            kwargs['prefetch_factor'] = self.prefetch_factor
        if self.multiprocessing_context is not None:
            kwargs['multiprocessing_context'] = self.multiprocessing_context
        return DataLoader(self.test_dataset, **kwargs)