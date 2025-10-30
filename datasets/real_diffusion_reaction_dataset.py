"""真实扩散-反应数据集加载器

专门处理真实扩散-反应数据集的时序数据加载，支持AR训练。
数据格式：E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5
结构：每个时间步为一个组，包含data数据集 (101, 128, 128, 2)
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import DictConfig


class RealDiffusionReactionDataset(Dataset):
    """真实扩散-反应数据集"""
    
    def __init__(
        self,
        data_path: str,
        T_in: int = 1,
        T_out: int = 20,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        time_step_start: int = 0,
        time_step_end: int = 980,
        time_step_stride: int = 1,
        normalize: bool = True,
        augmentation: Optional[Dict] = None,
        seed: int = 2025
    ):
        """
        Args:
            data_path: HDF5数据文件路径
            T_in: 输入时间步数
            T_out: 输出时间步数
            split: 数据集分割 ("train", "val", "test")
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            time_step_start: 开始时间步
            time_step_end: 结束时间步
            time_step_stride: 时间步间隔
            normalize: 是否归一化
            augmentation: 数据增强配置
            seed: 随机种子
        """
        self.data_path = data_path
        self.T_in = T_in
        self.T_out = T_out
        self.split = split
        self.normalize = normalize
        self.augmentation = augmentation or {}
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 检查数据文件
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 打开HDF5文件
        self.h5_file = h5py.File(data_path, 'r')
        
        # 获取时间步键
        self.time_keys = [k for k in self.h5_file.keys() if k.isdigit()]
        self.time_keys.sort()
        
        print(f"📅 发现 {len(self.time_keys)} 个时间步")
        print(f"   时间步范围: {self.time_keys[0]} ~ {self.time_keys[-1]}")
        
        # 过滤时间步
        filtered_keys = []
        for i, key in enumerate(self.time_keys):
            time_idx = int(key)
            if time_step_start <= time_idx <= time_step_end and i % time_step_stride == 0:
                filtered_keys.append(key)
        
        self.time_keys = filtered_keys
        print(f"   过滤后时间步数: {len(self.time_keys)}")
        
        # 检查数据结构
        self._analyze_data_structure()
        
        # 生成有效序列索引
        self._generate_sequence_indices()
        
        # 数据集分割
        self._split_dataset(train_ratio, val_ratio, test_ratio, seed)
        
        # 计算归一化统计量
        if self.normalize:
            self._compute_normalization_stats()
    
    def _analyze_data_structure(self):
        """分析数据结构"""
        first_key = self.time_keys[0]
        first_group = self.h5_file[first_key]
        
        if 'data' not in first_group:
            raise ValueError(f"时间步 {first_key} 中未找到 'data' 数据集")
        
        data_shape = first_group['data'].shape
        print(f"📊 数据形状: {data_shape}")
        
        # 解析数据维度 (101, 128, 128, 2)
        if len(data_shape) == 4:
            self.n_samples, self.height, self.width, self.n_channels = data_shape
            print(f"   样本数: {self.n_samples}")
            print(f"   空间分辨率: {self.height} x {self.width}")
            print(f"   通道数: {self.n_channels}")
        else:
            raise ValueError(f"不支持的数据形状: {data_shape}")
        
        # 检查数据范围
        sample_data = first_group['data'][:3]  # 取前3个样本
        print(f"   数据范围: [{sample_data.min():.6f}, {sample_data.max():.6f}]")
        print(f"   数据均值: {sample_data.mean():.6f}")
        print(f"   数据标准差: {sample_data.std():.6f}")
    
    def _generate_sequence_indices(self):
        """生成有效的序列索引"""
        self.sequence_indices = []
        
        # 对每个样本生成时序序列
        for sample_idx in range(self.n_samples):
            # 检查是否有足够的时间步
            max_start_time = len(self.time_keys) - (self.T_in + self.T_out)
            if max_start_time < 0:
                continue
            
            # 生成所有可能的起始时间点
            for start_time in range(0, max_start_time + 1):
                self.sequence_indices.append((sample_idx, start_time))
        
        print(f"🔢 生成 {len(self.sequence_indices)} 个序列样本")
    
    def _split_dataset(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
        """分割数据集"""
        # 确保比例和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        
        # 随机打乱索引
        np.random.seed(seed)
        indices = np.random.permutation(len(self.sequence_indices))
        
        # 计算分割点
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # 分割索引
        if self.split == "train":
            self.indices = indices[:n_train]
        elif self.split == "val":
            self.indices = indices[n_train:n_train + n_val]
        elif self.split == "test":
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"不支持的分割类型: {self.split}")
        
        print(f"📊 {self.split} 集样本数: {len(self.indices)}")
    
    def _compute_normalization_stats(self):
        """计算归一化统计量"""
        if self.split != "train":
            # 非训练集不计算统计量，使用预设值或从训练集加载
            self.mean = torch.zeros(self.n_channels)
            self.std = torch.ones(self.n_channels)
            return
        
        print("📈 计算归一化统计量...")
        
        # 采样部分数据计算统计量
        sample_size = min(1000, len(self.indices))
        sample_indices = np.random.choice(len(self.indices), sample_size, replace=False)
        
        all_data = []
        for idx in sample_indices:
            sample_idx, start_time = self.sequence_indices[self.indices[idx]]
            
            # 读取输入和输出序列
            for t in range(self.T_in + self.T_out):
                time_key = self.time_keys[start_time + t]
                data = self.h5_file[time_key]['data'][sample_idx]  # [H, W, C]
                all_data.append(data)
        
        # 转换为numpy数组并计算统计量
        all_data = np.stack(all_data, axis=0)  # [N, H, W, C]
        
        # 按通道计算均值和标准差
        self.mean = torch.tensor(all_data.mean(axis=(0, 1, 2)), dtype=torch.float32)
        self.std = torch.tensor(all_data.std(axis=(0, 1, 2)), dtype=torch.float32)
        
        print(f"   均值: {self.mean}")
        print(f"   标准差: {self.std}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 获取序列索引
        sample_idx, start_time = self.sequence_indices[self.indices[idx]]
        
        # 读取输入序列
        input_sequence = []
        for t in range(self.T_in):
            time_key = self.time_keys[start_time + t]
            data = self.h5_file[time_key]['data'][sample_idx]  # [H, W, C]
            data = torch.tensor(data, dtype=torch.float32)
            
            # 转换为 [C, H, W] 格式
            data = data.permute(2, 0, 1)
            input_sequence.append(data)
        
        # 读取目标序列
        target_sequence = []
        for t in range(self.T_out):
            time_key = self.time_keys[start_time + self.T_in + t]
            data = self.h5_file[time_key]['data'][sample_idx]  # [H, W, C]
            data = torch.tensor(data, dtype=torch.float32)
            
            # 转换为 [C, H, W] 格式
            data = data.permute(2, 0, 1)
            target_sequence.append(data)
        
        # 堆叠为时序张量
        input_sequence = torch.stack(input_sequence, dim=0)    # [T_in, C, H, W]
        target_sequence = torch.stack(target_sequence, dim=0)  # [T_out, C, H, W]
        
        # 归一化
        if self.normalize:
            input_sequence = (input_sequence - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
            target_sequence = (target_sequence - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
        
        # 数据增强
        if self.split == "train" and self.augmentation.get('enabled', False):
            input_sequence, target_sequence = self._apply_augmentation(input_sequence, target_sequence)
        
        return {
            'input_sequence': input_sequence,      # [T_in, C, H, W]
            'target_sequence': target_sequence,    # [T_out, C, H, W]
            'sample_idx': sample_idx,
            'start_time': start_time,
            'time_keys': [self.time_keys[start_time + t] for t in range(self.T_in + self.T_out)]
        }
    
    def _apply_augmentation(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用数据增强"""
        # 水平翻转
        if np.random.random() < self.augmentation.get('flip_prob', 0.0):
            input_seq = torch.flip(input_seq, dims=[-1])
            target_seq = torch.flip(target_seq, dims=[-1])
        
        # 旋转（90度的倍数）
        if np.random.random() < self.augmentation.get('rotate_prob', 0.0):
            k = np.random.randint(1, 4)  # 90, 180, 270度
            input_seq = torch.rot90(input_seq, k=k, dims=[-2, -1])
            target_seq = torch.rot90(target_seq, k=k, dims=[-2, -1])
        
        # 添加噪声
        noise_std = self.augmentation.get('noise_std', 0.0)
        if noise_std > 0:
            input_seq += torch.randn_like(input_seq) * noise_std
            target_seq += torch.randn_like(target_seq) * noise_std
        
        return input_seq, target_seq
    
    def __del__(self):
        """析构函数，关闭HDF5文件"""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()


class RealDiffusionReactionDataModule(pl.LightningDataModule):
    """真实扩散-反应数据模块"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.data_path = config.data.data_path
        
        # 时序配置
        self.T_in = config.data.get('T_in', 1)
        self.T_out = config.data.get('T_out', 20)
        
        # 数据集分割配置
        self.train_ratio = config.data.get('train_ratio', 0.7)
        self.val_ratio = config.data.get('val_ratio', 0.15)
        self.test_ratio = config.data.get('test_ratio', 0.15)
        
        # 时间步配置
        self.time_step_start = config.data.get('time_step_start', 0)
        self.time_step_end = config.data.get('time_step_end', 980)
        self.time_step_stride = config.data.get('time_step_stride', 1)
        
        # 其他配置
        self.normalize = config.data.get('normalize', True)
        self.augmentation = config.data.get('augmentation', {})
        self.seed = config.get('seed', 2025)
        
        # 数据加载器配置
        self.batch_size = config.training.get('batch_size', 8)
        self.num_workers = config.hardware.get('num_workers', 4)
        self.pin_memory = config.hardware.get('pin_memory', True)
        self.persistent_workers = config.hardware.get('persistent_workers', True)
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = RealDiffusionReactionDataset(
                data_path=self.data_path,
                T_in=self.T_in,
                T_out=self.T_out,
                split="train",
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                time_step_start=self.time_step_start,
                time_step_end=self.time_step_end,
                time_step_stride=self.time_step_stride,
                normalize=self.normalize,
                augmentation=self.augmentation,
                seed=self.seed
            )
            
            self.val_dataset = RealDiffusionReactionDataset(
                data_path=self.data_path,
                T_in=self.T_in,
                T_out=self.T_out,
                split="val",
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                time_step_start=self.time_step_start,
                time_step_end=self.time_step_end,
                time_step_stride=self.time_step_stride,
                normalize=self.normalize,
                seed=self.seed
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = RealDiffusionReactionDataset(
                data_path=self.data_path,
                T_in=self.T_in,
                T_out=self.T_out,
                split="test",
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                time_step_start=self.time_step_start,
                time_step_end=self.time_step_end,
                time_step_stride=self.time_step_stride,
                normalize=self.normalize,
                seed=self.seed
            )
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # HDF5文件不支持多进程
            pin_memory=False,  # 避免设备不匹配
            drop_last=True,
            persistent_workers=False
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False
        )


if __name__ == "__main__":
    """测试数据集加载器"""
    from omegaconf import DictConfig
    
    # 测试配置
    config = DictConfig({
        'data': {
            'data_path': 'E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
            'T_in': 1,
            'T_out': 20,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'time_step_start': 0,
            'time_step_end': 980,
            'time_step_stride': 1,
            'normalize': True,
            'augmentation': {
                'enabled': True,
                'flip_prob': 0.5,
                'rotate_prob': 0.3,
                'noise_std': 0.01
            }
        },
        'training': {
            'batch_size': 4
        },
        'hardware': {
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False
        },
        'seed': 2025
    })
    
    # 创建数据模块
    data_module = RealDiffusionReactionDataModule(config)
    data_module.setup()
    
    # 测试训练数据加载器
    train_loader = data_module.train_dataloader()
    print(f"训练集批次数: {len(train_loader)}")
    
    # 获取一个批次
    batch = next(iter(train_loader))
    print(f"输入序列形状: {batch['input_sequence'].shape}")
    print(f"目标序列形状: {batch['target_sequence'].shape}")
    print(f"样本索引: {batch['sample_idx']}")
    print(f"起始时间: {batch['start_time']}")
    
    print("✅ 数据集加载器测试成功！")