"""时序PDEBench数据处理模块

实现时序PDE数据读取器，支持时间序列数据的读取和处理。
严格按照开发手册要求，确保观测算子H与训练DC复用同一实现。
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import DictConfig
import cv2

from datasets.pdebench import PDEBenchBase
from ops.degradation import apply_degradation_operator


class TemporalPDEBenchBase(PDEBenchBase):
    """时序PDEBench数据集基类
    
    支持时间序列数据读取，输出格式为[B, T, C, H, W]
    """
    
    def __init__(
        self,
        data_path: str,
        keys: List[str],
        T_in: int = 1,
        T_out: int = 3,
        dt: float = 0.1,
        temporal_mode: str = "sequential",  # "sequential" or "random"
        sequence_length: Optional[int] = None,
        overlap_ratio: float = 0.0,
        **kwargs
    ):
        """初始化时序PDEBench数据集
        
        Args:
            T_in: 输入时间步数
            T_out: 输出时间步数
            dt: 时间步长
            temporal_mode: 时序采样模式
            sequence_length: 序列长度（如果None则使用T_in+T_out）
            overlap_ratio: 序列重叠比例
        """
        # 确保keys是列表格式
        if hasattr(keys, '__iter__') and not isinstance(keys, str):
            keys = list(keys)
        elif hasattr(keys, 'keys'):  # 处理可能的方法对象
            keys = ["u"]  # 使用默认值
        
        super().__init__(data_path, keys, **kwargs)
        
        self.T_in = T_in
        self.T_out = T_out
        self.dt = dt
        self.temporal_mode = temporal_mode
        self.sequence_length = sequence_length or (T_in + T_out)
        self.overlap_ratio = overlap_ratio
        
        # 获取时间维度信息
        self._analyze_temporal_structure()
        
        # 生成时序样本索引
        self._generate_temporal_indices()
        
        print(f"Temporal dataset initialized:")
        print(f"  T_in={T_in}, T_out={T_out}, dt={dt}")
        print(f"  Total temporal samples: {len(self.temporal_indices)}")
        print(f"  Available time steps: {self.n_timesteps}")
    
    def _analyze_temporal_structure(self):
        """分析数据的时间结构"""
        # 打开第一个case来分析时间结构
        if self.use_official_format:
            if "diff-react" in str(self.data_path).lower():
                # diff-react数据集特殊处理 - 检查样本组结构
                first_key = self.keys[0]
                sample_keys = [k for k in self.h5_file.keys() if k.isdigit()]
                if sample_keys and first_key in self.h5_file[sample_keys[0]]:
                    # 从第一个样本组中获取数据形状
                    data_shape = self.h5_file[sample_keys[0]][first_key].shape
                    # 数据格式是 [T, H, W, C]
                    self.n_timesteps = data_shape[0]
                elif first_key in self.h5_file:
                    data_shape = self.h5_file[first_key].shape
                    # 我们的数据格式是 [n_samples, n_timesteps, nx, ny]
                    self.n_timesteps = data_shape[1]
                else:
                    raise ValueError(f"Key '{first_key}' not found in HDF5 file")
            elif 'data' in self.h5_file:
                data_shape = self.h5_file['data'].shape
                if len(data_shape) == 5:
                    # [B, T, H, W, C]
                    self.n_timesteps = data_shape[1]
                elif len(data_shape) == 4:
                    # [B, H, W, C] - 单时间步
                    self.n_timesteps = 1
                else:
                    self.n_timesteps = 1
            else:
                # 使用变量键分析
                first_key = self.keys[0]
                if first_key in self.h5_file:
                    key_shape = self.h5_file[first_key].shape
                    if len(key_shape) >= 2:
                        # 假设第二维是时间维
                        self.n_timesteps = key_shape[1] if len(key_shape) > 1 else 1
                    else:
                        self.n_timesteps = 1
                else:
                    self.n_timesteps = 1
        else:
            # 原格式处理
            if 'data' in self.h5_file:
                data_shape = self.h5_file['data'].shape
                self.n_timesteps = data_shape[0]  # [T, C, H, W]
            else:
                first_key = self.keys[0]
                if first_key in self.h5_file:
                    key_shape = self.h5_file[first_key].shape
                    self.n_timesteps = key_shape[0]  # 假设第一维是时间维
                else:
                    self.n_timesteps = 1
    
    def _generate_temporal_indices(self):
        """生成时序样本索引"""
        self.temporal_indices = []
        
        for case_id in self.case_ids:
            # 为每个case生成时序索引
            max_start_time = max(0, self.n_timesteps - self.sequence_length)
            
            if self.temporal_mode == "sequential":
                # 顺序采样
                step = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
                for t_start in range(0, max_start_time + 1, step):
                    self.temporal_indices.append({
                        'case_id': case_id,
                        't_start': t_start,
                        't_end': t_start + self.sequence_length
                    })
            elif self.temporal_mode == "random":
                # 随机采样（这里先生成所有可能的起始点，实际随机在__getitem__中实现）
                for t_start in range(max_start_time + 1):
                    self.temporal_indices.append({
                        'case_id': case_id,
                        't_start': t_start,
                        't_end': t_start + self.sequence_length
                    })
    
    def __len__(self):
        return len(self.temporal_indices)
    
    def _load_temporal_data(self, case_id: str, t_start: int, t_end: int) -> torch.Tensor:
        """加载时序数据
        
        Returns:
            torch.Tensor: [T, C, H, W] 格式的时序数据
        """
        case_idx = int(case_id) if case_id.isdigit() else self.case_ids.index(case_id)
        
        if self.use_official_format:
            # 官方格式处理
            if "diff-react" in str(self.data_path).lower():
                # diff-react数据集特殊处理
                if case_id in self.h5_file:
                    group = self.h5_file[case_id]
                    if 'data' in group:
                        # data形状：[T, H, W, C]
                        data = torch.tensor(group['data'][t_start:t_end], dtype=torch.float32)  # [T, H, W, C]
                        data = data.permute(0, 3, 1, 2)  # [T, C, H, W]
                    else:
                        raise ValueError(f"No 'data' found in group '{case_id}'")
                else:
                    # 对于diff-react数据集，直接使用'u'键
                    if 'u' in self.h5_file:
                        data_shape = self.h5_file['u'].shape  # (100, 50, 128, 128) -> [B, T, H, W]
                        if case_idx >= data_shape[0]:
                            raise IndexError(f"Case index {case_idx} out of range for shape {data_shape}")
                        data = torch.tensor(self.h5_file['u'][case_idx, t_start:t_end], dtype=torch.float32)  # [T, H, W]
                        data = data.unsqueeze(1)  # [T, 1, H, W] - 添加通道维度
                    else:
                        # 使用数字索引获取组名
                        root_keys = list(self.h5_file.keys())
                        if case_idx < len(root_keys):
                            group_key = root_keys[case_idx]
                            group = self.h5_file[group_key]
                            if 'data' in group:
                                data = torch.tensor(group['data'][t_start:t_end], dtype=torch.float32)  # [T, H, W, C]
                                data = data.permute(0, 3, 1, 2)  # [T, C, H, W]
                            else:
                                raise ValueError(f"No 'data' found in group '{group_key}'")
                        else:
                            raise IndexError(f"Index {case_idx} out of range")
            
            elif 'data' in self.h5_file:
                # 标准格式：直接有'data'键
                data_shape = self.h5_file['data'].shape
                if len(data_shape) == 5:
                    # 5D格式：[B, T, H, W, C]
                    data = torch.tensor(self.h5_file['data'][0, t_start:t_end], dtype=torch.float32)  # [T, H, W, C]
                    data = data.permute(0, 3, 1, 2)  # [T, C, H, W]
                elif len(data_shape) == 4:
                    # 4D格式：[B, H, W, C] - 单时间步，复制多个时间步
                    single_data = torch.tensor(self.h5_file['data'][case_idx], dtype=torch.float32)  # [H, W, C]
                    single_data = single_data.permute(2, 0, 1)  # [C, H, W]
                    # 复制到所需的时间步数
                    data = single_data.unsqueeze(0).repeat(t_end - t_start, 1, 1, 1)  # [T, C, H, W]
                else:
                    raise ValueError(f"Unsupported data shape: {data_shape}")
            else:
                # 使用变量键读取数据
                data_list = []
                for key in self.keys:
                    if key in self.h5_file:
                        item = self.h5_file[key]
                        key_shape = item.shape
                        
                        if len(key_shape) == 5:
                            # 5D格式：[B, T, H, W, C]
                            var_data = torch.tensor(item[0, t_start:t_end], dtype=torch.float32)  # [T, H, W, C]
                            var_data = var_data.permute(0, 3, 1, 2)  # [T, C, H, W]
                        elif len(key_shape) == 4:
                            # 4D格式：[B, T, H, W] 或 [T, H, W, C] 或 [B, H, W, C]
                            if key_shape[1] > key_shape[0]:  # 第二维更大，可能是时间维 [B, T, H, W]
                                if case_idx >= key_shape[0]:
                                    raise IndexError(f"Case index {case_idx} out of range for shape {key_shape}")
                                var_data = torch.tensor(item[case_idx, t_start:t_end], dtype=torch.float32)  # [T, H, W]
                                var_data = var_data.unsqueeze(1)  # [T, 1, H, W]
                            elif key_shape[0] > 10:  # 第一维很大，可能是时间维 [T, H, W, C]
                                var_data = torch.tensor(item[t_start:t_end], dtype=torch.float32)  # [T, H, W, C]
                                if var_data.dim() == 3:  # [T, H, W] -> [T, 1, H, W]
                                    var_data = var_data.unsqueeze(1)
                                else:  # [T, H, W, C] -> [T, C, H, W]
                                    var_data = var_data.permute(0, 3, 1, 2)
                            else:  # 单时间步，复制 [B, H, W, C]
                                if case_idx >= key_shape[0]:
                                    raise IndexError(f"Case index {case_idx} out of range for shape {key_shape}")
                                single_data = torch.tensor(item[case_idx], dtype=torch.float32)  # [H, W, C]
                                if single_data.dim() == 2:  # [H, W] -> [1, H, W]
                                    single_data = single_data.unsqueeze(0)
                                else:  # [H, W, C] -> [C, H, W]
                                    single_data = single_data.permute(2, 0, 1)
                                var_data = single_data.unsqueeze(0).repeat(t_end - t_start, 1, 1, 1)
                        else:
                            raise ValueError(f"Unsupported key shape for {key}: {key_shape}")
                        
                        data_list.append(var_data)
                
                if data_list:
                    data = torch.cat(data_list, dim=1)  # [T, C, H, W]
                else:
                    raise ValueError(f"No valid data found for keys: {self.keys}")
        else:
            # 原格式处理
            if 'data' in self.h5_file:
                # [T, C, H, W] 格式
                data = torch.tensor(self.h5_file['data'][t_start:t_end], dtype=torch.float32)  # [T, C, H, W]
            else:
                # 使用变量键读取数据
                data_list = []
                for key in self.keys:
                    if key in self.h5_file:
                        # 假设形状为 [T, H, W] 或 [T, C, H, W]
                        var_data = torch.tensor(self.h5_file[key][t_start:t_end], dtype=torch.float32)
                        
                        if var_data.dim() == 3:  # [T, H, W] -> [T, 1, H, W]
                            var_data = var_data.unsqueeze(1)
                        
                        data_list.append(var_data)
                
                if data_list:
                    data = torch.cat(data_list, dim=1)  # [T, C, H, W]
                else:
                    raise ValueError(f"No valid data found for keys: {self.keys}")
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取时序样本
        
        Returns:
            Dict containing:
                - input_sequence: [T_in, C, H, W] 输入序列
                - target_sequence: [T_out, C, H, W] 目标序列
                - full_sequence: [T_in+T_out, C, H, W] 完整序列
                - case_id: str
                - time_info: Dict with temporal metadata
        """
        temporal_info = self.temporal_indices[idx]
        case_id = temporal_info['case_id']
        t_start = temporal_info['t_start']
        t_end = temporal_info['t_end']
        
        # 加载完整时序数据
        full_sequence = self._load_temporal_data(case_id, t_start, t_end)  # [T, C, H, W]
        
        # 处理多通道数据和归一化
        processed_sequence = []
        for t in range(full_sequence.shape[0]):
            frame = full_sequence[t]  # [C, H, W]
            
            # 处理每个通道的归一化
            frame_list = []
            for i, key in enumerate(self.keys):
                if i < frame.shape[0]:
                    channel_data = frame[i:i+1]  # [1, H, W]
                else:
                    channel_data = frame[0:1]  # 使用第一个通道
                
                # 归一化
                if self.normalize and self.norm_stats:
                    channel_data = self._normalize_data(channel_data, key)
                
                frame_list.append(channel_data)
            
            # 拼接通道
            processed_frame = torch.cat(frame_list, dim=0)  # [C, H, W]
            
            # 调整尺寸
            if processed_frame.shape[-2:] != self.image_size:
                processed_frame = F.interpolate(
                    processed_frame.unsqueeze(0),
                    size=self.image_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            processed_sequence.append(processed_frame)
        
        # 堆叠为时序数据
        full_sequence = torch.stack(processed_sequence, dim=0)  # [T, C, H, W]
        
        # 分割输入和目标序列
        input_sequence = full_sequence[:self.T_in]  # [T_in, C, H, W]
        target_sequence = full_sequence[self.T_in:self.T_in + self.T_out]  # [T_out, C, H, W]
        
        # 时间信息
        time_info = {
            't_start': t_start,
            't_end': t_end,
            'dt': self.dt,
            'T_in': self.T_in,
            'T_out': self.T_out,
            'timestamps': torch.arange(t_start, t_end, dtype=torch.float32) * self.dt
        }
        
        return {
            'input_sequence': input_sequence.cpu(),  # [T_in, C, H, W]
            'target_sequence': target_sequence.cpu(),  # [T_out, C, H, W]
            'full_sequence': full_sequence.cpu(),  # [T_in+T_out, C, H, W]
            'case_id': case_id,
            'time_info': time_info,
            'task_params': {'task': 'temporal'},
        }


class TemporalPDEBenchSR(TemporalPDEBenchBase):
    """时序超分辨率数据集"""
    
    def __init__(
        self,
        data_path: str,
        keys: List[str],
        scale: int,
        sigma: float = 1.0,
        blur_kernel: int = 5,
        boundary: str = "mirror",
        noise_std: float = 0.0,
        **kwargs
    ):
        super().__init__(data_path, keys, **kwargs)
        
        self.scale = scale
        self.sigma = sigma
        self.blur_kernel = blur_kernel
        self.boundary = boundary
        self.noise_std = noise_std
        
        # H算子参数
        self.h_params = {
            'task': 'SR',
            'scale': scale,
            'sigma': sigma,
            'blur_kernel': blur_kernel,
            'boundary': boundary,
            'noise_std': noise_std,
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取时序SR样本"""
        sample = super().__getitem__(idx)
        
        # 对输入序列应用SR降质
        input_sequence = sample['input_sequence']  # [T_in, C, H, W]
        target_sequence = sample['target_sequence']  # [T_out, C, H, W]
        
        # 对每个时间步应用降质
        degraded_input = []
        for t in range(input_sequence.shape[0]):
            frame = input_sequence[t]  # [C, H, W]
            degraded_frame = apply_degradation_operator(frame.unsqueeze(0), self.h_params).squeeze(0)
            degraded_input.append(degraded_frame)
        
        degraded_input = torch.stack(degraded_input, dim=0)  # [T_in, C, H, W]
        
        sample.update({
            'observation_sequence': degraded_input.cpu(),  # [T_in, C, H, W]
            'h_params': self.h_params,
        })
        
        return sample


class TemporalPDEBenchCrop(TemporalPDEBenchBase):
    """时序裁剪数据集"""
    
    def __init__(
        self,
        data_path: str,
        keys: List[str],
        crop_ratio: float,
        boundary: str = "mirror",
        noise_std: float = 0.0,
        **kwargs
    ):
        super().__init__(data_path, keys, **kwargs)
        
        self.crop_ratio = crop_ratio
        self.boundary = boundary
        self.noise_std = noise_std
        
        # H算子参数
        self.h_params = {
            'task': 'Crop',
            'crop_ratio': crop_ratio,
            'boundary': boundary,
            'noise_std': noise_std,
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取时序Crop样本"""
        sample = super().__getitem__(idx)
        
        # 对输入序列应用Crop降质
        input_sequence = sample['input_sequence']  # [T_in, C, H, W]
        target_sequence = sample['target_sequence']  # [T_out, C, H, W]
        
        # 对每个时间步应用降质
        degraded_input = []
        for t in range(input_sequence.shape[0]):
            frame = input_sequence[t]  # [C, H, W]
            degraded_frame = apply_degradation_operator(frame.unsqueeze(0), self.h_params).squeeze(0)
            degraded_input.append(degraded_frame)
        
        degraded_input = torch.stack(degraded_input, dim=0)  # [T_in, C, H, W]
        
        sample.update({
            'observation_sequence': degraded_input.cpu(),  # [T_in, C, H, W]
            'h_params': self.h_params,
        })
        
        return sample


class TemporalPDEBenchDataModule:
    """时序PDEBench数据模块"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # 时序参数
        self.T_in = config.temporal.T_in
        self.T_out = config.temporal.T_out
        self.dt = config.temporal.dt
        
        # 数据集参数
        self.data_path = config.data_path
        # 修复keys访问问题 - 直接从配置字典中获取
        try:
            # 尝试直接访问配置字典
            if 'keys' in config:
                self.keys = config['keys']
            else:
                self.keys = ["u"]  # 默认值
        except:
            self.keys = ["u"]  # 默认值
        self.task = config.task
        
        # 创建数据集
        self._create_datasets()
    
    def _create_datasets(self):
        """创建训练、验证、测试数据集"""
        dataset_kwargs = {
            'data_path': self.data_path,
            'keys': self.keys,
            'T_in': self.T_in,
            'T_out': self.T_out,
            'dt': self.dt,
            'temporal_mode': self.config.get('temporal_mode', 'sequential'),
            'sequence_length': self.config.get('sequence_length', None),
            'overlap_ratio': self.config.get('overlap_ratio', 0.0),
            'normalize': self.config.get('normalize', True),
            'image_size': self.config.get('image_size', 256),
            'use_official_format': self.config.get('use_official_format', False),
            'splits_dir': self.config.get('splits_dir', None),
        }
        
        # 根据任务类型选择数据集类
        if self.task == 'SR':
            dataset_class = TemporalPDEBenchSR
            dataset_kwargs.update({
                'scale': self.config.scale,
                'sigma': self.config.get('sigma', 1.0),
                'blur_kernel': self.config.get('blur_kernel', 5),
                'boundary': self.config.get('boundary', 'mirror'),
                'noise_std': self.config.get('noise_std', 0.0),
            })
        elif self.task == 'Crop':
            dataset_class = TemporalPDEBenchCrop
            dataset_kwargs.update({
                'crop_ratio': self.config.crop_ratio,
                'boundary': self.config.get('boundary', 'mirror'),
                'noise_std': self.config.get('noise_std', 0.0),
            })
        else:
            dataset_class = TemporalPDEBenchBase
        
        # 创建数据集
        self.train_dataset = dataset_class(split='train', **dataset_kwargs)
        self.val_dataset = dataset_class(split='val', **dataset_kwargs)
        self.test_dataset = dataset_class(split='test', **dataset_kwargs)
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        batch_size = self.config.get('dataloader', {}).get('batch_size', self.config.get('batch_size', 4))
        num_workers = self.config.get('dataloader', {}).get('num_workers', self.config.get('num_workers', 0))
        pin_memory = self.config.get('dataloader', {}).get('pin_memory', self.config.get('pin_memory', False))
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 由于h5py对象不能被pickle，设置为0
            pin_memory=False,  # 关闭pin_memory，避免设备不匹配
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        batch_size = self.config.get('dataloader', {}).get('batch_size', self.config.get('batch_size', 4))
        
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
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
        )