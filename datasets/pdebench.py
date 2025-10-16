"""PDEBench数据处理模块

实现PDEBench数据读取器、观测生成器和数据一致性算子。
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

from ops.degradation import apply_degradation_operator


class PDEBenchBase(Dataset):
    """PDEBench数据集基类
    
    支持HDF5格式数据读取，统一化处理多键名数据，z-score归一化。
    支持官方PDEBench格式：[batch, time, height, width, channels]
    """
    
    def __init__(
        self,
        data_path: str,
        keys: List[str],
        split: str = "train",
        splits_dir: Optional[str] = None,
        normalize: bool = True,
        image_size: int = 256,
        use_official_format: bool = False,  # 修改默认值为False
    ):
        """初始化PDEBench数据集
        
        Args:
            data_path: HDF5文件路径
            keys: 数据键名列表，如["u"]或["rho","ux","uy"]
            split: 数据切分，"train"/"val"/"test"
            splits_dir: 数据切分文件目录
            normalize: 是否进行z-score归一化
            image_size: 图像尺寸
            use_official_format: 是否使用官方PDEBench格式 [batch, time, height, width, channels]
        """
        self.data_path = data_path
        self.keys = keys
        self.split = split
        self.normalize = normalize
        self.image_size = image_size
        self.use_official_format = use_official_format
        
        # 先打开HDF5文件
        self.h5_file = h5py.File(data_path, 'r')
        
        # 加载数据切分
        self.case_ids = self._load_split_ids(splits_dir, split)
        
        # 加载归一化统计量
        self.norm_stats = self._load_norm_stats(splits_dir) if normalize else None
        
        # 验证数据完整性
        self._validate_data()
        
        # 打印调试信息
        print(f"DEBUG: use_official_format = {self.use_official_format}")
        print(f"DEBUG: keys = {self.keys}, type = {type(self.keys)}")
        print(f"DEBUG: keys is list: {isinstance(self.keys, list)}")
        print(f"DEBUG: HDF5 file keys: {list(self.h5_file.keys())}")
        if 'data' in self.h5_file:
            print(f"DEBUG: data shape = {self.h5_file['data'].shape}")
        
        # 转换keys为普通list（如果是ListConfig）
        if hasattr(self.keys, '__iter__') and not isinstance(self.keys, str):
            self.keys = list(self.keys)
            print(f"DEBUG: Converted keys to list: {self.keys}")
        
        if isinstance(self.keys, list) and len(self.keys) > 0 and self.keys[0] in self.h5_file:
            print(f"DEBUG: {self.keys[0]} shape = {self.h5_file[self.keys[0]].shape}")
        else:
            print(f"DEBUG: Key '{self.keys[0] if self.keys else 'None'}' not found in HDF5 file")
    
    def _load_split_ids(self, splits_dir: Optional[str], split: str) -> List[str]:
        """加载数据切分ID列表"""
        if splits_dir is None:
            # 如果没有指定切分目录，使用默认切分
            if hasattr(self, 'h5_file') and 'data' in self.h5_file:
                data_shape = self.h5_file['data'].shape
                if self.use_official_format:
                    # 官方格式：[batch, time, height, width, channels]
                    n_total = data_shape[1]  # 时间步数
                else:
                    # 原格式：[time, channels, height, width]
                    n_total = data_shape[0]  # 时间步数
            else:
                n_total = 20  # 默认20个样本
            
            total_ids = [str(i) for i in range(n_total)]
            if split == "train":
                return total_ids[:int(0.8 * n_total)]
            elif split == "val":
                return total_ids[int(0.8 * n_total):int(0.9 * n_total)]
            else:  # test
                return total_ids[int(0.9 * n_total):]
        
        # 获取数据根目录
        if self.data_path.endswith('.h5'):
            # 如果data_path是具体的h5文件，获取其父目录
            data_root = Path(self.data_path).parent.parent
        else:
            # 如果data_path是目录，直接使用
            data_root = Path(self.data_path)
        
        if isinstance(splits_dir, str):
            split_file = data_root / splits_dir / f"{split}.txt"
        else:
            split_file = splits_dir / f"{split}.txt"
        
        if not split_file.exists():
            # 如果切分文件不存在，使用默认切分
            # 获取数据总数
            if hasattr(self, 'h5_file') and isinstance(self.keys, list) and len(self.keys) > 0 and self.keys[0] in self.h5_file:
                n_total = self.h5_file[self.keys[0]].shape[0]
            else:
                n_total = 1000  # 默认值
            
            total_ids = [str(i) for i in range(n_total)]
            if split == "train":
                return total_ids[:int(0.8 * n_total)]
            elif split == "val":
                return total_ids[int(0.8 * n_total):int(0.9 * n_total)]
            else:  # test
                return total_ids[int(0.9 * n_total):]
        
        with open(split_file, 'r') as f:
            case_ids = [line.strip() for line in f if line.strip()]
        
        return case_ids
    
    def _load_norm_stats(self, splits_dir: Optional[str]) -> Dict[str, torch.Tensor]:
        """加载归一化统计量"""
        if splits_dir is None:
            return None
        
        # 获取数据根目录
        if self.data_path.endswith('.h5'):
            data_root = Path(self.data_path).parent.parent
        else:
            data_root = Path(self.data_path)
        stats_file = data_root / splits_dir / "norm_stat.npz"
        if not stats_file.exists():
            # 如果统计文件不存在，计算并保存
            return self._compute_norm_stats(stats_file)
        
        try:
            stats = np.load(stats_file)
            norm_stats = {}
            for key in self.keys:
                norm_stats[f"{key}_mean"] = torch.tensor(stats[f"{key}_mean"], dtype=torch.float32)
                norm_stats[f"{key}_std"] = torch.tensor(stats[f"{key}_std"], dtype=torch.float32)
            
            return norm_stats
        except Exception as e:
            print(f"Warning: Failed to load norm stats from {stats_file}: {e}")
            # 如果加载失败，重新计算
            return self._compute_norm_stats(stats_file)
    
    def _compute_norm_stats(self, save_path: Path) -> Dict[str, torch.Tensor]:
        """计算并保存归一化统计量"""
        print("Computing normalization statistics...")
        
        # 只在训练集上计算统计量
        if self.data_path.endswith('.h5'):
            data_root = Path(self.data_path).parent.parent
        else:
            data_root = Path(self.data_path)
        train_ids = self._load_split_ids(data_root / "splits", "train")
        
        stats = {}
        for i, key in enumerate(self.keys):
            values = []
            
            # 直接使用键名访问数据
            if key in self.h5_file:
                data = self.h5_file[key][:]
                print(f"Data shape for {key}: {data.shape}")
                values.append(data.flatten())
            else:
                print(f"Warning: Key {key} not found in HDF5 file")
                # 使用默认数据
                for case_id in train_ids:
                    # case_id 是文件名，需要从中提取数字索引
                    if case_id.endswith('.h5'):
                        # 从文件名中提取数字，如 "sample_data_2.h5" -> 2
                        case_idx = int(case_id.split('_')[-1].split('.')[0])
                    else:
                        case_idx = int(case_id)
                    
                    # 读取数据
                    if self.use_official_format:
                        # 官方格式：[batch, time, height, width, channels]
                        # 取第一个batch，指定时间步，所有空间点，第i个通道
                        if i < self.h5_file['data'].shape[4]:  # 确保通道索引有效
                            data = self.h5_file['data'][0, case_idx, :, :, i]  # [H, W]
                        else:
                            # 如果通道数不足，使用第一个通道
                            data = self.h5_file['data'][0, case_idx, :, :, 0]  # [H, W]
                    else:
                        # 原格式：[time, channels, height, width]
                        if i < self.h5_file['data'].shape[1]:  # 确保通道索引有效
                            data = self.h5_file['data'][case_idx, i, :, :]  # [H, W]
                        else:
                            # 如果通道数不足，使用第一个通道
                            data = self.h5_file['data'][case_idx, 0, :, :]  # [H, W]
                    
                    values.append(data.flatten())
            
            if values:
                all_values = np.concatenate(values)
                mean = np.mean(all_values)
                std = np.std(all_values)
                
                # 避免除零
                if std < 1e-8:
                    std = 1.0
                    print(f"Warning: std too small for {key}, set to 1.0")
                
                stats[f"{key}_mean"] = mean
                stats[f"{key}_std"] = std
        
        # 保存统计量
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **stats)
        
        # 转换为torch tensor
        norm_stats = {}
        for key in self.keys:
            norm_stats[f"{key}_mean"] = torch.tensor(stats[f"{key}_mean"], dtype=torch.float32)
            norm_stats[f"{key}_std"] = torch.tensor(stats[f"{key}_std"], dtype=torch.float32)
        
        return norm_stats
    
    def _validate_data(self):
        """验证数据完整性"""
        # 检查数据形状是否正确
        # 首先检查是否有'data'键，如果没有则检查是否有变量键
        print(f"DEBUG: Available keys in HDF5 file: {list(self.h5_file.keys())}")
        print(f"DEBUG: Looking for keys: {self.keys}")
        print(f"DEBUG: Keys type: {type(self.keys)}")
        print(f"DEBUG: Keys is list: {isinstance(self.keys, list)}")
        
        # 转换keys为普通list（如果是ListConfig）
        if hasattr(self.keys, '__iter__') and not isinstance(self.keys, str):
            self.keys = list(self.keys)
            print(f"DEBUG: Converted keys to list: {self.keys}")
        
        if 'data' in self.h5_file:
            data_shape = self.h5_file['data'].shape
            print(f"DEBUG: Using 'data' key with shape: {data_shape}")
        elif isinstance(self.keys, list) and len(self.keys) > 0:
            print(f"DEBUG: Checking if key '{self.keys[0]}' exists in file...")
            print(f"DEBUG: Key exists: {self.keys[0] in self.h5_file}")
            if self.keys[0] in self.h5_file:
                # 如果没有'data'键但有变量键，使用第一个变量的形状
                data_shape = self.h5_file[self.keys[0]].shape
                print(f"DEBUG: Using key '{self.keys[0]}' with shape: {data_shape}")
            else:
                print(f"ERROR: Key '{self.keys[0]}' not found in file")
                raise ValueError(f"HDF5 file must contain 'data' key or variable keys {self.keys}")
        else:
            print(f"ERROR: No keys specified or keys list is empty")
            print(f"DEBUG: Keys value: {self.keys}")
            print(f"DEBUG: Keys repr: {repr(self.keys)}")
            raise ValueError(f"HDF5 file must contain 'data' key or variable keys {self.keys}")
        
        if self.use_official_format:
            # 官方格式：[batch, time, height, width, channels]
            if len(data_shape) != 5:
                raise ValueError(f"Official format data must be 5D [B, T, H, W, C], got shape {data_shape}")
            
            if data_shape[4] < len(self.keys):
                raise ValueError(f"Data has {data_shape[4]} channels but {len(self.keys)} keys specified")
        else:
            # 原格式：[time, channels, height, width] 或 [time, height, width] (单通道)
            if len(data_shape) == 4:
                # 标准4D格式 [T, C, H, W]
                if data_shape[1] < len(self.keys):
                    raise ValueError(f"Data has {data_shape[1]} channels but {len(self.keys)} keys specified")
            elif len(data_shape) == 3:
                # 单通道3D格式 [T, H, W]，只支持单变量
                if len(self.keys) > 1:
                    raise ValueError(f"3D data [T, H, W] only supports single variable, but {len(self.keys)} keys specified")
            else:
                raise ValueError(f"Data must be 3D [T, H, W] or 4D [T, C, H, W], got shape {data_shape}")
    
    def _normalize_data(self, data: torch.Tensor, key: str) -> torch.Tensor:
        """对数据进行z-score归一化"""
        if not self.normalize or self.norm_stats is None:
            return data
        
        mean = self.norm_stats[f"{key}_mean"]
        std = self.norm_stats[f"{key}_std"]
        
        return (data - mean) / (std + 1e-8)
    
    def _denormalize_data(self, data: torch.Tensor, key: str) -> torch.Tensor:
        """反归一化数据到原值域"""
        if not self.normalize or self.norm_stats is None:
            return data
        
        mean = self.norm_stats[f"{key}_mean"]
        std = self.norm_stats[f"{key}_std"]
        
        return data * std + mean
    
    def __len__(self) -> int:
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项"""
        case_id = self.case_ids[idx]
        # case_id 可能是文件名或者包含路径的字符串
        if '/' in case_id:
            # 处理类似 "sample_data_2.h5/sample_6" 的情况
            parts = case_id.split('/')
            if len(parts) >= 2:
                # 从最后一部分提取数字，如 "sample_6" -> 6
                sample_part = parts[-1]
                if 'sample_' in sample_part:
                    case_idx = int(sample_part.split('_')[-1])
                else:
                    case_idx = idx  # 使用索引作为备选
            else:
                case_idx = idx
        elif case_id.endswith('.h5'):
            # 从文件名中提取数字，如 "sample_data_2.h5" -> 2
            case_idx = int(case_id.split('_')[-1].split('.')[0])
        else:
            try:
                case_idx = int(case_id)
            except ValueError:
                # 如果无法转换为整数，使用索引
                case_idx = idx
        
        # 读取数据
        if self.use_official_format:
            # 官方格式：[batch, time, height, width, channels]
            # 取第一个batch，指定时间步的数据
            if 'data' in self.h5_file:
                data = torch.tensor(self.h5_file['data'][0, case_idx], dtype=torch.float32)  # [H, W, C]
                # 转换为 [C, H, W] 格式
                data = data.permute(2, 0, 1)  # [C, H, W]
            else:
                # 使用变量键读取数据
                data_list = []
                for key in self.keys:
                    if key in self.h5_file:
                        var_data = torch.tensor(self.h5_file[key][0, case_idx], dtype=torch.float32)  # [H, W]
                        data_list.append(var_data.unsqueeze(0))  # [1, H, W]
                data = torch.cat(data_list, dim=0)  # [C, H, W]
        else:
            # 原格式：针对tensor数据的特殊处理
            if 'data' in self.h5_file:
                data = torch.tensor(self.h5_file['data'][case_idx], dtype=torch.float32)  # [C, H, W]
            else:
                # 使用变量键读取数据 - 专门处理tensor数据
                data_list = []
                for key in self.keys:
                    if key in self.h5_file:
                        # tensor数据形状为(10000, 1, 128, 128)，取第case_idx个样本
                        var_data = torch.tensor(self.h5_file[key][case_idx], dtype=torch.float32)  # [1, H, W] 或 [H, W]
                        
                        # 确保数据为3维 [C, H, W]
                        if var_data.dim() == 2:  # [H, W] -> [1, H, W]
                            var_data = var_data.unsqueeze(0)
                        elif var_data.dim() == 3:  # 已经是 [C, H, W] 格式
                            pass
                        else:
                            raise ValueError(f"Unexpected tensor dimension for key '{key}': {var_data.shape}")
                        
                        data_list.append(var_data)
                
                # 拼接所有键的数据
                if data_list:
                    data = torch.cat(data_list, dim=0)  # [C, H, W]
                else:
                    raise ValueError(f"No valid data found for keys: {self.keys}")
        
        # 处理多通道数据
        data_list = []
        for i, key in enumerate(self.keys):
            if i < data.shape[0]:  # 确保通道索引有效
                channel_data = data[i:i+1]  # [1, H, W]
            else:
                # 如果通道数不足，使用第一个通道
                channel_data = data[0:1]  # [1, H, W]
            
            # 归一化
            if self.normalize and self.norm_stats:
                channel_data = self._normalize_data(channel_data, key)
            data_list.append(channel_data)
        
        # 拼接多通道 [C, H, W]
        target = torch.cat(data_list, dim=0)
        
        # 调整尺寸到指定大小
        if target.shape[-2:] != (self.image_size, self.image_size):
            target = F.interpolate(
                target.unsqueeze(0), 
                size=(self.image_size, self.image_size),
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # 生成观测数据（这里需要根据观测模式生成）
        # 暂时返回target作为observation，后续会在子类中重写
        observation = target.clone()
        
        return {
            'target': target.cpu(),  # [C, H, W]
            'observation': observation.cpu(),  # [C, H, W] 观测数据
            'case_id': case_id,
            'task_params': {'task': 'base'},  # 基础任务参数
        }


class PDEBenchSR(PDEBenchBase):
    """超分辨率观测数据集
    
    实现SR模式的观测生成：高斯模糊 + 区域下采样
    """
    
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
        """初始化SR数据集
        
        Args:
            scale: 下采样倍率
            sigma: 高斯模糊标准差
            blur_kernel: 模糊核大小
            boundary: 边界策略 "mirror"/"wrap"/"zero"
            noise_std: 噪声标准差
        """
        # 过滤掉不属于基类的参数
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['split', 'splits_dir', 'normalize', 'image_size', 'use_official_format']}
        super().__init__(data_path, keys, **base_kwargs)
        
        self.scale = scale
        self.sigma = sigma
        self.blur_kernel = blur_kernel
        self.boundary = boundary
        self.noise_std = noise_std
        
        # H算子参数，确保与训练DC一致
        self.h_params = {
            'task': 'SR',
            'scale': scale,
            'sigma': sigma,
            'kernel_size': blur_kernel,
            'boundary': boundary,
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取SR观测数据"""
        data = super().__getitem__(idx)
        target = data['target']  # [C, H, W]
        
        # 生成SR观测
        obs_data = self._generate_sr_observation(target)
        
        # 添加噪声
        if self.noise_std > 0:
            noise = torch.randn_like(obs_data['baseline']) * self.noise_std
            obs_data['baseline'] = obs_data['baseline'] + noise
        
        # 更新数据字典
        data.update(obs_data)
        data['h_params'] = self.h_params
        data['task_params'] = self.h_params  # 使用h_params作为task_params
        # 设置observation为baseline
        data['observation'] = obs_data['baseline']
        
        return data
    
    def _generate_sr_observation(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成SR观测数据
        
        Args:
            target: 目标数据 [C, H, W]
            
        Returns:
            观测数据字典，包含baseline, coords, mask
        """
        C, H, W = target.shape
        
        # 使用统一的H算子生成观测
        # 注意：这里需要反归一化到原值域进行H操作
        target_orig = target.clone()
        if self.normalize and self.norm_stats is not None:
            for i, key in enumerate(self.keys):
                target_orig[i:i+1] = self._denormalize_data(target[i:i+1], key)
        
        # 应用H算子：blur + downsample
        lr_orig = apply_degradation_operator(
            target_orig.unsqueeze(0), 
            self.h_params
        ).squeeze(0)  # [C, H//scale, W//scale]
        
        # 上采样回原尺寸作为baseline
        baseline_orig = F.interpolate(
            lr_orig.unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # [C, H, W]
        
        # 重新归一化baseline
        baseline = baseline_orig.clone()
        if self.normalize and self.norm_stats is not None:
            for i, key in enumerate(self.keys):
                baseline[i:i+1] = self._normalize_data(baseline_orig[i:i+1], key)
        
        # 生成坐标网格
        coords = self._generate_coords(H, W)  # [2, H, W]
        
        # 生成mask（SR模式下全为1）
        mask = torch.ones(1, H, W)  # [1, H, W]
        
        return {
            'baseline': baseline.cpu(),  # [C, H, W] 上采样后的低分辨率观测
            'coords': coords.cpu(),      # [2, H, W] 坐标网格
            'mask': mask.cpu(),          # [1, H, W] 观测mask
            'lr_observation': lr_orig.cpu(),  # 原始低分辨率观测 [C, H//scale, W//scale]
            'original_observation': lr_orig.cpu(),  # 用于损失计算的原始观测数据
        }
    
    def _generate_coords(self, H: int, W: int) -> torch.Tensor:
        """生成归一化坐标网格"""
        y_coords = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
        x_coords = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
        coords = torch.stack([x_coords, y_coords], dim=0)  # [2, H, W]
        return coords


class PDEBenchCrop(PDEBenchBase):
    """裁剪观测数据集
    
    实现Crop模式的观测生成：窗口裁剪 + 对齐处理
    """
    
    def __init__(
        self,
        data_path: str,
        keys: List[str],
        crop_size: Tuple[int, int],
        patch_align: int = 8,
        center_sampler: str = "mixed",
        boundary: str = "mirror",
        **kwargs
    ):
        """初始化Crop数据集
        
        Args:
            crop_size: 裁剪窗口尺寸 (H, W)
            patch_align: patch对齐倍数
            center_sampler: 中心采样策略 "mixed"/"uniform"/"boundary"/"gradient"
            boundary: 边界处理策略
        """
        super().__init__(data_path, keys, **kwargs)
        
        self.crop_size = crop_size
        self.patch_align = patch_align
        self.center_sampler = center_sampler
        self.boundary = boundary
        
        # 确保裁剪尺寸对齐
        self.crop_h = (crop_size[0] // patch_align) * patch_align
        self.crop_w = (crop_size[1] // patch_align) * patch_align
        
        # H算子参数
        self.h_params = {
            'task': 'Crop',
            'crop_size': (self.crop_h, self.crop_w),
            'patch_align': patch_align,
            'boundary': boundary,
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取Crop观测数据"""
        data = super().__getitem__(idx)
        target = data['target']  # [C, H, W]
        
        # 生成Crop观测
        obs_data = self._generate_crop_observation(target)
        
        # 更新数据字典
        data.update(obs_data)
        data['h_params'] = self.h_params
        data['task_params'] = self.h_params  # 使用h_params作为task_params
        # 设置observation为baseline
        data['observation'] = obs_data['baseline']
        
        return data
    
    def _generate_crop_observation(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成Crop观测数据"""
        C, H, W = target.shape
        
        # 采样裁剪中心
        center_y, center_x = self._sample_crop_center(H, W)
        
        # 计算裁剪边界框
        y1 = max(0, center_y - self.crop_h // 2)
        y2 = min(H, y1 + self.crop_h)
        x1 = max(0, center_x - self.crop_w // 2)
        x2 = min(W, x1 + self.crop_w)
        
        # 调整边界框确保尺寸正确
        if y2 - y1 < self.crop_h:
            if y1 == 0:
                y2 = min(H, self.crop_h)
            else:
                y1 = max(0, H - self.crop_h)
                y2 = H
        
        if x2 - x1 < self.crop_w:
            if x1 == 0:
                x2 = min(W, self.crop_w)
            else:
                x1 = max(0, W - self.crop_w)
                x2 = W
        
        crop_box = (x1, y1, x2, y2)
        
        # 使用H算子生成观测
        target_orig = target.clone()
        if self.normalize and self.norm_stats is not None:
            for i, key in enumerate(self.keys):
                target_orig[i:i+1] = self._denormalize_data(target[i:i+1], key)
        
        # 应用H算子：裁剪
        h_params_with_box = self.h_params.copy()
        h_params_with_box['crop_box'] = crop_box
        
        cropped_orig = apply_degradation_operator(
            target_orig.unsqueeze(0),
            h_params_with_box
        ).squeeze(0)  # [C, crop_h, crop_w]
        
        # 创建baseline（零填充到原尺寸）
        baseline_orig = torch.zeros_like(target_orig)
        baseline_orig[:, y1:y2, x1:x2] = cropped_orig
        
        # 重新归一化
        baseline = baseline_orig.clone()
        if self.normalize and self.norm_stats is not None:
            for i, key in enumerate(self.keys):
                baseline[i:i+1] = self._normalize_data(baseline_orig[i:i+1], key)
        
        # 生成mask
        mask = torch.zeros(1, H, W)
        mask[:, y1:y2, x1:x2] = 1.0
        
        # 生成坐标
        coords = self._generate_coords(H, W)
        
        return {
            'baseline': baseline,     # [C, H, W] 零填充的观测
            'coords': coords,         # [2, H, W] 坐标网格
            'mask': mask,            # [1, H, W] 观测mask
            'crop_box': crop_box,    # (x1, y1, x2, y2) 裁剪边界框
            'cropped_observation': cropped_orig,  # [C, crop_h, crop_w] 原始裁剪观测
        }
    
    def _sample_crop_center(self, H: int, W: int) -> Tuple[int, int]:
        """采样裁剪中心点"""
        if self.center_sampler == "uniform":
            # 均匀采样
            center_y = np.random.randint(self.crop_h // 2, H - self.crop_h // 2)
            center_x = np.random.randint(self.crop_w // 2, W - self.crop_w // 2)
        
        elif self.center_sampler == "boundary":
            # 边界优先采样
            if np.random.random() < 0.5:
                # 靠近边界
                if np.random.random() < 0.5:
                    center_y = np.random.randint(0, self.crop_h)
                else:
                    center_y = np.random.randint(H - self.crop_h, H)
                center_x = np.random.randint(self.crop_w // 2, W - self.crop_w // 2)
            else:
                center_y = np.random.randint(self.crop_h // 2, H - self.crop_h // 2)
                if np.random.random() < 0.5:
                    center_x = np.random.randint(0, self.crop_w)
                else:
                    center_x = np.random.randint(W - self.crop_w, W)
        
        elif self.center_sampler == "mixed":
            # 混合采样：均匀40% + 边界30% + 高梯度30%
            rand = np.random.random()
            if rand < 0.4:
                # 均匀采样
                center_y = np.random.randint(self.crop_h // 2, H - self.crop_h // 2)
                center_x = np.random.randint(self.crop_w // 2, W - self.crop_w // 2)
            elif rand < 0.7:
                # 边界采样
                return self._sample_crop_center_boundary(H, W)
            else:
                # 高梯度区域采样（简化为随机）
                center_y = np.random.randint(self.crop_h // 2, H - self.crop_h // 2)
                center_x = np.random.randint(self.crop_w // 2, W - self.crop_w // 2)
        
        else:
            # 默认中心采样
            center_y = H // 2
            center_x = W // 2
        
        return center_y, center_x
    
    def _sample_crop_center_boundary(self, H: int, W: int) -> Tuple[int, int]:
        """边界区域采样"""
        boundary_width = min(32, min(H, W) // 4)  # 边界带宽度
        
        # 选择边界区域
        regions = []
        # 上边界
        if self.crop_h // 2 < boundary_width:
            regions.append('top')
        # 下边界  
        if H - self.crop_h // 2 > H - boundary_width:
            regions.append('bottom')
        # 左边界
        if self.crop_w // 2 < boundary_width:
            regions.append('left')
        # 右边界
        if W - self.crop_w // 2 > W - boundary_width:
            regions.append('right')
        
        if not regions:
            # 如果没有有效边界区域，使用中心
            return H // 2, W // 2
        
        region = np.random.choice(regions)
        
        if region == 'top':
            center_y = np.random.randint(self.crop_h // 2, min(boundary_width, H - self.crop_h // 2))
            center_x = np.random.randint(self.crop_w // 2, W - self.crop_w // 2)
        elif region == 'bottom':
            center_y = np.random.randint(max(H - boundary_width, self.crop_h // 2), H - self.crop_h // 2)
            center_x = np.random.randint(self.crop_w // 2, W - self.crop_w // 2)
        elif region == 'left':
            center_y = np.random.randint(self.crop_h // 2, H - self.crop_h // 2)
            center_x = np.random.randint(self.crop_w // 2, min(boundary_width, W - self.crop_w // 2))
        else:  # right
            center_y = np.random.randint(self.crop_h // 2, H - self.crop_h // 2)
            center_x = np.random.randint(max(W - boundary_width, self.crop_w // 2), W - self.crop_w // 2)
        
        return center_y, center_x
    
    def _generate_coords(self, H: int, W: int) -> torch.Tensor:
        """生成归一化坐标网格"""
        y_coords = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
        x_coords = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
        coords = torch.stack([x_coords, y_coords], dim=0)  # [2, H, W]
        return coords


class PDEBenchDataModule:
    """PDEBench数据模块
    
    统一管理训练、验证、测试数据加载器
    """
    
    def __init__(self, config: DictConfig):
        """初始化数据模块
        
        Args:
            config: 数据配置
        """
        self.config = config
        
        # 数据集类选择
        observation_config = config.get('observation', {})
        task = observation_config.get('mode', 'SR').lower()
        
        if task == "sr":
            self.dataset_class = PDEBenchSR
            sr_config = observation_config.get('sr', {})
            self.dataset_kwargs = {
                'scale': sr_config.get('scale_factor', 4),
                'sigma': sr_config.get('blur_sigma', 1.0),
                'blur_kernel': sr_config.get('blur_kernel_size', 5),
                'boundary': sr_config.get('boundary_mode', 'mirror'),
            }
        elif task == "crop":
            self.dataset_class = PDEBenchCrop
            crop_config = observation_config.get('crop', {})
            self.dataset_kwargs = {
                'crop_size': crop_config.get('crop_size', (64, 64)),
                'strategy': crop_config.get('crop_strategy', 'mixed'),
                'boundary': crop_config.get('boundary_mode', 'mirror'),
            }
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # 通用参数
        self.common_kwargs = {
            'normalize': config.get('normalize', True),
            'image_size': config.get('image_size', 256),
        }
    
    def setup(self, stage: Optional[str] = None) -> None:
        """设置数据集"""
        # 添加调试信息
        print(f"DEBUG: config = {self.config}")
        print(f"DEBUG: config type = {type(self.config)}")
        print(f"DEBUG: config keys = {list(self.config.keys())}")
        
        # 检查data_path配置
        data_path = self.config.get('data_path', None)
        print(f"DEBUG: data_path from config = {data_path}")
        print(f"DEBUG: data_path repr: {repr(data_path)}")
        
        # 如果配置中有data_path且是h5文件，直接使用
        if data_path and (data_path.endswith('.h5') or data_path.endswith('.hdf5')):
            print(f"DEBUG: Using configured data_path: {data_path}")
            is_h5_file = True
            print(f"DEBUG: is_h5_file: {is_h5_file}")
            
            # 检查文件是否存在
            if os.path.exists(data_path):
                print(f"DEBUG: File exists: {data_path}")
                # 直接使用配置的路径
                train_path = data_path
                val_path = data_path
                test_path = data_path
                splits_dir = None  # 单文件模式不需要splits_dir
            else:
                print(f"DEBUG: File does not exist: {data_path}")
                raise FileNotFoundError(f"Data file not found: {data_path}")
        else:
            print(f"DEBUG: data_path ends with .h5: {data_path.endswith('.h5') if data_path else False}")
            print(f"DEBUG: data_path ends with .hdf5: {data_path.endswith('.hdf5') if data_path else False}")
            is_h5_file = False
            print(f"DEBUG: is_h5_file: {is_h5_file}")
            
            # 使用目录结构
            data_dir = self.config.get('data_dir', 'data/pdebench')
            train_path = os.path.join(data_dir, 'DarcyFlow', '2D_DarcyFlow_beta1.0_Train.hdf5')
            val_path = os.path.join(data_dir, 'DarcyFlow', '2D_DarcyFlow_beta1.0_Valid.hdf5')
            test_path = os.path.join(data_dir, 'DarcyFlow', '2D_DarcyFlow_beta1.0_Test.hdf5')
            splits_dir = "splits"  # 多文件模式使用splits目录
            print(f"DEBUG: Using directory structure: {train_path}")
        
        # 训练集
        self.train_dataset = self.dataset_class(
            data_path=train_path,
            keys=self.config.get('keys', ['tensor']),  # 使用get方法提供默认值
            split="train",
            splits_dir=splits_dir,
            use_official_format=getattr(self.config, 'use_official_format', False),
            **self.common_kwargs,
            **self.dataset_kwargs
        )
        
        # 验证集
        self.val_dataset = self.dataset_class(
            data_path=val_path,
            keys=self.config.get('keys', ['tensor']),  # 使用get方法提供默认值
            split="val",
            splits_dir=splits_dir,
            use_official_format=getattr(self.config, 'use_official_format', False),
            **self.common_kwargs,
            **self.dataset_kwargs
        )
        
        # 测试集
        self.test_dataset = self.dataset_class(
            data_path=test_path,
            keys=self.config.get('keys', ['tensor']),
            split="test",
            splits_dir=splits_dir,
            use_official_format=getattr(self.config, 'use_official_format', False),
            **self.common_kwargs,
            **self.dataset_kwargs
        )
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            shuffle=True,
            num_workers=0,  # 由于h5py对象不能被pickle，设置为0
            pin_memory=False,  # 关闭pin_memory，避免设备不匹配
            persistent_workers=False,  # 关闭持久化worker
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            shuffle=False,
            num_workers=0,  # 由于h5py对象不能被pickle，设置为0
            pin_memory=False,  # 关闭pin_memory，避免设备不匹配
            persistent_workers=False,  # 关闭持久化worker
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # 测试时使用batch_size=1
            shuffle=False,
            num_workers=0,  # 测试时不使用多进程
            pin_memory=False,
        )
    
    def get_norm_stats(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取归一化统计量"""
        if hasattr(self, 'train_dataset') and self.train_dataset.norm_stats:
            # 转换为训练脚本期望的格式
            norm_stats = self.train_dataset.norm_stats
            # 假设只有一个变量，取第一个变量的统计量
            first_key = self.train_dataset.keys[0]
            return {
                'mean': norm_stats[f"{first_key}_mean"],
                'std': norm_stats[f"{first_key}_std"]
            }
        return None