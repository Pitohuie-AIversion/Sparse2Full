"""DarcyFlow真实数据集处理模块

专门处理真实的Darcy Flow HDF5数据，支持官方PDEBench格式。
确保与现有训练系统完全兼容。
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
from omegaconf import DictConfig

from ops.degradation import apply_degradation_operator


class DarcyFlowDataset(Dataset):
    """Darcy Flow真实数据集
    
    专门处理真实的2D_DarcyFlow_beta*.hdf5数据文件
    支持官方PDEBench格式：[batch, time, height, width, channels]
    """
    
    def __init__(
        self,
        data_path: str,
        keys: List[str] = ["u"],  # Darcy Flow通常只有一个通道
        split: str = "train",
        splits_dir: Optional[str] = None,
        normalize: bool = True,
        image_size: int = 256,
        task: str = "SR",  # "SR" 或 "Crop"
        scale: int = 4,  # SR模式的下采样倍率
        crop_size: Tuple[int, int] = (128, 128),  # Crop模式的裁剪尺寸
        sigma: float = 1.0,  # 高斯模糊标准差
        blur_kernel: int = 5,  # 模糊核大小
        boundary: str = "mirror",  # 边界策略
        patch_align: int = 8,  # patch对齐倍数
        center_sampler: str = "mixed",  # 中心采样策略
        noise_std: float = 0.0,  # 噪声标准差
    ):
        """初始化Darcy Flow数据集
        
        Args:
            data_path: HDF5文件路径
            keys: 数据键名列表，默认["u"]
            split: 数据切分，"train"/"val"/"test"
            splits_dir: 数据切分文件目录
            normalize: 是否进行z-score归一化
            image_size: 图像尺寸
            task: 任务类型 "SR"/"Crop"
            scale: SR下采样倍率
            crop_size: Crop裁剪尺寸
            sigma: 高斯模糊标准差
            blur_kernel: 模糊核大小
            boundary: 边界策略
            patch_align: patch对齐倍数
            center_sampler: 中心采样策略
            noise_std: 噪声标准差
        """
        self.data_path = data_path
        self.keys = keys
        self.split = split
        self.normalize = normalize
        self.image_size = image_size
        self.task = task
        self.scale = scale
        self.crop_size = crop_size
        self.sigma = sigma
        self.blur_kernel = blur_kernel
        self.boundary = boundary
        self.patch_align = patch_align
        self.center_sampler = center_sampler
        self.noise_std = noise_std
        
        # 打开HDF5文件
        self.h5_file = h5py.File(data_path, 'r')
        
        # 检查数据格式并获取数据信息
        self._analyze_data_format()
        
        # 加载数据切分
        self.case_ids = self._load_split_ids(splits_dir, split)
        
        # 加载归一化统计量
        self.norm_stats = self._load_norm_stats(splits_dir) if normalize else None
        
        # 设置H算子参数
        if task == "SR":
            self.h_params = {
                'task': 'SR',
                'scale': scale,
                'sigma': sigma,
                'kernel_size': blur_kernel,
                'boundary': boundary,
            }
        else:  # Crop
            self.h_params = {
                'task': 'Crop',
                'crop_size': crop_size,
                'patch_align': patch_align,
                'boundary': boundary,
            }
        
        print(f"DarcyFlow Dataset initialized:")
        print(f"  - Data path: {data_path}")
        print(f"  - Data format: {self.data_format}")
        print(f"  - Data shape: {self.data_shape}")
        print(f"  - Keys: {keys}")
        print(f"  - Split: {split} ({len(self.case_ids)} samples)")
        print(f"  - Task: {task}")
        print(f"  - Normalize: {normalize}")
    
    def _analyze_data_format(self):
        """分析数据格式"""
        # 检查可能的数据键
        possible_keys = ['data', 'u', 'velocity', 'pressure', 'solution']
        self.data_key = None
        
        for key in possible_keys:
            if key in self.h5_file:
                self.data_key = key
                break
        
        if self.data_key is None:
            # 列出所有可用的键
            available_keys = list(self.h5_file.keys())
            raise ValueError(f"No recognized data key found. Available keys: {available_keys}")
        
        # 获取数据形状
        self.data_shape = self.h5_file[self.data_key].shape
        
        # 判断数据格式
        if len(self.data_shape) == 5:
            # 5D数据：[batch, time, height, width, channels] (官方格式)
            self.data_format = "official"
            self.n_batch, self.n_time, self.height, self.width, self.n_channels = self.data_shape
        elif len(self.data_shape) == 4:
            # 4D数据：可能是 [time, channels, height, width] 或 [batch, height, width, channels]
            if self.data_shape[1] <= 10:  # 假设通道数不会超过10
                self.data_format = "tchw"  # [time, channels, height, width]
                self.n_time, self.n_channels, self.height, self.width = self.data_shape
                self.n_batch = 1
            else:
                self.data_format = "bhwc"  # [batch, height, width, channels]
                self.n_batch, self.height, self.width, self.n_channels = self.data_shape
                self.n_time = 1
        elif len(self.data_shape) == 3:
            # 3D数据：[time, height, width] (单通道)
            self.data_format = "thw"
            self.n_time, self.height, self.width = self.data_shape
            self.n_channels = 1
            self.n_batch = 1
        else:
            raise ValueError(f"Unsupported data shape: {self.data_shape}")
        
        print(f"Detected data format: {self.data_format}")
        print(f"Data shape: {self.data_shape}")
        print(f"Batch: {self.n_batch}, Time: {self.n_time}, H: {self.height}, W: {self.width}, C: {self.n_channels}")
    
    def _load_split_ids(self, splits_dir: Optional[str], split: str) -> List[str]:
        """加载数据切分ID列表"""
        if splits_dir is None:
            # 如果没有指定切分目录，使用默认切分
            if self.data_format == "official":
                n_total = self.n_time  # 使用时间步数
            elif self.data_format == "tchw":
                n_total = self.n_time
            elif self.data_format == "bhwc":
                n_total = self.n_batch
            else:  # thw
                n_total = self.n_time
            
            total_ids = [str(i) for i in range(n_total)]
            if split == "train":
                return total_ids[:int(0.8 * n_total)]
            elif split == "val":
                return total_ids[int(0.8 * n_total):int(0.9 * n_total)]
            else:  # test
                return total_ids[int(0.9 * n_total):]
        
        # 从文件加载切分
        data_root = Path(self.data_path).parent.parent
        split_file = data_root / splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            case_ids = [line.strip() for line in f if line.strip()]
        
        return case_ids
    
    def _load_norm_stats(self, splits_dir: Optional[str]) -> Dict[str, torch.Tensor]:
        """加载归一化统计量"""
        if splits_dir is None:
            return self._compute_norm_stats_auto()
        
        data_root = Path(self.data_path).parent.parent
        stats_file = data_root / splits_dir / "norm_stat.npz"
        if not stats_file.exists():
            return self._compute_norm_stats(stats_file)
        
        stats = np.load(stats_file)
        norm_stats = {}
        for key in self.keys:
            norm_stats[f"{key}_mean"] = torch.tensor(stats[f"{key}_mean"], dtype=torch.float32)
            norm_stats[f"{key}_std"] = torch.tensor(stats[f"{key}_std"], dtype=torch.float32)
        
        return norm_stats
    
    def _compute_norm_stats_auto(self) -> Dict[str, torch.Tensor]:
        """自动计算归一化统计量（不保存文件）"""
        print("Computing normalization statistics automatically...")
        
        # 使用训练集计算统计量
        train_ids = self._load_split_ids(None, "train")
        
        stats = {}
        for i, key in enumerate(self.keys):
            values = []
            for case_id in train_ids[:min(100, len(train_ids))]:  # 最多使用100个样本
                case_idx = int(case_id)
                data = self._load_raw_data(case_idx, i)
                values.append(data.flatten())
            
            if values:
                all_values = np.concatenate(values)
                mean = np.mean(all_values)
                std = np.std(all_values)
                
                stats[f"{key}_mean"] = torch.tensor(mean, dtype=torch.float32)
                stats[f"{key}_std"] = torch.tensor(std, dtype=torch.float32)
        
        return stats
    
    def _compute_norm_stats(self, save_path: Path) -> Dict[str, torch.Tensor]:
        """计算并保存归一化统计量"""
        print("Computing normalization statistics...")
        
        # 使用训练集计算统计量
        data_root = Path(self.data_path).parent.parent
        train_ids = self._load_split_ids(data_root / "splits", "train")
        
        stats = {}
        for i, key in enumerate(self.keys):
            values = []
            for case_id in train_ids:
                case_idx = int(case_id)
                data = self._load_raw_data(case_idx, i)
                values.append(data.flatten())
            
            if values:
                all_values = np.concatenate(values)
                mean = np.mean(all_values)
                std = np.std(all_values)
                
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
    
    def _load_raw_data(self, case_idx: int, channel_idx: int) -> np.ndarray:
        """加载原始数据"""
        if self.data_format == "official":
            # [batch, time, height, width, channels]
            if channel_idx < self.n_channels:
                data = self.h5_file[self.data_key][0, case_idx, :, :, channel_idx]
            else:
                data = self.h5_file[self.data_key][0, case_idx, :, :, 0]
        elif self.data_format == "tchw":
            # [time, channels, height, width]
            if channel_idx < self.n_channels:
                data = self.h5_file[self.data_key][case_idx, channel_idx, :, :]
            else:
                data = self.h5_file[self.data_key][case_idx, 0, :, :]
        elif self.data_format == "bhwc":
            # [batch, height, width, channels]
            if channel_idx < self.n_channels:
                data = self.h5_file[self.data_key][case_idx, :, :, channel_idx]
            else:
                data = self.h5_file[self.data_key][case_idx, :, :, 0]
        else:  # thw
            # [time, height, width]
            data = self.h5_file[self.data_key][case_idx, :, :]
        
        return data
    
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
        case_idx = int(case_id)
        
        # 读取多通道数据
        data_list = []
        for i, key in enumerate(self.keys):
            raw_data = self._load_raw_data(case_idx, i)
            channel_data = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
            
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
        
        # 生成观测数据
        obs_data = self._generate_observation(target)
        
        # 添加噪声
        if self.noise_std > 0:
            noise = torch.randn_like(obs_data['baseline']) * self.noise_std
            obs_data['baseline'] = obs_data['baseline'] + noise
        
        # 构建返回数据
        result = {
            'target': target,  # [C, H, W]
            'observation': obs_data['baseline'],  # [C, H, W]
            'baseline': obs_data['baseline'],
            'coords': obs_data['coords'],
            'mask': obs_data['mask'],
            'case_id': case_id,
            'h_params': self.h_params,
            'task_params': self.h_params,
        }
        
        return result
    
    def _generate_observation(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成观测数据"""
        if self.task == "SR":
            return self._generate_sr_observation(target)
        else:  # Crop
            return self._generate_crop_observation(target)
    
    def _generate_sr_observation(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成SR观测数据"""
        C, H, W = target.shape
        
        # 反归一化到原值域进行H操作
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
            'baseline': baseline,
            'coords': coords,
            'mask': mask,
        }
    
    def _generate_crop_observation(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成Crop观测数据"""
        C, H, W = target.shape
        
        # 生成裁剪框
        crop_h, crop_w = self.crop_size
        
        # 确保裁剪尺寸不超过图像尺寸
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)
        
        # 根据采样策略选择中心点
        if self.center_sampler == "uniform":
            # 均匀采样
            center_y = torch.randint(crop_h // 2, H - crop_h // 2, (1,)).item()
            center_x = torch.randint(crop_w // 2, W - crop_w // 2, (1,)).item()
        elif self.center_sampler == "boundary":
            # 边界采样
            if torch.rand(1) < 0.5:  # 选择边界
                if torch.rand(1) < 0.5:  # 上下边界
                    center_y = crop_h // 2 if torch.rand(1) < 0.5 else H - crop_h // 2
                    center_x = torch.randint(crop_w // 2, W - crop_w // 2, (1,)).item()
                else:  # 左右边界
                    center_y = torch.randint(crop_h // 2, H - crop_h // 2, (1,)).item()
                    center_x = crop_w // 2 if torch.rand(1) < 0.5 else W - crop_w // 2
            else:  # 均匀采样
                center_y = torch.randint(crop_h // 2, H - crop_h // 2, (1,)).item()
                center_x = torch.randint(crop_w // 2, W - crop_w // 2, (1,)).item()
        else:  # mixed
            # 混合采样：均匀40% + 边界30% + 高梯度30%
            rand_val = torch.rand(1).item()
            if rand_val < 0.4:  # 均匀采样
                center_y = torch.randint(crop_h // 2, H - crop_h // 2, (1,)).item()
                center_x = torch.randint(crop_w // 2, W - crop_w // 2, (1,)).item()
            elif rand_val < 0.7:  # 边界采样
                if torch.rand(1) < 0.5:  # 上下边界
                    center_y = crop_h // 2 if torch.rand(1) < 0.5 else H - crop_h // 2
                    center_x = torch.randint(crop_w // 2, W - crop_w // 2, (1,)).item()
                else:  # 左右边界
                    center_y = torch.randint(crop_h // 2, H - crop_h // 2, (1,)).item()
                    center_x = crop_w // 2 if torch.rand(1) < 0.5 else W - crop_w // 2
            else:  # 高梯度采样
                # 计算梯度幅度
                grad_x = torch.abs(target[:, :, 1:] - target[:, :, :-1])  # [C, H, W-1]
                grad_y = torch.abs(target[:, 1:, :] - target[:, :-1, :])  # [C, H-1, W]
                
                # 填充到相同尺寸
                grad_x = F.pad(grad_x, (0, 1), mode='replicate')  # [C, H, W]
                grad_y = F.pad(grad_y, (1, 0), mode='replicate')  # [C, H, W]
                
                # 计算总梯度
                grad_mag = torch.sqrt(grad_x**2 + grad_y**2).mean(dim=0)  # [H, W]
                
                # 平滑梯度图
                grad_smooth = F.avg_pool2d(
                    grad_mag.unsqueeze(0).unsqueeze(0), 
                    kernel_size=crop_h//4, 
                    stride=1, 
                    padding=crop_h//8
                ).squeeze()
                
                # 找到高梯度区域
                valid_y = torch.arange(crop_h // 2, H - crop_h // 2)
                valid_x = torch.arange(crop_w // 2, W - crop_w // 2)
                
                if len(valid_y) > 0 and len(valid_x) > 0:
                    grad_values = grad_smooth[valid_y][:, valid_x]
                    flat_idx = torch.multinomial(grad_values.flatten() + 1e-8, 1).item()
                    center_y = valid_y[flat_idx // len(valid_x)]
                    center_x = valid_x[flat_idx % len(valid_x)]
                else:
                    # 回退到均匀采样
                    center_y = H // 2
                    center_x = W // 2
        
        # 计算裁剪框边界
        y1 = max(0, center_y - crop_h // 2)
        y2 = min(H, y1 + crop_h)
        x1 = max(0, center_x - crop_w // 2)
        x2 = min(W, x1 + crop_w)
        
        # 调整边界确保尺寸正确
        if y2 - y1 < crop_h:
            if y1 == 0:
                y2 = min(H, crop_h)
            else:
                y1 = max(0, H - crop_h)
        
        if x2 - x1 < crop_w:
            if x1 == 0:
                x2 = min(W, crop_w)
            else:
                x1 = max(0, W - crop_w)
        
        # 创建baseline（全零，只有裁剪区域有值）
        baseline = torch.zeros_like(target)
        baseline[:, y1:y2, x1:x2] = target[:, y1:y2, x1:x2]
        
        # 生成mask
        mask = torch.zeros(1, H, W)
        mask[:, y1:y2, x1:x2] = 1.0
        
        # 生成坐标网格
        coords = self._generate_coords(H, W)  # [2, H, W]
        
        return {
            'baseline': baseline,
            'coords': coords,
            'mask': mask,
        }
    
    def _generate_coords(self, H: int, W: int) -> torch.Tensor:
        """生成坐标网格"""
        # 生成归一化坐标 [-1, 1]
        y_coords = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
        x_coords = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
        
        coords = torch.stack([x_coords, y_coords], dim=0)  # [2, H, W]
        
        return coords
    
    def close(self):
        """关闭HDF5文件"""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()