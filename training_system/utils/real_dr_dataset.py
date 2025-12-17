#!/usr/bin/env python3
"""
真实扩散-反应数据模块 - 适配training_system
基于PDEBench 2D Diffusion-Reaction数据集
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from omegaconf import DictConfig
import torch.nn.functional as F

# 统一使用项目内的退化算子实现，保证训练DC与观测H一致
try:
    from ops.degradation import apply_degradation_operator
except Exception:
    apply_degradation_operator = None  # 允许在极简环境下导入失败，代码路径兼容

logger = logging.getLogger(__name__)


class RealDiffusionReactionDataset(Dataset):
    """真实扩散-反应数据集"""
    
    def __init__(
        self,
        data_path: str,
        keys: List[str],
        split: str = "train",
        img_size: int = 128,
        channels: int = 2,
        T_in: int = 1,
        T_out: int = 20,
        time_step_start: int = 0,
        time_step_end: int = 100,
        time_step_stride: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        augmentation: Optional[Dict] = None,
        observation_config: Optional[Dict] = None,
        cache_data: bool = False
    ):
        self.data_path = data_path
        self.keys = keys
        self.split = split
        self.img_size = img_size
        self.channels = channels
        self.T_in = T_in
        self.T_out = T_out
        self.time_step_start = time_step_start
        self.time_step_end = time_step_end
        self.time_step_stride = time_step_stride
        self.normalize = normalize
        self.augmentation = augmentation or {}
        self.observation_config = observation_config
        self.cache_data = cache_data
        
        # 数据切分比例
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 缓存
        self._cache = {} if cache_data else None
        self._normalization_stats = None
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载数据"""
        logger.info(f"加载真实扩散-反应数据: {self.data_path}")
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                self.all_data = []
                self.sample_indices = []
                
                # 遍历所有样本
                for key in self.keys:
                    if key in f:
                        sample_group = f[key]
                        
                        # 查找数据
                        data = None
                        for data_key in sample_group.keys():
                            item = sample_group[data_key]
                            if hasattr(item, 'shape') and len(item.shape) >= 3:
                                data = item[:]
                                logger.info(f"样本 {key} 数据形状: {data.shape}")
                                break
                        
                        if data is not None:
                            # 确保数据是4D格式 [T, H, W, C]
                            if len(data.shape) == 3:
                                data = data[..., np.newaxis]
                            
                            # 调整通道数
                            if data.shape[-1] != self.channels:
                                if data.shape[-1] == 1 and self.channels == 2:
                                    # 复制单通道到双通道
                                    data = np.repeat(data, 2, axis=-1)
                                elif data.shape[-1] > self.channels:
                                    # 取前几个通道
                                    data = data[..., :self.channels]
                                else:
                                    # 填充通道
                                    padding = np.zeros((data.shape[0], data.shape[1], data.shape[2], 
                                                      self.channels - data.shape[-1]))
                                    data = np.concatenate([data, padding], axis=-1)
                            
                            self.all_data.append(data)
                            
                            # 创建样本索引
                            num_time_steps = min(data.shape[0], self.time_step_end) - self.T_in - self.T_out + 1
                            for t in range(self.time_step_start, num_time_steps, self.time_step_stride):
                                self.sample_indices.append({
                                    'sample_idx': len(self.all_data) - 1,
                                    'time_idx': t,
                                    'key': key
                                })
                
                logger.info(f"加载完成: {len(self.all_data)} 个样本, {len(self.sample_indices)} 个序列")
                
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
        
        # 数据切分
        self._split_data()
        
        # 计算归一化统计
        if self.normalize:
            self._compute_normalization_stats()
    
    def _split_data(self):
        """数据切分"""
        total_samples = len(self.sample_indices)
        train_end = int(total_samples * self.train_ratio)
        val_end = int(total_samples * (self.train_ratio + self.val_ratio))
        
        if self.split == "train":
            self.sample_indices = self.sample_indices[:train_end]
        elif self.split == "val":
            self.sample_indices = self.sample_indices[train_end:val_end]
        elif self.split == "test":
            self.sample_indices = self.sample_indices[val_end:]
        
        logger.info(f"{self.split} 集: {len(self.sample_indices)} 个序列")
    
    def _compute_normalization_stats(self):
        """计算归一化统计"""
        if self._normalization_stats is not None:
            return
            
        logger.info("计算归一化统计...")
        
        all_data = []
        for sample_data in self.all_data:
            all_data.append(sample_data)
        
        if all_data:
            all_data = np.concatenate(all_data, axis=0)
            mean_np = np.mean(all_data, axis=(0, 1, 2))
            std_np = np.std(all_data, axis=(0, 1, 2))
            self._normalization_stats = {
                'mean': mean_np,
                'std': std_np
            }
            # 兼容训练器读取：提供 Tensor 形式的 mean/std 属性
            try:
                self.mean = torch.tensor(mean_np, dtype=torch.float32)
                self.std = torch.tensor(std_np, dtype=torch.float32)
                self.n_channels = int(self.channels)
            except Exception:
                pass
            logger.info(f"归一化统计 - 均值: {mean_np}, 标准差: {std_np}")
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化数据"""
        if not self.normalize or self._normalization_stats is None:
            return data
        
        mean = self._normalization_stats['mean']
        std = self._normalization_stats['std']
        
        # 避免除零
        std = np.where(std == 0, 1, std)
        
        return (data - mean) / std
    
    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """反归一化数据"""
        if not self.normalize or self._normalization_stats is None:
            return data
        
        mean = self._normalization_stats['mean']
        std = self._normalization_stats['std']
        
        return data * std + mean
    
    def _apply_augmentation(self, data: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        if not self.augmentation.get('enabled', False):
            return data
        
        # 随机翻转
        if np.random.random() < self.augmentation.get('flip_prob', 0.5):
            data = np.flip(data, axis=-2)  # 垂直翻转
        
        if np.random.random() < self.augmentation.get('flip_prob', 0.5):
            data = np.flip(data, axis=-1)  # 水平翻转
        
        # 随机旋转90度
        if np.random.random() < self.augmentation.get('rotate_prob', 0.3):
            k = np.random.randint(1, 4)  # 1-3次90度旋转
            data = np.rot90(data, k=k, axes=(-2, -1))
        
        # 添加噪声
        if self.augmentation.get('noise_std', 0) > 0:
            noise = np.random.normal(0, self.augmentation['noise_std'], data.shape)
            data = data + noise
        
        return data
    
    def _generate_observation(self, data: np.ndarray) -> np.ndarray:
        """生成观测数据（SR/Crop），与训练DC复用同一H实现。

        返回与输入同尺寸的观测baseline（LR上采样回原尺寸），以兼容现有训练管线。
        若需要LR观测，可在 __getitem__ 中同时返回 'observed_lr_sequence'。
        """
        if self.observation_config is None:
            return data

        obs = self.observation_config or {}
        # 支持嵌套 sr 配置块，统一展开到顶层
        sr_block = obs.get('sr') if isinstance(obs.get('sr'), dict) else None
        if sr_block:
            obs = {**obs, **{k: v for k, v in sr_block.items() if k not in obs}}
        mode = str(obs.get('mode', obs.get('observation_mode', 'SR'))).lower()

        # 构造H参数，统一别名
        h_params = None
        if mode in ['sr', 'super_resolution', 'SR']:
            scale = obs.get('scale', obs.get('scale_factor', obs.get('sr_scale', 1)))
            sigma = obs.get('sigma', obs.get('blur_sigma', 1.0))
            kernel = obs.get('kernel_size', obs.get('blur_kernel_size', 5))
            boundary = obs.get('boundary', obs.get('boundary_mode', 'mirror'))
            h_params = {
                'task': 'SR',
                'scale': int(scale),
                'sigma': float(sigma),
                'kernel_size': int(kernel),
                'boundary': str(boundary),
            }
        elif mode in ['crop', 'cropping', 'crop_reconstruction']:
            crop_ratio = obs.get('crop_ratio', None)
            crop_size = obs.get('crop_size', None)
            boundary = obs.get('boundary', obs.get('boundary_mode', 'mirror'))
            # 若给定比例，则转为尺寸（中心裁剪）
            h, w = data.shape[-3:-1]
            if crop_size is None and crop_ratio is not None:
                crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
                crop_size = (crop_h, crop_w)
            h_params = {
                'task': 'Crop',
                'crop_size': crop_size,
                'boundary': str(boundary),
            }
        else:
            # 未知模式，直接返回原数据
            return data

        # 使用统一H算子：blur + INTER_AREA downsample（SR），或中心裁剪（Crop）
        # 输入 data: [T, H, W, C]
        T, H, W, C = data.shape
        # 转为张量 [T, C, H, W]
        t = torch.from_numpy(data).float().permute(0, 3, 1, 2).contiguous()
        if apply_degradation_operator is None:
            # 兜底：无ops实现时保持原数据
            degraded = t
        else:
            # 将时间维作为batch，一次性应用H
            degraded = apply_degradation_operator(t, h_params)  # [T, C, H', W']

        # 上采样回原尺寸作为baseline（与现有管线对齐）
        degraded_up = degraded
        if degraded.shape[-2:] != (H, W):
            degraded_up = F.interpolate(degraded, size=(H, W), mode='bilinear', align_corners=False)

        # 转回 [T, H, W, C]
        data_up = degraded_up.permute(0, 2, 3, 1).cpu().numpy()
        return data_up
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        
        # 获取样本索引信息
        sample_info = self.sample_indices[idx]
        sample_idx = sample_info['sample_idx']
        time_idx = sample_info['time_idx']
        
        # 获取数据
        sample_data = self.all_data[sample_idx]
        
        # 提取输入序列和目标序列
        input_seq = sample_data[time_idx:time_idx + self.T_in]
        target_seq = sample_data[time_idx + self.T_in:time_idx + self.T_in + self.T_out]
        
        # 调整图像大小
        if self.img_size != input_seq.shape[-2]:
            # 简单的插值调整大小
            from scipy.ndimage import zoom
            scale_factor = self.img_size / input_seq.shape[-2]
            input_seq = zoom(input_seq, (1, scale_factor, scale_factor, 1), order=1)
            target_seq = zoom(target_seq, (1, scale_factor, scale_factor, 1), order=1)
        
        # 观测生成应在原值域进行
        obs_seq_raw = self._generate_observation(sample_data[time_idx:time_idx + self.T_in])
        # 归一化（z-score）
        input_seq = self._normalize(input_seq)
        target_seq = self._normalize(target_seq)
        obs_seq = self._normalize(obs_seq_raw)
        
        # 应用数据增强（保持在归一化域）
        input_seq = self._apply_augmentation(input_seq)
        target_seq = self._apply_augmentation(target_seq)
        obs_seq = self._apply_augmentation(obs_seq)
        observed_lr_tensor = None
        if self.observation_config is not None and apply_degradation_operator is not None:
            # 构造与 _generate_observation 同步的参数
            obs = self.observation_config or {}
            mode = str(obs.get('mode', obs.get('observation_mode', 'SR'))).lower()
            h_params = None
            if mode in ['sr', 'super_resolution', 'SR']:
                scale = obs.get('scale', obs.get('scale_factor', obs.get('sr_scale', 1)))
                sigma = obs.get('sigma', obs.get('blur_sigma', 1.0))
                kernel = obs.get('kernel_size', obs.get('blur_kernel_size', 5))
                boundary = obs.get('boundary', obs.get('boundary_mode', 'mirror'))
                h_params = {
                    'task': 'SR',
                    'scale': int(scale),
                    'sigma': float(sigma),
                    'kernel_size': int(kernel),
                    'boundary': str(boundary),
                }
            elif mode in ['crop', 'cropping', 'crop_reconstruction']:
                crop_ratio = obs.get('crop_ratio', None)
                crop_size = obs.get('crop_size', None)
                boundary = obs.get('boundary', obs.get('boundary_mode', 'mirror'))
                h, w = input_seq.shape[-3:-1]
                if crop_size is None and crop_ratio is not None:
                    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
                    crop_size = (crop_h, crop_w)
                h_params = {
                    'task': 'Crop',
                    'crop_size': crop_size,
                    'boundary': str(boundary),
                }

            if h_params is not None:
                # 在原值域生成LR观测张量
                t_in = torch.from_numpy(sample_data[time_idx:time_idx + self.T_in]).float().permute(0, 3, 1, 2)
                degraded_lr = apply_degradation_operator(t_in, h_params)  # [T, C, h', w']
                observed_lr_tensor = degraded_lr  # 在结果中以张量形式返回
        
        # 转换为PyTorch张量
        input_tensor = torch.FloatTensor(input_seq).permute(0, 3, 1, 2)  # [T, C, H, W]
        target_tensor = torch.FloatTensor(target_seq).permute(0, 3, 1, 2)
        obs_tensor = torch.FloatTensor(obs_seq).permute(0, 3, 1, 2)
        Tn, Hn, Wn = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        # 像素中心坐标（align_corners=False）：x=2*(j+0.5)/W-1, y=2*(i+0.5)/H-1
        j_idx = np.arange(Wn, dtype=np.float32)
        i_idx = np.arange(Hn, dtype=np.float32)
        x_centers = (2.0 * (j_idx + 0.5) / float(Wn)) - 1.0
        y_centers = (2.0 * (i_idx + 0.5) / float(Hn)) - 1.0
        Xg, Yg = np.meshgrid(x_centers, y_centers)
        coords_hr = np.stack([Xg, Yg], axis=-1)
        coords_hr = np.repeat(coords_hr[np.newaxis, ...], Tn, axis=0)
        coords_hr_tensor = torch.FloatTensor(coords_hr).permute(0, 3, 1, 2)
        # fourier positional encoding（按配置开关）
        pe_hr_tensor = None
        try:
            include_pe = bool(getattr(self.config.data, 'include_fourier_pe', False))
            bands = int(getattr(self.config.data, 'fourier_pe_bands', 4))
        except Exception:
            include_pe, bands = False, 0
        if include_pe and bands > 0:
            xs = torch.from_numpy(Xg).float()
            ys = torch.from_numpy(Yg).float()
            pes = []
            for k in range(bands):
                f = float(2 ** k * np.pi)
                pes.append(torch.sin(f * xs))
                pes.append(torch.cos(f * xs))
                pes.append(torch.sin(f * ys))
                pes.append(torch.cos(f * ys))
            pe_stack = torch.stack(pes, dim=0)  # [P, H, W]
            pe_stack = pe_stack.unsqueeze(0).repeat(Tn, 1, 1, 1)  # [T, P, H, W]
            pe_hr_tensor = pe_stack
        mask_hr = np.zeros((Tn, 1, Hn, Wn), dtype=np.float32)
        if observed_lr_tensor is not None:
            h_lr = int(observed_lr_tensor.shape[-2])
            w_lr = int(observed_lr_tensor.shape[-1])
            mask = np.zeros((1, Hn, Wn), dtype=np.float32)
            for i in range(h_lr):
                ih0 = int(np.round(i * Hn / float(h_lr)))
                ih1 = int(np.round((i + 1) * Hn / float(h_lr)))
                ih0 = max(0, min(Hn, ih0))
                ih1 = max(ih0 + 1, min(Hn, ih1))
                for j in range(w_lr):
                    jw0 = int(np.round(j * Wn / float(w_lr)))
                    jw1 = int(np.round((j + 1) * Wn / float(w_lr)))
                    jw0 = max(0, min(Wn, jw0))
                    jw1 = max(jw0 + 1, min(Wn, jw1))
                    mask[0, ih0:ih1, jw0:jw1] = 1.0
            mask_hr = np.repeat(mask[np.newaxis, ...], Tn, axis=0)
        mask_hr_tensor = torch.FloatTensor(mask_hr)
        coords_lr_tensor = None
        mask_lr_tensor = None
        if observed_lr_tensor is not None:
            Tl, Cl, hl, wl = observed_lr_tensor.shape
            j_idx_lr = np.arange(wl, dtype=np.float32)
            i_idx_lr = np.arange(hl, dtype=np.float32)
            x_centers_lr = (2.0 * (j_idx_lr + 0.5) / float(wl)) - 1.0
            y_centers_lr = (2.0 * (i_idx_lr + 0.5) / float(hl)) - 1.0
            Xg_lr, Yg_lr = np.meshgrid(x_centers_lr, y_centers_lr)
            coords_lr = np.stack([Xg_lr, Yg_lr], axis=-1)
            coords_lr = np.repeat(coords_lr[np.newaxis, ...], Tl, axis=0)
            coords_lr_tensor = torch.FloatTensor(coords_lr).permute(0, 3, 1, 2)
            ones_lr = torch.ones((Tl, 1, hl, wl), dtype=torch.float32)
            mask_lr_tensor = ones_lr
        
        result = {
            'input_sequence': input_tensor,  # [T_in, C, H, W]
            'target_sequence': target_tensor,  # [T_out, C, H, W]
            'observed_sequence': obs_tensor,  # [T_in, C, H, W]
            'coords_sequence': coords_hr_tensor,
            'mask_sequence': mask_hr_tensor,
            'sample_idx': sample_idx,
            'start_time': time_idx,
            'key': sample_info['key']
        }
        if pe_hr_tensor is not None:
            result['fourier_pe_sequence'] = pe_hr_tensor

        # 可选：返回LR观测以供数据一致性损失使用
        if observed_lr_tensor is not None:
            result['observed_lr_sequence'] = observed_lr_tensor  # [T_in, C, h', w']
            if coords_lr_tensor is not None:
                result['coords_lr_sequence'] = coords_lr_tensor
            if mask_lr_tensor is not None:
                result['mask_lr_sequence'] = mask_lr_tensor
        
        if self._cache is not None:
            self._cache[idx] = result
        
        return result


class RealDiffusionReactionDataModule:
    """真实扩散-反应数据模块"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 统一的DataLoader参数获取（兼容多种配置位置）
        def _get_cfg(paths, default=None):
            cfg = self.config
            for path in paths:
                try:
                    cur = cfg
                    for key in path:
                        if isinstance(cur, dict):
                            cur = cur.get(key)
                        else:
                            # 兼容DictConfig/对象属性访问
                            cur = getattr(cur, key, None) if hasattr(cur, key) else cur.get(key, None)
                        if cur is None:
                            break
                    if cur is not None:
                        return cur
                except Exception:
                    continue
            return default
        
        # 提前解析DataLoader相关配置并缓存
        self._batch_size = _get_cfg([
            ['training', 'batch_size'],
            ['data', 'dataloader', 'batch_size'],
            ['dataloader', 'batch_size'],
        ], default=1)
        
        self._num_workers = _get_cfg([
            ['data', 'dataloader', 'num_workers'],
            ['hardware', 'num_workers'],
            ['dataloader', 'num_workers'],
        ], default=0)
        
        self._pin_memory = bool(_get_cfg([
            ['data', 'dataloader', 'pin_memory'],
            ['hardware', 'pin_memory'],
            ['dataloader', 'pin_memory'],
        ], default=False))
        
        # persistent_workers 仅在 num_workers>0 时启用，避免PyTorch报错
        _persistent_workers_cfg = bool(_get_cfg([
            ['data', 'dataloader', 'persistent_workers'],
            ['hardware', 'persistent_workers'],
            ['dataloader', 'persistent_workers'],
        ], default=False))
        self._persistent_workers = _persistent_workers_cfg and (self._num_workers and self._num_workers > 0)
        
    def setup(self, stage: str = None):
        data_config = self.config.data
        try:
            data_keys = data_config.get('keys') if hasattr(data_config, 'get') else getattr(data_config, 'keys', [])
        except Exception:
            data_keys = []
        splits_dir = getattr(data_config, 'splits_dir', None)
        train_keys_from_splits = None
        val_keys_from_splits = None
        test_keys_from_splits = None
        if splits_dir:
            base = Path(str(splits_dir))
            train_file = base / 'train.txt'
            val_file = base / 'val.txt'
            test_file = base / 'test.txt'
            def _read_split(fp):
                with open(fp, 'r') as f:
                    return [ln.strip() for ln in f if ln.strip()]
            if not (train_file.exists() and val_file.exists() and test_file.exists()):
                raise FileNotFoundError(
                    f"splits_dir={splits_dir} 下缺少 train/val/test.txt，"
                    f"请按照项目规范提供固定划分文件"
                )
            train_keys_from_splits = _read_split(train_file)
            val_keys_from_splits = _read_split(val_file)
            test_keys_from_splits = _read_split(test_file)
        
        # 根据是否启用AR/时序预测，强制纯空间T_out=1
        t_out_eff = getattr(data_config, 'T_out', 1)
        try:
            ar_cfg = getattr(self.config, 'ar', None)
            model_name = str(getattr(self.config.model, 'name', '')).lower()
            # 若显式关闭AR，或模型为纯空间（SwinUNet等），则覆盖为1
            if ar_cfg is not None and hasattr(ar_cfg, 'enabled') and (not getattr(ar_cfg, 'enabled')):
                t_out_eff = 1
            elif model_name in ['swinunet', 'unet', 'segformer', 'unetformer']:
                t_out_eff = 1
        except Exception:
            pass

        # 兼容字段别名与默认值
        img_size = getattr(data_config, 'img_size', getattr(data_config, 'image_size', 128))
        channels = getattr(data_config, 'channels', getattr(data_config, 'input_channels', 2))

        # 规范化与缓存配置（兼容两种写法）
        try:
            preprocessing_cfg = getattr(data_config, 'preprocessing', None)
        except Exception:
            preprocessing_cfg = None
        normalize_flag = True
        try:
            if hasattr(data_config, 'normalize'):
                normalize_flag = bool(getattr(data_config, 'normalize'))
            elif preprocessing_cfg is not None and hasattr(preprocessing_cfg, 'normalize'):
                normalize_flag = bool(getattr(preprocessing_cfg, 'normalize'))
        except Exception:
            normalize_flag = True
        cache_flag = False
        try:
            if preprocessing_cfg is not None and hasattr(preprocessing_cfg, 'cache_data'):
                cache_flag = bool(getattr(preprocessing_cfg, 'cache_data'))
        except Exception:
            cache_flag = False

        # 构建稳定的split：必须使用splits目录提供的固定划分
        if not (train_keys_from_splits and val_keys_from_splits and test_keys_from_splits):
            raise RuntimeError(
                "RealDiffusionReactionDataModule 需要固定的 train/val/test 划分文件。\n"
                "请在 data.splits_dir 下提供 train.txt、val.txt、test.txt，"
                "每行一个样本键（例如 0000, 0001, ...）。"
            )
        train_keys = train_keys_from_splits if (train_keys_from_splits and len(train_keys_from_splits) > 0) else data_keys
        self.train_dataset = RealDiffusionReactionDataset(
            data_path=data_config.data_path,
            keys=train_keys,
            split='train',
            img_size=img_size,
            channels=channels,
            T_in=data_config.T_in,
            T_out=t_out_eff,
            time_step_start=data_config.time_step_start,
            time_step_end=data_config.time_step_end,
            time_step_stride=data_config.time_step_stride,
            train_ratio=float(getattr(data_config, 'train_ratio', 0.8)),
            val_ratio=float(getattr(data_config, 'val_ratio', 0.1)),
            test_ratio=float(getattr(data_config, 'test_ratio', 0.1)),
            normalize=normalize_flag,
            augmentation=data_config.get('augmentation', {}),
            observation_config=data_config.get('observation'),
            cache_data=cache_flag
        )
        
        val_keys = val_keys_from_splits if (val_keys_from_splits and len(val_keys_from_splits) > 0) else data_keys
        self.val_dataset = RealDiffusionReactionDataset(
            data_path=data_config.data_path,
            keys=val_keys,
            split='val',
            img_size=img_size,
            channels=channels,
            T_in=data_config.T_in,
            T_out=t_out_eff,
            time_step_start=data_config.time_step_start,
            time_step_end=data_config.time_step_end,
            time_step_stride=data_config.time_step_stride,
            train_ratio=float(getattr(data_config, 'train_ratio', 0.8)),
            val_ratio=float(getattr(data_config, 'val_ratio', 0.1)),
            test_ratio=float(getattr(data_config, 'test_ratio', 0.1)),
            normalize=normalize_flag,
            augmentation={'enabled': False},  # 验证集不增强
            observation_config=data_config.get('observation'),
            cache_data=cache_flag
        )
        
        test_keys = test_keys_from_splits if (test_keys_from_splits and len(test_keys_from_splits) > 0) else data_keys
        self.test_dataset = RealDiffusionReactionDataset(
            data_path=data_config.data_path,
            keys=test_keys,
            split='test',
            img_size=img_size,
            channels=channels,
            T_in=data_config.T_in,
            T_out=t_out_eff,
            time_step_start=data_config.time_step_start,
            time_step_end=data_config.time_step_end,
            time_step_stride=data_config.time_step_stride,
            train_ratio=float(getattr(data_config, 'train_ratio', 0.8)),
            val_ratio=float(getattr(data_config, 'val_ratio', 0.1)),
            test_ratio=float(getattr(data_config, 'test_ratio', 0.1)),
            normalize=normalize_flag,
            augmentation={'enabled': False},  # 测试集不增强
            observation_config=data_config.get('observation'),
            cache_data=cache_flag
        )
        
        logger.info(f"数据模块设置完成:")
        logger.info(f"  训练集: {len(self.train_dataset)} 个样本")
        logger.info(f"  验证集: {len(self.val_dataset)} 个样本")
        logger.info(f"  测试集: {len(self.test_dataset)} 个样本")

    @property
    def norm_stats(self):
        """提供归一化统计，供损失与评估在原值域进行反归一化时使用"""
        try:
            train_ds = getattr(self, "train_dataset", None)
            if train_ds is None:
                return None
            mean = getattr(train_ds, "mean", None)
            std = getattr(train_ds, "std", None)
            if mean is None or std is None:
                ns = getattr(train_ds, "_normalization_stats", None)
                if ns is None:
                    return None
                mean = ns.get("mean", None)
                std = ns.get("std", None)
            if mean is None or std is None:
                return None
            mean_t = torch.as_tensor(mean, dtype=torch.float32)
            std_t = torch.as_tensor(std, dtype=torch.float32)
            return {
                "data_mean": mean_t,
                "data_std": std_t,
            }
        except Exception:
            return None
    
    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=self._collate_fn
        )

    def _ensure_tc_hw(self, t: torch.Tensor) -> torch.Tensor:
        """确保张量形状为 [T, C, H, W]。若为 [T, H, W, C] 则重排。
        通过维度大小启发式判断通道维（通常C≤8，H/W≥16）。"""
        if t.dim() != 4:
            return t
        # [T,H,W,C] 情况：最后一维可能是通道且较小
        if t.shape[-1] <= 8 and t.shape[1] >= 16 and t.shape[2] >= 16:
            return t.permute(0, 3, 1, 2).contiguous()
        return t

    def _collate_fn(self, batch):
        """安全的collate函数，统一样本张量形状后再堆叠。"""
        keys = batch[0].keys()
        out = {}
        for k in keys:
            items = [b[k] for b in batch]
            if isinstance(items[0], torch.Tensor):
                # 对序列张量进行形状统一
                if k in ("input_sequence", "target_sequence", "observed_sequence"):
                    items = [self._ensure_tc_hw(x) for x in items]
                out[k] = torch.stack(items, dim=0)
            else:
                out[k] = items
        return out
