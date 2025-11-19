"""
PDEBench数据集模块 - 支持多种PDE数据加载和处理
遵循黄金法则，确保观测算子一致性
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import logging
import cv2
import cv2

logger = logging.getLogger(__name__)


class PDEBenchDataset(Dataset):
    """PDEBench标准数据集类"""
    
    def __init__(
        self,
        data_path: str,
        data_key: str = "data",
        mode: str = "train",
        img_size: int = 256,
        T_in: int = 10,
        T_out: int = 10,
        normalize: bool = True,
        augmentation: bool = False,
        observation_mode: str = "super_resolution",
        sr_scale: int = 4,
        crop_size: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        初始化PDEBench数据集
        
        Args:
            data_path: HDF5数据文件路径
            data_key: HDF5中的数据键名
            mode: 数据集模式 ('train', 'val', 'test')
            img_size: 图像尺寸
            T_in: 输入时间步数
            T_out: 输出时间步数
            normalize: 是否归一化
            augmentation: 是否启用数据增强
            observation_mode: 观测模式 ('super_resolution', 'crop', 'full')
            sr_scale: 超分辨率下采样比例
            crop_size: 裁剪尺寸
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.data_path = data_path
        self.data_key = data_key
        self.mode = mode
        self.img_size = img_size
        self.T_in = T_in
        self.T_out = T_out
        self.normalize = normalize
        self.augmentation = augmentation
        # 观测模式别名统一
        mode_alias = str(observation_mode).lower()
        if mode_alias in ["sr", "super_resolution", "superresolution"]:
            self.observation_mode = "sr"
        elif mode_alias in ["crop", "patch"]:
            self.observation_mode = "crop"
        elif mode_alias in ["full", "identity", "none"]:
            self.observation_mode = "full"
        else:
            logger.warning(f"未知观测模式'{observation_mode}'，默认使用'sr'")
            self.observation_mode = "sr"
        self.sr_scale = sr_scale
        self.crop_size = crop_size or img_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # 加载数据
        self.data = self._load_data()
        # 尝试加载网格
        self.grid = self._load_grid()
        
        # 数据分割
        self.indices = self._split_data()
        
        # 计算归一化统计（z-score），保存为均值/方差
        self.norm_stats: Optional[Dict[str, np.ndarray]] = None
        if self.normalize:
            self.norm_stats = self._compute_norm_stats()
        
        logger.info(f"PDEBenchDataset初始化完成: {len(self.indices)}个样本")
    
    def _load_data(self) -> np.ndarray:
        """加载HDF5数据"""
        try:
            # 支持目录路径：如果传入目录，则在其中自动发现单个.h5/.hdf5文件
            data_path = self.data_path
            if os.path.isdir(data_path):
                candidates = [
                    f for f in os.listdir(data_path)
                    if f.lower().endswith((".h5", ".hdf5"))
                ]
                if not candidates:
                    raise FileNotFoundError(
                        f"数据目录 '{data_path}' 中未找到任何HDF5文件(.h5/.hdf5)"
                    )
                # 选择第一个候选（按名称排序保证确定性）
                candidates.sort()
                data_path = os.path.join(data_path, candidates[0])
                logger.info(f"检测到目录路径，使用文件: {data_path}")

            with h5py.File(data_path, 'r') as f:
                logger.info(f"尝试直接访问键: '{self.data_key}'")
                
                # 首先尝试直接访问完整路径
                if self.data_key in f:
                    data = f[self.data_key][:]
                    logger.info(f"直接访问成功: {self.data_key}")
                else:
                    # 如果直接键不存在，尝试逐级解析
                    logger.info(f"直接访问失败，尝试逐级解析路径: '{self.data_key}'")
                    keys = self.data_key.split('/')
                    current = f

                    # 特殊处理：如果根下是编号分组（如'0000'），自动选择第一个分组的'data'
                    root_keys = list(f.keys())
                    if 'data' not in root_keys and len(root_keys) > 0 and root_keys[0].isdigit():
                        first_group = root_keys[0]
                        logger.info(f"检测到编号分组格式，自动使用'{first_group}/data'")
                        group = f[first_group]
                        if 'data' in group:
                            data = group['data'][:]
                            logger.info(f"成功访问 {first_group}/data，形状: {data.shape}")
                            return data
                        else:
                            available_keys = list(group.keys())
                            raise KeyError(f"组 '{first_group}' 中没有 'data' 键，可用键: {available_keys}")
                    
                    logger.info(f"根级别可用键: {list(f.keys())}")
                    
                    # 特殊处理：如果data_key是"0000/data"这样的格式
                    if len(keys) == 2 and keys[0] in f and keys[1] == 'data':
                        logger.info(f"检测到标准格式: {keys[0]}/{keys[1]}")
                        group = f[keys[0]]
                        logger.info(f"组 '{keys[0]}' 内容: {list(group.keys())}")
                        if 'data' in group:
                            data = group['data'][:]
                            logger.info(f"成功访问 {keys[0]}/data，形状: {data.shape}")
                        else:
                            available_keys = list(group.keys())
                            logger.error(f"组 '{keys[0]}' 中没有 'data' 键，可用键: {available_keys}")
                            raise KeyError(f"组 '{keys[0]}' 中没有 'data' 键，可用键: {available_keys}")
                    elif len(keys) == 1 and keys[0] in f:
                        # 如果只有一个键，比如"0000"，尝试访问其中的"data"
                        logger.info(f"检测到单键格式: {keys[0]}")
                        group = f[keys[0]]
                        logger.info(f"组 '{keys[0]}' 内容: {list(group.keys())}")
                        if 'data' in group:
                            data = group['data'][:]
                            logger.info(f"成功访问 {keys[0]}/data，形状: {data.shape}")
                        else:
                            available_keys = list(group.keys())
                            logger.error(f"组 '{keys[0]}' 中没有 'data' 键，可用键: {available_keys}")
                            raise KeyError(f"组 '{keys[0]}' 中没有 'data' 键，可用键: {available_keys}")
                    else:
                        # 通用路径解析
                        for i, key in enumerate(keys):
                            logger.info(f"步骤 {i}: 查找键 '{key}'，当前级别可用键: {list(current.keys())}")
                            if key in current:
                                current = current[key]
                                logger.info(f"步骤 {i}: 成功进入 '{key}'，类型: {type(current)}")
                            else:
                                logger.error(f"步骤 {i}: 键 '{key}' 不存在")
                                available_keys = list(current.keys())
                                logger.error(f"键 '{key}' 不存在于HDF5文件中。可用键: {available_keys}")
                                logger.error(f"完整路径尝试: {self.data_key}")
                                logger.error(f"当前层级路径: {keys[:i+1]}")
                                logger.error(f"当前层级可用键: {available_keys}")
                                raise KeyError(f"键 '{key}' 不存在于HDF5文件中。可用键: {available_keys}")
                        
                        # 检查最终对象是否是数据集
                        if hasattr(current, 'shape'):
                            data = current[:]
                            logger.info(f"数据加载成功，形状: {data.shape}")
                        else:
                            logger.error(f"最终对象不是数据集，类型: {type(current)}")
                            logger.error(f"最终对象内容: {current}")
                            raise KeyError(f"最终对象不是数据集，类型: {type(current)}")
                
            logger.info(f"数据加载成功: shape={data.shape}")
            return data
        except KeyError as e:
            logger.error(f"数据加载失败 - KeyError: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"数据加载失败 - 其他错误: {str(e)}")
            raise

    def _load_grid(self) -> Optional[np.ndarray]:
        """加载网格数据（如果存在）"""
        try:
            with h5py.File(self.data_path, 'r') as f:
                # 尝试加载网格数据，通常在同一个组中
                if self.data_key and '/' in self.data_key:
                    group_key = self.data_key.rsplit('/', 1)[0]  # 获取组路径
                    grid_key = f"{group_key}/grid"
                else:
                    # 如果data_key是单个键，尝试在该组中找grid
                    group_key = self.data_key
                    grid_key = f"{group_key}/grid"
                
                if grid_key in f:
                    grid = f[grid_key][:]
                    logger.info(f"网格数据加载成功: {grid_key}, shape={grid.shape}")
                    return grid
                else:
                    logger.info(f"未找到网格数据: {grid_key}")
                    return None
        except Exception as e:
            logger.warning(f"网格数据加载失败: {str(e)}")
            return None

    def _compute_norm_stats(self) -> Dict[str, np.ndarray]:
        """计算z-score归一化统计，支持 [N,H,W]、[N,H,W,C]、[T,H,W] 形式"""
        data = np.asarray(self.data)
        # 统一为 [N,H,W] 或 [N,H,W,C]
        if data.ndim == 2:
            data = data[None, ...]
        elif data.ndim == 3:
            pass
        elif data.ndim == 4:
            # 视最后一维为通道
            pass
        else:
            raise ValueError(f"不支持的数据形状用于归一化统计: {data.shape}")

        # 逐通道统计，如果没有通道维则按整体统计
        if data.ndim == 4:
            mean = data.mean(axis=(0,1,2), dtype=np.float64)
            std = data.std(axis=(0,1,2), dtype=np.float64)
            std = np.where(std > 1e-8, std, 1.0)
        else:
            mean = data.mean(dtype=np.float64)
            std = data.std(dtype=np.float64)
            std = std if std > 1e-8 else 1.0

        stats = {
            'mean': np.array(mean, dtype=np.float32),
            'std': np.array(std, dtype=np.float32)
        }
        logger.info(f"归一化统计: mean={np.array(stats['mean']).flatten()[:4]}..., std={np.array(stats['std']).flatten()[:4]}...")
        return stats

    def get_normalization_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """返回归一化统计，用于损失在原值域的计算"""
        return self.norm_stats

    def _split_data(self) -> List[int]:
        """根据模式分割数据索引"""
        total_samples = len(self.data)
        indices = list(range(total_samples))
        
        # 设置随机种子以确保可重复性
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        # 计算分割点
        train_end = int(total_samples * self.train_ratio)
        val_end = train_end + int(total_samples * self.val_ratio)
        
        if self.mode == 'train':
            return indices[:train_end]
        elif self.mode == 'val':
            return indices[train_end:val_end]
        elif self.mode == 'test':
            return indices[val_end:]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _apply_observation(self, data: np.ndarray) -> np.ndarray:
        """应用观测模式"""
        if self.observation_mode == 'sr':
            # 超分辨率观测
            return self._apply_sr_observation(data)
        elif self.observation_mode == 'crop':
            # 裁剪观测
            return self._apply_crop_observation(data)
        elif self.observation_mode == 'full':
            # 完整观测（无降质）
            return data
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")

    def _apply_sr_observation(self, data: np.ndarray) -> np.ndarray:
        """应用超分辨率观测"""
        # 使用高斯模糊和下采样
        from scipy.ndimage import gaussian_filter
        
        # 应用高斯模糊
        blurred = gaussian_filter(data, sigma=1.0)
        
        # 下采样
        h, w = data.shape[-2:]
        new_h = max(1, h // self.sr_scale)
        new_w = max(1, w // self.sr_scale)
        
        # 使用numpy的resize进行下采样
        if data.ndim == 2:  # (H, W)
            sr_data = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif data.ndim == 3:
            # 可能是 (T, H, W) 或 (H, W, C)，此函数应当接收2D数据，确保__getitem__已选择单通道/单帧
            # 兜底：如果传入3D，则逐切片下采样后再选择第一片
            sr_slices = []
            for i in range(data.shape[0]):
                sr_slices.append(cv2.resize(blurred[i], (new_w, new_h), interpolation=cv2.INTER_AREA))
            sr_data = sr_slices[0]
        else:
            raise ValueError(f"Unsupported data shape for SR: {data.shape}")
        
        return sr_data

    def _apply_crop_observation(self, data: np.ndarray) -> np.ndarray:
        """应用裁剪观测"""
        h, w = data.shape[-2:]
        
        # 确保裁剪大小不超过图像尺寸
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)
        
        # 随机选择裁剪位置
        if h > crop_h:
            top = np.random.randint(0, h - crop_h + 1)
        else:
            top = 0
            
        if w > crop_w:
            left = np.random.randint(0, w - crop_w + 1)
        else:
            left = 0
        
        # 应用裁剪
        if data.ndim == 3:  # (T, H, W)
            return data[:, top:top+crop_h, left:left+crop_w]
        elif data.ndim == 2:  # (H, W)
            return data[top:top+crop_h, left:left+crop_w]
        else:
            raise ValueError(f"Unsupported data shape for crop: {data.shape}")

    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """获取数据项，返回统一接口：input/target/observation/h_params"""
        actual_idx = self.indices[idx]
        full = self.data[actual_idx]  # [H, W] 或 [T, H, W] 或 [H, W, C]

        # 选择用于纯空间任务的目标二维场
        target = self._select_spatial_target(full)

    def _select_spatial_target(self, sample: np.ndarray) -> np.ndarray:
        """从样本中选择二维空间场用于纯空间预测。
        支持形状：[H,W]、[T,H,W]、[H,W,C]、[T,H,W,C]（兜底）
        """
        if sample.ndim == 2:
            return sample
        if sample.ndim == 3:
            h, w = sample.shape[-2], sample.shape[-1]
            # 判断是否为(H,W,C)：最后一维作为通道（通常C较小）
            if sample.shape[-1] <= 16 and sample.shape[0] != h and sample.shape[0] != w:
                # (H, W, C)
                return sample[..., 0]
            else:
                # (T, H, W)
                return sample[0]
        if sample.ndim == 4:
            # 兜底处理：选择第一帧、第一通道
            return sample[0, ..., 0]
        raise ValueError(f"Unsupported data shape: {sample.shape}")

        # 生成观测
        observation = self._apply_observation(target)

        # 构造h_params（与ops.degradation参数一致）
        h_params: Dict[str, Any] = {}
        if self.observation_mode in ['sr', 'super_resolution']:
            h_params = {
                'task': 'SR',
                'scale': int(self.sr_scale),
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
        elif self.observation_mode == 'crop':
            h, w = target.shape[-2], target.shape[-1]
            crop_h = min(self.crop_size, h)
            crop_w = min(self.crop_size, w)
            h_params = {
                'task': 'Crop',
                'crop_size': (int(crop_h), int(crop_w)),
                'boundary': 'mirror'
            }
        else:
            h_params = {'task': 'SR', 'scale': 1, 'sigma': 0.0, 'kernel_size': 1, 'boundary': 'mirror'}

        # 输入打包：baseline=观测上采样到目标尺寸，附加坐标与掩码
        obs = observation.astype(np.float32)
        # 上采样到目标大小以作为baseline
        baseline = cv2.resize(obs, (target.shape[-1], target.shape[-2]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mask = np.ones_like(target, dtype=np.float32)
        # 坐标编码
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, target.shape[-2], dtype=np.float32),
            np.linspace(-1, 1, target.shape[-1], dtype=np.float32),
            indexing='ij'
        )
        # 组装输入张量 [C_in, H, W]：baseline(1) + mask(1) + coords(2)
        input_channels = np.stack([baseline, mask, yy, xx], axis=0)

        sample = {
            'input': input_channels,
            'target': target.astype(np.float32)[None, ...],  # [1,H,W]
            'observation': obs[None, ...],                   # [1,h',w']
            'h_params': h_params,
        }
        if self.grid is not None:
            sample['grid'] = self.grid.astype(np.float32)

        return sample