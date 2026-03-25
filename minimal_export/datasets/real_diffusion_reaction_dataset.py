"""真实扩散-反应数据集加载器

专门处理真实扩散-反应数据集的时序数据加载，支持AR训练。
数据格式：E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5
结构：每个时间步为一个组，包含data数据集 (101, 128, 128, 2)
"""

import os
from typing import Any

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


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
        normalize_sample_size: int = 64,
        augmentation: dict | None = None,
        seed: int = 2025,
        max_samples: int | None = None,
        # 缓存相关
        window_cache_enabled: bool = False,
        window_cache_max_gb: float | None = None,
        cache_lru_size: int = 1024,
        # 内存映射（将HDF5文件加载到内存，加速读取/增加内存占用）
        memory_map_dataset: bool = False,
        selected_channels: list[int] | None = None,
        channel_index: int | None = None,
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
        self.normalize_sample_size = int(normalize_sample_size)
        self.augmentation = augmentation or {}
        # 新增：时间步配置保存
        self.time_step_start = time_step_start
        self.time_step_end = time_step_end
        self.time_step_stride = time_step_stride

        # 新增：样本上限支持
        self.max_samples = max_samples if max_samples is not None else 0

        # 预加载缓存与文件句柄（多进程安全）
        self._preloaded: dict[str, np.ndarray] = {}
        self._use_ram_preload: bool = False
        self.h5_file_path = data_path
        self._h5: h5py.File | None = None  # worker-local handle, lazily opened
        # 是否启用内存映射（driver='core'）
        self._memory_map_dataset: bool = bool(memory_map_dataset)
        self.selected_channels = selected_channels
        self.channel_index = channel_index

        # 窗口LRU缓存（用于消耗内存并提升重复访问速度）
        from collections import OrderedDict

        self.window_cache_enabled = bool(window_cache_enabled)
        self._window_cache: OrderedDict[Any, torch.Tensor] = OrderedDict()
        self._window_cache_bytes: int = 0
        self._window_cache_max_bytes: int = (
            int((window_cache_max_gb or 0) * (1024**3)) if window_cache_max_gb else 0
        )
        self._window_cache_capacity = int(cache_lru_size)

        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 检查数据文件
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        # 不在构造器中长期持有文件句柄，改为延迟打开（多进程安全）
        # self.h5_file = h5py.File(data_path, 'r')

        # 获取样本ID键（0000-0999）
        with h5py.File(self.h5_file_path, "r") as f:
            self.sample_keys = [k for k in f.keys() if k.isdigit()]
        self.sample_keys.sort()

        print(f"📅 发现 {len(self.sample_keys)} 个样本")
        print(f"   样本ID范围: {self.sample_keys[0]} ~ {self.sample_keys[-1]}")

        # 过滤样本（如果需要限制样本数量）
        if hasattr(self, "max_samples") and self.max_samples > 0:
            self.sample_keys = self.sample_keys[: self.max_samples]
            print(f"   限制样本数: {len(self.sample_keys)}")

        print(f"   使用样本数: {len(self.sample_keys)}")

        # 检查数据结构
        self._analyze_data_structure()

        # 生成有效序列索引
        self._generate_sequence_indices()

        # 数据集分割
        self._split_dataset(train_ratio, val_ratio, test_ratio, seed)

        # 计算归一化统计量
        if self.normalize:
            self._compute_normalization_stats()

    def _ensure_h5_open(self) -> h5py.File:
        """在worker中延迟打开HDF5文件（多进程安全）"""
        if self._h5 is None:
            if self._memory_map_dataset:
                # 使用核心驱动将文件加载至内存，避免磁盘IO瓶颈
                # 注意：backing_store=False 不会将更改写回磁盘
                try:
                    self._h5 = h5py.File(
                        self.h5_file_path, "r", driver="core", backing_store=False
                    )
                    print("🧩 HDF5内存映射已启用 (driver='core')")
                except Exception:
                    # 回退到标准只读模式
                    self._h5 = h5py.File(self.h5_file_path, "r", swmr=True)
                    print("⚠️ HDF5内存映射启用失败，回退到标准模式")
            else:
                self._h5 = h5py.File(self.h5_file_path, "r", swmr=True)
        return self._h5

    def _close_h5(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __getstate__(self):
        """使数据集在DataLoader worker中pickle安全：移除打开的句柄"""
        state = self.__dict__.copy()
        state.pop("_h5", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None

    def preload_all_samples(self, limit: int | None = None):
        """可选：将全部样本的data预加载到RAM以提速"""
        keys = (
            self.sample_keys[:limit]
            if (limit is not None and limit > 0)
            else self.sample_keys
        )
        print(f"🧠 预加载 {len(keys)} 个样本到RAM...")
        with h5py.File(self.h5_file_path, "r") as f:
            for k in keys:
                self._preloaded[k] = f[k]["data"][...]
        self._use_ram_preload = True
        print("✅ RAM预加载完成")

    def _get_data(self, sample_key: str):
        """获取某个样本的data数据集，优先使用RAM预加载"""
        if self._use_ram_preload and sample_key in self._preloaded:
            return self._preloaded[sample_key]
        h5 = self._ensure_h5_open()
        return h5[sample_key]["data"]

    def _estimate_tensor_bytes(self, t: torch.Tensor) -> int:
        try:
            return int(t.element_size() * t.numel())
        except Exception:
            return 0

    def _window_cache_get(self, key: tuple[str, int, int]) -> torch.Tensor | None:
        if not self.window_cache_enabled:
            return None
        w = self._window_cache.get(key)
        if w is not None:
            # LRU: move to end
            self._window_cache.move_to_end(key)
        return w

    def _window_cache_put(self, key: tuple[str, int, int], tensor: torch.Tensor):
        if not self.window_cache_enabled:
            return
        # 若存在，先删除旧项以更新字节计数
        old = self._window_cache.pop(key, None)
        if old is not None:
            self._window_cache_bytes -= self._estimate_tensor_bytes(old)
        # 插入新项
        self._window_cache[key] = tensor
        self._window_cache_bytes += self._estimate_tensor_bytes(tensor)
        # 控制容量与字节上限
        while True:
            over_capacity = (
                self._window_cache_capacity > 0
                and len(self._window_cache) > self._window_cache_capacity
            )
            over_bytes = (
                self._window_cache_max_bytes > 0
                and self._window_cache_bytes > self._window_cache_max_bytes
            )
            if not (over_capacity or over_bytes):
                break
            # 弹出最旧项
            k, v = self._window_cache.popitem(last=False)
            self._window_cache_bytes -= self._estimate_tensor_bytes(v)

    def _analyze_data_structure(self):
        """分析数据结构"""
        first_key = self.sample_keys[0]
        with h5py.File(self.h5_file_path, "r") as f:
            first_group = f[first_key]
            if "data" not in first_group:
                raise ValueError(f"样本 {first_key} 中未找到 'data' 数据集")
            data_shape = first_group["data"].shape
        print(f"📊 数据形状: {data_shape}")

        # 解析数据维度 (101, 128, 128, 2)
        # 正确理解：101个时间步，128x128空间分辨率，2个物理通道
        if len(data_shape) == 4:
            self.n_timesteps, self.height, self.width, self.n_channels = data_shape
            # 样本数是HDF5文件中的键数量（样本ID: 0000-0999）
            self.n_samples = len(self.sample_keys)
            print(
                f"   样本数: {self.n_samples} (样本ID: {self.sample_keys[0]}-{self.sample_keys[-1]})"
            )
            print(f"   每个样本时间步数: {self.n_timesteps}")
            print(f"   空间分辨率: {self.height} x {self.width}")
            print(f"   物理通道数: {self.n_channels}")
        else:
            raise ValueError(f"不支持的数据形状: {data_shape}")

        # 检查数据范围
        with h5py.File(self.h5_file_path, "r") as f:
            sample_data = f[first_key]["data"][:3]
        print(f"   数据范围: [{sample_data.min():.6f}, {sample_data.max():.6f}]")
        print(f"   数据均值: {sample_data.mean():.6f}")
        print(f"   数据标准差: {sample_data.std():.6f}")

    def _generate_sequence_indices(self):
        """生成有效的序列索引"""
        self.sequence_indices = []

        # 对每个样本生成时序序列
        for sample_idx in range(self.n_samples):
            # 限制起始时间步范围并考虑stride
            start_min = max(0, self.time_step_start)
            start_max_global = self.n_timesteps - (self.T_in + self.T_out)
            start_max_cfg = self.time_step_end - (self.T_in + self.T_out)
            start_max = min(start_max_global, start_max_cfg)
            if start_max < start_min:
                continue
            for start_time in range(
                start_min, start_max + 1, max(1, self.time_step_stride)
            ):
                self.sequence_indices.append((sample_idx, start_time))

        print(f"🔢 生成 {len(self.sequence_indices)} 个序列样本")

    def _split_dataset(
        self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
    ):
        """分割数据集 - 按样本ID划分避免数据泄露"""
        # 确保比例和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

        # 🔥 关键修复：按样本ID进行划分，避免数据泄露
        np.random.seed(seed)
        sample_ids = np.random.permutation(self.n_samples)

        # 计算每个集合的样本数
        n_train = int(self.n_samples * train_ratio)
        n_val = int(self.n_samples * val_ratio)

        # 分配样本ID到不同集合
        if self.split == "train":
            assigned_samples = set(sample_ids[:n_train])
        elif self.split == "val":
            assigned_samples = set(sample_ids[n_train : n_train + n_val])
        elif self.split == "test":
            assigned_samples = set(sample_ids[n_train + n_val :])
        else:
            raise ValueError(f"不支持的分割类型: {self.split}")

        # 筛选属于当前集合的序列索引
        self.indices = []
        for i, (sample_idx, start_time) in enumerate(self.sequence_indices):
            if sample_idx in assigned_samples:
                self.indices.append(i)

        print(
            f"📊 {self.split} 集: {len(assigned_samples)} 个样本, {len(self.indices)} 个序列"
        )

    def _compute_normalization_stats(self):
        """计算归一化统计量"""
        if self.split != "train":
            # 非训练集不计算统计量，使用预设值或从训练集加载
            self.mean = torch.zeros(self.n_channels)
            self.std = torch.ones(self.n_channels)
            return

        print("📈 计算归一化统计量...")

        # 采样部分数据计算统计量（可配置，默认64以降低初始化IO）
        try:
            cfg_n = int(self.normalize_sample_size)
        except Exception:
            cfg_n = 64
        sample_size = max(1, min(cfg_n, len(self.indices)))
        sample_indices = np.random.choice(len(self.indices), sample_size, replace=False)

        all_data = []
        for idx in sample_indices:
            sample_idx, start_time = self.sequence_indices[self.indices[idx]]
            sample_key = self.sample_keys[sample_idx]
            sample_data = self._get_data(sample_key)  # [T, H, W, C]
            for t in range(self.T_in + self.T_out):
                time_idx = start_time + t
                data = sample_data[time_idx]
                all_data.append(data)

        all_data = np.stack(all_data, axis=0)  # [N, H, W, C]
        self.mean = torch.tensor(all_data.mean(axis=(0, 1, 2)), dtype=torch.float32)
        self.std = torch.tensor(all_data.std(axis=(0, 1, 2)), dtype=torch.float32)
        if isinstance(self.channel_index, int):
            s = int(self.channel_index)
            self.mean = self.mean[s : s + 1].clone()
            self.std = self.std[s : s + 1].clone()
        elif (
            isinstance(self.selected_channels, list) and len(self.selected_channels) > 0
        ):
            idx = [int(i) for i in self.selected_channels]
            self.mean = self.mean[idx].clone()
            self.std = self.std[idx].clone()

        print(f"   均值: {self.mean}")
        print(f"   标准差: {self.std}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取单个样本"""
        # 获取序列索引
        sample_idx, start_time = self.sequence_indices[self.indices[idx]]

        # 获取对应的样本键与数据
        sample_key = self.sample_keys[sample_idx]
        sample_data = self._get_data(sample_key)  # [T, H, W, C]

        # 窗口矢量化读取，减少Python层循环与GIL压力
        t_start = int(start_time)
        t_total = int(self.T_in + self.T_out)

        # 先尝试窗口缓存（键：sample_key, t_start, t_total）
        cache_key = (sample_key, t_start, t_total)
        cached = self._window_cache_get(cache_key)
        if cached is not None:
            window_t = cached
        else:
            try:
                window_np = sample_data[
                    t_start : t_start + t_total
                ]  # [T_total, H, W, C]
            except Exception:
                # 兼容极端情况：逐步读取回退
                arrs = []
                for t in range(t_total):
                    arrs.append(sample_data[t_start + t])
                window_np = np.stack(arrs, axis=0)

            # 转为Tensor并重排维度
            window_t = torch.from_numpy(window_np)
            if window_t.dtype != torch.float32:
                window_t = window_t.float()
            window_t = window_t.permute(0, 3, 1, 2).contiguous()
            if isinstance(self.channel_index, int):
                s = int(self.channel_index)
                window_t = window_t[:, s : s + 1]
            elif (
                isinstance(self.selected_channels, list)
                and len(self.selected_channels) > 0
            ):
                idx = torch.tensor(
                    [int(i) for i in self.selected_channels], dtype=torch.long
                )
                window_t = window_t.index_select(1, idx)
            # 写入缓存
            self._window_cache_put(cache_key, window_t)

        input_sequence = window_t[: self.T_in]
        target_sequence = window_t[self.T_in : self.T_in + self.T_out]

        # 归一化
        if self.normalize:
            input_sequence = (
                input_sequence - self.mean.view(1, -1, 1, 1)
            ) / self.std.view(1, -1, 1, 1)
            target_sequence = (
                target_sequence - self.mean.view(1, -1, 1, 1)
            ) / self.std.view(1, -1, 1, 1)

        # 数据增强
        if self.split == "train" and self.augmentation.get("enabled", False):
            input_sequence, target_sequence = self._apply_augmentation(
                input_sequence, target_sequence
            )

        return {
            "input_sequence": input_sequence,  # [T_in, C, H, W]
            "target_sequence": target_sequence,  # [T_out, C, H, W]
            "sample_idx": sample_idx,
            "start_time": start_time,
            "sample_key": sample_key,
            "time_indices": [start_time + t for t in range(self.T_in + self.T_out)],
        }

    def _apply_augmentation(
        self, input_seq: torch.Tensor, target_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """应用数据增强"""
        if np.random.random() < self.augmentation.get("flip_prob", 0.0):
            input_seq = torch.flip(input_seq, dims=[-1])
            target_seq = torch.flip(target_seq, dims=[-1])
        if np.random.random() < self.augmentation.get("rotate_prob", 0.0):
            k = np.random.randint(1, 4)
            input_seq = torch.rot90(input_seq, k=k, dims=[-2, -1])
            target_seq = torch.rot90(target_seq, k=k, dims=[-2, -1])
        noise_std = self.augmentation.get("noise_std", 0.0)
        if noise_std > 0:
            input_seq += torch.randn_like(input_seq) * noise_std
            target_seq += torch.randn_like(target_seq) * noise_std
        return input_seq, target_seq

    def __del__(self):
        """析构函数，关闭HDF5文件"""
        self._close_h5()


class RealDiffusionReactionDataModule(pl.LightningDataModule):
    """真实扩散-反应数据模块"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.data_path = config.data.data_path

        # 时序配置
        self.T_in = config.data.get("T_in", 1)
        self.T_out = config.data.get("T_out", 1)

        # 数据集分割配置
        self.train_ratio = config.data.get("train_ratio", 0.7)
        self.val_ratio = config.data.get("val_ratio", 0.15)
        self.test_ratio = config.data.get("test_ratio", 0.15)

        # 时间步配置
        self.time_step_start = config.data.get("time_step_start", 0)
        self.time_step_end = config.data.get("time_step_end", 980)
        self.time_step_stride = config.data.get("time_step_stride", 1)

        # 其他配置
        self.normalize = config.data.get("normalize", True)
        self.normalize_sample_size = int(config.data.get("normalize_sample_size", 64))
        self.augmentation = config.data.get("augmentation", {})
        self.seed = config.get("seed", 2025)

        # 样本限制
        self.max_samples = config.data.get("max_samples", None)

        # 安全读取data.dataloader配置，避免缺失键导致异常
        dl_cfg = getattr(config.data, "dataloader", DictConfig({}))

        # 数据加载器配置（先给出安全默认值，然后在下方用dl_cfg覆盖）
        self.batch_size = getattr(config, "training", DictConfig({})).get(
            "batch_size", 8
        )
        self.val_batch_size = getattr(dl_cfg, "val_batch_size", self.batch_size)
        self.test_batch_size = getattr(config, "testing", DictConfig({})).get(
            "batch_size", 1
        )
        self.num_workers = getattr(
            dl_cfg, "num_workers", getattr(config.hardware, "num_workers", 4)
        )
        self.pin_memory = getattr(
            dl_cfg, "pin_memory", getattr(config.hardware, "pin_memory", True)
        )
        self.persistent_workers = getattr(dl_cfg, "persistent_workers", True)

        # 新增：读取样本上限配置
        self.sample_limit = config.data.get("sample_limit", None)

        # 从data.dataloader读取参数为主，hardware为备选（完整覆盖）
        self.batch_size = dl_cfg.get(
            "batch_size", config.training.get("batch_size", self.batch_size)
        )
        self.val_batch_size = dl_cfg.get("val_batch_size", self.val_batch_size)
        self.test_batch_size = dl_cfg.get("test_batch_size", self.test_batch_size)
        self.num_workers = dl_cfg.get("num_workers", self.num_workers)
        try:
            import torch as _t

            self.pin_memory = bool(
                dl_cfg.get(
                    "pin_memory",
                    getattr(config.hardware, "pin_memory", _t.cuda.is_available()),
                )
            )
        except Exception:
            self.pin_memory = False
        self.persistent_workers = dl_cfg.get(
            "persistent_workers",
            self.persistent_workers if self.num_workers > 0 else False,
        )
        self.prefetch_factor = dl_cfg.get("prefetch_factor", 2)
        self.drop_last = dl_cfg.get("drop_last", True)
        self.shuffle = dl_cfg.get("shuffle", True)
        # 新增：支持multiprocessing_context/timeout/pin_memory_device
        mp_mode = dl_cfg.get("multiprocessing_context", None)
        self.mp_context = None
        self._mp_mode_str = mp_mode if isinstance(mp_mode, str) else None
        if isinstance(mp_mode, str) and mp_mode:
            try:
                import multiprocessing as mp

                self.mp_context = mp.get_context(mp_mode)
            except Exception:
                self.mp_context = None
        else:
            self.mp_context = mp_mode
        self.timeout = dl_cfg.get("timeout", 0)
        if int(self.num_workers) == 0:
            self.timeout = 0
        self.pin_memory_device = None

        # RAM预加载与缓存开关
        self.preload_entire_dataset = dl_cfg.get(
            "preload_entire_dataset", False
        ) or getattr(
            getattr(config, "hardware", DictConfig({})), "memory", DictConfig({})
        ).get(
            "preload_data", False
        )

        # 窗口/批次缓存配置
        self.window_cache_enabled = bool(dl_cfg.get("window_cache_enabled", False))
        self.window_cache_max_gb = dl_cfg.get("window_cache_max_gb", None)
        self.batch_cache_enabled = bool(dl_cfg.get("batch_cache_enabled", False))
        self.batch_cache_max_gb = dl_cfg.get("batch_cache_max_gb", None)
        self.cache_lru_size = int(dl_cfg.get("cache_lru_size", 1024))
        self.memory_map_dataset = bool(
            dl_cfg.get("memory_map_dataset", False)
            or getattr(
                getattr(config, "hardware", DictConfig({})), "memory", DictConfig({})
            ).get("memory_map_dataset", False)
        )
        from collections import OrderedDict

        self._batch_cache: OrderedDict[Any, dict[str, torch.Tensor]] = OrderedDict()
        self._batch_cache_bytes: int = 0
        self._batch_cache_max_bytes: int = (
            int((self.batch_cache_max_gb or 0) * (1024**3))
            if self.batch_cache_max_gb
            else 0
        )

        # 预填充批次缓存配置（用于快速提升内存占用）
        self.prefill_cache_on_setup = bool(dl_cfg.get("prefill_cache_on_setup", True))
        self.prefill_cache_batches = int(dl_cfg.get("prefill_cache_batches", 80))
        self.prefill_cache_batch_size = int(
            dl_cfg.get("prefill_cache_batch_size", self.batch_size)
        )

        # 若启用整数据RAM预加载：仅当使用非fork上下文时强制单进程，fork可共享页面
        if self.preload_entire_dataset:
            use_fork = False
            try:
                if self.mp_context is not None and hasattr(
                    self.mp_context, "get_start_method"
                ):
                    use_fork = self.mp_context.get_start_method() == "fork"
                elif isinstance(self._mp_mode_str, str):
                    use_fork = self._mp_mode_str == "fork"
            except Exception:
                use_fork = False
            if not use_fork:
                print(
                    "⚠️ preload_entire_dataset=True 且非fork 上下文，已将 num_workers 强制为 0 以避免多进程复制RAM缓存"
                )
                self.num_workers = 0
                self.persistent_workers = False

    def setup(self, stage: str | None = None):
        """设置数据集"""
        # 配置摘要日志，便于确认缓存与并行参数生效
        try:
            mp_mode = None
            if self.mp_context is not None and hasattr(
                self.mp_context, "get_start_method"
            ):
                mp_mode = self.mp_context.get_start_method()
            elif isinstance(self._mp_mode_str, str):
                mp_mode = self._mp_mode_str
            print(
                f"⚙️ DataLoader并行: num_workers={self.num_workers}, prefetch_factor={self.prefetch_factor}, "
                f"persistent_workers={self.persistent_workers}, multiprocessing_context={mp_mode}"
            )
            print(
                f"🧩 HDF5内存映射: memory_map_dataset={self.memory_map_dataset}; 🧠 RAM预加载: preload_entire_dataset={self.preload_entire_dataset}"
            )
            print(
                f"🗂️ 窗口缓存: enabled={self.window_cache_enabled}, max_gb={self.window_cache_max_gb}, lru_size={self.cache_lru_size}; "
                f"📦 批次缓存: enabled={self.batch_cache_enabled}, max_gb={self.batch_cache_max_gb}"
            )
        except Exception:
            pass
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
                normalize_sample_size=self.normalize_sample_size,
                augmentation=self.augmentation,
                seed=self.seed,
                max_samples=self.sample_limit,
                window_cache_enabled=self.window_cache_enabled,
                window_cache_max_gb=self.window_cache_max_gb,
                cache_lru_size=self.cache_lru_size,
                memory_map_dataset=self.memory_map_dataset,
                channel_index=0,
            )
            # 若启用整数据预加载，则在训练集上触发RAM预加载
            try:
                if bool(self.preload_entire_dataset):
                    limit = None
                    if isinstance(self.sample_limit, int) and self.sample_limit > 0:
                        limit = self.sample_limit
                    self.train_dataset.preload_all_samples(limit=limit)
            except Exception as _pre_err:
                print(f"⚠️ 训练集RAM预加载失败: {_pre_err}")

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
                normalize_sample_size=self.normalize_sample_size,
                seed=self.seed,
                max_samples=self.sample_limit,
                window_cache_enabled=self.window_cache_enabled,
                window_cache_max_gb=self.window_cache_max_gb,
                cache_lru_size=self.cache_lru_size,
                memory_map_dataset=self.memory_map_dataset,
                channel_index=0,
            )
            # 可选：对验证集进行轻量预加载（与训练集一致）
            try:
                if bool(self.preload_entire_dataset):
                    limit = None
                    if isinstance(self.sample_limit, int) and self.sample_limit > 0:
                        limit = self.sample_limit
                    self.val_dataset.preload_all_samples(limit=limit)
            except Exception as _pre_err_v:
                print(f"⚠️ 验证集RAM预加载失败: {_pre_err_v}")

            # 统一归一化统计：将训练集的mean/std传播到验证集
            if (
                getattr(self.train_dataset, "mean", None) is not None
                and getattr(self.train_dataset, "std", None) is not None
            ):
                self.val_dataset.mean = self.train_dataset.mean.clone()
                self.val_dataset.std = self.train_dataset.std.clone()
            else:
                # 回退：如果训练集未计算统计量（极端情况），设置默认值
                self.val_dataset.mean = torch.zeros(
                    getattr(self.train_dataset, "n_channels", 2), dtype=torch.float32
                )
                self.val_dataset.std = torch.ones(
                    getattr(self.train_dataset, "n_channels", 2), dtype=torch.float32
                )

            # RAM预加载（可选）：仅在训练集执行一次，并将缓存共享到验证集，避免重复占用内存
            if self.preload_entire_dataset:
                self.train_dataset.preload_all_samples(limit=self.sample_limit)
                # 共享缓存到验证集（浅拷贝引用，不重复分配）
                self.val_dataset._preloaded = self.train_dataset._preloaded
                self.val_dataset._use_ram_preload = True

            # 预填充批次缓存：基于训练集样本，构造若干大批次并缓存
            try:
                if (
                    self.batch_cache_enabled
                    and self.prefill_cache_on_setup
                    and self.prefill_cache_batches > 0
                ):
                    import random as _rnd

                    total_prefill = int(self.prefill_cache_batches)
                    bsz = int(self.prefill_cache_batch_size)
                    print(
                        f"🧺 预填充批次缓存: batches={total_prefill}, batch_size={bsz}"
                    )
                    for b in range(total_prefill):
                        # 从数据集长度随机采样索引（注意：__getitem__ 的 idx 是 self.indices 的位置）
                        sel = [
                            _rnd.randrange(len(self.train_dataset)) for _ in range(bsz)
                        ]
                        samples = [self.train_dataset[idx] for idx in sel]
                        batch = self._collate_fn(samples)
                        # 使用构造的键写入缓存
                        key = tuple(
                            (str(s["sample_key"]), int(s["start_time"]))
                            for s in samples
                        )
                        self._batch_cache_put(key, batch)
                        if (b + 1) % max(1, (total_prefill // 10)) == 0:
                            gb = self._batch_cache_bytes / (1024**3)
                            print(
                                f"   进度: {b+1}/{total_prefill}, 当前批次缓存占用≈{gb:.1f} GB"
                            )
                    gb = self._batch_cache_bytes / (1024**3)
                    print(f"✅ 批次缓存预填充完成，累计≈{gb:.1f} GB")
            except Exception as _prefill_err:
                print(f"⚠️ 批次缓存预填充失败: {_prefill_err}")

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
                normalize_sample_size=self.normalize_sample_size,
                seed=self.seed,
                max_samples=self.sample_limit,
                window_cache_enabled=self.window_cache_enabled,
                window_cache_max_gb=self.window_cache_max_gb,
                cache_lru_size=self.cache_lru_size,
                memory_map_dataset=self.memory_map_dataset,
                channel_index=0,
            )

            # 统一归一化统计：将训练集的mean/std传播到测试集
            if (
                getattr(self.train_dataset, "mean", None) is not None
                and getattr(self.train_dataset, "std", None) is not None
            ):
                self.test_dataset.mean = self.train_dataset.mean.clone()
                self.test_dataset.std = self.train_dataset.std.clone()
            else:
                # 回退：极端情况下使用默认零均值/单位方差
                self.test_dataset.mean = torch.zeros(
                    getattr(self.train_dataset, "n_channels", 2), dtype=torch.float32
                )
                self.test_dataset.std = torch.ones(
                    getattr(self.train_dataset, "n_channels", 2), dtype=torch.float32
                )

            # 若启用整数据 RAM 预加载，共享训练集缓存到测试集（浅拷贝引用）
            if self.preload_entire_dataset:
                self.test_dataset._preloaded = self.train_dataset._preloaded
                self.test_dataset._use_ram_preload = True

    def get_normalization_stats(self) -> dict[str, torch.Tensor] | None:
        """返回归一化统计，用于损失在原值域的计算"""
        train_ds = getattr(self, "train_dataset", None)
        if train_ds is None:
            return None
        mean = getattr(train_ds, "mean", None)
        std = getattr(train_ds, "std", None)
        if mean is None or std is None:
            return None
        try:
            mean_t = mean.detach().clone()
        except Exception:
            mean_t = torch.as_tensor(mean, dtype=torch.float32)
        try:
            std_t = std.detach().clone()
        except Exception:
            std_t = torch.as_tensor(std, dtype=torch.float32)
        return {
            "data_mean": mean_t,
            "data_std": std_t,
        }

    def _estimate_batch_bytes(self, batch: dict[str, torch.Tensor]) -> int:
        total = 0
        for k in ("input_sequence", "target_sequence"):
            if k in batch and isinstance(batch[k], torch.Tensor):
                total += self._tensor_nbytes(batch[k])
        return total

    def _tensor_nbytes(self, t: torch.Tensor) -> int:
        try:
            return int(t.element_size() * t.numel())
        except Exception:
            return 0

    def _batch_cache_get(
        self, key: tuple[tuple[str, int], ...]
    ) -> dict[str, torch.Tensor] | None:
        if not self.batch_cache_enabled:
            return None
        b = self._batch_cache.get(key)
        if b is not None:
            self._batch_cache.move_to_end(key)
        return b

    def _batch_cache_put(
        self, key: tuple[tuple[str, int], ...], batch: dict[str, torch.Tensor]
    ):
        if not self.batch_cache_enabled:
            return
        old = self._batch_cache.pop(key, None)
        if old is not None:
            self._batch_cache_bytes -= self._estimate_batch_bytes(old)
        self._batch_cache[key] = batch
        self._batch_cache_bytes += self._estimate_batch_bytes(batch)
        while True:
            over_capacity = (
                self.cache_lru_size > 0 and len(self._batch_cache) > self.cache_lru_size
            )
            over_bytes = (
                self._batch_cache_max_bytes > 0
                and self._batch_cache_bytes > self._batch_cache_max_bytes
            )
            if not (over_capacity or over_bytes):
                break
            k, v = self._batch_cache.popitem(last=False)
            self._batch_cache_bytes -= self._estimate_batch_bytes(v)

    def _collate_fn(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        # 以(batch内的 sample_key, start_time)列表作为键
        key = tuple((str(s["sample_key"]), int(s["start_time"])) for s in samples)
        cached = self._batch_cache_get(key)
        if cached is not None:
            return cached
        # 默认拼接
        input_seq = torch.stack([s["input_sequence"] for s in samples], dim=0)
        target_seq = torch.stack([s["target_sequence"] for s in samples], dim=0)
        batch = {
            "input_sequence": input_seq,
            "target_sequence": target_seq,
            "sample_idx": torch.tensor(
                [s["sample_idx"] for s in samples], dtype=torch.long
            ),
            "start_time": torch.tensor(
                [s["start_time"] for s in samples], dtype=torch.long
            ),
            "sample_key": [s["sample_key"] for s in samples],
            "time_indices": [s["time_indices"] for s in samples],
        }
        self._batch_cache_put(key, batch)
        return batch

    def _worker_init_fn(self, worker_id: int):
        """可选的worker初始化函数：设置随机种子"""
        np.random.seed(self.seed + worker_id)
        torch.manual_seed(self.seed + worker_id)

    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        # 仅在 num_workers > 0 时传递 prefetch_factor，避免 PyTorch 抛错
        _kwargs = dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
            multiprocessing_context=self.mp_context,
            timeout=self.timeout,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            _kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate_fn if self.batch_cache_enabled else None,
            **_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        # 强制单进程与CPU内存，确保稳定性
        _kwargs = dict(
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            worker_init_fn=None,
            multiprocessing_context=None,
            timeout=0,
        )
        # 移除 prefetch_factor 设置，因为 num_workers=0

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn if self.batch_cache_enabled else None,
            **_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        # 强制单进程与CPU内存，确保稳定性
        _kwargs = dict(
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            worker_init_fn=None,
            multiprocessing_context=None,
            timeout=0,
        )

        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn if self.batch_cache_enabled else None,
            **_kwargs,
        )


if __name__ == "__main__":
    """测试数据集加载器"""
    from omegaconf import DictConfig

    # 测试配置
    config = DictConfig(
        {
            "data": {
                "data_path": "E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5",
                "T_in": 1,
                "T_out": 20,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "time_step_start": 0,
                "time_step_end": 980,
                "time_step_stride": 1,
                "normalize": True,
                "augmentation": {
                    "enabled": True,
                    "flip_prob": 0.5,
                    "rotate_prob": 0.3,
                    "noise_std": 0.01,
                },
                "dataloader": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "prefetch_factor": 2,
                    "drop_last": True,
                    "shuffle": True,
                    "preload_entire_dataset": False,
                },
            },
            "training": {"batch_size": 4},
            "hardware": {
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "memory": {"preload_data": False},
            },
            "testing": {"batch_size": 1},
            "seed": 2025,
        }
    )

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
