#!/usr/bin/env python3
"""
真实扩散-反应数据AR训练脚本
专门用于训练真实数据集的20步AR预测模型
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import traceback
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# AMP统一接口：优先使用 torch.autocast（带 device_type），GradScaler 保持兼容
try:
    from torch import autocast  # torch.autocast(device_type=...)
except Exception:
    # 兼容旧版：退回到 torch.cuda.amp.autocast
    from torch.cuda.amp import autocast  # type: ignore
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import h5py
import psutil
import torch.distributed as dist
import numpy as np
import random
from functools import partial


def convert_numpy_types(obj):
    """递归转换numpy类型为JSON可序列化的Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# 添加项目根目录到路径，确保无论从哪个工作目录启动脚本都能正确导入包
project_root = Path(__file__).resolve().parents[2]
training_dir = Path(__file__).resolve().parent
for path in (training_dir, project_root):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper
from ops.losses import compute_total_loss, compute_ar_total_loss
from utils.metrics import compute_metrics
from utils.logger import setup_logger
from ops.degradation import SuperResolutionOperator, CropOperator
from utils.resource_monitor import ResourceMonitor

# 安全/快速collate（过滤None/低GIL压力），不可用时回退为None
try:
    from utils.collate import fast_collate_fn
except Exception:
    fast_collate_fn = None
try:
    from utils.collate import safe_collate_fn, fast_collate_fn
except Exception:
    safe_collate_fn = None
    fast_collate_fn = None

# 导入可视化模块
# 新增：CPU燃烧数据集与collate可选导入（不可用时回退）
try:
    from utils.cpu_burn_dataset import CpuBurnDataset
except Exception:
    CpuBurnDataset = None
try:
    from utils.cpu_burn_collate import cpu_burn_collate
except Exception:
    cpu_burn_collate = None

 
VISUALIZATION_AVAILABLE = False
try:
    # 先尝试导入轻量的 AR 可视化器；只要该模块可用即可开启可视化
    from utils.ar_visualizer import ARTrainingVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AR visualization not available: {e}")
    VISUALIZATION_AVAILABLE = False

# 尝试导入 PDEBench 综合可视化器（可选，不影响 VISUALIZATION_AVAILABLE）
try:
    from tools.visualization.pde_bench_visualizer import PDEBenchVisualizer
except ImportError as e:
    # 不禁用可视化，仅记录提示，AR 可视化仍然可用
    print(f"Note: PDEBench visualizer not available: {e}")

# 顶层 worker_init_fn，避免本地函数无法pickle
def seed_worker_fn(worker_id: int, base_seed: int = 2025):
    try:
        worker_seed = int(base_seed) + int(worker_id)
    except Exception:
        worker_seed = 2025 + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        torch.manual_seed(worker_seed)
    except Exception:
        pass

# 使用安全collate，避免None样本导致default_collate异常
try:
    from utils.collate import safe_collate_fn
except Exception:
    safe_collate_fn = None


class RealDataARTrainer:
    """真实数据AR训练器"""
    
    def _cfg_select(self, *keys, default=None):
        """Safely select first non-None config value from keys."""
        from omegaconf import OmegaConf
        for k in keys:
            try:
                val = OmegaConf.select(self.config, k, default=None)
            except Exception:
                val = None
            if val is not None:
                return val
        return default
    
    def __init__(self, config_path: str = None):
        """初始化训练器"""
        self.setup_config(config_path)
        # 动态配置校验与理顺，确保数据加载/内存/AMP/观测算子参数等一致性
        try:
            self.validate_config()
        except Exception as _vc_err:
            print(f"配置校验失败，继续使用原配置: {_vc_err}")
        # 课程学习状态初始化
        try:
            self.current_stage = 0
            # 初始化阶段内epoch计数，避免训练早期访问属性不存在
            self.stage_epoch = 0
            self._curriculum_enabled = bool(getattr(getattr(self.config, 'training', {}), 'curriculum', {}).get('enabled', False))
            self._curriculum_stages = list(getattr(getattr(self.config.training, 'curriculum', {}), 'stages', [])) if hasattr(self.config, 'training') else []
            # 预计算阶段边界（起始/结束epoch），用于快速查询
            self._stage_boundaries = []
            cum_epoch = 0
            if self._curriculum_enabled and self._curriculum_stages:
                for st in self._curriculum_stages:
                    dur = int(st.get('epochs', 0) or 0)
                    start = cum_epoch
                    end = cum_epoch + max(dur, 0)
                    self._stage_boundaries.append({'start': start, 'end': end, 'T_out': int(st.get('T_out', getattr(self.config.data, 'T_out', 20))), 'description': st.get('description', '')})
                    cum_epoch = end
        except Exception:
            # 保守初始化，避免训练过程中访问异常
            self.current_stage = 0
            self.stage_epoch = 0
            self._curriculum_enabled = False
            self._curriculum_stages = []
            self._stage_boundaries = []
        self.setup_logging()
        self.setup_device()
        self.setup_memory_management()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_monitoring()

    def get_current_T_out(self, epoch: int) -> int:
        """根据课程学习配置返回当前epoch的 T_out，并更新 current_stage。

        若未启用课程学习或配置为空，返回 data.T_out 的默认值。
        """
        try:
            if self._curriculum_enabled and self._stage_boundaries:
                for idx, st in enumerate(self._stage_boundaries):
                    if epoch >= st['start'] and epoch < st['end']:
                        self.current_stage = idx
                        return int(st['T_out'])
                # 超过最后阶段边界，停留在最后阶段
                self.current_stage = len(self._stage_boundaries) - 1
                return int(self._stage_boundaries[-1]['T_out'])
        except Exception:
            pass
        # 回退到默认 data.T_out
        try:
            return int(getattr(self.config.data, 'T_out', 20))
        except Exception:
            return 20

    def cleanup_distributed(self):
        """清理分布式进程组（若已初始化）。

        在所有训练退出路径调用，确保 destroy_process_group() 被正确执行，
        满足开发文档关于分布式清理的要求。
        """
        try:
            if hasattr(torch, 'distributed') and dist.is_available():
                if dist.is_initialized():
                    try:
                        # 尽量尝试一次同步，避免悬挂
                        dist.barrier()
                    except Exception:
                        pass
                    try:
                        dist.destroy_process_group()
                        print("[DDP] 已销毁进程组")
                    except Exception as e:
                        print(f"[DDP] 销毁进程组失败: {e}")
        except Exception:
            # 保守降级，避免在清理过程中影响主流程退出
            pass

    def validate_config(self):
        """动态配置校验与合理化，遵循开发文档的资源管理与一致性要求"""
        cfg = self.config
        # 1) DataLoader参数一致性：num_workers=0 时禁用 prefetch_factor/persistent_workers
        try:
            dl = getattr(cfg.data, 'dataloader', None)
            if dl is not None:
                nw = int(getattr(dl, 'num_workers', 0) or 0)
                if nw <= 0:
                    setattr(dl, 'prefetch_factor', None)
                    setattr(dl, 'persistent_workers', False)
                # pin_memory_device 在旧版PyTorch不支持时应避免设置
                if hasattr(dl, 'pin_memory_device') and getattr(dl, 'pin_memory_device') is None:
                    # 若显式设置为 None，则移除该键避免后续构造错误
                    try:
                        delattr(dl, 'pin_memory_device')
                    except Exception:
                        pass
        except Exception:
            pass

        # 2) AMP/精度合理化：优先bf16-mixed（A100以上显卡）、否则16-mixed；确保allow_tf32配置可用
        try:
            prec = str(getattr(cfg.experiment, 'precision', '16-mixed'))
            if torch.cuda.is_available():
                cap_major = torch.cuda.get_device_capability()[0]
                # A100/H100等通常cap>=8，优先bf16
                if cap_major >= 8:
                    cfg.experiment.precision = 'bf16-mixed'
                else:
                    cfg.experiment.precision = '16-mixed'
            else:
                cfg.experiment.precision = '32'
            # 允许TF32加速（与开发文档一致）
            hw = getattr(cfg, 'hardware', None)
            if hw is None:
                from omegaconf import DictConfig
                cfg.hardware = DictConfig({})
                hw = cfg.hardware
            if not hasattr(hw, 'allow_tf32'):
                hw.allow_tf32 = True
        except Exception:
            pass

        # 3) 观测算子参数校验：kernel_size为奇数，sigma非负；插值只能为area/bilinear/nearest
        try:
            obs = getattr(cfg, 'observation', None)
            if obs is not None:
                ks = int(getattr(obs, 'kernel_size', 5) or 5)
                if ks % 2 == 0:
                    ks = ks + 1
                    obs.kernel_size = ks
                sigma = float(getattr(obs, 'blur_sigma', 0.0) or 0.0)
                if sigma < 0:
                    obs.blur_sigma = 0.0
                interp = str(getattr(obs, 'downsample_interpolation', 'area'))
                if interp not in ('area', 'bilinear', 'nearest'):
                    obs.downsample_interpolation = 'area'
        except Exception:
            pass

        # 4) 早停参数校验：至少 patience>=20，min_delta默认1e-4
        try:
            tr = getattr(cfg, 'training', None)
            if tr is not None:
                es = getattr(tr, 'early_stopping', None)
                if es is None:
                    from omegaconf import DictConfig
                    tr.early_stopping = DictConfig({'enabled': True, 'patience': 50, 'min_delta': 1e-4, 'monitor': 'val_loss'})
                else:
                    if not hasattr(es, 'enabled'):
                        es.enabled = True
                    if not hasattr(es, 'patience') or int(getattr(es, 'patience', 0) or 0) < 20:
                        es.patience = 20
                    if not hasattr(es, 'min_delta'):
                        es.min_delta = 1e-4
                    if not hasattr(es, 'monitor'):
                        es.monitor = 'val_loss'
        except Exception:
            pass

        # 5) 检查点策略校验：最大保留数至少2；周期保存间隔为非负
        try:
            tr = getattr(cfg, 'training', None)
            if tr is not None:
                ck = getattr(tr, 'checkpoint', None)
                if ck is None:
                    from omegaconf import DictConfig
                    tr.checkpoint = DictConfig({'save_last': True, 'save_best': True, 'save_every_n_epochs': 0, 'max_keep': 2})
                else:
                    if not hasattr(ck, 'max_keep') or int(getattr(ck, 'max_keep', 0) or 0) < 2:
                        ck.max_keep = 2
                    if not hasattr(ck, 'save_every_n_epochs') or int(getattr(ck, 'save_every_n_epochs', 0) or 0) < 0:
                        ck.save_every_n_epochs = 0
        except Exception:
            pass

        # 6) Dataloader批次大小合理化：确保val/test bs存在，默认等于train bs
        try:
            dl = getattr(cfg.data, 'dataloader', None)
            if dl is not None:
                bs = int(getattr(dl, 'batch_size', getattr(cfg.training, 'batch_size', 32)))
                if not hasattr(dl, 'val_batch_size'):
                    dl.val_batch_size = bs
                if not hasattr(dl, 'test_batch_size'):
                    dl.test_batch_size = 1
        except Exception:
            pass
        
    def setup_config(self, config_path: str = None):
        """设置配置"""
        if config_path and os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
        else:
            # 默认配置
            self.config = DictConfig({
                'experiment': {
                    'name': 'Real-DR2D-AR-T20-128-SwinUNet-s2025',
                    'seed': 2025,
                    'seeds': [42, 123, 456],
                    'use_multi_seeds': False,
                    'output_dir': 'runs',
                    'device': 'cuda',
                    'precision': '16-mixed',
                    'log_every_n_steps': 10
                },
                'data': {
                    'data_path': 'E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
                    'T_in': 1,
                    'T_out': 20,
                    'img_size': 128,
                    'channels': 2,
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
                    },
                    'keys': ['u', 'v']
                },
                'model': {
                    'name': 'SwinUNet',
                    'in_channels': 2,
                    'out_channels': 2,
                    'img_size': 128,
                    'patch_size': 4,
                    'window_size': 8,
                    'depths': [2, 2, 6, 2],
                    'num_heads': [3, 6, 12, 24],
                    'embed_dim': 96,
                    'mlp_ratio': 4.0,
                    'drop_rate': 0.1,
                    'attn_drop_rate': 0.1,
                    'drop_path_rate': 0.2
                },
                'training': {
                    'epochs': 1,  # 测试配置
                    'batch_size': 8,  # 降低批次大小避免OOM
                    'accumulate_grad_batches': 1,  # 大批次不需要梯度累积
                    'optimizer': {
                        'name': 'AdamW',
                        'lr': 1e-4,  # 大批次使用稍大学习率
                        'weight_decay': 1e-4,
                        'betas': [0.9, 0.999]
                    },
                    'scheduler': {
                        'name': 'CosineAnnealingLR',
                        'T_max': 1,  # 单epoch测试
                        'eta_min': 1e-6,
                        'warmup_epochs': 0  # 测试阶段无需warmup
                    },
                    'gradient_clip_val': 1.0,
                    'amp': {
                        'enabled': True,
                        'opt_level': 'O1'
                    },
                    'curriculum': {
                        'enabled': True,
                        'stages': [
                            {'epochs': 1, 'T_out': 5, 'description': '测试阶段: 预测5步'},
                            {'epochs': 40, 'T_out': 15, 'description': '阶段3: 预测15步'},
                            {'epochs': 80, 'T_out': 20, 'description': '阶段4: 预测20步（最终目标）'}
                        ]
                    }
                },
                'loss': {
                    # 统一三件套配置：重建/频谱/DC
                    'reconstruction': {'weight': 1.0},
                    'spectral': {
                        'weight': 0.5,
                        'low_freq_modes': 16,
                        'use_rfft': False,
                        'normalize': False,
                        'boundary_mode': 'mirror'  # mirror/zero/none
                    },
                    'data_consistency': {
                        'weight': 1.0
                    },
                    # 兼容旧字段但不使用
                    'degradation_consistency': {
                        'weight': 0.0
                    },
                    # 额外项
                    'gradient_weight': 0.0
                },
                'observation': {
                    'mode': 'sr',
                    'scale_factor': 2,
                    'blur_sigma': 1.0,
                    'kernel_size': 5,
                    'boundary': 'mirror',
                    'crop_size': None,
                    'crop_box': None
                },
                'validation': {
                    'check_val_every_n_epoch': 5,
                    'val_check_interval': 0.5,
                    'metrics': ['mse', 'mae', 'rel_l2', 'psnr', 'ssim', 'temporal_mse', 'long_term_stability']
                },
                'performance_monitoring': {
                    'enabled': True,
                    'report_interval_seconds': 30,
                    'gpu_low_threshold': 0.90,
                    'iowait_high_threshold': 0.12,
                    'cpu_low_threshold': 0.80,
                    'num_workers_step': 4,
                    'prefetch_factor_step': 2,
                    'batch_size_step': 8
                },
                'hardware': {
                    'num_workers': 0,
                    'pin_memory': False,
                    'persistent_workers': False
                },
                'testing': {
                    'enabled': True,
                    'run_final_test': True,
                    'batch_size': 1
                },
                'paper_package': {
                    'auto_generate': True
                },
            })

        # 保守默认：仅在缺失时提供安全参数，优先尊重外部YAML配置
        try:
            # DataLoader 默认（小批量、低并发，避免OOM与初始化问题）
            if not hasattr(self.config.data, 'dataloader') or getattr(self.config.data, 'dataloader') is None:
                bs_default = int(getattr(getattr(self.config, 'training', DictConfig({})), 'batch_size', 8))
                self.config.data.dataloader = DictConfig({
                    'batch_size': bs_default,
                    'val_batch_size': bs_default,
                    'test_batch_size': 1,
                    'num_workers': 0,
                    'pin_memory': False,
                    'persistent_workers': False,
                    'prefetch_factor': None,
                    'drop_last': True,
                    'shuffle': True,
                })
            else:
                dl = self.config.data.dataloader
                dl.batch_size = int(getattr(dl, 'batch_size', getattr(self.config.training, 'batch_size', 8)))
                dl.val_batch_size = int(getattr(dl, 'val_batch_size', dl.batch_size))
                dl.test_batch_size = int(getattr(dl, 'test_batch_size', 1))
                # 并发相关仅在未配置时提供保守默认
                dl.num_workers = int(getattr(dl, 'num_workers', getattr(self.config, 'hardware', DictConfig({})).get('num_workers', 0)))
                dl.pin_memory = bool(getattr(dl, 'pin_memory', getattr(getattr(self.config, 'hardware', DictConfig({})), 'pin_memory', False)))
                # 当 num_workers==0 时禁用持久化与预取
                dl.persistent_workers = bool(getattr(dl, 'persistent_workers', False if int(getattr(dl, 'num_workers', 0)) <= 0 else True))
                dl.prefetch_factor = (None if int(getattr(dl, 'num_workers', 0)) <= 0 else int(getattr(dl, 'prefetch_factor', 2)))
                dl.drop_last = bool(getattr(dl, 'drop_last', True))
                dl.shuffle = bool(getattr(dl, 'shuffle', True))

            # 训练与调度：仅在缺省时设置保守默认
            if not hasattr(self.config, 'training') or getattr(self.config, 'training') is None:
                self.config.training = DictConfig({'epochs': 15, 'batch_size': 8, 'scheduler': {'T_max': 15}})
            else:
                self.config.training.epochs = int(getattr(self.config.training, 'epochs', 15))
                self.config.training.batch_size = int(getattr(self.config.training, 'batch_size', getattr(self.config.data.dataloader, 'batch_size', 8)))
                if hasattr(self.config.training, 'scheduler'):
                    try:
                        self.config.training.scheduler.T_max = int(getattr(self.config.training.scheduler, 'T_max', self.config.training.epochs))
                    except Exception:
                        pass

            # 硬件并行默认：仅在缺省时设置保守默认
            if not hasattr(self.config, 'hardware') or getattr(self.config, 'hardware') is None:
                self.config.hardware = DictConfig({'num_workers': 0, 'pin_memory': False, 'persistent_workers': False})
            else:
                self.config.hardware.num_workers = int(getattr(self.config.hardware, 'num_workers', 0))
                self.config.hardware.pin_memory = bool(getattr(self.config.hardware, 'pin_memory', False))
                self.config.hardware.persistent_workers = bool(getattr(self.config.hardware, 'persistent_workers', False))

            # 合成数据规模（若真实数据不可用）
            if not hasattr(self.config.data, 'max_samples'):
                self.config.data.max_samples = 512
        except Exception:
            pass
        
        # 设置随机种子
        torch.manual_seed(self.config.experiment.seed)
        np.random.seed(self.config.experiment.seed)
        
    def _cfg_select(self, *keys, default=None):
        """安全选择配置中的第一个非空值，统一OmegaConf.select的默认写法"""
        try:
            for key in keys:
                if key is None:
                    continue
                val = OmegaConf.select(self.config, key, default=None)
                if val is not None:
                    return val
        except Exception:
            pass
        return default
        
    def setup_logging(self):
        """设置日志"""
        self.output_dir = Path(self.config.experiment.output_dir) / self.config.experiment.name
        # 目录创建允许并发；由所有rank执行无害
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        # 依据环境变量判定主进程，避免依赖尚未调用的 setup_device
        log_file_path = None
        try:
            env_local_rank = os.environ.get("LOCAL_RANK")
            env_rank = os.environ.get("RANK")
            rank_val = int(env_local_rank if env_local_rank is not None else (env_rank if env_rank is not None else "0"))
        except Exception:
            rank_val = 0
        is_primary = (rank_val == 0)
        self.is_primary = is_primary
        if is_primary:
            log_file_path = self.output_dir / "training.log"
        else:
            # 非主进程写入独立日志文件，避免文件句柄并发冲突
            log_file_path = self.output_dir / f"training_rank{rank_val}.log"

        self.logger = setup_logger(
            name="RealDataARTrainer",
            log_file=log_file_path,
            level=logging.INFO
        )
        
        self.logger.info(f"输出目录: {self.output_dir}")
        
        # TensorBoard：仅主进程创建，避免事件文件并发冲突
        self.writer = None
        if is_primary:
            try:
                self.writer = SummaryWriter(self.output_dir / "tensorboard")
            except Exception as _tb_err:
                # 不中断训练，记录并继续
                self.logger.warning(f"TensorBoard创建失败（继续训练）: {_tb_err}")

        # 保存合并后的配置快照（仅主进程），满足黄金法则与复现要求
        if is_primary:
            try:
                merged_yaml = OmegaConf.to_yaml(self.config)
                cfg_snapshot = self.output_dir / "config_merged.yaml"
                with open(cfg_snapshot, 'w') as f:
                    f.write(merged_yaml)
                self.logger.info(f"📝 已保存配置快照: {cfg_snapshot}")
            except Exception as _cfg_err:
                self.logger.warning(f"⚠️ 配置快照保存失败: {_cfg_err}")
        
    def setup_device(self):
        """设置设备 - 支持多GPU，并启用TF32/cuDNN与CPU线程优化"""
        # 在DDP初始化前设置关键NCCL稳定性环境变量
        try:
            os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
            os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
            # 避免NCCL在某些环境下的Socket连接问题
            os.environ.setdefault('NCCL_SOCKET_NTHREADS', '4')
            os.environ.setdefault('NCCL_NSOCKS_PERTHREAD', '4')
            # 当网络IB不可用时强制使用PCIe
            os.environ.setdefault('NCCL_IB_DISABLE', '1')
            # 提升超时，避免大批次初始化阶段误判超时
            os.environ.setdefault('NCCL_BLOCKING_WAIT_TIMEOUT', '600')
            # 自动检测主机网络接口并设置 NCCL/GLOO 的 IFNAME（优先选择处于 up 状态的以太网接口）
            try:
                import glob
                def _get_up_ifnames():
                    candidates = []
                    for path in glob.glob('/sys/class/net/*'):
                        name = os.path.basename(path)
                        try:
                            with open(os.path.join(path, 'operstate'), 'r') as f:
                                state = f.read().strip()
                        except Exception:
                            state = ''
                        if state == 'up':
                            candidates.append(name)
                    return candidates
                up_ifaces = _get_up_ifnames()
                # 过滤掉虚拟/环回接口，优先 eno*/eth* 其次 ib*
                preferred = [n for n in up_ifaces if n.startswith(('eno', 'eth'))]
                if not preferred:
                    preferred = [n for n in up_ifaces if n.startswith('ib')]
                # 兜底：如果没有 up 状态接口，则不覆盖已有设置
                if preferred:
                    ifname = preferred[0]
                    os.environ.setdefault('NCCL_SOCKET_IFNAME', ifname)
                    os.environ.setdefault('GLOO_SOCKET_IFNAME', ifname)
            except Exception:
                pass
            # 设置本地主从地址端口（仅当未由 torch.run 设定时）
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29500')
        except Exception:
            pass
        # 设备选择（统一规范：'gpu'→'cuda'；优先 device.accelerator 其次 experiment.device）
        try:
            raw_device = str(self._cfg_select('device.accelerator', 'experiment.device', default='cuda')).lower()
        except Exception:
            raw_device = 'cuda'
        # 归一化映射
        if raw_device in ('gpu', 'cuda'):
            normalized = 'cuda'
        elif raw_device in ('cpu',):
            normalized = 'cpu'
        elif raw_device in ('mps',):
            normalized = 'mps'
        else:
            # 未知设备类型，保守回退到cpu
            normalized = 'cpu'

        if normalized == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif normalized == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        # 记录设备选择过程，便于诊断
        try:
            self.logger.info(f"设备选择: raw='{raw_device}', normalized='{normalized}', final='{self.device}'")
        except Exception:
            pass

        # DDP 初始化（环境变量 WORLD_SIZE/RANK 存在时）
        self.distributed = False
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            # 从环境获取rank并更新is_primary标记
            try:
                env_local_rank = os.environ.get("LOCAL_RANK")
                env_rank = os.environ.get("RANK")
                rank_val = int(env_local_rank if env_local_rank is not None else (env_rank if env_rank is not None else "0"))
            except Exception:
                rank_val = 0
            self.is_primary = (rank_val == 0)
            if world_size > 1:
                backend = 'nccl' if self.device.type == 'cuda' else 'gloo'
                try:
                    dist.init_process_group(backend=backend)
                    self.distributed = True
                    self.local_rank = rank_val
                    if self.device.type == 'cuda':
                        torch.cuda.set_device(self.local_rank % max(1, torch.cuda.device_count()))
                        self.device = torch.device(f"cuda:{self.local_rank % max(1, torch.cuda.device_count())}")
                    self.logger.info(f"DDP已初始化: backend={backend}, rank={self.local_rank}, world_size={dist.get_world_size()}")
                except Exception as _ddp_err:
                    # NCCL失败时，尝试回退到GLOO后端
                    if backend == 'nccl':
                        self.logger.warning(f"NCCL初始化失败，回退到GLOO后端: {_ddp_err}")
                        try:
                            os.environ.setdefault('GLOO_SOCKET_IFNAME', os.environ.get('NCCL_SOCKET_IFNAME', 'lo'))
                            dist.init_process_group(backend='gloo')
                            self.distributed = True
                            self.local_rank = rank_val
                            # GLOO也支持CUDA张量，但性能较低；设备保持不变
                            self.logger.info(f"DDP已初始化: backend=gloo, rank={self.local_rank}, world_size={dist.get_world_size()}")
                        except Exception as _gloo_err:
                            self.logger.error(f"GLOO初始化也失败，回退到非分布式: {_gloo_err}")
                            self.distributed = False
                    else:
                        self.logger.error(f"DDP初始化失败，回退到非分布式: {_ddp_err}")
        except Exception as e:
            self.logger.warning(f"DDP初始化失败，回退到非分布式: {e}")

        # CPU线程与库线程数设置（根据hardware.*与hardware.cpu.*）
        try:
            import os as _os
            torch_threads = int(self._cfg_select('hardware.cpu.torch_threads', 'hardware.torch_threads', default=0) or 0)
            mkl_threads = int(self._cfg_select('hardware.cpu.mkl_threads', 'hardware.mkl_threads', default=0) or 0)
            omp_threads = int(self._cfg_select('hardware.cpu.omp_threads', 'hardware.omp_threads', default=0) or 0)
            numexpr_threads = int(self._cfg_select('hardware.cpu.numexpr_threads', default=0) or 0)
            interop_threads = int(self._cfg_select('hardware.cpu.interop_threads', 'hardware.interop_threads', default=0) or 0)
            openblas_threads = int(self._cfg_select('hardware.cpu.blas_threads', default=0) or 0)
            if torch_threads > 0:
                torch.set_num_threads(torch_threads)
            if interop_threads > 0 and hasattr(torch, 'set_num_interop_threads'):
                try:
                    torch.set_num_interop_threads(interop_threads)
                except Exception:
                    pass
            if mkl_threads > 0:
                _os.environ['MKL_NUM_THREADS'] = str(mkl_threads)
            if omp_threads > 0:
                _os.environ['OMP_NUM_THREADS'] = str(omp_threads)
            if numexpr_threads > 0:
                _os.environ['NUMEXPR_NUM_THREADS'] = str(numexpr_threads)
            if openblas_threads > 0:
                _os.environ['OPENBLAS_NUM_THREADS'] = str(openblas_threads)
            self.logger.info(f"CPU线程设置: torch={torch_threads}, interop={interop_threads}, MKL={mkl_threads}, OMP={omp_threads}, numexpr={numexpr_threads}, openblas={openblas_threads}")
        except Exception as e:
            self.logger.warning(f"CPU线程设置失败: {e}")

        # 多GPU配置
        self.use_multi_gpu = False
        if self.device.type == 'cuda':
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"检测到 {gpu_count} 张GPU")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f} GB")
            # 如果有多张GPU且配置允许，启用多GPU训练
            if gpu_count > 1 and hasattr(self.config, 'device') and getattr(self.config.device, 'devices', 1) > 1:
                self.use_multi_gpu = True
                self.logger.info(f"启用多GPU训练，使用 {getattr(self.config.device, 'devices', gpu_count)} 张GPU")
            else:
                self.logger.info(f"使用单GPU训练: {self.device}")

            # 启用TF32与cuDNN优化
            try:
                allow_tf32 = bool(self._cfg_select('hardware.memory.allow_tf32', default=False))
                cudnn_bench = bool(self._cfg_select('hardware.memory.cudnn_benchmark', default=False))
                if allow_tf32:
                    torch.set_float32_matmul_precision('medium')
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = cudnn_bench
                # 保持默认确定性配置，由配置决定是否开启benchmark
                self.logger.info(f"TF32: {allow_tf32}, cuDNN benchmark: {cudnn_bench}")
            except Exception as e:
                self.logger.warning(f"TF32/cuDNN优化设置失败: {e}")
        else:
            self.logger.info(f"使用设备: {self.device}")
    
    def setup_memory_management(self):
        """设置内存管理"""
        # 内存管理配置
        self.memory_config = {
            'gradient_accumulation_steps': getattr(self.config.training, 'gradient_accumulation_steps', 1),
            'memory_cleanup_frequency': getattr(self.config.training, 'memory_cleanup_frequency', 10),
            'auto_batch_size_reduction': getattr(self.config.training, 'auto_batch_size_reduction', True),
            'memory_threshold': getattr(self.config.training, 'memory_threshold', 0.9),  # 90%显存使用率阈值
        }
        
        # 线程与环境变量配置（从YAML获取，支持极限CPU/RAM设置）
        try:
            torch_threads = int(self._cfg_select('hardware.cpu.torch_threads', 'hardware.torch_threads', default=0) or 0)
            mkl_threads = int(self._cfg_select('hardware.cpu.mkl_threads', 'hardware.mkl_threads', default=0) or 0)
            omp_threads = int(self._cfg_select('hardware.cpu.omp_threads', 'hardware.omp_threads', default=0) or 0)
            numexpr_threads = int(self._cfg_select('hardware.cpu.numexpr_threads', default=0) or 0)
            if torch_threads > 0:
                try:
                    torch.set_num_threads(torch_threads)
                except Exception:
                    pass
            if mkl_threads > 0:
                os.environ['MKL_NUM_THREADS'] = str(mkl_threads)
            if omp_threads > 0:
                os.environ['OMP_NUM_THREADS'] = str(omp_threads)
            if numexpr_threads > 0:
                os.environ['NUMEXPR_NUM_THREADS'] = str(numexpr_threads)
        except Exception as e:
            self.logger.warning(f"线程/环境变量设置失败: {e}")
        
        # 设置CUDA内存与TF32/cuDNN性能开关
        if self.device.type == 'cuda':
            # 启用内存池
            torch.cuda.empty_cache()
            # 设置内存分配策略（使用expandable_segments，兼容当前PyTorch版本）
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
            # NCCL稳定性与错误处理（单机多卡）
            os.environ['TORCH_NCCL_BLOCKING_WAIT'] = os.environ.get('TORCH_NCCL_BLOCKING_WAIT', '1')
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = os.environ.get('NCCL_ASYNC_ERROR_HANDLING', '1')
            os.environ['NCCL_BLOCKING_WAIT'] = os.environ.get('NCCL_BLOCKING_WAIT', '1')
            os.environ['NCCL_DEBUG'] = os.environ.get('NCCL_DEBUG', 'WARN')
            os.environ['NCCL_IB_DISABLE'] = os.environ.get('NCCL_IB_DISABLE', '1')  # 单机默认禁用IB，避免误配置
            # 显式启用阻塞等待，提升稳定性
            os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
            
            # TF32与cuDNN性能优化
            try:
                from omegaconf import OmegaConf
                allow_tf32 = bool(self._cfg_select("hardware.memory.allow_tf32", default=True))
                cudnn_bench = bool(self._cfg_select("hardware.memory.cudnn_benchmark", default=True))
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.allow_tf32 = allow_tf32
                    torch.backends.cudnn.benchmark = cudnn_bench
                # Matmul精度（中档可启用TF32路径）
                try:
                    torch.set_float32_matmul_precision('medium' if allow_tf32 else 'high')
                except Exception:
                    pass
                # 统一设置AMP自动转换默认dtype（优先BF16）
                try:
                    amp_cfg = getattr(self.config.training, 'amp', None)
                    dtype_name = None
                    if amp_cfg is not None:
                        dtype_name = getattr(amp_cfg, 'autocast_dtype', None) or getattr(amp_cfg, 'cast_model_type', None)
                    if dtype_name:
                        dtype_str = str(dtype_name).lower()
                        if 'bf16' in dtype_str or 'bfloat16' in dtype_str:
                            if hasattr(torch, 'set_autocast_gpu_dtype'):
                                torch.set_autocast_gpu_dtype(torch.bfloat16)
                                self.logger.info("AMP autocast dtype 设置为 BF16")
                        elif 'fp16' in dtype_str or 'float16' in dtype_str:
                            if hasattr(torch, 'set_autocast_gpu_dtype'):
                                torch.set_autocast_gpu_dtype(torch.float16)
                                self.logger.info("AMP autocast dtype 设置为 FP16")
                except Exception as _amp_err:
                    self.logger.warning(f"AMP dtype 设置失败: {_amp_err}")
                self.logger.info(f"TF32: {allow_tf32}, cuDNN benchmark: {cudnn_bench}")
            except Exception as e:
                self.logger.warning(f"TF32/cuDNN设置失败: {e}")
            
        # 选择 AMP autocast dtype（float16/bfloat16），缺省为 None 由 PyTorch 决定
        autocast_dtype = None
        try:
            cast_type = str(self._cfg_select('training.amp.cast_model_type', 'device.precision', 'training.precision', default=''))
            cast_type_l = cast_type.lower()
            if 'bf16' in cast_type_l or 'bfloat16' in cast_type_l:
                autocast_dtype = torch.bfloat16
            elif '16' in cast_type_l or 'fp16' in cast_type_l or 'float16' in cast_type_l:
                autocast_dtype = torch.float16
        except Exception:
            autocast_dtype = None
        self.autocast_dtype = autocast_dtype

        self.logger.info(f"内存管理配置: {self.memory_config}, AMP dtype: {('default' if autocast_dtype is None else ('bfloat16' if autocast_dtype is torch.bfloat16 else 'float16'))}")
        
    def check_memory_usage(self) -> float:
        """检查GPU内存使用率"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            usage_ratio = allocated / total
            
            return usage_ratio
        return 0.0
    
    def cleanup_memory(self):
        """清理GPU内存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    def setup_data(self):
        """设置数据"""
        self.logger.info("设置数据模块...")
        
        try:
            # 获取批次大小配置，优先使用dataloader中的配置
            batch_size = int(self._cfg_select('data.dataloader.batch_size', 'training.batch_size', default=128))
            
            # 获取验证批次大小配置
            val_batch_size = int(self._cfg_select('data.dataloader.val_batch_size', default=batch_size))
            
            # 获取测试批次大小配置
            test_batch_size = int(self._cfg_select('data.dataloader.test_batch_size', 'testing.batch_size', default=1))

            # 统一保护：num_workers==0 时禁用 prefetch_factor 并关闭 persistent_workers，避免 DataLoader 冲突
            try:
                num_workers = int(self._cfg_select('data.dataloader.num_workers', 'hardware.num_workers', default=32) or 32)
                if num_workers == 0 and hasattr(self.config, 'data') and hasattr(self.config.data, 'dataloader'):
                    self.config.data.dataloader.prefetch_factor = None
                    self.config.data.dataloader.persistent_workers = False
                    self.logger.info("num_workers=0: 已将 prefetch_factor=None 且 persistent_workers=False")
                elif num_workers > 0 and hasattr(self.config, 'data') and hasattr(self.config.data, 'dataloader'):
                    # 当可并行时，保持用户的 pin_memory 设置，仅启用持久workers并合理提升预取
                    try:
                        current_pin = bool(getattr(self.config.data.dataloader, 'pin_memory', False))
                    except Exception:
                        current_pin = False
                    self.config.data.dataloader.persistent_workers = True
                    # 统一设置更高预取因子（若未设置或为0则提升到16）
                    prefetch_cfg = getattr(self.config.data.dataloader, 'prefetch_factor', None)
                    if prefetch_cfg in (None, 0):
                        self.config.data.dataloader.prefetch_factor = 16
                    self.logger.info(f"num_workers={num_workers}: 保持 pin_memory={current_pin}，启用 persistent_workers=True，预取因子={self.config.data.dataloader.prefetch_factor}")
            except Exception as e:
                self.logger.warning(f"设置 prefetch_factor 保护失败: {e}")
            
            # 记录使用的批次大小
            self.logger.info(f"使用训练批次大小: {batch_size}")
            self.logger.info(f"使用验证批次大小: {val_batch_size}")
            self.logger.info(f"使用测试批次大小: {test_batch_size}")
            
            # 使用新版本的数据模块，传入完整配置
            self.data_module = RealDiffusionReactionDataModule(self.config)
            using_dm = True
            try:
                self.data_module.setup()
                # 获取数据加载器（若数据模块内部强制num_workers=0，则在此处重建以支持并行加载）
                dm_train = self.data_module.train_dataloader()
                dm_val = self.data_module.val_dataloader()
                dm_test = self.data_module.test_dataloader()
            except Exception as e:
                self.logger.warning(f"数据模块setup失败，启用合成数据回退: {e}")
                using_dm = False
                # 合成数据集：匹配配置的时序与空间维度
                class SyntheticARSequenceDataset(torch.utils.data.Dataset):
                    def __init__(self, n=4096, T_in=1, T_out=20, C=2, H=128, W=128, seed=2025):
                        self.n = n
                        self.T_in = T_in
                        self.T_out = T_out
                        self.C = C
                        self.H = H
                        self.W = W
                        torch.manual_seed(seed)
                    def __len__(self):
                        return self.n
                    def __getitem__(self, idx):
                        input_seq = torch.randn(self.T_in, self.C, self.H, self.W)
                        target_seq = torch.randn(self.T_out, self.C, self.H, self.W)
                        return {
                            'input_sequence': input_seq,
                            'target_sequence': target_seq,
                            'sample_idx': idx,
                            'start_time': 0
                        }
                T_in = int(self._cfg_select('data.T_in', default=1))
                T_out = int(self._cfg_select('data.T_out', default=20))
                C = int(self._cfg_select('model.out_channels', default=2))
                H = int(self._cfg_select('model.img_size', default=128))
                W = H
                synth_n = int(self._cfg_select('data.max_samples', default=4096) or 4096)
                seed = int(self._cfg_select('experiment.seed', default=2025))
                synth_ds = SyntheticARSequenceDataset(n=synth_n, T_in=T_in, T_out=T_out, C=C, H=H, W=W, seed=seed)
                # 划分训练/验证/测试
                n_train = int(synth_n * 0.7)
                n_val = int(synth_n * 0.15)
                self.train_dataset = torch.utils.data.Subset(synth_ds, range(0, n_train))
                self.val_dataset = torch.utils.data.Subset(synth_ds, range(n_train, n_train + n_val))
                self.test_dataset = torch.utils.data.Subset(synth_ds, range(n_train + n_val, synth_n))
                dm_train = dm_val = dm_test = None
            
            # 提取底层Dataset并重建DataLoader以应用data.dataloader配置
            try:
                from torch.utils.data import DataLoader as _DL
                dl_cfg = getattr(self.config.data, 'dataloader', None)
                # 若未提供data.dataloader，则构造安全默认配置以便重建DataLoader
                if dl_cfg is None:
                    try:
                        default_num_workers = int(self._cfg_select('hardware.num_workers', default=0) or 0)
                        default_pin_memory = bool(self._cfg_select('hardware.pin_memory', default=False))
                        default_test_bs = int(self._cfg_select('data.dataloader.test_batch_size', 'testing.batch_size', default=1))
                    except Exception:
                        default_num_workers, default_pin_memory, default_test_bs = 0, False, 1
                    from omegaconf import DictConfig
                    dl_cfg = DictConfig({
                        'num_workers': default_num_workers,
                        'pin_memory': default_pin_memory,
                        'persistent_workers': False,
                        'prefetch_factor': None,
                        'drop_last': True,
                        'shuffle': True,
                        'val_batch_size': batch_size,
                        'test_batch_size': default_test_bs,
                    })
                if dl_cfg is not None:
                    num_workers = int(getattr(dl_cfg, 'num_workers', 12))
                    pin_memory = bool(getattr(dl_cfg, 'pin_memory', True))
                    persistent_workers = bool(getattr(dl_cfg, 'persistent_workers', True)) and num_workers > 0
                    prefetch_factor = int(getattr(dl_cfg, 'prefetch_factor', 4)) if num_workers > 0 else None
                    drop_last = bool(getattr(dl_cfg, 'drop_last', True))
                    shuffle = bool(getattr(dl_cfg, 'shuffle', True))
                    mp_ctx_opt = getattr(dl_cfg, 'multiprocessing_context', None)
                    timeout_opt = int(getattr(dl_cfg, 'timeout', 0) or 0)
                    # 训练/验证/测试的batch_size
                    train_bs = batch_size
                    val_bs = int(getattr(dl_cfg, 'val_batch_size', train_bs))
                    test_bs = int(getattr(dl_cfg, 'test_batch_size', 1))
                    # 取出底层dataset（优先DataModule，其次训练器的合成数据回退）
                    train_ds = getattr(self.data_module, 'train_dataset', None)
                    val_ds = getattr(self.data_module, 'val_dataset', None)
                    test_ds = getattr(self.data_module, 'test_dataset', None)
                    if train_ds is None and hasattr(self, 'train_dataset'):
                        train_ds = getattr(self, 'train_dataset', None)
                    if val_ds is None and hasattr(self, 'val_dataset'):
                        val_ds = getattr(self, 'val_dataset', None)
                    if test_ds is None and hasattr(self, 'test_dataset'):
                        test_ds = getattr(self, 'test_dataset', None)
                    # 若DataModule未暴露dataset，则从默认DataLoader中提取
                    if (train_ds is None or val_ds is None or test_ds is None) and (dm_train is not None and dm_val is not None and dm_test is not None):
                        try:
                            train_ds = dm_train.dataset if train_ds is None else train_ds
                            val_ds = dm_val.dataset if val_ds is None else val_ds
                            test_ds = dm_test.dataset if test_ds is None else test_ds
                            self.logger.info("🔧 从默认DataLoader提取底层dataset用于重建")
                        except Exception:
                            pass
                    # 移除CPU燃烧包装：保持干净的数据加载，避免额外CPU占用与阻塞
                    self.logger.info("ℹ️ 已禁用CPU燃烧包装，直接使用原始Dataset")
                    if train_ds is not None and val_ds is not None and test_ds is not None:
                        # DDP下使用DistributedSampler
                        if getattr(self, 'distributed', False):
                            # 为 train/val 显式创建 DistributedSampler，绑定 world_size/rank
                            world_size = dist.get_world_size() if dist.is_initialized() else 1
                            rank = dist.get_rank() if dist.is_initialized() else 0
                            sampler_train = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
                            sampler_val = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
                        else:
                            sampler_train = None
                            sampler_val = None
                        self.train_sampler = sampler_train
                        self.val_sampler = sampler_val
                    # 兼容PyTorch新版本：仅在num_workers>0时传入prefetch_factor
                    dl_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=drop_last)
                    # 预先初始化 supports_pmd，避免在未进入下面分支时引用未定义变量
                    supports_pmd = False
                    if num_workers > 0 and prefetch_factor is not None:
                        dl_kwargs['prefetch_factor'] = prefetch_factor
                        # multiprocessing_context: 支持'fork'/'spawn'/'forkserver'或直接传入上下文
                        if mp_ctx_opt is not None:
                            try:
                                import multiprocessing as mp
                                if isinstance(mp_ctx_opt, str) and mp_ctx_opt:
                                    dl_kwargs['multiprocessing_context'] = mp.get_context(mp_ctx_opt)
                                else:
                                    dl_kwargs['multiprocessing_context'] = mp_ctx_opt
                            except Exception:
                                pass
                        # timeout（秒）
                        if timeout_opt > 0:
                            dl_kwargs['timeout'] = timeout_opt
                        # 根据DataLoader签名条件性传入pin_memory_device，避免旧版本报错
                        try:
                            import inspect
                            sig = inspect.signature(_DL.__init__)
                            supports_pmd = ('pin_memory_device' in sig.parameters)
                        except Exception:
                            supports_pmd = False
                    if supports_pmd and pin_memory:
                        try:
                            if torch.cuda.is_available():
                                # 使用明确的 cuda:{index}
                                if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                                    dev_index = 0 if (self.device.index is None) else int(self.device.index)
                                elif getattr(self, 'distributed', False):
                                    dev_index = int(getattr(self, 'local_rank', 0))
                                else:
                                    dev_index = 0
                                dl_kwargs['pin_memory_device'] = f"cuda:{dev_index}"
                            else:
                                dl_kwargs['pin_memory_device'] = 'cpu'
                        except Exception:
                            dl_kwargs['pin_memory_device'] = 'cpu'
                    else:
                        if not supports_pmd:
                            self.logger.info("DataLoader不支持pin_memory_device参数，已跳过该设置")
                        elif not pin_memory:
                            self.logger.info("pin_memory=False，跳过pin_memory_device设置")
                    # 调试兼容修复：若pin_memory_device未设置，关闭pin_memory以避免len(None)错误
                    if dl_kwargs.get('pin_memory', False) and ('pin_memory_device' not in dl_kwargs or dl_kwargs.get('pin_memory_device') is None):
                        self.logger.info("调试兼容：检测到pin_memory_device缺失，禁用pin_memory避免迭代错误")
                        dl_kwargs['pin_memory'] = False
                        dl_kwargs.pop('pin_memory_device', None)
                        # generator与worker_init_fn：提高复现性与CPU并行稳定性
                        base_seed = int(self._cfg_select('experiment.seed', default=2025))
                        data_gen = torch.Generator(device='cpu')
                        try:
                            data_gen.manual_seed(base_seed)
                        except Exception:
                            pass
                        dl_kwargs['generator'] = data_gen
                        # 在DataLoader中绑定安全collate与worker_init_fn
                        # 在 DataLoader 中绑定安全collate，过滤None样本，避免default_collate异常
                        dl_collate = (
                            fast_collate_fn if ('fast_collate_fn' in globals() and fast_collate_fn is not None)
                            else (safe_collate_fn if ('safe_collate_fn' in globals() and safe_collate_fn is not None) else None)
                        )
                        # 创建DataLoader，若因pin_memory_device产生TypeError则回退移除此参数重建
                        try:
                            self.train_loader = _DL(
                                train_ds,
                                batch_size=train_bs,
                                sampler=sampler_train,
                                shuffle=(sampler_train is None and shuffle),
                                collate_fn=dl_collate,
                                worker_init_fn=partial(seed_worker_fn, base_seed=base_seed) if num_workers > 0 else None,
                                **dl_kwargs,
                            )
                        except TypeError as e:
                            if 'pin_memory_device' in str(e):
                                self.logger.info("移除pin_memory_device后重建train_loader以兼容旧版PyTorch")
                                dl_kwargs.pop('pin_memory_device', None)
                                self.train_loader = _DL(
                                    train_ds,
                                    batch_size=train_bs,
                                    sampler=sampler_train,
                                    shuffle=(sampler_train is None and shuffle),
                                    collate_fn=dl_collate,
                                    worker_init_fn=partial(seed_worker_fn, base_seed=base_seed) if num_workers > 0 else None,
                                    **dl_kwargs,
                                )
                            else:
                                raise
                        try:
                            val_kwargs = {**dl_kwargs, 'drop_last': False}
                            self.val_loader = _DL(
                                val_ds,
                                batch_size=val_bs,
                                sampler=sampler_val,
                                shuffle=False,
                                collate_fn=dl_collate,
                                worker_init_fn=partial(seed_worker_fn, base_seed=base_seed) if num_workers > 0 else None,
                                **val_kwargs,
                            )
                        except TypeError as e:
                            if 'pin_memory_device' in str(e):
                                self.logger.info("移除pin_memory_device后重建val_loader以兼容旧版PyTorch")
                                val_kwargs.pop('pin_memory_device', None)
                                self.val_loader = _DL(
                                    val_ds,
                                    batch_size=val_bs,
                                    sampler=sampler_val,
                                    shuffle=False,
                                    collate_fn=dl_collate,
                                    worker_init_fn=partial(seed_worker_fn, base_seed=base_seed) if num_workers > 0 else None,
                                    **val_kwargs,
                                )
                            else:
                                raise
                        # 强制修复DataLoader的pin_memory_device为空导致迭代报错的问题
                        for _name, _dl in (('train', getattr(self, 'train_loader', None)),
                                           ('val', getattr(self, 'val_loader', None))):
                            try:
                                if _dl is not None and hasattr(_dl, 'pin_memory_device'):
                                    _pmd = getattr(_dl, 'pin_memory_device', None)
                                    if _pmd is None or (isinstance(_pmd, str) and len(_pmd) == 0):
                                        if torch.cuda.is_available():
                                            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                                                dev_index = 0 if (self.device.index is None) else int(self.device.index)
                                            elif getattr(self, 'distributed', False):
                                                dev_index = int(getattr(self, 'local_rank', 0))
                                            else:
                                                dev_index = 0
                                            setattr(_dl, 'pin_memory_device', f"cuda:{dev_index}")
                                        else:
                                            setattr(_dl, 'pin_memory_device', 'cpu')
                            except Exception:
                                pass
                        try:
                            test_kwargs = {**dl_kwargs, 'drop_last': False}
                            self.test_loader = _DL(
                                test_ds,
                                batch_size=test_bs,
                                shuffle=False,
                                collate_fn=dl_collate,
                                worker_init_fn=partial(seed_worker_fn, base_seed=base_seed) if num_workers > 0 else None,
                                **test_kwargs,
                            )
                        except TypeError as e:
                            if 'pin_memory_device' in str(e):
                                self.logger.info("移除pin_memory_device后重建test_loader以兼容旧版PyTorch")
                                test_kwargs.pop('pin_memory_device', None)
                                self.test_loader = _DL(
                                    test_ds,
                                    batch_size=test_bs,
                                    shuffle=False,
                                    collate_fn=dl_collate,
                                    worker_init_fn=partial(seed_worker_fn, base_seed=base_seed) if num_workers > 0 else None,
                                    **test_kwargs,
                                )
                        except Exception as e:
                            # 其他异常直接抛出，避免静默失败
                            raise
                        # 日志：重建DataLoader配置
                        self.logger.info(
                            f"🔧 重建DataLoader: num_workers={num_workers}, pin_memory={pin_memory}, "
                            f"prefetch_factor={prefetch_factor if (num_workers>0 and prefetch_factor is not None) else 'N/A'}, "
                            f"pin_memory_device={(dl_kwargs.get('pin_memory_device') if (pin_memory and 'pin_memory_device' in dl_kwargs) else 'skip')}, "
                            f"persistent_workers={persistent_workers}, timeout={timeout_opt}, "
                            f"multiprocessing_context={'set' if ('multiprocessing_context' in dl_kwargs and dl_kwargs['multiprocessing_context'] is not None) else 'default'}, "
                            f"sampler={'DDP(train/val)' if sampler_train else 'None'}"
                        )
                    else:
                        # 若仍无法获取dataset，则回退到DataModule默认DataLoader（若它们也为None则抛错）
                        self.train_loader, self.val_loader, self.test_loader = dm_train, dm_val, dm_test
                        if any(dl is None for dl in (self.train_loader, self.val_loader, self.test_loader)):
                            raise RuntimeError("无法获取底层dataset且默认DataLoader不可用")
                        self.logger.warning("⚠️ 无法提取底层dataset，使用数据模块的默认DataLoader")
                else:
                    self.train_loader, self.val_loader, self.test_loader = dm_train, dm_val, dm_test
            except Exception as e:
                self.logger.warning(f"⚠️ 重建DataLoader失败，回退到默认: {e}")
                self.train_loader, self.val_loader, self.test_loader = dm_train, dm_val, dm_test
            # 若仍存在None的DataLoader，进行最终兜底重建，确保训练不因None中断
            if any(dl is None for dl in (self.train_loader, self.val_loader, self.test_loader)):
                self.logger.warning("⚠️ DataLoader仍为None，使用最小配置强制重建")
                try:
                    # 尝试从已有属性中获取dataset
                    train_ds_fb = getattr(self, 'train_dataset', None) or getattr(self.data_module, 'train_dataset', None)
                    val_ds_fb = getattr(self, 'val_dataset', None) or getattr(self.data_module, 'val_dataset', None)
                    test_ds_fb = getattr(self, 'test_dataset', None) or getattr(self.data_module, 'test_dataset', None)
                    # 如果仍为空且默认DataLoader存在，尝试取其dataset
                    if (train_ds_fb is None or val_ds_fb is None or test_ds_fb is None):
                        try:
                            if dm_train is not None and train_ds_fb is None:
                                train_ds_fb = getattr(dm_train, 'dataset', None)
                            if dm_val is not None and val_ds_fb is None:
                                val_ds_fb = getattr(dm_val, 'dataset', None)
                            if dm_test is not None and test_ds_fb is None:
                                test_ds_fb = getattr(dm_test, 'dataset', None)
                        except Exception:
                            pass
                    from torch.utils.data import DataLoader as _DL2
                    minimal_kwargs = dict(num_workers=0, pin_memory=False, persistent_workers=False)
                    dl_collate_fb = (
                        fast_collate_fn if ('fast_collate_fn' in globals() and fast_collate_fn is not None)
                        else (safe_collate_fn if ('safe_collate_fn' in globals() and safe_collate_fn is not None) else None)
                    )
                    if self.train_loader is None and train_ds_fb is not None:
                        self.train_loader = _DL2(train_ds_fb, batch_size=batch_size, shuffle=True, collate_fn=dl_collate_fb, **minimal_kwargs)
                    if self.val_loader is None and val_ds_fb is not None:
                        self.val_loader = _DL2(val_ds_fb, batch_size=int(self._cfg_select('data.dataloader.val_batch_size', default=batch_size)), shuffle=False, collate_fn=dl_collate_fb, **minimal_kwargs)
                    if self.test_loader is None and test_ds_fb is not None:
                        self.test_loader = _DL2(test_ds_fb, batch_size=int(self._cfg_select('data.dataloader.test_batch_size', 'testing.batch_size', default=1)), shuffle=False, collate_fn=dl_collate_fb, **minimal_kwargs)
                    if any(dl is None for dl in (self.train_loader, self.val_loader, self.test_loader)):
                        raise RuntimeError("最终兜底重建失败：仍有DataLoader为None")
                except Exception as e:
                    self.logger.error(f"❌ 兜底重建DataLoader失败: {e}")
                    raise
            
            # 存储原始批次大小用于动态调整
            self.original_batch_size = batch_size
            self.current_batch_size = batch_size

            # 统一修复：确保所有DataLoader的pin_memory_device为有效字符串，避免len(None)错误
            def _fix_pmd(loader):
                try:
                    if loader is None:
                        return
                    if hasattr(loader, 'pin_memory_device'):
                        pmd = getattr(loader, 'pin_memory_device', None)
                        if pmd is None or (isinstance(pmd, str) and len(pmd) == 0):
                            if torch.cuda.is_available():
                                if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                                    dev_index = 0 if (self.device.index is None) else int(self.device.index)
                                elif getattr(self, 'distributed', False):
                                    dev_index = int(getattr(self, 'local_rank', 0))
                                else:
                                    dev_index = 0
                                setattr(loader, 'pin_memory_device', f"cuda:{dev_index}")
                            else:
                                setattr(loader, 'pin_memory_device', 'cpu')
                except Exception:
                    pass

            _fix_pmd(self.train_loader)
            _fix_pmd(self.val_loader)
            _fix_pmd(self.test_loader)
            
            self.logger.info(f"训练集批次数: {len(self.train_loader)}")
            self.logger.info(f"验证集批次数: {len(self.val_loader)}")
            self.logger.info(f"测试集批次数: {len(self.test_loader)}")
            
            # 测试数据加载（兼容安全collate返回None的情况）
            sample_batch = None
            for _ in range(10):  # 最多尝试10次，跳过None批次
                sample_batch = next(iter(self.train_loader))
                if sample_batch is not None:
                    break
            if sample_batch is None:
                raise RuntimeError("DataLoader返回连续None批次，无法获取样本用于形状检查")
            self.logger.info(f"✅ 输入序列形状: {sample_batch['input_sequence'].shape}")
            self.logger.info(f"✅ 目标序列形状: {sample_batch['target_sequence'].shape}")

            # 观测算子与H参数设置（支持 config.observation 与 config.data.observation 两种路径，兼容嵌套 sr/crop 配置）
            obs_cfg = getattr(self.config, 'observation', None)
            if obs_cfg is None:
                try:
                    obs_cfg = getattr(self.config, 'data', None)
                    obs_cfg = getattr(obs_cfg, 'observation', None) if obs_cfg is not None else None
                except Exception:
                    obs_cfg = None
            self.h_params = None
            self.observation_op = None
            if obs_cfg is not None:
                # 兼容嵌套结构：obs_cfg 可能包含 {'mode': 'sr', 'sr': {...}} 或 {'mode': 'crop', 'crop': {...}}
                mode_raw = obs_cfg.get('mode', 'sr')
                mode = str(mode_raw[0] if isinstance(mode_raw, (list, tuple)) else mode_raw).lower()
                # 顶层通用边界键名，若未设置，尝试从子配置读取
                boundary = obs_cfg.get('boundary', obs_cfg.get('boundary_mode', 'mirror'))
                if mode == 'sr':
                    sr_sub = obs_cfg.get('sr', {}) if isinstance(obs_cfg.get('sr', {}), dict) else {}
                    # 从顶层或sr子配置中解析
                    scale = obs_cfg.get('scale_factor', sr_sub.get('scale_factor', 2))
                    sigma = obs_cfg.get('blur_sigma', sr_sub.get('blur_sigma', 1.0))
                    kernel_size = obs_cfg.get('kernel_size', sr_sub.get('blur_kernel_size', 5))
                    # 边界优先级：顶层 -> 子配置
                    boundary = boundary if boundary is not None else sr_sub.get('boundary_mode', 'mirror')
                    self.h_params = {
                        'task': 'sr',
                        'scale': scale,
                        'sigma': sigma,
                        'kernel_size': kernel_size,
                        'boundary': boundary
                    }
                    self.observation_op = SuperResolutionOperator(scale=scale, sigma=sigma, kernel_size=kernel_size, boundary=boundary)
                elif mode == 'crop':
                    crop_sub = obs_cfg.get('crop', {}) if isinstance(obs_cfg.get('crop', {}), dict) else {}
                    crop_size = obs_cfg.get('crop_size', crop_sub.get('crop_size', None))
                    crop_box = obs_cfg.get('crop_box', crop_sub.get('crop_box', None))
                    boundary = boundary if boundary is not None else crop_sub.get('boundary_mode', 'mirror')
                    self.h_params = {
                        'task': 'crop',
                        'crop_size': crop_size,
                        'crop_box': crop_box,
                        'boundary': boundary
                    }
                    self.observation_op = CropOperator(crop_size=crop_size, crop_box=crop_box, boundary=boundary)
                else:
                    self.logger.warning(f"未知的观测模式: {mode}，跳过观测算子初始化")
                    self.h_params = None
                    self.observation_op = None
                self.logger.info(f"✅ 观测算子配置: {self.h_params}")

            # 归一化统计量，用于反归一化到原值域
            self.norm_stats = None
            try:
                train_ds = getattr(self.data_module, 'train_dataset', None)
                if train_ds is not None and hasattr(train_ds, 'mean') and hasattr(train_ds, 'std'):
                    mean = train_ds.mean
                    std = train_ds.std
                    if isinstance(mean, torch.Tensor):
                        mean = mean.detach().cpu()
                    if isinstance(std, torch.Tensor):
                        std = std.detach().cpu()
                    self.norm_stats = {
                        'u_mean': torch.tensor(float(mean[0])),
                        'u_std': torch.tensor(float(std[0] if std[0] != 0 else 1.0)),
                        'v_mean': torch.tensor(float(mean[1])),
                        'v_std': torch.tensor(float(std[1] if std[1] != 0 else 1.0))
                    }
                    self.logger.info(f"✅ 归一化统计: u_mean={self.norm_stats['u_mean']:.3f}, u_std={self.norm_stats['u_std']:.3f}, v_mean={self.norm_stats['v_mean']:.3f}, v_std={self.norm_stats['v_std']:.3f}")
                else:
                    self.logger.warning("⚠️ 未找到训练集归一化统计，DC与谱损失将跳过反归一化")
            except Exception as e:
                self.logger.warning(f"⚠️ 归一化统计提取失败: {e}")

            # 一次性形状与归一化检查日志
            try:
                inp = sample_batch['input_sequence']
                tgt = sample_batch['target_sequence']
                # 形状断言
                assert inp.ndim == 5 and tgt.ndim == 5, f"Input/Target dims incorrect: {inp.ndim}/{tgt.ndim}"
                assert inp.shape[2] == tgt.shape[2], f"Channel mismatch: {inp.shape[2]} vs {tgt.shape[2]}"
                assert inp.shape[-2:] == tgt.shape[-2:], f"Spatial mismatch: {inp.shape[-2:]} vs {tgt.shape[-2:]}"
                # 归一化域统计（训练集）
                mean = inp.mean().item()
                std = inp.std().item()
                self.logger.info(f"🔎 训练样本归一化域: mean={mean:.3f}, std={std:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ 形状/归一化检查失败: {e}")

            # 统一数据键，用于反归一化与损失装配
            try:
                if not hasattr(self.config, 'data'):
                    self.config.data = DictConfig({})
                self.config.data.keys = ['u', 'v']
                self.logger.info(f"✅ 数据键设置: {self.config.data.keys}")
            except Exception as e:
                self.logger.warning(f"⚠️ 设置数据键失败: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ 数据设置失败: {e}")
            raise
    
    def adjust_batch_size_on_oom(self):
        """在内存不足时动态调整批次大小"""
        if self.memory_config['auto_batch_size_reduction'] and self.current_batch_size > 1:
            new_batch_size = max(1, self.current_batch_size // 2)
            self.logger.warning(f"内存不足，将批次大小从 {self.current_batch_size} 调整为 {new_batch_size}")
            
            # 在无多进程(num_workers=0)时，强制禁用prefetch_factor以避免ValueError
            try:
                num_workers = int(self._cfg_select('data.dataloader.num_workers', 'hardware.num_workers', default=0) or 0)
                if num_workers == 0:
                    if hasattr(self.config, 'data') and hasattr(self.config.data, 'dataloader'):
                        try:
                            self.config.data.dataloader.prefetch_factor = None
                            self.config.data.dataloader.persistent_workers = False
                            self.logger.info("⚙️ OOM调整: num_workers=0 → 设置 prefetch_factor=None, persistent_workers=False")
                        except Exception as e:
                            self.logger.warning(f"设置prefetch_factor=None失败: {e}")
                # 更新批次大小到配置
                if hasattr(self.config, 'data') and hasattr(self.config.data, 'dataloader'):
                    try:
                        self.config.data.dataloader.batch_size = new_batch_size
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning(f"OOM批次调整时配置更新失败: {e}")
            
            # 重建数据模块与数据加载器以应用最新配置
            try:
                try:
                    self.data_module.batch_size = new_batch_size
                except Exception:
                    pass
                self.data_module = RealDiffusionReactionDataModule(self.config)
                self.data_module.setup()
                self.train_loader = self.data_module.train_dataloader()
                self.val_loader = self.data_module.val_dataloader()
            except Exception as e:
                self.logger.error(f"重建数据加载器失败: {e}")
                return False
            
            self.current_batch_size = new_batch_size
            
            # 相应调整梯度累积步数
            if self.memory_config['gradient_accumulation_steps'] == 1:
                self.memory_config['gradient_accumulation_steps'] = 2
            
            return True
        return False
    
    def setup_model(self):
        """设置模型"""
        self.logger.info("🏗️ 设置模型...")
        
        try:
            # 创建基础模型
            # 读取注意力/SDPA相关配置
            use_flash = bool(self._cfg_select('training.use_flash_attention', 'model.use_flash_attention', default=False))
            sdpa_kernel = str(self._cfg_select('training.sdpa_kernel', 'model.sdpa_kernel', default='auto'))
            # 读取模型/设备相关的尺寸与通道设置
            # 统一通过安全选择函数读取配置，提供严格默认值（与SwinUNet文档一致）
            # 先进行 window_size 合法性校验与修正，避免在 BasicLayer/window_partition 视图时形状非法
            try:
                img_size = int(self._cfg_select('model.img_size', 'data.img_size', default=128))
                patch_size = int(self._cfg_select('model.patch_size', 'training.patch_size', default=4))
                depths = list(self._cfg_select('model.depths', default=[2, 2, 6, 2]))
                win = int(self._cfg_select('model.window_size', default=8))
                # 计算每层的分辨率（以patch为单位）
                from math import gcd
                patch_res = img_size // max(patch_size, 1)
                stage_res = [max(patch_res // (2 ** i), 1) for i in range(len(depths))]
                # 计算所有层分辨率的最大公约数，确保整除
                g = stage_res[0]
                for r in stage_res[1:]:
                    g = gcd(g, r)
                # 根据分辨率下限调整window_size：不超过任一层分辨率，且能整除
                safe_win = max(1, min(win, g))
                if safe_win != win:
                    self.logger.warning(
                        f"⚠️ 调整window_size: {win}→{safe_win} 以匹配阶段分辨率 {stage_res}"
                    )
                    # 尝试写回配置（若存在），否则仅在本地变量中使用
                    try:
                        self.config.model.window_size = safe_win
                    except Exception:
                        pass
            except Exception as _werr:
                # 保守降级：若校验失败则保持原值，但记录日志
                self.logger.warning(f"⚠️ window_size校验失败，保留原配置: {_werr}")

            # 安全读取所有关键模型超参，提供默认值
            in_channels = int(self._cfg_select('model.in_channels', 'data.channels', default=1))
            out_channels = int(self._cfg_select('model.out_channels', 'data.channels', default=in_channels))
            embed_dim = int(self._cfg_select('model.embed_dim', default=96))
            num_heads = list(self._cfg_select('model.num_heads', default=[3, 6, 12, 24]))
            mlp_ratio = float(self._cfg_select('model.mlp_ratio', default=4.0))
            drop_rate = float(self._cfg_select('model.drop_rate', default=0.0))
            attn_drop_rate = float(self._cfg_select('model.attn_drop_rate', default=0.0))
            drop_path_rate = float(self._cfg_select('model.drop_path_rate', default=0.1))

            base_model = SwinUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=img_size,
                patch_size=patch_size,
                window_size=safe_win if 'safe_win' in locals() else int(self._cfg_select('model.window_size', default=8)),
                depths=depths,
                num_heads=num_heads,
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                # 将SDPA/Flash相关参数向下传递到编码器与解码器
                use_checkpoint=bool(self._cfg_select('device.memory_management.gradient_checkpointing', 'training.gradient_checkpointing', default=False)),
                **{
                    'use_sdpa': use_flash,
                    'sdpa_kernel': sdpa_kernel
                }
            )
            
            # 包装为AR模型
            self.model = ARWrapper(
                single_frame_model=base_model,
                detach_rollout=True,
                scheduled_sampling=False
            )
            
            # 可选：转换为SyncBatchNorm以配合DDP
            try:
                if bool(getattr(self.config.training, 'sync_batchnorm', False)):
                    base = self.model
                    if hasattr(base, 'module'):
                        base = base.module
                    base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
                    self.model = base
                    self.logger.info("✅ 已转换模型为 SyncBatchNorm")
            except Exception as e:
                self.logger.warning(f"⚠️ 转换 SyncBatchNorm 失败: {e}")

            self.model = self.model.to(self.device)
            # 明确记录基础模型类型与设备
            try:
                base_cls = type(base_model).__name__
                wrapped_cls = type(self.model).__name__
                real_device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
                self.logger.info(f"🧩 BaseModel={base_cls}, Wrapper={wrapped_cls}, Device={real_device}")
            except Exception:
                pass
            
            # 性能优化：channels_last 与 torch.compile（在DDP包裹之前进行）
            try:
                use_channels_last = bool(self._cfg_select('training.channels_last', 'device.channels_last', default=False))
            except Exception:
                use_channels_last = False
            if use_channels_last and self.device.type == 'cuda':
                try:
                    # 全局设置模型为channels_last内存格式
                    try:
                        self.model.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                    # 兜底：逐参数设置内存格式
                    for p in self.model.parameters():
                        if p.is_cuda and p.dim() >= 4:
                            p.data = p.data.contiguous(memory_format=torch.channels_last)
                    self.logger.info("🧠 模型设为channels_last内存格式（包含逐参数兜底）")
                except Exception as e:
                    self.logger.warning(f"⚠️ 设置channels_last失败: {e}")

            # 按配置启用 torch.compile（Inductor，reduce-overhead），在DDP之前编译
            compile_enabled = False
            compile_backend = 'inductor'
            compile_mode = 'reduce-overhead'
            try:
                # 支持 training.torch_compile 与 device.compile_model 两种入口
                compile_enabled = bool(self._cfg_select('training.torch_compile', 'device.compile_model', default=False))
                compile_backend = str(self._cfg_select('training.torch_compile_backend', default='inductor'))
                compile_mode = str(self._cfg_select('training.torch_compile_mode', default='reduce-overhead'))
            except Exception:
                pass
            if compile_enabled:
                try:
                    self.model = torch.compile(self.model, backend=compile_backend, mode=compile_mode)
                    self.logger.info(f"🚀 已启用torch.compile: backend={compile_backend}, mode={compile_mode}")
                except Exception as e:
                    self.logger.warning(f"⚠️ torch.compile失败，回退未编译: {e}")

            # 将TF32设置日志化（在setup_device中已设置），这里补充记录sdpa与kernel选择
            try:
                allow_tf32 = bool(self._cfg_select('hardware.memory.allow_tf32', default=False))
                self.logger.info(f"🔧 注意力: use_sdpa/flash={use_flash}, sdpa_kernel={sdpa_kernel}, TF32={allow_tf32}")
            except Exception:
                pass

            # 记录AMP dtype选择
            try:
                dtype_str = 'default'
                if self.autocast_dtype is torch.float16:
                    dtype_str = 'float16'
                elif self.autocast_dtype is torch.bfloat16:
                    dtype_str = 'bfloat16'
                self.logger.info(f"🧪 AMP autocast dtype: {dtype_str}")
            except Exception:
                pass

            # DDP优先，其次DataParallel
            if getattr(self, 'distributed', False):
                self.logger.info("🔄 启用DistributedDataParallel")
                # 在CPU模式下，必须将 device_ids/output_device 设为 None；仅在CUDA下设置为本地设备索引
                if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                    dev_id = self.local_rank if hasattr(self, 'local_rank') else (self.device.index if self.device.index is not None else None)
                    device_ids = [dev_id] if dev_id is not None else None
                    output_device = dev_id
                else:
                    device_ids = None
                    output_device = None
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=device_ids,
                    output_device=output_device,
                    find_unused_parameters=False
                )
            else:
                if self.use_multi_gpu and torch.cuda.device_count() > 1:
                    self.logger.info("⚠️ 非DDP模式下暂不启用DataParallel，避免开销和不一致性")

            # 计算参数量与记录FLOPs/推理延迟（单次采样）
            model_for_params = self.model.module if hasattr(self.model, 'module') else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"✅ 模型参数量: {total_params:,} (可训练: {trainable_params:,})")

            # 资源统计：FLOPs与延迟（以当前img_size/通道配置为准，输入形状[B,C,H,W]）
            try:
                from utils.performance import PerformanceProfiler
                profiler = PerformanceProfiler(device=self.device.type)
                input_shape = (1, self.config.model.in_channels, self.config.model.img_size, self.config.model.img_size)
                # 移动到设备
                model_for_perf = self.model
                if hasattr(model_for_perf, 'module'):
                    model_for_perf = model_for_perf.module
                model_for_perf.eval()
                dummy = torch.randn(input_shape, device=self.device)
                flops_info = profiler.calculate_flops(model_for_perf, dummy)
                latency_info = profiler.measure_inference_latency(model_for_perf, dummy, num_runs=20, warmup_runs=5)
                # 记录到日志与保存资源信息文件
                resource_info = {
                    'params': total_params,
                    'params_trainable': trainable_params,
                    'flops_total': int(flops_info.get('total', 0)),
                    'flops_g': float(flops_info.get('total_gflops', 0.0)),
                    'inference_latency_ms_mean': float(latency_info.get('mean_ms', 0.0)),
                    'inference_latency_ms_std': float(latency_info.get('std_ms', 0.0)),
                    'input_shape': input_shape
                }
                self.logger.info(
                    f"📊 资源: FLOPs={resource_info['flops_g']:.3f}G@{input_shape[2]}², "
                    f"延迟={resource_info['inference_latency_ms_mean']:.2f}±{resource_info['inference_latency_ms_std']:.2f}ms"
                )
                try:
                    with open(self.output_dir / 'model_resources.json', 'w') as f:
                        json.dump(resource_info, f, indent=2)
                except Exception as _wr_err:
                    self.logger.warning(f"写入资源文件失败: {_wr_err}")
            except Exception as e:
                self.logger.warning(f"资源统计失败，继续训练: {e}")

            # 已在DDP前处理channels_last与torch.compile


            # 存储归一化统计到trainer，供损失函数统一使用
            try:
                # RealDiffusionReactionDataset 使用 mean/std 为 Tensor[C]
                train_ds = getattr(self.data_module, 'train_dataset', None)
                if train_ds is not None and hasattr(train_ds, 'mean') and hasattr(train_ds, 'std'):
                    mean = train_ds.mean if isinstance(train_ds.mean, torch.Tensor) else torch.as_tensor(train_ds.mean, dtype=torch.float32)
                    std = train_ds.std if isinstance(train_ds.std, torch.Tensor) else torch.as_tensor(train_ds.std, dtype=torch.float32)
                    # 组装为 norm_stats 字典，包含通道级 'mean'/'std' 以及兼容旧键名（u/v）
                    self.norm_stats = {
                        'mean': mean.clone(),
                        'std': std.clone()
                    }
                    # 兼容: 若为双通道，提供 u/v 键名，避免旧代码报错
                    try:
                        if mean.numel() >= 1:
                            self.norm_stats['u_mean'] = mean[0]
                            self.norm_stats['u_std'] = std[0]
                        if mean.numel() >= 2:
                            self.norm_stats['v_mean'] = mean[1]
                            self.norm_stats['v_std'] = std[1]
                    except Exception:
                        pass
                    self.logger.info("✅ 已提取归一化统计用于损失：提供 'mean/std' 通道级统计")
                else:
                    # 回退：使用零均值、单位方差
                    C = self.config.model.out_channels
                    zeros = torch.zeros(C)
                    ones = torch.ones(C)
                    self.norm_stats = {
                        'mean': zeros.clone(),
                        'std': ones.clone()
                    }
                    # 兼容旧键名
                    try:
                        self.norm_stats['u_mean'] = zeros[0]
                        self.norm_stats['u_std'] = ones[0]
                        if C > 1:
                            self.norm_stats['v_mean'] = zeros[1]
                            self.norm_stats['v_std'] = ones[1]
                    except Exception:
                        pass
                    self.logger.warning("⚠️ 数据集未提供mean/std，使用默认0/1 归一化统计（含通道级 'mean/std'）")
            except Exception as e:
                self.logger.warning(f"⚠️ 归一化统计组装失败: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ 模型设置失败: {e}")
            raise
    
    def setup_optimizer(self):
        """设置优化器"""
        self.logger.info("⚙️ 设置优化器...")
        
        # 优化器 - 支持 fused/foreach/eps/amsgrad（若缺失则提供健壮回退）
        try:
            opt_cfg = self.config.training.optimizer
        except Exception:
            from omegaconf import DictConfig
            self.logger.warning("⚠️ 未找到 training.optimizer，使用默认AdamW配置回退")
            if not hasattr(self.config, 'training'):
                self.config.training = DictConfig({})
            self.config.training.optimizer = DictConfig({
                'name': 'AdamW',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'amsgrad': False,
                'fused': False,
                'foreach': False,
            })
            opt_cfg = self.config.training.optimizer

        adamw_kwargs = {
            'lr': float(getattr(opt_cfg, 'lr', 1e-3)),
            'weight_decay': float(getattr(opt_cfg, 'weight_decay', 1e-4)),
            'betas': tuple(getattr(opt_cfg, 'betas', (0.9, 0.999))),
            'eps': float(getattr(opt_cfg, 'eps', 1e-8)),
            'amsgrad': bool(getattr(opt_cfg, 'amsgrad', False))
        }
        # PyTorch 2.0+ 支持 fused/foreach 标志
        fused_flag = bool(getattr(opt_cfg, 'fused', False))
        foreach_flag = bool(getattr(opt_cfg, 'foreach', False))
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **adamw_kwargs,
                fused=fused_flag,
                foreach=foreach_flag
            )
            self.logger.info(f"✅ 优化器: AdamW (fused={fused_flag}, foreach={foreach_flag}, eps={adamw_kwargs['eps']}, amsgrad={adamw_kwargs['amsgrad']})")
        except TypeError:
            # 回退：不支持fused/foreach的环境
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **adamw_kwargs
            )
            self.logger.info(f"✅ 优化器: AdamW (fallback, eps={adamw_kwargs['eps']}, amsgrad={adamw_kwargs['amsgrad']})")
        
        # 学习率调度器（若缺失则提供回退到 CosineAnnealingLR）
        try:
            sch_cfg = self.config.training.scheduler
        except Exception:
            from omegaconf import DictConfig
            self.logger.warning("⚠️ 未找到 training.scheduler，使用默认CosineAnnealingLR回退")
            if not hasattr(self.config, 'training'):
                self.config.training = DictConfig({})
            self.config.training.scheduler = DictConfig({
                'name': 'CosineAnnealingLR',
                'T_max': int(getattr(self.config.training, 'epochs', 1)),
                'eta_min': 1e-6,
                'warmup_epochs': 0,
            })
            sch_cfg = self.config.training.scheduler

        try:
            name = str(getattr(sch_cfg, 'name', 'CosineAnnealingLR'))
            if name.lower().startswith('cosine'):
                T_max = int(getattr(sch_cfg, 'T_max', getattr(self.config.training, 'epochs', 1)))
                eta_min = float(getattr(sch_cfg, 'eta_min', 1e-6))
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
                self.logger.info(f"✅ 调度器: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
            else:
                # 简化：其它名称暂不实现，使用Cosine回退
                T_max = int(getattr(self.config.training, 'epochs', 1))
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=1e-6)
                self.logger.info("ℹ️ 未识别调度器名称，已回退到 CosineAnnealingLR")
        except Exception as e:
            self.scheduler = None
            self.logger.warning(f"⚠️ 学习率调度器设置失败，继续训练: {e}")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.scheduler.T_max,
            eta_min=self.config.training.scheduler.eta_min
        )
        
        # 梯度缩放器（仅在FP16下启用；BF16无需GradScaler）
        try:
            amp_cfg = getattr(self.config.training, 'amp', None)
            init_scale = float(getattr(amp_cfg, 'init_scale', 2.0 ** 16)) if amp_cfg is not None else (2.0 ** 16)
            growth_factor = float(getattr(amp_cfg, 'growth_factor', 2.0)) if amp_cfg is not None else 2.0
            backoff_factor = float(getattr(amp_cfg, 'backoff_factor', 0.5)) if amp_cfg is not None else 0.5
            growth_interval = int(getattr(amp_cfg, 'growth_interval', 1000)) if amp_cfg is not None else 1000
            use_fp16_scaler = (self.device.type == 'cuda') and (getattr(self, 'autocast_dtype', torch.bfloat16) is torch.float16)
            self.scaler = GradScaler(enabled=use_fp16_scaler, init_scale=init_scale, growth_factor=growth_factor, backoff_factor=backoff_factor, growth_interval=growth_interval) if use_fp16_scaler else None
        except Exception:
            from torch.cuda.amp import GradScaler as _LegacyGradScaler  # type: ignore
            use_fp16_scaler = (self.device.type == 'cuda') and (getattr(self, 'autocast_dtype', torch.bfloat16) is torch.float16)
            self.scaler = _LegacyGradScaler() if use_fp16_scaler else None
        amp_enabled = bool(self.scaler is not None)
        self.logger.info(f"✅ AMP: dtype={'fp16' if amp_enabled else 'bf16'}, GradScaler={'on' if amp_enabled else 'off'}, device={self.device}")
        
        self.logger.info(f"✅ 学习率: {opt_cfg.lr}")
        
    def setup_monitoring(self):
        """设置监控"""
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        # 初始化当前epoch与训练历史结构
        self.current_epoch = 0
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': [],
            'learning_rates': [],
            'epochs': [],
            'curriculum_stages': [],
        }
        # 初始化性能计数器
        self._perf_samples = 0
        self._perf_fetch_time = 0.0
        self._perf_data_time = 0.0
        self._perf_compute_time = 0.0
        # 初始化TensorBoard writer
        try:
            self.writer = SummaryWriter(log_dir=str(self.output_dir))
            self.logger.info("TensorBoard 监控已启用")
        except Exception as _tb_err:
            self.logger.warning(f"TensorBoard初始化失败，继续训练: {_tb_err}")

    def run_quick_benchmark(self, num_batches: int = 50, outfile: str = 'benchmark.json'):
        """轻量级基准测试：评估数据加载和前向吞吐，写入指定文件
        按统一接口调用模型 forward(x)->y，并对ARWrapper使用 (input_seq, T_out, target_seq)。
        """
        self.logger.info(f"⚡ 运行轻量级基准测试，采样批次={num_batches}")
        results = {
            'num_batches': int(num_batches),
            'data_fetch_time_sec': 0.0,
            'forward_time_sec': 0.0,
            'samples': 0,
            'throughput_samples_per_sec': 0.0,
        }
        fetch_t, fwd_t, samples = 0.0, 0.0, 0
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            self.logger.info("训练前基准：train_loader 不存在，跳过")
            return
        import itertools
        it = itertools.islice(iter(self.train_loader), num_batches)
        current_T_out = 1
        try:
            current_T_out = self.get_current_T_out(self.current_epoch if hasattr(self, 'current_epoch') else 0)
        except Exception:
            pass
        for batch in it:
            t0 = time.time()
            # 统计取数时间（batch已在上面取到，此处仅统计处理开销）
            t1 = time.time()
            fetch_t += (t1 - t0)
            # 前向测试（不进行反向与优化）
            try:
                self.model.eval()
                with torch.no_grad():
                    if isinstance(batch, dict) and 'input_sequence' in batch and 'target_sequence' in batch:
                        x = batch['input_sequence'].to(self.device, non_blocking=True)
                        tgt = batch['target_sequence'].to(self.device, non_blocking=True)
                    else:
                        continue
                    t2 = time.time()
                    # 统一调用：ARWrapper需要 (x, T_out, tgt)
                    try:
                        _ = self.model(x, current_T_out, tgt)
                    except TypeError:
                        # 退化为通用接口 forward(x)
                        _ = self.model(x)
                    t3 = time.time()
                    fwd_t += (t3 - t2)
                    samples += int(x.shape[0])
            except Exception as _bm_fwd_err:
                self.logger.debug(f"基准前向失败，跳过该批次: {_bm_fwd_err}")
        total_t = (fetch_t + fwd_t)
        results['data_fetch_time_sec'] = fetch_t
        results['forward_time_sec'] = fwd_t
        results['samples'] = samples
        results['throughput_samples_per_sec'] = (float(samples) / total_t) if total_t > 0 else 0.0
        try:
            with open(self.output_dir / outfile, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"✅ 轻量基准完成：吞吐={results['throughput_samples_per_sec']:.2f} samples/s")
        except Exception as _b_err:
            self.logger.debug(f"写入benchmark失败: {_b_err}")
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': [],
            'curriculum_stages': [],
            'val_metrics': []
        }
        
        # 课程学习状态
        self.current_stage = 0
        self.stage_epoch = 0
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0

        # 性能窗口累计（耗时分解/CPU/IO监控）
        self.perf_last_report_time = time.time()
        self._perf_fetch_time = 0.0   # DataLoader取数耗时
        self._perf_data_time = 0.0    # 设备搬运耗时（host→device）
        self._perf_compute_time = 0.0 # 计算耗时（前向+损失+反向+优化）
        self._perf_batches = 0
        self._perf_samples = 0
        try:
            self._process = psutil.Process(os.getpid())
            # 初始化一次CPU使用率（第一次调用返回0.0）
            _ = self._process.cpu_percent(interval=None)
            # 配置CPU亲和性：hardware.cpu.cpu_affinity 或 thread_pool_size/num_workers 映射
            try:
                aff_cfg = getattr(self.config, 'hardware', None)
                cpu_cfg = getattr(aff_cfg, 'cpu', None) if aff_cfg is not None else None
                affinity = None
                if cpu_cfg is not None and hasattr(cpu_cfg, 'cpu_affinity'):
                    affinity = getattr(cpu_cfg, 'cpu_affinity')
                # 如果未显式配置亲和性，且存在num_workers或thread_pool_size，使用一个合理映射（不强制）
                if affinity is None:
                    tp_size = int(self._cfg_select('hardware.cpu.thread_pool_size', 'data.dataloader.num_workers', 'hardware.num_workers', default=0) or 0)
                    if tp_size > 0:
                        # 将主进程绑定到前tp_size个逻辑CPU，避免过度迁移
                        affinity = list(range(min(tp_size, psutil.cpu_count(logical=True) or tp_size)))
                if affinity is not None:
                    # 亲和性可以是列表或区间描述，统一为列表
                    if isinstance(affinity, (list, tuple)):
                        cpu_list = [int(x) for x in affinity if isinstance(x, (int, float))]
                    elif isinstance(affinity, dict) and 'range' in affinity:
                        start = int(affinity['range'][0])
                        end = int(affinity['range'][1])
                        cpu_list = list(range(start, end + 1))
                    else:
                        cpu_list = None
                    if cpu_list and len(cpu_list) > 0:
                        try:
                            self._process.cpu_affinity(cpu_list)
                            self.logger.info(f"CPU亲和性已设置: {cpu_list}")
                        except Exception as _aff_e:
                            self.logger.warning(f"CPU亲和性设置失败，跳过: {_aff_e}")
            except Exception as _aff_outer:
                self.logger.debug(f"CPU亲和性配置跳过: {_aff_outer}")
        except Exception:
            self._process = None
        
        # 初始化可视化器
        if VISUALIZATION_AVAILABLE:
            self.visualizer = ARTrainingVisualizer(str(self.output_dir))
        else:
            self.visualizer = None
            self.logger.warning("Visualization modules not available; disabling visualizations.")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            # 修复PyTorch 2.6兼容性问题 - 添加安全全局列表
            import torch.serialization
            from omegaconf.listconfig import ListConfig
            from omegaconf.dictconfig import DictConfig as OmegaDictConfig
            
            # 添加OmegaConf类到安全全局列表
            safe_globals = [ListConfig, OmegaDictConfig]
            
            # 尝试使用安全全局列表加载
            try:
                with torch.serialization.safe_globals(safe_globals):
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except Exception as safe_load_error:
                # 如果安全加载失败，回退到weights_only=False
                self.logger.warning(f"安全加载失败，回退到非安全模式: {safe_load_error}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 加载模型状态 - 处理结构不匹配
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except RuntimeError as e:
                self.logger.warning(f"严格模式加载失败: {e}")
                self.logger.info("尝试非严格模式加载...")
                
                # 获取当前模型和检查点的状态字典
                model_state = self.model.state_dict()
                checkpoint_state = checkpoint['model_state_dict']
                
                # 过滤掉不匹配的键
                filtered_state = {}
                for key, value in checkpoint_state.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            filtered_state[key] = value
                        else:
                            self.logger.warning(f"跳过形状不匹配的参数: {key} "
                                              f"(模型: {model_state[key].shape} vs 检查点: {value.shape})")
                    else:
                        self.logger.warning(f"跳过不存在的参数: {key}")
                
                # 检查缺失的参数
                missing_keys = set(model_state.keys()) - set(filtered_state.keys())
                if missing_keys:
                    self.logger.warning(f"以下参数将使用随机初始化: {missing_keys}")
                
                # 加载过滤后的状态字典
                self.model.load_state_dict(filtered_state, strict=False)
                self.logger.info(f"✅ 非严格模式加载成功，加载了 {len(filtered_state)}/{len(checkpoint_state)} 个参数")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # 加载训练状态
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', self.training_history)
            
            self.logger.info(f"✅ 成功加载检查点: {checkpoint_path}")
            self.logger.info(f"恢复到 epoch {self.current_epoch}, 最佳验证损失: {self.best_val_loss:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 加载检查点失败: {str(e)}")
            return False
    
    def create_visualizations(self, sample_batch: Optional[Dict] = None, epoch: int = 0):
        """创建可视化（统一ARTrainingVisualizer用法）"""
        # 简化版：若可视化器不可用则跳过
        try:
            viz_root = self.output_dir / "visualizations"
            viz_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if self.visualizer is None:
            self.logger.debug("可视化器不可用，跳过create_visualizations")
            return
        try:
            # 当提供sample_batch时，绘制输入/预测/误差热图
            if sample_batch is not None:
                self.visualizer.save_sample_batch(sample_batch, epoch)
            # 同时保存训练曲线（loss/val_loss），与黄金法则一致统一色标由可视化器处理
            try:
                self.visualizer.save_training_curves(self.training_history)
            except Exception:
                pass
            self.logger.info(f"🎨 已保存第{epoch}轮的可视化样本与训练曲线")
        except Exception as _viz_err:
            self.logger.debug(f"create_visualizations失败: {_viz_err}")

    def create_test_visualizations(self, final_test_metrics: Optional[Dict] = None):
        """测试阶段简单可视化占位实现"""
        if self.visualizer is None:
            self.logger.debug("测试可视化跳过：visualizer不可用")
            return
        try:
            # 保存测试指标摘要到可视化目录
            out_dir = self.output_dir / "visualizations"
            out_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(final_test_metrics, dict):
                with open(out_dir / "final_test_metrics.json", 'w') as f:
                    json.dump(convert_numpy_types(final_test_metrics), f, indent=2)
            self.logger.info("🖼️ 测试阶段可视化与指标保存完成")
        except Exception as _tviz_err:
            self.logger.debug(f"create_test_visualizations失败: {_tviz_err}")

    def get_current_T_out(self, epoch: int) -> int:
        """根据课程学习配置返回当前T_out，并维护当前阶段索引"""
        try:
            cur_cfg = getattr(self.config.training, 'curriculum', None)
            if not (cur_cfg and bool(getattr(cur_cfg, 'enabled', False))):
                return int(getattr(self.config.data, 'T_out', 1))
            stages = list(getattr(cur_cfg, 'stages', []))
            if not stages:
                return int(getattr(self.config.data, 'T_out', 1))
            # 累积epoch定位阶段
            total = 0
            for idx, st in enumerate(stages):
                e = int(st.get('epochs', 0))
                total += e
                if epoch < total:
                    self.current_stage = idx
                    return int(st.get('T_out', getattr(self.config.data, 'T_out', 1)))
            # 超出课程阶段范围，返回最后一个阶段的T_out
            self.current_stage = len(stages) - 1
            return int(stages[-1].get('T_out', getattr(self.config.data, 'T_out', 1)))
        except Exception:
            return int(getattr(self.config.data, 'T_out', 1))

    def _is_primary_process(self) -> bool:
        """DDP/多进程下仅主进程进行文件写入操作"""
        try:
            if torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
        except Exception:
            pass
        # 非分布式环境
        return True

    def save_checkpoint(self, epoch: int, is_best: bool):
        """保存检查点：始终保存last.ckpt；按需保存best.ckpt与周期性epoch_*.ckpt"""
        if not self._is_primary_process():
            return
        try:
            ck_cfg = getattr(self.config.training, 'checkpoint', None)
            save_last = True
            save_best = True
            save_every_n = 0
            max_keep = 2
            try:
                if ck_cfg is not None:
                    save_last = bool(getattr(ck_cfg, 'save_last', True))
                    save_best = bool(getattr(ck_cfg, 'save_best', True))
                    save_every_n = int(getattr(ck_cfg, 'save_every_n_epochs', 0) or 0)
                    max_keep = int(getattr(ck_cfg, 'max_keep', 2) or 2)
            except Exception:
                pass

            # 组装状态
            state = {
                'epoch': int(epoch),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') and self.optimizer is not None else {},
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler is not None else {},
                'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler is not None else None,
                'best_val_loss': float(getattr(self, 'best_val_loss', float('inf'))),
                'training_history': self.training_history,
                'config': OmegaConf.to_container(self.config, resolve=True),
                'timestamp': time.time(),
            }

            # 始终保存last
            if save_last:
                last_path = self.output_dir / 'last.ckpt'
                try:
                    torch.save(state, last_path)
                    self.logger.info(f"💾 已保存最后检查点: {last_path}")
                except Exception as _sl_err:
                    self.logger.warning(f"保存last.ckpt失败: {_sl_err}")

            # 保存最佳
            if save_best and is_best:
                best_path = self.output_dir / 'best.ckpt'
                try:
                    torch.save(state, best_path)
                    self.logger.info(f"🏅 已更新最佳检查点: {best_path}")
                except Exception as _sb_err:
                    self.logger.warning(f"保存best.ckpt失败: {_sb_err}")

            # 周期性保存（保留一定数量）
            if save_every_n > 0 and ((epoch + 1) % save_every_n == 0):
                ep_path = self.output_dir / f"epoch_{epoch+1:04d}.ckpt"
                try:
                    torch.save(state, ep_path)
                    self.logger.info(f"🧱 已保存周期检查点: {ep_path}")
                except Exception as _se_err:
                    self.logger.warning(f"保存周期检查点失败: {_se_err}")

                # 清理多余的周期检查点
                try:
                    import glob
                    ckpts = sorted(glob.glob(str(self.output_dir / "epoch_*.ckpt")))
                    if len(ckpts) > max_keep:
                        remove_count = len(ckpts) - max_keep
                        for p in ckpts[:remove_count]:
                            try:
                                os.remove(p)
                                self.logger.info(f"🧹 已清理旧检查点: {p}")
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(f"保存检查点失败: {e}")

    def generate_resource_summary(self):
        """汇总资源监控与每epoch资源，输出 resource_summary.json"""
        if not self._is_primary_process():
            return
        summary = {
            'epochs': int(len(self.training_history.get('epochs', []))),
            'avg_throughput_samples_per_sec': 0.0,
            'avg_epoch_time_sec': 0.0,
            'max_gpu_peak_allocated_gb': 0.0,
            'max_gpu_peak_reserved_gb': 0.0,
            'avg_cpu_percent': 0.0,
            'avg_system_memory_percent': 0.0,
            'avg_iowait_percent': 0.0,
        }
        # 读取每epoch资源
        epoch_records = []
        try:
            ep_file = self.output_dir / 'resources_epoch.jsonl'
            if ep_file.exists():
                with open(ep_file, 'r') as f:
                    for line in f:
                        try:
                            epoch_records.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass
        if epoch_records:
            import numpy as _np
            def _avg(key):
                vals = [float(r.get(key, 0.0)) for r in epoch_records]
                return float(_np.mean(vals)) if vals else 0.0
            def _max(key):
                vals = [float(r.get(key, 0.0)) for r in epoch_records]
                return float(max(vals)) if vals else 0.0
            summary['avg_throughput_samples_per_sec'] = _avg('throughput_samples_per_sec')
            summary['avg_epoch_time_sec'] = _avg('time_sec')
            summary['max_gpu_peak_allocated_gb'] = _max('gpu_peak_allocated_gb')
            summary['max_gpu_peak_reserved_gb'] = _max('gpu_peak_reserved_gb')
            summary['avg_cpu_percent'] = _avg('cpu_percent')
            summary['avg_system_memory_percent'] = _avg('system_memory_percent')
            summary['avg_iowait_percent'] = _avg('iowait_percent')
        # 写出
        try:
            out_path = self.output_dir / 'resource_summary.json'
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"📊 资源摘要已保存: {out_path}")
        except Exception as _sum_err:
            self.logger.debug(f"资源摘要写入失败: {_sum_err}")
        # 配置开关：可视化总开关
        try:
            viz_enabled = bool(self._cfg_select('visualization.enabled', default=True))
        except Exception:
            viz_enabled = True

        if not viz_enabled:
            self.logger.info("⚪ 配置关闭可视化，跳过生成")
            return

        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("可视化模块不可用，跳过可视化生成")
            return
        
        try:
            # 创建可视化目录
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 使用AR专用可视化器
            from utils.ar_visualizer import ARTrainingVisualizer
            ar_visualizer = ARTrainingVisualizer(str(viz_dir))
            
            # 可视化训练曲线
            if hasattr(self, 'training_history') and self.training_history:
                ar_visualizer.plot_training_curves(self.training_history, f"training_curves_epoch_{epoch}")
            
            # 如果有样本数据，创建AR预测可视化
            if sample_batch is not None:
                input_seq = sample_batch['input_sequence']
                target_seq = sample_batch['target_sequence'] 
                pred_seq = sample_batch.get('predictions', None)
                
                if pred_seq is None:
                    # 兜底：若未提供预测，则进行一次前向以生成
                    self.model.eval()
                    with torch.no_grad():
                        current_T_out = self.get_current_T_out(epoch)
                        pred_seq = self.model(input_seq.to(self.device), current_T_out).cpu()
                    self.model.train()
                
                # 创建AR预测可视化
                ar_visualizer.visualize_ar_predictions(
                    input_seq, target_seq, pred_seq, timestep_idx=epoch, 
                    save_name=f"ar_predictions_epoch_{epoch}"
                )
                
                # 创建误差分析
                ar_visualizer.create_error_analysis(target_seq, pred_seq, 
                                                   save_name=f"error_analysis_epoch_{epoch}")
                
                # 创建时间分析
                ar_visualizer.create_temporal_analysis(pred_seq, target_seq,
                                                     save_name=f"temporal_analysis_epoch_{epoch}")
            
            self.logger.info(f"✅ 可视化已保存到 {viz_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
        
    def get_current_T_out(self, epoch: int) -> int:
        """获取当前阶段的T_out（健壮处理缺失配置）"""
        # 安全读取课程学习开关
        try:
            curriculum_enabled = bool(self._cfg_select("training.curriculum.enabled", default=False))
        except Exception:
            curriculum_enabled = False
        
        # 若未启用课程学习，则回退到 data.T_out 或默认20
        if not curriculum_enabled:
            try:
                return int(self.config.data.T_out)
            except Exception:
                return 20
        
        # 安全读取课程阶段
        try:
            stages = self._cfg_select("training.curriculum.stages", default=[])
        except Exception:
            stages = []
        
        # 若阶段为空，回退到 data.T_out 或默认20
        if not stages:
            try:
                return int(self.config.data.T_out)
            except Exception:
                return 20
        
        cumulative_epochs = 0
        for i, stage in enumerate(stages):
            # 兼容字典或对象风格
            stage_epochs = stage.get('epochs', 0) if isinstance(stage, dict) else getattr(stage, 'epochs', 0)
            cumulative_epochs += stage_epochs
            if epoch < cumulative_epochs:
                if i != self.current_stage:
                    self.current_stage = i
                    self.stage_epoch = 0
                    desc = stage.get('description', f"阶段{i}") if isinstance(stage, dict) else getattr(stage, 'description', f"阶段{i}")
                    self.logger.info(f"🎯 进入{desc}")
                stage_T_out = stage.get('T_out', None) if isinstance(stage, dict) else getattr(stage, 'T_out', None)
                if stage_T_out is None:
                    try:
                        stage_T_out = int(self._cfg_select("data.T_out", default=20))
                    except Exception:
                        stage_T_out = 20
                return int(stage_T_out)
        
        # 若超出所有阶段，使用最后一个阶段的T_out，若缺失则回退
        last = stages[-1]
        last_T_out = last.get('T_out', None) if isinstance(last, dict) else getattr(last, 'T_out', None)
        if last_T_out is None:
            try:
                last_T_out = int(self._cfg_select("data.T_out", default=20))
            except Exception:
                last_T_out = 20
        return int(last_T_out)
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        # 重置性能窗口累计（按epoch）
        self._perf_fetch_time = 0.0
        self._perf_data_time = 0.0
        self._perf_compute_time = 0.0
        self._perf_batches = 0
        self._perf_samples = 0
        
        # 获取当前T_out
        current_T_out = self.get_current_T_out(epoch)
        
        # 梯度累积配置
        accumulation_steps = self.memory_config['gradient_accumulation_steps']

        # 记录并重置本epoch的显存峰值统计
        gpu_total = 0.0
        if self.device.type == 'cuda':
            try:
                torch.cuda.reset_peak_memory_stats()
                gpu_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
            except Exception:
                gpu_total = 0.0
        
        # 性能监控与压测配置（按新YAML路径）
        try:
            perf_cfg = getattr(self.config, 'performance_monitoring', None)
            # 窗口报告秒级间隔，兼容命名
            if perf_cfg is not None:
                perf_window_sec = int(getattr(perf_cfg, 'report_interval_seconds', getattr(perf_cfg, 'interval_sec', 30)))
            else:
                perf_window_sec = 30
            if perf_window_sec <= 0:
                perf_window_sec = 30
        except Exception:
            perf_window_sec = 30
        try:
            bm_cfg = getattr(self.config, 'benchmark', None)
            bench_enabled = bool(getattr(bm_cfg, 'enabled', True)) if bm_cfg is not None else True
            warmup_steps = int(getattr(bm_cfg, 'warmup_steps', 10)) if bm_cfg is not None else 10
            measure_steps = int(getattr(bm_cfg, 'measure_steps', 100)) if bm_cfg is not None else 100
            step_report_interval = int(getattr(bm_cfg, 'report_interval', 5)) if bm_cfg is not None else 5
            max_runtime_seconds = int(getattr(bm_cfg, 'max_runtime_seconds', 60)) if bm_cfg is not None else 60
        except Exception:
            bench_enabled = True
            warmup_steps = 10
            measure_steps = 100
            step_report_interval = 5
            max_runtime_seconds = 60
        # 启用吞吐日志
        log_throughput = True
        measured_steps = 0
        throughput_samples = []
        epoch_start_wall = time.time()
        bench_gpu_utils = []
        # 重置性能窗口起点（避免跨epoch累计）
        self.perf_last_report_time = time.time()

        # DDP下需每个epoch设置sampler的epoch以确保不同进程的shuffle一致
        if getattr(self, 'distributed', False) and hasattr(self, 'train_sampler') and self.train_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except Exception:
                pass
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", mininterval=0.1, smoothing=0.01)
        
        # 初始化梯度累积
        self.optimizer.zero_grad()
        # 记录本epoch内发生的优化步次数，用于确保调度器步进顺序正确
        self._epoch_opt_steps = 0
        # 记录上一个batch结束时间，用于估算下一次DataLoader取数耗时
        prev_batch_end_cpu = time.perf_counter()
        
        for batch_idx, batch in enumerate(progress_bar):
            # 跳过None批次（安全collate过滤可能导致None返回）
            if batch is None:
                continue
            t0 = time.perf_counter()
            batch_start_wall = time.time()
            try:
                # 检查内存使用率
                memory_usage = self.check_memory_usage()
                if memory_usage > self.memory_config['memory_threshold']:
                    self.logger.warning(f"内存使用率过高: {memory_usage:.2%}, 执行内存清理")
                    self.cleanup_memory()
                
                # 分解DataLoader取数与设备搬运耗时（近似：当前batch开始CPU时间与上一个batch结束CPU时间差）
                self._perf_fetch_time += max(0.0, t0 - prev_batch_end_cpu)
                
                # 设备搬运耗时起点
                load_t0 = time.perf_counter()
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(self.device, non_blocking=True)  # [B, T_in, C, H, W]
                target_seq = batch['target_sequence'].to(self.device, non_blocking=True)  # [B, T_out, C, H, W]
                data_end = time.perf_counter()
                self._perf_data_time += (data_end - load_t0)
                self._perf_samples += int(input_seq.shape[0])
                
                # 根据课程学习调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]
                
                # 计算耗时起点
                comp_t0 = time.perf_counter()
                # 前向传播（AMP按设备启用，显式dtype，CPU禁用）
                use_amp = (self.device.type == 'cuda')
                if use_amp:
                    amp_ctx = autocast(device_type='cuda', dtype=getattr(self, 'autocast_dtype', torch.bfloat16), enabled=True)
                else:
                    # CPU上禁用autocast
                    class _NullCtx:
                        def __enter__(self):
                            return None
                        def __exit__(self, exc_type, exc, tb):
                            return False
                    amp_ctx = _NullCtx()
                with amp_ctx:
                    pred_seq = self.model(input_seq, current_T_out, target_seq)
                    
                    # 统一损失装配（z-score域重建 + 原值域谱/DC）
                    from ops.losses import compute_ar_total_loss
                    
                    # 组装观测字典（AR：按时间维度）
                    # 计算观测序列（原值域），使用GT反归一化并应用观测算子
                    observation_seq = None
                    if hasattr(self, 'observation_op') and self.observation_op is not None and getattr(self, 'norm_stats', None) is not None:
                        try:
                            B, T, C, H, W = target_seq.shape
                            # 优先使用通道级 'mean/std' 进行反归一化，避免 keys 被误用为样本ID导致偏差
                            if isinstance(self.norm_stats, dict) and ('mean' in self.norm_stats and 'std' in self.norm_stats):
                                mean_t = self.norm_stats['mean'].to(self.device).reshape(1, C, 1, 1)
                                std_t = self.norm_stats['std'].to(self.device).reshape(1, C, 1, 1)
                            else:
                                # 回退：按键名（u/v）查找
                                keys = getattr(self.config.data, 'keys', ['u', 'v'])
                                if callable(keys) or not isinstance(keys, (list, tuple)):
                                    keys = ['u', 'v']
                                means = []
                                stds = []
                                for k in keys:
                                    m = self.norm_stats.get(f"{k}_mean")
                                    s = self.norm_stats.get(f"{k}_std")
                                    if m is None or s is None:
                                        m = torch.tensor(0.0, device=self.device)
                                        s = torch.tensor(1.0, device=self.device)
                                    means.append(m.to(self.device))
                                    stds.append(s.to(self.device))
                                mean_t = torch.stack(means).reshape(1, C, 1, 1)
                                std_t = torch.stack(stds).reshape(1, C, 1, 1)
                            gt_flat = target_seq.reshape(B * T, C, H, W).contiguous()
                            gt_orig_flat = gt_flat * std_t + mean_t
                            obs_flat = self.observation_op(gt_orig_flat)
                            obs_h, obs_w = obs_flat.shape[-2:]
                            observation_seq = obs_flat.reshape(B, T, C, obs_h, obs_w).contiguous()
                        except Exception as obs_err:
                            self.logger.warning(f"观测序列生成失败，跳过DC: {obs_err}")
                            observation_seq = None
                    # 仅提供观测序列与观测参数；当提供observation_seq时不再传baseline_seq，避免在损失中错误展平(T_in≠T_out)
                    obs_data = {
                        'observation_seq': observation_seq,
                        'h_params': self.h_params if hasattr(self, 'h_params') and self.h_params is not None else {
                            'task': 'sr', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
                        }
                    }
                    # 计算损失（返回字典，包括 total_loss）
                    losses = compute_ar_total_loss(
                        pred_seq=pred_seq,
                        gt_seq=target_seq,
                        obs_data=obs_data,
                        norm_stats=self.norm_stats,
                        config=self.config
                    )
                    loss = losses['total_loss']

                    # 梯度累积：损失除以累积步数
                    loss = loss / accumulation_steps
                
                # 反向传播与优化（BF16无需GradScaler）
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 每accumulation_steps步或最后一个batch时更新参数
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    # 梯度裁剪
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)

                    # 更新参数
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    # 统计优化步次数
                    try:
                        self._epoch_opt_steps += 1
                    except Exception:
                        self._epoch_opt_steps = 1

                    # 清零梯度
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps  # 恢复原始损失值
                compute_end = time.perf_counter()
                # 计算部分耗时（前向+损失+反向+优化）
                self._perf_compute_time += (compute_end - comp_t0)
                self._perf_batches += 1
                
                # 定期清理内存
                if batch_idx % self.memory_config['memory_cleanup_frequency'] == 0:
                    self.cleanup_memory()
                
                # 性能窗口报告（每 perf_window_sec 秒）
                if (time.time() - self.perf_last_report_time) >= perf_window_sec:
                    window = time.time() - self.perf_last_report_time
                    throughput = (self._perf_samples / window) if window > 0 else 0.0
                    try:
                        cpu_pct = self._process.cpu_percent(interval=None) if self._process else 0.0
                        iowait = psutil.cpu_times_percent(interval=None).iowait
                    except Exception:
                        cpu_pct, iowait = 0.0, 0.0
                    self.logger.info(
                        f"[Perf] window={window:.1f}s | fetch={self._perf_fetch_time:.2f}s | data={self._perf_data_time:.2f}s | compute={self._perf_compute_time:.2f}s | batches={self._perf_batches} | throughput={throughput:.1f} samples/s | CPU={cpu_pct:.1f}% | IOwait={iowait:.1f}%"
                    )
                    # 重置性能窗口累计
                    self.perf_last_report_time = time.time()
                    self._perf_fetch_time = 0.0
                    self._perf_data_time = 0.0
                    self._perf_compute_time = 0.0
                    self._perf_batches = 0
                    self._perf_samples = 0
                
                # 记录当前batch结束CPU时间用于下一次fetch耗时估算
                prev_batch_end_cpu = time.perf_counter()

                # 可选CPU燃烧段：仅在配置开启时提高CPU占用（不参与梯度）
                try:
                    cpu_burn_cfg = getattr(self.config.data, 'cpu_burn', None)
                    if cpu_burn_cfg is not None and bool(getattr(cpu_burn_cfg, 'enabled', False)):
                        burn_size = int(getattr(cpu_burn_cfg, 'burn_size', 2048))
                        burn_repeat = int(getattr(cpu_burn_cfg, 'burn_repeat', 1))
                        with torch.no_grad():
                            for _r in range(burn_repeat):
                                _a = torch.randn(burn_size, burn_size, dtype=torch.float32)
                                _b = torch.randn(burn_size, burn_size, dtype=torch.float32)
                                _ = _a.mm(_b)
                                del _a, _b, _
                except Exception as _cpu_err:
                    if hasattr(self, 'logger'):
                        self.logger.debug(f"CPU burn skipped: {_cpu_err}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"CUDA内存不足在batch {batch_idx}: {e}")
                    # 清理内存
                    self.cleanup_memory()
                    
                    # 尝试动态调整批次大小
                    if self.adjust_batch_size_on_oom():
                        self.logger.info("批次大小已调整，重新开始当前epoch")
                        # 重新开始当前epoch
                        return self.train_epoch(epoch)
                    else:
                        # 如果无法调整批次大小，跳过这个batch
                        self.logger.warning("无法进一步减小批次大小，跳过当前batch")
                        continue
                else:
                    raise e
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'T_out': current_T_out,
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录到TensorBoard
            try:
                log_every = int(self._cfg_select("experiment.log_every_n_steps", default=100))
            except Exception:
                log_every = 100
            if log_every > 0 and batch_idx % log_every == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
                self.writer.add_scalar('Train/T_out', current_T_out, global_step)
            
            # 吞吐与GPU监控
            if log_throughput:
                try:
                    batch_size = int(input_seq.shape[0])
                except Exception:
                    batch_size = 1
                step_time = time.time() - batch_start_wall
                if step_time > 0:
                    samples_per_sec = batch_size / step_time
                    throughput_samples.append(samples_per_sec)
                    if step_report_interval > 0 and (batch_idx % step_report_interval == 0):
                        avg_tput = float(np.mean(throughput_samples[-step_report_interval:])) if throughput_samples else float(samples_per_sec)
                        gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**3 if self.device.type == 'cuda' else 0.0
                        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3 if self.device.type == 'cuda' else 0.0
                        gpu_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3 if self.device.type == 'cuda' else 0.0
                        gpu_mem_util = (gpu_mem_alloc / gpu_total) if gpu_total > 0 else 0.0
                        # GPU核心利用率（可选）
                        gpu_util = None
                        if self.device.type == 'cuda':
                            try:
                                import pynvml
                                if not hasattr(self, '_nvml_inited') or not getattr(self, '_nvml_inited'):
                                    pynvml.nvmlInit()
                                    self._nvml_inited = True
                                    self._nvml_device = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                                util_rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_device)
                                gpu_util = float(util_rates.gpu)
                            except Exception:
                                gpu_util = None
                        # 日志输出
                        msg = f"[Perf] step={batch_idx} tput={avg_tput:.1f} samples/s | vram={gpu_mem_alloc:.2f}/{gpu_total:.2f} GB ({gpu_mem_util*100:.1f}%) | reserved={gpu_mem_reserved:.2f} GB"
                        if gpu_util is not None:
                            bench_gpu_utils.append(gpu_util)
                            msg += f" | gpu_util={gpu_util:.0f}%"
                        self.logger.info(msg)
                        # 写入TensorBoard
                        self.writer.add_scalar('Perf/Throughput_samples_per_sec', avg_tput, global_step)
                        self.writer.add_scalar('Perf/GPU_Memory_Util', gpu_mem_util, global_step)
                        self.writer.add_scalar('Perf/GPU_Memory_Allocated_GB', gpu_mem_alloc, global_step)
                        self.writer.add_scalar('Perf/GPU_Memory_Reserved_GB', gpu_mem_reserved, global_step)
                        if gpu_util is not None:
                            self.writer.add_scalar('Perf/GPU_Utilization_pct', gpu_util, global_step)
            
            # 压测模式：达到测量步数或时间限制即跳出本epoch的训练循环
            if bench_enabled:
                if batch_idx >= warmup_steps:
                    measured_steps += 1
                    if (max_runtime_seconds > 0 and (time.time() - epoch_start_wall) > max_runtime_seconds) or (measure_steps > 0 and measured_steps >= measure_steps):
                        avg_gpu_util = float(np.mean(bench_gpu_utils)) if bench_gpu_utils else float('nan')
                        if not np.isnan(avg_gpu_util):
                            self.logger.info(f"[Benchmark] 平均GPU利用率={avg_gpu_util:.1f}%")
                            try:
                                self.writer.add_scalar('Perf/Benchmark_Avg_GPU_Util_pct', avg_gpu_util, epoch)
                            except Exception:
                                pass
                        self.logger.info(f"[Benchmark] 结束: measured_steps={measured_steps}, epoch_time={time.time()-epoch_start_wall:.1f}s")
                        break
        
        avg_loss = total_loss / num_batches
        # 记录本epoch的显存峰值并对比阈值
        if self.device.type == 'cuda':
            try:
                peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
                util_pct = (peak_alloc / gpu_total) if gpu_total > 0 else 0.0
                self.logger.info(f"[VRAM] epoch={epoch+1} peak_alloc={peak_alloc:.2f} GB | reserved={peak_reserved:.2f} GB | util={util_pct*100:.1f}%")
                try:
                    self.writer.add_scalar('Perf/VRAM_Peak_Allocated_GB', peak_alloc, epoch)
                    self.writer.add_scalar('Perf/VRAM_Peak_Reserved_GB', peak_reserved, epoch)
                    self.writer.add_scalar('Perf/VRAM_Peak_Util_pct', util_pct, epoch)
                except Exception:
                    pass
                try:
                    vram_threshold = float(self._cfg_select('hardware.vram_threshold', 'training.memory_threshold', default=0.95))
                except Exception:
                    vram_threshold = 0.95
                if util_pct > vram_threshold:
                    self.logger.warning(f"[VRAM] 峰值利用率 {util_pct*100:.1f}% 超过阈值 {vram_threshold*100:.0f}%")
            except Exception as _vram_err:
                try:
                    self.logger.debug(f"VRAM峰值记录失败: {_vram_err}")
                except Exception:
                    pass
        self.stage_epoch += 1
        
        return avg_loss
    
    def _validate_epoch_legacy(self, epoch: int) -> Tuple[float, Dict[str, float], Optional[Dict]]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        num_batches = len(self.val_loader)
        
        current_T_out = self.get_current_T_out(epoch)
        sample_batch = None  # 保存一个样本用于可视化
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                try:
                    input_seq = batch['input_sequence'].to(self.device)
                    target_seq = batch['target_sequence'].to(self.device)
                    
                    # 根据课程学习调整目标序列长度
                    if target_seq.shape[1] > current_T_out:
                        target_seq = target_seq[:, :current_T_out]
                    
                    with autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                        pred_seq = self.model(input_seq, current_T_out)
                        # 统一损失装配
                        from ops.losses import compute_ar_total_loss
                        # 计算观测序列（原值域），使用GT反归一化并应用观测算子
                        observation_seq = None
                        if hasattr(self, 'observation_op') and self.observation_op is not None and getattr(self, 'norm_stats', None) is not None:
                            try:
                                B, T, C, H, W = target_seq.shape
                                keys = getattr(self.config.data, 'keys', ['u', 'v'])
                                # 防护：若读取到方法或非序列，回退默认键名
                                if callable(keys) or not isinstance(keys, (list, tuple)):
                                    keys = ['u', 'v']
                                means = []
                                stds = []
                                for k in keys:
                                    m = self.norm_stats.get(f"{k}_mean")
                                    s = self.norm_stats.get(f"{k}_std")
                                    if m is None or s is None:
                                        m = torch.tensor(0.0, device=self.device)
                                        s = torch.tensor(1.0, device=self.device)
                                    means.append(m.to(self.device))
                                    stds.append(s.to(self.device))
                                mean_t = torch.stack(means).reshape(1, C, 1, 1)
                                std_t = torch.stack(stds).reshape(1, C, 1, 1)
                                gt_flat = target_seq.reshape(B * T, C, H, W)
                                gt_orig_flat = gt_flat * std_t + mean_t
                                obs_flat = self.observation_op(gt_orig_flat)
                                obs_h, obs_w = obs_flat.shape[-2:]
                                observation_seq = obs_flat.reshape(B, T, C, obs_h, obs_w)
                            except Exception as obs_err:
                                self.logger.warning(f"观测序列生成失败，跳过DC: {obs_err}")
                                observation_seq = None
                        obs_data = {
                            'observation_seq': observation_seq,
                            'baseline_seq': input_seq,
                            'h_params': self.h_params if self.h_params is not None else {
                                'task': 'sr', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
                            }
                        }
                        losses = compute_ar_total_loss(
                            pred_seq=pred_seq,
                            gt_seq=target_seq,
                            obs_data=obs_data,
                            norm_stats=self.norm_stats,
                            config=self.config
                        )
                        loss = losses['total_loss']
                    
                    total_loss += loss.item()
                    
                    # 计算详细指标
                    # 指标：统一使用最后一个时间步
                    pred_last = pred_seq[:, -1]
                    target_last = target_seq[:, -1]
                    pred_np = pred_last.cpu().numpy()
                    target_np = target_last.cpu().numpy()
                    try:
                        batch_metrics = compute_metrics(pred_np, target_np, image_size=target_last.shape[-2:], include_freq_metrics=False)
                        all_metrics.append(batch_metrics)
                    except Exception as metrics_error:
                        self.logger.warning(f"指标计算失败 batch {batch_idx}: {metrics_error}")
                        # 跳过这个batch的指标计算，但继续验证
                        continue
                    
                    # 保存第一个batch用于可视化
                    if batch_idx == 0:
                        sample_batch = {
                            'input_sequence': batch['input_sequence'],
                            'target_sequence': batch['target_sequence']
                        }
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.warning(f"验证时CUDA内存不足 batch {batch_idx}: {e}")
                        self.cleanup_memory()
                        continue
                    else:
                        self.logger.error(f"验证时发生错误 batch {batch_idx}: {e}")
                        continue
        
        avg_loss = total_loss / num_batches
        
        # 计算平均指标
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_loss, avg_metrics, sample_batch
    
    def test_epoch(self) -> Dict[str, float]:
        """测试集评估"""
        self.logger.info("🧪 开始测试集评估...")
        self.model.eval()
        
        total_loss = 0.0
        all_metrics = []
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # 模型预测（测试时不使用teacher forcing），输出长度与目标序列一致
                test_T_out = target_seq.shape[1]
                pred_seq = self.model(input_seq, test_T_out)
                
                # 计算损失
                loss = F.mse_loss(pred_seq, target_seq)
                total_loss += loss.item()
                
                # 计算详细指标
                pred_np = pred_seq.cpu().numpy()
                target_np = target_seq.cpu().numpy()
                
                batch_metrics = compute_metrics(pred_np, target_np, image_size=target_seq.shape[-2:], include_freq_metrics=False)
                all_metrics.append(batch_metrics)
        
        # 聚合指标
        avg_loss = total_loss / num_batches
        
        # 计算平均指标
        final_metrics = {}
        if all_metrics:
            try:
                for key in all_metrics[0].keys():
                    # 收集所有批次的指标值并转换为标量
                    values = []
                    for m in all_metrics:
                        try:
                            metric_val = m[key]
                            if isinstance(metric_val, torch.Tensor):
                                # 如果是张量，取平均值转为标量
                                if metric_val.numel() > 1:
                                    values.append(metric_val.mean().item())
                                else:
                                    values.append(metric_val.item())
                            elif isinstance(metric_val, (list, np.ndarray)):
                                # 如果是列表或数组，取平均值
                                values.append(np.mean(metric_val))
                            else:
                                # 如果已经是标量，直接使用
                                values.append(float(metric_val))
                        except Exception as e:
                            self.logger.warning(f"处理指标 {key} 时出错: {e}")
                            continue
                    
                    # 计算所有批次的平均值
                    if values:
                        final_metrics[key] = np.mean(values)
                    else:
                        self.logger.warning(f"指标 {key} 没有有效值")
                        
            except Exception as e:
                self.logger.error(f"指标聚合失败: {e}")
                final_metrics = {'error': 'metrics_aggregation_failed'}
        
        final_metrics['test_loss'] = avg_loss
        
        self.logger.info(f"✅ 测试完成 - 损失: {avg_loss:.6f}")
        for key, value in final_metrics.items():
            if key != 'test_loss':
                self.logger.info(f"  {key}: {value:.6f}")
        
        return final_metrics
    
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], Optional[Dict]]:
        """验证一个epoch（聚合损失分量并健壮处理空验证集）"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        loss_components_list = []
        # 兜底处理：val_loader可能为None或长度不可用
        try:
            num_batches = len(self.val_loader) if self.val_loader is not None else 0
        except Exception:
            num_batches = 0
        sample_batch = None
        
        # 获取当前T_out
        current_T_out = self.get_current_T_out(epoch)
        
        # 若无有效val_loader，直接返回训练损失的占位与空指标
        if num_batches == 0:
            self.logger.warning("验证加载器不可用（None或空），跳过验证阶段")
            try:
                # 若训练历史存在最近一次训练损失，作为占位
                last_train_loss = self.training_history['train_losses'][-1] if 'train_losses' in self.training_history and len(self.training_history['train_losses']) > 0 else float('nan')
            except Exception:
                last_train_loss = float('nan')
            return last_train_loss, {}, None

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(self.device, non_blocking=True)  # [B, T_in, C, H, W]
                target_seq = batch['target_sequence'].to(self.device, non_blocking=True)  # [B, T_out, C, H, W]

                # 根据课程学习调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]

                # 模型预测（AMP加速推理；与训练一致传入teacher）
                use_amp = (self.device.type == 'cuda')
                amp_ctx = autocast(device_type='cuda', dtype=getattr(self, 'autocast_dtype', torch.bfloat16), enabled=use_amp) if use_amp else None
                if amp_ctx is None:
                    class _NullCtx:
                        def __enter__(self):
                            return None
                        def __exit__(self, exc_type, exc, tb):
                            return False
                    amp_ctx = _NullCtx()
                with amp_ctx:
                    pred_seq = self.model(input_seq, current_T_out, target_seq)

                    # 构造观测序列（原值域），与训练阶段保持一致
                    observation_seq = None
                    if hasattr(self, 'observation_op') and self.observation_op is not None and getattr(self, 'norm_stats', None) is not None:
                        try:
                            B, T, C, H, W = target_seq.shape
                            if isinstance(self.norm_stats, dict) and ('mean' in self.norm_stats and 'std' in self.norm_stats):
                                mean_t = self.norm_stats['mean'].to(self.device).reshape(1, C, 1, 1)
                                std_t = self.norm_stats['std'].to(self.device).reshape(1, C, 1, 1)
                            else:
                                keys = getattr(self.config.data, 'keys', ['u', 'v'])
                                if callable(keys) or not isinstance(keys, (list, tuple)):
                                    keys = ['u', 'v']
                                means = []
                                stds = []
                                for k in keys:
                                    m = self.norm_stats.get(f"{k}_mean")
                                    s = self.norm_stats.get(f"{k}_std")
                                    if m is None or s is None:
                                        m = torch.tensor(0.0, device=self.device)
                                        s = torch.tensor(1.0, device=self.device)
                                    means.append(m.to(self.device))
                                    stds.append(s.to(self.device))
                                mean_t = torch.stack(means).reshape(1, C, 1, 1)
                                std_t = torch.stack(stds).reshape(1, C, 1, 1)
                            gt_flat = target_seq.reshape(B * T, C, H, W).contiguous()
                            gt_orig_flat = gt_flat * std_t + mean_t
                            obs_flat = self.observation_op(gt_orig_flat)
                            obs_h, obs_w = obs_flat.shape[-2:]
                            observation_seq = obs_flat.reshape(B, T, C, obs_h, obs_w).contiguous()
                        except Exception as obs_err:
                            self.logger.warning(f"观测序列生成失败，跳过DC: {obs_err}")
                            observation_seq = None

                    obs_data = {
                        'observation_seq': observation_seq,
                        'h_params': self.h_params if hasattr(self, 'h_params') and self.h_params is not None else {
                            'task': 'sr', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
                        }
                    }

                    # 统一损失/DC计算
                    losses = compute_ar_total_loss(
                        pred_seq=pred_seq,
                        gt_seq=target_seq,
                        obs_data=obs_data,
                        norm_stats=self.norm_stats,
                        config=self.config
                    )
                    loss = losses['total_loss']
                    # 收集损失分量
                    try:
                        loss_rec = {
                            'dc_loss': float(losses.get('dc_loss', 0.0)),
                            'spectral_loss': float(losses.get('spectral_loss', 0.0)),
                            'reconstruction_loss': float(losses.get('reconstruction_loss', 0.0)),
                            'rel_l2': float(losses.get('rel_l2', 0.0)),
                            'mae': float(losses.get('mae', 0.0)),
                        }
                        loss_components_list.append(loss_rec)
                    except Exception:
                        pass
                total_loss += loss.item()

                # 计算详细指标（使用最后一个时间步）
                pred_last = pred_seq[:, -1]
                target_last = target_seq[:, -1]
                pred_np = pred_last.cpu().numpy()
                target_np = target_last.cpu().numpy()

                batch_metrics = compute_metrics(pred_np, target_np, image_size=target_last.shape[-2:], include_freq_metrics=False)
                all_metrics.append(batch_metrics)

                # 保存第一个批次用于可视化
                if batch_idx == 0:
                    sample_batch = {
                        'input_sequence': input_seq.cpu(),
                        'target_sequence': target_seq.cpu(),
                        'predictions': pred_seq.cpu()
                    }
        
        # 聚合指标
        avg_loss = total_loss / max(1, num_batches)
        
        # 计算平均指标
        final_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                # 收集所有批次的指标值并转换为标量
                values = []
                for m in all_metrics:
                    metric_val = m[key]
                    if isinstance(metric_val, torch.Tensor):
                        # 如果是张量，取平均值转为标量
                        values.append(metric_val.mean().item())
                    else:
                        # 如果已经是标量，直接使用
                        values.append(float(metric_val))
                
                # 计算所有批次的平均值
                final_metrics[key] = np.mean(values)

        # 聚合损失分量
        if loss_components_list:
            try:
                for k in ['dc_loss', 'spectral_loss', 'reconstruction_loss', 'rel_l2', 'mae']:
                    vals = [d.get(k) for d in loss_components_list if d.get(k) is not None]
                    if vals:
                        final_metrics[k] = float(np.mean(vals))
            except Exception:
                pass

        final_metrics['val_loss'] = avg_loss
        
        return avg_loss, final_metrics, sample_batch

    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点（CPU offload + 原子写 + 仅主进程）"""
        # DDP: 仅在rank 0保存
        try:
            if getattr(self, 'distributed', False) and dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass
        t0 = time.perf_counter()
        # CPU offload（避免GPU张量持有导致序列化阻塞）
        def _move_to_cpu(obj):
            import torch as _torch
            if isinstance(obj, _torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: _move_to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_move_to_cpu(v) for v in obj)
            else:
                return obj
        try:
            model_state_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        except Exception:
            model_state_cpu = self.model.state_dict()
        opt_state = self.optimizer.state_dict() if hasattr(self, 'optimizer') and self.optimizer is not None else {}
        sch_state = self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler is not None else {}
        scl_state = self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler is not None else {}
        try:
            opt_state = _move_to_cpu(opt_state)
            sch_state = _move_to_cpu(sch_state)
            scl_state = _move_to_cpu(scl_state)
        except Exception:
            pass
        checkpoint = {
            'epoch': int(epoch),
            'model_state_dict': model_state_cpu,
            'optimizer_state_dict': opt_state,
            'scheduler_state_dict': sch_state,
            'scaler_state_dict': scl_state,
            'best_val_loss': float(self.best_val_loss),
            'config': OmegaConf.to_yaml(self.config),
            'training_history': self.training_history
        }
        
        # 读取检查点策略
        ck_cfg = getattr(self.config.training, 'checkpoint', None)
        save_last = True if ck_cfg is None else bool(getattr(ck_cfg, 'save_last', True))
        save_best = True if ck_cfg is None else bool(getattr(ck_cfg, 'save_best', True))
        save_every = int(getattr(ck_cfg, 'save_every_n_epochs', 0) or 0) if ck_cfg is not None else 0
        max_keep = int(getattr(ck_cfg, 'max_keep', 2) or 2) if ck_cfg is not None else 2

        # 保存函数：原子写
        def _atomic_save(obj, path: Path):
            tmp_path = Path(str(path) + '.tmp')
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
        write_times = {}
        # 保存最新检查点
        if save_last:
            w0 = time.perf_counter()
            _atomic_save(checkpoint, self.output_dir / 'last.ckpt')
            write_times['last_ckpt_ms'] = (time.perf_counter() - w0) * 1000.0
        
        # 保存最佳检查点
        if save_best and is_best:
            w0 = time.perf_counter()
            _atomic_save(checkpoint, self.output_dir / 'best.ckpt')
            write_times['best_ckpt_ms'] = (time.perf_counter() - w0) * 1000.0
            self.logger.info(f"💾 保存最佳模型 (验证损失: {self.best_val_loss:.6f})")

        # 周期性保存
        if save_every > 0 and ((epoch + 1) % save_every == 0):
            ep_path = self.output_dir / f'epoch_{epoch+1:04d}.ckpt'
            w0 = time.perf_counter()
            _atomic_save(checkpoint, ep_path)
            write_times['periodic_ckpt_ms'] = (time.perf_counter() - w0) * 1000.0

        # 保留最近 max_keep 个周期检查点
        try:
            ep_ckpts = sorted(list(self.output_dir.glob('epoch_*.ckpt')))
            if len(ep_ckpts) > max_keep:
                to_delete = ep_ckpts[:-max_keep]
                for p in to_delete:
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        # 将检查点耗时记录到训练历史，便于外部报告总结
        try:
            if 'checkpoint_times_ms' not in self.training_history:
                self.training_history['checkpoint_times_ms'] = []
            write_times['total_ckpt_ms'] = (time.perf_counter() - t0) * 1000.0
            write_times['epoch'] = int(epoch)
            self.training_history['checkpoint_times_ms'].append(write_times)
        except Exception:
            pass

    def generate_resource_summary(self):
        """汇总资源指标，生成 JSON 与 Markdown 报告"""
        import json
        epoch_file = self.output_dir / 'resources_epoch.jsonl'
        metrics_file = self.output_dir / 'resource_metrics.jsonl'
        summary = {
            'epochs': 0,
            'avg_throughput_samples_per_sec': 0.0,
            'max_gpu_peak_allocated_gb': 0.0,
            'max_gpu_peak_reserved_gb': 0.0,
            'avg_epoch_time_sec': 0.0,
        }
        try:
            throughputs, times, peak_allocs, peak_resv = [], [], [], []
            if epoch_file.exists():
                with open(epoch_file, 'r') as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            throughputs.append(float(rec.get('throughput_samples_per_sec', 0.0)))
                            times.append(float(rec.get('time_sec', 0.0)))
                            peak_allocs.append(float(rec.get('gpu_peak_allocated_gb', 0.0)))
                            peak_resv.append(float(rec.get('gpu_peak_reserved_gb', 0.0)))
                        except Exception:
                            continue
            if throughputs:
                summary['avg_throughput_samples_per_sec'] = float(np.mean(throughputs))
            if times:
                summary['avg_epoch_time_sec'] = float(np.mean(times))
                summary['epochs'] = int(len(times))
            if peak_allocs:
                summary['max_gpu_peak_allocated_gb'] = float(np.max(peak_allocs))
            if peak_resv:
                summary['max_gpu_peak_reserved_gb'] = float(np.max(peak_resv))
            # 写入JSON
            with open(self.output_dir / 'resource_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            # 写入Markdown
            md = (
                f"# 资源摘要\n\n"
                f"- 训练轮数: {summary['epochs']}\n"
                f"- 平均吞吐: {summary['avg_throughput_samples_per_sec']:.2f} samples/s\n"
                f"- 平均每轮耗时: {summary['avg_epoch_time_sec']:.2f} s\n"
                f"- GPU峰值(alloc): {summary['max_gpu_peak_allocated_gb']:.3f} GB\n"
                f"- GPU峰值(reserved): {summary['max_gpu_peak_reserved_gb']:.3f} GB\n"
            )
            with open(self.output_dir / 'resource_summary.md', 'w') as f:
                f.write(md)
            self.logger.info("📋 资源摘要已生成: resource_summary.json / resource_summary.md")
        except Exception as _sum_err:
            self.logger.debug(f"资源摘要生成失败: {_sum_err}")
    
    # 注意：上方已实现的 create_visualizations 为统一版本；移除重复实现避免维护成本
    
    def create_test_visualizations(self, test_metrics: Dict[str, float]):
        """创建测试阶段的完整可视化报告"""
        # 配置开关：测试阶段可视化
        try:
            save_test_viz = bool(self._cfg_select('visualization.save_test_visualizations', default=True))
        except Exception:
            save_test_viz = True
        if not save_test_viz:
            self.logger.info("⚪ 配置关闭测试可视化，跳过生成")
            return

        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("可视化模块不可用，跳过测试可视化生成")
            return
        
        try:
            self.logger.info("🎨 开始生成测试阶段可视化...")
            
            # 创建测试可视化目录
            test_viz_dir = self.output_dir / "test_visualizations"
            test_viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建paper_package测试可视化目录
            paper_test_dir = Path("paper_package/figs") / f"{self.config.experiment.name}_test"
            paper_test_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化AR可视化器
            ar_visualizer = ARTrainingVisualizer(str(test_viz_dir))
            
            # 获取测试数据样本进行可视化
            self.model.eval()
            test_samples_visualized = 0
            max_test_samples = 5  # 可视化前5个测试样本
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_loader):
                    if test_samples_visualized >= max_test_samples:
                        break
                    
                    # 准备输入数据
                    input_seq = batch['input_sequence'].to(self.device)
                    target_seq = batch['target_sequence'].to(self.device)
                    
                    # 获取当前T_out
                    current_T_out = target_seq.shape[1]
                    
                    # AR预测
                    pred_seq = self.model(input_seq, T_out=current_T_out)
                    
                    # 转换为numpy数组用于可视化
                    input_np = input_seq.cpu().numpy()
                    target_np = target_seq.cpu().numpy()
                    pred_np = pred_seq.cpu().numpy()
                    
                    # 为每个样本生成可视化
                    batch_size = input_np.shape[0]
                    for sample_idx in range(min(batch_size, max_test_samples - test_samples_visualized)):
                        sample_name = f"test_sample_{test_samples_visualized + 1}"
                        
                        # 提取单个样本
                        sample_input = input_np[sample_idx:sample_idx+1]  # [1, T_in, C, H, W]
                        sample_target = target_np[sample_idx:sample_idx+1]  # [1, T_out, C, H, W]
                        sample_pred = pred_np[sample_idx:sample_idx+1]  # [1, T_out, C, H, W]
                        
                        self.logger.info(f"📊 生成测试样本 {test_samples_visualized + 1} 的可视化...")
                        
                        # 1. AR预测可视化
                        ar_visualizer.visualize_ar_predictions(
                            sample_input, sample_target, sample_pred,
                            save_name=f"{sample_name}_ar_predictions"
                        )
                        
                        # 2. 误差分析
                        ar_visualizer.create_error_analysis(
                            sample_target, sample_pred,
                            save_name=f"{sample_name}_error_analysis"
                        )
                        
                        # 3. 时间分析
                        ar_visualizer.create_temporal_analysis(
                            sample_pred, sample_target,
                            save_name=f"{sample_name}_temporal_analysis"
                        )
                        
                        test_samples_visualized += 1
                        
                        if test_samples_visualized >= max_test_samples:
                            break
            
            # 生成测试指标可视化
            self.logger.info("📈 生成测试指标可视化...")
            self._create_test_metrics_visualization(test_metrics, test_viz_dir)
            
            # 生成测试阶段HTML报告
            self.logger.info("📄 生成测试阶段HTML报告...")
            self._create_test_html_report(test_metrics, test_viz_dir, paper_test_dir)
            
            # 复制可视化文件到paper_package
            import shutil
            if test_viz_dir.exists():
                # 复制所有可视化文件
                for file_path in test_viz_dir.glob("*.png"):
                    shutil.copy2(file_path, paper_test_dir)
                for file_path in test_viz_dir.glob("*.html"):
                    shutil.copy2(file_path, paper_test_dir)
                
                self.logger.info(f"📋 测试可视化文件已复制到 {paper_test_dir}")
            
            self.logger.info(f"✅ 测试可视化已完成，保存到 {test_viz_dir} 和 {paper_test_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ 测试可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_test_metrics_visualization(self, test_metrics: Dict[str, float], output_dir: Path):
        """创建测试指标可视化"""
        try:
            import matplotlib.pyplot as plt
            
            # 准备指标数据
            metrics_names = list(test_metrics.keys())
            metrics_values = list(test_metrics.values())
            
            # 创建指标柱状图
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(metrics_names, metrics_values, color='skyblue', alpha=0.7)
            
            # 添加数值标签
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom')
            
            ax.set_title('Test Metrics Results', fontsize=16, fontweight='bold')
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.set_xlabel('Metrics', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(output_dir / 'test_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("📊 测试指标可视化已生成")
            
        except Exception as e:
            self.logger.error(f"❌ 测试指标可视化生成失败: {e}")
    
    def _create_test_html_report(self, test_metrics: Dict[str, float], viz_dir: Path, paper_dir: Path):
        """创建测试阶段HTML报告"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AR模型测试报告 - {self.config.experiment.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #4CAF50; margin-top: 30px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .metrics-table th {{ background-color: #4CAF50; color: white; }}
        .metrics-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .image-item {{ text-align: center; }}
        .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .image-item h3 {{ margin: 10px 0 5px 0; color: #333; }}
        .info-box {{ background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; }}
        .timestamp {{ color: #666; font-size: 0.9em; text-align: center; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AR模型测试报告</h1>
        
        <div class="info-box">
            <strong>实验名称:</strong> {self.config.experiment.name}<br>
            <strong>模型类型:</strong> {self.config.model.name}<br>
            <strong>测试时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>数据集:</strong> 真实扩散-反应数据
        </div>
        
        <h2>📊 测试指标结果</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>指标名称</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
            </thead>
            <tbody>
"""
            
            # 添加指标说明
            metric_descriptions = {
                'mse': 'Mean Squared Error - 均方误差',
                'mae': 'Mean Absolute Error - 平均绝对误差',
                'rel_l2': 'Relative L2 Error - 相对L2误差',
                'psnr': 'Peak Signal-to-Noise Ratio - 峰值信噪比',
                'ssim': 'Structural Similarity Index - 结构相似性指数',
                'temporal_mse': 'Temporal MSE - 时间一致性误差',
                'long_term_stability': 'Long-term Stability - 长期稳定性'
            }
            
            for metric_name, metric_value in test_metrics.items():
                description = metric_descriptions.get(metric_name, '测试指标')
                html_content += f"""
                <tr>
                    <td><strong>{metric_name.upper()}</strong></td>
                    <td>{metric_value:.6f}</td>
                    <td>{description}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
        
        <h2>📈 测试指标可视化</h2>
        <div class="image-grid">
            <div class="image-item">
                <h3>测试指标总览</h3>
                <img src="test_metrics.png" alt="测试指标">
            </div>
        </div>
        
        <h2>🎯 测试样本可视化</h2>
        <div class="image-grid">
"""
            
            # 添加测试样本可视化
            for i in range(1, 6):  # 最多5个测试样本
                sample_files = [
                    f"test_sample_{i}_ar_predictions.png",
                    f"test_sample_{i}_error_analysis.png", 
                    f"test_sample_{i}_temporal_analysis.png"
                ]
                
                for file_name in sample_files:
                    if (viz_dir / file_name).exists():
                        title = file_name.replace('.png', '').replace('_', ' ').title()
                        html_content += f"""
            <div class="image-item">
                <h3>{title}</h3>
                <img src="{file_name}" alt="{title}">
            </div>
"""
            
            html_content += f"""
        </div>
        
        <div class="timestamp">
            报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
            
            # 保存HTML报告
            report_path = viz_dir / 'test_report.html'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # 也保存到paper_package目录
            paper_report_path = paper_dir / 'test_report.html'
            with open(paper_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"📄 测试HTML报告已生成: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 测试HTML报告生成失败: {e}")

    def create_final_report(self):
        """创建最终可视化报告"""
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("可视化模块不可用，跳过最终报告生成")
            return
        
        try:
            # 创建paper_package目录
            paper_dir = Path("paper_package/figs") / self.config.experiment.name
            paper_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用统一可视化器创建综合报告
            visualizer = PDEBenchVisualizer(str(paper_dir))
            
            # 创建综合报告
            visualizer.create_comprehensive_report(str(self.output_dir))
            
            self.logger.info(f"📊 最终可视化报告已保存到 {paper_dir}")
            
            # 复制到paper_package目录
            import shutil
            viz_source = self.output_dir / "visualizations"
            if viz_source.exists():
                shutil.copytree(viz_source, paper_dir, dirs_exist_ok=True)
                self.logger.info(f"📋 可视化文件已复制到 paper_package")
            
        except Exception as e:
            self.logger.error(f"❌ 最终报告生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    def train(self):
        """主训练循环"""
        self.logger.info("🚀 开始训练...")

        start_time = time.time()
        start_epoch = self.current_epoch

        # 启动资源监控（采样间隔从配置读取，默认30秒）
        resource_monitor = None
        try:
            from utils.resource_monitor import ResourceMonitor
            # 采样间隔：performance_monitoring.report_interval_seconds（秒）
            try:
                perf_cfg = getattr(self.config, 'performance_monitoring', None)
                # 兼容不同命名：report_interval_seconds 或 interval_sec
                if perf_cfg is not None:
                    mon_interval = int(getattr(perf_cfg, 'report_interval_seconds', getattr(perf_cfg, 'interval_sec', 30)))
                else:
                    mon_interval = 30
                if mon_interval <= 0:
                    mon_interval = 30
            except Exception:
                mon_interval = 30
            # ResourceMonitor 不接受 device 参数，保持最小化调用以避免构造错误
            resource_monitor = ResourceMonitor(str(self.output_dir), interval_sec=mon_interval)
            resource_monitor.start()
            self.logger.info("📈 资源监控已启动，写入 resource_metrics.jsonl")
        except Exception as e:
            self.logger.warning(f"资源监控启动失败，继续训练: {e}")
        
        # 自适应资源调优配置与工具
        # 自适应监控配置：健壮读取，避免缺失键导致异常
        perf_cfg = getattr(self.config, 'performance_monitoring', None)
        def _perf_bool(key: str, default: bool) -> bool:
            try:
                return bool(getattr(perf_cfg, key, default)) if perf_cfg is not None else default
            except Exception:
                return default
        def _perf_float(key: str, default: float) -> float:
            try:
                val = getattr(perf_cfg, key, default) if perf_cfg is not None else default
                return float(val)
            except Exception:
                return default
        def _perf_int(key: str, default: int) -> int:
            try:
                val = getattr(perf_cfg, key, default) if perf_cfg is not None else default
                return int(val)
            except Exception:
                return default

        adaptive_enabled = _perf_bool('enabled', True)
        gpu_low_threshold = _perf_float('gpu_low_threshold', 0.90)
        iowait_high_threshold = _perf_float('iowait_high_threshold', 0.12)
        cpu_low_threshold = _perf_float('cpu_low_threshold', 0.80)
        nw_step = _perf_int('num_workers_step', 4)
        pf_step = _perf_int('prefetch_factor_step', 2)
        bs_step = _perf_int('batch_size_step', 8)

        def _read_last_resource_record() -> Optional[dict]:
            try:
                metrics_file = self.output_dir / 'resource_metrics.jsonl'
                if not metrics_file.exists():
                    return None
                with open(metrics_file, 'rb') as f:
                    try:
                        f.seek(-4096, os.SEEK_END)
                    except Exception:
                        pass
                    lines = f.read().decode('utf-8', errors='ignore').strip().splitlines()
                for line in reversed(lines):
                    try:
                        rec = json.loads(line)
                        return rec
                    except Exception:
                        continue
                return None
            except Exception:
                return None

        def _try_adjust_params(rec: dict) -> None:
            if not adaptive_enabled:
                return
            try:
                gpus = rec.get('gpus', []) or []
                avg_gpu_util = float(np.mean([g.get('util', 0.0) for g in gpus])) / 100.0 if gpus else 0.0
                # 计算平均显存占用比例（用于批次增长的安全阈值控制）
                try:
                    mem_ratios = []
                    for g in gpus:
                        used = float(g.get('mem_used_mib', 0.0))
                        total = float(g.get('mem_total_mib', 1.0))
                        if total > 0:
                            mem_ratios.append(used / total)
                    avg_mem_ratio = float(np.mean(mem_ratios)) if mem_ratios else 0.0
                except Exception:
                    avg_mem_ratio = 0.0
                cpu_pct = float(rec.get('cpu', {}).get('percent', 0.0)) / 100.0
                iowait_pct = float(rec.get('cpu_times_percent', {}).get('iowait', 0.0)) / 100.0

                dl_cfg = getattr(self.config.data, 'dataloader', None)
                if dl_cfg is None:
                    return
                cur_nw = int(getattr(dl_cfg, 'num_workers', 0) or 0)
                cur_pf = int(getattr(dl_cfg, 'prefetch_factor', 0) or 0)
                cur_bs = int(getattr(dl_cfg, 'batch_size', getattr(self.config.training, 'batch_size', 32)))

                changed = False

                # GPU低利用率且IO等待不高：增加workers/prefetch/batch
                if avg_gpu_util < gpu_low_threshold and iowait_pct < (iowait_high_threshold * 0.8):
                    new_nw = min(cur_nw + nw_step, os.cpu_count() or 96)
                    new_pf = cur_pf + pf_step if new_nw > 0 else cur_pf
                    # 批次增长需考虑显存阈值，避免OOM
                    vram_threshold = float(getattr(getattr(self.config, 'hardware', {}), 'vram_threshold', 0.94) or 0.94)
                    if avg_mem_ratio < (vram_threshold * 0.92):
                        new_bs = cur_bs + bs_step
                    else:
                        new_bs = cur_bs
                    if new_nw != cur_nw:
                        setattr(dl_cfg, 'num_workers', new_nw)
                        changed = True
                    if new_pf != cur_pf and new_nw > 0:
                        setattr(dl_cfg, 'prefetch_factor', new_pf)
                        changed = True
                    if new_bs != cur_bs:
                        setattr(dl_cfg, 'batch_size', new_bs)
                        setattr(self.config.training, 'batch_size', new_bs)
                        changed = True
                        self.logger.info(f"⚙️ 自适应↑ GPU低利用率: workers {cur_nw}->{new_nw}, prefetch {cur_pf}->{new_pf}, batch {cur_bs}->{new_bs} (avg_mem_ratio={avg_mem_ratio:.3f} < {vram_threshold*0.90:.3f})")
                    else:
                        self.logger.info(f"⚖️ 批次未增长：avg_mem_ratio={avg_mem_ratio:.3f} 接近阈值，避免显存溢出")

                # IO等待偏高：下调workers，提升prefetch以缓冲IO
                if iowait_pct > iowait_high_threshold:
                    new_nw = max(cur_nw - max(1, nw_step // 2), 0)
                    new_pf = max(cur_pf, pf_step)
                    if new_nw != cur_nw:
                        setattr(dl_cfg, 'num_workers', new_nw)
                        changed = True
                    if new_pf != cur_pf and new_nw > 0:
                        setattr(dl_cfg, 'prefetch_factor', new_pf)
                        changed = True
                    self.logger.info(f"⚙️ 自适应↓ IO等待偏高: workers {cur_nw}->{new_nw}, prefetch {cur_pf}->{new_pf}")

                # CPU低利用率且GPU也低：增加workers
                if cpu_pct < cpu_low_threshold and avg_gpu_util < gpu_low_threshold:
                    new_nw = min(cur_nw + nw_step, os.cpu_count() or 96)
                    if new_nw != cur_nw:
                        setattr(dl_cfg, 'num_workers', new_nw)
                        changed = True
                        self.logger.info(f"⚙️ 自适应↑ CPU低利用率: workers {cur_nw}->{new_nw}")

                if changed:
                    try:
                        self.logger.info("🔄 自适应调优：重建DataLoader应用新配置")
                        # 复用现有逻辑重建DataLoader
                        self.setup_data()
                    except Exception as e:
                        self.logger.warning(f"自适应重建DataLoader失败: {e}")
            except Exception as e:
                self.logger.debug(f"自适应调优跳过: {e}")
        
        try:
            # 预热基准测试：在训练前进行轻量级数据加载吞吐评估
            try:
                bm = getattr(self.config, 'benchmark', None)
                if bm is not None and bool(getattr(bm, 'enabled', False)) and bool(getattr(bm, 'run_before_training', True)):
                    nb = int(getattr(bm, 'num_batches', 50) or 50)
                    self.run_quick_benchmark(nb)
            except Exception as _bm_err:
                self.logger.debug(f"基准测试跳过: {_bm_err}")

            for epoch in range(start_epoch, self.config.training.epochs):
                epoch_start_time = time.time()
                
                # 训练
                train_loss = self.train_epoch(epoch)
                
                # 验证（每个epoch都执行，确保history包含验证项）
                val_loss, val_metrics, sample_batch = self.validate_epoch(epoch)

                # 记录历史
                self.training_history['train_losses'].append(train_loss)
                self.training_history['val_losses'].append(val_loss)
                self.training_history['val_metrics'].append(val_metrics)
                self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                self.training_history['epochs'].append(epoch)

                # 记录课程学习阶段
                current_T_out = self.get_current_T_out(epoch)
                self.training_history['curriculum_stages'].append({
                    'epoch': epoch,
                    'T_out': current_T_out,
                    'stage': self.current_stage
                })

                # 检查是否为最佳模型（支持min_delta避免微小噪声触发）
                min_delta = 0.0
                try:
                    es_cfg = getattr(self.config.training, 'early_stopping', None)
                    if es_cfg is not None:
                        min_delta = float(getattr(es_cfg, 'min_delta', 0.0) or 0.0)
                except Exception:
                    min_delta = 0.0
                is_best = val_loss < (self.best_val_loss - min_delta)
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # 保存检查点
                self.save_checkpoint(epoch, is_best)

                # 生成可视化（遵循配置开关，降低额外开销）
                viz_enabled = False
                try:
                    viz_enabled = bool(getattr(self.config.visualization, 'enabled', False))
                except Exception:
                    viz_enabled = False
                if viz_enabled and ((epoch + 1) % 10 == 0 or is_best):
                    self.create_visualizations(sample_batch, epoch)

                # 记录到TensorBoard
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                # 分量记录：若存在则写入
                try:
                    if isinstance(val_metrics, dict):
                        if 'dc_loss' in val_metrics:
                            self.writer.add_scalar('Val/DC_Loss', float(val_metrics['dc_loss']), epoch)
                        if 'spectral_loss' in val_metrics:
                            self.writer.add_scalar('Val/Spectral_Loss', float(val_metrics['spectral_loss']), epoch)
                        if 'rel_l2' in val_metrics:
                            self.writer.add_scalar('Val/RelL2', float(val_metrics['rel_l2']), epoch)
                        if 'mae' in val_metrics:
                            self.writer.add_scalar('Val/MAE', float(val_metrics['mae']), epoch)
                except Exception:
                    pass

                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{self.config.training.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Best: {self.best_val_loss:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                # 早停：patience 与 min_delta
                try:
                    es_cfg = getattr(self.config.training, 'early_stopping', None)
                    if es_cfg and bool(getattr(es_cfg, 'enabled', False)):
                        patience = int(getattr(es_cfg, 'patience', 50))
                        if self.patience_counter >= patience:
                            self.logger.info(f"⏹️ 早停触发: patience={patience}, best_val_loss={self.best_val_loss:.6f}, last_val_loss={val_loss:.6f}")
                            break
                except Exception as _es_err:
                    self.logger.debug(f"早停检查失败，已跳过: {_es_err}")
                
                # 记录显存峰值与吞吐（样本/秒）
                try:
                    if self.device.type == 'cuda':
                        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                        peak_res_gb = torch.cuda.max_memory_reserved() / 1024**3
                    else:
                        peak_alloc_gb, peak_res_gb = 0.0, 0.0
                except Exception:
                    peak_alloc_gb, peak_res_gb = 0.0, 0.0
                throughput = 0.0
                try:
                    throughput = float(self._perf_samples) / float(max(epoch_time, 1e-6))
                except Exception:
                    throughput = 0.0
                self.writer.add_scalar('Resources/GPU_Peak_Allocated_GB', peak_alloc_gb, epoch)
                self.writer.add_scalar('Resources/GPU_Peak_Reserved_GB', peak_res_gb, epoch)
                self.writer.add_scalar('Resources/Throughput_Samples_per_sec', throughput, epoch)
                self.logger.info(
                    f"📈 资源 | GPU峰值: alloc={peak_alloc_gb:.3f}GB, reserved={peak_res_gb:.3f}GB | 吞吐={throughput:.2f} samples/s"
                )

                # 写入每epoch资源JSONL
                try:
                    epoch_resource = {
                        'epoch': epoch,
                        'time_sec': epoch_time,
                        'gpu_peak_allocated_gb': peak_alloc_gb,
                        'gpu_peak_reserved_gb': peak_res_gb,
                        'fetch_time_sec': float(self._perf_fetch_time),
                        'data_time_sec': float(self._perf_data_time),
                        'compute_time_sec': float(self._perf_compute_time),
                        'samples': int(self._perf_samples),
                        'throughput_samples_per_sec': throughput,
                        # 额外资源指标：CPU/系统内存/IOwait
                        'cpu_percent': float(self._process.cpu_percent(interval=None)) if getattr(self, '_process', None) else 0.0,
                        'system_memory_percent': float(psutil.virtual_memory().percent) if psutil else 0.0,
                        'iowait_percent': float(psutil.cpu_times_percent(interval=None).iowait) if psutil else 0.0,
                    }
                    with open(self.output_dir / 'resources_epoch.jsonl', 'a') as f:
                        f.write(json.dumps(epoch_resource) + '\n')
                except Exception as _res_err:
                    self.logger.debug(f"epoch资源写入失败: {_res_err}")

                # 资源监控指标写入与自适应调优
                try:
                    rec = _read_last_resource_record()
                    if rec:
                        try:
                            avg_gpu_util = float(np.mean([g.get('util', 0.0) for g in rec.get('gpus', [])]))
                        except Exception:
                            avg_gpu_util = 0.0
                        self.writer.add_scalar('Resources/Mon_GPU_Util_percent', avg_gpu_util, epoch)
                        self.writer.add_scalar('Resources/Mon_CPU_percent', float(rec.get('cpu', {}).get('percent', 0.0)), epoch)
                        self.writer.add_scalar('Resources/Mon_Mem_used_GiB', float(rec.get('memory', {}).get('used_gib', 0.0)), epoch)
                        self.writer.add_scalar('Resources/Mon_Mem_total_GiB', float(rec.get('memory', {}).get('total_gib', 0.0)), epoch)
                        iowait = float(rec.get('cpu_times_percent', {}).get('iowait', 0.0))
                        self.writer.add_scalar('Resources/Mon_CPU_iowait_percent', iowait, epoch)
                        dio = rec.get('disk_io', {})
                        self.writer.add_scalar('Resources/Mon_Disk_read_bytes', float(dio.get('read_bytes', 0.0)), epoch)
                        self.writer.add_scalar('Resources/Mon_Disk_write_bytes', float(dio.get('write_bytes', 0.0)), epoch)
                        # 执行自适应参数调整
                        _try_adjust_params(rec)
                except Exception as _mon_err:
                    self.logger.debug(f"资源监控处理失败/调优跳过: {_mon_err}")

                
                # 更新学习率（仅当本epoch中发生过optimizer.step时才步进）
                try:
                    if hasattr(self, 'scheduler') and self.scheduler is not None:
                        if int(getattr(self, '_epoch_opt_steps', 0) or 0) > 0:
                            self.scheduler.step()
                        else:
                            # 避免PyTorch关于步进顺序的警告：若未发生优化步则跳过调度器步进
                            self.logger.debug("本epoch未发生optimizer.step，跳过scheduler.step() 以避免警告")
                except Exception as _sch_err:
                    self.logger.warning(f"学习率调度器步进失败，已跳过: {_sch_err}")
                
                # 保存训练历史
                with open(self.output_dir / 'training_history.json', 'w') as f:
                    json.dump(self.training_history, f, indent=2)
        
        except KeyboardInterrupt:
            self.logger.info("⚠️ 训练被用户中断")
        except Exception as e:
            self.logger.error(f"❌ 训练过程中出现错误: {e}")
            traceback.print_exc()
        finally:
            # 分布式清理（所有退出路径）
            try:
                self.cleanup_distributed()
            except Exception:
                pass
            # 停止资源监控
            try:
                if resource_monitor is not None:
                    resource_monitor.stop()
                    self.logger.info("🛑 资源监控已停止")
            except Exception as e:
                self.logger.warning(f"资源监控停止失败: {e}")
            total_time = time.time() - start_time
            self.logger.info(f"🏁 训练完成，总用时: {total_time/3600:.2f} 小时")
            
            # 在训练完成后，根据配置决定是否进行最终测试
            try:
                testing_enabled = bool(getattr(self.config.testing, 'enabled', True))
                run_final_test = bool(getattr(self.config.testing, 'run_final_test', True))
            except Exception:
                testing_enabled, run_final_test = True, True

            if testing_enabled and run_final_test:
                best_ckpt_path = self.output_dir / 'best.ckpt'
                if best_ckpt_path.exists():
                    self.logger.info("📊 使用最佳模型进行最终测试评估...")
                    self.load_checkpoint(str(best_ckpt_path))
                    final_test_metrics = self.test_epoch()
                    
                    # 保存测试结果
                    test_results = {
                        'final_test_metrics': final_test_metrics,
                        'test_time': time.time(),
                        'model_path': str(best_ckpt_path)
                    }
                    
                    # 转换numpy类型为JSON可序列化类型
                    test_results = convert_numpy_types(test_results)
                    
                    with open(self.output_dir / 'test_results.json', 'w') as f:
                        json.dump(test_results, f, indent=2)
                    
                    self.logger.info("✅ 最终测试结果已保存到 test_results.json")
                    
                    # 生成测试阶段可视化
                    self.logger.info("🎨 开始生成测试阶段可视化...")
                    self.create_test_visualizations(final_test_metrics)
                else:
                    self.logger.info("ℹ️ 未找到最佳检查点，跳过最终测试评估")
            else:
                self.logger.info("⏭️ 配置 testing.enabled=false 或 run_final_test=false，跳过最终测试阶段")
            
            # 生成最终可视化报告
            self.create_final_report()

            # 生成资源摘要报告（平均吞吐/耗时/GPU峰值）
            try:
                self.generate_resource_summary()
            except Exception as _sum_err:
                self.logger.debug(f"资源摘要生成失败: {_sum_err}")

            # 训练结束自动触发评估与论文材料生成（汇总）
            try:
                # 汇总与结果生成
                from tools.summarize_runs import summarize_runs  # 若存在
            except Exception:
                summarize_runs = None
            try:
                # 触发论文包生成入口（若配置开启）
                generate_paper = bool(getattr(self.config, 'paper_package', {}).get('auto_generate', True))
            except Exception:
                generate_paper = True
            if generate_paper:
                try:
                    # 直接调用生成器脚本入口
                    from tools.generate_paper_package import PaperPackageGenerator
                    # 合并后的配置快照写入 paper_package/configs
                    paper_root = Path('paper_package')
                    paper_root.mkdir(exist_ok=True, parents=True)
                    cfg_out = paper_root / 'configs' / 'config_merged.yaml'
                    cfg_out.parent.mkdir(parents=True, exist_ok=True)
                    with open(cfg_out, 'w') as f:
                        yaml_dump = OmegaConf.to_yaml(self.config)
                        f.write(yaml_dump)
                    generator = PaperPackageGenerator(self.config, paper_root)
                    generator.generate_package()
                    self.logger.info("📦 已自动生成论文材料包")
                except Exception as _pp_err:
                    self.logger.warning(f"论文材料自动生成失败: {_pp_err}")
            
            # 在分布式环境下，显式销毁进程组避免资源泄漏
            try:
                if getattr(self, 'distributed', False) and torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                    self.logger.info("🧹 已销毁分布式进程组")
            except Exception as e:
                self.logger.warning(f"⚠️ 销毁分布式进程组失败: {e}")
            
            self.writer.close()
            # 显式清理 DataLoader 以避免解释器关闭阶段线程创建错误
            try:
                if hasattr(self, 'train_loader'):
                    self.train_loader = None
                if hasattr(self, 'val_loader'):
                    self.val_loader = None
                if hasattr(self, 'test_loader'):
                    self.test_loader = None
                import gc
                gc.collect()
            except Exception as _dl_err:
                self.logger.debug(f"DataLoader cleanup skipped: {_dl_err}")


# 注意：convert_numpy_types 已在文件顶部定义，避免重复定义


def main():
    """主函数"""
    import argparse
    import traceback as _tb
    from datetime import datetime as _dt
    import os as _os
    
    parser = argparse.ArgumentParser(description="真实扩散-反应数据AR训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的随机种子列表，如 42,123,456")
    args = parser.parse_args()
    
    # 如果提供了多种子列表，则顺序运行并聚合结果
    try:
        # 将关键环境变量记录到标准输出，便于分布式诊断
        env_snapshot = {
            'WORLD_SIZE': _os.environ.get('WORLD_SIZE'),
            'RANK': _os.environ.get('RANK'),
            'LOCAL_RANK': _os.environ.get('LOCAL_RANK'),
            'CUDA_VISIBLE_DEVICES': _os.environ.get('CUDA_VISIBLE_DEVICES'),
        }
        print(f"[Env] DDP 环境变量快照: {env_snapshot}")

        if args.seeds:
            try:
                seed_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
            except Exception:
                seed_list = []
            if len(seed_list) < 1:
                # 回退到配置中的 seeds
                base_cfg = OmegaConf.load(args.config) if args.config else None
                if base_cfg is not None and hasattr(base_cfg.experiment, 'seeds'):
                    seed_list = list(getattr(base_cfg.experiment, 'seeds'))
                else:
                    seed_list = [42, 123, 456]

            aggregated = {}
            per_seed_results = []
            base_name = None
            for s in seed_list:
                # 为每个种子创建临时配置文件
                base_cfg = OmegaConf.load(args.config) if args.config else None
                if base_cfg is None:
                    # 使用默认配置对象（通过临时trainer获取）
                    tmp_trainer = RealDataARTrainer(None)
                    base_cfg = tmp_trainer.config
                cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
                # 更新种子与实验名
                try:
                    old_name = str(cfg.experiment.name)
                except Exception:
                    old_name = f"Real-DR2D-AR"
                if base_name is None:
                    base_name = old_name.split('-s')[0]
                new_name = f"{base_name}-s{s}"
                cfg.experiment.name = new_name
                cfg.experiment.seed = int(s)
                # 写入临时配置文件
                tmp_dir = Path('runs') / 'tmp_configs'
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_cfg_path = tmp_dir / f"{new_name}.yaml"
                with open(tmp_cfg_path, 'w') as f:
                    f.write(OmegaConf.to_yaml(cfg))

                # 运行该种子训练
                trainer = RealDataARTrainer(str(tmp_cfg_path))
                if args.resume:
                    trainer.load_checkpoint(args.resume)
                trainer.train()

                # 收集测试结果
                try:
                    test_json = Path(cfg.experiment.output_dir) / cfg.experiment.name / 'test_results.json'
                    if test_json.exists():
                        with open(test_json, 'r') as f:
                            res = json.load(f)
                            per_seed_results.append({'seed': s, 'metrics': res.get('final_test_metrics', {})})
                except Exception:
                    pass

            # 聚合均值±标准差
            try:
                # 统一所有指标键
                keys = set()
                for item in per_seed_results:
                    keys.update(item.get('metrics', {}).keys())
                summary = {}
                for k in keys:
                    vals = [float(item['metrics'].get(k, float('nan'))) for item in per_seed_results]
                    vals = [v for v in vals if not (np.isnan(v) or np.isinf(v))]
                    if len(vals) >= 1:
                        mean_v = float(np.mean(vals))
                        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        summary[k] = {'mean': mean_v, 'std': std_v, 'n': len(vals)}
                out_summary = {
                    'experiment_group': base_name or 'Real-DR2D-AR',
                    'seeds': seed_list,
                    'metrics': summary,
                    'timestamp': time.time()
                }
                out_path = Path('runs') / f"{(base_name or 'Real-DR2D-AR')}_multi_seed_summary.json"
                with open(out_path, 'w') as f:
                    json.dump(out_summary, f, indent=2)
                print(f"✅ 多种子汇总已保存: {out_path}")
            except Exception as _agg_err:
                print(f"⚠️ 多种子汇总失败: {_agg_err}")
        else:
            # 单次训练
            trainer = RealDataARTrainer(args.config)
            if args.resume:
                trainer.load_checkpoint(args.resume)
            trainer.train()
    except Exception as _main_err:
        # 捕获顶层异常，按rank写入独立错误日志
        try:
            rank_val = int(_os.environ.get('LOCAL_RANK', _os.environ.get('RANK', '0')))
        except Exception:
            rank_val = 0
        exp_name = None
        try:
            # 尝试从配置文件读取实验名，构造输出目录
            if args.config:
                cfg = OmegaConf.load(args.config)
                exp_name = str(getattr(cfg.experiment, 'name', 'AR-DR2D-Unknown'))
        except Exception:
            exp_name = 'AR-DR2D-Unknown'
        err_dir = Path('runs') / (exp_name or 'AR-DR2D-Unknown')
        err_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        err_file = err_dir / f"error_rank{rank_val}_{ts}.log"
        with open(err_file, 'w') as f:
            f.write("Top-level exception captured in main()\n")
            f.write("Environment snapshot:\n")
            f.write(json.dumps(env_snapshot, indent=2) + "\n")
            f.write("Traceback:\n")
            f.write(''.join(_tb.format_exception(type(_main_err), _main_err, _main_err.__traceback__)))
        print(f"❌ 发生异常，已写入错误日志: {err_file}")
        # 异常路径也执行分布式清理
        try:
            # 使用临时trainer的清理逻辑，避免重复代码
            tmp = RealDataARTrainer(None)
            tmp.cleanup_distributed()
        except Exception:
            # 若构造失败，直接调用底层dist清理
            try:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
        raise
        try:
            seed_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except Exception:
            seed_list = []
        if len(seed_list) < 1:
            # 回退到配置中的 seeds
            base_cfg = OmegaConf.load(args.config) if args.config else None
            if base_cfg is not None and hasattr(base_cfg.experiment, 'seeds'):
                seed_list = list(getattr(base_cfg.experiment, 'seeds'))
            else:
                seed_list = [42, 123, 456]

        aggregated = {}
        per_seed_results = []
        base_name = None
        for s in seed_list:
            # 为每个种子创建临时配置文件
            base_cfg = OmegaConf.load(args.config) if args.config else None
            if base_cfg is None:
                # 使用默认配置对象（通过临时trainer获取）
                tmp_trainer = RealDataARTrainer(None)
                base_cfg = tmp_trainer.config
            cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
            # 更新种子与实验名
            try:
                old_name = str(cfg.experiment.name)
            except Exception:
                old_name = f"Real-DR2D-AR"
            if base_name is None:
                base_name = old_name.split('-s')[0]
            new_name = f"{base_name}-s{s}"
            cfg.experiment.name = new_name
            cfg.experiment.seed = int(s)
            # 写入临时配置文件
            tmp_dir = Path('runs') / 'tmp_configs'
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_cfg_path = tmp_dir / f"{new_name}.yaml"
            with open(tmp_cfg_path, 'w') as f:
                f.write(OmegaConf.to_yaml(cfg))

            # 运行该种子训练
            trainer = RealDataARTrainer(str(tmp_cfg_path))
            if args.resume:
                trainer.load_checkpoint(args.resume)
            trainer.train()

            # 收集测试结果
            try:
                test_json = Path(cfg.experiment.output_dir) / cfg.experiment.name / 'test_results.json'
                if test_json.exists():
                    with open(test_json, 'r') as f:
                        res = json.load(f)
                        per_seed_results.append({'seed': s, 'metrics': res.get('final_test_metrics', {})})
            except Exception:
                pass

        # 聚合均值±标准差
        try:
            # 统一所有指标键
            keys = set()
            for item in per_seed_results:
                keys.update(item.get('metrics', {}).keys())
            summary = {}
            for k in keys:
                vals = [float(item['metrics'].get(k, float('nan'))) for item in per_seed_results]
                vals = [v for v in vals if not (np.isnan(v) or np.isinf(v))]
                if len(vals) >= 1:
                    mean_v = float(np.mean(vals))
                    std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    summary[k] = {'mean': mean_v, 'std': std_v, 'n': len(vals)}
            out_summary = {
                'experiment_group': base_name or 'Real-DR2D-AR',
                'seeds': seed_list,
                'metrics': summary,
                'timestamp': time.time()
            }
            out_path = Path('runs') / f"{(base_name or 'Real-DR2D-AR')}_multi_seed_summary.json"
            with open(out_path, 'w') as f:
                json.dump(out_summary, f, indent=2)
            print(f"✅ 多种子汇总已保存: {out_path}")
        except Exception as _agg_err:
            print(f"⚠️ 多种子汇总失败: {_agg_err}")
    finally:
        # 正常结束路径执行分布式清理
        try:
            tmp = RealDataARTrainer(None)
            tmp.cleanup_distributed()
        except Exception:
            try:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
    # 结束


if __name__ == "__main__":
    main()
