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
import sys
from logging import StreamHandler, FileHandler
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
import psutil
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
# 优先将项目根与训练目录插入到 sys.path 头部，避免与系统中同名包冲突（如 site-packages 下的 models）
for path in (project_root, training_dir):
    p = str(path)
    if p in sys.path:
        # 移动到最前确保优先级
        try:
            sys.path.remove(p)
        except Exception:
            pass
    sys.path.insert(0, p)

try:
    from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
except Exception:
    from datasets.real_dr_dataset import RealDiffusionReactionDataModule
from models.spatial import SwinUNet
from models.temporal import ARWrapper
from ops.losses import compute_total_loss, compute_ar_total_loss
from utils.metrics import compute_metrics
from utils.logger import setup_logger
from ops.degradation import apply_degradation_operator
# 分阶段预测架构模块
from models.temporal.components.sequential_spatiotemporal import (
    SequentialSpatiotemporalModel, 
    SpatialPredictionModule,
    TemporalPredictionModule
)
from models.temporal.components.sequential_dc_consistency import SequentialConsistencyChecker
from models.temporal.components.sequential_trainer import (
    SequentialSpatiotemporalTrainer,
    SpatialTrainer,
    TemporalTrainer
)
# 模型加载器
from tools.training.model_loader import create_model_with_loader, list_models, get_model_info
from tools.training.model_loader_improved import create_improved_model, list_improved_models, get_improved_model_info
from tools.training.model_loader_enhanced import create_enhanced_model, list_enhanced_models, get_enhanced_model_info, test_enhanced_model
 # 资源监控器的导入在运行时根据可用实现动态处理，避免签名不兼容

# 安全/快速collate（过滤None/低GIL压力），不可用时回退为None
try:
    from utils.collate import safe_collate_fn, fast_collate_fn
except Exception:
    safe_collate_fn = None
    fast_collate_fn = None

 
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
    PDEBenchVisualizer = None

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

# 使用安全collate，避免None样本导致default_collate异常（已在上方统一导入）


class RealDataARTrainer:
    """真实数据AR训练器 - 支持分阶段时空预测架构"""
    
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
    
    def __init__(self, config_path: str = None, model_name: str = None):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
            model_name: 模型架构名称（可选，会覆盖配置文件中的模型设置）
        """
        self.model_name = model_name  # 保存模型名称参数
        self.setup_config(config_path)
        
        # 如果指定了模型名称，更新配置
        if model_name is not None and hasattr(self, 'config'):
            try:
                if not hasattr(self.config, 'model'):
                    self.config.model = OmegaConf.create({})
                self.config.model.name = model_name
                try:
                    if hasattr(self, 'logger') and self.logger is not None:
                        self.logger.info(f"使用命令行指定的模型: {model_name}")
                    else:
                        print(f"使用命令行指定的模型: {model_name}")
                except Exception:
                    print(f"使用命令行指定的模型: {model_name}")
                try:
                    if hasattr(self.config, 'experiment') and hasattr(self.config.experiment, 'name'):
                        base_name = str(self.config.experiment.name)
                        suffix = f"-model_{model_name}"
                        if suffix not in base_name:
                            self.config.experiment.name = base_name + suffix
                except Exception:
                    pass
            except Exception as e:
                print(f"更新模型配置失败: {e}")
        
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
        
        # 分阶段预测架构相关属性
        self.sequential_trainer = None
        self.spatial_trainer = None
        self.temporal_trainer = None
        self.consistency_checker = None
        self.sequential_model = None
        self.training_phase = 'spatial'  # 'spatial', 'temporal', 'joint'
        self.phase_epochs = {'spatial': 0, 'temporal': 0, 'joint': 0}
        self.setup_logging()
        self.setup_device()
        self.setup_memory_management()
        try:
            _bs0 = int(self._cfg_select('data.dataloader.batch_size', 'training.batch_size', default=1))
        except Exception:
            _bs0 = 1
        self.original_batch_size = _bs0
        self.current_batch_size = _bs0
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
            print(f"✅ 成功加载配置文件: {config_path}")
            # 打印关键配置项进行验证
            print(f"📊 配置验证 - T_in: {getattr(self.config.data, 'T_in', '未设置')}")
            print(f"📊 配置验证 - T_out: {getattr(self.config.data, 'T_out', '未设置')}")
            print(f"📊 配置验证 - use_synthetic_data: {getattr(self.config.data, 'use_synthetic_data', '未设置')}")
        else:
            raise FileNotFoundError(f"必须提供有效的配置文件路径 --config，当前: {config_path}")

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
        try:
            s = int(getattr(self.config.experiment, 'seed', 2025))
            torch.manual_seed(s)
            np.random.seed(s)
        except Exception:
            pass

        try:
            if not hasattr(self.config, 'experiment'):
                self.config.experiment = DictConfig({})
            base_name = str(getattr(self.config.experiment, 'name', 'AR-DR2D-Exp'))
            tag = str(getattr(self, 'model_name', getattr(getattr(self.config, 'model', DictConfig({})), 'name', 'unknown')))
            if '-model_' in base_name:
                base_name = base_name.split('-model_')[0]
            self.config.experiment.name = base_name + f"-model_{tag}"
            if not hasattr(self.config.experiment, 'output_dir') or not getattr(self.config.experiment, 'output_dir'):
                self.config.experiment.output_dir = 'runs'
        except Exception:
            pass
        
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
        # 动态命名：附加日期时间戳，避免覆盖，并便于区分不同运行
        try:
            ts = time.strftime('%Y%m%d_%H%M%S')
        except Exception:
            ts = 'time'
        self.output_dir = Path(self.config.experiment.output_dir) / f"{self.config.experiment.name}_{ts}"
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
            try:
                vis = os.environ.get('CUDA_VISIBLE_DEVICES')
                if vis is not None and len(vis.strip()) > 0:
                    idx = 0
                    try:
                        idx = int(vis.strip().split(',')[0])
                    except Exception:
                        idx = 0
                    self.device = torch.device(f'cuda:{idx}')
                else:
                    self.device = torch.device('cuda')
            except Exception:
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

        # 应用CPU线程与TF32设置（来自配置hardware.*），确保充分利用硬件
        try:
            hw = getattr(self.config, 'hardware', None)
            if hw is not None:
                omp_threads = int(getattr(hw, 'omp_threads', 0) or 0)
                mkl_threads = int(getattr(hw, 'mkl_threads', 0) or 0)
                torch_threads = int(getattr(hw, 'torch_threads', 0) or 0)
                if omp_threads > 0:
                    os.environ['OMP_NUM_THREADS'] = str(omp_threads)
                if mkl_threads > 0:
                    os.environ['MKL_NUM_THREADS'] = str(mkl_threads)
                if torch_threads > 0:
                    try:
                        torch.set_num_threads(torch_threads)
                    except Exception:
                        pass
                # 允许TF32加速（如果配置开启）
                allow_tf32 = bool(getattr(hw, 'allow_tf32', True))
                try:
                    if self.device.type == 'cuda':
                        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                        torch.backends.cudnn.allow_tf32 = allow_tf32
                except Exception:
                    pass
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
                cfg_backend = None
                try:
                    if hasattr(self.config, 'device'):
                        cfg_backend = getattr(self.config.device, 'backend', None)
                except Exception:
                    cfg_backend = None
                backend = cfg_backend if (isinstance(cfg_backend, str) and cfg_backend in {'nccl', 'gloo'}) else ('nccl' if self.device.type == 'cuda' else 'gloo')
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
            torch_threads = int(self._cfg_select('hardware.torch_threads', default=0) or 0)
            mkl_threads = int(self._cfg_select('hardware.mkl_threads', default=0) or 0)
            omp_threads = int(self._cfg_select('hardware.omp_threads', default=0) or 0)
            numexpr_threads = int(self._cfg_select('hardware.numexpr_threads', default=0) or 0)
            interop_threads = int(self._cfg_select('hardware.interop_threads', default=0) or 0)
            openblas_threads = int(self._cfg_select('hardware.blas_threads', default=0) or 0)
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
                try:
                    max_thr = int(_os.environ.get('NUMEXPR_MAX_THREADS', '64'))
                except Exception:
                    max_thr = 64
                clamped = max(1, min(numexpr_threads, max_thr))
                _os.environ['NUMEXPR_MAX_THREADS'] = str(max_thr)
                _os.environ['NUMEXPR_NUM_THREADS'] = str(clamped)
            if openblas_threads > 0:
                try:
                    ob_max = 64
                except Exception:
                    ob_max = 64
                _os.environ['OPENBLAS_NUM_THREADS'] = str(max(1, min(openblas_threads, ob_max)))
            self.logger.info(f"CPU线程设置: torch={torch_threads}, interop={interop_threads}, MKL={mkl_threads}, OMP={omp_threads}, numexpr={numexpr_threads}, openblas={openblas_threads}")
        except Exception as e:
            self.logger.warning(f"CPU线程设置失败: {e}")

        # 多GPU配置
        self.use_multi_gpu = False
        if self.device.type == 'cuda':
            gpu_count = torch.cuda.device_count()
            # 记录GPU信息仅在首次初始化，不重复输出
            self.logger.debug(f"检测到 {gpu_count} 张GPU")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.debug(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f} GB")
            # 标记是否希望使用多GPU（与分布式解耦）
            self.use_multi_gpu = (gpu_count > 1 and hasattr(self.config, 'device') and getattr(self.config.device, 'devices', 1) > 1)
            if getattr(self, 'distributed', False) and self.use_multi_gpu:
                self.logger.info(f"启用多GPU训练（DDP），使用 {getattr(self.config.device, 'devices', gpu_count)} 张GPU")
            elif self.use_multi_gpu:
                self.logger.info(f"检测到多GPU且配置期望使用 {getattr(self.config.device, 'devices', gpu_count)} 张GPU；将尝试DataParallel回退")
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
            except Exception:
                pass
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
                try:
                    max_thr = int(os.environ.get('NUMEXPR_MAX_THREADS', '64'))
                except Exception:
                    max_thr = 64
                clamped = max(1, min(numexpr_threads, max_thr))
                os.environ['NUMEXPR_MAX_THREADS'] = str(max_thr)
                os.environ['NUMEXPR_NUM_THREADS'] = str(clamped)
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
                allow_tf32 = bool(self._cfg_select("hardware.allow_tf32", "hardware.memory.allow_tf32", default=True))
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
                pass
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
                    self.logger.debug("num_workers=0: prefetch_factor=None, persistent_workers=False")
                elif num_workers > 0 and hasattr(self.config, 'data') and hasattr(self.config.data, 'dataloader'):
                    try:
                        current_pin = bool(getattr(self.config.data.dataloader, 'pin_memory', False))
                    except Exception:
                        current_pin = False
                    self.config.data.dataloader.persistent_workers = True
                    prefetch_cfg = getattr(self.config.data.dataloader, 'prefetch_factor', None)
                    if prefetch_cfg in (None, 0):
                        self.config.data.dataloader.prefetch_factor = 16
                    self.logger.debug(f"num_workers={num_workers}: pin_memory={current_pin}, persistent_workers=True, prefetch_factor={self.config.data.dataloader.prefetch_factor}")
            except Exception as e:
                self.logger.warning(f"设置 prefetch_factor 保护失败: {e}")
            
            # 记录使用的批次大小
            self.logger.info(f"使用训练批次大小: {batch_size}")
            self.logger.info(f"使用验证批次大小: {val_batch_size}")
            self.logger.info(f"使用测试批次大小: {test_batch_size}")
            
            # 检查是否使用合成数据模式
            use_synthetic = bool(self._cfg_select('data.use_synthetic_data', default=False))
            self.using_synthetic = use_synthetic
            
            # 初始化DataModule变量，避免作用域问题
            dm_train = dm_val = dm_test = None
            
            if use_synthetic:
                self.logger.info("🧪 使用合成数据模式")
                using_dm = False
            else:
                # 使用新版本的数据模块，传入完整配置
                try:
                    self.data_module = RealDiffusionReactionDataModule(self.config)
                    using_dm = True
                    self.data_module.setup()
                    # 获取数据加载器（若数据模块内部强制num_workers=0，则在此处重建以支持并行加载）
                    dm_train = self.data_module.train_dataloader()
                    dm_val = self.data_module.val_dataloader()
                    dm_test = self.data_module.test_dataloader()
                except Exception as e:
                    self.logger.warning(f"数据模块setup失败，启用合成数据回退: {e}")
                    using_dm = False
                    use_synthetic = True  # 强制切换到合成数据模式
            if use_synthetic:
                # 合成数据集：匹配配置的时序与空间维度
                class SyntheticARSequenceDataset(torch.utils.data.Dataset):
                    def __init__(self, n=4096, T_in=1, T_out=1, C=2, H=128, W=128, seed=2025):
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
                T_out = int(self._cfg_select('data.T_out', default=1))
                C = int(self._cfg_select('model.out_channels', default=2))
                H = int(self._cfg_select('model.img_size', default=128))
                W = H
                synth_n = int(self._cfg_select('data.synthetic_data_config.num_samples', 'data.max_samples', default=1000) or 1000)
                seed = int(self._cfg_select('experiment.seed', default=2025))
                synth_ds = SyntheticARSequenceDataset(n=synth_n, T_in=T_in, T_out=T_out, C=C, H=H, W=W, seed=seed)
                # 划分训练/验证/测试
                n_train = int(synth_n * 0.7)
                n_val = int(synth_n * 0.15)
                self.train_dataset = torch.utils.data.Subset(synth_ds, range(0, n_train))
                self.val_dataset = torch.utils.data.Subset(synth_ds, range(n_train, n_train + n_val))
                self.test_dataset = torch.utils.data.Subset(synth_ds, range(n_train + n_val, synth_n))
                dm_train = dm_val = dm_test = None
                
                self.logger.info(f"✅ 合成数据集创建完成: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")
                self.logger.info(f"🔍 合成数据集属性检查: has_train_dataset={hasattr(self, 'train_dataset')}, has_val_dataset={hasattr(self, 'val_dataset')}, has_test_dataset={hasattr(self, 'test_dataset')}")

                # 构建DataLoader
                try:
                    from torch.utils.data import DataLoader as _DL
                    _collate = (
                        fast_collate_fn if ('fast_collate_fn' in globals() and fast_collate_fn is not None)
                        else (safe_collate_fn if ('safe_collate_fn' in globals() and safe_collate_fn is not None) else None)
                    )
                    self.train_loader = _DL(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0, pin_memory=False, persistent_workers=False)
                    self.val_loader = _DL(self.val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=_collate, num_workers=0, pin_memory=False, persistent_workers=False)
                    self.test_loader = _DL(self.test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=_collate, num_workers=0, pin_memory=False, persistent_workers=False)
                except Exception:
                    self.train_loader = None
                    self.val_loader = None
                    self.test_loader = None


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
                        self.norm_stats['u_mean'] = zeros[0] if C >= 1 else zeros
                        self.norm_stats['u_std'] = ones[0] if C >= 1 else ones
                        self.norm_stats['v_mean'] = zeros[1] if C >= 2 else zeros
                        self.norm_stats['v_std'] = ones[1] if C >= 2 else ones
                        self.logger.info("ℹ️ 使用默认归一化统计（零均值，单位方差）")
                except Exception as e:
                    self.logger.warning(f"提取归一化统计失败: {e}")

                self.logger.info("✅ 模型设置完成")
            
            # 将数据加载器分配给实例变量（仅在使用真实数据模块时）
            try:
                if 'using_dm' in locals() and using_dm:
                    self.train_loader = dm_train
                    self.val_loader = dm_val
                    self.test_loader = dm_test
            except Exception:
                pass

        except Exception as e:
            self.logger.error(f"❌ 模型设置失败: {e}")
            raise
    
    def setup_optimizer(self):
            try:
                batch_size = int(self._cfg_select('data.dataloader.batch_size', 'training.batch_size', default=(getattr(self, 'current_batch_size', None) or getattr(self, 'original_batch_size', 1) or 1)))
            except Exception:
                batch_size = 1
            # 首先确保DataLoader属性存在，如果不存在则初始化为None
            if not hasattr(self, 'train_loader'):
                self.train_loader = None
            if not hasattr(self, 'val_loader'):
                self.val_loader = None
            if not hasattr(self, 'test_loader'):
                self.test_loader = None
                
            if any(dl is None for dl in (self.train_loader, self.val_loader, self.test_loader)):
                self.logger.warning("⚠️ DataLoader仍为None，使用最小配置强制重建")
                try:
                    # 尝试从已有属性中获取dataset
                    train_ds_fb = getattr(self, 'train_dataset', None)
                    val_ds_fb = getattr(self, 'val_dataset', None)
                    test_ds_fb = getattr(self, 'test_dataset', None)
                    
                    self.logger.info(f"📊 数据集状态: train={train_ds_fb is not None}, val={val_ds_fb is not None}, test={test_ds_fb is not None}")
                    
                    # 如果self中没有，尝试从data_module获取
                    if train_ds_fb is None and hasattr(self, 'data_module') and hasattr(self.data_module, 'train_dataset'):
                        train_ds_fb = getattr(self.data_module, 'train_dataset', None)
                    if val_ds_fb is None and hasattr(self, 'data_module') and hasattr(self.data_module, 'val_dataset'):
                        val_ds_fb = getattr(self.data_module, 'val_dataset', None)
                    if test_ds_fb is None and hasattr(self, 'data_module') and hasattr(self.data_module, 'test_dataset'):
                        test_ds_fb = getattr(self.data_module, 'test_dataset', None)
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
            
            try:
                tl = len(self.train_loader) if self.train_loader is not None else 0
            except Exception:
                tl = 0
            try:
                vl = len(self.val_loader) if self.val_loader is not None else 0
            except Exception:
                vl = 0
            try:
                tsl = len(self.test_loader) if self.test_loader is not None else 0
            except Exception:
                tsl = 0
            self.logger.info(f"训练集批次数: {tl}")
            self.logger.info(f"验证集批次数: {vl}")
            self.logger.info(f"测试集批次数: {tsl}")
            
            # 测试数据加载（兼容安全collate返回None的情况）
            sample_batch = None
            try:
                it = iter(self.train_loader)
                for _ in range(10):
                    sample_batch = next(it)
                    if sample_batch is not None:
                        break
            except Exception:
                sample_batch = None
            if sample_batch is None:
                # 构造一个最小占位批次以继续初始化流程
                B = max(1, batch_size)
                T_in = int(self._cfg_select('data.T_in', default=1))
                T_out = int(self._cfg_select('data.T_out', default=1))
                C = int(self._cfg_select('model.out_channels', default=1))
                H = int(self._cfg_select('model.img_size', default=64))
                W = H
                sample_batch = {
                    'input_sequence': torch.randn(B, T_in, C, H, W),
                    'target_sequence': torch.randn(B, T_out, C, H, W),
                }
            self.logger.info(f"✅ 输入序列形状: {sample_batch['input_sequence'].shape}")
            self.logger.info(f"✅ 目标序列形状: {sample_batch['target_sequence'].shape}")
            try:
                in_shape = sample_batch['input_sequence'].shape  # [B, T_in, C, H, W]
                tgt_shape = sample_batch['target_sequence'].shape  # [B, T_out, C, H, W]
                self.logger.info(
                    f"✅ 使用通道数: input.C={in_shape[2]}, target.C={tgt_shape[2]} (单通道预测应为1)"
                )
            except Exception:
                pass

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
                boundary = obs_cfg.get('boundary', obs_cfg.get('boundary_mode', 'mirror'))
                if mode == 'sr':
                    sr_sub = obs_cfg.get('sr', {}) if isinstance(obs_cfg.get('sr', {}), dict) else {}
                    scale = obs_cfg.get('scale_factor', sr_sub.get('scale_factor', 2))
                    sigma = obs_cfg.get('blur_sigma', sr_sub.get('blur_sigma', 1.0))
                    kernel_size = obs_cfg.get('kernel_size', sr_sub.get('blur_kernel_size', 5))
                    boundary = boundary if boundary is not None else sr_sub.get('boundary_mode', 'mirror')
                    downsample = obs_cfg.get('downsample_interpolation', sr_sub.get('downsample_mode', 'area'))
                    self.h_params = {
                        'task': 'SR',
                        'scale': scale,
                        'sigma': sigma,
                        'kernel_size': kernel_size,
                        'boundary': boundary,
                        'downsample_interpolation': downsample
                    }
                    self.observation_op = lambda x: apply_degradation_operator(x, {
                        'task': 'SR', 'scale': scale, 'sigma': sigma, 'kernel_size': kernel_size, 'boundary': boundary
                    })
                elif mode == 'crop':
                    crop_sub = obs_cfg.get('crop', {}) if isinstance(obs_cfg.get('crop', {}), dict) else {}
                    crop_size = obs_cfg.get('crop_size', crop_sub.get('crop_size', None))
                    crop_box = obs_cfg.get('crop_box', crop_sub.get('crop_box', None))
                    boundary = boundary if boundary is not None else crop_sub.get('boundary_mode', 'mirror')
                    self.h_params = {
                        'task': 'Crop',
                        'crop_size': crop_size,
                        'crop_box': crop_box,
                        'boundary': boundary
                    }
                    self.observation_op = lambda x: apply_degradation_operator(x, {
                        'task': 'Crop', 'crop_size': crop_size, 'crop_box': crop_box, 'boundary': boundary
                    })
                else:
                    self.logger.warning(f"未知的观测模式: {mode}，跳过观测算子初始化")
                    self.h_params = None
                    self.observation_op = None
                self.logger.info(f"✅ 观测算子配置: {self.h_params}")

            # 归一化统计量，用于反归一化到原值域
            # 只有在未初始化时才设置为None，避免重复初始化时重置
            if not hasattr(self, 'norm_stats'):
                self.norm_stats = None
            try:
                if not getattr(self, 'using_synthetic', False):
                    train_ds = getattr(self.data_module, 'train_dataset', None)
                else:
                    train_ds = None
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
                    # 提供默认归一化统计，避免后续代码出错
                    C = self.config.model.out_channels
                    self.norm_stats = {
                        'mean': torch.zeros(C),
                        'std': torch.ones(C),
                        'u_mean': torch.tensor(0.0),
                        'u_std': torch.tensor(1.0),
                        'v_mean': torch.tensor(0.0),
                        'v_std': torch.tensor(1.0)
                    }
            except Exception as e:
                self.logger.warning(f"⚠️ 归一化统计提取失败: {e}")
                # 提供默认归一化统计，避免后续代码出错
                C = self.config.model.out_channels
                self.norm_stats = {
                    'mean': torch.zeros(C),
                    'std': torch.ones(C),
                    'u_mean': torch.tensor(0.0),
                    'u_std': torch.tensor(1.0),
                    'v_mean': torch.tensor(0.0),
                    'v_std': torch.tensor(1.0)
                }

            # 一次性形状与归一化检查日志
            try:
                inp = sample_batch['input_sequence']
                tgt = sample_batch['target_sequence']
                # 形状断言
                assert inp.ndim == 5 and tgt.ndim == 5, f"Input/Target dims incorrect: {inp.ndim}/{tgt.ndim}"
                assert inp.shape[2] == tgt.shape[2], f"Channel mismatch: {inp.shape[2]} vs {tgt.shape[2]}"
                assert inp.shape[-2:] == tgt.shape[-2:], f"Spatial mismatch: {inp.shape[-2:]} vs {tgt.shape[-2:]}"
                # 严格使用配置中的通道数，避免运行时修改
                try:
                    in_ch = int(self._cfg_select('model.in_channels', default=4))
                    out_ch = int(self._cfg_select('model.out_channels', default=1))
                    if in_ch != 4 or out_ch != 1:
                        self.logger.warning(f"建议使用固定通道配置 in=4(out1)：当前 in={in_ch}, out={out_ch}")
                except Exception:
                    pass
                # 归一化域统计（训练集）
                mean = inp.mean().item()
                std = inp.std().item()
                self.logger.info(f"🔎 训练样本归一化域: mean={mean:.3f}, std={std:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ 形状/归一化检查失败: {e}")

            # 统一数据键：尊重配置，官方扩展数据采用样本组内 'data' 键，单通道预测
            try:
                if not hasattr(self.config, 'data'):
                    self.config.data = DictConfig({})
                if not hasattr(self.config.data, 'keys') or not self.config.data.keys:
                    self.config.data.keys = ['data']
                self.logger.info(f"✅ 数据键设置: {self.config.data.keys}")
            except Exception as e:
                self.logger.warning(f"⚠️ 设置数据键失败: {e}")
            
            except Exception as e:
                self.logger.error(f"❌ 数据设置失败: {e}")
                raise
    
    def handle_cuda_error(self, error: Exception, phase: str = "training") -> bool:
        """处理CUDA相关错误，包括内存不足和其他CUDA错误"""
        error_msg = str(error).lower()
        
        # 检查是否是内存相关错误
        is_oom = any(keyword in error_msg for keyword in [
            'out of memory', 'cuda out of memory', 'oom', 'memory',
            'cuda runtime error', 'allocation', 'insufficient memory'
        ])
        
        if is_oom:
            return self.adjust_batch_size_on_oom(error, phase)
        else:
            # 其他CUDA错误，记录详细信息
            self.logger.error(f"❌ CUDA错误在{phase}阶段: {error}")
            self.logger.error(f"错误类型: {type(error).__name__}")
            return False
    
    def adjust_batch_size_on_oom(self, error: Exception = None, phase: str = "training") -> bool:
        """在内存不足时动态调整批次大小"""
        try:
            curr_bs = int(getattr(self, 'current_batch_size', 0) or 0)
        except Exception:
            curr_bs = 0
            
        # 记录详细的OOM信息
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.warning(f"💾 GPU内存状态: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB, 总计 {total:.2f}GB")
        
        if self.memory_config['auto_batch_size_reduction'] and curr_bs > 1:
            new_batch_size = max(1, curr_bs // 2)
            self.logger.warning(f"内存不足，将批次大小从 {curr_bs} 调整为 {new_batch_size}")
            if error:
                self.logger.warning(f"OOM错误详情: {error}")
            
            # 在无多进程(num_workers=0)时，强制禁用prefetch_factor以避免ValueError
            try:
                num_workers = int(self._cfg_select('data.dataloader.num_workers', 'hardware.num_workers', default=0) or 0)
                if num_workers == 0:
                    if hasattr(self.config, 'data') and hasattr(self.config.data, 'dataloader'):
                        try:
                            self.config.data.dataloader.prefetch_factor = None
                            self.config.data.dataloader.persistent_workers = False
                            self.logger.debug("⚙️ OOM调整: num_workers=0 → prefetch_factor=None, persistent_workers=False")
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
        """设置模型 - 支持分阶段预测架构"""
        # 检查是否启用分阶段预测架构 - 修复配置解析
        try:
            # 尝试多种配置路径
            sequential_enabled = False
            if hasattr(self.config, 'model') and hasattr(self.config.model, 'sequential'):
                sequential_enabled = bool(self.config.model.sequential.get('enabled', False))
            elif hasattr(self.config, 'sequential'):
                sequential_enabled = bool(self.config.sequential.get('enabled', False))
            else:
                # 最后尝试从配置字典获取
                sequential_enabled = bool(self._cfg_select('model.sequential.enabled', 'sequential.enabled', default=False))
            
            self.logger.info(f"时序模型检测: sequential_enabled={sequential_enabled}")
            
        except Exception as e:
            self.logger.warning(f"配置解析失败，回退到传统AR模型: {e}")
            sequential_enabled = False
        
        if sequential_enabled:
            self.logger.info("启用分阶段时空预测架构")
            self.setup_sequential_model()
        else:
            self.logger.info("使用传统AR模型架构")
            self.setup_traditional_model()
    
    def setup_traditional_model(self):
        """设置传统AR模型 - 支持多种模型架构"""
        self.logger.info("🏗️ 设置模型...")
        
        try:
            # 获取模型名称和配置
            model_name = str(self._cfg_select('model.name', 'model.type', 'model.architecture', default='swin_unet')).lower()
            if getattr(self, 'model_name', None):
                model_name = str(self.model_name).lower()
            self.logger.info(f"使用模型架构: {model_name}")
            
            # 获取所有可用模型列表
            available_models = list_models()
            self.logger.info(f"可用模型: {available_models}")
            
            # 检查模型是否可用
            if model_name not in available_models:
                self.logger.warning(f"模型 {model_name} 不在可用模型列表中，尝试使用模型加载器创建")
            
            # 获取模型配置参数
            model_config = {
                'in_channels': int(self._cfg_select('model.in_channels', 'data.channels', default=1)),
                'out_channels': int(self._cfg_select('model.out_channels', 'data.channels', default=1)),
                'img_size': int(self._cfg_select('model.img_size', 'data.img_size', default=128)),
            }
            
            # 根据模型类型添加特定参数
            if model_name == 'swin_unet':
                # SwinUNet特定参数
                try:
                    patch_size = int(self._cfg_select('model.patch_size', 'training.patch_size', default=4))
                    depths = list(self._cfg_select('model.depths', default=[2, 2, 6, 2]))
                    win = int(self._cfg_select('model.window_size', default=8))
                    
                    # window_size合法性校验
                    if model_config['img_size'] % max(patch_size, 1) != 0:
                        self.logger.warning(f"img_size({model_config['img_size']}) 不能被 patch_size({patch_size}) 整除")
                    
                    from math import gcd
                    patch_res = model_config['img_size'] // max(patch_size, 1)
                    stage_res = [max(patch_res // (2 ** i), 1) for i in range(len(depths))]
                    g = stage_res[0]
                    for r in stage_res[1:]:
                        g = gcd(g, r)
                    safe_win = max(1, min(win, g))
                    
                    if safe_win != win:
                        self.logger.warning(f"⚠️ 调整window_size: {win}→{safe_win} 以匹配阶段分辨率 {stage_res}")
                        try:
                            self.config.model.window_size = safe_win
                        except Exception:
                            pass
                    
                    model_config.update({
                        'patch_size': patch_size,
                        'depths': depths,
                        'window_size': safe_win if 'safe_win' in locals() else win,
                        'embed_dim': int(self._cfg_select('model.embed_dim', default=96)),
                        'num_heads': list(self._cfg_select('model.num_heads', default=[3, 6, 12, 24])),
                        'mlp_ratio': float(self._cfg_select('model.mlp_ratio', default=4.0)),
                        'drop_rate': float(self._cfg_select('model.drop_rate', default=0.0)),
                        'attn_drop_rate': float(self._cfg_select('model.attn_drop_rate', default=0.0)),
                        'drop_path_rate': float(self._cfg_select('model.drop_path_rate', default=0.1)),
                        'use_checkpoint': bool(self._cfg_select('device.memory_management.gradient_checkpointing', 'training.gradient_checkpointing', default=False)),
                        'use_sdpa': bool(self._cfg_select('training.use_flash_attention', 'model.use_flash_attention', default=False)),
                        'sdpa_kernel': str(self._cfg_select('training.sdpa_kernel', 'model.sdpa_kernel', default='auto'))
                    })
                except Exception as _werr:
                    self.logger.warning(f"⚠️ SwinUNet参数设置失败: {_werr}")
            
            # 添加通用参数
            additional_params = {}
            for key in ['embed_dim', 'num_heads', 'depths', 'mlp_ratio', 'drop_rate', 
                       'attn_drop_rate', 'drop_path_rate', 'patch_size', 'window_size',
                       'use_checkpoint', 'use_sdpa', 'sdpa_kernel']:
                try:
                    value = self._cfg_select(f'model.{key}', default=None)
                    if value is not None:
                        if key in ['depths', 'num_heads']:
                            additional_params[key] = list(value) if isinstance(value, (list, tuple)) else value
                        elif key in ['mlp_ratio', 'drop_rate', 'attn_drop_rate', 'drop_path_rate']:
                            additional_params[key] = float(value)
                        elif key in ['embed_dim', 'patch_size', 'window_size']:
                            additional_params[key] = int(value)
                        else:
                            additional_params[key] = value
                except Exception:
                    pass
            
            model_config.update(additional_params)
            
            # 使用增强模型加载器创建基础模型（四层回退策略）
            base_model = None
            model_creation_errors = []
            creation_method = None
            
            # 第一层：使用最终增强模型加载器（最高兼容性）
            try:
                base_model = create_enhanced_model(model_name, self.config, **model_config)
                creation_method = "enhanced_loader"
                self.logger.info(f"✅ 成功使用增强加载器创建基础模型: {type(base_model).__name__}")
                
                # 测试模型前向传播
                try:
                    test_success = test_enhanced_model(model_name, self.config, **model_config)
                    if test_success:
                        self.logger.info("✅ 模型前向传播测试通过")
                    else:
                        self.logger.warning("⚠️ 模型前向传播测试失败，但仍可使用")
                except Exception as test_error:
                    self.logger.warning(f"⚠️ 模型测试失败: {test_error}")
                    
            except Exception as enhanced_error:
                model_creation_errors.append(f"增强加载器: {enhanced_error}")
                self.logger.warning(f"⚠️ 增强模型加载器失败: {enhanced_error}")
                
                # 第二层：回退到改进模型加载器
                try:
                    base_model = create_improved_model(model_name, self.config, **model_config)
                    creation_method = "improved_loader"
                    self.logger.info(f"✅ 成功使用改进加载器创建基础模型: {type(base_model).__name__}")
                except Exception as improved_error:
                    model_creation_errors.append(f"改进加载器: {improved_error}")
                    self.logger.warning(f"⚠️ 改进模型加载器失败: {improved_error}")
                    
                    # 第三层：回退到原始模型加载器
                    try:
                        base_model = create_model_with_loader(model_name, self.config, **model_config)
                        creation_method = "original_loader"
                        self.logger.info(f"✅ 成功使用原始加载器创建基础模型: {type(base_model).__name__}")
                    except Exception as original_error:
                        model_creation_errors.append(f"原始加载器: {original_error}")
                        self.logger.error(f"❌ 原始模型加载器也失败: {original_error}")
                        
                        # 第四层：最终回退到默认SwinUNet实现
                        self.logger.info("回退到默认SwinUNet实现")
                        try:
                            base_model = SwinUNet(**model_config)
                            creation_method = "fallback_swinunet"
                            self.logger.info("✅ 成功回退到默认SwinUNet")
                        except Exception as swin_error:
                            self.logger.error(f"❌ 所有模型创建方式都失败: {model_creation_errors}")
                            raise RuntimeError(f"无法创建模型 {model_name}: {model_creation_errors}") from swin_error
            
            # 根据配置禁用时间预测，仅空间预测时直接使用基础模型
            try:
                ar_enabled = bool(getattr(self.config, 'ar', {}).get('enabled', True))
            except Exception:
                ar_enabled = True
                
            if ar_enabled:
                # 包装为AR模型
                self.model = ARWrapper(
                    single_frame_model=base_model,
                    detach_rollout=True,
                    scheduled_sampling=False
                )
            else:
                # 仅空间预测：直接使用单帧模型，统一 forward(x)->y
                self.model = base_model
            
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
                    base_cls = type(self.model).__name__.lower()
                    if 'swin' in base_cls:
                        raise RuntimeError("skip compile for SwinUNet due to CUDA graphs overwrite issue")
                    self.model = torch.compile(self.model, backend=compile_backend, mode=compile_mode)
                    self.logger.info(f"🚀 已启用torch.compile: backend={compile_backend}, mode={compile_mode}")
                except Exception as e:
                    self.logger.warning(f"⚠️ torch.compile失败或已跳过: {e}")

            # 将TF32设置日志化（在setup_device中已设置），这里补充记录sdpa与kernel选择
            try:
                allow_tf32 = bool(self._cfg_select('hardware.memory.allow_tf32', default=False))
                use_flash = bool(self._cfg_select('training.use_flash_attention', 'model.use_flash_attention', default=False))
                sdpa_kernel = str(self._cfg_select('training.sdpa_kernel', 'model.sdpa_kernel', default='auto'))
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
            try:
                if getattr(self, 'distributed', False):
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
                    allow_dp = False
                    try:
                        if hasattr(self.config, 'device'):
                            allow_dp = bool(getattr(self.config.device, 'allow_data_parallel_fallback', True)) or (getattr(self.config.device, 'strategy', '').lower() == 'dp')
                    except Exception:
                        allow_dp = True
                    cfg_devices = 1
                    try:
                        if hasattr(self.config, 'device'):
                            cfg_devices = int(getattr(self.config.device, 'devices', 1) or 1)
                    except Exception:
                        cfg_devices = 1
                    if torch.cuda.device_count() > 1 and allow_dp and cfg_devices > 1:
                        vis = os.environ.get('CUDA_VISIBLE_DEVICES')
                        try:
                            if vis is None or len(vis.strip()) == 0:
                                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(min(cfg_devices, torch.cuda.device_count())))
                        except Exception:
                            pass
                        self.model = torch.nn.DataParallel(self.model)
            except Exception as e:
                self.logger.warning(f"⚠️ 并行处理设置失败: {e}")

            # 计算参数量与记录FLOPs/推理延迟（单次采样）
            model_for_params = self.model.module if hasattr(self.model, 'module') else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"✅ 模型参数量: {total_params:,} (可训练: {trainable_params:,})")

            try:
                import json as _json
                selected_model = str(getattr(self, 'model_name', getattr(getattr(self.config, 'model', {}), 'name', 'unknown')))
                config_model_name = str(getattr(getattr(self.config, 'model', {}), 'name', 'unknown'))
                model_class = type(model_for_params).__name__
                info = {
                    'selected_model': selected_model,
                    'config_model_name': config_model_name,
                    'model_class': model_class,
                    'total_params': int(total_params),
                    'trainable_params': int(trainable_params)
                }
                outp = self.output_dir / 'model_info.json'
                with open(outp, 'w') as f:
                    _json.dump(info, f, indent=2)
                self.logger.info(f"ActiveModelClass={model_class} SelectedModel={selected_model} ConfigModelName={config_model_name}")
                self.logger.info(f"📝 写入模型信息: {outp}")
            except Exception:
                pass

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
        # 确定要优化的模型
        model_to_optimize = None
        if hasattr(self, 'model') and self.model is not None:
            model_to_optimize = self.model
        elif hasattr(self, 'sequential_model') and self.sequential_model is not None:
            model_to_optimize = self.sequential_model
        else:
            raise RuntimeError("未找到可优化的模型 (既无self.model也无self.sequential_model)")
        
        # PyTorch 2.0+ 支持 fused/foreach 标志
        fused_flag = bool(getattr(opt_cfg, 'fused', False))
        foreach_flag = bool(getattr(opt_cfg, 'foreach', False))
        try:
            self.optimizer = torch.optim.AdamW(
                model_to_optimize.parameters(),
                **adamw_kwargs,
                fused=fused_flag,
                foreach=foreach_flag
            )
            self.logger.info(f"✅ 优化器: AdamW (fused={fused_flag}, foreach={foreach_flag}, eps={adamw_kwargs['eps']}, amsgrad={adamw_kwargs['amsgrad']})")
        except TypeError:
            # 回退：不支持fused/foreach的环境
            self.optimizer = torch.optim.AdamW(
                model_to_optimize.parameters(),
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
            T_max = int(getattr(sch_cfg, 'T_max', getattr(self.config.training, 'epochs', 1)))
            eta_min = float(getattr(sch_cfg, 'eta_min', 1e-6))
            warmup_epochs = int(getattr(sch_cfg, 'warmup_epochs', 0))

            if warmup_epochs > 0:
                base_lr = float(getattr(self.config.training.optimizer, 'lr', 1e-3))
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, T_max - warmup_epochs),
                    eta_min=eta_min
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs]
                )
                self.logger.info(f"✅ 调度器: LinearLR warmup({warmup_epochs}) → CosineAnnealingLR(T_max={max(1, T_max - warmup_epochs)}, eta_min={eta_min})")
            else:
                if name.lower().startswith('cosine'):
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
                    self.logger.info(f"✅ 调度器: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
                else:
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
                    self.logger.info("ℹ️ 未识别调度器名称，已回退到 CosineAnnealingLR")
        except Exception as e:
            self.scheduler = None
            self.logger.warning(f"⚠️ 学习率调度器设置失败，继续训练: {e}")
        
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
    
    def setup_sequential_model(self):
        """设置分阶段预测架构模型"""
        # 安全获取配置 - 修复OmegaConf配置访问
        try:
            sequential_cfg = self._cfg_select('model.sequential', 'sequential', default={})
            # 兼容键：优先 model.sequential.spatial / temporal，其次 fallback 到 model.spatial / model.temporal
            spatial_config = self._cfg_select('model.sequential.spatial', 'sequential.spatial', default=self._cfg_select('model.spatial', 'spatial', default={}))
            temporal_config = self._cfg_select('model.sequential.temporal', 'sequential.temporal', default=self._cfg_select('model.temporal', 'temporal', default={}))
            
            self.logger.info(f"空间配置: {spatial_config}")
            self.logger.info(f"时序配置: {temporal_config}")
            
        except Exception as e:
            self.logger.error(f"配置解析失败: {e}")
            raise
        
        # 初始化分阶段模型
        self.sequential_model = SequentialSpatiotemporalModel(
            spatial_config=spatial_config,
            temporal_config=temporal_config,
            data_config=self.config.data,
            device=self.device
        ).to(self.device)
        
        # 设置模型精度 - 安全获取AMP配置
        try:
            amp_enabled = bool(self._cfg_select('training.amp.enabled', 'model.amp.enabled', default=False))
            if amp_enabled:
                # 保持权重为FP32；通过autocast控制计算精度，避免权重类型转换导致数值不稳定
                pass
        except Exception as e:
            self.logger.warning(f"AMP配置解析失败，使用默认设置: {e}")
            amp_enabled = False
        
        # 分布式训练设置 - 安全获取配置
        try:
            distributed_enabled = bool(self._cfg_select('training.distributed.enabled', 'distributed.enabled', default=False))
            if distributed_enabled:
                self.sequential_model = nn.parallel.DistributedDataParallel(
                    self.sequential_model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True
                )
        except Exception as e:
            self.logger.warning(f"分布式配置解析失败，使用单GPU模式: {e}")
        
        # 初始化一致性检查器 - 安全获取配置
        try:
            consistency_config = self._cfg_select('model.sequential.consistency', 'sequential.consistency', default={})
            self.consistency_checker = SequentialConsistencyChecker(config=consistency_config)
        except Exception as e:
            self.logger.warning(f"一致性检查器配置失败，使用默认配置: {e}")
            self.consistency_checker = SequentialConsistencyChecker(config={})
        
        # 初始化分阶段训练器
        self.setup_sequential_trainers()
        
        self.logger.info(f"分阶段模型设置完成: {type(self.sequential_model).__name__}")
        self.logger.info(f"模型参数量: {sum(p.numel() for p in self.sequential_model.parameters()):,}")
    
    def setup_sequential_trainers(self):
        """设置分阶段训练器"""
        # 安全获取配置
        spatial_config = self._cfg_select('model.sequential.spatial', 'sequential.spatial', default={})
        temporal_config = self._cfg_select('model.sequential.temporal', 'sequential.temporal', default={})
        sequential_config = self._cfg_select('model.sequential', 'sequential', default={})
        
        # 空间预测训练器
        self.spatial_trainer = SpatialTrainer(
            model=self.sequential_model.spatial_module,
            config=spatial_config
        )
        
        # 时序预测训练器
        self.temporal_trainer = TemporalTrainer(
            model=self.sequential_model.temporal_module,
            config=temporal_config
        )
        
        # 联合训练器 - 使用已有的模型实例
        self.sequential_trainer = SequentialSpatiotemporalTrainer(
            config=sequential_config
        )
        # 覆盖模型为已创建的实例
        self.sequential_trainer.model = self.sequential_model
        
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
        # 初始化TensorBoard writer（若尚未创建），避免重复事件文件
        try:
            if not hasattr(self, 'writer') or self.writer is None:
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
                with torch.no_grad():
                    if isinstance(batch, dict) and 'input_sequence' in batch and 'target_sequence' in batch:
                        x = batch['input_sequence'].to(self.device, non_blocking=True)
                        tgt = batch['target_sequence'].to(self.device, non_blocking=True)
                    else:
                        continue
                    t2 = time.time()
                    # 统一调用：支持SequentialSpatiotemporalModel和传统模型
                    model = self.get_model()
                    try:
                        if hasattr(model, 'forward') and hasattr(model, 'spatial_forward'):
                            # SequentialSpatiotemporalModel模式 - 需要完整的时序输入
                            _ = model(x, tgt)
                        else:
                            # 传统模型模式：ARWrapper需要 (x, T_out, tgt)
                            _ = model(x, current_T_out, tgt)
                    except TypeError:
                        # 退化为通用接口 forward(x)
                        _ = model(x)
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
            if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer') and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception:
                    pass
            if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler') and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception:
                    pass
            
            if 'scaler_state_dict' in checkpoint and getattr(self, 'scaler', None) is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # 加载训练状态
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', self.training_history)
            
            self.logger.info(f"✅ Successfully loaded checkpoint: {checkpoint_path}")
            self.logger.info(f"Restored to epoch {self.current_epoch}, best val loss: {self.best_val_loss:.6f}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint, fallback to current model: {str(e)}")
            return False
    
    def create_visualizations(self, sample_batch: Optional[Dict] = None, epoch: int = 0):
        try:
            out_dir_runs = self.output_dir / "visualizations"
            out_dir_runs.mkdir(parents=True, exist_ok=True)
            out_dir_pkg = Path("paper_package/figs") / self.output_dir.name
            out_dir_pkg.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        if self.visualizer is not None:
            try:
                if sample_batch is not None:
                    self.visualizer.save_sample_batch(sample_batch, epoch)
                try:
                    self.visualizer.save_training_curves(self.training_history)
                except Exception:
                    pass
                self.logger.info(f"Saved visualization samples and training curves for epoch {epoch}")
            except Exception:
                pass
            return
        if sample_batch is None:
            return
        try:
            device = self.device
            input_seq = sample_batch.get("input_sequence")
            target_seq = sample_batch.get("target_sequence")
            if not isinstance(input_seq, torch.Tensor) or not isinstance(target_seq, torch.Tensor):
                return
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            x = input_seq[:, 0, 0:1]
            x_in = x  # 仅使用观测数据，禁用坐标和掩码
            self.get_model().eval()
            with torch.no_grad():
                y = self.get_model()(x_in)
            gt = target_seq[:, 0, 0:1]
            err = (y - gt).abs()
            import numpy as np
            import matplotlib.pyplot as plt
            
            # 获取归一化统计信息进行反归一化
            norm_stats = getattr(self, 'norm_stats', None)
            if norm_stats is not None and 'mean' in norm_stats and 'std' in norm_stats:
                mean = norm_stats['mean']
                std = norm_stats['std']
                # 确保mean和std是标量或可以广播到正确形状
                if isinstance(mean, torch.Tensor):
                    mean_val = float(mean[0]) if mean.numel() > 0 else 0.0
                else:
                    mean_val = float(mean) if np.isscalar(mean) else 0.0
                    
                if isinstance(std, torch.Tensor):
                    std_val = float(std[0]) if std.numel() > 0 else 1.0
                else:
                    std_val = float(std) if np.isscalar(std) else 1.0
            else:
                # 如果没有归一化统计信息，使用默认值
                mean_val = 0.0
                std_val = 1.0
                self.logger.warning("⚠️ 未找到归一化统计信息，可视化使用z-score域数据")
            
            n = min(B, 8)
            paths = []
            for b in range(n):
                # 反归一化到真实数据尺度
                obs_img = x[b, 0].detach().cpu().numpy() * std_val + mean_val
                gt_img = gt[b, 0].detach().cpu().numpy() * std_val + mean_val
                pr_img = y[b, 0].detach().cpu().numpy() * std_val + mean_val
                er_img = err[b, 0].detach().cpu().numpy() * std_val  # 误差也需要缩放
                
                # 统一颜色范围用于Obs/GT/Pred（物理量）
                vmin_phys = float(min(np.min(obs_img), np.min(gt_img), np.min(pr_img)))
                vmax_phys = float(max(np.max(obs_img), np.max(gt_img), np.max(pr_img)))
                
                # 误差图使用对称范围，便于观察正负误差
                abs_max_err = float(max(np.abs(np.min(er_img)), np.abs(np.max(er_img))))
                vmin_err = -abs_max_err
                vmax_err = abs_max_err
                
                # 创建更合理的布局：4连图，colorbar在最右侧
                fig = plt.figure(figsize=(16, 4))
                
                # 创建网格布局，为colorbar预留右侧空间
                gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.05)
                
                # Obs - 使用统一物理量颜色范围
                ax0 = fig.add_subplot(gs[0])
                im0 = ax0.imshow(obs_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys)
                ax0.set_title("Obs", fontsize=11, fontweight='bold')
                
                # GT - 使用统一物理量颜色范围  
                ax1 = fig.add_subplot(gs[1])
                im1 = ax1.imshow(gt_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys)
                ax1.set_title("GT", fontsize=11, fontweight='bold')
                
                # Pred - 使用统一物理量颜色范围
                ax2 = fig.add_subplot(gs[2])
                im2 = ax2.imshow(pr_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys)
                ax2.set_title("Pred", fontsize=11, fontweight='bold')
                
                # Error - 使用对称的误差颜色范围
                ax3 = fig.add_subplot(gs[3])
                im3 = ax3.imshow(er_img, cmap="coolwarm", vmin=vmin_err, vmax=vmax_err)
                ax3.set_title("Error", fontsize=11, fontweight='bold')
                
                # 移除坐标轴刻度，保持简洁
                for ax in [ax0, ax1, ax2, ax3]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # 添加统一的颜色条 - 物理量（在最右侧）
                cbar_phys = fig.colorbar(im0, cax=fig.add_subplot(gs[4]), orientation='vertical')
                cbar_phys.set_label('Physical Value', fontsize=10, fontweight='bold')
                cbar_phys.ax.tick_params(labelsize=9)
                
                fig.suptitle(f'Epoch {epoch} - Sample {b}', fontsize=13, fontweight='bold', y=0.95)
                fig.tight_layout()
                
                # 保存图像
                p_run_png = out_dir_runs / f"epoch_{epoch:04d}_sample_{b:03d}.png"
                p_pkg_png = out_dir_pkg / f"epoch_{epoch:04d}_sample_{b:03d}.png"
                p_run_svg = out_dir_runs / f"epoch_{epoch:04d}_sample_{b:03d}.svg"
                p_pkg_svg = out_dir_pkg / f"epoch_{epoch:04d}_sample_{b:03d}.svg"
                
                plt.savefig(p_run_png, dpi=200, bbox_inches='tight', pad_inches=0.1)
                plt.savefig(p_pkg_png, dpi=200, bbox_inches='tight', pad_inches=0.1)
                plt.savefig(p_run_svg, bbox_inches='tight', pad_inches=0.1)
                plt.savefig(p_pkg_svg, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                
                paths.append(p_pkg_svg)
            index_html = out_dir_pkg / "index.html"
            try:
                with open(index_html, "w") as f:
                    f.write("<html><body>")
                    for p in paths:
                        f.write(f"<img src='{p.name}' style='width:800px'><br/>")
                    f.write("</body></html>")
            except Exception:
                pass
            self.logger.info("Saved fallback visualizations")
        except Exception:
            pass

    def create_test_visualizations(self, final_test_metrics: Optional[Dict] = None):
        """Generate test-phase visualizations and export to paper_package/figs."""
        try:
            out_dir = self.output_dir / "visualizations"
            out_dir.mkdir(parents=True, exist_ok=True)
            paper_dir = Path("paper_package/figs") / self.output_dir.name
            paper_dir.mkdir(parents=True, exist_ok=True)

            # 保存测试指标摘要
            if isinstance(final_test_metrics, dict):
                with open(out_dir / "final_test_metrics.json", 'w') as f:
                    json.dump(convert_numpy_types(final_test_metrics), f, indent=2)

            # 若可视化器可用，导出若干测试样本图像
            samples_saved = 0
            try:
                if hasattr(self, 'visualizer') and self.visualizer is not None:
                    samples_saved = self.visualizer.save_test_samples(self.test_loader, out_dir, max_batches=3)
            except Exception:
                pass

            # 生成简易索引页面
            idx_path = out_dir / "index.html"
            with open(idx_path, 'w') as f:
                f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Test Visualizations</title></head><body>")
                f.write("<h1>Test Visualizations</h1>")
                f.write(f"<p>Samples saved: {samples_saved}</p>")
                f.write("<ul>")
                for p in sorted(out_dir.glob('*.png')):
                    f.write(f"<li><img src='{p.name}' style='max-width:512px' /></li>")
                f.write("</ul></body></html>")

            # 拷贝到论文包目录
            try:
                import shutil
                for p in out_dir.glob('*'):
                    shutil.copy2(p, paper_dir / p.name)
            except Exception:
                pass

            self.logger.info("🖼️ Test-phase visualizations exported to paper_package/figs")
        except Exception as _tviz_err:
            self.logger.warning(f"create_test_visualizations failed: {_tviz_err}")

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
                'model_state_dict': self.get_model().state_dict(),
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
        # Write summary
        try:
            out_path = self.output_dir / 'resource_summary.json'
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"📊 Resource summary saved: {out_path}")
        except Exception as _sum_err:
            self.logger.debug(f"Failed to write resource summary: {_sum_err}")
        # 配置开关：可视化总开关
        try:
            viz_enabled = bool(self._cfg_select('visualization.enabled', default=True))
        except Exception:
            viz_enabled = True

        if not viz_enabled:
            self.logger.info("⚪ Visualization disabled by config, skipping generation")
            return

        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization module unavailable, skipping visualization generation")
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
                    self.get_model().eval()
                    with torch.no_grad():
                        current_T_out = self.get_current_T_out(epoch)
                        pred_seq = self.get_model()(input_seq.to(self.device), current_T_out).cpu()
                    self.get_model().train()
                
                # 创建AR预测可视化
                # 仅在使用ARWrapper时可视化AR预测；顺序模型跳过该可视化以避免维度不匹配
                if hasattr(self.get_model(), 'autoregressive_predict'):
                    ar_visualizer.visualize_ar_predictions(
                        input_seq, target_seq, pred_seq, timestep_idx=epoch, 
                        save_name=f"ar_predictions_epoch_{epoch}",
                        norm_stats=self.norm_stats
                    )
                
                # 创建误差分析
                # 确保norm_stats存在
                self.ensure_norm_stats()
                ar_visualizer.create_error_analysis(target_seq, pred_seq, 
                                                   save_name=f"error_analysis_epoch_{epoch}",
                                                   norm_stats=self.norm_stats)
                
                # 创建时间分析
                # 确保norm_stats存在
                self.ensure_norm_stats()
                ar_visualizer.create_temporal_analysis(pred_seq, target_seq,
                                                     save_name=f"temporal_analysis_epoch_{epoch}",
                                                     norm_stats=self.norm_stats)
            
            self.logger.info(f"✅ Visualizations saved to {viz_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate visualizations: {e}")
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
                        stage_T_out = int(self._cfg_select("data.T_out", default=1))
                    except Exception:
                        stage_T_out = 1
                return int(stage_T_out)
        
        # 若超出所有阶段，使用最后一个阶段的T_out，若缺失则回退
        last = stages[-1]
        last_T_out = last.get('T_out', None) if isinstance(last, dict) else getattr(last, 'T_out', None)
        if last_T_out is None:
            try:
                last_T_out = int(self._cfg_select("data.T_out", default=1))
            except Exception:
                last_T_out = 1
        return int(last_T_out)
    
    def ensure_norm_stats(self):
        """确保norm_stats已初始化，避免AttributeError"""
        if not hasattr(self, 'norm_stats') or self.norm_stats is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("⚠️ norm_stats未初始化，使用默认值")
            # 安全获取输出通道数
            try:
                C = int(self._cfg_select('model.out_channels', 'data.target_channels', default=1))
            except Exception:
                C = 1  # 最坏情况默认值
            self.norm_stats = {
                'mean': torch.zeros(C),
                'std': torch.ones(C),
                'u_mean': torch.tensor(0.0),
                'u_std': torch.tensor(1.0),
                'v_mean': torch.tensor(0.0),
                'v_std': torch.tensor(1.0)
            }

    def get_model(self):
        """获取当前使用的模型（兼容ARWrapper和SequentialSpatiotemporalModel）"""
        if hasattr(self, 'model') and self.model is not None:
            return self.model
        elif hasattr(self, 'sequential_model') and self.sequential_model is not None:
            return self.sequential_model
        else:
            raise RuntimeError("未找到可用的模型 (既无self.model也无self.sequential_model)")
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        model_to_train = self.get_model()
        model_to_train.train()
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
        try:
            import torch.distributed as dist
            is_primary = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        except Exception:
            is_primary = True
        progress_bar = (tqdm(self.train_loader, desc=f"Epoch {epoch+1}", mininterval=0.5, smoothing=0.0, leave=True, dynamic_ncols=True) if is_primary else self.train_loader)
        
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
                # 设备搬运耗时起点
                load_t0 = time.perf_counter()
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(self.device, non_blocking=True)  # [B, T_in, C, H, W]
                target_seq = batch['target_sequence'].to(self.device, non_blocking=True)  # [B, T_out, C, H, W]
                data_end = time.perf_counter()
                
                
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
                    # 统一：若是SequentialSpatiotemporalModel，则始终做时序前向并计算序列损失
                    model = self.get_model()
                    is_seq_model = hasattr(model, 'spatial_forward') and hasattr(model, 'temporal_forward')
                    if is_seq_model:
                        if torch.isnan(input_seq).any() or torch.isinf(input_seq).any():
                            print(f"WARNING: NaN/Inf detected in input_seq")
                            input_seq = torch.nan_to_num(input_seq, nan=0.0, posinf=1e6, neginf=-1e6)
                        if torch.isnan(target_seq).any() or torch.isinf(target_seq).any():
                            print(f"WARNING: NaN/Inf detected in target_seq")
                            target_seq = torch.nan_to_num(target_seq, nan=0.0, posinf=1e6, neginf=-1e6)
                        model_output = model(input_seq, target_seq)
                        pred_seq = model_output['final_pred']
                        if torch.isnan(pred_seq).any() or torch.isinf(pred_seq).any():
                            print(f"WARNING: NaN/Inf detected in pred_seq from model")
                            pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=1e6, neginf=-1e6)
                    else:
                        # 非顺序模型：按原逻辑（ARWrapper/传统模型）
                        if hasattr(self, 'config') and hasattr(self.config, 'ar') and not bool(getattr(self.config.ar, 'enabled', True)):
                            # 空间-only路径：保持原来单帧损失
                            x_single = input_seq[:, 0]
                            if hasattr(model, 'spatial_forward'):
                                spatial_output = model.spatial_forward(x_single)
                                y_single = spatial_output.spatial_pred
                            else:
                                y_single = model(x_single)
                            target_single = target_seq[:, 0]
                            obs_data_single = {
                                'observation': None,
                                'baseline': x_single,
                                'h_params': self.h_params if hasattr(self, 'h_params') and self.h_params is not None else {
                                    'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
                                }
                            }
                            from ops.losses import compute_total_loss
                            self.ensure_norm_stats()
                            losses = compute_total_loss(
                                pred_z=y_single,
                                target_z=target_single,
                                obs_data=obs_data_single,
                                norm_stats=self.norm_stats,
                                config=self.config
                            )
                            loss = losses['total_loss']
                        else:
                            pred_seq = model(input_seq, current_T_out, target_seq)

                    # 统一损失装配（z-score域重建 + 原值域谱/DC）
                    # 顺序模型或AR路径下的序列损失
                    from ops.losses import compute_ar_total_loss
                    if is_seq_model or (hasattr(self, 'config') and hasattr(self.config, 'ar') and bool(getattr(self.config.ar, 'enabled', True))):
                        obs_data = {
                            'observation_seq': None,
                            'baseline_seq': input_seq,
                            'h_params': {
                                'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
                            }
                        }
                        self.ensure_norm_stats()
                        losses = compute_ar_total_loss(
                            pred_seq=pred_seq,
                            gt_seq=target_seq,
                            obs_data=obs_data,
                            norm_stats=self.norm_stats,
                            config=self.config
                        )
                        loss = losses['total_loss']

                    loss = loss / accumulation_steps
                
                model = self.get_model()
                use_no_sync = hasattr(model, 'no_sync') and accumulation_steps > 1 and ((batch_idx + 1) % accumulation_steps != 0) and ((batch_idx + 1) != num_batches)
                if use_no_sync:
                    ctx = model.no_sync()
                else:
                    class _NullCtx2:
                        def __enter__(self):
                            return None
                        def __exit__(self, exc_type, exc, tb):
                            return False
                    ctx = _NullCtx2()
                with ctx:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # 每accumulation_steps步或最后一个batch时更新参数
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    # 梯度裁剪
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    model = self.get_model()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clip_val)

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
                
                # 记录当前batch结束CPU时间用于下一次fetch耗时估算
                prev_batch_end_cpu = time.perf_counter()

                
                
            except RuntimeError as e:
                if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                    # 使用改进的CUDA错误处理
                    if self.handle_cuda_error(e, "training"):
                        self.logger.info("CUDA错误已处理，重新开始当前epoch")
                        # 重新开始当前epoch
                        return self.train_epoch(epoch)
                    else:
                        # 如果无法处理CUDA错误，跳过这个batch
                        self.logger.warning("无法处理CUDA错误，跳过当前batch")
                        continue
                else:
                    raise e
            
            # 更新进度条
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}', 'T_out': current_T_out})
            
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
            
            
            
            
        
        avg_loss = total_loss / max(1, num_batches)
        
        self.stage_epoch += 1
        
        return avg_loss
    
    def _validate_epoch_legacy(self, epoch: int) -> Tuple[float, Dict[str, float], Optional[Dict]]:
        """验证一个epoch"""
        # 获取当前模型（兼容ARWrapper和SequentialSpatiotemporalModel）
        model_to_validate = self.get_model()
        model_to_validate.eval()
        total_loss = 0.0
        all_metrics = []
        num_batches = len(self.val_loader)
        
        current_T_out = self.get_current_T_out(epoch)
        sample_batch = None  # 保存一个样本用于可视化
        
        with torch.no_grad():
            try:
                import torch.distributed as dist
                is_primary = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
            except Exception:
                is_primary = True
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", leave=False) if is_primary else self.val_loader):
                try:
                    input_seq = batch['input_sequence'].to(self.device)
                    target_seq = batch['target_sequence'].to(self.device)
                    
                    # 根据课程学习调整目标序列长度
                    if target_seq.shape[1] > current_T_out:
                        target_seq = target_seq[:, :current_T_out]
                    
                    with autocast(device_type='cuda', dtype=getattr(self, 'autocast_dtype', torch.bfloat16), enabled=(self.device.type == 'cuda')):
                        # 使用专用时序模型或传统模型进行验证预测
                        model = self.get_model()
                        if hasattr(model, 'forward') and hasattr(model, 'spatial_forward'):
                            # SequentialSpatiotemporalModel模式 - 需要完整的时序输入和目标
                            model_output = model(input_seq, target_seq)
                            pred_seq = model_output['final_pred']
                        else:
                            # 传统模型模式
                            pred_seq = model(input_seq, current_T_out)
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
                                'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
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
                    # 指标：统一使用最后一个时间步 - GPU优化版本
                    # 确保张量维度正确 [B, C, H, W]
                    print(f"DEBUG: pred_seq shape = {pred_seq.shape}, dim = {pred_seq.dim()}")
                    print(f"DEBUG: target_seq shape = {target_seq.shape}, dim = {target_seq.dim()}")
                    
                    if pred_seq.dim() == 5:  # [B, T, C, H, W]
                        pred_last = pred_seq[:, -1]  # [B, C, H, W]
                    elif pred_seq.dim() == 4:  # [B, C, H, W] or [B, T, H, W]
                        if pred_seq.shape[1] == 1:  # [B, 1, H, W] - assume this is [B, T, H, W]
                            pred_last = pred_seq.squeeze(1)  # [B, H, W]
                            pred_last = pred_last.unsqueeze(1)  # [B, 1, H, W] - add channel dim
                        else:
                            pred_last = pred_seq  # [B, C, H, W]
                    else:
                        self.logger.warning(f"Unexpected pred_seq dimensions: {pred_seq.dim()}")
                        continue
                        
                    if target_seq.dim() == 5:  # [B, T, C, H, W]
                        target_last = target_seq[:, -1]  # [B, C, H, W]
                    elif target_seq.dim() == 4:  # [B, C, H, W] or [B, T, H, W]
                        if target_seq.shape[1] == 1:  # [B, 1, H, W] - assume this is [B, T, H, W]
                            target_last = target_seq.squeeze(1)  # [B, H, W]
                            target_last = target_last.unsqueeze(1)  # [B, 1, H, W] - add channel dim
                        else:
                            target_last = target_seq  # [B, C, H, W]
                    else:
                        self.logger.warning(f"Unexpected target_seq dimensions: {target_seq.dim()}")
                        continue
                        
                    print(f"DEBUG: pred_last shape = {pred_last.shape}")
                    print(f"DEBUG: target_last shape = {target_last.shape}")
                    try:
                        # 使用GPU优化的指标计算，避免CPU转移
                        from ops.metrics import compute_all_metrics
                        batch_metrics = compute_all_metrics(pred_last, target_last, use_gpu_ssim=True)
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
                    if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                        # 使用改进的CUDA错误处理
                        if self.handle_cuda_error(e, "validation"):
                            self.logger.info("验证时CUDA错误已处理，继续下一个batch")
                            continue
                        else:
                            self.logger.warning("无法处理验证时CUDA错误，跳过当前batch")
                            continue
                    else:
                        self.logger.error(f"验证时发生错误 batch {batch_idx}: {e}")
                        continue
        
        avg_loss = total_loss / max(1, num_batches)
        
        # 计算平均指标
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_loss, avg_metrics, sample_batch
    
    def test_epoch(self) -> Dict[str, float]:
        """测试集评估"""
        self.logger.info("🧪 开始测试集评估...")
        # 获取当前模型（兼容ARWrapper和SequentialSpatiotemporalModel）
        model_to_test = self.get_model()
        model_to_test.eval()
        
        # 检查test_loader是否存在且不为None
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            self.logger.warning("⚠️ test_loader不存在或为None，跳过测试评估")
            return {'test_loss': 0.0, 'test_metrics': {}}
        
        total_loss = 0.0
        all_metrics = []
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            try:
                import torch.distributed as dist
                is_primary = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
            except Exception:
                is_primary = True
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing", leave=False) if is_primary else self.test_loader):
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # 模型预测（测试时不使用teacher forcing），输出长度与目标序列一致
                test_T_out = target_seq.shape[1]
                model = self.get_model()
                is_seq_model = hasattr(model, 'spatial_forward') and hasattr(model, 'temporal_forward')
                if is_seq_model:
                    model_output = model(input_seq, target_seq)
                    pred_seq = model_output['final_pred']
                else:
                    try:
                        ar_enabled = bool(getattr(self.config, 'ar', {}).get('enabled', True))
                    except Exception:
                        ar_enabled = True
                    if ar_enabled:
                        if hasattr(model, 'autoregressive_predict'):
                            pred_seq = model.autoregressive_predict(input_seq, test_T_out, teacher=None, train_mode=False)
                        else:
                            pred_seq = model(input_seq, test_T_out)
                    else:
                        x_single = input_seq[:, 0]
                        if hasattr(model, 'spatial_forward'):
                            y_single = model.spatial_forward(x_single).spatial_pred
                        else:
                            y_single = model(x_single)
                        pred_seq = y_single[:, None]
                
                # 计算损失（与训练/验证口径一致：Rel-L2 + MAE）
                from ops.losses import rel_l2, l1_mae
                loss = rel_l2(pred_seq, target_seq) + l1_mae(pred_seq, target_seq)
                total_loss += loss.item()
                
                # 计算详细指标 - GPU优化版本
                try:
                    # 使用GPU优化的指标计算，避免CPU转移
                    from ops.metrics import compute_all_metrics
                    batch_metrics = compute_all_metrics(pred_seq, target_seq, use_gpu_ssim=True)
                    all_metrics.append(batch_metrics)
                except Exception as metrics_error:
                    self.logger.warning(f"指标计算失败 batch {batch_idx}: {metrics_error}")
                    # 跳过这个batch的指标计算，但继续训练
                    continue
        
        # 聚合指标
        avg_loss = total_loss / max(1, num_batches)
        
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
        self.get_model().eval()
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
            try:
                import torch.distributed as dist
                is_primary = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
            except Exception:
                is_primary = True
            val_iter = self.val_loader
            for batch_idx, batch in enumerate(val_iter):
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(self.device, non_blocking=True)  # [B, T_in, C, H, W]
                target_seq = batch['target_sequence'].to(self.device, non_blocking=True)  # [B, T_out, C, H, W]

                # 根据课程学习调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]
                # 根据配置分支：空间-only 与 AR
                try:
                    ar_enabled = bool(getattr(self.config, 'ar', {}).get('enabled', True))
                except Exception:
                    ar_enabled = True

                use_amp = (self.device.type == 'cuda')
                amp_ctx = autocast(device_type='cuda', dtype=getattr(self, 'autocast_dtype', torch.bfloat16), enabled=use_amp) if use_amp else None
                if amp_ctx is None:
                    class _NullCtx:
                        def __enter__(self):
                            return None
                        def __exit__(self, exc_type, exc, tb):
                            return False
                    amp_ctx = _NullCtx()

                if not ar_enabled:
                    # 空间-only：单帧前向与空间损失
                    with amp_ctx:
                        x_single = input_seq[:, 0]
                        if isinstance(x_single, torch.Tensor):
                            B, C, H, W = x_single.shape
                            if C == 1:
                                # 仅使用观测数据，禁用坐标和掩码
                                pass  # x_single 保持单通道
                            elif C == 2:
                                # 保持2通道（观测+掩码），禁用坐标
                                pass
                        # 使用专用时序模型进行空间预测
                        model = self.get_model()
                        if hasattr(model, 'spatial_forward'):
                            # SequentialSpatiotemporalModel模式
                            spatial_output = model.spatial_forward(x_single)
                            y_single = spatial_output.spatial_pred
                        else:
                            # 传统模型模式
                            y_single = model(x_single)
                        target_single = target_seq[:, 0]
                        obs_data_single = {
                            'observation': None,
                            'baseline': x_single,
                            'h_params': self.h_params if hasattr(self, 'h_params') and self.h_params is not None else {
                                'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
                            }
                        }
                        from ops.losses import compute_total_loss
                        losses = compute_total_loss(
                            pred_z=y_single,
                            target_z=target_single,
                            obs_data=obs_data_single,
                            norm_stats=self.norm_stats,
                            config=self.config
                        )
                        loss = losses['total_loss']
                    total_loss += loss.item()

                    
                else:
                    # AR验证路径：与原逻辑一致
                    with amp_ctx:
                        # 使用专用时序模型或ARWrapper进行训练预测
                        model = self.get_model()
                        if hasattr(model, 'forward') and hasattr(model, 'spatial_forward'):
                            # SequentialSpatiotemporalModel模式 - 需要完整的时序输入和目标
                            model_output = model(input_seq, target_seq)
                            pred_seq = model_output['final_pred']
                        else:
                            # 传统模型模式：ARWrapper需要 (input_seq, current_T_out, target_seq)
                            pred_seq = model(input_seq, current_T_out, target_seq)
                        observation_seq = None

                        obs_data = {
                            'observation_seq': observation_seq,
                            'baseline_seq': input_seq,
                            'h_params': self.h_params if hasattr(self, 'h_params') and self.h_params is not None else {
                                'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
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
            model = self.get_model()
            model_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        except Exception:
            model = self.get_model()
            model_state_cpu = model.state_dict()
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
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = Path(str(path) + '.tmp')
                torch.save(obj, tmp_path)
                # 如果临时文件未创建成功，直接回退到普通保存
                if not tmp_path.exists():
                    torch.save(obj, path)
                    return
                os.replace(tmp_path, path)
            except Exception as e:
                # 原子写失败时回退到普通保存，避免训练中断
                try:
                    torch.save(obj, path)
                    self.logger.warning(f"⚠️ 原子保存失败，已回退普通保存: {e}")
                except Exception as e2:
                    self.logger.error(f"❌ 检查点保存失败: {e2}")
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
            paper_test_dir = Path("paper_package/figs") / f"{self.output_dir.name}_test"
            paper_test_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化AR可视化器
            ar_visualizer = ARTrainingVisualizer(str(test_viz_dir))
            
            # 获取测试数据样本进行可视化
            self.get_model().eval()
            test_samples_visualized = 0
            max_test_samples = 5  # 可视化前5个测试样本
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_loader):
                    if test_samples_visualized >= max_test_samples:
                        break
                    
                    # 准备输入/目标数据
                    # 优先使用SR观测作为可视化的输入帧，确保与训练退化一致
                    target_seq = batch['target_sequence'].to(self.device)

                    # 原始输入序列（可能是高分辨率或已退化回升尺寸的序列）
                    input_seq_raw = batch.get('input_sequence', None)
                    if input_seq_raw is not None:
                        input_seq_raw = input_seq_raw.to(self.device)

                    # 选择用于可视化的输入序列，优先观测/基线
                    input_seq_vis = None
                    try:
                        # 1) 直接使用时序观测（若提供）
                        if 'observation_sequence' in batch and batch['observation_sequence'] is not None:
                            obs_seq = batch['observation_sequence']
                            input_seq_vis = obs_seq.to(self.device)
                        # 2) 使用通用观测字段（形状可能非时序）
                        elif 'observation' in batch and batch['observation'] is not None:
                            obs = batch['observation']
                            # 期望形状：[B, T_in, C, H, W] 或 [T_in, C, H, W]
                            if obs.dim() == 5:
                                input_seq_vis = obs.to(self.device)
                            else:
                                # 将非时序观测扩展为时序长度，使用最后一帧重复
                                t_in = input_seq_raw.shape[0] if input_seq_raw is not None and input_seq_raw.dim() >= 1 else 1
                                if obs.dim() == 4:  # [B, C, H, W]
                                    obs = obs[0]  # 取第一个样本
                                if obs.dim() == 3:  # [C, H, W]
                                    obs = obs.unsqueeze(0).repeat(t_in, 1, 1, 1)  # [T_in, C, H, W]
                                input_seq_vis = obs.to(self.device)
                        # 3) 使用baseline（可能为上采样后的SR观测）
                        elif 'baseline' in batch and batch['baseline'] is not None:
                            base = batch['baseline']
                            t_in = input_seq_raw.shape[0] if input_seq_raw is not None and input_seq_raw.dim() >= 1 else 1
                            if base.dim() == 5:
                                input_seq_vis = base.to(self.device)
                            else:
                                if base.dim() == 4:  # [B, C, H, W]
                                    base = base[0]
                                if base.dim() == 3:  # [C, H, W]
                                    base = base.unsqueeze(0).repeat(t_in, 1, 1, 1)
                                else:
                                    # 处理 [T_in*C, H, W] 的flatten格式（来自某些时序数据集）
                                    if base.dim() == 3 and input_seq_raw is not None and input_seq_raw.dim() == 4:
                                        C = input_seq_raw.shape[1]
                                        H, W = base.shape[-2:]
                                        # 尝试恢复为 [T_in, C, H, W]
                                        try:
                                            base = base.view(t_in, C, H, W)
                                        except Exception:
                                            # 回退：重复单帧
                                            base = base.unsqueeze(0).repeat(t_in, 1, 1, 1)
                                input_seq_vis = base.to(self.device)
                    except Exception:
                        # 回退到原始输入序列
                        input_seq_vis = input_seq_raw if input_seq_raw is not None else None

                    if input_seq_vis is None:
                        # 最终回退：使用原始输入序列
                        input_seq_vis = input_seq_raw

                    # 模型前向仍使用原始输入序列，避免训练逻辑变化
                    input_seq = input_seq_raw
                    
                    # 获取当前T_out
                    current_T_out = target_seq.shape[1]
                    
                    if hasattr(self.get_model(), 'autoregressive_predict'):
                        pred_seq = self.get_model().autoregressive_predict(input_seq, T_out=current_T_out, teacher=None, train_mode=False)
                    else:
                        x = input_seq
                        if x.dim() == 5:
                            x = x[:, -1]
                        y = self.get_model()(x)
                        # Handle dictionary output from SequentialSpatiotemporalModel
                        if isinstance(y, dict):
                            if 'final_pred' in y:
                                pred_seq = y['final_pred']
                            else:
                                # Fallback: use the first tensor value
                                pred_seq = list(y.values())[0]
                        else:
                            pred_seq = y.unsqueeze(1)
                    
                    # 转换为numpy数组用于可视化
                    input_np = input_seq_vis.cpu().numpy()
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
                        
                        self.logger.info(f"📊 Generating visualization for test sample {test_samples_visualized + 1}...")
                        
                    # 1. 预测可视化（顺序模型与AR分别处理）
                    self.ensure_norm_stats()
                    if hasattr(self.get_model(), 'autoregressive_predict'):
                        ar_visualizer.visualize_ar_predictions(
                            sample_input, sample_target, sample_pred,
                            save_name=f"{sample_name}_ar_predictions",
                            norm_stats=self.norm_stats
                        )
                    else:
                        # 顺序模型：可视化最后一步的 Obs/GT/Pred，避免序列图与形状不一致
                        try:
                            obs = sample_input[:, -1]
                            gt = sample_target[:, -1]
                            pr = sample_pred[:, -1]
                            ar_visualizer.visualize_single_frame(obs, gt, pr,
                                save_name=f"{sample_name}_seq_last_frame",
                                norm_stats=self.norm_stats)
                        except Exception:
                            pass
                        
                    # 2. 误差分析（先对齐时间长度与空间维度）
                    self.ensure_norm_stats()
                    try:
                        tgt = sample_target
                        pred = sample_pred
                        # 对齐T
                        T_tgt = tgt.shape[1]
                        T_pred = pred.shape[1]
                        if T_pred != T_tgt:
                            if T_pred > T_tgt:
                                pred = pred[:, :T_tgt]
                            else:
                                pred = np.concatenate([pred, pred[:, -1:].repeat(T_tgt - T_pred, axis=1)], axis=1)
                        # 对齐H,W
                        H_t, W_t = tgt.shape[-2], tgt.shape[-1]
                        H_p, W_p = pred.shape[-2], pred.shape[-1]
                        if (H_p != H_t) or (W_p != W_t):
                            # 简化处理：只做最后帧误差分析
                            ar_visualizer.create_error_analysis(
                                tgt[:, -1:], pred[:, -1:],
                                save_name=f"{sample_name}_error_analysis_last",
                                norm_stats=self.norm_stats)
                        else:
                            ar_visualizer.create_error_analysis(
                                tgt, pred,
                                save_name=f"{sample_name}_error_analysis",
                                norm_stats=self.norm_stats)
                    except Exception:
                        pass
                        
                    # 3. 时间分析（仅当T一致）
                    self.ensure_norm_stats()
                    try:
                        if sample_pred.shape[1] == sample_target.shape[1]:
                            ar_visualizer.create_temporal_analysis(
                                sample_pred, sample_target,
                                save_name=f"{sample_name}_temporal_analysis",
                                norm_stats=self.norm_stats)
                        else:
                            # 回退：仅分析最后帧（不做时序分析）
                            pass
                    except Exception:
                        pass
                        
                        test_samples_visualized += 1
                        
                        if test_samples_visualized >= max_test_samples:
                            break
            
            # 生成测试指标可视化
            self.logger.info("📈 Generating test metrics visualization...")
            self._create_test_metrics_visualization(test_metrics, test_viz_dir)
            
            # 生成测试阶段HTML报告
            self.logger.info("📄 Generating test phase HTML report...")
            self._create_test_html_report(test_metrics, test_viz_dir, paper_test_dir)
            
            # 复制可视化文件到paper_package
            import shutil
            if test_viz_dir.exists():
                # 复制所有可视化文件
                for file_path in test_viz_dir.glob("*.png"):
                    shutil.copy2(file_path, paper_test_dir)
                for file_path in test_viz_dir.glob("*.html"):
                    shutil.copy2(file_path, paper_test_dir)
                
                self.logger.info(f"📋 Test visualization files copied to {paper_test_dir}")
            
            self.logger.info(f"✅ Test visualizations completed, saved to {test_viz_dir} and {paper_test_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate test visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_test_metrics_visualization(self, test_metrics: Dict[str, float], output_dir: Path):
        """Create test metrics visualization (English labels)."""
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
            
            self.logger.info("📊 Test metrics visualization generated")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate test metrics visualization: {e}")
    
    def _create_test_html_report(self, test_metrics: Dict[str, float], viz_dir: Path, paper_dir: Path):
        """Create English HTML report for the test phase."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AR Model Test Report - {self.config.experiment.name}</title>
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
        <h1>AR Model Test Report</h1>
        
        <div class="info-box">
            <strong>Experiment Name:</strong> {self.config.experiment.name}<br>
            <strong>Model Type:</strong> {self.config.model.name}<br>
            <strong>Test Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Dataset:</strong> Real diffusion-reaction data
        </div>
        
        <h2>📊 Test Metrics Results</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
"""
            
            # 添加指标说明
            metric_descriptions = {
                'mse': 'Mean Squared Error',
                'mae': 'Mean Absolute Error',
                'rel_l2': 'Relative L2 Error',
                'psnr': 'Peak Signal-to-Noise Ratio',
                'ssim': 'Structural Similarity Index',
                'temporal_mse': 'Temporal MSE (temporal consistency error)',
                'long_term_stability': 'Long-term Stability'
            }
            
            for metric_name, metric_value in test_metrics.items():
                description = metric_descriptions.get(metric_name, 'Test Metric')
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
        
        <h2>📈 Metrics Visualization</h2>
        <div class="image-grid">
            <div class="image-item">
                <h3>Metrics Overview</h3>
                <img src="test_metrics.png" alt="Metrics Overview">
            </div>
        </div>
        
        <h2>🎯 Test Samples Visualization</h2>
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
            Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
            
            self.logger.info(f"📄 Test HTML report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate test HTML report: {e}")

    def create_final_report(self):
        """Create final visualization report (English logs)."""
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization module unavailable, skipping final report generation")
            return
        
        try:
            # 创建paper_package目录
            paper_dir = Path("paper_package/figs") / self.output_dir.name
            paper_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用统一可视化器创建综合报告
            visualizer = PDEBenchVisualizer(str(paper_dir))
            
            # 创建综合报告
            visualizer.create_comprehensive_report(str(self.output_dir))
            
            self.logger.info(f"📊 Final visualization report saved to {paper_dir}")
            
            # 复制到paper_package目录
            import shutil
            viz_source = self.output_dir / "visualizations"
            if viz_source.exists():
                shutil.copytree(viz_source, paper_dir, dirs_exist_ok=True)
                self.logger.info(f"📋 Visualization files copied to paper_package")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate final report: {e}")
            import traceback
            traceback.print_exc()
    
    def train(self):
        """Main training loop"""
        self.logger.info("🚀 Starting training...")
        # 明确记录当前模式：空间-only 或 AR
        try:
            ar_cfg = getattr(self.config, 'ar', None)
            ar_enabled = bool(getattr(ar_cfg, 'enabled', True)) if ar_cfg is not None else True
        except Exception:
            ar_enabled = True
        if not ar_enabled:
            # 当禁用AR且 T_in=T_out=1 时，进一步标注空间-only
            try:
                t_in = int(getattr(self.config.data, 'T_in', 1))
                t_out = int(getattr(self.config.data, 'T_out', 1))
            except Exception:
                t_in, t_out = 1, 1
            if t_in == 1 and t_out == 1:
                self.logger.info("🌐 当前训练模式：空间-only（禁用时间预测，T_in=T_out=1）")
            else:
                self.logger.info(f"🌐 当前训练模式：部分时间禁用（AR禁用，T_in={t_in}, T_out={t_out}）")
        else:
            self.logger.info("🕒 当前训练模式：自回归（启用时间预测）")

        start_time = time.time()
        start_epoch = self.current_epoch

        resource_monitor = None
        
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
                try:
                    if self.device.type == 'cuda':
                        self.writer.add_scalar('Resources/GPU_PeakAllocated_GB', float(torch.cuda.max_memory_allocated() / 1024**3), epoch)
                        self.writer.add_scalar('Resources/GPU_PeakReserved_GB', float(torch.cuda.max_memory_reserved() / 1024**3), epoch)
                    if getattr(self, '_process', None) is not None:
                        self.writer.add_scalar('Resources/CPU_Percent', float(self._process.cpu_percent(interval=None)), epoch)
                        self.writer.add_scalar('Resources/SystemMem_Percent', float(psutil.virtual_memory().percent), epoch)
                except Exception:
                    pass
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
                try:
                    rec = {
                        'epoch': int(epoch),
                        'time_sec': float(epoch_time),
                        'throughput_samples_per_sec': float(self._perf_samples / max(1e-6, (self._perf_fetch_time + self._perf_data_time + self._perf_compute_time))),
                        'gpu_peak_allocated_gb': 0.0,
                        'gpu_peak_reserved_gb': 0.0,
                        'cpu_percent': 0.0,
                        'system_memory_percent': 0.0,
                        'iowait_percent': 0.0,
                    }
                    if self.device.type == 'cuda':
                        try:
                            rec['gpu_peak_allocated_gb'] = float(torch.cuda.max_memory_allocated() / 1024**3)
                            rec['gpu_peak_reserved_gb'] = float(torch.cuda.max_memory_reserved() / 1024**3)
                        except Exception:
                            pass
                    if getattr(self, '_process', None) is not None:
                        try:
                            rec['cpu_percent'] = float(self._process.cpu_percent(interval=None))
                            vm = psutil.virtual_memory()
                            rec['system_memory_percent'] = float(vm.percent)
                            ctp = psutil.cpu_times_percent(interval=None)
                            rec['iowait_percent'] = float(getattr(ctp, 'iowait', 0.0))
                        except Exception:
                            pass
                    with open(self.output_dir / 'resources_epoch.jsonl', 'a') as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception as _ep_err:
                    self.logger.debug(f"资源记录写入失败: {_ep_err}")
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
                
                

                # 写入每epoch资源JSONL
                

                # 资源监控指标写入与自适应调优
                

                
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
                    self.logger.info("🛑 Resource monitoring stopped")
            except Exception as e:
                self.logger.warning(f"Failed to stop resource monitoring: {e}")
            total_time = time.time() - start_time
            self.logger.info(f"🏁 Training finished, total time: {total_time/3600:.2f} hours")
            
            # 在训练完成后，根据配置决定是否进行最终测试
            try:
                testing_enabled = bool(getattr(self.config.testing, 'enabled', True))
                run_final_test = bool(getattr(self.config.testing, 'run_final_test', True))
            except Exception:
                testing_enabled, run_final_test = True, True

            if testing_enabled and run_final_test:
                best_ckpt_path = self.output_dir / 'best.ckpt'
                if best_ckpt_path.exists():
                    self.logger.info("📊 Using best model for final test evaluation...")
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
                    
                    self.logger.info("✅ Final test results saved to test_results.json")
                    
                    # 生成测试阶段可视化
                    self.logger.info("🎨 Generating test-phase visualizations...")
                    self.create_test_visualizations(final_test_metrics)
                else:
                    self.logger.info("ℹ️ Best checkpoint not found, using current model for final test evaluation")
                    final_test_metrics = self.test_epoch()
                    test_results = {
                        'final_test_metrics': final_test_metrics,
                        'test_time': time.time(),
                        'model_path': 'current_model'
                    }
                    test_results = convert_numpy_types(test_results)
                    with open(self.output_dir / 'test_results.json', 'w') as f:
                        json.dump(test_results, f, indent=2)
                    self.logger.info("✅ Final test results saved to test_results.json")
                    self.logger.info("🎨 Generating test-phase visualizations...")
                    self.create_test_visualizations(final_test_metrics)
            else:
                self.logger.info("⏭️ testing.disabled; running minimal test-phase visualizations")
                try:
                    final_test_metrics = {}
                    self.create_test_visualizations(final_test_metrics)
                except Exception as _min_viz_err:
                    self.logger.warning(f"Minimal test-phase visualization failed: {_min_viz_err}")
            
            # 生成最终可视化报告
            self.create_final_report()

            # 生成资源摘要报告（平均吞吐/耗时/GPU峰值）
            try:
                self.generate_resource_summary()
            except Exception as _sum_err:
                self.logger.debug(f"Resource summary generation failed: {_sum_err}")

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
    
    # 新增模型选择参数
    parser.add_argument("--model", type=str, default=None, help="模型架构名称（如 swin_unet, unet, fno2d, segformer 等）")
    parser.add_argument("--list-models", action="store_true", help="列出所有可用模型")
    
    args = parser.parse_args()
    
    # 如果请求列出模型，显示后退出
    if args.list_models:
        available_models = list_models()
        print("\n可用模型架构:")
        for i, model in enumerate(available_models, 1):
            info = get_model_info(model)
            if info:
                print(f"  {i:2d}. {model:20s} - {info.get('class_name', 'Unknown')}")
            else:
                print(f"  {i:2d}. {model}")
        print(f"\n总计: {len(available_models)} 个模型\n")
        return
    
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
                trainer = RealDataARTrainer(str(tmp_cfg_path), model_name=args.model)
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
            trainer = RealDataARTrainer(args.config, model_name=args.model)
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
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        raise
    finally:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
    # 结束


if __name__ == "__main__":
    main()
if not logging.getLogger().handlers:
    logging.getLogger().setLevel(logging.INFO)
    sh = StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
