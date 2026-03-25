#!/usr/bin/env python3
"""
真实扩散-反应数据AR训练脚本
专门用于训练真实数据集的20步AR预测模型
"""

import os
import sys

# 将项目根目录添加到系统路径，确保能够导入 utils 模块
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import json
import logging
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# AMP统一接口：优先使用 torch.autocast（带 device_type），GradScaler 保持兼容
try:
    from torch import autocast  # torch.autocast(device_type=...)
except Exception:
    # 兼容旧版：退回到 torch.cuda.amp.autocast
    from torch.cuda.amp import autocast  # type: ignore

import numpy as np
import psutil
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.logger import setup_logger

try:
    from utils.visualization import ARVisualizer
except ImportError:
    ARVisualizer = None


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
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))

# 强制使用唯一的 Dataset 实现
from datasets.darcy_flow_dataset import DarcyFlowDataModule
from datasets.real_diffusion_reaction_dataset import (
    RealDiffusionReactionDataModule,
)
from models.temporal import ARWrapper
from models.temporal.components.sequential_dc_consistency import (
    SequentialConsistencyChecker,
)

# 分阶段预测架构模块
from models.temporal.components.sequential_spatiotemporal import (
    SequentialSpatiotemporalModel,
)
from models.temporal.components.sequential_trainer import (
    SequentialSpatiotemporalTrainer,
    SpatialTrainer,
    TemporalTrainer,
)
from ops.degradation import apply_degradation_operator
from ops.losses import compute_ar_total_loss

# 模型加载器
from tools.training.model_loader import (
    get_model_info,
    list_models,
)
from tools.training.model_loader_enhanced import (
    create_enhanced_model,
)

# 资源监控器的导入在运行时根据可用实现动态处理，避免签名不兼容

# 安全/快速collate（过滤None/低GIL压力），不可用时回退为None
try:
    from utils.collate import fast_collate_fn, safe_collate_fn
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

    def __init__(
        self,
        config_path: str = None,
        model_name: str = None,
        use_liif_decoder: bool = False,
        output_dir_override: Path = None,
        skip_optimizer: bool = False,
        skip_monitoring: bool = False,
        minimal_init: bool = False,
        overrides: list = None,
    ):
        """初始化训练器

        Args:
            config_path: 配置文件路径
            model_name: 模型架构名称（可选，会覆盖配置文件中的模型设置）
            use_liif_decoder: 是否使用LIIF解码器（覆盖配置）
            output_dir_override: 强制指定输出目录（若提供，则不再自动创建带时间戳的新目录）
            overrides: 命令行配置覆盖列表 (list of "key=value")
        """
        self.model_name = model_name  # 保存模型名称参数
        self.use_liif_decoder = use_liif_decoder  # 保存LIIF解码器配置
        self.output_dir_override = (
            Path(output_dir_override) if output_dir_override else None
        )
        self._skip_optimizer = bool(skip_optimizer)
        self._skip_monitoring = bool(skip_monitoring)
        self.overrides = overrides
        self._minimal_init = bool(minimal_init)
        # 统一初始化，避免后续属性访问报错
        self.data_module = None
        self.setup_config(config_path)

        # 如果命令行指定了LIIF解码器，强制覆盖配置
        if use_liif_decoder:
            if not hasattr(self.config, "model"):
                self.config.model = OmegaConf.create({})
            # 设置LIIF参数
            self.config.model.use_liif_decoder = True
            # 如果没有设置hidden size，给个默认值
            if not hasattr(self.config.model, "liif_mlp_hidden"):
                self.config.model.liif_mlp_hidden = 64
            print(
                f"CommandLine: Force enabling LIIF decoder with hidden size {self.config.model.liif_mlp_hidden}"
            )

        try:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        except Exception:
            pass

        # 如果指定了模型名称，更新配置
        if model_name is not None and hasattr(self, "config"):
            try:
                if not hasattr(self.config, "model"):
                    self.config.model = OmegaConf.create({})
                self.config.model.name = model_name
                try:
                    if hasattr(self, "logger") and self.logger is not None:
                        self.logger.info(f"使用命令行指定的模型: {model_name}")
                    else:
                        print(f"使用命令行指定的模型: {model_name}")
                except Exception:
                    print(f"使用命令行指定的模型: {model_name}")
                try:
                    if hasattr(self.config, "experiment") and hasattr(
                        self.config.experiment, "name"
                    ):
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
            self._curriculum_enabled = bool(
                getattr(getattr(self.config, "training", {}), "curriculum", {}).get(
                    "enabled", False
                )
            )
            self._curriculum_stages = (
                list(
                    getattr(
                        getattr(self.config.training, "curriculum", {}), "stages", []
                    )
                )
                if hasattr(self.config, "training")
                else []
            )
            # 预计算阶段边界（起始/结束epoch），用于快速查询
            self._stage_boundaries = []
            cum_epoch = 0
            if self._curriculum_enabled and self._curriculum_stages:
                for st in self._curriculum_stages:
                    dur = int(st.get("epochs", 0) or 0)
                    start = cum_epoch
                    end = cum_epoch + max(dur, 0)
                    self._stage_boundaries.append(
                        {
                            "start": start,
                            "end": end,
                            "T_out": int(
                                st.get("T_out", getattr(self.config.data, "T_out", 20))
                            ),
                            "description": st.get("description", ""),
                        }
                    )
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
        self.training_phase = "spatial"  # 'spatial', 'temporal', 'joint'
        self.phase_epochs = {"spatial": 0, "temporal": 0, "joint": 0}
        self._current_training_phase = None
        self._phase_initialized = False
        self.h_params = None
        self.observation_op = None
        self._test_viz_done = False
        self._io_debug_cfg = {
            "enabled": bool(
                self._cfg_select(
                    "logging.visualization.io_debug.enabled", default=False
                )
            ),
            "train_every_n_steps": int(
                self._cfg_select(
                    "logging.visualization.io_debug.train_every_n_steps", default=200
                )
                or 200
            ),
            "val_every_n_steps": int(
                self._cfg_select(
                    "logging.visualization.io_debug.val_every_n_steps", default=50
                )
                or 50
            ),
            "max_batches": int(
                self._cfg_select(
                    "logging.visualization.io_debug.max_batches", default=1
                )
                or 1
            ),
            "max_time_steps": int(
                self._cfg_select(
                    "logging.visualization.io_debug.max_time_steps", default=4
                )
                or 4
            ),
        }
        self.setup_logging()
        if self._minimal_init:
            self.device = torch.device("cpu")
            return
        self.setup_device()
        self.setup_memory_management()
        try:
            _bs0 = int(
                self._cfg_select(
                    "data.dataloader.batch_size", "training.batch_size", default=1
                )
            )
        except Exception:
            _bs0 = 1
        self.original_batch_size = _bs0
        self.current_batch_size = _bs0
        self.setup_data()
        self.setup_model()
        if not self._skip_optimizer:
            self.setup_optimizer()
        else:
            self.optimizer = None
            self.scheduler = None
            self.scaler = None
        if not self._skip_monitoring:
            self.setup_monitoring()

    def get_current_T_out(self, epoch: int) -> int:
        """根据课程学习配置返回当前epoch的 T_out，并更新 current_stage。

        若未启用课程学习或配置为空，返回 data.T_out 的默认值。
        """
        try:
            if self._curriculum_enabled and self._stage_boundaries:
                for idx, st in enumerate(self._stage_boundaries):
                    if epoch >= st["start"] and epoch < st["end"]:
                        self.current_stage = idx
                        return int(st["T_out"])
                # 超过最后阶段边界，停留在最后阶段
                self.current_stage = len(self._stage_boundaries) - 1
                return int(self._stage_boundaries[-1]["T_out"])
        except Exception:
            pass
        # 回退到默认 data.T_out
        try:
            return int(getattr(self.config.data, "T_out", 20))
        except Exception:
            return 20

    def cleanup_distributed(self):
        """清理分布式进程组（若已初始化）。

        在所有训练退出路径调用，确保 destroy_process_group() 被正确执行，
        满足开发文档关于分布式清理的要求。
        """
        try:
            if hasattr(torch, "distributed") and dist.is_available():
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
        changes = []

        # 1) DataLoader参数一致性：num_workers=0 时禁用 prefetch_factor/persistent_workers
        try:
            dl = getattr(cfg.data, "dataloader", None)
            if dl is not None:
                nw = int(getattr(dl, "num_workers", 0) or 0)
                if nw <= 0:
                    if getattr(dl, "prefetch_factor", None) is not None:
                        dl.prefetch_factor = None
                        changes.append(
                            "dataloader.prefetch_factor -> None (num_workers=0)"
                        )
                    if getattr(dl, "persistent_workers", False):
                        dl.persistent_workers = False
                        changes.append(
                            "dataloader.persistent_workers -> False (num_workers=0)"
                        )
                # pin_memory_device 在旧版PyTorch不支持时应避免设置
                if hasattr(dl, "pin_memory_device") and dl.pin_memory_device is None:
                    # 避免使用 delattr，由后续逻辑处理 None 值
                    pass
        except Exception:
            pass

        # 2) AMP/精度合理化：优先bf16-mixed（A100以上显卡）、否则16-mixed；确保allow_tf32配置可用
        try:
            prec = str(getattr(cfg.experiment, "precision", "16-mixed"))
            if torch.cuda.is_available():
                cap_major = torch.cuda.get_device_capability()[0]
                # A100/H100等通常cap>=8，优先bf16
                target_prec = "bf16-mixed" if cap_major >= 8 else "16-mixed"
                if prec != target_prec and prec not in ("32", "64"):
                    cfg.experiment.precision = target_prec
                    changes.append(f"experiment.precision {prec} -> {target_prec}")
            else:
                if prec != "32":
                    cfg.experiment.precision = "32"
                    changes.append(f"experiment.precision {prec} -> 32 (No CUDA)")

            # 允许TF32加速（与开发文档一致）
            hw = getattr(cfg, "hardware", None)
            if hw is None:
                from omegaconf import DictConfig

                cfg.hardware = DictConfig({})
                hw = cfg.hardware
            if not hasattr(hw, "allow_tf32"):
                hw.allow_tf32 = True
                changes.append("hardware.allow_tf32 -> True")
        except Exception:
            pass

        # 3) 观测算子参数校验：kernel_size为奇数，sigma非负；插值只能为area/bilinear/nearest
        try:
            obs = getattr(cfg, "observation", None)
            if obs is not None:
                ks = int(getattr(obs, "kernel_size", 5) or 5)
                if ks % 2 == 0:
                    ks = ks + 1
                    obs.kernel_size = ks
                    changes.append(f"observation.kernel_size -> {ks} (must be odd)")
                sigma = float(getattr(obs, "blur_sigma", 0.0) or 0.0)
                if sigma < 0:
                    obs.blur_sigma = 0.0
                    changes.append("observation.blur_sigma -> 0.0 (non-negative)")
                interp = str(getattr(obs, "downsample_interpolation", "area"))
                if interp not in ("area", "bilinear", "nearest"):
                    obs.downsample_interpolation = "area"
                    changes.append(
                        f"observation.downsample_interpolation {interp} -> area"
                    )
        except Exception:
            pass

        # 4) 早停参数校验：至少 patience>=20，min_delta默认1e-4
        try:
            tr = getattr(cfg, "training", None)
            if tr is not None:
                es = getattr(tr, "early_stopping", None)
                if es is None:
                    from omegaconf import DictConfig

                    tr.early_stopping = DictConfig(
                        {
                            "enabled": True,
                            "patience": 50,
                            "min_delta": 1e-4,
                            "monitor": "val_loss",
                        }
                    )
                    changes.append(
                        "training.early_stopping -> enabled=True, patience=50"
                    )
                else:
                    if not hasattr(es, "enabled"):
                        es.enabled = True
                    if (
                        not hasattr(es, "patience")
                        or int(getattr(es, "patience", 0) or 0) < 20
                    ):
                        es.patience = 20
                        changes.append("training.early_stopping.patience -> 20 (min)")
                    if not hasattr(es, "min_delta"):
                        es.min_delta = 1e-4
                    if not hasattr(es, "monitor"):
                        es.monitor = "val_loss"
        except Exception:
            pass

        # 5) 检查点策略校验：最大保留数至少2；周期保存间隔为非负
        try:
            tr = getattr(cfg, "training", None)
            if tr is not None:
                ck = getattr(tr, "checkpoint", None)
                if ck is None:
                    from omegaconf import DictConfig

                    tr.checkpoint = DictConfig(
                        {
                            "save_last": True,
                            "save_best": True,
                            "save_every_n_epochs": 0,
                            "max_keep": 2,
                        }
                    )
                else:
                    if (
                        not hasattr(ck, "max_keep")
                        or int(getattr(ck, "max_keep", 0) or 0) < 2
                    ):
                        ck.max_keep = 2
                        changes.append("training.checkpoint.max_keep -> 2 (min)")
                    if (
                        not hasattr(ck, "save_every_n_epochs")
                        or int(getattr(ck, "save_every_n_epochs", 0) or 0) < 0
                    ):
                        ck.save_every_n_epochs = 0
        except Exception:
            pass

        # 6) Dataloader批次大小合理化：确保val/test bs存在，默认等于train bs
        try:
            dl = getattr(cfg.data, "dataloader", None)
            if dl is not None:
                bs = int(
                    getattr(dl, "batch_size", getattr(cfg.training, "batch_size", 32))
                )
                if not hasattr(dl, "val_batch_size"):
                    dl.val_batch_size = bs
                    changes.append(f"data.dataloader.val_batch_size -> {bs}")
                if not hasattr(dl, "test_batch_size"):
                    dl.test_batch_size = 1
                    changes.append("data.dataloader.test_batch_size -> 1")
        except Exception:
            pass

        if changes and hasattr(self, "logger"):
            for change in changes:
                self.logger.info(f"ℹ️ 配置隐式修改: {change}")

    def setup_config(self, config_path: str = None):
        """设置配置"""
        if config_path and os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
            print(f"✅ 成功加载配置文件: {config_path}")

            # 应用命令行Overrides
            if hasattr(self, "overrides") and self.overrides:
                try:
                    print(f"🔧 应用命令行覆盖配置: {self.overrides}")
                    override_conf = OmegaConf.from_dotlist(self.overrides)
                    self.config = OmegaConf.merge(self.config, override_conf)
                except Exception as e:
                    print(f"⚠️ 应用命令行覆盖配置失败: {e}")

            # 打印关键配置项进行验证
            print(f"📊 配置验证 - T_in: {getattr(self.config.data, 'T_in', '未设置')}")
            print(
                f"📊 配置验证 - T_out: {getattr(self.config.data, 'T_out', '未设置')}"
            )
            print(
                f"📊 配置验证 - use_synthetic_data: {getattr(self.config.data, 'use_synthetic_data', '未设置')}"
            )
        else:
            raise FileNotFoundError(
                f"必须提供有效的配置文件路径 --config，当前: {config_path}"
            )

        # 保守默认：仅在缺失时提供安全参数，优先尊重外部YAML配置
        try:
            # DataLoader 默认（小批量、低并发，避免OOM与初始化问题）
            if (
                not hasattr(self.config.data, "dataloader")
                or self.config.data.dataloader is None
            ):
                bs_default = int(
                    getattr(
                        getattr(self.config, "training", DictConfig({})),
                        "batch_size",
                        8,
                    )
                )
                self.config.data.dataloader = DictConfig(
                    {
                        "batch_size": bs_default,
                        "val_batch_size": bs_default,
                        "test_batch_size": 1,
                        "num_workers": 0,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": None,
                        "drop_last": True,
                        "shuffle": True,
                    }
                )
            else:
                dl = self.config.data.dataloader
                dl.batch_size = int(
                    getattr(
                        dl, "batch_size", getattr(self.config.training, "batch_size", 8)
                    )
                )
                dl.val_batch_size = int(getattr(dl, "val_batch_size", dl.batch_size))
                dl.test_batch_size = int(getattr(dl, "test_batch_size", 1))
                # 并发相关仅在未配置时提供保守默认
                dl.num_workers = int(
                    getattr(
                        dl,
                        "num_workers",
                        getattr(self.config, "hardware", DictConfig({})).get(
                            "num_workers", 0
                        ),
                    )
                )
                dl.pin_memory = bool(
                    getattr(
                        dl,
                        "pin_memory",
                        getattr(
                            getattr(self.config, "hardware", DictConfig({})),
                            "pin_memory",
                            False,
                        ),
                    )
                )
                # 当 num_workers==0 时禁用持久化与预取
                dl.persistent_workers = bool(
                    getattr(
                        dl,
                        "persistent_workers",
                        False if int(getattr(dl, "num_workers", 0)) <= 0 else True,
                    )
                )
                dl.prefetch_factor = (
                    None
                    if int(getattr(dl, "num_workers", 0)) <= 0
                    else int(getattr(dl, "prefetch_factor", 2))
                )
                dl.drop_last = bool(getattr(dl, "drop_last", True))
                dl.shuffle = bool(getattr(dl, "shuffle", True))

            # 训练与调度：仅在缺省时设置保守默认
            if not hasattr(self.config, "training") or self.config.training is None:
                self.config.training = DictConfig(
                    {"epochs": 15, "batch_size": 8, "scheduler": {"T_max": 15}}
                )
            else:
                self.config.training.epochs = int(
                    getattr(self.config.training, "epochs", 15)
                )
                self.config.training.batch_size = int(
                    getattr(
                        self.config.training,
                        "batch_size",
                        getattr(self.config.data.dataloader, "batch_size", 8),
                    )
                )
                if hasattr(self.config.training, "scheduler"):
                    try:
                        self.config.training.scheduler.T_max = int(
                            getattr(
                                self.config.training.scheduler,
                                "T_max",
                                self.config.training.epochs,
                            )
                        )
                    except Exception:
                        pass

            # 硬件并行默认：仅在缺省时设置保守默认
            if not hasattr(self.config, "hardware") or self.config.hardware is None:
                self.config.hardware = DictConfig(
                    {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
                )
            else:
                self.config.hardware.num_workers = int(
                    getattr(self.config.hardware, "num_workers", 0)
                )
                self.config.hardware.pin_memory = bool(
                    getattr(self.config.hardware, "pin_memory", False)
                )
                self.config.hardware.persistent_workers = bool(
                    getattr(self.config.hardware, "persistent_workers", False)
                )

            # 合成数据规模（若真实数据不可用）
            if not hasattr(self.config.data, "max_samples"):
                self.config.data.max_samples = 512
        except Exception:
            pass

        # 设置随机种子与确定性
        try:
            s = int(getattr(self.config.experiment, "seed", 2025))
            torch.manual_seed(s)
            np.random.seed(s)
            try:
                torch.cuda.manual_seed_all(s)
            except Exception:
                pass
            try:
                import torch.backends.cudnn as cudnn

                cudnn.deterministic = True
                cudnn.benchmark = False
            except Exception:
                pass
        except Exception:
            pass

        try:
            if not hasattr(self.config, "experiment"):
                self.config.experiment = DictConfig({})
            base_name = str(getattr(self.config.experiment, "name", "AR-DR2D-Exp"))
            raw_tag = getattr(self, "model_name", None)
            if raw_tag in (None, "", "None"):
                raw_tag = getattr(
                    getattr(self.config, "model", DictConfig({})), "name", "unknown"
                )
            tag = str(raw_tag)
            if "-model_" in base_name:
                base_name = base_name.split("-model_")[0]
            date_str = time.strftime("%Y%m%d")
            seed_val = str(
                getattr(getattr(self.config, "experiment", DictConfig({})), "seed", "")
            )
            name_core = base_name
            if tag and f"-model_{tag}" not in name_core:
                name_core = name_core + f"-model_{tag}"
            if seed_val and f"-s{seed_val}" not in name_core:
                name_core = name_core + f"-s{seed_val}"
            if not name_core.endswith(date_str):
                name_core = name_core + f"-{date_str}"
            self.config.experiment.name = name_core
            if (
                not hasattr(self.config.experiment, "output_dir")
                or not self.config.experiment.output_dir
            ):
                self.config.experiment.output_dir = "runs"
        except Exception:
            pass

    def setup_logging(self):
        """设置日志"""
        if self.output_dir_override:
            self.output_dir = self.output_dir_override
            # 确保目录存在但不覆盖
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger_name_suffix = "_test"  # 区分日志名
        else:
            # 严格使用配置中的 output_dir，不自动拼接实验名
            # 如果配置中没有 output_dir，则回退到 runs/{experiment.name}
            base_out = Path(self.config.experiment.output_dir)
            if base_out.name == "runs":  # 默认值情况，保持原有行为以兼容旧脚本
                self.output_dir = base_out / f"{self.config.experiment.name}"
            else:
                # 用户显式指定了非默认目录 (如 run_sw_4x)，则直接使用该目录，不拼接子目录
                self.output_dir = base_out

            # 目录创建允许并发；由所有rank执行无害
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger_name_suffix = ""

        # 设置日志
        # 依据环境变量判定主进程，避免依赖尚未调用的 setup_device
        log_file_path = None
        try:
            env_local_rank = os.environ.get("LOCAL_RANK")
            env_rank = os.environ.get("RANK")
            rank_val = int(
                env_local_rank
                if env_local_rank is not None
                else (env_rank if env_rank is not None else "0")
            )
        except Exception:
            rank_val = 0
        is_primary = rank_val == 0
        self.is_primary = is_primary
        if is_primary:
            log_file_path = self.output_dir / "training.log"
        else:
            # 非主进程写入独立日志文件，避免文件句柄并发冲突
            log_file_path = self.output_dir / f"training_rank{rank_val}.log"

        self.logger = setup_logger(
            name="RealDataARTrainer", log_file=log_file_path, level=logging.INFO
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
                with open(cfg_snapshot, "w") as f:
                    f.write(merged_yaml)
                self.logger.info(f"📝 已保存配置快照: {cfg_snapshot}")
            except Exception as _cfg_err:
                self.logger.warning(f"⚠️ 配置快照保存失败: {_cfg_err}")

    def setup_device(self):
        """设置设备 - 支持多GPU，并启用TF32/cuDNN与CPU线程优化"""
        # 在DDP初始化前设置关键NCCL稳定性环境变量
        try:
            os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
            os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
            # 避免NCCL在某些环境下的Socket连接问题
            os.environ.setdefault("NCCL_SOCKET_NTHREADS", "4")
            os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")
            # 当网络IB不可用时强制使用PCIe
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            # 提升超时，避免大批次初始化阶段误判超时
            os.environ.setdefault("NCCL_BLOCKING_WAIT_TIMEOUT", "600")
            # 自动检测主机网络接口并设置 NCCL/GLOO 的 IFNAME（优先选择处于 up 状态的以太网接口）
            try:
                import glob

                def _get_up_ifnames():
                    candidates = []
                    for path in glob.glob("/sys/class/net/*"):
                        name = os.path.basename(path)
                        try:
                            with open(os.path.join(path, "operstate")) as f:
                                state = f.read().strip()
                        except Exception:
                            state = ""
                        if state == "up":
                            candidates.append(name)
                    return candidates

                up_ifaces = _get_up_ifnames()
                # 过滤掉虚拟/环回接口，优先 eno*/eth* 其次 ib*
                preferred = [n for n in up_ifaces if n.startswith(("eno", "eth"))]
                if not preferred:
                    preferred = [n for n in up_ifaces if n.startswith("ib")]
                # 兜底：如果没有 up 状态接口，则不覆盖已有设置
                if preferred:
                    ifname = preferred[0]
                    os.environ.setdefault("NCCL_SOCKET_IFNAME", ifname)
                    os.environ.setdefault("GLOO_SOCKET_IFNAME", ifname)
            except Exception:
                pass
            # 设置本地主从地址端口（仅当未由 torch.run 设定时）
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
        except Exception:
            pass
        # 设备选择（统一规范：'gpu'→'cuda'；优先 device.accelerator 其次 experiment.device）
        try:
            raw_device = str(
                self._cfg_select(
                    "device.accelerator", "experiment.device", default="cuda"
                )
            ).lower()
        except Exception:
            raw_device = "cuda"
        # 归一化映射
        if raw_device in ("gpu", "cuda"):
            normalized = "cuda"
        elif raw_device in ("cpu",):
            normalized = "cpu"
        elif raw_device in ("mps",):
            normalized = "mps"
        else:
            # 未知设备类型，保守回退到cpu
            normalized = "cpu"

        print(
            f"DEBUG: normalized={normalized}, cuda_avail={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}"
        )
        import os

        print(f"DEBUG: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

        if normalized == "cuda":
            if torch.cuda.is_available():
                try:
                    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
                    if vis is not None and len(vis.strip()) > 0:
                        self.device = torch.device("cuda:0")
                    else:
                        self.device = torch.device("cuda")
                except Exception:
                    self.device = torch.device("cuda")
            else:
                print(
                    "⚠️ [Critical] Config requested CUDA but torch.cuda.is_available() is False!"
                )
                try:
                    torch.cuda.init()
                except Exception as e:
                    print(f"❌ CUDA Init Error: {e}")
                # Fallback to CPU but warn loudly
                self.device = torch.device("cpu")
        elif (
            normalized == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # 记录设备选择过程，便于诊断
        try:
            self.logger.info(
                f"设备选择: raw='{raw_device}', normalized='{normalized}', final='{self.device}'"
            )
        except Exception:
            pass

        # 应用CPU线程与TF32设置（来自配置hardware.*），确保充分利用硬件
        try:
            hw = getattr(self.config, "hardware", None)
            if hw is not None:
                omp_threads = int(getattr(hw, "omp_threads", 0) or 0)
                mkl_threads = int(getattr(hw, "mkl_threads", 0) or 0)
                torch_threads = int(getattr(hw, "torch_threads", 0) or 0)
                if omp_threads > 0:
                    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
                if mkl_threads > 0:
                    os.environ["MKL_NUM_THREADS"] = str(mkl_threads)
                if torch_threads > 0:
                    try:
                        torch.set_num_threads(torch_threads)
                    except Exception:
                        pass
                # 允许TF32加速（如果配置开启）
                allow_tf32 = bool(getattr(hw, "allow_tf32", True))
                try:
                    if self.device.type == "cuda":
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
                rank_val = int(
                    env_local_rank
                    if env_local_rank is not None
                    else (env_rank if env_rank is not None else "0")
                )
            except Exception:
                rank_val = 0
            self.is_primary = rank_val == 0
            if world_size > 1:
                cfg_backend = None
                try:
                    if hasattr(self.config, "device"):
                        cfg_backend = getattr(self.config.device, "backend", None)
                except Exception:
                    cfg_backend = None
                backend = (
                    cfg_backend
                    if (
                        isinstance(cfg_backend, str) and cfg_backend in {"nccl", "gloo"}
                    )
                    else ("nccl" if self.device.type == "cuda" else "gloo")
                )
                try:
                    dist.init_process_group(backend=backend)
                    self.distributed = True
                    self.local_rank = rank_val
                    if self.device.type == "cuda":
                        torch.cuda.set_device(
                            self.local_rank % max(1, torch.cuda.device_count())
                        )
                        self.device = torch.device(
                            f"cuda:{self.local_rank % max(1, torch.cuda.device_count())}"
                        )
                    self.logger.info(
                        f"DDP已初始化: backend={backend}, rank={self.local_rank}, world_size={dist.get_world_size()}"
                    )
                except Exception as _ddp_err:
                    # NCCL失败时，尝试回退到GLOO后端
                    if backend == "nccl":
                        self.logger.warning(
                            f"NCCL初始化失败，回退到GLOO后端: {_ddp_err}"
                        )
                        try:
                            os.environ.setdefault(
                                "GLOO_SOCKET_IFNAME",
                                os.environ.get("NCCL_SOCKET_IFNAME", "lo"),
                            )
                            dist.init_process_group(backend="gloo")
                            self.distributed = True
                            self.local_rank = rank_val
                            # GLOO也支持CUDA张量，但性能较低；设备保持不变
                            self.logger.info(
                                f"DDP已初始化: backend=gloo, rank={self.local_rank}, world_size={dist.get_world_size()}"
                            )
                        except Exception as _gloo_err:
                            self.logger.error(
                                f"GLOO初始化也失败，回退到非分布式: {_gloo_err}"
                            )
                            self.distributed = False
                    else:
                        self.logger.error(f"DDP初始化失败，回退到非分布式: {_ddp_err}")
        except Exception as e:
            self.logger.warning(f"DDP初始化失败，回退到非分布式: {e}")

        # CPU线程与库线程数设置（根据hardware.*与hardware.cpu.*）
        try:
            import os as _os

            torch_threads = int(
                self._cfg_select("hardware.torch_threads", default=0) or 0
            )
            mkl_threads = int(self._cfg_select("hardware.mkl_threads", default=0) or 0)
            omp_threads = int(self._cfg_select("hardware.omp_threads", default=0) or 0)
            numexpr_threads = int(
                self._cfg_select("hardware.numexpr_threads", default=0) or 0
            )
            interop_threads = int(
                self._cfg_select("hardware.interop_threads", default=0) or 0
            )
            openblas_threads = int(
                self._cfg_select("hardware.blas_threads", default=0) or 0
            )
            if torch_threads > 0:
                torch.set_num_threads(torch_threads)
            if interop_threads > 0 and hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(interop_threads)
                except Exception:
                    pass
            if mkl_threads > 0:
                _os.environ["MKL_NUM_THREADS"] = str(mkl_threads)
            if omp_threads > 0:
                _os.environ["OMP_NUM_THREADS"] = str(omp_threads)
            if numexpr_threads > 0:
                try:
                    max_thr = int(_os.environ.get("NUMEXPR_MAX_THREADS", "64"))
                except Exception:
                    max_thr = 64
                clamped = max(1, min(numexpr_threads, max_thr))
                _os.environ["NUMEXPR_MAX_THREADS"] = str(max_thr)
                _os.environ["NUMEXPR_NUM_THREADS"] = str(clamped)
            if openblas_threads > 0:
                try:
                    ob_max = 64
                except Exception:
                    ob_max = 64
                _os.environ["OPENBLAS_NUM_THREADS"] = str(
                    max(1, min(openblas_threads, ob_max))
                )
            self.logger.info(
                f"CPU线程设置: torch={torch_threads}, interop={interop_threads}, MKL={mkl_threads}, OMP={omp_threads}, numexpr={numexpr_threads}, openblas={openblas_threads}"
            )
        except Exception as e:
            self.logger.warning(f"CPU线程设置失败: {e}")

        # 多GPU配置
        self.use_multi_gpu = False
        if self.device.type == "cuda":
            gpu_count = torch.cuda.device_count()
            # 记录GPU信息仅在首次初始化，不重复输出
            self.logger.debug(f"检测到 {gpu_count} 张GPU")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.debug(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f} GB")
            # 标记是否希望使用多GPU（与分布式解耦）
            self.use_multi_gpu = (
                gpu_count > 1
                and hasattr(self.config, "device")
                and getattr(self.config.device, "devices", 1) > 1
            )
            if getattr(self, "distributed", False) and self.use_multi_gpu:
                self.logger.info(
                    f"启用多GPU训练（DDP），使用 {getattr(self.config.device, 'devices', gpu_count)} 张GPU"
                )
            elif self.use_multi_gpu:
                self.logger.info(
                    f"检测到多GPU且配置期望使用 {getattr(self.config.device, 'devices', gpu_count)} 张GPU；将尝试DataParallel回退"
                )
            else:
                self.logger.info(f"使用单GPU训练: {self.device}")

            # 启用TF32与cuDNN优化
            try:
                allow_tf32 = bool(
                    self._cfg_select("hardware.memory.allow_tf32", default=False)
                )
                cudnn_bench = bool(
                    self._cfg_select("hardware.memory.cudnn_benchmark", default=False)
                )
                if allow_tf32:
                    torch.set_float32_matmul_precision("medium")
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
            "gradient_accumulation_steps": getattr(
                self.config.training, "gradient_accumulation_steps", 1
            ),
            "memory_cleanup_frequency": getattr(
                self.config.training, "memory_cleanup_frequency", 10
            ),
            "auto_batch_size_reduction": getattr(
                self.config.training, "auto_batch_size_reduction", True
            ),
            "memory_threshold": getattr(self.config.training, "memory_threshold", 0.9),
        }

        # 线程与环境变量配置（从YAML获取，支持极限CPU/RAM设置）
        try:
            torch_threads = int(
                self._cfg_select(
                    "hardware.cpu.torch_threads", "hardware.torch_threads", default=0
                )
                or 0
            )
            mkl_threads = int(
                self._cfg_select(
                    "hardware.cpu.mkl_threads", "hardware.mkl_threads", default=0
                )
                or 0
            )
            omp_threads = int(
                self._cfg_select(
                    "hardware.cpu.omp_threads", "hardware.omp_threads", default=0
                )
                or 0
            )
            numexpr_threads = int(
                self._cfg_select("hardware.cpu.numexpr_threads", default=0) or 0
            )
            if torch_threads > 0:
                try:
                    torch.set_num_threads(torch_threads)
                except Exception:
                    pass
            if mkl_threads > 0:
                os.environ["MKL_NUM_THREADS"] = str(mkl_threads)
            if omp_threads > 0:
                os.environ["OMP_NUM_THREADS"] = str(omp_threads)
            if numexpr_threads > 0:
                try:
                    max_thr = int(os.environ.get("NUMEXPR_MAX_THREADS", "64"))
                except Exception:
                    max_thr = 64
                clamped = max(1, min(numexpr_threads, max_thr))
                os.environ["NUMEXPR_MAX_THREADS"] = str(max_thr)
                os.environ["NUMEXPR_NUM_THREADS"] = str(clamped)
        except Exception as e:
            self.logger.warning(f"线程/环境变量设置失败: {e}")

        # 设置CUDA内存与TF32/cuDNN性能开关
        if self.device.type == "cuda":
            # 启用内存池
            torch.cuda.empty_cache()
            # 设置内存分配策略（使用expandable_segments，兼容当前PyTorch版本）
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True,max_split_size_mb:128"
            )
            # NCCL稳定性与错误处理（单机多卡）
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = os.environ.get(
                "TORCH_NCCL_BLOCKING_WAIT", "1"
            )
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = os.environ.get(
                "NCCL_ASYNC_ERROR_HANDLING", "1"
            )
            os.environ["NCCL_BLOCKING_WAIT"] = os.environ.get("NCCL_BLOCKING_WAIT", "1")
            os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")
            os.environ["NCCL_IB_DISABLE"] = os.environ.get(
                "NCCL_IB_DISABLE", "1"
            )  # 单机默认禁用IB，避免误配置
            # 显式启用阻塞等待，提升稳定性
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

            # TF32与cuDNN性能优化
            try:
                allow_tf32 = bool(
                    self._cfg_select(
                        "hardware.allow_tf32",
                        "hardware.memory.allow_tf32",
                        default=True,
                    )
                )
                cudnn_bench = bool(
                    self._cfg_select("hardware.memory.cudnn_benchmark", default=True)
                )
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = allow_tf32
                    torch.backends.cudnn.benchmark = cudnn_bench
                # Matmul精度（中档可启用TF32路径）
                try:
                    torch.set_float32_matmul_precision(
                        "medium" if allow_tf32 else "high"
                    )
                except Exception:
                    pass
                # 统一设置AMP自动转换默认dtype（优先BF16）
                try:
                    amp_cfg = getattr(self.config.training, "amp", None)
                    dtype_name = None
                    if amp_cfg is not None:
                        dtype_name = getattr(
                            amp_cfg, "autocast_dtype", None
                        ) or getattr(amp_cfg, "cast_model_type", None)
                    if dtype_name:
                        dtype_str = str(dtype_name).lower()
                        if "bf16" in dtype_str or "bfloat16" in dtype_str:
                            if hasattr(torch, "set_autocast_gpu_dtype"):
                                torch.set_autocast_gpu_dtype(torch.bfloat16)
                                self.logger.info("AMP autocast dtype 设置为 BF16")
                        elif "fp16" in dtype_str or "float16" in dtype_str:
                            if hasattr(torch, "set_autocast_gpu_dtype"):
                                torch.set_autocast_gpu_dtype(torch.float16)
                                self.logger.info("AMP autocast dtype 设置为 FP16")
                except Exception as _amp_err:
                    self.logger.warning(f"AMP dtype 设置失败: {_amp_err}")
                pass
            except Exception as e:
                self.logger.warning(f"TF32/cuDNN设置失败: {e}")

        # 选择 AMP autocast dtype - 仅依赖 experiment.precision
        autocast_dtype = None
        try:
            # experiment.precision 已在 validate_config 中被规范化
            prec = str(getattr(self.config.experiment, "precision", "16-mixed")).lower()
            if "bf16" in prec:
                autocast_dtype = torch.bfloat16
            elif "16" in prec:
                autocast_dtype = torch.float16
        except Exception:
            autocast_dtype = None
        self.autocast_dtype = autocast_dtype

        self.logger.info(
            f"内存管理配置: {self.memory_config}, AMP dtype: {('default' if autocast_dtype is None else ('bfloat16' if autocast_dtype is torch.bfloat16 else 'float16'))}"
        )

    def check_memory_usage(self) -> float:
        """检查GPU内存使用率"""
        try:
            if torch.cuda.is_available() and self.device.type == "cuda":
                # Fix B: Use current device instead of hardcoded 0
                device_idx = torch.cuda.current_device()
                if self.device.index is not None:
                    device_idx = self.device.index

                allocated = torch.cuda.memory_allocated(device_idx) / 1024**3  # GB
                total = (
                    torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
                )  # GB
                return allocated / total
        except Exception as e:
            self.logger.warning(f"显存监控读取失败: {e}")
        return 0.0

    def cleanup_memory(self):
        """清理GPU内存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def cleanup_cuda(self, full: bool = True):
        """统一的CUDA清理函数"""
        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)

            # 显式删除可能持有大Tensor的属性
            # 注意：不要删除 self.model 或 self.data_module
            for attr in [
                "loss",
                "losses",
                "pred_seq",
                "target_seq",
                "input_seq",
                "batch",
            ]:
                if hasattr(self, attr):
                    try:
                        delattr(self, attr)
                    except Exception:
                        pass

            # 清理局部变量（如果在frame中）- 这里主要靠GC
            import gc

            gc.collect()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                # 仅在调试模式下同步，避免性能损耗
                if self._cfg_select("training.debug_cuda", default=False):
                    torch.cuda.synchronize()

                # 重置峰值统计，以便观察后续是否下降
                torch.cuda.reset_peak_memory_stats()

        except Exception as e:
            self.logger.warning(f"cleanup_cuda 执行期间发生次级错误: {e}")

    def setup_data(self):
        """设置数据"""
        self.logger.info("设置数据模块...")
        try:
            self._setup_data_module()
            # 移除合成数据回退，强制使用真实数据
            # self._setup_synthetic_data()
            self._setup_dataloaders()
            self._setup_observation_operator()
            self._setup_norm_stats()
        except Exception as e:
            self.logger.exception(f"❌ 数据模块设置失败/数据设置失败: {e}")
            raise

    def _setup_data_module(self):
        """内部函数：设置数据模块"""
        # 获取批次大小配置
        self.batch_size = int(
            self._cfg_select(
                "data.dataloader.batch_size", "training.batch_size", default=128
            )
        )
        self.val_batch_size = int(
            self._cfg_select("data.dataloader.val_batch_size", default=self.batch_size)
        )
        self.test_batch_size = int(
            self._cfg_select(
                "data.dataloader.test_batch_size", "testing.batch_size", default=1
            )
        )

        # Task A: 硬性一致性校验
        T_in = int(self._cfg_select("data.T_in", default=1))
        T_out = int(self._cfg_select("data.T_out", default=1))
        time_step_start = int(self._cfg_select("data.time_step_start", default=0))
        time_step_end = int(self._cfg_select("data.time_step_end", default=100))
        time_step_stride = int(self._cfg_select("data.time_step_stride", default=1))

        # 计算可用时间点数
        N_timesteps = (time_step_end - time_step_start) // time_step_stride + 1

        if N_timesteps < T_in + T_out:
            raise ValueError(
                f"❌ 时间步配置不自洽！可用时间点数 N_timesteps ({N_timesteps}) 小于 T_in ({T_in}) + T_out ({T_out})。\n"
                f"当前配置: start={time_step_start}, end={time_step_end}, stride={time_step_stride} -> N={(time_step_end-time_step_start)}//{time_step_stride}+1 = {N_timesteps}。\n"
                f"建议修复: 减小 stride (例如设为1) 或 增大 end (至少需要 {time_step_start + (T_in + T_out - 1) * time_step_stride})。"
            )
        self.logger.info(
            f"✅ 时间步一致性校验通过: N_timesteps={N_timesteps} >= T_in({T_in}) + T_out({T_out})"
        )

        # 记录使用的批次大小
        self.logger.info(f"使用训练批次大小: {self.batch_size}")
        self.logger.info(f"使用验证批次大小: {self.val_batch_size}")
        self.logger.info(f"使用测试批次大小: {self.test_batch_size}")

        # 强制使用真实数据，无任何 fallback
        self.using_synthetic = False
        self.using_dm = True

        dataset_name = self._cfg_select(
            "data.dataset_name", default="RealDiffusionReaction"
        )
        self.logger.info(f"使用数据集: {dataset_name}")

        if dataset_name == "darcy_flow":
            self.logger.info("初始化 DarcyFlowDataModule...")
            self.data_module = DarcyFlowDataModule(self.config)
        else:
            self.logger.info("初始化 RealDiffusionReactionDataModule (默认)...")
            self.data_module = RealDiffusionReactionDataModule(self.config)

        self.data_module.setup()

    def _setup_synthetic_data(self):
        """内部函数：设置合成数据"""
        if self.using_synthetic:
            self.logger.info("🧪 使用合成数据模式")

            # 合成数据集定义
            class SyntheticARSequenceDataset(torch.utils.data.Dataset):
                def __init__(
                    self, n=4096, T_in=1, T_out=1, C=2, H=128, W=128, seed=2025
                ):
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
                        "input_sequence": input_seq,
                        "target_sequence": target_seq,
                        "sample_idx": idx,
                        "start_time": 0,
                    }

            T_in = int(self._cfg_select("data.T_in", default=1))
            T_out = int(self._cfg_select("data.T_out", default=1))
            C = int(self._cfg_select("model.out_channels", default=2))
            H = int(self._cfg_select("model.img_size", default=128))
            W = H
            synth_n = int(
                self._cfg_select(
                    "data.synthetic_data_config.num_samples",
                    "data.max_samples",
                    default=1000,
                )
                or 1000
            )
            seed = int(self._cfg_select("experiment.seed", default=2025))

            synth_ds = SyntheticARSequenceDataset(
                n=synth_n, T_in=T_in, T_out=T_out, C=C, H=H, W=W, seed=seed
            )

            n_train = int(synth_n * 0.7)
            n_val = int(synth_n * 0.15)
            self.train_dataset = torch.utils.data.Subset(synth_ds, range(0, n_train))
            self.val_dataset = torch.utils.data.Subset(
                synth_ds, range(n_train, n_train + n_val)
            )
            self.test_dataset = torch.utils.data.Subset(
                synth_ds, range(n_train + n_val, synth_n)
            )

            self.logger.info(
                f"✅ 合成数据集创建完成: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}"
            )

        self.test_loader = None
        self.using_dm = False

        # Fix 3: One-time flag for DDP logging
        self._ddp_loadercheck_logged = set()

        # 尝试获取DataModule
        try:
            self._setup_data_module()
            # 移除合成数据回退，强制使用真实数据
            # self._setup_synthetic_data()
            self._setup_dataloaders()
            self._setup_observation_operator()
            self._setup_norm_stats()
        except Exception as e:
            self.logger.exception(f"❌ 数据模块设置失败/数据设置失败: {e}")
            raise

    def _setup_dataloaders(self):
        # 统一保护：num_workers==0 时禁用 prefetch_factor 并关闭 persistent_workers
        try:
            num_workers = int(
                self._cfg_select(
                    "data.dataloader.num_workers", "hardware.num_workers", default=32
                )
                or 32
            )
            if (
                num_workers == 0
                and hasattr(self.config, "data")
                and hasattr(self.config.data, "dataloader")
            ):
                self.config.data.dataloader.prefetch_factor = None
                self.config.data.dataloader.persistent_workers = False
            elif (
                num_workers > 0
                and hasattr(self.config, "data")
                and hasattr(self.config.data, "dataloader")
            ):
                self.config.data.dataloader.persistent_workers = True
                prefetch_cfg = getattr(
                    self.config.data.dataloader, "prefetch_factor", None
                )
                if prefetch_cfg in (None, 0):
                    self.config.data.dataloader.prefetch_factor = 16
        except Exception as e:
            self.logger.warning(f"设置 prefetch_factor 保护失败: {e}")

        if self.using_synthetic:
            try:
                from torch.utils.data import DataLoader as _DL

                _collate = (
                    fast_collate_fn
                    if ("fast_collate_fn" in globals() and fast_collate_fn is not None)
                    else (
                        safe_collate_fn
                        if (
                            "safe_collate_fn" in globals()
                            and safe_collate_fn is not None
                        )
                        else None
                    )
                )
                # Fix D: Use sanitize helper for construction
                base_kwargs = dict(
                    num_workers=0, pin_memory=False, persistent_workers=False
                )
                dl_kwargs = self._get_compatible_loader_kwargs(base_kwargs)
                self.train_loader = _DL(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=_collate,
                    **dl_kwargs,
                )
                self.val_loader = _DL(
                    self.val_dataset,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    collate_fn=_collate,
                    **dl_kwargs,
                )
                self.test_loader = _DL(
                    self.test_dataset,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    collate_fn=_collate,
                    **dl_kwargs,
                )
            except Exception:
                self.logger.exception(
                    "❌ Failed to create DataLoaders from synthetic datasets"
                )
                raise
        elif self.using_dm:
            self.train_loader = self.data_module.train_dataloader()
            self.val_loader = self.data_module.val_dataloader()
            self.test_loader = self.data_module.test_dataloader()

            # Fix D: Rebuild DataModule loaders if they have dangerous pin_memory_device=None
            for name in ["train_loader", "val_loader", "test_loader"]:
                loader = getattr(self, name, None)
                if (
                    loader
                    and hasattr(loader, "pin_memory_device")
                    and loader.pin_memory_device is None
                ):
                    try:
                        # Rebuild using robust helper
                        new_loader = self._rebuild_loader_from_existing(
                            loader,
                            shuffle=(name == "train_loader"),
                            sampler=loader.sampler,  # Will be overridden if DDP
                        )
                        setattr(self, name, new_loader)
                        self.logger.info(
                            f"Rebuilt {name} to fix pin_memory_device=None"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to rebuild {name}, keeping original: {e}"
                        )

        # DDP处理与兜底
        self._ensure_dataloader()

        # 修复pin_memory_device

        # 记录批次数
        tl = len(self.train_loader) if self.train_loader else 0
        vl = len(self.val_loader) if self.val_loader else 0
        tsl = len(self.test_loader) if self.test_loader else 0
        self.logger.info(f"训练集批次数: {tl}, 验证集批次数: {vl}, 测试集批次数: {tsl}")

    def _setup_observation_operator(self):
        """内部函数：设置观测算子"""
        obs_cfg = getattr(self.config, "observation", None)
        if obs_cfg is None:
            try:
                obs_cfg = getattr(self.config, "data", None)
                obs_cfg = (
                    getattr(obs_cfg, "observation", None)
                    if obs_cfg is not None
                    else None
                )
            except Exception:
                obs_cfg = None

        # 将 DictConfig 转为 dict 以避免 .get() 风险
        if obs_cfg is not None:
            try:
                from omegaconf import DictConfig, OmegaConf

                if isinstance(obs_cfg, DictConfig):
                    obs_cfg = OmegaConf.to_container(obs_cfg, resolve=True)
            except Exception:
                pass

        self.h_params = None
        self.observation_op = None
        if obs_cfg is not None and isinstance(obs_cfg, dict):
            mode_raw = obs_cfg.get("mode", "sr")
            mode = str(
                mode_raw[0] if isinstance(mode_raw, (list, tuple)) else mode_raw
            ).lower()
            boundary = obs_cfg.get("boundary", obs_cfg.get("boundary_mode", "mirror"))

            if mode == "sr":
                sr_sub = (
                    obs_cfg.get("sr", {})
                    if isinstance(obs_cfg.get("sr", {}), dict)
                    else {}
                )
                # 优先查找 'scale'，其次 'scale_factor'，最后是 sr 子字典中的配置
                scale = obs_cfg.get(
                    "scale", obs_cfg.get("scale_factor", sr_sub.get("scale_factor", 2))
                )
                # 确保 scale 是 int
                try:
                    scale = int(scale)
                except Exception:
                    scale = 2

                sigma = obs_cfg.get("blur_sigma", sr_sub.get("blur_sigma", 1.0))
                kernel_size = obs_cfg.get(
                    "kernel_size", sr_sub.get("blur_kernel_size", 5)
                )
                boundary = (
                    boundary
                    if boundary is not None
                    else sr_sub.get("boundary_mode", "mirror")
                )
                downsample = obs_cfg.get(
                    "downsample_interpolation", sr_sub.get("downsample_mode", "area")
                )
                self.h_params = {
                    "task": "SR",
                    "scale": scale,
                    "sigma": sigma,
                    "kernel_size": kernel_size,
                    "boundary": boundary,
                    "downsample_interpolation": downsample,
                }
                self.observation_op = lambda x: apply_degradation_operator(
                    x,
                    {
                        "task": "SR",
                        "scale": scale,
                        "sigma": sigma,
                        "kernel_size": kernel_size,
                        "boundary": boundary,
                    },
                )
                self.logger.info(
                    f"✅ 观测算子初始化 (SR): scale={scale}, sigma={sigma}"
                )
            elif mode == "crop":
                crop_sub = (
                    obs_cfg.get("crop", {})
                    if isinstance(obs_cfg.get("crop", {}), dict)
                    else {}
                )
                # 修复: 同时支持 'crop_size' 和 'size' 键名
                crop_size = obs_cfg.get(
                    "crop_size", crop_sub.get("crop_size", crop_sub.get("size", None))
                )

                # degradation.py 期望 crop_size 为列表/元组 [h, w]
                deg_crop_size = crop_size
                if crop_size is not None and not isinstance(crop_size, (list, tuple)):
                    deg_crop_size = [crop_size, crop_size]

                crop_box = obs_cfg.get("crop_box", crop_sub.get("crop_box", None))
                boundary = (
                    boundary
                    if boundary is not None
                    else crop_sub.get("boundary_mode", "mirror")
                )
                self.h_params = {
                    "task": "Crop",
                    "crop_size": deg_crop_size,
                    "crop_box": crop_box,
                    "boundary": boundary,
                }
                self.observation_op = lambda x: apply_degradation_operator(
                    x,
                    {
                        "task": "Crop",
                        "crop_size": deg_crop_size,
                        "crop_box": crop_box,
                        "boundary": boundary,
                    },
                )
            elif mode == "identity":
                self.h_params = {"task": "Identity", "boundary": boundary}
                self.observation_op = nn.Identity()
            else:
                self.logger.warning(f"未知的观测模式: {mode}，跳过观测算子初始化")

            if self.h_params:
                self.logger.info(f"✅ 观测算子配置: {self.h_params}")

        # 设置训练专用退化算子 (支持 Mismatch Experiment)
        self.training_degradation_op = self.observation_op  # 默认与观测算子一致

        # 尝试从 training.degradation 读取独立配置
        train_deg_cfg = getattr(self.config, "training", {}).get("degradation", None)
        if train_deg_cfg is not None:
            try:
                if isinstance(train_deg_cfg, DictConfig):
                    train_deg_cfg = OmegaConf.to_container(train_deg_cfg, resolve=True)

                if isinstance(train_deg_cfg, dict):
                    self.logger.info(f"🔧 发现独立训练退化配置: {train_deg_cfg}")
                    # 复用 apply_degradation_operator 逻辑
                    mode_raw = train_deg_cfg.get("mode", "sr")
                    mode = str(
                        mode_raw[0] if isinstance(mode_raw, (list, tuple)) else mode_raw
                    ).lower()
                    boundary = train_deg_cfg.get(
                        "boundary", train_deg_cfg.get("boundary_mode", "mirror")
                    )

                    if mode == "sr":
                        sr_sub = (
                            train_deg_cfg.get("sr", {})
                            if isinstance(train_deg_cfg.get("sr", {}), dict)
                            else {}
                        )
                        # 优先查找 'scale'，其次 'scale_factor'，最后是 sr 子字典中的配置
                        scale = train_deg_cfg.get(
                            "scale",
                            train_deg_cfg.get(
                                "scale_factor", sr_sub.get("scale_factor", 2)
                            ),
                        )
                        sigma = train_deg_cfg.get(
                            "blur_sigma", sr_sub.get("blur_sigma", 1.0)
                        )
                        kernel_size = train_deg_cfg.get(
                            "kernel_size", sr_sub.get("blur_kernel_size", 5)
                        )
                        boundary = (
                            boundary
                            if boundary is not None
                            else sr_sub.get("boundary_mode", "mirror")
                        )

                        # 确保 scale 是 int
                        try:
                            scale = int(scale)
                        except Exception:
                            scale = 2

                        self.logger.info(
                            f"🔧 SR退化配置解析: raw_scale={train_deg_cfg.get('scale')}, raw_scale_factor={train_deg_cfg.get('scale_factor')}, resolved_scale={scale}"
                        )

                        # training_degradation_op 构造
                        self.training_degradation_op = (
                            lambda x: apply_degradation_operator(
                                x,
                                {
                                    "task": "SR",
                                    "scale": scale,
                                    "sigma": sigma,
                                    "kernel_size": kernel_size,
                                    "boundary": boundary,
                                },
                            )
                        )
                        self.logger.info(
                            f"✅ 训练退化算子已重写 (SR): sigma={sigma}, scale={scale}"
                        )
                    elif mode == "crop":
                        # 类似的 Crop 逻辑...
                        crop_sub = (
                            train_deg_cfg.get("crop", {})
                            if isinstance(train_deg_cfg.get("crop", {}), dict)
                            else {}
                        )
                        # 修复: 同时支持 'crop_size' 和 'size' 键名
                        crop_size = train_deg_cfg.get(
                            "crop_size",
                            crop_sub.get("crop_size", crop_sub.get("size", None)),
                        )

                        # degradation.py 期望 crop_size 为列表/元组 [h, w]
                        deg_crop_size = crop_size
                        if crop_size is not None and not isinstance(
                            crop_size, (list, tuple)
                        ):
                            deg_crop_size = [crop_size, crop_size]

                        crop_box = train_deg_cfg.get(
                            "crop_box", crop_sub.get("crop_box", None)
                        )
                        boundary = (
                            boundary
                            if boundary is not None
                            else crop_sub.get("boundary_mode", "mirror")
                        )
                        self.training_degradation_op = (
                            lambda x: apply_degradation_operator(
                                x,
                                {
                                    "task": "Crop",
                                    "crop_size": deg_crop_size,
                                    "crop_box": crop_box,
                                    "boundary": boundary,
                                },
                            )
                        )
                        self.logger.info(
                            f"✅ 训练退化算子已重写 (Crop): size={crop_size}"
                        )
                    elif mode == "identity":
                        self.training_degradation_op = nn.Identity()
                        self.logger.info("✅ 训练退化算子已重写 (Identity)")
            except Exception as e:
                self.logger.warning(
                    f"⚠️ 初始化训练退化算子失败，回退到默认观测算子: {e}"
                )
                self.training_degradation_op = self.observation_op

    def _setup_norm_stats(self):
        """内部函数：设置归一化统计量（优先级：缓存文件 > DataModule > 现场计算 > 默认）"""
        self.norm_stats = None
        source = "unknown"

        # 1. 尝试从缓存文件加载
        try:
            from pathlib import Path

            import numpy as np

            candidates = []
            splits_dir = getattr(getattr(self.config, "data", None), "splits_dir", None)
            if splits_dir:
                candidates.append(Path(str(splits_dir)) / "norm_stat.npz")
                candidates.append(Path(str(splits_dir)) / "norm_stats.npz")
            candidates.append(Path(self.output_dir) / "norm_stats.npz")

            for p in candidates:
                if p.exists():
                    d = np.load(str(p))
                    mean = torch.tensor(d.get("mean")).float()
                    std = torch.tensor(d.get("std")).float()
                    self.norm_stats = {"mean": mean, "std": std}
                    source = f"cache_file({p})"
                    break
        except Exception:
            pass

        # 2. 尝试从DataModule获取
        if self.norm_stats is None and self.using_dm:
            try:
                dm = getattr(self, "data_module", None)
                train_ds = getattr(dm, "train_dataset", None)
                if (
                    train_ds is not None
                    and hasattr(train_ds, "mean")
                    and hasattr(train_ds, "std")
                ):
                    mean = train_ds.mean
                    std = train_ds.std
                    if isinstance(mean, torch.Tensor):
                        mean = mean.detach().cpu()
                    if isinstance(std, torch.Tensor):
                        std = std.detach().cpu()
                    self.norm_stats = {"mean": mean, "std": std}
                    source = "data_module"
            except Exception:
                pass

        # 3. 现场计算（仅在真实数据模式下）
        if (
            self.norm_stats is None
            and not self.using_synthetic
            and self.train_loader is not None
        ):
            if self._compute_norm_stats_from_data():
                source = "computed_from_data"

        # 4. 默认值（合成数据或兜底）
        if self.norm_stats is None:
            C = int(self._cfg_select("model.out_channels", default=2))
            self.norm_stats = {"mean": torch.zeros(C), "std": torch.ones(C)}
            source = "default_zeros_ones"

        # 统一填充兼容键名
        if self.norm_stats is not None:
            mean = self.norm_stats["mean"]
            std = self.norm_stats["std"]
            if mean.numel() >= 1:
                self.norm_stats["u_mean"] = mean[0]
                self.norm_stats["u_std"] = std[0]
            if mean.numel() >= 2:
                self.norm_stats["v_mean"] = mean[1]
                self.norm_stats["v_std"] = std[1]
            self.norm_stats["data_mean"] = self.norm_stats.get(
                "u_mean", torch.tensor(0.0)
            )
            self.norm_stats["data_std"] = self.norm_stats.get(
                "u_std", torch.tensor(1.0)
            )

        self.logger.info(f"✅ 归一化统计来源: {source}")
        if self.norm_stats:
            self.logger.info(
                f"   u_mean={self.norm_stats.get('u_mean', 0):.3f}, u_std={self.norm_stats.get('u_std', 1):.3f}"
            )

    def _compute_norm_stats_from_data(self) -> bool:
        """从数据集中计算归一化统计量"""
        try:
            import torch

            ds = getattr(self, "train_dataset", None)
            dl = getattr(self, "train_loader", None)
            if dl is None and ds is not None:
                from torch.utils.data import DataLoader

                dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            if dl is None:
                return False
            sums = None
            sumsq = None
            count = 0
            max_batches = 8
            b = 0
            for batch in dl:
                if b >= max_batches:
                    break
                b += 1
                x = None
                if isinstance(batch, dict):
                    # 优先取输入或目标序列张量
                    for k in ["input_seq", "target_seq", "x", "y"]:
                        v = batch.get(k, None)
                        if torch.is_tensor(v) and v.dim() >= 4:
                            x = v
                            break
                    if x is None:
                        for v in batch.values():
                            if torch.is_tensor(v) and v.dim() >= 4:
                                x = v
                                break
                elif isinstance(batch, (list, tuple)):
                    for v in batch:
                        if torch.is_tensor(v) and v.dim() >= 4:
                            x = v
                            break
                elif torch.is_tensor(batch):
                    x = batch
                if x is None:
                    continue
                if x.dim() == 5:
                    B, T, C, H, W = x.shape
                    x = x.view(B * T, C, H, W)
                elif x.dim() == 4:
                    B, C, H, W = x.shape
                else:
                    continue
                x = x.float()
                c_sum = x.sum(dim=(0, 2, 3))
                c_sumsq = (x * x).sum(dim=(0, 2, 3))
                pixels = x.shape[0] * x.shape[2] * x.shape[3]
                sums = c_sum if sums is None else (sums + c_sum)
                sumsq = c_sumsq if sumsq is None else (sumsq + c_sumsq)
                count += pixels
            if count == 0 or sums is None:
                return False
            mean = sums / count
            var = sumsq / count - mean * mean
            var = torch.clamp(var, min=1e-8)
            std = torch.sqrt(var)
            self.norm_stats = {"mean": mean.clone(), "std": std.clone()}
            if mean.numel() >= 1:
                self.norm_stats["u_mean"] = mean[0]
                self.norm_stats["u_std"] = std[0]
            if mean.numel() >= 2:
                self.norm_stats["v_mean"] = mean[1]
                self.norm_stats["v_std"] = std[1]
            # 缓存到运行目录
            try:
                from pathlib import Path

                import numpy as np

                out = Path(self.output_dir) / "norm_stats.npz"
                np.savez(str(out), mean=mean.cpu().numpy(), std=std.cpu().numpy())
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _get_compatible_loader_kwargs(self, base_kwargs: dict) -> dict:
        """Fix D: Helper to sanitize DataLoader kwargs for compatibility"""
        kwargs = base_kwargs.copy()

        # If pin_memory is False, device is irrelevant
        if not kwargs.get("pin_memory", False):
            kwargs.pop("pin_memory_device", None)
            return kwargs

        # Check if current PyTorch supports pin_memory_device
        import inspect

        from torch.utils.data import DataLoader

        sig = inspect.signature(DataLoader)
        if "pin_memory_device" not in sig.parameters:
            kwargs.pop("pin_memory_device", None)
            return kwargs

        # Supported: Sanitize the value
        pmd = kwargs.get("pin_memory_device")
        if not pmd:  # None or empty string
            if torch.cuda.is_available():
                dev_idx = torch.cuda.current_device()
                if hasattr(self, "device") and self.device.index is not None:
                    dev_idx = self.device.index
                kwargs["pin_memory_device"] = f"cuda:{dev_idx}"
            else:
                kwargs["pin_memory_device"] = "cpu"
        return kwargs

    def _rebuild_loader_from_existing(
        self, loader, *, shuffle: bool, sampler=None, batch_size=None
    ):
        """Fix 3: Robust loader reconstruction preserving attributes"""
        from torch.utils.data import DataLoader as _DL

        dataset = loader.dataset
        collate_fn = loader.collate_fn
        bs = batch_size if batch_size is not None else loader.batch_size

        # Inherit worker settings but sanitize pin_memory
        base_kwargs = {
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory,
            "persistent_workers": getattr(loader, "persistent_workers", False),
        }
        # Fix 4: num_workers compatibility
        if loader.num_workers > 0 and hasattr(loader, "prefetch_factor"):
            base_kwargs["prefetch_factor"] = loader.prefetch_factor
        elif loader.num_workers == 0:
            base_kwargs["persistent_workers"] = False

        dl_kwargs = self._get_compatible_loader_kwargs(base_kwargs)

        # Handle DDP sampler injection if needed
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            from torch.utils.data.distributed import DistributedSampler

            # Ensure we create a fresh DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        return _DL(
            dataset,
            batch_size=bs,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            **dl_kwargs,
        )

    def _log_ddp_setup(self, name, loader):
        """Fix 1: Log DDP loader setup details on rank 0"""
        import torch.distributed as dist

        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() == 0
            and name not in self._ddp_loadercheck_logged
        ):
            sampler = loader.sampler
            s_type = type(sampler).__name__
            replicas = getattr(sampler, "num_replicas", "N/A")
            rank = getattr(sampler, "rank", "N/A")
            shuffle = getattr(sampler, "shuffle", "N/A")
            bs = loader.batch_size
            self.logger.info(
                f"DDP LoaderCheck {name}: sampler={s_type} replicas={replicas} rank={rank} shuffle={shuffle} bs={bs}"
            )
            self._ddp_loadercheck_logged.add(name)

    def _ensure_dataloader(self):
        """确保DataLoader已正确初始化（原setup_optimizer中的逻辑提取）"""
        try:
            batch_size = int(
                self._cfg_select(
                    "data.dataloader.batch_size",
                    "training.batch_size",
                    default=(
                        getattr(self, "current_batch_size", None)
                        or getattr(self, "original_batch_size", 1)
                        or 1
                    ),
                )
            )
        except Exception:
            self.logger.warning("⚠️ 无法获取batch_size配置，默认为1", exc_info=True)
            batch_size = 1
        # 首先确保DataLoader属性存在，如果不存在则初始化为None
        if not hasattr(self, "train_loader"):
            self.train_loader = None
        if not hasattr(self, "val_loader"):
            self.val_loader = None
        if not hasattr(self, "test_loader"):
            self.test_loader = None

        if any(
            dl is None for dl in (self.train_loader, self.val_loader, self.test_loader)
        ):
            self.logger.warning("⚠️ DataLoader仍为None，使用最小配置强制重建")
            try:
                # 尝试从已有属性中获取dataset
                train_ds_fb = getattr(self, "train_dataset", None)
                val_ds_fb = getattr(self, "val_dataset", None)
                test_ds_fb = getattr(self, "test_dataset", None)

                self.logger.info(
                    f"📊 数据集状态: train={train_ds_fb is not None}, val={val_ds_fb is not None}, test={test_ds_fb is not None}"
                )

                # 如果self中没有，尝试从data_module获取
                if (
                    train_ds_fb is None
                    and hasattr(self, "data_module")
                    and hasattr(self.data_module, "train_dataset")
                ):
                    train_ds_fb = getattr(self.data_module, "train_dataset", None)
                if (
                    val_ds_fb is None
                    and hasattr(self, "data_module")
                    and hasattr(self.data_module, "val_dataset")
                ):
                    val_ds_fb = getattr(self.data_module, "val_dataset", None)
                if (
                    test_ds_fb is None
                    and hasattr(self, "data_module")
                    and hasattr(self.data_module, "test_dataset")
                ):
                    test_ds_fb = getattr(self.data_module, "test_dataset", None)
                # 如果仍为空且默认DataLoader存在，尝试取其dataset
                dm_train = getattr(self, "train_loader", None)
                dm_val = getattr(self, "val_loader", None)
                dm_test = getattr(self, "test_loader", None)

                if train_ds_fb is None or val_ds_fb is None or test_ds_fb is None:
                    try:
                        if dm_train is not None and train_ds_fb is None:
                            train_ds_fb = getattr(dm_train, "dataset", None)
                        if dm_val is not None and val_ds_fb is None:
                            val_ds_fb = getattr(dm_val, "dataset", None)
                        if dm_test is not None and test_ds_fb is None:
                            test_ds_fb = getattr(dm_test, "dataset", None)
                    except Exception:
                        self.logger.warning(
                            "⚠️ 从现有DataLoader提取dataset失败", exc_info=True
                        )
                from torch.utils.data import DataLoader as _DL2

                minimal_kwargs = dict(
                    num_workers=0, pin_memory=False, persistent_workers=False
                )

                # Fix A: Robust DDP detection
                import torch.distributed as dist

                use_ddp = dist.is_available() and dist.is_initialized()

                # Fix D: Use sanitized kwargs
                dl_kwargs = self._get_compatible_loader_kwargs(minimal_kwargs)

                dl_collate_fb = (
                    fast_collate_fn
                    if ("fast_collate_fn" in globals() and fast_collate_fn is not None)
                    else (
                        safe_collate_fn
                        if (
                            "safe_collate_fn" in globals()
                            and safe_collate_fn is not None
                        )
                        else None
                    )
                )
                if self.train_loader is None and train_ds_fb is not None:
                    sampler = None
                    if use_ddp:
                        from torch.utils.data.distributed import DistributedSampler

                        sampler = DistributedSampler(train_ds_fb, shuffle=True)
                    self.train_loader = _DL2(
                        train_ds_fb,
                        batch_size=batch_size,
                        shuffle=(sampler is None),
                        sampler=sampler,
                        collate_fn=dl_collate_fb,
                        **dl_kwargs,
                    )
                    if use_ddp:
                        self._log_ddp_setup("train", self.train_loader)

                if self.val_loader is None and val_ds_fb is not None:
                    sampler_v = None
                    if use_ddp:
                        from torch.utils.data.distributed import DistributedSampler

                        sampler_v = DistributedSampler(val_ds_fb, shuffle=False)
                    self.val_loader = _DL2(
                        val_ds_fb,
                        batch_size=int(
                            self._cfg_select(
                                "data.dataloader.val_batch_size", default=batch_size
                            )
                        ),
                        shuffle=False,
                        sampler=sampler_v,
                        collate_fn=dl_collate_fb,
                        **dl_kwargs,
                    )
                    if use_ddp:
                        self._log_ddp_setup("val", self.val_loader)

                if self.test_loader is None and test_ds_fb is not None:
                    sampler_t = None
                    if use_ddp:
                        from torch.utils.data.distributed import DistributedSampler

                        sampler_t = DistributedSampler(test_ds_fb, shuffle=False)
                    self.test_loader = _DL2(
                        test_ds_fb,
                        batch_size=int(
                            self._cfg_select(
                                "data.dataloader.test_batch_size",
                                "testing.batch_size",
                                default=1,
                            )
                        ),
                        shuffle=False,
                        sampler=sampler_t,
                        collate_fn=dl_collate_fb,
                        **dl_kwargs,
                    )
                    if use_ddp:
                        self._log_ddp_setup("test", self.test_loader)

                if any(
                    dl is None
                    for dl in (self.train_loader, self.val_loader, self.test_loader)
                ):
                    raise RuntimeError("最终兜底重建失败：仍有DataLoader为None")
            except Exception as e:
                self.logger.exception(f"❌ 兜底重建DataLoader失败: {e}")
                raise

        # 存储原始批次大小用于动态调整
        self.original_batch_size = batch_size
        self.current_batch_size = batch_size

        # Fix E: Runtime monkey patching removed.

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
            T_in = int(self._cfg_select("data.T_in", default=1))
            T_out = int(self._cfg_select("data.T_out", default=1))
            C = int(self._cfg_select("model.out_channels", default=1))
            H = int(self._cfg_select("model.img_size", default=64))
            W = H
            sample_batch = {
                "input_sequence": torch.randn(B, T_in, C, H, W),
                "target_sequence": torch.randn(B, T_out, C, H, W),
            }
        self.logger.info(f"✅ 输入序列形状: {sample_batch['input_sequence'].shape}")
        self.logger.info(f"✅ 目标序列形状: {sample_batch['target_sequence'].shape}")

        try:
            in_shape = sample_batch["input_sequence"].shape  # [B, T_in, C, H, W]
            tgt_shape = sample_batch["target_sequence"].shape  # [B, T_out, C, H, W]
            self.logger.info(
                f"✅ 使用通道数: input.C={in_shape[2]}, target.C={tgt_shape[2]} (单通道预测应为1)"
            )
        except Exception:
            pass

            # 观测算子与H参数设置（支持 config.observation 与 config.data.observation 两种路径，兼容嵌套 sr/crop 配置）
            obs_cfg = getattr(self.config, "observation", None)
            if obs_cfg is None:
                try:
                    obs_cfg = getattr(self.config, "data", None)
                    obs_cfg = (
                        getattr(obs_cfg, "observation", None)
                        if obs_cfg is not None
                        else None
                    )
                except Exception:
                    obs_cfg = None
            self.h_params = None
            self.observation_op = None
            if obs_cfg is not None:
                # 兼容嵌套结构：obs_cfg 可能包含 {'mode': 'sr', 'sr': {...}} 或 {'mode': 'crop', 'crop': {...}}
                mode_raw = obs_cfg.get("mode", "sr")
                mode = str(
                    mode_raw[0] if isinstance(mode_raw, (list, tuple)) else mode_raw
                ).lower()
                boundary = obs_cfg.get(
                    "boundary", obs_cfg.get("boundary_mode", "mirror")
                )
                if mode == "sr":
                    sr_sub = (
                        obs_cfg.get("sr", {})
                        if isinstance(obs_cfg.get("sr", {}), dict)
                        else {}
                    )
                    scale = obs_cfg.get("scale_factor", sr_sub.get("scale_factor", 2))
                    sigma = obs_cfg.get("blur_sigma", sr_sub.get("blur_sigma", 1.0))
                    kernel_size = obs_cfg.get(
                        "kernel_size", sr_sub.get("blur_kernel_size", 5)
                    )
                    boundary = (
                        boundary
                        if boundary is not None
                        else sr_sub.get("boundary_mode", "mirror")
                    )
                    downsample = obs_cfg.get(
                        "downsample_interpolation",
                        sr_sub.get("downsample_mode", "area"),
                    )
                    self.h_params = {
                        "task": "SR",
                        "scale": scale,
                        "sigma": sigma,
                        "kernel_size": kernel_size,
                        "boundary": boundary,
                        "downsample_interpolation": downsample,
                    }
                    self.observation_op = lambda x: apply_degradation_operator(
                        x,
                        {
                            "task": "SR",
                            "scale": scale,
                            "sigma": sigma,
                            "kernel_size": kernel_size,
                            "boundary": boundary,
                        },
                    )
                elif mode == "crop":
                    crop_sub = (
                        obs_cfg.get("crop", {})
                        if isinstance(obs_cfg.get("crop", {}), dict)
                        else {}
                    )
                    crop_size = obs_cfg.get(
                        "crop_size", crop_sub.get("crop_size", None)
                    )
                    crop_box = obs_cfg.get("crop_box", crop_sub.get("crop_box", None))
                    boundary = (
                        boundary
                        if boundary is not None
                        else crop_sub.get("boundary_mode", "mirror")
                    )
                    self.h_params = {
                        "task": "Crop",
                        "crop_size": crop_size,
                        "crop_box": crop_box,
                        "boundary": boundary,
                    }
                    self.observation_op = lambda x: apply_degradation_operator(
                        x,
                        {
                            "task": "Crop",
                            "crop_size": crop_size,
                            "crop_box": crop_box,
                            "boundary": boundary,
                        },
                    )
                else:
                    self.logger.warning(f"未知的观测模式: {mode}，跳过观测算子初始化")
                    self.h_params = None
                    self.observation_op = None
                self.logger.info(f"✅ 观测算子配置: {self.h_params}")

            # 归一化统计量，用于反归一化到原值域
            # 只有在未初始化时才设置为None，避免重复初始化时重置
            if not hasattr(self, "norm_stats"):
                self.norm_stats = None
            try:
                if self.norm_stats is None:
                    from pathlib import Path

                    import numpy as np

                    candidates = []
                    try:
                        splits_dir = getattr(
                            getattr(self.config, "data", DictConfig({})),
                            "splits_dir",
                            None,
                        )
                    except Exception:
                        splits_dir = None
                    if splits_dir:
                        candidates.append(Path(str(splits_dir)) / "norm_stat.npz")
                        candidates.append(Path(str(splits_dir)) / "norm_stats.npz")
                    candidates.append(Path(self.output_dir) / "norm_stats.npz")
                    for p in candidates:
                        try:
                            if p.exists():
                                d = np.load(str(p))
                                mean = torch.tensor(d.get("mean")).float()
                                std = torch.tensor(d.get("std")).float()
                                self.norm_stats = {"mean": mean, "std": std}
                                if mean.numel() >= 1:
                                    self.norm_stats["u_mean"] = mean[0]
                                    self.norm_stats["u_std"] = std[0]
                                if mean.numel() >= 2:
                                    self.norm_stats["v_mean"] = mean[1]
                                    self.norm_stats["v_std"] = std[1]
                                self.norm_stats["data_mean"] = self.norm_stats.get(
                                    "u_mean", torch.tensor(0.0)
                                )
                                self.norm_stats["data_std"] = self.norm_stats.get(
                                    "u_std", torch.tensor(1.0)
                                )
                                break
                        except Exception:
                            pass
                if not getattr(self, "using_synthetic", False):
                    dm = getattr(self, "data_module", None)
                    train_ds = (
                        getattr(dm, "train_dataset", None) if dm is not None else None
                    )
                else:
                    train_ds = None
                if (
                    train_ds is not None
                    and hasattr(train_ds, "mean")
                    and hasattr(train_ds, "std")
                ):
                    mean = train_ds.mean
                    std = train_ds.std
                    if isinstance(mean, torch.Tensor):
                        mean = mean.detach().cpu()
                    if isinstance(std, torch.Tensor):
                        std = std.detach().cpu()
                    self.norm_stats = {
                        "u_mean": torch.tensor(float(mean[0])),
                        "u_std": torch.tensor(float(std[0] if std[0] != 0 else 1.0)),
                        "v_mean": torch.tensor(float(mean[1])),
                        "v_std": torch.tensor(float(std[1] if std[1] != 0 else 1.0)),
                    }
                    try:
                        keys = getattr(self.config.data, "keys", ["data"])
                    except Exception:
                        keys = ["data"]
                    if isinstance(keys, (list, tuple)) and ("data" in keys):
                        self.norm_stats["data_mean"] = self.norm_stats["u_mean"]
                        self.norm_stats["data_std"] = self.norm_stats["u_std"]
                    self.logger.info(
                        f"✅ 归一化统计: u_mean={self.norm_stats['u_mean']:.3f}, u_std={self.norm_stats['u_std']:.3f}, v_mean={self.norm_stats['v_mean']:.3f}, v_std={self.norm_stats['v_std']:.3f}"
                    )
                else:
                    # 优先进行全量统计并写入 splits_dir/norm_stat.npz
                    full_ok = False
                    try:
                        from pathlib import Path

                        import numpy as np

                        splits_dir = getattr(
                            getattr(self.config, "data", DictConfig({})),
                            "splits_dir",
                            None,
                        )
                        out_path = None
                        if splits_dir:
                            out_path = Path(str(splits_dir)) / "norm_stat.npz"
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                        if (
                            hasattr(self, "train_loader")
                            and self.train_loader is not None
                        ):
                            sums = None
                            sumsq = None
                            count = 0
                            for batch in self.train_loader:
                                x = None
                                if isinstance(batch, dict):
                                    for v in batch.values():
                                        if torch.is_tensor(v) and v.dim() >= 4:
                                            x = v
                                            break
                                elif isinstance(batch, (list, tuple)):
                                    for v in batch:
                                        if torch.is_tensor(v) and v.dim() >= 4:
                                            x = v
                                            break
                                elif torch.is_tensor(batch):
                                    x = batch
                                if x is None:
                                    continue
                                if x.dim() == 5:
                                    B, T, Cx, Hx, Wx = x.shape
                                    x = x.view(B * T, Cx, Hx, Wx)
                                elif x.dim() != 4:
                                    continue
                                x = x.float()
                                c_sum = x.sum(dim=(0, 2, 3))
                                c_sumsq = (x * x).sum(dim=(0, 2, 3))
                                pixels = x.shape[0] * x.shape[2] * x.shape[3]
                                sums = c_sum if sums is None else (sums + c_sum)
                                sumsq = c_sumsq if sumsq is None else (sumsq + c_sumsq)
                                count += pixels
                            if count > 0 and sums is not None:
                                mean = sums / count
                                var = sumsq / count - mean * mean
                                var = torch.clamp(var, min=1e-8)
                                std = torch.sqrt(var)
                                self.norm_stats = {
                                    "mean": mean.clone(),
                                    "std": std.clone(),
                                }
                                if mean.numel() >= 1:
                                    self.norm_stats["u_mean"] = mean[0]
                                    self.norm_stats["u_std"] = std[0]
                                if mean.numel() >= 2:
                                    self.norm_stats["v_mean"] = mean[1]
                                    self.norm_stats["v_std"] = std[1]
                                self.norm_stats["data_mean"] = self.norm_stats.get(
                                    "u_mean", torch.tensor(0.0)
                                )
                                self.norm_stats["data_std"] = self.norm_stats.get(
                                    "u_std", torch.tensor(1.0)
                                )
                                if out_path is not None:
                                    try:
                                        np.savez(
                                            str(out_path),
                                            mean=mean.cpu().numpy(),
                                            std=std.cpu().numpy(),
                                        )
                                    except Exception:
                                        pass
                                full_ok = True
                    except Exception:
                        full_ok = False
                    if not full_ok:
                        ok = False
                        try:
                            if hasattr(
                                self, "_compute_norm_stats_from_data"
                            ) and callable(self._compute_norm_stats_from_data):
                                ok = self._compute_norm_stats_from_data()
                        except Exception:
                            ok = False
                        if (
                            ok
                            and hasattr(self, "norm_stats")
                            and self.norm_stats is not None
                        ):
                            if (
                                "u_mean" not in self.norm_stats
                                and "mean" in self.norm_stats
                            ):
                                m = self.norm_stats["mean"]
                                s = self.norm_stats["std"]
                                self.norm_stats["u_mean"] = m[0]
                                self.norm_stats["u_std"] = s[0]
                            self.norm_stats["data_mean"] = self.norm_stats.get(
                                "u_mean", torch.tensor(0.0)
                            )
                            self.norm_stats["data_std"] = self.norm_stats.get(
                                "u_std", torch.tensor(1.0)
                            )
                        else:
                            C = self.config.model.out_channels
                            self.norm_stats = {
                                "mean": torch.zeros(C),
                                "std": torch.ones(C),
                                "u_mean": torch.tensor(0.0),
                                "u_std": torch.tensor(1.0),
                                "v_mean": torch.tensor(0.0),
                                "v_std": torch.tensor(1.0),
                            }
                            self.norm_stats["data_mean"] = self.norm_stats["u_mean"]
                            self.norm_stats["data_std"] = self.norm_stats["u_std"]
            except Exception as e:
                self.logger.warning(f"⚠️ 归一化统计提取失败: {e}")
                # 提供默认归一化统计，避免后续代码出错
                C = self.config.model.out_channels
                self.norm_stats = {
                    "mean": torch.zeros(C),
                    "std": torch.ones(C),
                    "u_mean": torch.tensor(0.0),
                    "u_std": torch.tensor(1.0),
                    "v_mean": torch.tensor(0.0),
                    "v_std": torch.tensor(1.0),
                }

            # 一次性形状与归一化检查日志
            try:
                inp = sample_batch["input_sequence"]
                tgt = sample_batch["target_sequence"]
                # 形状断言
                assert (
                    inp.ndim == 5 and tgt.ndim == 5
                ), f"Input/Target dims incorrect: {inp.ndim}/{tgt.ndim}"
                assert (
                    inp.shape[2] == tgt.shape[2]
                ), f"Channel mismatch: {inp.shape[2]} vs {tgt.shape[2]}"
                assert (
                    inp.shape[-2:] == tgt.shape[-2:]
                ), f"Spatial mismatch: {inp.shape[-2:]} vs {tgt.shape[-2:]}"
                # 严格使用配置中的通道数，避免运行时修改
                try:
                    in_ch = int(self._cfg_select("model.in_channels", default=4))
                    out_ch = int(self._cfg_select("model.out_channels", default=1))
                    if in_ch != 4 or out_ch != 1:
                        self.logger.warning(
                            f"建议使用固定通道配置 in=4(out1)：当前 in={in_ch}, out={out_ch}"
                        )
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
                if not hasattr(self.config, "data"):
                    self.config.data = DictConfig({})
                if not hasattr(self.config.data, "keys") or not self.config.data.keys:
                    self.config.data.keys = ["data"]
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
        is_oom = any(
            keyword in error_msg
            for keyword in [
                "out of memory",
                "cuda out of memory",
                "oom",
                "memory",
                "cuda runtime error",
                "allocation",
                "insufficient memory",
            ]
        )

        if is_oom:
            return self.adjust_batch_size_on_oom(error, phase)
        else:
            # 其他CUDA错误，记录详细信息
            self.logger.error(f"❌ CUDA错误在{phase}阶段: {error}")
            self.logger.error(f"错误类型: {type(error).__name__}")
            return False

    def adjust_batch_size_on_oom(
        self, error: Exception = None, phase: str = "training"
    ) -> bool:
        """在内存不足时动态调整批次大小"""
        try:
            curr_bs = int(getattr(self, "current_batch_size", 0) or 0)
        except Exception:
            curr_bs = 0

        # 记录详细的OOM信息
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.warning(
                f"💾 GPU内存状态: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB, 总计 {total:.2f}GB"
            )

        if not (self.memory_config["auto_batch_size_reduction"] and curr_bs > 1):
            return False
        if error:
            self.logger.warning(f"OOM错误详情: {error}")
        # 逐步减半直到成功或到1
        new_bs = curr_bs
        while new_bs > 1:
            new_bs = max(1, new_bs // 2)
            self.logger.warning(f"内存不足，将批次大小从 {curr_bs} 调整为 {new_bs}")
            try:
                # 清理缓存，减少碎片影响
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                # DataLoader参数调整
                num_workers = int(
                    self._cfg_select(
                        "data.dataloader.num_workers", "hardware.num_workers", default=0
                    )
                    or 0
                )
                if (
                    num_workers == 0
                    and hasattr(self.config, "data")
                    and hasattr(self.config.data, "dataloader")
                ):
                    try:
                        self.config.data.dataloader.prefetch_factor = None
                        self.config.data.dataloader.persistent_workers = False
                        self.logger.debug(
                            "⚙️ OOM调整: num_workers=0 → prefetch_factor=None, persistent_workers=False"
                        )
                    except Exception as e:
                        self.logger.warning(f"设置prefetch_factor=None失败: {e}")
                # 更新所有批次尺寸配置
                if hasattr(self.config, "data") and hasattr(
                    self.config.data, "dataloader"
                ):
                    try:
                        self.config.data.dataloader.batch_size = new_bs
                        if hasattr(self.config.data.dataloader, "val_batch_size"):
                            self.config.data.dataloader.val_batch_size = new_bs
                        if hasattr(self.config.data.dataloader, "test_batch_size"):
                            self.config.data.dataloader.test_batch_size = new_bs
                    except Exception:
                        pass
                if hasattr(self.config, "training"):
                    try:
                        self.config.training.batch_size = new_bs
                        if hasattr(self.config.training, "dataloader") and hasattr(
                            self.config.training.dataloader, "batch_size"
                        ):
                            self.config.training.dataloader.batch_size = new_bs
                    except Exception:
                        pass
                # 重建数据加载器
                try:
                    try:
                        self.data_module.batch_size = new_bs
                    except Exception:
                        pass
                    self.data_module = (
                        _DMCfg(self.config)
                        if " _DMCfg" in globals()
                        else self.data_module
                    )
                    self.data_module.setup()
                    self.train_loader = self.data_module.train_dataloader()
                    self.val_loader = self.data_module.val_dataloader()
                    self.test_loader = self.data_module.test_dataloader()
                except Exception as e:
                    self.logger.error(f"重建数据加载器失败: {e}")
                    return False
                self.current_batch_size = new_bs
                # 动态提升梯度累积以保持稳定
                try:
                    if (
                        int(
                            self.memory_config.get("gradient_accumulation_steps", 1)
                            or 1
                        )
                        == 1
                        and new_bs < curr_bs
                    ):
                        self.memory_config["gradient_accumulation_steps"] = 2
                except Exception:
                    pass
                return True
            except Exception as e:
                self.logger.warning(f"批次调整失败，继续尝试更小批量: {e}")
                continue
        return False

    def setup_model(self):
        """设置模型 - 支持分阶段预测架构"""
        # 检查是否启用分阶段预测架构 - 修复配置解析
        try:
            # 尝试多种配置路径
            sequential_enabled = False
            if hasattr(self.config, "model") and hasattr(
                self.config.model, "sequential"
            ):
                sequential_enabled = bool(
                    self.config.model.sequential.get("enabled", False)
                )
            elif hasattr(self.config, "sequential"):
                sequential_enabled = bool(self.config.sequential.get("enabled", False))
            else:
                # 最后尝试从配置字典获取
                sequential_enabled = bool(
                    self._cfg_select(
                        "model.sequential.enabled", "sequential.enabled", default=False
                    )
                )

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
            model_name = str(
                self._cfg_select(
                    "model.name",
                    "model.type",
                    "model.architecture",
                    default="swin_unet",
                )
            ).lower()
            if getattr(self, "model_name", None):
                model_name = str(self.model_name).lower()
            self.logger.info(f"使用模型架构: {model_name}")

            # 获取所有可用模型列表
            available_models = list_models()
            self.logger.info(f"可用模型: {available_models}")

            # 检查模型是否可用
            if model_name not in available_models:
                self.logger.warning(
                    f"模型 {model_name} 不在可用模型列表中，尝试使用模型加载器创建"
                )

            # 获取模型配置参数
            # Robustly handle img_size which can be int or list/ListConfig
            img_size_val = self._cfg_select(
                "model.img_size", "data.img_size", default=128
            )
            if hasattr(img_size_val, "__iter__") and not isinstance(img_size_val, str):
                # If list-like (e.g. ListConfig or list), use first element assuming square
                try:
                    img_size_val = int(img_size_val[0])
                except (IndexError, ValueError, TypeError):
                    self.logger.warning(
                        f"Could not parse img_size {img_size_val}, defaulting to 128"
                    )
                    img_size_val = 128
            else:
                img_size_val = int(img_size_val)

            model_config = {
                "in_channels": int(
                    self._cfg_select("model.in_channels", "data.channels", default=1)
                ),
                "out_channels": int(
                    self._cfg_select("model.out_channels", "data.channels", default=1)
                ),
                "img_size": img_size_val,
            }
            # 透传包装器特有参数
            enc_out = self._cfg_select("model.encoder_out_channels", default=None)
            if enc_out is not None:
                try:
                    model_config["encoder_out_channels"] = int(enc_out)
                except Exception:
                    pass
            post_head = self._cfg_select("model.post_conv3x3", default=None)
            if post_head is not None:
                try:
                    model_config["post_conv3x3"] = bool(post_head)
                except Exception:
                    pass

            # 根据模型类型添加特定参数
            if model_name == "swin_unet":
                # SwinUNet特定参数
                try:
                    patch_size = int(
                        self._cfg_select(
                            "model.patch_size", "training.patch_size", default=4
                        )
                    )
                    depths = list(
                        self._cfg_select("model.depths", default=[2, 2, 6, 2])
                    )
                    win = int(self._cfg_select("model.window_size", default=8))

                    # window_size合法性校验
                    if model_config["img_size"] % max(patch_size, 1) != 0:
                        self.logger.warning(
                            f"img_size({model_config['img_size']}) 不能被 patch_size({patch_size}) 整除"
                        )

                    from math import gcd

                    patch_res = model_config["img_size"] // max(patch_size, 1)
                    stage_res = [
                        max(patch_res // (2**i), 1) for i in range(len(depths))
                    ]
                    g = stage_res[0]
                    for r in stage_res[1:]:
                        g = gcd(g, r)
                    safe_win = max(1, min(win, g))

                    if safe_win != win:
                        self.logger.warning(
                            f"⚠️ 调整window_size: {win}→{safe_win} 以匹配阶段分辨率 {stage_res}"
                        )
                        try:
                            self.config.model.window_size = safe_win
                        except Exception:
                            pass

                    model_config.update(
                        {
                            "patch_size": patch_size,
                            "depths": depths,
                            "window_size": safe_win if "safe_win" in locals() else win,
                            "embed_dim": int(
                                self._cfg_select("model.embed_dim", default=96)
                            ),
                            "num_heads": list(
                                self._cfg_select(
                                    "model.num_heads", default=[3, 6, 12, 24]
                                )
                            ),
                            "mlp_ratio": float(
                                self._cfg_select("model.mlp_ratio", default=4.0)
                            ),
                            "drop_rate": float(
                                self._cfg_select("model.drop_rate", default=0.0)
                            ),
                            "attn_drop_rate": float(
                                self._cfg_select("model.attn_drop_rate", default=0.0)
                            ),
                            "drop_path_rate": float(
                                self._cfg_select("model.drop_path_rate", default=0.1)
                            ),
                            "use_checkpoint": bool(
                                self._cfg_select(
                                    "device.memory_management.gradient_checkpointing",
                                    "training.gradient_checkpointing",
                                    default=False,
                                )
                            ),
                            "use_sdpa": bool(
                                self._cfg_select(
                                    "training.use_flash_attention",
                                    "model.use_flash_attention",
                                    default=False,
                                )
                            ),
                            "sdpa_kernel": str(
                                self._cfg_select(
                                    "training.sdpa_kernel",
                                    "model.sdpa_kernel",
                                    default="auto",
                                )
                            ),
                        }
                    )
                except Exception as _werr:
                    self.logger.warning(f"⚠️ SwinUNet参数设置失败: {_werr}")
            elif ("swin" in model_name) and (model_name != "swin_unet"):
                try:
                    patch_size = int(
                        self._cfg_select(
                            "model.patch_size", "training.patch_size", default=4
                        )
                    )
                    depths = list(
                        self._cfg_select("model.depths", default=[2, 2, 6, 2])
                    )
                    win = int(self._cfg_select("model.window_size", default=7))
                    from math import gcd

                    img_sz = int(model_config["img_size"])
                    patch_res = img_sz // max(patch_size, 1)
                    stage_res = [
                        max(patch_res // (2**i), 1) for i in range(len(depths))
                    ]
                    g = stage_res[0]
                    for r in stage_res[1:]:
                        g = gcd(g, r)
                    safe_win = max(1, min(win, g))
                    if safe_win != win:
                        self.logger.warning(
                            f"⚠️ 调整Swin window_size: {win}→{safe_win} 以匹配阶段分辨率 {stage_res}"
                        )
                        try:
                            self.config.model.window_size = safe_win
                        except Exception:
                            pass
                    model_config.update(
                        {
                            "patch_size": patch_size,
                            "depths": depths,
                            "window_size": safe_win if "safe_win" in locals() else win,
                            "embed_dim": int(
                                self._cfg_select("model.embed_dim", default=96)
                            ),
                            "num_heads": list(
                                self._cfg_select(
                                    "model.num_heads", default=[3, 6, 12, 24]
                                )
                            ),
                            "mlp_ratio": float(
                                self._cfg_select("model.mlp_ratio", default=4.0)
                            ),
                            "drop_rate": float(
                                self._cfg_select("model.drop_rate", default=0.0)
                            ),
                            "attn_drop_rate": float(
                                self._cfg_select("model.attn_drop_rate", default=0.0)
                            ),
                            "drop_path_rate": float(
                                self._cfg_select("model.drop_path_rate", default=0.1)
                            ),
                            "use_checkpoint": bool(
                                self._cfg_select(
                                    "device.memory_management.gradient_checkpointing",
                                    "training.gradient_checkpointing",
                                    default=False,
                                )
                            ),
                        }
                    )
                except Exception as _st_err:
                    self.logger.warning(f"⚠️ Swin家族参数设置失败: {_st_err}")

            # 添加通用参数
            additional_params = {}

            # 1. 优先处理标准参数列表
            standard_keys = [
                "embed_dim",
                "num_heads",
                "depths",
                "mlp_ratio",
                "drop_rate",
                "attn_drop_rate",
                "drop_path_rate",
                "patch_size",
                "window_size",
                "use_checkpoint",
                "use_sdpa",
                "sdpa_kernel",
            ]

            for key in standard_keys:
                try:
                    value = self._cfg_select(f"model.{key}", default=None)
                    if value is not None:
                        if key in ["depths", "num_heads"]:
                            additional_params[key] = (
                                list(value)
                                if isinstance(value, (list, tuple))
                                else value
                            )
                        elif key in [
                            "mlp_ratio",
                            "drop_rate",
                            "attn_drop_rate",
                            "drop_path_rate",
                        ]:
                            additional_params[key] = float(value)
                        elif key in ["embed_dim", "patch_size", "window_size"]:
                            additional_params[key] = int(value)
                        else:
                            additional_params[key] = value
                except Exception:
                    pass

            # 2. 补充其他未被包含的模型参数 (支持 EDSR, RDN 等自定义参数)
            if hasattr(self.config, "model") and not isinstance(self.config.model, str):
                if hasattr(self.config.model, "items"):
                    for key, value in self.config.model.items():
                        if (
                            key not in model_config
                            and key not in additional_params
                            and key
                            not in [
                                "name",
                                "type",
                                "architecture",
                                "sequential",
                                "ar_config",
                            ]
                        ):
                            # 简单的类型转换尝试
                            if hasattr(value, "__iter__") and not isinstance(
                                value, str
                            ):
                                try:
                                    additional_params[key] = list(value)
                                except:
                                    additional_params[key] = value
                            else:
                                additional_params[key] = value

            # 添加LIIF相关参数
            if self.use_liif_decoder or (
                hasattr(self.config.model, "use_liif_decoder")
                and self.config.model.use_liif_decoder
            ):
                additional_params["use_liif_decoder"] = True
                additional_params["liif_mlp_hidden"] = int(
                    getattr(self.config.model, "liif_mlp_hidden", 64)
                )
                print(
                    f"Adding LIIF params to model config: {additional_params['use_liif_decoder']}, hidden={additional_params['liif_mlp_hidden']}"
                )

            model_config.update(additional_params)

            try:
                mb = getattr(self.config, "model_budget", None)
                target_m = None
                tol_m = 0.5
                auto_tune = False
                if mb is not None:
                    target_m = (
                        float(getattr(mb, "target_params_m", None))
                        if getattr(mb, "target_params_m", None) is not None
                        else None
                    )
                    tol_m = float(getattr(mb, "tolerance_m", 0.5))
                    auto_tune = bool(getattr(mb, "auto_tune", False))
                if auto_tune and target_m is not None:
                    # 增强的自动调优逻辑
                    try:
                        # 针对特定架构的调优策略
                        base_model_cls = model_name.lower()

                        # 1. UNO (通常层数过多，参数量巨大)
                        if "uno" in base_model_cls:
                            # UNO 默认配置往往很大，大幅削减宽度和深度
                            current_width = int(model_config.get("width", 64))
                            model_config["width"] = min(current_width, 32)  # 限制宽度
                            if "in_channels" not in model_config:
                                model_config["in_channels"] = 1
                            if "out_channels" not in model_config:
                                model_config["out_channels"] = 1

                        # 2. NAFNet / Restormer (通常通道数和块数过多)
                        elif (
                            "nafnet" in base_model_cls or "restormer" in base_model_cls
                        ):
                            current_width = int(model_config.get("width", 32))
                            if current_width > 16:
                                model_config["width"] = 16  # 激进压缩宽度

                            # 减少块数
                            if "enc_blk_nums" in model_config:
                                model_config["enc_blk_nums"] = [1, 1, 1, 1]
                            if "dec_blk_nums" in model_config:
                                model_config["dec_blk_nums"] = [1, 1, 1, 1]
                            if "middle_blk_num" in model_config:
                                model_config["middle_blk_num"] = 1

                        # 3. RCAN / RDN / EDSR (基于残差块，容易过大或过小)
                        elif any(x in base_model_cls for x in ["rcan", "rdn", "edsr"]):
                            # 先尝试默认调优，如果太小则增加，太大则减少
                            pass  # 让后续通用逻辑处理，或者在这里特化

                        # 4. PerceiverIO (通常太小)
                        elif "perceiver" in base_model_cls:
                            # 增加 latent 维度或数量
                            if "num_latents" in model_config:
                                model_config["num_latents"] = max(
                                    int(model_config["num_latents"]), 256
                                )
                            if "latent_dim" in model_config:
                                model_config["latent_dim"] = max(
                                    int(model_config["latent_dim"]), 256
                                )

                        # 执行通用自动调优
                        model_config = self._auto_tune_model_params(
                            model_name, model_config, target_m, tol_m
                        )

                    except Exception as e_tune_enhance:
                        self.logger.warning(
                            f"增强自动调优失败，回退到标准调优: {e_tune_enhance}"
                        )
                        model_config = self._auto_tune_model_params(
                            model_name, model_config, target_m, tol_m
                        )
            except Exception:
                pass

            # 使用增强模型加载器创建基础模型（四层回退策略）
            base_model = None
            model_creation_errors = []
            creation_method = None

            # -----------------------------------------------------------
            # 简化且健壮的模型创建逻辑 (Robust Model Creation)
            # -----------------------------------------------------------

            # 优先级 1: 标准注册表 (Standard Registry) - 最可靠
            # 支持 aliases (如 UformerLite -> ConvUNetLite)
            try:
                from models import create_model as registry_create_model

                # 注意: registry_create_model 接受 (name, **kwargs)
                base_model = registry_create_model(model_name, **model_config)
                creation_method = "standard_registry"
                self.logger.info(
                    f"✅ 使用标准注册表成功创建模型: {type(base_model).__name__} (Request: {model_name})"
                )
            except Exception as e_reg:
                self.logger.warning(f"⚠️ 标准注册表创建失败 ({model_name}): {e_reg}")

                # 优先级 2: 原始模型加载器 (Original ModelLoader) - 支持扫描和兼容性
                try:
                    from tools.training.model_loader import create_model_with_loader

                    base_model = create_model_with_loader(
                        model_name, self.config, **model_config
                    )
                    creation_method = "original_loader"
                    self.logger.info(
                        f"✅ 使用 ModelLoader 成功创建模型: {type(base_model).__name__}"
                    )
                except Exception as e_loader:
                    self.logger.error(f"❌ ModelLoader 创建失败: {e_loader}")

                    # 优先级 3: 增强/改进加载器 (Enhanced/Improved) - 仅作为最后手段
                    # 之前导致了 alias 解析问题，现在仅作为备用
                    try:
                        from tools.training.model_loader_enhanced import (
                            create_enhanced_model,
                        )

                        base_model = create_enhanced_model(
                            model_name, self.config, **model_config
                        )
                        creation_method = "enhanced_loader"
                        self.logger.info(
                            f"✅ 使用增强加载器成功创建模型: {type(base_model).__name__}"
                        )
                    except Exception as e_enhanced:
                        self.logger.error(f"❌ 增强加载器也失败: {e_enhanced}")
                        raise RuntimeError(
                            f"无法创建模型 {model_name}. \nRegistry Error: {e_reg}\nLoader Error: {e_loader}\nEnhanced Error: {e_enhanced}"
                        )

            # 根据配置禁用时间预测，仅空间预测时直接使用基础模型
            try:
                ar_enabled = bool(getattr(self.config, "ar", {}).get("enabled", True))
            except Exception:
                ar_enabled = True

            if ar_enabled:
                # 包装为AR模型
                self.model = ARWrapper(
                    single_frame_model=base_model,
                    detach_rollout=True,
                    scheduled_sampling=False,
                )
            else:
                # 仅空间预测：直接使用单帧模型，统一 forward(x)->y
                self.model = base_model

            # 可选：转换为SyncBatchNorm以配合DDP
            try:
                if bool(getattr(self.config.training, "sync_batchnorm", False)):
                    base = self.model
                    if hasattr(base, "module"):
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
                real_device = (
                    self.device
                    if isinstance(self.device, torch.device)
                    else torch.device(self.device)
                )
                self.logger.info(
                    f"🧩 BaseModel={base_cls}, Wrapper={wrapped_cls}, Device={real_device}"
                )
            except Exception:
                pass

            # 性能优化：channels_last 与 torch.compile（在DDP包裹之前进行）
            try:
                use_channels_last = bool(
                    self._cfg_select(
                        "training.channels_last", "device.channels_last", default=False
                    )
                )
            except Exception:
                use_channels_last = False
            if use_channels_last and self.device.type == "cuda":
                try:
                    # 全局设置模型为channels_last内存格式
                    try:
                        self.model.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                    # 兜底：逐参数设置内存格式
                    for p in self.model.parameters():
                        if p.is_cuda and p.dim() >= 4:
                            p.data = p.data.contiguous(
                                memory_format=torch.channels_last
                            )
                    self.logger.info(
                        "🧠 模型设为channels_last内存格式（包含逐参数兜底）"
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ 设置channels_last失败: {e}")

            # 按配置启用 torch.compile（Inductor，reduce-overhead），在DDP之前编译
            compile_enabled = False
            compile_backend = "inductor"
            compile_mode = "reduce-overhead"
            try:
                # 支持 training.torch_compile 与 device.compile_model 两种入口
                compile_enabled = bool(
                    self._cfg_select(
                        "training.torch_compile", "device.compile_model", default=False
                    )
                )
                compile_backend = str(
                    self._cfg_select(
                        "training.torch_compile_backend", default="inductor"
                    )
                )
                compile_mode = str(
                    self._cfg_select(
                        "training.torch_compile_mode", default="reduce-overhead"
                    )
                )
            except Exception:
                pass
            if compile_enabled:
                try:
                    base_cls = type(self.model).__name__.lower()
                    if "swin" in base_cls:
                        raise RuntimeError(
                            "skip compile for SwinUNet due to CUDA graphs overwrite issue"
                        )
                    self.model = torch.compile(
                        self.model, backend=compile_backend, mode=compile_mode
                    )
                    self.logger.info(
                        f"🚀 已启用torch.compile: backend={compile_backend}, mode={compile_mode}"
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ torch.compile失败或已跳过: {e}")

            # 将TF32设置日志化（在setup_device中已设置），这里补充记录sdpa与kernel选择
            try:
                allow_tf32 = bool(
                    self._cfg_select("hardware.memory.allow_tf32", default=False)
                )
                use_flash = bool(
                    self._cfg_select(
                        "training.use_flash_attention",
                        "model.use_flash_attention",
                        default=False,
                    )
                )
                sdpa_kernel = str(
                    self._cfg_select(
                        "training.sdpa_kernel", "model.sdpa_kernel", default="auto"
                    )
                )
                self.logger.info(
                    f"🔧 注意力: use_sdpa/flash={use_flash}, sdpa_kernel={sdpa_kernel}, TF32={allow_tf32}"
                )
            except Exception:
                pass

            # 记录AMP dtype选择
            try:
                dtype_str = "default"
                if self.autocast_dtype is torch.float16:
                    dtype_str = "float16"
                elif self.autocast_dtype is torch.bfloat16:
                    dtype_str = "bfloat16"
                self.logger.info(f"🧪 AMP autocast dtype: {dtype_str}")
            except Exception:
                pass

            # DDP优先，其次DataParallel
            try:
                if getattr(self, "distributed", False):
                    if (
                        isinstance(self.device, torch.device)
                        and self.device.type == "cuda"
                    ):
                        dev_id = (
                            self.local_rank
                            if hasattr(self, "local_rank")
                            else (
                                self.device.index
                                if self.device.index is not None
                                else None
                            )
                        )
                        device_ids = [dev_id] if dev_id is not None else None
                        output_device = dev_id
                    else:
                        device_ids = None
                        output_device = None
                    self.model = torch.nn.parallel.DistributedDataParallel(
                        self.model,
                        device_ids=device_ids,
                        output_device=output_device,
                        find_unused_parameters=False,
                    )
                else:
                    # Explicitly disable DataParallel fallback
                    pass
            except Exception as e:
                self.logger.warning(f"⚠️ 并行处理设置失败: {e}")

            # 计算参数量与记录FLOPs/推理延迟（单次采样）
            model_for_params = (
                self.model.module if hasattr(self.model, "module") else self.model
            )

            # 详细打印分模块参数量
            self.logger.info("-" * 50)
            self.logger.info("📊 模型参数分布 (Parameter Breakdown):")

            total_params = 0
            trainable_params = 0

            # 1. 如果是分阶段模型 (SequentialSpatiotemporalModel)
            if hasattr(model_for_params, "spatial_module") and hasattr(
                model_for_params, "temporal_module"
            ):
                # 空间模块
                spatial_p = sum(
                    p.numel() for p in model_for_params.spatial_module.parameters()
                )
                spatial_trainable = sum(
                    p.numel()
                    for p in model_for_params.spatial_module.parameters()
                    if p.requires_grad
                )
                self.logger.info(
                    f"  🔹 [Spatial Module]: {spatial_p:,} params (Trainable: {spatial_trainable:,})"
                )

                # 时序模块
                temporal_p = sum(
                    p.numel() for p in model_for_params.temporal_module.parameters()
                )
                temporal_trainable = sum(
                    p.numel()
                    for p in model_for_params.temporal_module.parameters()
                    if p.requires_grad
                )
                self.logger.info(
                    f"  🔹 [Temporal Module]: {temporal_p:,} params (Trainable: {temporal_trainable:,})"
                )

                total_params = spatial_p + temporal_p
                trainable_params = spatial_trainable + temporal_trainable

            else:
                # 2. 如果是普通模型，尝试打印第一层子模块
                for name, child in model_for_params.named_children():
                    child_p = sum(p.numel() for p in child.parameters())
                    child_trainable = sum(
                        p.numel() for p in child.parameters() if p.requires_grad
                    )
                    self.logger.info(f"  🔹 [{name}]: {child_p:,} params")

                total_params = sum(p.numel() for p in model_for_params.parameters())
                trainable_params = sum(
                    p.numel() for p in model_for_params.parameters() if p.requires_grad
                )

            self.logger.info("-" * 50)
            self.logger.info(
                f"✅ 总参数量 (Total): {total_params:,} (可训练: {trainable_params:,})"
            )
            self.logger.info("-" * 50)

            # -----------------------------------------------------------
            # 严格参数限制检查 (Strict Parameter Budget Check)
            # -----------------------------------------------------------
            try:
                mb = getattr(self.config, "model_budget", None)
                if mb is not None:
                    target_m = float(getattr(mb, "target_params_m", 0))
                    # 默认关闭严格模式，允许参数量不达标的模型继续运行
                    strict_mode = bool(getattr(mb, "strict_mode", False))

                    if target_m > 0:
                        current_m = total_params / 1e6
                        tolerance = float(getattr(mb, "tolerance_m", 0.5))

                        diff = abs(current_m - target_m)
                        if diff > tolerance:
                            msg = (
                                f"❌ 模型参数量 ({current_m:.2f}M) 超出预算目标 "
                                f"({target_m:.2f}M ± {tolerance:.2f}M). "
                                f"差异: {diff:.2f}M"
                            )

                            if strict_mode:
                                self.logger.error(msg)
                                self.logger.error(
                                    "已启用严格模式 (strict_mode=True)，终止训练。"
                                )
                                self.logger.error(
                                    "建议: 调整模型配置或增大 model_budget.tolerance_m"
                                )
                                raise RuntimeError(msg)
                            else:
                                self.logger.warning(
                                    f"⚠️ {msg} (strict_mode=False, 继续训练)"
                                )
                        else:
                            self.logger.info(
                                f"✅ 模型参数量 ({current_m:.2f}M) 符合预算目标 ({target_m}M ± {tolerance}M)"
                            )
            except Exception as e_budget:
                if isinstance(e_budget, RuntimeError):
                    raise e_budget
                self.logger.warning(f"⚠️ 参数预算检查执行失败: {e_budget}")

            # 写入模型信息（健壮的 try/except 包裹，避免解析期错误）
            try:
                import json as _json
            except Exception:
                _json = None
            selected_model = str(
                getattr(
                    self,
                    "model_name",
                    getattr(getattr(self.config, "model", {}), "name", "unknown"),
                )
            )
            config_model_name = str(
                getattr(getattr(self.config, "model", {}), "name", "unknown")
            )
            model_class = type(model_for_params).__name__
            info = {
                "selected_model": selected_model,
                "config_model_name": config_model_name,
                "model_class": model_class,
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
            }
            try:
                outp = self.output_dir / "model_info.json"
                if _json is not None:
                    with open(outp, "w") as f:
                        _json.dump(info, f, indent=2)
                self.logger.info(
                    f"ActiveModelClass={model_class} SelectedModel={selected_model} ConfigModelName={config_model_name}"
                )
                self.logger.info(f"📝 写入模型信息: {outp}")
            except Exception as _wr_info_err:
                self.logger.warning(f"写入模型信息失败: {_wr_info_err}")

            # 资源统计：FLOPs与延迟（以当前img_size/通道配置为准，输入形状[B,C,H,W]）
            try:
                from utils.performance import PerformanceProfiler

                profiler = PerformanceProfiler(device=self.device.type)
                img_size = getattr(self.config.model, "img_size", None)
                try:
                    from omegaconf import ListConfig  # type: ignore

                    if isinstance(img_size, ListConfig):
                        img_size = list(img_size)
                except Exception:
                    pass
                if isinstance(img_size, (list, tuple)):
                    if len(img_size) >= 2:
                        h, w = int(img_size[0]), int(img_size[1])
                    elif len(img_size) == 1:
                        h = w = int(img_size[0])
                    else:
                        h = w = int(getattr(self.config.data, "img_size", 224))
                else:
                    s = int(
                        img_size
                        if img_size is not None
                        else getattr(self.config.data, "img_size", 224)
                    )
                    h = w = s
                input_shape = (1, int(self.config.model.in_channels), int(h), int(w))
                # 移动到设备
                model_for_perf = self.model
                if hasattr(model_for_perf, "module"):
                    model_for_perf = model_for_perf.module
                model_for_perf.eval()
                dummy = torch.randn(input_shape, device=self.device)
                flops_info = profiler.calculate_flops(model_for_perf, dummy)
                latency_info = profiler.measure_inference_latency(
                    model_for_perf, dummy, num_runs=20, warmup_runs=5
                )
                # 记录到日志与保存资源信息文件
                resource_info = {
                    "params": total_params,
                    "params_trainable": trainable_params,
                    "flops_total": int(flops_info.get("total", 0)),
                    "flops_g": float(flops_info.get("total_gflops", 0.0)),
                    "inference_latency_ms_mean": float(
                        latency_info.get("mean_ms", 0.0)
                    ),
                    "inference_latency_ms_std": float(latency_info.get("std_ms", 0.0)),
                    "input_shape": input_shape,
                }
                self.logger.info(
                    f"📊 资源: FLOPs={resource_info['flops_g']:.3f}G@{input_shape[2]}x{input_shape[3]}, "
                    f"延迟={resource_info['inference_latency_ms_mean']:.2f}±{resource_info['inference_latency_ms_std']:.2f}ms"
                )
                try:
                    with open(self.output_dir / "model_resources.json", "w") as f:
                        json.dump(resource_info, f, indent=2)
                except Exception as _wr_err:
                    self.logger.warning(f"写入资源文件失败: {_wr_err}")
            except Exception as e:
                self.logger.warning(f"资源统计失败，继续训练: {e}")

            # 已在DDP前处理channels_last与torch.compile

            # 存储归一化统计到trainer，供损失函数统一使用
            try:
                # RealDiffusionReactionDataset 使用 mean/std 为 Tensor[C]
                train_ds = getattr(self.data_module, "train_dataset", None)
                if (
                    train_ds is not None
                    and hasattr(train_ds, "mean")
                    and hasattr(train_ds, "std")
                ):
                    mean = (
                        train_ds.mean
                        if isinstance(train_ds.mean, torch.Tensor)
                        else torch.as_tensor(train_ds.mean, dtype=torch.float32)
                    )
                    std = (
                        train_ds.std
                        if isinstance(train_ds.std, torch.Tensor)
                        else torch.as_tensor(train_ds.std, dtype=torch.float32)
                    )
                    # 组装为 norm_stats 字典，包含通道级 'mean'/'std' 以及兼容旧键名（u/v）
                    self.norm_stats = {"mean": mean.clone(), "std": std.clone()}
                    # 兼容: 若为双通道，提供 u/v 键名，避免旧代码报错
                    try:
                        if mean.numel() >= 1:
                            self.norm_stats["u_mean"] = mean[0]
                            self.norm_stats["u_std"] = std[0]
                        if mean.numel() >= 2:
                            self.norm_stats["v_mean"] = mean[1]
                            self.norm_stats["v_std"] = std[1]
                    except Exception:
                        pass
                    self.logger.info(
                        "✅ 已提取归一化统计用于损失：提供 'mean/std' 通道级统计"
                    )
                else:
                    # 回退：使用零均值、单位方差
                    C = self.config.model.out_channels
                    zeros = torch.zeros(C)
                    ones = torch.ones(C)
                    self.norm_stats = {"mean": zeros.clone(), "std": ones.clone()}
                    # 兼容旧键名
                    try:
                        self.norm_stats["u_mean"] = zeros[0]
                        self.norm_stats["u_std"] = ones[0]
                        if C > 1:
                            self.norm_stats["v_mean"] = zeros[1]
                            self.norm_stats["v_std"] = ones[1]
                    except Exception:
                        pass
                    try:
                        keys = getattr(self.config.data, "keys", ["data"])
                    except Exception:
                        keys = ["data"]
                    if isinstance(keys, (list, tuple)) and ("data" in keys):
                        self.norm_stats["data_mean"] = self.norm_stats.get(
                            "u_mean", torch.tensor(0.0)
                        )
                        self.norm_stats["data_std"] = self.norm_stats.get(
                            "u_std", torch.tensor(1.0)
                        )
                    self.logger.warning(
                        "⚠️ 数据集未提供mean/std，使用默认0/1 归一化统计（含通道级 'mean/std'）"
                    )
            except Exception as e:
                self.logger.warning(f"⚠️ 归一化统计组装失败: {e}")

        except Exception as e:
            self.logger.error(f"❌ 模型设置失败: {e}")
            raise

    def _auto_tune_model_params(
        self,
        model_name: str,
        model_config: dict[str, Any],
        target_params_m: float,
        tolerance_m: float,
    ) -> dict[str, Any]:
        try:
            low, high = 0.5, 4.0
            best = model_config.copy()
            best_err = float("inf")

            def build(cfg: dict[str, Any]) -> int:
                try:
                    # 对于MLP模型，需要将embed_dim映射为hidden_dims
                    if (
                        "mlp" in str(model_name).lower()
                        and "mixer" not in str(model_name).lower()
                        and "embed_dim" in cfg
                    ):
                        width = cfg["embed_dim"]
                        # 默认4层结构
                        cfg["hidden_dims"] = [width // 2, width, width, width // 2]

                    # 对于LIIF模型，需要将width映射为hidden_list
                    if "liif" in str(model_name).lower() and "width" in cfg:
                        width = cfg["width"]
                        cfg["hidden_list"] = [width, width, width, width]

                    m = create_enhanced_model(model_name, self.config, **cfg)
                    pc = sum(p.numel() for p in m.parameters())
                    return int(pc)
                except Exception:
                    return 0

            target = target_params_m * 1e6
            name_l = str(model_name).lower()

            if "deeponet" in name_l:
                # DeepONet tuning
                base_latent = int(model_config.get("latent_dim", 128))
                base_hidden = (
                    int(model_config.get("trunk_hidden", [128, 128, 128])[0])
                    if isinstance(model_config.get("trunk_hidden"), list)
                    else 128
                )

                lo, hi = 0.1, 10.0
                for _ in range(15):
                    scale = (lo + hi) / 2
                    latent = max(16, int(base_latent * scale) // 16 * 16)
                    hidden_dim = max(32, int(base_hidden * scale) // 16 * 16)

                    cfg = {
                        **model_config,
                        "latent_dim": latent,
                        "trunk_hidden": [hidden_dim] * 3,
                        "branch_channels": [
                            hidden_dim // 2,
                            hidden_dim,
                            hidden_dim * 2,
                        ],
                    }
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = scale
                    else:
                        hi = scale

            elif "convunetlite" in name_l or "uformerlite" in name_l:
                # Lite models using embed_dim
                base_dim = int(model_config.get("embed_dim", 64))
                lo, hi = 16, 1024
                for _ in range(15):
                    mid = int((lo + hi) // 2) // 8 * 8
                    cfg = {**model_config, "embed_dim": mid}
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + 8
                    else:
                        hi = mid - 8

            elif (
                "unet" in name_l
                and "former" not in name_l
                and "swin" not in name_l
                and "ufno" not in name_l
            ):
                base = [64, 128, 256, 512]
                high = 2.0
                max_feat = 2048
                for _ in range(12):
                    mid = (low + high) * 0.5
                    fs = [
                        max(8, min(max_feat, int(int(f * mid) // 8 * 8))) for f in base
                    ]
                    cfg = {**model_config, "features": fs, "bilinear": True}
                    pc = build(cfg)
                    if pc == 0:
                        # 回退安全特征
                        cfg = {**model_config, "features": base}
                        pc = build(cfg)
                        if pc == 0:
                            continue
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        low = mid
                    else:
                        high = mid
                # 超预算兜底收缩
                try:
                    final_pc = build(best)
                    if final_pc > int(target * 1.2) or final_pc < int(target * 0.8):
                        scale = (target / max(1, final_pc)) ** 0.5
                        fs = [
                            max(16, min(max_feat, int(int(f * scale) // 8 * 8)))
                            for f in best.get("features", base)
                        ]
                        best = {**best, "features": fs}
                except Exception:
                    pass
            elif (
                "swin" in name_l
                or "former" in name_l
                or "vit" in name_l
                or "visiontransformer" in name_l
                or "transformer" in name_l
                or "ufno" in name_l
            ):
                base = int(model_config.get("embed_dim", 96))
                if "ufno" in name_l:
                    base = int(model_config.get("width", 64))
                heads = model_config.get("num_heads", 12)
                if isinstance(heads, (list, tuple)):
                    head_list = [int(max(1, h)) for h in heads]
                else:
                    head_list = [int(max(1, heads))]
                dnh = model_config.get("decoder_num_heads", None)
                if isinstance(dnh, (list, tuple)):
                    dhead_list = [int(max(1, h)) for h in dnh]
                elif isinstance(dnh, int):
                    dhead_list = [int(max(1, dnh))]
                else:
                    dhead_list = []
                import math

                l = 1
                for h in head_list + dhead_list:
                    l = math.lcm(l, h)
                l = max(1, l)

                # Allow scaling down significantly and up
                # Reduced hi to 512 to avoid OOM for large Transformers
                lo, hi = max(l, 16), max(192, 512)

                for _ in range(15):
                    mid = int(((lo + hi) // 2) // l * l)
                    mid = max(l, mid)

                    if "ufno" in name_l:
                        # UFNOUNet tuning: Scale features list
                        # Default features: [64, 128, 256, 512] -> huge (1100M)
                        # Need ~10M. Scale factor ~0.1 -> features ~ [6, 12, 25, 50]
                        if _ == 0:
                            lo, hi = (
                                4,
                                32,
                            )  # Override search range for UFNO features scale
                        mid_scale = max(4, int(((lo + hi) // 2) // 4 * 4))

                        scale_f = mid_scale / 64.0  # normalized scale
                        fs = [
                            max(8, int(f * scale_f) // 4 * 4)
                            for f in [64, 128, 256, 512]
                        ]
                        cfg = {**model_config, "features": fs}

                        # Also reduce FNO complexity
                        cfg["fno_modes1"] = max(4, min(12, int(mid_scale / 4)))
                        cfg["fno_modes2"] = max(4, min(12, int(mid_scale / 4)))
                        cfg["fno_layers"] = 1

                        # Update loop bounds based on mid_scale
                        if _ > 0:  # Skip first iteration check for bounds update
                            mid = mid_scale  # Logic consistency
                    else:
                        cfg = {**model_config, "embed_dim": int(mid)}

                    pc = build(cfg)
                    if pc == 0:
                        pc = 1e9  # treat as too big/fail

                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + l
                    else:
                        hi = mid - l

                # 兜底：确保 Swin 的 window_size 与 patch/grid 对齐
                try:
                    if "swin" in name_l:
                        ps = int(
                            best.get("patch_size", model_config.get("patch_size", 4))
                        )
                        img_sz = int(
                            best.get("img_size", model_config.get("img_size", 128))
                        )
                        win = int(
                            best.get("window_size", model_config.get("window_size", 7))
                        )
                        grid = img_sz // ps
                        if grid % win != 0:
                            from math import gcd

                            safe_win = gcd(grid, win)
                            if safe_win <= 1:
                                candidates = [
                                    w for w in [4, 5, 6, 7, 8, 10, 12] if grid % w == 0
                                ]
                                if candidates:
                                    safe_win = candidates[0]
                                else:
                                    safe_win = max(1, win)
                            best["window_size"] = int(safe_win)
                except Exception:
                    pass

                try:
                    final_pc = build(best)
                    if final_pc > 0 and (
                        final_pc > int(target * 1.2) or final_pc < int(target * 0.8)
                    ):
                        scale = (target / final_pc) ** 0.5
                        if "ufno" in name_l:
                            ed = int(best.get("width", base))
                            ed2 = max(16, min(512, (int(ed * scale) // 8) * 8))
                            best = {**best, "width": ed2}
                        else:
                            ed = int(best.get("embed_dim", base))
                            ed2 = max(l, min(1024, (int(ed * scale) // l) * l))
                            best = {**best, "embed_dim": ed2}
                except Exception:
                    pass
            elif "fno" in name_l and "ufno" not in name_l:
                base_w = int(model_config.get("width", 64))
                lo, hi = max(16, base_w // 4), min(1024, base_w * 4)
                depth = int(model_config.get("n_layers", 4))
                for _ in range(12):
                    mid = int(((lo + hi) // 2) // 8 * 8)
                    cfg = {**model_config, "width": max(16, mid), "n_layers": depth}
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + 8
                    else:
                        hi = mid - 8
                try:
                    final_pc = build(best)
                    if final_pc > int(target * 1.2):
                        scale = (target / final_pc) ** 0.5
                        wd = int(best.get("width", base_w))
                        wd2 = max(16, min(1024, int(wd * scale) // 8 * 8))
                        best = {**best, "width": wd2, "n_layers": depth}
                except Exception:
                    pass
            elif "mlp" in name_l and "mixer" not in name_l:
                base = int(model_config.get("embed_dim", 512))
                lo, hi = max(64, base // 4), max(4096, base * 4)
                for _ in range(15):
                    mid = int(((lo + hi) // 2) // 16 * 16)
                    cfg = {**model_config, "embed_dim": mid}
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + 16
                    else:
                        hi = mid - 16
                try:
                    final_pc = build(best)
                    if final_pc > 0:
                        scale = (target / final_pc) ** 0.5
                        ed = int(best.get("embed_dim", base))
                        ed2 = max(64, int(ed * scale) // 16 * 16)
                        best = {**best, "embed_dim": ed2}
                        width = best["embed_dim"]
                        best["hidden_dims"] = [width // 2, width, width, width // 2]
                except Exception:
                    pass
            elif "mixer" in name_l:
                # MLPMixer tuning
                base = int(model_config.get("embed_dim", 512))
                lo, hi = 64, 4096
                for _ in range(15):
                    mid = int(((lo + hi) // 2) // 16 * 16)
                    cfg = {**model_config, "embed_dim": mid}
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + 16
                    else:
                        hi = mid - 16
                try:
                    final_pc = build(best)
                    if final_pc > 0 and (
                        final_pc > int(target * 1.2) or final_pc < int(target * 0.8)
                    ):
                        scale = (target / final_pc) ** 0.5
                        ed = int(best.get("embed_dim", base))
                        ed2 = max(64, int(ed * scale) // 16 * 16)
                        best = {**best, "embed_dim": ed2}
                except Exception:
                    pass
            elif "hybrid" in name_l:
                # Hybrid model tuning - mainly scale attention and fno width
                base_attn = int(model_config.get("attn_embed_dim", 256))
                base_fno = int(model_config.get("fno_width", 64))
                base_unet = int(model_config.get("unet_base_channels", 64))

                lo, hi = 0.25, 4.0
                for _ in range(15):
                    scale = (lo + hi) / 2
                    cfg = {
                        **model_config,
                        "attn_embed_dim": max(32, int(base_attn * scale) // 8 * 8),
                        "fno_width": max(16, int(base_fno * scale) // 8 * 8),
                        "unet_base_channels": max(16, int(base_unet * scale) // 8 * 8),
                    }
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = scale
                    else:
                        hi = scale
            elif "liif" in name_l:
                lo, hi = 64, 4096
                for _ in range(15):
                    mid = int(((lo + hi) // 2) // 16 * 16)
                    cfg = {**model_config, "width": mid}
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + 16
                    else:
                        hi = mid - 16
                if "width" in best:
                    w = best["width"]
                    best["hidden_list"] = [w, w, w, w]
                    del best["width"]
            elif "lite" in name_l or "restormer" in name_l or "nafnet" in name_l:
                base = int(model_config.get("embed_dim", 48))
                orig_enc_blk = (
                    list(model_config.get("enc_num", [2, 2, 4, 8]))
                    if "enc_num" in model_config
                    else None
                )
                orig_dec_blk = (
                    list(model_config.get("dec_num", [2, 2, 2, 2]))
                    if "dec_num" in model_config
                    else None
                )

                lo, hi = 8, 2048
                for _ in range(15):
                    mid = int(((lo + hi) // 2) // 8 * 8)
                    cfg = {**model_config, "embed_dim": mid}

                    if mid < 24 and orig_enc_blk:
                        cfg["enc_num"] = [1, 1, 1, 1]
                        cfg["dec_num"] = [1, 1, 1, 1]
                    elif mid < 32 and orig_enc_blk:
                        cfg["enc_num"] = [max(1, x // 2) for x in orig_enc_blk]
                        cfg["dec_num"] = [max(1, x // 2) for x in orig_dec_blk]

                    if (
                        "restormer" in name_l
                        and mid < 24
                        and "num_heads" in model_config
                    ):
                        base_heads = list(model_config.get("num_heads", [1, 2, 4, 8]))
                        cfg["num_heads"] = [max(1, h // 2) for h in base_heads]

                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + 8
                    else:
                        hi = mid - 8
                try:
                    final_pc = build(best)
                    if final_pc > 0 and abs(final_pc - target) > target * 0.1:
                        # Fallback generic scale
                        pass
                except Exception:
                    pass
            elif "segformer" in name_l:
                # SegFormer tuning: Scale embed_dims list
                if "embed_dims" not in model_config:
                    model_config["embed_dims"] = [32, 64, 160, 256]
                base = 32
                lo, hi = 4, 128
                for _ in range(15):
                    mid = int(((lo + hi) // 2) // 4 * 4)
                    ratio = mid / base
                    new_dims = [
                        max(16, int(d * ratio) // 8 * 8) for d in [32, 64, 160, 256]
                    ]
                    cfg = {**model_config, "embed_dims": new_dims}
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + 4
                    else:
                        hi = mid - 4
            elif "edsr" in name_l:
                # EDSR tuning: Scale n_feats
                base = int(model_config.get("n_feats", 64))
                lo, hi = 8, 512
                for _ in range(15):
                    mid = int(((lo + hi) // 2) // 4 * 4)
                    cfg = {**model_config, "n_feats": max(16, mid)}
                    pc = build(cfg)
                    err = abs(pc - target)
                    if err < best_err:
                        best, best_err = cfg, err
                    if pc < target:
                        lo = mid + 4
                    else:
                        hi = mid - 4
            else:
                pc = build(model_config)
                if pc == 0:
                    return model_config
                scale = (target / pc) ** 0.5
                if "embed_dim" in model_config:
                    val = int(max(32, int(model_config["embed_dim"] * scale) // 8 * 8))
                    best = {**model_config, "embed_dim": val}
                elif "width" in model_config:
                    val = int(max(16, int(model_config["width"] * scale) // 8 * 8))
                    best = {**model_config, "width": val}
                elif "n_feats" in model_config:
                    val = int(max(16, int(model_config["n_feats"] * scale) // 4 * 4))
                    best = {**model_config, "n_feats": val}
                else:
                    best = model_config
            final_pc = build(best)
            if final_pc > 0:
                pm = final_pc / 1e6
                if abs(pm - target_params_m) <= tolerance_m:
                    return best
            return best
        except Exception:
            return model_config

    def setup_optimizer(self):
        """设置优化器"""
        self.logger.info("⚙️ 设置优化器...")

        # 优化器 - 支持 fused/foreach/eps/amsgrad（若缺失则提供健壮回退）
        try:
            opt_cfg = self.config.training.optimizer
        except Exception:
            from omegaconf import DictConfig

            self.logger.warning("⚠️ 未找到 training.optimizer，使用默认AdamW配置回退")
            if not hasattr(self.config, "training"):
                self.config.training = DictConfig({})
            self.config.training.optimizer = DictConfig(
                {
                    "name": "AdamW",
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "amsgrad": False,
                    "fused": False,
                    "foreach": False,
                }
            )
            opt_cfg = self.config.training.optimizer

        adamw_kwargs = {
            "lr": float(getattr(opt_cfg, "lr", 1e-3)),
            "weight_decay": float(getattr(opt_cfg, "weight_decay", 1e-4)),
            "betas": tuple(getattr(opt_cfg, "betas", (0.9, 0.999))),
            "eps": float(getattr(opt_cfg, "eps", 1e-8)),
            "amsgrad": bool(getattr(opt_cfg, "amsgrad", False)),
        }
        # 确定要优化的模型
        model_to_optimize = None
        if hasattr(self, "model") and self.model is not None:
            model_to_optimize = self.model
        elif hasattr(self, "sequential_model") and self.sequential_model is not None:
            model_to_optimize = self.sequential_model
        else:
            raise RuntimeError(
                "未找到可优化的模型 (既无self.model也无self.sequential_model)"
            )

        # PyTorch 2.0+ 支持 fused/foreach 标志
        fused_flag = bool(getattr(opt_cfg, "fused", False))
        foreach_flag = bool(getattr(opt_cfg, "foreach", False))
        try:
            self.optimizer = torch.optim.AdamW(
                model_to_optimize.parameters(),
                **adamw_kwargs,
                fused=fused_flag,
                foreach=foreach_flag,
            )
            self.logger.info(
                f"✅ 优化器: AdamW (fused={fused_flag}, foreach={foreach_flag}, eps={adamw_kwargs['eps']}, amsgrad={adamw_kwargs['amsgrad']})"
            )
        except TypeError:
            # 回退：不支持fused/foreach的环境
            self.optimizer = torch.optim.AdamW(
                model_to_optimize.parameters(), **adamw_kwargs
            )
            self.logger.info(
                f"✅ 优化器: AdamW (fallback, eps={adamw_kwargs['eps']}, amsgrad={adamw_kwargs['amsgrad']})"
            )

        # 学习率调度器（若缺失则提供回退到 CosineAnnealingLR）
        try:
            sch_cfg = self.config.training.scheduler
        except Exception:
            from omegaconf import DictConfig

            self.logger.warning(
                "⚠️ 未找到 training.scheduler，使用默认CosineAnnealingLR回退"
            )
            if not hasattr(self.config, "training"):
                self.config.training = DictConfig({})
            self.config.training.scheduler = DictConfig(
                {
                    "name": "CosineAnnealingLR",
                    "T_max": int(getattr(self.config.training, "epochs", 1)),
                    "eta_min": 1e-6,
                    "warmup_epochs": 0,
                }
            )
            sch_cfg = self.config.training.scheduler

        try:
            name = str(getattr(sch_cfg, "name", "CosineAnnealingLR"))
            T_max = int(
                getattr(sch_cfg, "T_max", getattr(self.config.training, "epochs", 1))
            )
            eta_min = float(getattr(sch_cfg, "eta_min", 1e-6))
            warmup_epochs = int(getattr(sch_cfg, "warmup_epochs", 0))

            if warmup_epochs > 0:
                base_lr = float(getattr(self.config.training.optimizer, "lr", 1e-3))
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max(1, T_max - warmup_epochs), eta_min=eta_min
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
                self.logger.info(
                    f"✅ 调度器: LinearLR warmup({warmup_epochs}) → CosineAnnealingLR(T_max={max(1, T_max - warmup_epochs)}, eta_min={eta_min})"
                )
            else:
                if name.lower().startswith("cosine"):
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=T_max, eta_min=eta_min
                    )
                    self.logger.info(
                        f"✅ 调度器: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})"
                    )
                else:
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=T_max, eta_min=eta_min
                    )
                    self.logger.info("ℹ️ 未识别调度器名称，已回退到 CosineAnnealingLR")
        except Exception as e:
            self.scheduler = None
            self.logger.warning(f"⚠️ 学习率调度器设置失败，继续训练: {e}")

        # AMP逻辑统一
        amp_cfg = getattr(getattr(self.config, "training", None), "amp", None)
        amp_enabled = (
            bool(getattr(amp_cfg, "enabled", False)) if amp_cfg is not None else False
        )
        autocast_dtype = getattr(self, "autocast_dtype", torch.bfloat16)

        # 梯度缩放器（仅在FP16下启用；BF16无需GradScaler）
        try:
            init_scale = (
                float(getattr(amp_cfg, "init_scale", 2.0**16))
                if amp_cfg is not None
                else (2.0**16)
            )
            growth_factor = (
                float(getattr(amp_cfg, "growth_factor", 2.0))
                if amp_cfg is not None
                else 2.0
            )
            backoff_factor = (
                float(getattr(amp_cfg, "backoff_factor", 0.5))
                if amp_cfg is not None
                else 0.5
            )
            growth_interval = (
                int(getattr(amp_cfg, "growth_interval", 1000))
                if amp_cfg is not None
                else 1000
            )

            use_scaler = (
                amp_enabled
                and (self.device.type == "cuda")
                and (autocast_dtype == torch.float16)
            )

            if use_scaler:
                self.scaler = GradScaler(
                    enabled=True,
                    init_scale=init_scale,
                    growth_factor=growth_factor,
                    backoff_factor=backoff_factor,
                    growth_interval=growth_interval,
                )
            else:
                self.scaler = None
        except Exception:
            from torch.cuda.amp import GradScaler as _LegacyGradScaler  # type: ignore

            use_scaler = (
                amp_enabled
                and (self.device.type == "cuda")
                and (autocast_dtype == torch.float16)
            )
            self.scaler = _LegacyGradScaler() if use_scaler else None

        self.logger.info(
            f"✅ AMP: enabled={amp_enabled}, autocast_dtype={autocast_dtype}, scaler.enabled={use_scaler}"
        )

        self.logger.info(f"✅ 学习率: {opt_cfg.lr}")

    def setup_sequential_model(self):
        """设置分阶段预测架构模型"""
        # 安全获取配置 - 修复OmegaConf配置访问
        try:
            sequential_cfg = self._cfg_select(
                "model.sequential", "sequential", default={}
            )
            # 兼容键：优先 model.sequential.spatial / temporal，其次 fallback 到 model.spatial / model.temporal
            spatial_config = self._cfg_select(
                "model.sequential.spatial",
                "sequential.spatial",
                default=self._cfg_select("model.spatial", "spatial", default={}),
            )
            temporal_config = self._cfg_select(
                "model.sequential.temporal",
                "sequential.temporal",
                default=self._cfg_select("model.temporal", "temporal", default={}),
            )

            self.logger.info(f"空间配置: {spatial_config}")
            self.logger.info(f"时序配置: {temporal_config}")

            # 智能检测：如果空间损失权重为0且启用了时序模型，且空间配置不是Identity，则强制转为Identity
            # 这通常发生在"仅时序预测"的控制变量实验中，此时需要直接使用GT或观测作为时序输入，
            # 而不是使用未训练（随机初始化）的空间模型输出的噪声
            try:
                spatial_loss_w = float(
                    self._cfg_select(
                        "model.sequential.training.spatial_loss_weight",
                        "training.loss_weights.reconstruction",
                        default=1.0,
                    )
                )
                temporal_loss_w = float(
                    self._cfg_select(
                        "model.sequential.training.temporal_loss_weight",
                        "training.loss_weights.r2.weight",
                        default=1.0,
                    )
                )

                # 检查是否为 FNO/UNet 等需要训练的骨干
                bk_type = str(spatial_config.get("backbone_type", "")).lower()
                is_trainable_backbone = bk_type not in ["identity", "none", ""]

                if (
                    is_trainable_backbone
                    and spatial_loss_w <= 1e-6
                    and temporal_loss_w > 0
                ):
                    self.logger.warning(
                        "⚠️ 检测到仅时序训练模式（空间损失≈0），但空间骨干为可训练模型。"
                    )
                    self.logger.warning(
                        f"   原骨干类型: {bk_type} -> 强制转换为: identity"
                    )
                    self.logger.warning(
                        "   这样做是为了确保时序模块接收有效输入（如上一帧），而不是未训练模型的随机噪声。"
                    )

                    # 强制修改配置
                    if isinstance(spatial_config, (dict, list)):  # Dict or ListConfig
                        spatial_config["backbone_type"] = "identity"
                        spatial_config["spatial_feature_dim"] = (
                            0  # 标记为无特征(但实际placeholder输出1通道)
                        )
                    else:  # OmegaConf object
                        spatial_config.backbone_type = "identity"
                        spatial_config.spatial_feature_dim = 0

                    # 同时更新时序配置，使其匹配Identity模式下的输出维度
                    # Identity模式下，SequentialSpatiotemporalModel产生1通道的占位符特征
                    if isinstance(temporal_config, (dict, list)):
                        temporal_config["spatial_feature_dim"] = 1
                    else:
                        temporal_config.spatial_feature_dim = 1

            except Exception as _chk_err:
                self.logger.warning(f"智能模式检测失败: {_chk_err}")

        except Exception as e:
            self.logger.error(f"配置解析失败: {e}")
            raise

        # 初始化分阶段模型
        self.sequential_model = SequentialSpatiotemporalModel(
            spatial_config=spatial_config,
            temporal_config=temporal_config,
            data_config=self.config.data,
            device=self.device,
        ).to(self.device)

        # 可选：冻结空间模块仅训练时序模块（验证纯时序能力）
        try:
            freeze_spatial = bool(
                self._cfg_select(
                    "model.sequential.training.freeze_spatial",
                    "sequential.training.freeze_spatial",
                    default=False,
                )
            )
        except Exception:
            freeze_spatial = False
        if freeze_spatial and hasattr(self.sequential_model, "spatial_module"):
            try:
                for p in self.sequential_model.spatial_module.parameters():
                    p.requires_grad = False
                # 重建优化器以剔除被冻结参数
                self._rebuild_optimizer()
                self.logger.info("✅ 已冻结空间模块参数，仅训练时序模块")
            except Exception as _frz_err:
                self.logger.warning(f"冻结空间模块失败: {_frz_err}")

        # 设置模型精度 - 安全获取AMP配置
        try:
            amp_enabled = bool(
                self._cfg_select(
                    "training.amp.enabled", "model.amp.enabled", default=False
                )
            )
            if amp_enabled:
                # 保持权重为FP32；通过autocast控制计算精度，避免权重类型转换导致数值不稳定
                pass
        except Exception as e:
            self.logger.warning(f"AMP配置解析失败，使用默认设置: {e}")
            amp_enabled = False

        # 分布式训练设置 - 安全获取配置
        try:
            distributed_enabled = bool(
                self._cfg_select(
                    "training.distributed.enabled", "distributed.enabled", default=False
                )
            )
            if distributed_enabled:
                self.sequential_model = nn.parallel.DistributedDataParallel(
                    self.sequential_model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                )
        except Exception as e:
            self.logger.warning(f"分布式配置解析失败，使用单GPU模式: {e}")

        # 初始化一致性检查器 - 安全获取配置
        try:
            consistency_config = self._cfg_select(
                "model.sequential.consistency", "sequential.consistency", default={}
            )
            self.consistency_checker = SequentialConsistencyChecker(
                config=consistency_config
            )
        except Exception as e:
            self.logger.warning(f"一致性检查器配置失败，使用默认配置: {e}")
            self.consistency_checker = SequentialConsistencyChecker(config={})

        # 初始化分阶段训练器
        self.setup_sequential_trainers()

        self.logger.info(f"分阶段模型设置完成: {type(self.sequential_model).__name__}")
        self.logger.info(f"模型结构详情:\n{self.sequential_model}")
        self.logger.info(
            f"模型参数量: {sum(p.numel() for p in self.sequential_model.parameters()):,}"
        )

    def setup_sequential_trainers(self):
        """设置分阶段训练器"""
        # 安全获取配置
        spatial_config = self._cfg_select(
            "model.sequential.spatial", "sequential.spatial", default={}
        )
        temporal_config = self._cfg_select(
            "model.sequential.temporal", "sequential.temporal", default={}
        )
        sequential_config = self._cfg_select(
            "model.sequential", "sequential", default={}
        )

        # 获取核心模型（解包DDP）
        if isinstance(self.sequential_model, torch.nn.parallel.DistributedDataParallel):
            model_core = self.sequential_model.module
        else:
            model_core = self.sequential_model

        # 空间预测训练器（如禁用空间，则跳过创建）
        spatial_disabled = False
        try:
            sf_dim = int(
                spatial_config.get(
                    "spatial_feature_dim", spatial_config.get("feature_dim", 0)
                )
            )
            bk_type = str(spatial_config.get("backbone_type", "")).lower()
            spatial_disabled = (sf_dim == 0) or (bk_type == "identity")
        except Exception:
            pass

        if spatial_disabled:
            self.logger.info("已禁用空间阶段，跳过 SpatialTrainer 创建")
            self.spatial_trainer = None
        else:
            self.spatial_trainer = SpatialTrainer(
                model=model_core.spatial_module, config=spatial_config
            )

        # 时序预测训练器
        self.temporal_trainer = TemporalTrainer(
            model=model_core.temporal_module, config=temporal_config
        )

        # 联合训练器 - 使用已有的模型实例
        self.sequential_trainer = SequentialSpatiotemporalTrainer(
            config=sequential_config
        )
        # 覆盖模型为已创建的实例（保留DDP包装，用于联合训练）
        self.sequential_trainer.model = self.sequential_model

    def setup_monitoring(self):
        """设置监控"""
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        # 初始化当前epoch与训练历史结构
        self.current_epoch = 0
        self.training_history = {
            "train_losses": [],
            "val_losses": [],
            "val_metrics": [],
            "learning_rates": [],
            "epochs": [],
            "curriculum_stages": [],
        }
        # 初始化性能计数器
        self._perf_samples = 0
        self._perf_fetch_time = 0.0
        self._perf_data_time = 0.0
        self._perf_compute_time = 0.0
        # 初始化TensorBoard writer（若尚未创建），避免重复事件文件
        try:
            if not hasattr(self, "writer") or self.writer is None:
                self.writer = SummaryWriter(log_dir=str(self.output_dir))
                self.logger.info("TensorBoard 监控已启用")
        except Exception as _tb_err:
            self.logger.warning(f"TensorBoard初始化失败，继续训练: {_tb_err}")

    def run_quick_benchmark(
        self, num_batches: int = 50, outfile: str = "benchmark.json"
    ):
        """轻量级基准测试：评估数据加载和前向吞吐，写入指定文件
        按统一接口调用模型 forward(x)->y，并对ARWrapper使用 (input_seq, T_out, target_seq)。
        """
        self.logger.info(f"⚡ 运行轻量级基准测试，采样批次={num_batches}")
        results = {
            "num_batches": int(num_batches),
            "data_fetch_time_sec": 0.0,
            "forward_time_sec": 0.0,
            "samples": 0,
            "throughput_samples_per_sec": 0.0,
        }
        fetch_t, fwd_t, samples = 0.0, 0.0, 0
        if not hasattr(self, "train_loader") or self.train_loader is None:
            self.logger.info("训练前基准：train_loader 不存在，跳过")
            return
        import itertools

        it = itertools.islice(iter(self.train_loader), num_batches)
        current_T_out = 1
        try:
            current_T_out = self.get_current_T_out(
                self.current_epoch if hasattr(self, "current_epoch") else 0
            )
        except Exception:
            pass
        for batch in it:
            t0 = time.time()
            # 统计取数时间（batch已在上面取到，此处仅统计处理开销）
            t1 = time.time()
            fetch_t += t1 - t0
            # 前向测试（不进行反向与优化）
            try:
                with torch.no_grad():
                    if (
                        isinstance(batch, dict)
                        and "input_sequence" in batch
                        and "target_sequence" in batch
                    ):
                        x = batch["input_sequence"].to(self.device, non_blocking=True)
                        tgt = batch["target_sequence"].to(
                            self.device, non_blocking=True
                        )
                    else:
                        continue
                    t2 = time.time()
                    # 统一调用：支持SequentialSpatiotemporalModel和传统模型
                    model = self.get_model()
                    try:
                        if hasattr(model, "forward") and hasattr(
                            model, "spatial_forward"
                        ):
                            # SequentialSpatiotemporalModel模式 - 需要完整的时序输入
                            _ = model(x, tgt)
                        else:
                            # 传统模型模式：ARWrapper需要 (x, T_out, tgt)
                            _ = model(x, current_T_out, tgt)
                    except TypeError:
                        # 退化为通用接口 forward(x)
                        _ = model(x)
                    t3 = time.time()
                    fwd_t += t3 - t2
                    samples += int(x.shape[0])
            except Exception as _bm_fwd_err:
                self.logger.debug(f"基准前向失败，跳过该批次: {_bm_fwd_err}")
        total_t = fetch_t + fwd_t
        results["data_fetch_time_sec"] = fetch_t
        results["forward_time_sec"] = fwd_t
        results["samples"] = samples
        results["throughput_samples_per_sec"] = (
            (float(samples) / total_t) if total_t > 0 else 0.0
        )
        try:
            with open(self.output_dir / outfile, "w") as f:
                json.dump(results, f, indent=2)
            self.logger.info(
                f"✅ 轻量基准完成：吞吐={results['throughput_samples_per_sec']:.2f} samples/s"
            )
        except Exception as _b_err:
            self.logger.debug(f"写入benchmark失败: {_b_err}")
        self.training_history = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
            "epochs": [],
            "curriculum_stages": [],
            "val_metrics": [],
        }

        # 课程学习状态
        self.current_stage = 0
        self.stage_epoch = 0

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0

        # 性能窗口累计（耗时分解/CPU/IO监控）
        self.perf_last_report_time = time.time()
        self._perf_fetch_time = 0.0  # DataLoader取数耗时
        self._perf_data_time = 0.0  # 设备搬运耗时（host→device）
        self._perf_compute_time = 0.0  # 计算耗时（前向+损失+反向+优化）
        self._perf_batches = 0
        self._perf_samples = 0
        try:
            self._process = psutil.Process(os.getpid())
            # 初始化一次CPU使用率（第一次调用返回0.0）
            _ = self._process.cpu_percent(interval=None)
            # 配置CPU亲和性：hardware.cpu.cpu_affinity 或 thread_pool_size/num_workers 映射
            try:
                aff_cfg = getattr(self.config, "hardware", None)
                cpu_cfg = getattr(aff_cfg, "cpu", None) if aff_cfg is not None else None
                affinity = None
                if cpu_cfg is not None and hasattr(cpu_cfg, "cpu_affinity"):
                    affinity = cpu_cfg.cpu_affinity
                # 如果未显式配置亲和性，且存在num_workers或thread_pool_size，使用一个合理映射（不强制）
                if affinity is None:
                    tp_size = int(
                        self._cfg_select(
                            "hardware.cpu.thread_pool_size",
                            "data.dataloader.num_workers",
                            "hardware.num_workers",
                            default=0,
                        )
                        or 0
                    )
                    if tp_size > 0:
                        # 将主进程绑定到前tp_size个逻辑CPU，避免过度迁移
                        affinity = list(
                            range(
                                min(tp_size, psutil.cpu_count(logical=True) or tp_size)
                            )
                        )
                if affinity is not None:
                    # 亲和性可以是列表或区间描述，统一为列表
                    if isinstance(affinity, (list, tuple)):
                        cpu_list = [
                            int(x) for x in affinity if isinstance(x, (int, float))
                        ]
                    elif isinstance(affinity, dict) and "range" in affinity:
                        start = int(affinity["range"][0])
                        end = int(affinity["range"][1])
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
            self.logger.warning(
                "Visualization modules not available; disabling visualizations."
            )

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False

        try:
            # 修复PyTorch 2.6兼容性问题 - 添加安全全局列表
            import torch.serialization
            from omegaconf.dictconfig import DictConfig as OmegaDictConfig
            from omegaconf.listconfig import ListConfig

            # 添加OmegaConf类到安全全局列表
            safe_globals = [ListConfig, OmegaDictConfig]

            # 尝试使用安全全局列表加载
            try:
                with torch.serialization.safe_globals(safe_globals):
                    checkpoint = torch.load(
                        checkpoint_path, map_location=self.device, weights_only=True
                    )
            except Exception as safe_load_error:
                # 如果安全加载失败，回退到weights_only=False
                self.logger.warning(
                    f"安全加载失败，回退到非安全模式: {safe_load_error}"
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=False
                )

            # 加载模型状态 - 处理结构不匹配
            model_to_load = self.get_model()
            if hasattr(model_to_load, "module"):
                model_to_load = model_to_load.module

            # 预处理 state_dict：移除 'module.' 前缀（处理 DDP 保存的检查点）
            state_dict = checkpoint["model_state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            try:
                model_to_load.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as e:
                self.logger.warning(f"严格模式加载失败: {e}")
                self.logger.info("尝试非严格模式加载...")
                model_state = model_to_load.state_dict()
                checkpoint_state = new_state_dict
                filtered_state = {}
                for key, value in checkpoint_state.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            filtered_state[key] = value
                        else:
                            self.logger.warning(
                                f"跳过形状不匹配的参数: {key} (模型: {model_state[key].shape} vs 检查点: {value.shape})"
                            )
                    else:
                        self.logger.warning(f"跳过不存在的参数: {key}")
                missing_keys = set(model_state.keys()) - set(filtered_state.keys())
                if missing_keys:
                    self.logger.warning(f"以下参数将使用随机初始化: {missing_keys}")
                model_to_load.load_state_dict(filtered_state, strict=False)
                self.logger.info(
                    f"✅ 非严格模式加载成功，加载了 {len(filtered_state)}/{len(checkpoint_state)} 个参数"
                )
            if (
                "optimizer_state_dict" in checkpoint
                and hasattr(self, "optimizer")
                and self.optimizer is not None
            ):
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    pass
            if (
                "scheduler_state_dict" in checkpoint
                and hasattr(self, "scheduler")
                and self.scheduler is not None
            ):
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception:
                    pass

            if (
                "scaler_state_dict" in checkpoint
                and getattr(self, "scaler", None) is not None
            ):
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            # 加载训练状态
            self.current_epoch = checkpoint.get("epoch", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

            # 安全加载 training_history
            if "training_history" in checkpoint:
                self.training_history = checkpoint["training_history"]
            elif not hasattr(self, "training_history"):
                self.training_history = {
                    "train_losses": [],
                    "val_losses": [],
                    "val_metrics": [],
                    "learning_rates": [],
                    "epochs": [],
                    "curriculum_stages": [],
                }

            self.logger.info(f"✅ Successfully loaded checkpoint: {checkpoint_path}")
            self.logger.info(
                f"Restored to epoch {self.current_epoch}, best val loss: {self.best_val_loss:.6f}"
            )
            return True

        except Exception as e:
            self.logger.warning(
                f"Failed to load checkpoint, fallback to current model: {str(e)}"
            )
            return False

    def create_visualizations(self, sample_batch: dict | None = None, epoch: int = 0):
        """创建可视化（统一ARVisualizer用法）"""
        try:
            out_dir_runs = self.output_dir / "visualizations"
            out_dir_runs.mkdir(parents=True, exist_ok=True)
            out_dir_pkg = Path("paper_package/figs") / self.output_dir.name
            out_dir_pkg.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        # 1. 尝试使用self.visualizer（如果已初始化）
        if self.visualizer is not None:
            try:
                if sample_batch is not None:
                    pass
                try:
                    self.visualizer.save_training_curves(self.training_history)
                except Exception:
                    pass
                self.logger.info(
                    f"Saved visualization samples and training curves for epoch {epoch}"
                )
            except Exception:
                pass
        # 2. 如果self.visualizer不可用，或者需要强制使用ARVisualizer进行标准可视化
        # 我们在这里实例化ARVisualizer来生成标准图
        if ARVisualizer is not None:
            try:
                viz = ARVisualizer(str(out_dir_runs))
                # 绘制训练曲线
                viz.save_training_curves(
                    self.training_history, str(out_dir_runs / "training_curves.svg")
                )
                viz.save_training_curves(
                    self.training_history, str(out_dir_pkg / "training_curves.svg")
                )

                # 如果有样本batch，绘制Obs/GT/Pred/Err
                if sample_batch is not None:
                    input_seq = sample_batch.get("input_sequence")
                    target_seq = sample_batch.get("target_sequence")
                    if isinstance(input_seq, torch.Tensor) and isinstance(
                        target_seq, torch.Tensor
                    ):
                        device = self.device
                        input_seq = input_seq.to(device)
                        target_seq = target_seq.to(device)

                        # 获取预测结果
                        self.get_model().eval()
                        with torch.no_grad():
                            x_in = input_seq[:, 0]
                            if x_in.dim() == 4 and x_in.shape[1] > 1:
                                x_in = x_in[:, 0:1]  # 只取第一个通道作为图像输入
                            y = self.get_model()(x_in)

                        # 准备数据 [B,C,H,W]
                        # 反归一化
                        norm_stats = getattr(self, "norm_stats", None)
                        mean_val = 0.0
                        std_val = 1.0
                        if (
                            norm_stats is not None
                            and "mean" in norm_stats
                            and "std" in norm_stats
                        ):
                            mean = norm_stats["mean"]
                            std = norm_stats["std"]
                            if isinstance(mean, torch.Tensor):
                                mean_val = float(mean[0]) if mean.numel() > 0 else 0.0
                            else:
                                mean_val = float(mean) if np.isscalar(mean) else 0.0
                            if isinstance(std, torch.Tensor):
                                std_val = float(std[0]) if std.numel() > 0 else 1.0
                            else:
                                std_val = float(std) if np.isscalar(std) else 1.0

                        # 反归一化处理
                        obs_denorm = x_in * std_val + mean_val
                        gt_denorm = target_seq[:, 0, 0:1] * std_val + mean_val
                        pred_denorm = y * std_val + mean_val

                        # 保存可视化
                        viz_path = viz.plot_obs_gt_pred_err_horizontal(
                            obs_denorm,
                            gt_denorm,
                            pred_denorm,
                            save_path=str(
                                out_dir_runs / f"epoch_{epoch:04d}_sample.png"
                            ),
                            num_samples=min(4, input_seq.size(0)),
                            # 传递 crop_params
                            crop_params=getattr(
                                self.cfg.data.observation, "crop", None
                            ),
                        )
                        # 复制到paper_package
                        import shutil

                        shutil.copy2(viz_path, out_dir_pkg / Path(viz_path).name)

                        self.logger.info(
                            f"Saved standard visualizations via ARVisualizer for epoch {epoch}"
                        )
                        return  # 成功使用ARVisualizer后返回
            except Exception as e:
                self.logger.warning(f"ARVisualizer failed: {e}")

        # 3. Fallback: 原有的matplotlib可视化逻辑（如果ARVisualizer失败）
        if sample_batch is None:
            return
        try:
            device = self.device
            input_seq = sample_batch.get("input_sequence")
            target_seq = sample_batch.get("target_sequence")
            if not isinstance(input_seq, torch.Tensor) or not isinstance(
                target_seq, torch.Tensor
            ):
                return
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            x = input_seq[:, 0]
            if x.dim() == 4 and x.shape[1] > 1:
                x = x[:, 0:1]
            x_in = x
            self.get_model().eval()
            with torch.no_grad():
                y = self.get_model()(x_in)
            gt = target_seq[:, 0, 0:1]
            err = (y - gt).abs()
            import matplotlib.pyplot as plt
            import numpy as np

            # 获取归一化统计信息进行反归一化
            norm_stats = getattr(self, "norm_stats", None)
            if norm_stats is not None and "mean" in norm_stats and "std" in norm_stats:
                mean = norm_stats["mean"]
                std = norm_stats["std"]
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

            n = min(input_seq.size(0), 8)
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
                gs = fig.add_gridspec(
                    1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.05
                )

                # Obs - 使用统一物理量颜色范围
                ax0 = fig.add_subplot(gs[0])
                im0 = ax0.imshow(
                    obs_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys
                )
                ax0.set_title("Obs", fontsize=11, fontweight="bold")

                # GT - 使用统一物理量颜色范围
                ax1 = fig.add_subplot(gs[1])
                im1 = ax1.imshow(gt_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys)
                ax1.set_title("GT", fontsize=11, fontweight="bold")

                # Pred - 使用统一物理量颜色范围
                ax2 = fig.add_subplot(gs[2])
                im2 = ax2.imshow(pr_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys)
                ax2.set_title("Pred", fontsize=11, fontweight="bold")

                # Error - 使用对称的误差颜色范围
                ax3 = fig.add_subplot(gs[3])
                im3 = ax3.imshow(er_img, cmap="coolwarm", vmin=vmin_err, vmax=vmax_err)
                ax3.set_title("Error", fontsize=11, fontweight="bold")

                # 移除坐标轴刻度，保持简洁
                for ax in [ax0, ax1, ax2, ax3]:
                    ax.set_xticks([])
                    ax.set_yticks([])

                # 添加统一的颜色条 - 物理量（在最右侧）
                cbar_phys = fig.colorbar(
                    im0, cax=fig.add_subplot(gs[4]), orientation="vertical"
                )
                cbar_phys.set_label("Physical Value", fontsize=10, fontweight="bold")
                cbar_phys.ax.tick_params(labelsize=9)

                fig.suptitle(
                    f"Epoch {epoch} - Sample {b}",
                    fontsize=13,
                    fontweight="bold",
                    y=0.95,
                )
                fig.tight_layout()

                # 保存图像
                p_run_png = out_dir_runs / f"epoch_{epoch:04d}_sample_{b:03d}.png"
                p_pkg_png = out_dir_pkg / f"epoch_{epoch:04d}_sample_{b:03d}.png"
                p_run_svg = out_dir_runs / f"epoch_{epoch:04d}_sample_{b:03d}.svg"
                p_pkg_svg = out_dir_pkg / f"epoch_{epoch:04d}_sample_{b:03d}.svg"

                plt.savefig(p_run_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
                plt.savefig(p_pkg_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
                plt.savefig(p_run_svg, bbox_inches="tight", pad_inches=0.1)
                plt.savefig(p_pkg_svg, bbox_inches="tight", pad_inches=0.1)
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

    def create_test_visualizations(self, final_test_metrics: dict | None = None):
        """Generate test-phase visualizations and export to paper_package/figs."""
        if getattr(self, "_test_viz_done", False):
            self.logger.info(
                "⚪ Test visualizations already generated; skipping duplicate run"
            )
            return
        try:
            out_dir = self.output_dir / "visualizations"
            out_dir.mkdir(parents=True, exist_ok=True)
            paper_dir = Path("paper_package/figs") / self.output_dir.name
            paper_dir.mkdir(parents=True, exist_ok=True)

            # 保存测试指标摘要
            if isinstance(final_test_metrics, dict):
                with open(out_dir / "final_test_metrics.json", "w") as f:
                    json.dump(convert_numpy_types(final_test_metrics), f, indent=2)

            # 若可视化器可用，导出若干测试样本图像
            samples_saved = 0
            try:
                if hasattr(self, "visualizer") and self.visualizer is not None:
                    samples_saved = self.visualizer.save_test_samples(
                        self.test_loader, out_dir, max_batches=3
                    )
            except Exception:
                pass

            # 生成简易索引页面
            idx_path = out_dir / "index.html"
            with open(idx_path, "w") as f:
                f.write(
                    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Test Visualizations</title></head><body>"
                )
                f.write("<h1>Test Visualizations</h1>")
                f.write(f"<p>Samples saved: {samples_saved}</p>")
                f.write("<ul>")
                for p in sorted(out_dir.glob("**/*.png")):
                    rel = p.relative_to(out_dir)
                    f.write(
                        f"<li><img src='{rel.as_posix()}' style='max-width:512px' /></li>"
                    )
                f.write("</ul></body></html>")

            # 拷贝到论文包目录
            try:
                import shutil

                for p in out_dir.glob("*"):
                    shutil.copy2(p, paper_dir / p.name)
            except Exception:
                pass

            self.logger.info(
                "🖼️ Test-phase visualizations exported to paper_package/figs"
            )
            self._test_viz_done = True
        except Exception as _tviz_err:
            self.logger.warning(f"create_test_visualizations failed: {_tviz_err}")

    def get_current_T_out(self, epoch: int) -> int:
        """根据课程学习配置返回当前T_out，并维护当前阶段索引"""
        try:
            cur_cfg = getattr(self.config.training, "curriculum", None)
            if not (cur_cfg and bool(getattr(cur_cfg, "enabled", False))):
                return int(getattr(self.config.data, "T_out", 1))
            stages = list(getattr(cur_cfg, "stages", []))
            if not stages:
                return int(getattr(self.config.data, "T_out", 1))
            # 累积epoch定位阶段
            total = 0
            for idx, st in enumerate(stages):
                e = int(st.get("epochs", 0))
                total += e
                if epoch < total:
                    self.current_stage = idx
                    return int(st.get("T_out", getattr(self.config.data, "T_out", 1)))
            # 超出课程阶段范围，返回最后一个阶段的T_out
            self.current_stage = len(stages) - 1
            return int(stages[-1].get("T_out", getattr(self.config.data, "T_out", 1)))
        except Exception:
            return int(getattr(self.config.data, "T_out", 1))

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
            ck_cfg = getattr(self.config.training, "checkpoint", None)
            save_last = True
            save_best = True
            save_every_n = 0
            max_keep = 2
            try:
                if ck_cfg is not None:
                    save_last = bool(getattr(ck_cfg, "save_last", True))
                    save_best = bool(getattr(ck_cfg, "save_best", True))
                    save_every_n = int(getattr(ck_cfg, "save_every_n_epochs", 0) or 0)
                    max_keep = int(getattr(ck_cfg, "max_keep", 2) or 2)
            except Exception:
                pass

            # 组装状态
            state = {
                "epoch": int(epoch),
                "model_state_dict": self.get_model().state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                    if hasattr(self, "optimizer") and self.optimizer is not None
                    else {}
                ),
                "scheduler_state_dict": (
                    self.scheduler.state_dict()
                    if hasattr(self, "scheduler") and self.scheduler is not None
                    else {}
                ),
                "scaler_state_dict": (
                    self.scaler.state_dict()
                    if hasattr(self, "scaler") and self.scaler is not None
                    else None
                ),
                "best_val_loss": float(getattr(self, "best_val_loss", float("inf"))),
                "training_history": self.training_history,
                "config": OmegaConf.to_container(self.config, resolve=True),
                "timestamp": time.time(),
            }

            # 始终保存last
            if save_last:
                last_path = self.output_dir / "last.ckpt"
                try:
                    torch.save(state, last_path)
                    self.logger.info(f"💾 已保存最后检查点: {last_path}")
                except Exception as _sl_err:
                    self.logger.warning(f"保存last.ckpt失败: {_sl_err}")

            # 保存最佳
            if save_best and is_best:
                best_path = self.output_dir / "best.ckpt"
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
            "epochs": int(len(self.training_history.get("epochs", []))),
            "avg_throughput_samples_per_sec": 0.0,
            "avg_epoch_time_sec": 0.0,
            "max_gpu_peak_allocated_gb": 0.0,
            "max_gpu_peak_reserved_gb": 0.0,
            "avg_cpu_percent": 0.0,
            "avg_system_memory_percent": 0.0,
            "avg_iowait_percent": 0.0,
        }
        # 读取每epoch资源
        epoch_records = []
        try:
            ep_file = self.output_dir / "resources_epoch.jsonl"
            if ep_file.exists():
                with open(ep_file) as f:
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

            summary["avg_throughput_samples_per_sec"] = _avg(
                "throughput_samples_per_sec"
            )
            summary["avg_epoch_time_sec"] = _avg("time_sec")
            summary["max_gpu_peak_allocated_gb"] = _max("gpu_peak_allocated_gb")
            summary["max_gpu_peak_reserved_gb"] = _max("gpu_peak_reserved_gb")
            summary["avg_cpu_percent"] = _avg("cpu_percent")
            summary["avg_system_memory_percent"] = _avg("system_memory_percent")
            summary["avg_iowait_percent"] = _avg("iowait_percent")
        # Write summary
        try:
            out_path = self.output_dir / "resource_summary.json"
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"📊 Resource summary saved: {out_path}")
        except Exception as _sum_err:
            self.logger.debug(f"Failed to write resource summary: {_sum_err}")
        # 配置开关：可视化总开关
        try:
            viz_enabled = bool(self._cfg_select("visualization.enabled", default=True))
        except Exception:
            viz_enabled = True

        if not viz_enabled:
            self.logger.info("⚪ Visualization disabled by config, skipping generation")
            return

        if not VISUALIZATION_AVAILABLE:
            self.logger.warning(
                "Visualization module unavailable, skipping visualization generation"
            )
            return

        try:
            # 创建可视化目录
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # 使用AR专用可视化器
            import os as _os

            from utils.ar_visualizer import ARTrainingVisualizer

            max_cols_cfg = int(
                self._cfg_select(
                    "visualization.max_time_cols",
                    "logging.visualization.max_time_cols",
                    default=6,
                )
                or 6
            )
            try:
                _os.environ["VIZ_MAX_TIME_COLS"] = str(max_cols_cfg)
            except Exception:
                pass
            ar_visualizer = ARTrainingVisualizer(str(viz_dir))

            # 可视化训练曲线
            if hasattr(self, "training_history") and self.training_history:
                ar_visualizer.plot_training_curves(
                    self.training_history, f"training_curves_epoch_{epoch}"
                )

            # 如果有样本数据，创建AR预测可视化
            if sample_batch is not None:
                input_seq = sample_batch["input_sequence"]
                target_seq = sample_batch["target_sequence"]
                pred_seq = sample_batch.get("predictions", None)

                if pred_seq is None:
                    # 兜底：若未提供预测，则进行一次前向以生成
                    self.get_model().eval()
                    with torch.no_grad():
                        current_T_out = self.get_current_T_out(epoch)
                        pred_seq = self.get_model()(
                            input_seq.to(self.device), current_T_out
                        ).cpu()
                    self.get_model().train()

                # 创建AR预测可视化（首行显示 H(GT) 观测近似），统一色标
                # 传递一致的 H 参数用于构造观测帧
                h_params = self.h_params
                ar_visualizer.visualize_ar_predictions(
                    input_seq,
                    target_seq,
                    pred_seq,
                    timestep_idx=epoch,
                    save_name=f"ar_predictions_epoch_{epoch}",
                    norm_stats=self.norm_stats,
                    h_params=h_params,
                )

                # 创建误差分析
                # 确保norm_stats存在
                self.ensure_norm_stats()
                ar_visualizer.create_error_analysis(
                    target_seq,
                    pred_seq,
                    save_name=f"error_analysis_epoch_{epoch}",
                    norm_stats=self.norm_stats,
                )

                # 创建时间分析
                # 确保norm_stats存在
                self.ensure_norm_stats()
                ar_visualizer.create_temporal_analysis(
                    pred_seq,
                    target_seq,
                    save_name=f"temporal_analysis_epoch_{epoch}",
                    norm_stats=self.norm_stats,
                )

            self.logger.info(f"✅ Visualizations saved to {viz_dir}")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate visualizations: {e}")
            import traceback

            traceback.print_exc()

    def get_current_T_out(self, epoch: int) -> int:
        """获取当前阶段的T_out（健壮处理缺失配置）"""
        # 安全读取课程学习开关
        try:
            curriculum_enabled = bool(
                self._cfg_select("training.curriculum.enabled", default=False)
            )
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
            stage_epochs = (
                stage.get("epochs", 0)
                if isinstance(stage, dict)
                else getattr(stage, "epochs", 0)
            )
            cumulative_epochs += stage_epochs
            if epoch < cumulative_epochs:
                if i != self.current_stage:
                    self.current_stage = i
                    self.stage_epoch = 0
                    desc = (
                        stage.get("description", f"阶段{i}")
                        if isinstance(stage, dict)
                        else getattr(stage, "description", f"阶段{i}")
                    )
                    self.logger.info(f"🎯 进入{desc}")
                stage_T_out = (
                    stage.get("T_out", None)
                    if isinstance(stage, dict)
                    else getattr(stage, "T_out", None)
                )
                if stage_T_out is None:
                    try:
                        stage_T_out = int(self._cfg_select("data.T_out", default=1))
                    except Exception:
                        stage_T_out = 1
                return int(stage_T_out)

        # 若超出所有阶段，使用最后一个阶段的T_out，若缺失则回退
        last = stages[-1]
        last_T_out = (
            last.get("T_out", None)
            if isinstance(last, dict)
            else getattr(last, "T_out", None)
        )
        if last_T_out is None:
            try:
                last_T_out = int(self._cfg_select("data.T_out", default=1))
            except Exception:
                last_T_out = 1
        return int(last_T_out)

    def ensure_norm_stats(self):
        """确保norm_stats已初始化，避免AttributeError"""
        if not hasattr(self, "norm_stats") or self.norm_stats is None:
            # 优先从训练数据计算真实z-score统计
            try:
                fn = getattr(self, "_compute_norm_stats_from_data", None)
            except Exception:
                fn = None
            if fn is None:
                # 内联实现：从train_loader估计均值/方差
                def _compute_norm_stats_from_data(_self) -> bool:
                    try:
                        import torch

                        dl = getattr(_self, "train_loader", None)
                        ds = getattr(_self, "train_dataset", None)
                        if dl is None and ds is not None:
                            from torch.utils.data import DataLoader

                            dl = DataLoader(
                                ds, batch_size=1, shuffle=False, num_workers=0
                            )
                        if dl is None:
                            return False
                        sums = None
                        sumsq = None
                        count = 0
                        max_batches = 16
                        b = 0
                        for batch in dl:
                            if b >= max_batches:
                                break
                            b += 1
                            x = None
                            if isinstance(batch, dict):
                                for k in [
                                    "input_sequence",
                                    "target_sequence",
                                    "x",
                                    "y",
                                ]:
                                    v = batch.get(k, None)
                                    if torch.is_tensor(v) and v.dim() >= 4:
                                        x = v
                                        break
                            if x is None:
                                continue
                            x = x.to("cpu")
                            B = x.shape[0]
                            C = x.shape[-3]
                            X = x.reshape(B, -1)
                            s = X.sum(dim=1).sum()
                            ss = (X**2).sum(dim=1).sum()
                            cnt = X.numel()
                            sums = (sums + s) if sums is not None else s
                            sumsq = (sumsq + ss) if sumsq is not None else ss
                            count += cnt
                        if count <= 0:
                            return False
                        mean = (sums / count).unsqueeze(0)
                        var = (sumsq / count) - (mean.squeeze() ** 2)
                        std = torch.sqrt(torch.clamp(var, min=1e-8)).unsqueeze(0)
                        _self.norm_stats = {
                            "mean": mean,
                            "std": std,
                            "u_mean": (
                                mean[0] if mean.numel() >= 1 else torch.tensor(0.0)
                            ),
                            "u_std": std[0] if std.numel() >= 1 else torch.tensor(1.0),
                        }
                        return True
                    except Exception:
                        return False

                self._compute_norm_stats_from_data = _compute_norm_stats_from_data
                fn = _compute_norm_stats_from_data
            if not fn(self):
                import logging

                logger = logging.getLogger(__name__)
                logger.warning("⚠️ norm_stats未初始化，使用默认值")
                try:
                    C = int(
                        self._cfg_select(
                            "model.out_channels", "data.target_channels", default=1
                        )
                    )
                except Exception:
                    C = 1
                self.norm_stats = {"mean": torch.zeros(C), "std": torch.ones(C)}
                # 兼容旧键名
                self.norm_stats["u_mean"] = (
                    self.norm_stats["mean"][0] if C >= 1 else torch.tensor(0.0)
                )
                self.norm_stats["u_std"] = (
                    self.norm_stats["std"][0] if C >= 1 else torch.tensor(1.0)
                )
                if C >= 2:
                    self.norm_stats["v_mean"] = self.norm_stats["mean"][1]
                    self.norm_stats["v_std"] = self.norm_stats["std"][1]

    def _is_spatial_phase(self, epoch: int) -> bool:
        try:
            two_stage = bool(
                self._cfg_select(
                    "model.sequential.training.two_stage",
                    "sequential.training.two_stage",
                    default=False,
                )
            )
            if not two_stage:
                return False
            s1 = int(
                self._cfg_select(
                    "model.sequential.training.stage1_epochs",
                    "sequential.training.stage1_epochs",
                    default=0,
                )
            )
            return epoch < s1
        except Exception:
            return False

    def _freeze_temporal(self):
        try:
            if self.sequential_model is not None and hasattr(
                self.sequential_model, "temporal_module"
            ):
                for p in self.sequential_model.temporal_module.parameters():
                    p.requires_grad = False
        except Exception:
            pass

    def _unfreeze_temporal(self):
        try:
            if self.sequential_model is not None and hasattr(
                self.sequential_model, "temporal_module"
            ):
                for p in self.sequential_model.temporal_module.parameters():
                    p.requires_grad = True
        except Exception:
            pass

    def _rebuild_optimizer(self):
        try:
            opt_cfg = getattr(self.config.training, "optimizer", None)
            if opt_cfg is None:
                return
            model_to_optimize = self.get_model()
            params = [p for p in model_to_optimize.parameters() if p.requires_grad]
            adamw_kwargs = {
                "lr": float(getattr(opt_cfg, "lr", 1e-3)),
                "weight_decay": float(getattr(opt_cfg, "weight_decay", 1e-4)),
                "betas": tuple(getattr(opt_cfg, "betas", (0.9, 0.999))),
                "eps": float(getattr(opt_cfg, "eps", 1e-8)),
                "amsgrad": bool(getattr(opt_cfg, "amsgrad", False)),
            }
            fused_flag = bool(getattr(opt_cfg, "fused", False))
            foreach_flag = bool(getattr(opt_cfg, "foreach", False))
            try:
                self.optimizer = torch.optim.AdamW(
                    params, **adamw_kwargs, fused=fused_flag, foreach=foreach_flag
                )
            except TypeError:
                self.optimizer = torch.optim.AdamW(params, **adamw_kwargs)
            sch_cfg = getattr(self.config.training, "scheduler", None)
            if sch_cfg is not None:
                name = str(getattr(sch_cfg, "name", "CosineAnnealingLR"))
                T_max = int(
                    getattr(
                        sch_cfg, "T_max", getattr(self.config.training, "epochs", 1)
                    )
                )
                eta_min = float(getattr(sch_cfg, "eta_min", 1e-6))
                warmup_epochs = int(getattr(sch_cfg, "warmup_epochs", 0))
                if warmup_epochs > 0:
                    warmup = torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    )
                    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=max(1, T_max - warmup_epochs),
                        eta_min=eta_min,
                    )
                    self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                        self.optimizer,
                        schedulers=[warmup, cosine],
                        milestones=[warmup_epochs],
                    )
                else:
                    if name.lower().startswith("cosine"):
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, T_max=T_max, eta_min=eta_min
                        )
                    else:
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, T_max=T_max, eta_min=eta_min
                        )
        except Exception:
            pass

    def _update_phase(self, epoch: int):
        try:
            is_spatial = self._is_spatial_phase(epoch)
            target_phase = "spatial" if is_spatial else "joint"
            if self._current_training_phase != target_phase:
                if target_phase == "spatial":
                    self._freeze_temporal()
                else:
                    self._unfreeze_temporal()
                self._rebuild_optimizer()
                self._current_training_phase = target_phase
        except Exception:
            pass

    def get_model(self):
        """获取当前使用的模型（兼容ARWrapper和SequentialSpatiotemporalModel）"""
        if hasattr(self, "model") and self.model is not None:
            return self.model
        if hasattr(self, "sequential_model") and self.sequential_model is not None:
            return self.sequential_model
        # 兜底：尝试初始化模型
        try:
            self.setup_model()
            if hasattr(self, "model") and self.model is not None:
                return self.model
            if hasattr(self, "sequential_model") and self.sequential_model is not None:
                return self.sequential_model
        except Exception:
            pass
        raise RuntimeError("未找到可用的模型 (既无self.model也无self.sequential_model)")

    def handle_cuda_error(self, e, context="training"):
        """统一处理CUDA错误的逻辑

        Returns:
            bool: 是否可以恢复（True=尝试恢复并重试，False=不可恢复或已降级失败）
        """
        import traceback

        err_msg = str(e).lower()
        err_type = type(e).__name__

        # 1. 致命错误：Fail-fast
        fatal_patterns = [
            "illegal memory access",
            "device-side assert",
            "cublas",
            "cudnn",
            "unspecified launch failure",
            "an illegal instruction",
        ]
        if any(p in err_msg for p in fatal_patterns):
            self.logger.critical(f"❌ 致命CUDA错误 ({context}): {err_type} - {e}")
            self.logger.critical(
                "此类错误无法恢复，通常意味着显存越界访问、NaN/Inf传播或硬件问题。"
            )
            self.logger.critical(f"堆栈:\n{traceback.format_exc()}")
            self.logger.critical(
                "建议: 设置 export CUDA_LAUNCH_BLOCKING=1 重新运行以定位具体算子。"
            )
            raise e  # 直接抛出，终止进程

        # 2. OOM错误：尝试恢复
        is_oom = "out of memory" in err_msg or isinstance(
            e, torch.cuda.OutOfMemoryError
        )

        if is_oom:
            self.logger.warning(f"⚠️ CUDA OOM ({context}): {e}")

            # 打印显存状态
            try:
                mem_summary = torch.cuda.memory_summary(abbreviated=True)
                # 只取最后几行，避免刷屏
                mem_lines = mem_summary.split("\n")[-10:]
                self.logger.info("显存状态摘要:\n" + "\n".join(mem_lines))
            except Exception:
                pass

            # 执行清理
            self.cleanup_cuda()

            # 检查是否允许重试配置
            oom_cfg = self._cfg_select("training.oom_recovery", default={})
            # 如果是字典，包装一下方便取值；如果是Config对象则直接取
            if isinstance(oom_cfg, dict):
                enabled = oom_cfg.get("enabled", True)
            else:
                enabled = getattr(oom_cfg, "enabled", True)

            if enabled:
                self.logger.info("尝试 OOM 恢复流程...")
                return True
            else:
                self.logger.warning("OOM恢复未启用，跳过Batch")
                return False

        # 3. 其他未知RuntimeError
        self.logger.error(f"❌ 未知RuntimeError ({context}): {e}")
        self.logger.error(f"堆栈:\n{traceback.format_exc()}")
        return False  # 默认不恢复，除非是明确的OOM

    def _apply_random_masking(
        self, target_hr: torch.Tensor, crop_config: DictConfig
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """应用随机掩码（Crop & Pad）生成稀疏观测

        Args:
            target_hr: 高分辨率目标序列 [B, T, C, H, W]
            crop_config: 裁剪配置

        Returns:
            inputs_sparse: [B*N, T, C, H, W] (稀疏观测)
            targets_dense: [B*N, T, C, H, W] (全图目标)
            masks: [B*N, T, C, H, W] (掩码, 1为观测区域, 0为缺失)
        """
        # 维度兼容性修复：如果输入是4D [B, C, H, W]，扩展为 [B, 1, C, H, W]
        if target_hr.dim() == 4:
            target_hr = target_hr.unsqueeze(1)

        B, T, C, H, W = target_hr.shape
        try:
            crop_size = int(crop_config.size)
        except TypeError:
            # Handle ListConfig/list case by taking the first element
            if isinstance(crop_config.size, (list, tuple)) or hasattr(
                crop_config.size, "__iter__"
            ):
                crop_size = int(crop_config.size[0])
            else:
                # Fallback: try parsing string representation
                import ast

                try:
                    val = ast.literal_eval(str(crop_config.size))
                    if isinstance(val, (list, tuple)):
                        crop_size = int(val[0])
                    else:
                        crop_size = int(val)
                except Exception:
                    raise TypeError(
                        f"Cannot convert crop_config.size ({type(crop_config.size)}: {crop_config.size}) to int"
                    )

        try:
            n_patches = int(crop_config.patches_per_image)
        except TypeError:
            n_patches = 1

        inputs_list = []
        targets_list = []
        masks_list = []

        for _ in range(n_patches):
            # 为了效率，整个Batch使用相同的随机裁剪位置
            h_start = torch.randint(0, H - crop_size + 1, (1,)).item()
            w_start = torch.randint(0, W - crop_size + 1, (1,)).item()

            mask = torch.zeros_like(target_hr)
            mask[..., h_start : h_start + crop_size, w_start : w_start + crop_size] = (
                1.0
            )

            input_sparse = target_hr * mask

            inputs_list.append(input_sparse)
            targets_list.append(target_hr)
            masks_list.append(mask)

        inputs = torch.cat(inputs_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        masks = torch.cat(masks_list, dim=0)

        return inputs, targets, masks

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        # [紧急修复] Test Only 模式拦截
        if self.config.get("testing", {}).get("test_only", False):
            self.logger.info("🛑 检测到 Test Only 模式，跳过 train_epoch")
            return 0.0

        model_to_train = self.get_model()
        model_to_train.train()
        total_loss = 0.0
        total_loss_unscaled = 0.0
        num_batches = len(self.train_loader)
        train_nar_sum = 0.0
        train_tf_sum = 0.0
        train_nar_count = 0
        train_tf_count = 0
        train_dc_sum = 0.0
        train_spec_sum = 0.0
        train_dc_count = 0
        train_spec_count = 0
        train_batches_count = 0

        # 重置性能窗口累计（按epoch）
        self._perf_fetch_time = 0.0
        self._perf_data_time = 0.0
        self._perf_compute_time = 0.0
        self._perf_batches = 0
        self._perf_samples = 0

        # 获取当前T_out
        current_T_out = self.get_current_T_out(epoch)

        # 梯度累积配置
        accumulation_steps = self.memory_config["gradient_accumulation_steps"]

        # 记录并重置本epoch的显存峰值统计
        gpu_total = 0.0
        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
                gpu_total = (
                    torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).total_memory
                    / 1024**3
                )
            except Exception:
                gpu_total = 0.0

        # 性能监控与压测配置（按新YAML路径）
        try:
            perf_cfg = getattr(self.config, "performance_monitoring", None)
            # 窗口报告秒级间隔，兼容命名
            if perf_cfg is not None:
                perf_window_sec = int(
                    getattr(
                        perf_cfg,
                        "report_interval_seconds",
                        getattr(perf_cfg, "interval_sec", 30),
                    )
                )
            else:
                perf_window_sec = 30
            if perf_window_sec <= 0:
                perf_window_sec = 30
        except Exception:
            perf_window_sec = 30
        try:
            bm_cfg = getattr(self.config, "benchmark", None)
            bench_enabled = (
                bool(getattr(bm_cfg, "enabled", True)) if bm_cfg is not None else True
            )
            warmup_steps = (
                int(getattr(bm_cfg, "warmup_steps", 10)) if bm_cfg is not None else 10
            )
            measure_steps = (
                int(getattr(bm_cfg, "measure_steps", 100))
                if bm_cfg is not None
                else 100
            )
            step_report_interval = (
                int(getattr(bm_cfg, "report_interval", 5)) if bm_cfg is not None else 5
            )
            max_runtime_seconds = (
                int(getattr(bm_cfg, "max_runtime_seconds", 60))
                if bm_cfg is not None
                else 60
            )
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
        if getattr(self, "distributed", False):
            # 优先使用 train_loader.sampler
            if (
                hasattr(self, "train_loader")
                and self.train_loader is not None
                and hasattr(self.train_loader, "sampler")
                and hasattr(self.train_loader.sampler, "set_epoch")
            ):
                try:
                    self.train_loader.sampler.set_epoch(epoch)
                except Exception:
                    pass
            # 兼容旧逻辑
            elif (
                hasattr(self, "train_sampler")
                and self.train_sampler is not None
                and hasattr(self.train_sampler, "set_epoch")
            ):
                try:
                    self.train_sampler.set_epoch(epoch)
                except Exception:
                    pass
        try:
            import torch.distributed as dist

            is_primary = (
                (not dist.is_available())
                or (not dist.is_initialized())
                or (dist.get_rank() == 0)
            )
        except Exception:
            is_primary = True
        progress_bar = (
            tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}",
                mininterval=0.5,
                smoothing=0.0,
                leave=True,
                dynamic_ncols=True,
            )
            if is_primary
            else self.train_loader
        )

        # 初始化梯度累积与TF调度
        self.optimizer.zero_grad()
        # 记录本epoch内发生的优化步次数，用于确保调度器步进顺序正确
        self._epoch_opt_steps = 0

        # Task C: 初始化Batch跳过统计
        self._epoch_skip_count = 0
        self._epoch_total_batches = 0

        # 当使用SequentialSpatiotemporalModel时，按epoch设置teacher forcing概率
        try:
            m = self.get_model()
            if hasattr(m, "set_epoch"):
                m.set_epoch(
                    epoch,
                    decay=getattr(self.config.training, "teacher_forcing_decay", 0.95),
                )
            try:
                tm = getattr(m, "_last_teacher_mask", None)
                if tm is not None and hasattr(self, "output_dir"):
                    import json as _json

                    import numpy as _np

                    out = self.output_dir / "diagnostics"
                    out.mkdir(parents=True, exist_ok=True)
                    _np.save(
                        out / f"teacher_mask_epoch_{epoch+1}.npy",
                        tm.detach().cpu().numpy(),
                    )
                    with open(out / f"teacher_mask_epoch_{epoch+1}.json", "w") as f:
                        _json.dump(
                            {
                                "mean": float(tm.mean().item()),
                                "std": float(tm.std().item()),
                            },
                            f,
                        )
            except Exception:
                pass
        except Exception:
            pass
        # 记录上一个batch结束时间，用于估算下一次DataLoader取数耗时
        prev_batch_end_cpu = time.perf_counter()

        for batch_idx, batch in enumerate(progress_bar):
            # 跳过None批次（安全collate过滤可能导致None返回）
            if batch is None:
                self._epoch_skip_count += 1
                self._epoch_total_batches += 1
                continue

            self._epoch_total_batches += 1

            # 初始化关键变量，防止UnboundLocalError
            pred_seq = None
            pred_seq_tf = None

            t0 = time.perf_counter()
            batch_start_wall = time.time()
            try:
                # 设备搬运耗时起点
                load_t0 = time.perf_counter()
                # 移动数据到设备
                input_seq = batch["input_sequence"].to(
                    self.device, non_blocking=True
                )  # [B, T_in, C, H, W]
                target_seq = batch["target_sequence"].to(
                    self.device, non_blocking=True
                )  # [B, T_out, C, H, W]

                # Crop训练逻辑 (Masking)
                crop_cfg = getattr(getattr(self.config, "training", None), "crop", None)
                masks_seq = None
                if crop_cfg and getattr(crop_cfg, "enabled", False):
                    # 使用随机掩码生成 input_seq (Sparse) 和 target_seq (Dense)
                    input_seq, target_seq, masks_seq = self._apply_random_masking(
                        target_seq, crop_cfg
                    )
                    # 强制清空 observed_lr_sequence，防止后续逻辑误用全图LR
                    if isinstance(batch, dict):
                        batch["observed_lr_sequence"] = None
                        # 将生成的 masks_seq 放回 batch，以便后续逻辑（如拼接）能使用
                        batch["mask_sequence"] = masks_seq

                data_end = time.perf_counter()

                # 根据课程学习调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]

                # 计算耗时起点
                comp_t0 = time.perf_counter()
                # 前向传播（AMP按配置与设备启用）
                use_amp = bool(
                    getattr(getattr(self, "config", None), "training", None)
                    and getattr(self.config.training.amp, "enabled", False)
                ) and (self.device.type == "cuda")
                if use_amp:
                    amp_ctx = autocast(
                        device_type="cuda",
                        dtype=getattr(self, "autocast_dtype", torch.bfloat16),
                        enabled=True,
                    )
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
                    is_seq_model = hasattr(model, "spatial_forward") and hasattr(
                        model, "temporal_forward"
                    )
                    if is_seq_model:
                        if self._is_spatial_phase(epoch):
                            # LR输入模式：优先使用observed_lr_sequence；否则退回baseline首帧
                            if (
                                isinstance(batch, dict)
                                and ("observed_lr_sequence" in batch)
                                and (batch["observed_lr_sequence"] is not None)
                            ):
                                lr_seq = batch["observed_lr_sequence"]
                                x_single = lr_seq[:, 0]
                                # 拼接低分辨率坐标/掩码（如存在）
                                try:
                                    if ("coords_lr_sequence" in batch) and (
                                        batch["coords_lr_sequence"] is not None
                                    ):
                                        coords_lr = batch["coords_lr_sequence"][:, 0]
                                        x_single = torch.cat(
                                            [x_single, coords_lr], dim=1
                                        )
                                    if ("mask_lr_sequence" in batch) and (
                                        batch["mask_lr_sequence"] is not None
                                    ):
                                        mask_lr = batch["mask_lr_sequence"][:, 0]
                                        x_single = torch.cat([x_single, mask_lr], dim=1)
                                    if ("fourier_pe_sequence" in batch) and (
                                        batch["fourier_pe_sequence"] is not None
                                    ):
                                        # 若提供的是HR PE，则在LR模式下不拼接；此处预留接口
                                        pass
                                except Exception:
                                    pass
                                # 对齐模型期望的输入通道数
                                try:
                                    exp_in = int(
                                        getattr(
                                            self.config.model,
                                            "in_channels",
                                            x_single.shape[1],
                                        )
                                    )
                                    if x_single.shape[1] > exp_in:
                                        x_single = x_single[:, :exp_in]
                                    elif x_single.shape[1] < exp_in:
                                        pad = exp_in - x_single.shape[1]
                                        zeros = torch.zeros(
                                            x_single.size(0),
                                            pad,
                                            x_single.size(2),
                                            x_single.size(3),
                                            dtype=x_single.dtype,
                                            device=x_single.device,
                                        )
                                        x_single = torch.cat([x_single, zeros], dim=1)
                                except Exception:
                                    pass
                            else:
                                x_single = input_seq[:, 0]
                                # Apply training degradation if available (e.g. for SR/Crop)
                                if (
                                    hasattr(self, "training_degradation_op")
                                    and self.training_degradation_op is not None
                                ):
                                    x_single = self.training_degradation_op(x_single)
                                # baseline模式下拼接HR坐标与掩码（如存在）
                                try:
                                    if ("coords_sequence" in batch) and (
                                        batch["coords_sequence"] is not None
                                    ):
                                        coords_hr = batch["coords_sequence"][:, 0]
                                        x_single = torch.cat(
                                            [x_single, coords_hr], dim=1
                                        )
                                    if ("mask_sequence" in batch) and (
                                        batch["mask_sequence"] is not None
                                    ):
                                        mask_hr = batch["mask_sequence"][:, 0]
                                        x_single = torch.cat([x_single, mask_hr], dim=1)
                                    if ("fourier_pe_sequence" in batch) and (
                                        batch["fourier_pe_sequence"] is not None
                                    ):
                                        pe_hr = batch["fourier_pe_sequence"][:, 0]
                                        x_single = torch.cat([x_single, pe_hr], dim=1)
                                except Exception:
                                    pass
                                try:
                                    # Debug info
                                    # self.logger.info(f"DEBUG: x_single shape: {x_single.shape}")

                                    exp_in = int(
                                        getattr(
                                            self.config.model,
                                            "in_channels",
                                            x_single.shape[1],
                                        )
                                    )
                                    if x_single.shape[1] > exp_in:
                                        x_single = x_single[:, :exp_in]
                                    elif x_single.shape[1] < exp_in:
                                        pad = exp_in - x_single.shape[1]
                                        zeros = torch.zeros(
                                            x_single.size(0),
                                            pad,
                                            x_single.size(2),
                                            x_single.size(3),
                                            dtype=x_single.dtype,
                                            device=x_single.device,
                                        )
                                        x_single = torch.cat([x_single, zeros], dim=1)
                                        # self.logger.info(f"DEBUG: Padded x_single to {x_single.shape} (Config based)")

                                    # 自动适配输入通道数（Model based）- Double check
                                    raw_model = (
                                        model.module
                                        if hasattr(model, "module")
                                        else model
                                    )
                                    if hasattr(raw_model, "in_channels"):
                                        expected_in = raw_model.in_channels
                                        current_in = x_single.shape[1]
                                        if current_in < expected_in:
                                            pad_c = expected_in - current_in
                                            # Pad with zeros: [B, pad_c, H, W]
                                            padding = torch.zeros(
                                                x_single.size(0),
                                                pad_c,
                                                x_single.size(2),
                                                x_single.size(3),
                                                device=x_single.device,
                                                dtype=x_single.dtype,
                                            )
                                            x_single = torch.cat(
                                                [x_single, padding], dim=1
                                            )
                                            # self.logger.info(f"DEBUG: Padded x_single to {x_single.shape} (Model based)")
                                except Exception as e:
                                    print(f"❌ Error in input padding: {e}")
                                    pass

                            # 最终形状检查，如果还是不对，打印警告
                            if hasattr(model, "module") and hasattr(
                                model.module, "in_channels"
                            ):
                                exp_c = model.module.in_channels
                                if x_single.shape[1] != exp_c:
                                    # 强制填充
                                    pad_c = exp_c - x_single.shape[1]
                                    if pad_c > 0:
                                        padding = torch.zeros(
                                            x_single.size(0),
                                            pad_c,
                                            x_single.size(2),
                                            x_single.size(3),
                                            device=x_single.device,
                                            dtype=x_single.dtype,
                                        )
                                        x_single = torch.cat([x_single, padding], dim=1)
                            elif hasattr(model, "in_channels"):
                                exp_c = model.in_channels
                                if x_single.shape[1] != exp_c:
                                    # 强制填充
                                    pad_c = exp_c - x_single.shape[1]
                                    if pad_c > 0:
                                        padding = torch.zeros(
                                            x_single.size(0),
                                            pad_c,
                                            x_single.size(2),
                                            x_single.size(3),
                                            device=x_single.device,
                                            dtype=x_single.dtype,
                                        )
                                        x_single = torch.cat([x_single, padding], dim=1)

                            if hasattr(model, "spatial_forward"):
                                spatial_output = model.spatial_forward(x_single)
                                y_single = spatial_output.spatial_pred
                            else:
                                try:
                                    dtype = next(model.parameters()).dtype
                                    x_single = x_single.to(dtype=dtype)
                                except Exception:
                                    pass
                                try:
                                    device = next(model.parameters()).device
                                    x_single = x_single.to(device=device)
                                except Exception:
                                    pass
                                y_single = model(x_single)
                            # 若输入为LR，损失前上采样到HR尺寸
                            try:
                                if y_single.dim() == 4 and (
                                    y_single.shape[-2:] != target_seq[:, 0].shape[-2:]
                                ):
                                    y_single = torch.nn.functional.interpolate(
                                        y_single,
                                        size=target_seq[:, 0].shape[-2:],
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                            except Exception:
                                pass
                            try:
                                if y_single.dim() == 4 and (
                                    y_single.shape[-2:] != target_seq[:, 0].shape[-2:]
                                ):
                                    y_single = torch.nn.functional.interpolate(
                                        y_single,
                                        size=target_seq[:, 0].shape[-2:],
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                            except Exception:
                                pass
                            target_single = target_seq[:, 0]
                            # 优先使用dataset返回的observed_lr_sequence（已降质）
                            observation_single = None
                            if masks_seq is not None:
                                # 如果是随机Mask训练，input_seq[:,0] 就是观测值
                                observation_single = input_seq[:, 0]
                            elif (
                                isinstance(batch, dict)
                                and ("observed_lr_sequence" in batch)
                                and (batch["observed_lr_sequence"] is not None)
                            ):
                                obs_lr_seq = batch["observed_lr_sequence"]
                                if obs_lr_seq.dim() == 5:  # [B, T, C, H, W]
                                    observation_single = obs_lr_seq[:, 0]
                                elif obs_lr_seq.dim() == 4:  # [B, C, H, W]
                                    observation_single = obs_lr_seq

                            # 计算 pred_obs (H(Pred)) 用于 DC Loss
                            pred_obs_single = None
                            baseline_single = None
                            try:
                                if (
                                    hasattr(self, "norm_stats")
                                    and self.norm_stats is not None
                                ):
                                    m = self.norm_stats.get(
                                        "data_mean",
                                        self.norm_stats.get(
                                            "u_mean", torch.tensor(0.0)
                                        ),
                                    )
                                    s = self.norm_stats.get(
                                        "data_std",
                                        self.norm_stats.get("u_std", torch.tensor(1.0)),
                                    )
                                    mean_t = torch.as_tensor(
                                        m, device=self.device
                                    ).reshape(1, -1, 1, 1)
                                    std_t = torch.as_tensor(
                                        s, device=self.device
                                    ).reshape(1, -1, 1, 1)

                                    # 反归一化 Pred
                                    pred_orig = y_single * std_t + mean_t
                                    # 反归一化 GT
                                    gt_orig = target_single * std_t + mean_t

                                    # 应用观测算子 H(Pred) (Mismatch Experiment: 使用 training_degradation_op)
                                    if masks_seq is not None:
                                        # 如果有掩码，直接应用掩码
                                        mask_single = masks_seq[:, 0]
                                        pred_obs_single = pred_orig * mask_single
                                        baseline_single = gt_orig * mask_single
                                    elif (
                                        hasattr(self, "training_degradation_op")
                                        and self.training_degradation_op is not None
                                    ):
                                        pred_obs_single = self.training_degradation_op(
                                            pred_orig
                                        )
                                        baseline_single = self.training_degradation_op(
                                            gt_orig
                                        )
                                    elif (
                                        hasattr(self, "observation_op")
                                        and self.observation_op is not None
                                    ):
                                        # 回退逻辑
                                        pred_obs_single = self.observation_op(pred_orig)
                                        baseline_single = self.observation_op(gt_orig)
                            except Exception:
                                pred_obs_single = None
                                baseline_single = None

                            obs_data_single = {
                                "y": observation_single,
                                "observation": observation_single,
                                "pred_obs": pred_obs_single,
                                "baseline": baseline_single,
                                "baseline_type": "H_gt",
                                "h_params": (
                                    self.h_params
                                    if self.h_params is not None
                                    else {
                                        "task": "SR",
                                        "scale": int(
                                            self._cfg_select(
                                                "data.observation.sr.scale_factor",
                                                "data.observation.scale_factor",
                                                default=2,
                                            )
                                        ),
                                        "sigma": float(
                                            self._cfg_select(
                                                "data.observation.sr.blur_sigma",
                                                "data.observation.blur_sigma",
                                                default=1.0,
                                            )
                                        ),
                                        "kernel_size": int(
                                            self._cfg_select(
                                                "data.observation.sr.blur_kernel_size",
                                                "data.observation.kernel_size",
                                                default=5,
                                            )
                                        ),
                                        "boundary": str(
                                            self._cfg_select(
                                                "data.observation.sr.boundary_mode",
                                                "data.observation.boundary",
                                                default="mirror",
                                            )
                                        ),
                                        "downsample_interpolation": str(
                                            self._cfg_select(
                                                "data.observation.sr.downsample_mode",
                                                "data.observation.downsample_interpolation",
                                                default="area",
                                            )
                                        ),
                                    }
                                ),
                            }
                            from ops.losses import compute_total_loss

                            self.ensure_norm_stats()
                            try:
                                losses = compute_total_loss(
                                    pred_z=y_single,
                                    target_z=target_single,
                                    obs_data=obs_data_single,
                                    norm_stats=self.norm_stats,
                                    config=self.config,
                                )
                                loss = losses["total_loss"]
                            except Exception as e:
                                # 训练阶段容错：跳过该batch
                                if not hasattr(self, "_has_warned_metric_failure"):
                                    self.logger.warning(
                                        f"⚠️ 训练阶段 (Single) Loss/Metrics 计算失败 (batch {batch_idx})，将跳过此Batch。错误详情: {e}"
                                    )
                                    self._has_warned_metric_failure = True
                                self._epoch_skip_count += 1
                                continue

                            # 显式赋值pred_seq，防止后续UnboundLocalError
                            if y_single is not None:
                                pred_seq = y_single.unsqueeze(1)
                        else:
                            if (
                                torch.isnan(input_seq).any()
                                or torch.isinf(input_seq).any()
                            ):
                                input_seq = torch.nan_to_num(
                                    input_seq, nan=0.0, posinf=1e6, neginf=-1e6
                                )
                            if (
                                torch.isnan(target_seq).any()
                                or torch.isinf(target_seq).any()
                            ):
                                target_seq = torch.nan_to_num(
                                    target_seq, nan=0.0, posinf=1e6, neginf=-1e6
                                )
                            # 支持滚动训练：与验证分布一致
                            try:
                                rollout_train = bool(
                                    self._cfg_select(
                                        "training.rollout_training", default=False
                                    )
                                )
                            except Exception:
                                rollout_train = False
                            if rollout_train and hasattr(model, "rollout_inference"):
                                # 训练阶段需要保留梯度
                                pred_seq = model.rollout_inference(
                                    input_seq,
                                    current_T_out,
                                    step_by_step=True,
                                    preserve_grad=True,
                                )
                            else:
                                model_output = model(input_seq, target_seq)
                                if (
                                    isinstance(model_output, dict)
                                    and "final_pred" in model_output
                                ):
                                    pred_seq = model_output["final_pred"]
                                elif torch.is_tensor(model_output):
                                    pred_seq = model_output
                                else:
                                    # Fallback
                                    pred_seq = None

                                # 维度对齐 [B, T, C, H, W]
                                if pred_seq is not None and pred_seq.dim() == 4:
                                    pred_seq = pred_seq.unsqueeze(1)

                            if pred_seq is not None and (
                                torch.isnan(pred_seq).any()
                                or torch.isinf(pred_seq).any()
                            ):
                                pred_seq = torch.nan_to_num(
                                    pred_seq, nan=0.0, posinf=1e6, neginf=-1e6
                                )
                    else:
                        # 非顺序模型：按原逻辑（ARWrapper/传统模型）
                        if (
                            hasattr(self, "config")
                            and hasattr(self.config, "ar")
                            and not bool(getattr(self.config.ar, "enabled", True))
                        ):
                            # 空间-only路径：保持原来单帧损失
                            if (
                                isinstance(batch, dict)
                                and ("observed_lr_sequence" in batch)
                                and (batch["observed_lr_sequence"] is not None)
                            ):
                                lr_seq = batch["observed_lr_sequence"]
                                x_single = lr_seq[:, 0]
                                try:
                                    if ("coords_lr_sequence" in batch) and (
                                        batch["coords_lr_sequence"] is not None
                                    ):
                                        coords_lr = batch["coords_lr_sequence"][:, 0]
                                        x_single = torch.cat(
                                            [x_single, coords_lr], dim=1
                                        )
                                    if ("mask_lr_sequence" in batch) and (
                                        batch["mask_lr_sequence"] is not None
                                    ):
                                        mask_lr = batch["mask_lr_sequence"][:, 0]
                                        x_single = torch.cat([x_single, mask_lr], dim=1)
                                except Exception:
                                    pass
                                try:
                                    exp_in = int(
                                        getattr(
                                            self.config.model,
                                            "in_channels",
                                            x_single.shape[1],
                                        )
                                    )
                                    if x_single.shape[1] > exp_in:
                                        x_single = x_single[:, :exp_in]
                                    elif x_single.shape[1] < exp_in:
                                        pad = exp_in - x_single.shape[1]
                                        zeros = torch.zeros(
                                            x_single.size(0),
                                            pad,
                                            x_single.size(2),
                                            x_single.size(3),
                                            dtype=x_single.dtype,
                                            device=x_single.device,
                                        )
                                        x_single = torch.cat([x_single, zeros], dim=1)
                                except Exception:
                                    pass
                            else:
                                x_single = input_seq[:, 0]
                                # Apply training degradation if available
                                if (
                                    hasattr(self, "training_degradation_op")
                                    and self.training_degradation_op is not None
                                ):
                                    x_single = self.training_degradation_op(x_single)
                                try:
                                    if ("coords_sequence" in batch) and (
                                        batch["coords_sequence"] is not None
                                    ):
                                        coords_hr = batch["coords_sequence"][:, 0]
                                        x_single = torch.cat(
                                            [x_single, coords_hr], dim=1
                                        )
                                    if ("mask_sequence" in batch) and (
                                        batch["mask_sequence"] is not None
                                    ):
                                        mask_hr = batch["mask_sequence"][:, 0]
                                        x_single = torch.cat([x_single, mask_hr], dim=1)
                                except Exception:
                                    pass
                                try:
                                    exp_in = int(
                                        getattr(
                                            self.config.model,
                                            "in_channels",
                                            x_single.shape[1],
                                        )
                                    )
                                    if x_single.shape[1] > exp_in:
                                        x_single = x_single[:, :exp_in]
                                    elif x_single.shape[1] < exp_in:
                                        pad = exp_in - x_single.shape[1]
                                        zeros = torch.zeros(
                                            x_single.size(0),
                                            pad,
                                            x_single.size(2),
                                            x_single.size(3),
                                            dtype=x_single.dtype,
                                            device=x_single.device,
                                        )
                                        x_single = torch.cat([x_single, zeros], dim=1)
                                except Exception:
                                    pass
                            if hasattr(model, "spatial_forward"):
                                spatial_output = model.spatial_forward(x_single)
                                y_single = spatial_output.spatial_pred
                            else:
                                try:
                                    dtype = next(model.parameters()).dtype
                                    x_single = x_single.to(dtype=dtype)
                                except Exception:
                                    pass
                                try:
                                    device = next(model.parameters()).device
                                    dtype = next(model.parameters()).dtype
                                    x_single = x_single.to(device=device, dtype=dtype)
                                except Exception:
                                    pass
                                y_single = model(x_single)
                            try:
                                if y_single.dim() == 4 and (
                                    y_single.shape[-2:] != target_seq[:, 0].shape[-2:]
                                ):
                                    y_single = torch.nn.functional.interpolate(
                                        y_single,
                                        size=target_seq[:, 0].shape[-2:],
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                            except Exception:
                                pass
                            try:
                                if y_single.dim() == 4 and (
                                    y_single.shape[-2:] != target_seq[:, 0].shape[-2:]
                                ):
                                    y_single = torch.nn.functional.interpolate(
                                        y_single,
                                        size=target_seq[:, 0].shape[-2:],
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                            except Exception:
                                pass
                            target_single = target_seq[:, 0]
                            # 移除单步独立 Loss 计算，统一由后续序列 Loss 处理
                            # 初始化pred_seq用于后续序列损失计算（空间-only模式）
                            pred_seq = y_single.unsqueeze(1)
                        else:
                            # 针对裸空间模型（如UNet）适配 5D -> 4D 及通道 Padding
                            # 注意：input_seq是5D [B, T, C, H, W]
                            model_input = input_seq
                            is_raw_spatial = False

                            # 启发式判断：如果不是 ARWrapper/SequenceModel，且输入是 5D
                            if input_seq.dim() == 5:
                                raw_m = (
                                    model.module if hasattr(model, "module") else model
                                )
                                # 检查是否有序列处理方法
                                has_seq_method = (
                                    hasattr(raw_m, "autoregressive_predict")
                                    or hasattr(raw_m, "temporal_forward")
                                    or (
                                        hasattr(raw_m, "is_temporal")
                                        and raw_m.is_temporal
                                    )
                                )

                                if not has_seq_method:
                                    # 假设是单帧模型，只取第一帧
                                    model_input = input_seq[:, 0]
                                    is_raw_spatial = True

                                    # [Fix] Apply degradation during training if available
                                    # This ensures the model sees masked/degraded input instead of full GT
                                    if (
                                        hasattr(self, "training_degradation_op")
                                        and self.training_degradation_op is not None
                                    ):
                                        model_input = self.training_degradation_op(
                                            model_input
                                        )
                                    elif (
                                        hasattr(self, "observation_op")
                                        and self.observation_op is not None
                                    ):
                                        model_input = self.observation_op(model_input)

                                    # 自动适配输入通道数
                                    if hasattr(raw_m, "in_channels"):
                                        exp_in = raw_m.in_channels
                                        if model_input.shape[1] < exp_in:
                                            pad_c = exp_in - model_input.shape[1]
                                            padding = torch.zeros(
                                                model_input.size(0),
                                                pad_c,
                                                model_input.size(2),
                                                model_input.size(3),
                                                device=model_input.device,
                                                dtype=model_input.dtype,
                                            )
                                            model_input = torch.cat(
                                                [model_input, padding], dim=1
                                            )

                            # 尝试自适应调用 forward
                            try:
                                if is_raw_spatial:
                                    out = model(model_input)
                                else:
                                    # 尝试带有 T_out 的调用 (针对 ARWrapper 等)
                                    out = model(input_seq, current_T_out, target_seq)
                            except TypeError:
                                # 回退到标准调用 (针对 SequentialSpatiotemporalModel 等)
                                if is_raw_spatial:
                                    out = model(model_input)
                                else:
                                    out = model(input_seq, target_seq)

                            if isinstance(out, dict) and "final_pred" in out:
                                pred_seq = out["final_pred"]
                            elif torch.is_tensor(out):
                                pred_seq = out
                            else:
                                pred_seq = None

                            if pred_seq is not None and pred_seq.dim() == 4:
                                pred_seq = pred_seq.unsqueeze(1)

                # 统一损失装配（z-score域重建 + 原值域谱/DC）
                # 顺序模型或AR路径下的序列损失
                from ops.losses import compute_ar_total_loss

                # 最终保护：确保pred_seq已赋值
                if pred_seq is None:
                    try:
                        self.logger.warning(
                            f"Batch {batch_idx}: pred_seq is None. Using target_seq as fallback."
                        )
                        pred_seq = target_seq.detach().clone()
                    except Exception:
                        pass

                if pred_seq is None:
                    try:
                        self.logger.warning(
                            f"Batch {batch_idx}: Failed to recover pred_seq. Skipping batch."
                        )
                    except Exception:
                        pass
                    continue

                if True:  # 只要有序列输出，就计算序列损失
                    observation_seq = None
                    pred_obs_seq = None
                    # 仅使用 Trainer 中的观测算子生成观测数据
                    try:
                        if (
                            hasattr(self, "training_degradation_op")
                            and self.training_degradation_op is not None
                            and hasattr(self, "norm_stats")
                            and self.norm_stats is not None
                        ):
                            B, T, C, H, W = target_seq.shape
                            # 使用完整通道的统计量
                            m = self.norm_stats["mean"].to(self.device)
                            s = self.norm_stats["std"].to(self.device)
                            # [1, 1, C, 1, 1] for broadcasting
                            mean_t = m.reshape(1, 1, -1, 1, 1)
                            std_t = s.reshape(1, 1, -1, 1, 1)

                            # 生成 observation_seq (GT -> Degradation -> Obs)
                            gt_orig = target_seq * std_t + mean_t
                            gt_flat = gt_orig.reshape(B * T, C, H, W)

                            # Mismatch Experiment: 使用 training_degradation_op
                            obs_flat = self.training_degradation_op(gt_flat)
                            oh, ow = obs_flat.shape[-2:]
                            observation_seq = obs_flat.reshape(B, T, C, oh, ow)

                            # 生成 pred_obs (Degradation(Pred)) 用于 DC Loss
                            if pred_seq.dim() == 5:
                                pred_orig = pred_seq * std_t + mean_t
                                pred_flat = pred_orig.reshape(B * T, C, H, W)
                                pred_obs_flat = self.training_degradation_op(pred_flat)
                                pred_obs_seq = pred_obs_flat.reshape(B, T, C, oh, ow)
                        # 回退兼容旧代码 (若未初始化 training_degradation_op)
                        elif (
                            hasattr(self, "observation_op")
                            and self.observation_op is not None
                            and hasattr(self, "norm_stats")
                            and self.norm_stats is not None
                        ):
                            B, T, C, H, W = target_seq.shape
                            m = self.norm_stats["mean"].to(self.device)
                            s = self.norm_stats["std"].to(self.device)
                            mean_t = m.reshape(1, 1, -1, 1, 1)
                            std_t = s.reshape(1, 1, -1, 1, 1)
                            gt_orig = target_seq * std_t + mean_t
                            gt_flat = gt_orig.reshape(B * T, C, H, W)
                            obs_flat = self.observation_op(gt_flat)
                            oh, ow = obs_flat.shape[-2:]
                            observation_seq = obs_flat.reshape(B, T, C, oh, ow)
                            if pred_seq.dim() == 5:
                                pred_orig = pred_seq * std_t + mean_t
                                pred_flat = pred_orig.reshape(B * T, C, H, W)
                                pred_obs_flat = self.observation_op(pred_flat)
                                pred_obs_seq = pred_obs_flat.reshape(B, T, C, oh, ow)
                    except Exception:
                        observation_seq = None
                        pred_obs_seq = None

                    obs_last = (
                        observation_seq[:, -1] if observation_seq is not None else None
                    )
                    obs_data = {
                        "y": obs_last,
                        "observation_seq": observation_seq,
                        "observation": obs_last,
                        "pred_obs": pred_obs_seq,
                        "baseline_seq": observation_seq,
                        "baseline": obs_last,
                        "h_params": self.h_params,
                        "baseline_type": "H_gt",
                    }
                    self.ensure_norm_stats()
                    try:
                        losses = compute_ar_total_loss(
                            pred_seq=pred_seq,
                            gt_seq=target_seq,
                            obs_data=obs_data,
                            norm_stats=self.norm_stats,
                            config=self.config,
                        )
                        loss = losses["total_loss"]
                    except Exception as e:
                        # 训练阶段容错：跳过该batch的metrics/loss统计，并打印一次性告警
                        if not hasattr(self, "_has_warned_metric_failure_ar"):
                            self.logger.warning(
                                f"⚠️ 训练阶段 (AR) Loss/Metrics 计算失败 (batch {batch_idx})，将跳过此Batch。错误详情: {e}"
                            )
                            self._has_warned_metric_failure_ar = True
                        self._epoch_skip_count += 1
                        continue
                    try:
                        recon_nar = losses.get("reconstruction_loss", None)
                        if torch.is_tensor(recon_nar):
                            train_nar_sum += float(recon_nar.detach().mean().item())
                            train_nar_count += 1
                        elif recon_nar is not None:
                            train_nar_sum += float(recon_nar)
                            train_nar_count += 1
                    except Exception:
                        pass
                    try:
                        dc_term = losses.get("dc_loss", None)
                        if torch.is_tensor(dc_term):
                            train_dc_sum += float(dc_term.detach().mean().item())
                            train_dc_count += 1
                        elif dc_term is not None:
                            train_dc_sum += float(dc_term)
                            train_dc_count += 1
                    except Exception:
                        pass
                    try:
                        spec_term = losses.get("spectral_loss", None)
                        if torch.is_tensor(spec_term):
                            train_spec_sum += float(spec_term.detach().mean().item())
                            train_spec_count += 1
                        elif spec_term is not None:
                            train_spec_sum += float(spec_term)
                            train_spec_count += 1
                    except Exception:
                        pass

                # IO Debug: 训练期可视化输入/输出（受开关与步频控制）
                try:
                    if bool(self._io_debug_cfg.get("enabled", False)):
                        step_interval = int(
                            self._io_debug_cfg.get("train_every_n_steps", 200)
                        )
                        if step_interval > 0 and ((batch_idx + 1) % step_interval == 0):
                            viz_root = (
                                self.output_dir
                                / "io_debug"
                                / f"epoch_{epoch+1:03d}"
                                / "train"
                            )
                            viz_root.mkdir(parents=True, exist_ok=True)
                            max_t = int(self._io_debug_cfg.get("max_time_steps", 4))
                            try:
                                from utils.ar_visualizer import ARTrainingVisualizer

                                dbg_viz = ARTrainingVisualizer(str(viz_root))
                                # 统一裁切时间长度
                                ps = pred_seq.detach().cpu()
                                ts = target_seq.detach().cpu()
                                if ps.shape[1] > max_t:
                                    ps = ps[:, :max_t]
                                if ts.shape[1] > max_t:
                                    ts = ts[:, :max_t]
                                # 可视化最后帧与序列对比
                                dbg_viz.visualize_obs_gt_pred_error(
                                    ts,
                                    ps,
                                    save_name=f"b{batch_idx+1:05d}_obs_gt_pred_error",
                                    norm_stats=self.norm_stats,
                                )
                                dbg_viz.create_temporal_analysis(
                                    ps.numpy(),
                                    ts.numpy(),
                                    save_name=f"b{batch_idx+1:05d}_temporal_analysis",
                                    norm_stats=self.norm_stats,
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

                    # 上方已记录 NAR 重建损失；此处不再重复计算

                    # 梯度累积归一化与数值稳定化
                    loss = loss / accumulation_steps
                    try:
                        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)
                    except Exception:
                        pass

                model = self.get_model()
                use_no_sync = (
                    hasattr(model, "no_sync")
                    and accumulation_steps > 1
                    and ((batch_idx + 1) % accumulation_steps != 0)
                    and ((batch_idx + 1) != num_batches)
                )
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
                if (batch_idx + 1) % accumulation_steps == 0 or (
                    batch_idx + 1
                ) == num_batches:
                    # 梯度裁剪
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    model = self.get_model()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.training.gradient_clip_val
                    )

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

                total_loss += loss.item()
                total_loss_unscaled += loss.item() * accumulation_steps
                compute_end = time.perf_counter()

                # 记录当前batch结束CPU时间用于下一次fetch耗时估算
                prev_batch_end_cpu = time.perf_counter()

            except RuntimeError as e:
                # 使用改进的CUDA错误处理
                should_retry = self.handle_cuda_error(e, "training")
                if should_retry:
                    # OOM 恢复策略：Micro-batching
                    try:
                        splits = (
                            int(
                                getattr(
                                    self._cfg_select(
                                        "training.oom_recovery", default={}
                                    ),
                                    "microbatch_splits",
                                    4,
                                )
                            )
                            or 4
                        )
                        self.logger.info(
                            f"🔄 尝试降级：Micro-batching (splits={splits})"
                        )

                        # 清理显存
                        self.cleanup_cuda()

                        # 手动拆分数据
                        batch_size = input_seq.size(0)
                        micro_bs = max(1, batch_size // splits)

                        # 确保梯度清零
                        self.optimizer.zero_grad(set_to_none=True)

                        micro_loss_sum = 0.0

                        # 遍历微批次
                        for i in range(0, batch_size, micro_bs):
                            mb_input = input_seq[i : i + micro_bs]
                            mb_target = target_seq[i : i + micro_bs]

                            # Forward
                            with autocast(
                                device_type=self.device.type,
                                dtype=self.autocast_dtype,
                                enabled=True,
                            ):
                                # 简化调用逻辑，仅支持核心路径
                                try:
                                    mb_out = model(mb_input, mb_target)
                                except TypeError:
                                    mb_out = model(mb_input)

                                if isinstance(mb_out, dict):
                                    mb_out = mb_out["final_pred"]

                                # Loss
                                from ops.losses import (
                                    compute_total_loss,
                                    rel_l2,
                                )

                                # 简单Loss回退
                                if mb_out.ndim == 4 and mb_target.ndim == 5:
                                    target_to_compare = mb_target[:, 0]
                                elif mb_out.ndim == 5 and mb_target.ndim == 5:
                                    min_t = min(mb_out.shape[1], mb_target.shape[1])
                                    mb_out = mb_out[:, :min_t]
                                    target_to_compare = mb_target[:, :min_t]
                                else:
                                    target_to_compare = mb_target

                                mb_loss = rel_l2(mb_out, target_to_compare)

                                # Scale loss by splits for accumulation
                                mb_loss = mb_loss / (batch_size / micro_bs)  # 近似缩放

                            # Backward
                            if self.scaler:
                                self.scaler.scale(mb_loss).backward()
                            else:
                                mb_loss.backward()

                            micro_loss_sum += mb_loss.item()

                            # 释放微批次图
                            del mb_out, mb_loss

                        # Optimizer Step
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.logger.info(
                            f"✅ Micro-batch recovery successful. Loss: {micro_loss_sum:.4f}"
                        )
                        # 恢复成功，继续下一个Batch
                        continue

                    except Exception as retry_err:
                        self.logger.error(f"❌ OOM Recovery failed: {retry_err}")
                        self.cleanup_cuda()
                        self._epoch_skip_count += 1
                        continue
                else:
                    # 无法恢复，跳过
                    self._epoch_skip_count += 1
                    continue

                # 更新进度条（仅当progress_bar是tqdm对象）
                try:
                    if hasattr(progress_bar, "set_postfix"):
                        progress_bar.set_postfix(
                            {
                                "Loss": f"{float(loss.detach().item()):.6f}",
                                "T_out": int(current_T_out),
                            },
                            refresh=True,
                        )
                except Exception:
                    pass

            # 记录到TensorBoard
            try:
                log_every = int(
                    self._cfg_select("experiment.log_every_n_steps", default=100)
                )
            except Exception:
                log_every = 100
            if log_every > 0 and batch_idx % log_every == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar("Train/Loss", loss.item(), global_step)
                self.writer.add_scalar(
                    "Train/LR", self.optimizer.param_groups[0]["lr"], global_step
                )
                self.writer.add_scalar("Train/T_out", current_T_out, global_step)

        avg_loss = total_loss / max(1, num_batches)
        avg_loss_unscaled = total_loss_unscaled / max(1, num_batches)

        # Task C: 检查跳过比例
        skip_ratio = self._epoch_skip_count / max(1, self._epoch_total_batches)
        self.logger.info(
            f"Batch Skip Ratio: {skip_ratio:.2%} ({self._epoch_skip_count}/{self._epoch_total_batches})"
        )

        skip_threshold = float(
            self._cfg_select("training.max_skip_ratio", default=0.05)
        )
        if skip_ratio > skip_threshold:
            raise RuntimeError(
                f"❌ 训练异常终止：Batch跳过比例 ({skip_ratio:.2%}) 超过阈值 ({skip_threshold:.2%})。\n"
                f"总Batch数: {self._epoch_total_batches}, 跳过数: {self._epoch_skip_count}。\n"
                f"常见原因：数据加载失败(None)、NaN/Inf导致Loss计算失败、CUDA OOM。\n"
                f"请检查日志中之前的警告信息以定位具体原因。"
            )

        avg_train_nar = train_nar_sum / max(1, train_nar_count)
        avg_train_tf = train_tf_sum / max(1, train_tf_count)
        try:
            comp_dc = float(train_dc_sum / max(1, train_dc_count))
        except Exception:
            comp_dc = float("nan")
        try:
            comp_spec = float(train_spec_sum / max(1, train_spec_count))
        except Exception:
            comp_spec = float("nan")

        self.stage_epoch += 1

        try:
            step_count = max(1, len(self.train_batch_losses))
            avg_per_step = float(avg_loss_unscaled) / float(step_count)
            self.logger.info(
                f"Train AvgLoss(per-step)={avg_per_step:.6f} | AvgLoss(unscaled)={avg_loss_unscaled:.6f} | "
                f"Recon(NAR)={avg_train_nar:.6f} | Recon(TF)={avg_train_tf:.6f} | "
                f"DC={comp_dc:.6f} | Spec={comp_spec:.6f}"
            )
        except Exception:
            pass
        try:
            if hasattr(self, "writer") and self.writer is not None:
                global_step = (epoch + 1) * max(1, num_batches)
                self.writer.add_scalar("Train/Recon_NAR", avg_train_nar, global_step)
                self.writer.add_scalar("Train/Recon_TF", avg_train_tf, global_step)
                if not math.isnan(comp_dc):
                    self.writer.add_scalar("Train/DC", comp_dc, global_step)
                if not math.isnan(comp_spec):
                    self.writer.add_scalar("Train/Spec", comp_spec, global_step)
        except Exception:
            pass
        try:
            self._last_train_loss_scaled = float(avg_loss)
            self._last_train_loss_unscaled = float(avg_loss_unscaled)
        except Exception:
            pass
        return avg_loss

    def test_epoch(self) -> dict[str, float]:
        """测试集评估"""
        self.logger.info("🧪 开始测试集评估...")
        # 获取当前模型（兼容ARWrapper和SequentialSpatiotemporalModel）
        model_to_test = self.get_model()
        model_to_test.eval()

        # 检查test_loader是否存在且不为None
        if not hasattr(self, "test_loader") or self.test_loader is None:
            self.logger.warning("⚠️ test_loader不存在或为None，跳过测试评估")
            return {"test_loss": 0.0, "test_metrics": {}}

        # -----------------------------------------------------------
        # [Bug Fix] DDP State Dict Key Mismatch Handling
        # -----------------------------------------------------------
        # 在测试开始前，确保模型权重正确加载。
        # 如果模型当前在DDP包装下（key有 module. 前缀），但checkpoint没有（或相反），会导致加载失败。
        # 虽然 load_checkpoint 已经尝试处理，但为了双重保险，这里进行一次运行时检查。
        # 特别是针对 "加载了 0/70 个参数" 的情况。

        try:
            # 获取模型参数
            model_params = dict(model_to_test.named_parameters())
            # 尝试获取检查点中的state_dict（如果 self.best_model_path 存在）
            ckpt_path = (
                self.best_model_path
                if hasattr(self, "best_model_path")
                and self.best_model_path
                and self.best_model_path.exists()
                else None
            )

            if ckpt_path:
                self.logger.info(f"🔍 验证模型权重加载状态: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint)

                # 检查key是否匹配（module. 前缀）
                model_keys = set(model_params.keys())
                ckpt_keys = set(state_dict.keys())

                # 情况1: 模型有module.，ckpt没有 -> 需要在ckpt加module. 或 模型去module.
                # 情况2: 模型无module.，ckpt有 -> 需要去ckpt的module.

                has_module_model = any(k.startswith("module.") for k in model_keys)
                has_module_ckpt = any(k.startswith("module.") for k in ckpt_keys)

                if has_module_model != has_module_ckpt:
                    self.logger.warning(
                        f"⚠️ 发现 DDP 前缀不匹配: Model(module={has_module_model}) vs Ckpt(module={has_module_ckpt})"
                    )
                    self.logger.warning("🔄 尝试重新加载权重以修复前缀问题...")

                    new_state_dict = {}
                    if has_module_model and not has_module_ckpt:
                        # 模型有module，ckpt没有 -> 给ckpt加
                        for k, v in state_dict.items():
                            new_state_dict[f"module.{k}"] = v
                    elif not has_module_model and has_module_ckpt:
                        # 模型无module，ckpt有 -> 去掉ckpt的module
                        for k, v in state_dict.items():
                            new_state_dict[k.replace("module.", "")] = v

                    # 重新加载
                missing, unexpected = model_to_test.load_state_dict(
                    new_state_dict, strict=False
                )
                # 计算加载成功的参数数量
                loaded_params = len(new_state_dict) - len(missing)
                total_params = len(new_state_dict)
                self.logger.info(
                    f"✅ 重新加载完成: missing={len(missing)}, unexpected={len(unexpected)}, loaded={loaded_params}/{total_params}"
                )

                # 如果missing全是module前缀差异导致的，其实是加载成功的
                if len(missing) > 0:
                    self.logger.warning(f"Missing keys: {missing[:5]}...")
            else:
                self.logger.info("✅ DDP前缀匹配，无需特殊处理")

        except Exception as e:
            self.logger.warning(f"⚠️ 权重验证过程中出错 (不影响继续测试): {e}")
            pass

        total_loss = 0.0
        all_metrics = []
        num_batches = len(self.test_loader)

        with torch.no_grad():
            try:
                import torch.distributed as dist

                is_primary = (
                    (not dist.is_available())
                    or (not dist.is_initialized())
                    or (dist.get_rank() == 0)
                )
            except Exception:
                is_primary = True

            # 使用 tqdm 显示进度
            iterator = self.test_loader
            if is_primary:
                iterator = tqdm(self.test_loader, desc="Testing", leave=False)

            for batch_idx, batch in enumerate(iterator):
                # 移动数据到设备
                input_seq = batch["input_sequence"].to(self.device)
                target_seq = batch["target_sequence"].to(self.device)

                # [Patch] 应用随机掩码（如果配置启用）- 确保测试集与训练任务一致
                crop_cfg = getattr(getattr(self.config, "training", None), "crop", None)
                masks_seq = None
                if crop_cfg and getattr(crop_cfg, "enabled", False):
                    # 测试阶段强制 patches_per_image=1
                    from omegaconf import OmegaConf

                    if OmegaConf.is_config(crop_cfg):
                        test_crop_cfg = OmegaConf.to_container(crop_cfg, resolve=True)
                    else:
                        test_crop_cfg = (
                            crop_cfg.copy() if hasattr(crop_cfg, "copy") else crop_cfg
                        )
                    if isinstance(test_crop_cfg, dict):
                        test_crop_cfg["patches_per_image"] = 1
                        test_crop_cfg = OmegaConf.create(test_crop_cfg)

                    input_seq, target_seq, masks_seq = self._apply_random_masking(
                        target_seq, test_crop_cfg
                    )

                    if isinstance(batch, dict):
                        batch["input_sequence"] = input_seq
                        batch["observed_lr_sequence"] = None
                        batch["mask_sequence"] = masks_seq

                # 模型预测（测试时不使用teacher forcing），输出长度与目标序列一致
                test_T_out = target_seq.shape[1]
                model = self.get_model()
                is_seq_model = hasattr(model, "spatial_forward") and hasattr(
                    model, "temporal_forward"
                )
                if is_seq_model:
                    model_output = model(input_seq, target_seq)
                    pred_seq = model_output["final_pred"]
                else:
                    try:
                        ar_enabled = bool(
                            getattr(self.config, "ar", {}).get("enabled", True)
                        )
                    except Exception:
                        ar_enabled = True
                    if ar_enabled:
                        if hasattr(model, "autoregressive_predict"):
                            pred_seq = model.autoregressive_predict(
                                input_seq, test_T_out, teacher=None, train_mode=False
                            )
                        else:
                            # [Fix] Apply degradation for raw spatial models in AR test path
                            model_input = input_seq
                            if not hasattr(model, "spatial_forward") and not hasattr(
                                model, "temporal_forward"
                            ):
                                # Assume raw spatial model taking one frame
                                x_single = input_seq[:, 0]
                                if (
                                    hasattr(self, "training_degradation_op")
                                    and self.training_degradation_op is not None
                                ):
                                    x_single = self.training_degradation_op(x_single)
                                elif (
                                    hasattr(self, "observation_op")
                                    and self.observation_op is not None
                                ):
                                    x_single = self.observation_op(x_single)

                                # Use degraded input for prediction
                                # Note: This only works if model expects single frame.
                                # If model expects sequence, we need more complex logic, but for UNet scan it's single frame.
                                model_input = x_single

                                # Handle channel padding if needed (copy from below)
                                raw_model = (
                                    model.module if hasattr(model, "module") else model
                                )
                                if hasattr(raw_model, "in_channels"):
                                    expected_in = raw_model.in_channels
                                    current_in = model_input.shape[1]
                                    if current_in < expected_in:
                                        pad_c = expected_in - current_in
                                        padding = torch.zeros(
                                            model_input.size(0),
                                            pad_c,
                                            model_input.size(2),
                                            model_input.size(3),
                                            device=model_input.device,
                                            dtype=model_input.dtype,
                                        )
                                        model_input = torch.cat(
                                            [model_input, padding], dim=1
                                        )

                                pred_seq = (
                                    model(model_input, test_T_out)
                                    if "test_T_out"
                                    in model.forward.__code__.co_varnames
                                    else model(model_input)
                                )
                                # Ensure 5D output
                                if pred_seq.dim() == 4:
                                    pred_seq = pred_seq.unsqueeze(1)
                            else:
                                pred_seq = model(input_seq, test_T_out)
                    else:
                        if (
                            isinstance(batch, dict)
                            and ("observed_lr_sequence" in batch)
                            and (batch["observed_lr_sequence"] is not None)
                        ):
                            lr_seq = batch["observed_lr_sequence"]
                            x_single = lr_seq[:, 0]
                            try:
                                if ("coords_lr_sequence" in batch) and (
                                    batch["coords_lr_sequence"] is not None
                                ):
                                    coords_lr = batch["coords_lr_sequence"][:, 0]
                                    x_single = torch.cat([x_single, coords_lr], dim=1)
                                if ("mask_lr_sequence" in batch) and (
                                    batch["mask_lr_sequence"] is not None
                                ):
                                    mask_lr = batch["mask_lr_sequence"][:, 0]
                                    x_single = torch.cat([x_single, mask_lr], dim=1)
                            except Exception:
                                pass
                        else:
                            x_single = input_seq[:, 0]
                            # Apply degradation (Validation)
                            if (
                                hasattr(self, "training_degradation_op")
                                and self.training_degradation_op is not None
                            ):
                                x_single = self.training_degradation_op(x_single)
                            elif (
                                hasattr(self, "observation_op")
                                and self.observation_op is not None
                            ):
                                x_single = self.observation_op(x_single)
                            try:
                                if ("coords_sequence" in batch) and (
                                    batch["coords_sequence"] is not None
                                ):
                                    coords_hr = batch["coords_sequence"][:, 0]
                                    x_single = torch.cat([x_single, coords_hr], dim=1)
                                if ("mask_sequence" in batch) and (
                                    batch["mask_sequence"] is not None
                                ):
                                    mask_hr = batch["mask_sequence"][:, 0]
                                    x_single = torch.cat([x_single, mask_hr], dim=1)
                            except Exception:
                                pass
                        if hasattr(model, "spatial_forward"):
                            y_single = model.spatial_forward(x_single).spatial_pred
                        else:
                            try:
                                device = next(model.parameters()).device
                                dtype = next(model.parameters()).dtype
                                x_single = x_single.to(device=device, dtype=dtype)
                            except Exception:
                                pass

                            # 自动适配输入通道数
                            raw_model = (
                                model.module if hasattr(model, "module") else model
                            )
                            if hasattr(raw_model, "in_channels"):
                                expected_in = raw_model.in_channels
                                current_in = x_single.shape[1]
                                if current_in < expected_in:
                                    pad_c = expected_in - current_in
                                    padding = torch.zeros(
                                        x_single.size(0),
                                        pad_c,
                                        x_single.size(2),
                                        x_single.size(3),
                                        device=x_single.device,
                                        dtype=x_single.dtype,
                                    )
                                    x_single = torch.cat([x_single, padding], dim=1)

                            y_single = model(x_single)

                        # [Patch] Auto-interpolate for UNet SR in Test
                        if y_single.shape[-2:] != target_seq.shape[-2:]:
                            y_single = torch.nn.functional.interpolate(
                                y_single,
                                size=target_seq.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )

                        pred_seq = y_single[:, None]

                # 计算损失（与训练/验证口径一致：Rel-L2 + MAE）
                from ops.losses import l1_mae, rel_l2

                loss = rel_l2(pred_seq, target_seq) + l1_mae(pred_seq, target_seq)
                total_loss += loss.item()

                # 计算详细指标 - GPU优化版本
                try:
                    # 使用GPU优化的指标计算，避免CPU转移
                    from utils.metrics import compute_all_metrics

                    observation_seq = None
                    pred_obs_seq = None
                    try:
                        if (
                            hasattr(self, "observation_op")
                            and self.observation_op is not None
                            and hasattr(self, "norm_stats")
                            and self.norm_stats is not None
                        ):
                            B, T, C, H, W = target_seq.shape
                            m = self.norm_stats.get(
                                "mean",
                                self.norm_stats.get(
                                    "data_mean",
                                    self.norm_stats.get("u_mean", torch.tensor(0.0)),
                                ),
                            )
                            s = self.norm_stats.get(
                                "std",
                                self.norm_stats.get(
                                    "data_std",
                                    self.norm_stats.get("u_std", torch.tensor(1.0)),
                                ),
                            )
                            m_t = torch.as_tensor(m, device=self.device).reshape(-1)
                            s_t = torch.as_tensor(s, device=self.device).reshape(-1)
                            if m_t.numel() == 1 and C > 1:
                                m_t = m_t.repeat(C)
                            if s_t.numel() == 1 and C > 1:
                                s_t = s_t.repeat(C)
                            mean_t = m_t.reshape(1, 1, C, 1, 1)
                            std_t = s_t.reshape(1, 1, C, 1, 1)

                            gt_orig = target_seq * std_t + mean_t
                            gt_flat = gt_orig.reshape(B * T, C, H, W)
                            obs_flat = self.observation_op(gt_flat)
                            oh, ow = obs_flat.shape[-2:]
                            observation_seq = obs_flat.reshape(B, T, C, oh, ow)

                            if pred_seq.dim() == 5:
                                pred_orig = pred_seq * std_t + mean_t
                                pred_flat = pred_orig.reshape(B * T, C, H, W)
                                pred_obs_flat = self.observation_op(pred_flat)
                                pred_obs_seq = pred_obs_flat.reshape(B, T, C, oh, ow)
                    except Exception:
                        observation_seq = None
                        pred_obs_seq = None

                    obs_data = None
                    obs_last = None
                    if observation_seq is not None:
                        if observation_seq.dim() == 5:
                            obs_last = observation_seq[:, -1]
                        elif observation_seq.dim() == 4:
                            obs_last = observation_seq

                        baseline_seq = observation_seq
                        baseline_last = obs_last

                        obs_data = {
                            "y": obs_last,
                            "observation_seq": observation_seq,
                            "observation": obs_last,
                            "pred_obs": pred_obs_seq,
                            "baseline_seq": baseline_seq,
                            "baseline": baseline_last,
                            "baseline_type": "H_gt",
                            "h_params": self.h_params,
                        }

                    if batch_idx == 0:
                        self.logger.info("🔍 First Batch Shapes (Test):")
                        self.logger.info(f"  GT (Target): {target_seq.shape}")
                        if pred_seq is not None:
                            self.logger.info(f"  Pred: {pred_seq.shape}")
                        if obs_last is not None:
                            self.logger.info(f"  Obs (Last): {obs_last.shape}")
                        else:
                            self.logger.info("  Obs (Last): None")
                        if pred_obs_seq is not None:
                            self.logger.info(f"  Pred Obs: {pred_obs_seq.shape}")

                    # [Fix] 传递 crop_params
                    try:
                        crop_params = getattr(
                            self.config.data.observation, "crop", None
                        )
                    except Exception:
                        crop_params = None

                    image_size = target_seq.shape[-2:]
                    batch_metrics_dict = compute_all_metrics(
                        pred_seq,
                        target_seq,
                        obs_data=obs_data,
                        norm_stats=self.norm_stats,
                        image_size=image_size,
                        include_freq_metrics=True,
                        crop_params=crop_params,
                    )

                    batch_metrics = {}
                    for k, v in batch_metrics_dict.items():
                        if isinstance(v, torch.Tensor):
                            batch_metrics[k] = float(v.mean().item())
                        else:
                            batch_metrics[k] = float(v)
                    batch_metrics["rel_l2_domain"] = "zscore"

                    all_metrics.append(batch_metrics)

                except Exception as e:
                    self.logger.error(
                        f"compute_all_metrics failed in test. Pred shape: {pred_seq.shape}, Target shape: {target_seq.shape}"
                    )
                    self.logger.error(f"Error details: {e}")
                    raise e

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
                            elif isinstance(metric_val, (int, float)):
                                # 如果已经是数值标量，直接使用
                                values.append(float(metric_val))
                            elif isinstance(metric_val, str):
                                # 字符串指标（如域标记）不参与数值聚合，跳过
                                continue
                            else:
                                # 其它类型尝试提取可数值化内容，否则跳过
                                try:
                                    values.append(float(metric_val))
                                except Exception:
                                    continue
                        except Exception as e:
                            self.logger.warning(f"处理指标 {key} 时出错: {e}")
                            continue

                    # 计算所有批次的平均值
                    if values:
                        final_metrics[key] = np.mean(values)
                    else:
                        # 非数值型指标（如 *_domain 标签）不应聚合，也不记录警告
                        try:
                            if isinstance(
                                batch_metrics.get(key, None), str
                            ) or key.endswith("_domain"):
                                pass
                            else:
                                self.logger.warning(f"指标 {key} 没有有效值")
                        except Exception:
                            self.logger.warning(f"指标 {key} 没有有效值")

            except Exception as e:
                self.logger.error(f"指标聚合失败: {e}")
                final_metrics = {"error": "metrics_aggregation_failed"}

        final_metrics["test_loss"] = avg_loss

        self.logger.info(f"✅ 测试完成 - 损失: {avg_loss:.6f}")
        for key, value in final_metrics.items():
            if key != "test_loss":
                self.logger.info(f"  {key}: {value:.6f}")

        return final_metrics

    def validate_epoch(self, epoch: int) -> tuple[float, dict[str, float], dict | None]:
        """验证一个epoch（聚合损失分量并健壮处理空验证集）"""
        self.get_model().eval()
        total_loss = 0.0
        total_loss_tf = 0.0
        total_loss_nar = 0.0
        all_metrics = []
        loss_components_list = []
        # 兜底处理：val_loader可能为None或长度不可用
        try:
            num_batches = len(self.val_loader) if self.val_loader is not None else 0
        except Exception:
            num_batches = 0
        sample_batch = None
        denom_vals = []
        err_vals = []

        # 获取当前T_out
        current_T_out = self.get_current_T_out(epoch)

        # 若无有效val_loader，直接返回训练损失的占位与空指标
        if num_batches == 0:
            self.logger.warning(
                "验证加载器不可用（None或空），跳过验证阶段，不更新最佳checkpoint"
            )
            # Fix C: Return nan to indicate no valid validation occurred
            return float("nan"), {}, None

        with torch.no_grad():
            try:
                import torch.distributed as dist

                is_primary = (
                    (not dist.is_available())
                    or (not dist.is_initialized())
                    or (dist.get_rank() == 0)
                )
            except Exception:
                is_primary = True
            val_iter = self.val_loader
            for batch_idx, batch in enumerate(val_iter):
                # 移动数据到设备
                input_seq = batch["input_sequence"].to(
                    self.device, non_blocking=True
                )  # [B, T_in, C, H, W]
                target_seq = batch["target_sequence"].to(
                    self.device, non_blocking=True
                )  # [B, T_out, C, H, W]

                # [Patch] 应用随机掩码（如果配置启用）- 确保验证集与训练任务一致
                crop_cfg = getattr(getattr(self.config, "training", None), "crop", None)
                masks_seq = None
                if crop_cfg and getattr(crop_cfg, "enabled", False):
                    # 验证阶段强制 patches_per_image=1 以节省显存
                    from omegaconf import OmegaConf

                    if OmegaConf.is_config(crop_cfg):
                        val_crop_cfg = OmegaConf.to_container(crop_cfg, resolve=True)
                    else:
                        val_crop_cfg = (
                            crop_cfg.copy() if hasattr(crop_cfg, "copy") else crop_cfg
                        )
                    if isinstance(val_crop_cfg, dict):
                        val_crop_cfg["patches_per_image"] = 1
                        val_crop_cfg = OmegaConf.create(
                            val_crop_cfg
                        )  # Wrap back for getattr access in _apply

                    input_seq, target_seq, masks_seq = self._apply_random_masking(
                        target_seq, val_crop_cfg
                    )

                    if isinstance(batch, dict):
                        batch["input_sequence"] = input_seq
                        # batch['target_sequence'] = target_seq
                        batch["observed_lr_sequence"] = None
                        batch["mask_sequence"] = masks_seq
                        # 确保 sample_batch 更新，以便 create_visualizations 使用正确的输入
                        if batch_idx == num_batches - 1:
                            sample_batch = batch

                # 根据课程学习调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]
                # 根据配置分支：空间-only 与 AR
                try:
                    ar_enabled = bool(
                        getattr(self.config, "ar", {}).get("enabled", True)
                    )
                except Exception:
                    ar_enabled = True
                try:
                    _m = self.get_model()
                    _is_seq = hasattr(_m, "spatial_forward") and hasattr(
                        _m, "temporal_forward"
                    )
                    if _is_seq:
                        ar_enabled = True
                except Exception:
                    pass

                use_amp = bool(
                    getattr(getattr(self, "config", None), "training", None)
                    and getattr(self.config.training.amp, "enabled", False)
                ) and (self.device.type == "cuda")
                amp_ctx = (
                    autocast(
                        device_type="cuda",
                        dtype=getattr(self, "autocast_dtype", torch.bfloat16),
                        enabled=use_amp,
                    )
                    if use_amp
                    else None
                )
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
                        if (
                            isinstance(batch, dict)
                            and ("observed_lr_sequence" in batch)
                            and (batch["observed_lr_sequence"] is not None)
                        ):
                            lr_seq = batch["observed_lr_sequence"]
                            x_single = lr_seq[:, 0]
                            try:
                                if ("coords_lr_sequence" in batch) and (
                                    batch["coords_lr_sequence"] is not None
                                ):
                                    coords_lr = batch["coords_lr_sequence"][:, 0]
                                    x_single = torch.cat([x_single, coords_lr], dim=1)
                                if ("mask_lr_sequence" in batch) and (
                                    batch["mask_lr_sequence"] is not None
                                ):
                                    mask_lr = batch["mask_lr_sequence"][:, 0]
                                    x_single = torch.cat([x_single, mask_lr], dim=1)
                            except Exception:
                                pass
                        else:
                            x_single = input_seq[:, 0]
                            # Apply degradation (Validation/Testing - Spatial Only)
                            if (
                                hasattr(self, "training_degradation_op")
                                and self.training_degradation_op is not None
                            ):
                                x_single = self.training_degradation_op(x_single)
                            elif (
                                hasattr(self, "observation_op")
                                and self.observation_op is not None
                            ):
                                x_single = self.observation_op(x_single)
                            try:
                                if ("coords_sequence" in batch) and (
                                    batch["coords_sequence"] is not None
                                ):
                                    coords_hr = batch["coords_sequence"][:, 0]
                                    x_single = torch.cat([x_single, coords_hr], dim=1)
                                if ("mask_sequence" in batch) and (
                                    batch["mask_sequence"] is not None
                                ):
                                    mask_hr = batch["mask_sequence"][:, 0]
                                    x_single = torch.cat([x_single, mask_hr], dim=1)
                            except Exception:
                                pass
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
                        if hasattr(model, "spatial_forward"):
                            # SequentialSpatiotemporalModel模式
                            spatial_output = model.spatial_forward(x_single)
                            y_single = spatial_output.spatial_pred
                        else:
                            # 传统模型模式
                            try:
                                device = next(model.parameters()).device
                                dtype = next(model.parameters()).dtype
                                x_single = x_single.to(device=device, dtype=dtype)
                            except Exception:
                                pass

                            # 自动适配输入通道数
                            raw_model = (
                                model.module if hasattr(model, "module") else model
                            )
                            if hasattr(raw_model, "in_channels"):
                                expected_in = raw_model.in_channels
                                current_in = x_single.shape[1]
                                if current_in < expected_in:
                                    pad_c = expected_in - current_in
                                    padding = torch.zeros(
                                        x_single.size(0),
                                        pad_c,
                                        x_single.size(2),
                                        x_single.size(3),
                                        device=x_single.device,
                                        dtype=x_single.dtype,
                                    )
                                    x_single = torch.cat([x_single, padding], dim=1)

                            y_single = model(x_single)
                        target_single = target_seq[:, 0]
                        # [Patch] Auto-interpolate if output size mismatches target (for UNet SR)
                        if y_single.shape[-2:] != target_single.shape[-2:]:
                            y_single = torch.nn.functional.interpolate(
                                y_single,
                                size=target_single.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        obs_data_single = {
                            "observation": None,
                            "baseline": x_single,
                            "h_params": self.h_params,
                        }
                        from ops.losses import compute_total_loss

                        losses = compute_total_loss(
                            pred_z=y_single,
                            target_z=target_single,
                            obs_data=obs_data_single,
                            norm_stats=self.norm_stats,
                            config=self.config,
                        )
                        from ops.losses import l1_mae, rel_l2

                        base_rel = rel_l2(y_single, target_single)
                        base_mae = l1_mae(y_single, target_single)
                        base_loss = base_rel + base_mae
                        loss_components_list.append(
                            {
                                "reconstruction_loss": base_loss.item(),
                                "rel_l2": base_rel.item(),
                                "mae": base_mae.item(),
                            }
                        )
                    total_loss += base_loss.item()
                    total_loss_nar += base_loss.item()

                else:
                    # AR验证路径：与原逻辑一致
                    with amp_ctx:
                        # 使用专用时序模型或ARWrapper进行训练预测
                        model = self.get_model()
                        if hasattr(model, "spatial_forward") and hasattr(
                            model, "temporal_forward"
                        ):
                            # 使用teacher forcing路径以确保预测长度与target一致
                            mo_tf = model(input_seq, target_seq)
                            pred_seq = mo_tf["final_pred"]
                            pred_seq_tf = pred_seq
                        else:
                            # 统一使用teacher forcing以避免T_out不匹配

                            # 针对裸空间模型（如UNet）适配 5D -> 4D 及通道 Padding
                            # 注意：input_seq是5D [B, T, C, H, W]
                            model_input = input_seq
                            is_raw_spatial = False

                            if input_seq.dim() == 5:
                                raw_m = (
                                    model.module if hasattr(model, "module") else model
                                )
                                has_seq_method = (
                                    hasattr(raw_m, "autoregressive_predict")
                                    or hasattr(raw_m, "temporal_forward")
                                    or (
                                        hasattr(raw_m, "is_temporal")
                                        and raw_m.is_temporal
                                    )
                                )

                                if not has_seq_method:
                                    model_input = input_seq[:, 0]
                                    is_raw_spatial = True

                                    # [Fix] Apply degradation for raw spatial models in AR path
                                    if (
                                        hasattr(self, "training_degradation_op")
                                        and self.training_degradation_op is not None
                                    ):
                                        model_input = self.training_degradation_op(
                                            model_input
                                        )
                                    elif (
                                        hasattr(self, "observation_op")
                                        and self.observation_op is not None
                                    ):
                                        model_input = self.observation_op(model_input)

                                    if hasattr(raw_m, "in_channels"):
                                        exp_in = raw_m.in_channels
                                        if model_input.shape[1] < exp_in:
                                            pad_c = exp_in - model_input.shape[1]
                                            padding = torch.zeros(
                                                model_input.size(0),
                                                pad_c,
                                                model_input.size(2),
                                                model_input.size(3),
                                                device=model_input.device,
                                                dtype=model_input.dtype,
                                            )
                                            model_input = torch.cat(
                                                [model_input, padding], dim=1
                                            )

                            if is_raw_spatial:
                                out = model(model_input)
                            else:
                                out = model(input_seq, target_seq)

                            # 兼容性处理：ARWrapper 可能直接返回 Tensor
                            if isinstance(out, dict):
                                pred_seq = out["final_pred"]
                            else:
                                pred_seq = out

                            if pred_seq.dim() == 4:
                                pred_seq = pred_seq.unsqueeze(1)

                            pred_seq_tf = pred_seq

                        # 计算观测序列与预测观测（用于DC Loss）
                        observation_seq = None
                        pred_obs_seq = None
                        try:
                            if (
                                hasattr(self, "observation_op")
                                and self.observation_op is not None
                                and hasattr(self, "norm_stats")
                                and self.norm_stats is not None
                            ):
                                B, T, C, H, W = target_seq.shape
                                m = self.norm_stats.get(
                                    "data_mean",
                                    self.norm_stats.get("u_mean", torch.tensor(0.0)),
                                )
                                s = self.norm_stats.get(
                                    "data_std",
                                    self.norm_stats.get("u_std", torch.tensor(1.0)),
                                )
                                mean_t = torch.as_tensor(m, device=self.device).reshape(
                                    1, 1, -1, 1, 1
                                )
                                std_t = torch.as_tensor(s, device=self.device).reshape(
                                    1, 1, -1, 1, 1
                                )

                                # 生成 observation_seq
                                gt_orig = target_seq * std_t + mean_t
                                gt_flat = gt_orig.reshape(B * T, C, H, W)
                                obs_flat = self.observation_op(gt_flat)
                                oh, ow = obs_flat.shape[-2:]
                                observation_seq = obs_flat.reshape(B, T, C, oh, ow)

                                # 生成 pred_obs_seq
                                if pred_seq.dim() == 5:
                                    pred_orig = pred_seq * std_t + mean_t
                                    pred_flat = pred_orig.reshape(B * T, C, H, W)
                                    pred_obs_flat = self.observation_op(pred_flat)
                                    pred_obs_seq = pred_obs_flat.reshape(
                                        B, T, C, oh, ow
                                    )
                        except Exception:
                            observation_seq = None
                            pred_obs_seq = None

                        # 构造完整的 obs_data
                        obs_last = None
                        if observation_seq is not None:
                            if observation_seq.dim() == 5:
                                obs_last = observation_seq[:, -1]
                            elif observation_seq.dim() == 4:
                                obs_last = observation_seq

                        # Baseline 必须等于 H(gt)
                        baseline_seq = observation_seq
                        baseline_last = obs_last

                        obs_data = {
                            "y": obs_last,  # [B, C, h, w] Real Observation
                            "observation_seq": observation_seq,
                            "observation": obs_last,
                            "pred_obs": pred_obs_seq,
                            "baseline_seq": baseline_seq,
                            "baseline": baseline_last,
                            "baseline_type": "H_gt",
                            "h_params": self.h_params,
                        }

                        # 首次batch打印形状 (Validation)
                        if batch_idx == 0:
                            self.logger.info("🔍 First Batch Shapes (Validation):")
                            self.logger.info(f"  GT (Target): {target_seq.shape}")
                            if pred_seq is not None:
                                self.logger.info(f"  Pred: {pred_seq.shape}")
                            if obs_last is not None:
                                self.logger.info(f"  Obs (Last): {obs_last.shape}")
                            else:
                                self.logger.info("  Obs (Last): None")
                            if pred_obs_seq is not None:
                                self.logger.info(f"  Pred Obs: {pred_obs_seq.shape}")

                        losses = compute_ar_total_loss(
                            pred_seq=pred_seq,
                            gt_seq=target_seq,
                            obs_data=obs_data,
                            norm_stats=self.norm_stats,
                            config=self.config,
                        )
                        losses_tf = compute_ar_total_loss(
                            pred_seq=pred_seq_tf,
                            gt_seq=target_seq,
                            obs_data=obs_data,
                            norm_stats=self.norm_stats,
                            config=self.config,
                        )
                        # AR Evaluation Strategy (last/mean)
                        try:
                            eval_strategy = "mean"
                            try:
                                # New consolidated eval logic
                                if hasattr(self.config, "ar") and hasattr(
                                    self.config.ar, "eval_time_strategy"
                                ):
                                    eval_strategy = str(
                                        self.config.ar.eval_time_strategy
                                    ).lower()
                                else:
                                    eval_strategy = "mean"

                                p_eval = pred_seq
                                t_eval = target_seq
                                if eval_strategy == "last":
                                    p_eval = pred_seq[:, -1:]
                                    t_eval = target_seq[:, -1:]

                                from ops.losses import l1_mae, rel_l2

                                base_rel = rel_l2(p_eval, t_eval)
                                base_mae = l1_mae(p_eval, t_eval)

                                # Always use recomputed metrics for consistent reporting
                                add_rel = base_rel.item()
                                add_mae = base_mae.item()

                                # Update base_loss if it was missing or if we want to reflect the eval strategy
                                base_loss = losses.get("reconstruction_loss", None)
                                if base_loss is None or not torch.is_tensor(base_loss):
                                    base_loss = base_rel + base_mae

                            except Exception as _eval_err:
                                self.logger.warning(f"Metric calc failed: {_eval_err}")
                                add_rel = 0.0
                                add_mae = 0.0
                                if "base_loss" not in locals():
                                    base_loss = torch.tensor(0.0, device=self.device)
                                else:
                                    add_rel = float(
                                        losses.get("rel2_loss", float("nan"))
                                    )
                                    add_mae = float(
                                        losses.get("mae_loss", float("nan"))
                                    )
                        except Exception:
                            from ops.losses import l1_mae, rel_l2

                            # Fallback re-evaluation
                            base_rel = rel_l2(pred_seq, target_seq)
                            base_mae = l1_mae(pred_seq, target_seq)
                            base_loss = base_rel + base_mae
                            add_rel = base_rel.item()
                            add_mae = base_mae.item()
                        loss_components_list.append(
                            {
                                "reconstruction_loss": (
                                    float(base_loss.item())
                                    if torch.is_tensor(base_loss)
                                    else float(base_loss)
                                ),
                                "rel_l2": add_rel,
                                "mae": add_mae,
                                "dc_loss": (
                                    float(losses.get("dc_loss", 0.0))
                                    if torch.is_tensor(losses.get("dc_loss", 0.0))
                                    else float(losses.get("dc_loss", 0.0))
                                ),
                                "spectral_loss": (
                                    float(losses.get("spectral_loss", 0.0))
                                    if torch.is_tensor(losses.get("spectral_loss", 0.0))
                                    else float(losses.get("spectral_loss", 0.0))
                                ),
                            }
                        )
                        try:
                            d = torch.sqrt((target_seq**2).sum(dim=(2, 3, 4))).mean(
                                dim=1
                            )
                            e = torch.sqrt(
                                ((pred_seq - target_seq) ** 2).sum(dim=(2, 3, 4))
                            ).mean(dim=1)
                            denom_vals.append(d.detach().cpu().numpy())
                            err_vals.append(e.detach().cpu().numpy())
                        except Exception:
                            pass
                    total_loss += base_loss.item()
                    total_loss_nar += base_loss.item()
                    try:
                        base_loss_tf = losses_tf.get("reconstruction_loss", None)
                        if base_loss_tf is None or not torch.is_tensor(base_loss_tf):
                            from ops.losses import l1_mae, rel_l2

                            _rel_tf = rel_l2(pred_seq_tf, target_seq)
                            _mae_tf = l1_mae(pred_seq_tf, target_seq)
                            base_loss_tf = _rel_tf + _mae_tf
                        total_loss_tf += (
                            float(base_loss_tf.item())
                            if torch.is_tensor(base_loss_tf)
                            else float(base_loss_tf)
                        )
                    except Exception:
                        pass

                # IO Debug: 验证期可视化输入/输出（受开关与步频控制）
                try:
                    if bool(self._io_debug_cfg.get("enabled", False)):
                        step_interval = int(
                            self._io_debug_cfg.get("val_every_n_steps", 50)
                        )
                        if step_interval > 0 and ((batch_idx + 1) % step_interval == 0):
                            viz_root = (
                                self.output_dir
                                / "io_debug"
                                / f"epoch_{epoch+1:03d}"
                                / "val"
                            )
                            viz_root.mkdir(parents=True, exist_ok=True)
                            max_t = int(self._io_debug_cfg.get("max_time_steps", 4))
                            try:
                                from utils.ar_visualizer import ARTrainingVisualizer

                                dbg_viz = ARTrainingVisualizer(str(viz_root))
                                ps = pred_seq.detach().cpu()
                                ts = target_seq.detach().cpu()
                                if ps.shape[1] > max_t:
                                    ps = ps[:, :max_t]
                                if ts.shape[1] > max_t:
                                    ts = ts[:, :max_t]
                                dbg_viz.visualize_obs_gt_pred_error(
                                    ts,
                                    ps,
                                    save_name=f"b{batch_idx+1:05d}_obs_gt_pred_error",
                                    norm_stats=self.norm_stats,
                                )
                                dbg_viz.create_temporal_analysis(
                                    ps.numpy(),
                                    ts.numpy(),
                                    save_name=f"b{batch_idx+1:05d}_temporal_analysis",
                                    norm_stats=self.norm_stats,
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

        # 聚合指标
        avg_loss = total_loss / max(1, num_batches)
        avg_loss_nar = total_loss_nar / max(1, num_batches)
        avg_loss_tf = total_loss_tf / max(1, num_batches)

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

            # 诊断一致性差值的分布（last/seq）
            try:

                def _dist_stats(vals):
                    vals = np.array(vals, dtype=np.float64)
                    return {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "q25": float(np.quantile(vals, 0.25)),
                        "q50": float(np.quantile(vals, 0.50)),
                        "q75": float(np.quantile(vals, 0.75)),
                    }

                last_vals = [
                    float(m["diff_last_rel_l2"])
                    for m in all_metrics
                    if "diff_last_rel_l2" in m
                ]
                seq_vals = [
                    float(m["diff_seq_rel_l2"])
                    for m in all_metrics
                    if "diff_seq_rel_l2" in m
                ]
                if last_vals:
                    s = _dist_stats(last_vals)
                    final_metrics["diff_last_rel_l2_mean"] = s["mean"]
                    final_metrics["diff_last_rel_l2_std"] = s["std"]
                    final_metrics["diff_last_rel_l2_q25"] = s["q25"]
                    final_metrics["diff_last_rel_l2_q50"] = s["q50"]
                    final_metrics["diff_last_rel_l2_q75"] = s["q75"]
                    try:
                        self.logger.info(
                            f"DiffLast rel_l2: mean={s['mean']:.6f} std={s['std']:.6f} q25={s['q25']:.6f} q50={s['q50']:.6f} q75={s['q75']:.6f}"
                        )
                    except Exception:
                        pass
                if seq_vals:
                    s = _dist_stats(seq_vals)
                    final_metrics["diff_seq_rel_l2_mean"] = s["mean"]
                    final_metrics["diff_seq_rel_l2_std"] = s["std"]
                    final_metrics["diff_seq_rel_l2_q25"] = s["q25"]
                    final_metrics["diff_seq_rel_l2_q50"] = s["q50"]
                    final_metrics["diff_seq_rel_l2_q75"] = s["q75"]
                    try:
                        self.logger.info(
                            f"DiffSeq rel_l2:  mean={s['mean']:.6f} std={s['std']:.6f} q25={s['q25']:.6f} q50={s['q50']:.6f} q75={s['q75']:.6f}"
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        # 聚合损失分量
        if loss_components_list:
            try:
                for k in [
                    "dc_loss",
                    "spectral_loss",
                    "reconstruction_loss",
                    "rel_l2",
                    "mae",
                ]:
                    vals = [
                        d.get(k) for d in loss_components_list if d.get(k) is not None
                    ]
                    if vals:
                        final_metrics[k] = float(np.mean(vals))

                # 显式记录验证集的 DC 和 Spectral 损失到 TensorBoard
                if hasattr(self, "writer") and self.writer is not None:
                    if "dc_loss" in final_metrics:
                        self.writer.add_scalar(
                            "Val/DC", final_metrics["dc_loss"], self.global_step
                        )
                    if "spectral_loss" in final_metrics:
                        self.writer.add_scalar(
                            "Val/Spec", final_metrics["spectral_loss"], self.global_step
                        )
            except Exception:
                pass

        final_metrics["val_loss"] = avg_loss_nar
        final_metrics["val_loss_nar"] = avg_loss_nar
        final_metrics["val_loss_tf"] = avg_loss_tf

        # If fallback re-evaluation was triggered (e.g. no loss_components_list),
        # ensure we still have rel_l2 and mae in final_metrics if possible.
        if "rel_l2" not in final_metrics and "add_rel" in locals():
            final_metrics["rel_l2"] = add_rel
        if "mae" not in final_metrics and "add_mae" in locals():
            final_metrics["mae"] = add_mae

        try:
            if denom_vals and err_vals:
                import numpy as _np

                dv = _np.concatenate(denom_vals, axis=0)
                ev = _np.concatenate(err_vals, axis=0)
                final_metrics["den_mean"] = float(_np.mean(dv))
                final_metrics["den_std"] = float(_np.std(dv))
                final_metrics["den_q25"] = float(_np.quantile(dv, 0.25))
                final_metrics["den_q50"] = float(_np.quantile(dv, 0.50))
                final_metrics["den_q75"] = float(_np.quantile(dv, 0.75))
                final_metrics["err_mean"] = float(_np.mean(ev))
                final_metrics["err_std"] = float(_np.std(ev))
                final_metrics["err_q25"] = float(_np.quantile(ev, 0.25))
                final_metrics["err_q50"] = float(_np.quantile(ev, 0.50))
                final_metrics["err_q75"] = float(_np.quantile(ev, 0.75))
                try:
                    self.logger.info(
                        f"ScaleStats den_mean={final_metrics['den_mean']:.6f} den_std={final_metrics['den_std']:.6f} err_mean={final_metrics['err_mean']:.6f} err_std={final_metrics['err_std']:.6f}"
                    )
                except Exception:
                    pass
        except Exception:
            pass

        return avg_loss, final_metrics, sample_batch

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点（CPU offload + 原子写 + 仅主进程）"""
        # DDP: 仅在rank 0保存
        try:
            if (
                getattr(self, "distributed", False)
                and dist.is_initialized()
                and dist.get_rank() != 0
            ):
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
            model_state_cpu = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }
        except Exception:
            model = self.get_model()
            model_state_cpu = model.state_dict()
        opt_state = (
            self.optimizer.state_dict()
            if hasattr(self, "optimizer") and self.optimizer is not None
            else {}
        )
        sch_state = (
            self.scheduler.state_dict()
            if hasattr(self, "scheduler") and self.scheduler is not None
            else {}
        )
        scl_state = (
            self.scaler.state_dict()
            if hasattr(self, "scaler") and self.scaler is not None
            else {}
        )
        try:
            opt_state = _move_to_cpu(opt_state)
            sch_state = _move_to_cpu(sch_state)
            scl_state = _move_to_cpu(scl_state)
        except Exception:
            pass
        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": model_state_cpu,
            "optimizer_state_dict": opt_state,
            "scheduler_state_dict": sch_state,
            "scaler_state_dict": scl_state,
            "best_val_loss": float(self.best_val_loss),
            "config": OmegaConf.to_yaml(self.config),
            "training_history": self.training_history,
        }

        # 读取检查点策略
        ck_cfg = getattr(self.config.training, "checkpoint", None)
        save_last = True if ck_cfg is None else bool(getattr(ck_cfg, "save_last", True))
        save_best = True if ck_cfg is None else bool(getattr(ck_cfg, "save_best", True))
        save_every = (
            int(getattr(ck_cfg, "save_every_n_epochs", 0) or 0)
            if ck_cfg is not None
            else 0
        )
        max_keep = int(getattr(ck_cfg, "max_keep", 2) or 2) if ck_cfg is not None else 2

        # 保存函数：原子写
        def _atomic_save(obj, path: Path):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = Path(str(path) + ".tmp")
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
            _atomic_save(checkpoint, self.output_dir / "last.ckpt")
            write_times["last_ckpt_ms"] = (time.perf_counter() - w0) * 1000.0

        # 保存最佳检查点
        if save_best and is_best:
            w0 = time.perf_counter()
            _atomic_save(checkpoint, self.output_dir / "best.ckpt")
            write_times["best_ckpt_ms"] = (time.perf_counter() - w0) * 1000.0
            self.logger.info(f"💾 保存最佳模型 (验证损失: {self.best_val_loss:.6f})")

        # 周期性保存
        if save_every > 0 and ((epoch + 1) % save_every == 0):
            ep_path = self.output_dir / f"epoch_{epoch+1:04d}.ckpt"
            w0 = time.perf_counter()
            _atomic_save(checkpoint, ep_path)
            write_times["periodic_ckpt_ms"] = (time.perf_counter() - w0) * 1000.0

        # 保留最近 max_keep 个周期检查点
        try:
            ep_ckpts = sorted(list(self.output_dir.glob("epoch_*.ckpt")))
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
            if "checkpoint_times_ms" not in self.training_history:
                self.training_history["checkpoint_times_ms"] = []
            write_times["total_ckpt_ms"] = (time.perf_counter() - t0) * 1000.0
            write_times["epoch"] = int(epoch)
            self.training_history["checkpoint_times_ms"].append(write_times)
        except Exception:
            pass

    def generate_resource_summary(self):
        """汇总资源指标，生成 JSON 与 Markdown 报告"""
        import json

        epoch_file = self.output_dir / "resources_epoch.jsonl"
        metrics_file = self.output_dir / "resource_metrics.jsonl"
        summary = {
            "epochs": 0,
            "avg_throughput_samples_per_sec": 0.0,
            "max_gpu_peak_allocated_gb": 0.0,
            "max_gpu_peak_reserved_gb": 0.0,
            "avg_epoch_time_sec": 0.0,
        }
        try:
            throughputs, times, peak_allocs, peak_resv = [], [], [], []
            if epoch_file.exists():
                with open(epoch_file) as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            throughputs.append(
                                float(rec.get("throughput_samples_per_sec", 0.0))
                            )
                            times.append(float(rec.get("time_sec", 0.0)))
                            peak_allocs.append(
                                float(rec.get("gpu_peak_allocated_gb", 0.0))
                            )
                            peak_resv.append(
                                float(rec.get("gpu_peak_reserved_gb", 0.0))
                            )
                        except Exception:
                            continue
            if throughputs:
                summary["avg_throughput_samples_per_sec"] = float(np.mean(throughputs))
            if times:
                summary["avg_epoch_time_sec"] = float(np.mean(times))
                summary["epochs"] = int(len(times))
            if peak_allocs:
                summary["max_gpu_peak_allocated_gb"] = float(np.max(peak_allocs))
            if peak_resv:
                summary["max_gpu_peak_reserved_gb"] = float(np.max(peak_resv))
            # 写入JSON
            with open(self.output_dir / "resource_summary.json", "w") as f:
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
            with open(self.output_dir / "resource_summary.md", "w") as f:
                f.write(md)
            self.logger.info(
                "📋 资源摘要已生成: resource_summary.json / resource_summary.md"
            )
        except Exception as _sum_err:
            self.logger.debug(f"资源摘要生成失败: {_sum_err}")

    # 注意：上方已实现的 create_visualizations 为统一版本；移除重复实现避免维护成本

    def create_test_visualizations(self, test_metrics: dict[str, float]):
        """创建测试阶段的完整可视化报告"""
        if getattr(self, "_test_viz_done", False):
            self.logger.info(
                "⚪ Test visualizations already generated; skipping duplicate run"
            )
            return
        # 配置开关：测试阶段可视化
        try:
            save_test_viz = bool(
                self._cfg_select("visualization.save_test_visualizations", default=True)
            )
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
            h_params = self.h_params

            # Force re-parse if Crop task has no crop_size (fix for bad h_params init)
            if (
                h_params
                and h_params.get("task") == "Crop"
                and h_params.get("crop_size") is None
            ):
                h_params = None

            if h_params is None:
                try:
                    obs_cfg = getattr(self.config, "observation", None)
                    if obs_cfg is None:
                        data_cfg = getattr(self.config, "data", None)
                        obs_cfg = (
                            getattr(data_cfg, "observation", None)
                            if data_cfg is not None
                            else None
                        )
                    if obs_cfg is not None:
                        try:
                            from omegaconf import DictConfig, OmegaConf

                            if isinstance(obs_cfg, DictConfig):
                                obs_cfg = OmegaConf.to_container(obs_cfg, resolve=True)
                        except Exception:
                            pass
                        mode_raw = obs_cfg.get("mode", "sr")
                        mode = str(
                            mode_raw[0]
                            if isinstance(mode_raw, (list, tuple))
                            else mode_raw
                        ).lower()
                        boundary = obs_cfg.get(
                            "boundary", obs_cfg.get("boundary_mode", "mirror")
                        )
                        if mode == "sr":
                            sr_sub = (
                                obs_cfg.get("sr", {})
                                if isinstance(obs_cfg.get("sr", {}), dict)
                                else {}
                            )
                            # 优先查找 'scale'，其次 'scale_factor'，最后是 sr 子字典中的配置
                            scale = obs_cfg.get(
                                "scale",
                                obs_cfg.get(
                                    "scale_factor", sr_sub.get("scale_factor", 2)
                                ),
                            )
                            try:
                                scale = int(scale)
                            except Exception:
                                scale = 2
                            sigma = obs_cfg.get(
                                "blur_sigma", sr_sub.get("blur_sigma", 1.0)
                            )
                            kernel_size = obs_cfg.get(
                                "kernel_size", sr_sub.get("blur_kernel_size", 5)
                            )
                            boundary = (
                                boundary
                                if boundary is not None
                                else sr_sub.get("boundary_mode", "mirror")
                            )
                            downsample = obs_cfg.get(
                                "downsample_interpolation",
                                sr_sub.get("downsample_mode", "area"),
                            )
                            h_params = {
                                "task": "SR",
                                "scale": int(scale),
                                "sigma": float(sigma),
                                "kernel_size": int(kernel_size),
                                "boundary": str(boundary),
                                "downsample_interpolation": str(downsample),
                            }
                        elif mode == "crop":
                            crop_sub = (
                                obs_cfg.get("crop", {})
                                if isinstance(obs_cfg.get("crop", {}), dict)
                                else {}
                            )
                            # Fix: Try 'size' if 'crop_size' is None (as seen in config)
                            crop_size = obs_cfg.get(
                                "crop_size",
                                crop_sub.get("crop_size", crop_sub.get("size", None)),
                            )

                            # Force extraction from OmegaConf if needed
                            if crop_size is None and "crop" in obs_cfg:
                                try:
                                    c = obs_cfg["crop"]
                                    if "size" in c:
                                        crop_size = c["size"]
                                except:
                                    pass

                            # Ensure crop_size is a list for degradation.py compatibility
                            if crop_size is not None and not isinstance(
                                crop_size, (list, tuple)
                            ):
                                crop_size = [crop_size, crop_size]

                            crop_box = obs_cfg.get(
                                "crop_box", crop_sub.get("crop_box", None)
                            )
                            boundary = (
                                boundary
                                if boundary is not None
                                else crop_sub.get("boundary_mode", "mirror")
                            )
                            h_params = {
                                "task": "Crop",
                                "crop_size": crop_size,
                                "crop_box": crop_box,
                                "boundary": str(boundary),
                            }
                        else:
                            h_params = None
                except Exception:
                    h_params = None
            # 保存训练曲线到测试可视化目录
            try:
                ar_visualizer.plot_training_curves(
                    self.training_history, save_name="training_curves"
                )
            except Exception:
                pass

            # 获取测试数据样本进行可视化
            self.get_model().eval()
            test_samples_visualized = 0
            # 来自配置控制测试阶段可视化样本数
            try:
                max_test_samples = int(
                    self._cfg_select(
                        "testing.num_visualization_samples",
                        "logging.visualization.num_test_samples",
                        default=2,
                    )
                )
            except Exception:
                max_test_samples = 2
            # 限制每个样本生成的图像数量，避免过多文件
            try:
                max_images_per_sample = int(
                    self._cfg_select(
                        "visualization.max_images_per_sample",
                        "logging.visualization.max_images_per_sample",
                        default=3,
                    )
                )
            except Exception:
                max_images_per_sample = 3

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_loader):
                    if test_samples_visualized >= max_test_samples:
                        break

                    # 准备输入/目标数据
                    # 优先使用SR观测作为可视化的输入帧，确保与训练退化一致
                    target_seq = batch["target_sequence"].to(self.device)

                    # [Patch] 应用随机掩码（如果配置启用）- 确保可视化与训练一致
                    crop_cfg = getattr(
                        getattr(self.config, "training", None), "crop", None
                    )
                    masks_seq = None
                    if crop_cfg and getattr(crop_cfg, "enabled", False):
                        # 测试阶段强制 patches_per_image=1
                        from omegaconf import OmegaConf

                        if OmegaConf.is_config(crop_cfg):
                            viz_crop_cfg = OmegaConf.to_container(
                                crop_cfg, resolve=True
                            )
                        else:
                            viz_crop_cfg = (
                                crop_cfg.copy()
                                if hasattr(crop_cfg, "copy")
                                else crop_cfg
                            )
                        if isinstance(viz_crop_cfg, dict):
                            viz_crop_cfg["patches_per_image"] = 1
                            viz_crop_cfg = OmegaConf.create(viz_crop_cfg)

                        input_seq, target_seq, masks_seq = self._apply_random_masking(
                            target_seq, viz_crop_cfg
                        )

                        if isinstance(batch, dict):
                            batch["input_sequence"] = input_seq
                            batch["observed_lr_sequence"] = None
                            batch["mask_sequence"] = masks_seq
                            batch["observation_sequence"] = input_seq

                    # 原始输入序列（可能是高分辨率或已退化回升尺寸的序列）
                    input_seq_raw = batch.get("input_sequence", None)
                    if input_seq_raw is not None:
                        input_seq_raw = input_seq_raw.to(self.device)

                    # 选择用于可视化的输入序列，优先观测/基线
                    input_seq_vis = None
                    try:
                        # 1) 直接使用时序观测（若提供）
                        if (
                            "observation_sequence" in batch
                            and batch["observation_sequence"] is not None
                        ):
                            obs_seq = batch["observation_sequence"]
                            input_seq_vis = obs_seq.to(self.device)
                        # 2) 使用通用观测字段（形状可能非时序）
                        elif (
                            "observation" in batch and batch["observation"] is not None
                        ):
                            obs = batch["observation"]
                            # 期望形状：[B, T_in, C, H, W] 或 [T_in, C, H, W]
                            if obs.dim() == 5:
                                input_seq_vis = obs.to(self.device)
                            else:
                                # 将非时序观测扩展为时序长度
                                # 获取 Batch Size 和 Time Steps
                                B = (
                                    input_seq_raw.shape[0]
                                    if input_seq_raw is not None
                                    and input_seq_raw.dim() >= 1
                                    else 1
                                )
                                T_in = (
                                    input_seq_raw.shape[1]
                                    if input_seq_raw is not None
                                    and input_seq_raw.dim() >= 2
                                    else 1
                                )

                                if obs.dim() == 4:  # [B, C, H, W]
                                    # 扩展时间维度: [B, C, H, W] -> [B, T_in, C, H, W]
                                    obs = obs.unsqueeze(1).repeat(1, T_in, 1, 1, 1)
                                elif obs.dim() == 3:  # [C, H, W]
                                    # 扩展Batch和Time: [C, H, W] -> [B, T_in, C, H, W]
                                    obs = (
                                        obs.unsqueeze(0)
                                        .unsqueeze(0)
                                        .repeat(B, T_in, 1, 1, 1)
                                    )

                                input_seq_vis = obs.to(self.device)
                        # 3) 使用baseline（可能为上采样后的SR观测）
                        elif "baseline" in batch and batch["baseline"] is not None:
                            base = batch["baseline"]
                            B = (
                                input_seq_raw.shape[0]
                                if input_seq_raw is not None
                                and input_seq_raw.dim() >= 1
                                else 1
                            )
                            T_in = (
                                input_seq_raw.shape[1]
                                if input_seq_raw is not None
                                and input_seq_raw.dim() >= 2
                                else 1
                            )

                            if base.dim() == 5:
                                input_seq_vis = base.to(self.device)
                            else:
                                if base.dim() == 4:  # [B, C, H, W]
                                    # [B, C, H, W] -> [B, T_in, C, H, W]
                                    base = base.unsqueeze(1).repeat(1, T_in, 1, 1, 1)
                                elif base.dim() == 3:  # [C, H, W]
                                    # [C, H, W] -> [B, T_in, C, H, W]
                                    base = (
                                        base.unsqueeze(0)
                                        .unsqueeze(0)
                                        .repeat(B, T_in, 1, 1, 1)
                                    )
                                else:
                                    # 处理 [T_in*C, H, W] 的flatten格式（来自某些时序数据集）
                                    if (
                                        base.dim() == 3
                                        and input_seq_raw is not None
                                        and input_seq_raw.dim() == 4
                                    ):
                                        C = input_seq_raw.shape[1]
                                        H, W = base.shape[-2:]
                                        # 尝试恢复为 [T_in, C, H, W]
                                        try:
                                            base = base.view(t_in, C, H, W)
                                        except Exception:
                                            # 回退：重复单帧
                                            base = base.unsqueeze(0).repeat(
                                                t_in, 1, 1, 1
                                            )
                                input_seq_vis = base.to(self.device)
                    except Exception:
                        # 回退到原始输入序列
                        input_seq_vis = (
                            input_seq_raw if input_seq_raw is not None else None
                        )

                    if input_seq_vis is None:
                        # 最终回退：使用原始输入序列
                        input_seq_vis = input_seq_raw

                    # 模型前向使用4D输入
                    # FIX: Use input_seq_vis (which contains the correct observation/degraded input) instead of raw input
                    # Prioritize 'observation' directly from batch to avoid any ambiguity
                    if "observation" in batch and batch["observation"] is not None:
                        input_seq = batch["observation"].to(self.device)
                    elif input_seq_vis is not None:
                        input_seq = input_seq_vis
                    else:
                        input_seq = input_seq_raw

                    # if input_seq is not None and input_seq.dim() == 5:
                    #     input_seq = input_seq[:, 0]

                    # Debug: Print input shape to log
                    if batch_idx == 0:
                        self.logger.info(
                            f"🔍 Test Visualization Input Shape: {input_seq.shape if input_seq is not None else 'None'}"
                        )
                        # Print batch keys and shapes for debugging
                        if isinstance(batch, dict):
                            keys_info = {
                                k: (
                                    str(v.shape)
                                    if isinstance(v, torch.Tensor)
                                    else "N/A"
                                )
                                for k, v in batch.items()
                            }
                            self.logger.info(f"📦 Batch Keys: {keys_info}")

                    # Safety: If model expects LR input but we got HR input (same size as target), downsample it!
                    # This prevents OOM when EDSR (scale 32) receives 128x128 input and tries to output 4096x4096
                    model = self.get_model()
                    upscale = getattr(model, "upscale", None)
                    if upscale is None:
                        upscale = getattr(
                            getattr(model, "module", None), "upscale", None
                        )
                    if upscale is None:
                        try:
                            upscale = int(self._cfg_select("model.upscale", default=1))
                        except:
                            upscale = 1
                    upscale = int(upscale)

                    if batch_idx == 0:
                        self.logger.info(f"🔍 Detected model upscale: {upscale}")

                    if upscale > 1 and input_seq is not None:
                        # Check if input resolution matches target resolution
                        if (
                            input_seq.shape[-1] == target_seq.shape[-1]
                            and input_seq.shape[-2] == target_seq.shape[-2]
                        ):
                            self.logger.warning(
                                f"⚠️ Input shape {input_seq.shape} matches target shape {target_seq.shape} but model upscale is {upscale}. Downsampling input to avoid OOM!"
                            )
                            if input_seq.dim() == 5:
                                # 5D: [B, T, C, H, W] -> flatten T -> interpolate -> reshape
                                B, T, C, H, W = input_seq.shape
                                input_seq_flat = input_seq.view(B * T, C, H, W)
                                input_seq_flat = torch.nn.functional.interpolate(
                                    input_seq_flat,
                                    scale_factor=1.0 / upscale,
                                    mode="area",
                                )
                                _, _, H_new, W_new = input_seq_flat.shape
                                input_seq = input_seq_flat.view(B, T, C, H_new, W_new)
                            else:
                                input_seq = torch.nn.functional.interpolate(
                                    input_seq, scale_factor=1.0 / upscale, mode="area"
                                )
                            self.logger.info(
                                f"⬇️ Downsampled input to {input_seq.shape}"
                            )

                    # 预测
                    images_generated = 0
                    try:
                        sample_name = f"test_sample_{batch_idx}"
                        sample_key = int(test_samples_visualized + 1)

                        # 确保输入在设备上
                        if input_seq is not None:
                            input_seq = input_seq.to(self.device)

                        # 1. 序列预测 (AR Forward)
                        # 检查模型类型
                        model = self.get_model()
                        if hasattr(model, "spatial_forward") and hasattr(
                            model, "temporal_forward"
                        ):
                            # Sequential Model
                            mo_out = model(input_seq, target_seq)  # TF
                            pred_seq = mo_out["final_pred"]
                        else:
                            # Standard Model
                            # 自动适配输入通道数
                            raw_model = (
                                model.module if hasattr(model, "module") else model
                            )

                            # 针对裸空间模型（如UNet）适配 5D -> 4D 及通道 Padding
                            # 注意：input_seq是5D [B, T, C, H, W]
                            model_input = input_seq
                            is_raw_spatial = False

                            if input_seq.dim() == 5:
                                has_seq_method = (
                                    hasattr(raw_model, "autoregressive_predict")
                                    or hasattr(raw_model, "temporal_forward")
                                    or (
                                        hasattr(raw_model, "is_temporal")
                                        and raw_model.is_temporal
                                    )
                                )

                                if not has_seq_method:
                                    model_input = input_seq[:, 0]
                                    is_raw_spatial = True

                                    if hasattr(raw_model, "in_channels"):
                                        exp_in = raw_model.in_channels
                                        if model_input.shape[1] < exp_in:
                                            pad_c = exp_in - model_input.shape[1]
                                            padding = torch.zeros(
                                                model_input.size(0),
                                                pad_c,
                                                model_input.size(2),
                                                model_input.size(3),
                                                device=model_input.device,
                                                dtype=model_input.dtype,
                                            )
                                            model_input = torch.cat(
                                                [model_input, padding], dim=1
                                            )

                            if is_raw_spatial:
                                out = model(model_input)
                            else:
                                out = model(input_seq, target_seq)  # TF

                            if isinstance(out, dict):
                                pred_seq = out["final_pred"]
                            else:
                                pred_seq = out

                            if pred_seq.dim() == 4:
                                pred_seq = pred_seq.unsqueeze(1)

                        # [Patch] Auto-interpolate if output size mismatches target (for UNet SR in Test)
                        if pred_seq.shape[-2:] != target_seq.shape[-2:]:
                            # If sequence length matches
                            if pred_seq.shape[1] == target_seq.shape[1]:
                                B, T, C, H, W = target_seq.shape
                                pred_seq = pred_seq.flatten(0, 1)
                                pred_seq = torch.nn.functional.interpolate(
                                    pred_seq,
                                    size=(H, W),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                pred_seq = pred_seq.view(B, T, C, H, W)
                            else:
                                # Fallback: interpolate last frame only
                                pass

                        sample_input = input_seq.detach().cpu()
                        sample_target = target_seq.detach().cpu()
                        sample_pred = pred_seq.detach().cpu()
                        current_T_out = sample_target.shape[1]

                        # 1. 整体序列可视化 (Gif)
                        if len(sample_target.shape) >= 4:
                            ar_visualizer.visualize_ar_predictions(
                                sample_input,
                                sample_target,
                                sample_pred,
                                timestep_idx=0,
                                save_name=f"{sample_name}",
                                norm_stats=self.norm_stats,
                                h_params=h_params,
                                sample_idx=sample_key,
                            )
                            images_generated += 1

                        # 单帧和误差分析 (使用最后一帧)
                        if current_T_out > 0:
                            last_idx = current_T_out - 1
                            ar_visualizer.visualize_single_frame(
                                (
                                    sample_input[0, -1]
                                    if sample_input.ndim == 5
                                    else sample_input[0]
                                ),
                                sample_target[0, last_idx],
                                sample_pred[0, last_idx],
                                save_name=f"{sample_name}_seq_last_frame",
                                norm_stats=self.norm_stats,
                            )
                            images_generated += 1

                        # 2. 四宫格对比图 (Obs | GT | Pred | Error)
                        # 生成前3个时间步的对比图
                        for t in range(min(3, current_T_out)):
                            ar_visualizer.visualize_obs_gt_pred_error(
                                sample_target,
                                sample_pred,
                                timestep_idx=t,
                                save_name=f"{sample_name}_obs_gt_pred_error",
                                norm_stats=self.norm_stats,
                                h_params=h_params,
                                sample_idx=sample_key,
                                # observation_seq=sample_input  <-- Removed to force re-generation using h_params
                            )
                            images_generated += 1

                        # 3. 误差分析
                        ar_visualizer.create_error_analysis(
                            sample_target,
                            sample_pred,
                            save_name=f"{sample_name}_error_analysis",
                            norm_stats=self.norm_stats,
                        )
                        images_generated += 1

                    except Exception as e:
                        self.logger.warning(f"AR visualizer failed: {e}")
                        import traceback

                        self.logger.warning(traceback.format_exc())
                    else:
                        # 顺序模型/普通模型：统一使用 AR 序列可视化
                        try:
                            # 序列可视化（整体）
                            if len(sample_target.shape) >= 4:
                                ar_visualizer.visualize_ar_predictions(
                                    sample_input,
                                    sample_target,
                                    sample_pred,
                                    timestep_idx=0,
                                    save_name=f"{sample_name}",
                                    norm_stats=self.norm_stats,
                                    h_params=h_params,
                                    sample_idx=sample_key,
                                )
                                images_generated += 1

                            # 单帧和误差分析 (使用最后一帧)
                            if current_T_out > 0:
                                last_idx = current_T_out - 1
                                ar_visualizer.visualize_single_frame(
                                    (
                                        sample_input[0, -1]
                                        if sample_input.ndim == 5
                                        else sample_input[0]
                                    ),
                                    sample_target[0, last_idx],
                                    sample_pred[0, last_idx],
                                    save_name=f"{sample_name}_seq_last_frame",
                                    norm_stats=self.norm_stats,
                                )
                                images_generated += 1

                            # 误差分析
                            ar_visualizer.create_error_analysis(
                                sample_target,
                                sample_pred,
                                save_name=f"{sample_name}_error_analysis",
                                norm_stats=self.norm_stats,
                            )
                            images_generated += 1

                        except Exception as e:
                            self.logger.warning(f"Sequence visualization failed: {e}")
                            import traceback

                            self.logger.warning(traceback.format_exc())

                    if images_generated < max_images_per_sample:
                        try:
                            obs_seq = None
                            try:
                                if ("observation_sequence" in sample) and (
                                    sample["observation_sequence"] is not None
                                ):
                                    obs_seq = sample["observation_sequence"]
                                elif ("observed_lr_sequence" in sample) and (
                                    sample["observed_lr_sequence"] is not None
                                ):
                                    obs_seq = sample["observed_lr_sequence"]
                            except Exception:
                                obs_seq = None
                            # Visualize multiple key timesteps (0, 20, 50, 70, etc.)
                            base_start_time = (
                                start_time
                                if start_time is not None
                                else int(
                                    self._cfg_select("data.time_step_start", default=0)
                                )
                            )
                            # Define timesteps to visualize (relative to sequence start)
                            # t=0 is always visualized (start_time)
                            # We also want t=20, t=50, t=70 if available
                            vis_steps = [0, 20, 50, 70]
                            # Add start_time if not 0 (though usually it's 0)
                            if (
                                base_start_time != 0
                                and base_start_time not in vis_steps
                            ):
                                vis_steps.insert(0, base_start_time)

                            for t_idx in vis_steps:
                                # Check if t_idx is within the sequence length
                                if t_idx < current_T_out:
                                    try:
                                        ar_visualizer.visualize_obs_gt_pred_error(
                                            sample_target,
                                            sample_pred,
                                            save_name=f"{sample_name}_obs_gt_pred_error_t{t_idx}",
                                            norm_stats=self.norm_stats,
                                            h_params=h_params,
                                            timestep_idx=t_idx,
                                            sample_idx=(
                                                sample_key
                                                if sample_key is not None
                                                else int(test_samples_visualized + 1)
                                            ),
                                            observation_seq=obs_seq,
                                        )
                                        # Only count as generated image once per sample to avoid hitting limit too early
                                        if t_idx == 0:
                                            images_generated += 1
                                    except Exception as _obs_err:
                                        self.logger.warning(
                                            f"Obs/GT/Pred/Error visualization failed for t={t_idx}: {_obs_err}"
                                        )

                            test_samples_visualized += 1
                            if test_samples_visualized >= max_test_samples:
                                break
                        except Exception as _obs_err:
                            self.logger.warning(
                                f"Obs/GT/Pred/Error visualization failed: {_obs_err}"
                            )

                    # 3. 时间分析（仅当T一致）
                    self.ensure_norm_stats()
                    try:
                        if (images_generated < max_images_per_sample) and (
                            sample_pred.shape[1] == sample_target.shape[1]
                        ):
                            ar_visualizer.create_temporal_analysis(
                                sample_pred,
                                sample_target,
                                save_name=f"{sample_name}_temporal_analysis",
                                norm_stats=self.norm_stats,
                            )
                            images_generated += 1
                            if images_generated >= max_images_per_sample:
                                test_samples_visualized += 1
                                if test_samples_visualized >= max_test_samples:
                                    break
                        else:
                            # 回退：仅分析最后帧（不做时序分析）
                            pass
                    except Exception:
                        pass

                    # 4. 边界带误差与频域RMSE诊断
                    try:
                        if images_generated < max_images_per_sample:
                            ar_visualizer.create_boundary_and_frequency_metrics(
                                sample_pred,
                                sample_target,
                                save_name=f"{sample_name}_boundary_frequency_metrics",
                                band_width=16,
                            )
                            images_generated += 1
                            if images_generated >= max_images_per_sample:
                                test_samples_visualized += 1
                                if test_samples_visualized >= max_test_samples:
                                    break
                    except Exception:
                        pass

                        try:
                            import numpy as np

                            self.ensure_norm_stats()
                            pred_last = sample_pred[:, -1]
                            tgt_last = sample_target[:, -1]
                            diff = pred_last - tgt_last
                            rel_l2 = np.linalg.norm(diff) / (
                                np.linalg.norm(tgt_last) + 1e-8
                            )
                            mae = float(np.mean(np.abs(diff)))
                            mse = float(np.mean(diff**2))
                            psnr = float(20.0 * np.log10(1.0 / (np.sqrt(mse) + 1e-8)))
                            metrics_list.append(
                                {
                                    "sample": int(test_samples_visualized + 1),
                                    "rel_l2": float(rel_l2),
                                    "mae": float(mae),
                                    "mse": float(mse),
                                    "psnr": float(psnr),
                                }
                            )
                        except Exception as _metrics_err:
                            self.logger.warning(
                                f"Metrics collection failed for {sample_name}: {_metrics_err}"
                            )

                        test_samples_visualized += 1

                        if test_samples_visualized >= max_test_samples:
                            break

                    # 定义 input_np 以解决后续 NameError
                    input_np = input_seq.detach().cpu().numpy()
                    target_np = target_seq.detach().cpu().numpy()
                    pred_np = pred_seq.detach().cpu().numpy()

                    batch_size = input_np.shape[0]
                    samples_to_take = int(
                        min(batch_size, max_test_samples - test_samples_visualized)
                    )
                    # 若当前批次无需可视化样本，则跳过后续生成逻辑，避免重复生成同一个样本的图
                    if samples_to_take <= 0:
                        continue

                    # 将 batch 数据迁移到循环内处理，因为每次循环处理一个样本
                    for sample_idx in range(samples_to_take):
                        # 从批次元信息读取真实 sample 与时间信息
                        try:
                            batch_meta = batch  # 原始tensor批次
                            sample_key = None
                            start_time = None
                            time_indices = None
                            if isinstance(batch_meta, dict):
                                # 与 DataLoader 字段对应
                                if "sample_key" in batch_meta:
                                    sk = batch_meta["sample_key"]
                                    sample_key = (
                                        sk[sample_idx]
                                        if hasattr(sk, "__getitem__")
                                        else sk
                                    )
                                if "start_time" in batch_meta:
                                    st = batch_meta["start_time"]
                                    start_time = (
                                        int(st[sample_idx])
                                        if hasattr(st, "__getitem__")
                                        else int(st)
                                    )
                                if "time_indices" in batch_meta:
                                    ti = batch_meta["time_indices"]
                                    time_indices = (
                                        ti[sample_idx]
                                        if hasattr(ti, "__getitem__")
                                        else ti
                                    )
                            # 统一使用 sample_key 作为前缀
                            sample_name = f"sample_{str(sample_key) if sample_key is not None else (test_samples_visualized + 1)}"
                        except Exception:
                            sample_name = f"test_sample_{test_samples_visualized + 1}"

                        # 提取单个样本
                        sample_input = input_np[
                            sample_idx : sample_idx + 1
                        ]  # [1, T_in, C, H, W]
                        sample_target = target_np[
                            sample_idx : sample_idx + 1
                        ]  # [1, T_out, C, H, W]
                        sample_pred = pred_np[
                            sample_idx : sample_idx + 1
                        ]  # [1, T_out, C, H, W]

                        if images_generated >= max_images_per_sample:
                            break
                        if images_generated == 0:
                            self.logger.info(
                                f"📊 Generating visualization for test sample {test_samples_visualized + 1}..."
                            )

                        # 1. 预测可视化（顺序模型与AR分别处理）
                        self.ensure_norm_stats()

                        # 总是使用AR可视化器，因为它现在支持序列可视化
                        # 正确处理5D到4D的转换 (传递完整5D Tensor给visualize_ar_predictions)
                        try:
                            # 序列可视化（整体）
                            if len(sample_target.shape) >= 4:
                                ar_visualizer.visualize_ar_predictions(
                                    sample_input,
                                    sample_target,
                                    sample_pred,
                                    timestep_idx=0,
                                    save_name=f"{sample_name}",
                                    norm_stats=self.norm_stats,
                                    h_params=h_params,
                                    sample_idx=sample_key,
                                )
                                images_generated += 1

                            # 单帧和误差分析 (使用最后一帧)
                            if current_T_out > 0:
                                last_idx = current_T_out - 1
                                ar_visualizer.visualize_single_frame(
                                    (
                                        sample_input[0, -1]
                                        if sample_input.ndim == 5
                                        else sample_input[0]
                                    ),
                                    sample_target[0, last_idx],
                                    sample_pred[0, last_idx],
                                    save_name=f"{sample_name}_seq_last_frame",
                                    norm_stats=self.norm_stats,
                                )
                                images_generated += 1

                            # 2. 四宫格对比图 (Obs | GT | Pred | Error)
                            # 生成前3个时间步的对比图
                            for t in range(min(3, current_T_out)):
                                ar_visualizer.visualize_obs_gt_pred_error(
                                    sample_target,
                                    sample_pred,
                                    timestep_idx=t,
                                    save_name=f"{sample_name}_obs_gt_pred_error",
                                    norm_stats=self.norm_stats,
                                    h_params=h_params,
                                    sample_idx=sample_key,
                                    # observation_seq=sample_input  <-- Removed to force re-generation using h_params
                                )
                                images_generated += 1

                            # 3. 误差分析
                            ar_visualizer.create_error_analysis(
                                sample_target,
                                sample_pred,
                                save_name=f"{sample_name}_error_analysis",
                                norm_stats=self.norm_stats,
                            )
                            images_generated += 1

                        except Exception as e:
                            self.logger.warning(f"AR visualizer failed: {e}")
                            import traceback

                            self.logger.warning(traceback.format_exc())
                    else:
                        # 顺序模型/普通模型：统一使用 AR 序列可视化
                        try:
                            # 序列可视化（整体）
                            if len(sample_target.shape) >= 4:
                                ar_visualizer.visualize_ar_predictions(
                                    sample_input,
                                    sample_target,
                                    sample_pred,
                                    timestep_idx=0,
                                    save_name=f"{sample_name}",
                                    norm_stats=self.norm_stats,
                                    h_params=h_params,
                                    sample_idx=sample_key,
                                )
                                images_generated += 1

                            # 单帧和误差分析 (使用最后一帧)
                            if current_T_out > 0:
                                last_idx = current_T_out - 1
                                ar_visualizer.visualize_single_frame(
                                    (
                                        sample_input[0, -1]
                                        if sample_input.ndim == 5
                                        else sample_input[0]
                                    ),
                                    sample_target[0, last_idx],
                                    sample_pred[0, last_idx],
                                    save_name=f"{sample_name}_seq_last_frame",
                                    norm_stats=self.norm_stats,
                                )
                                images_generated += 1

                            # 误差分析
                            ar_visualizer.create_error_analysis(
                                sample_target,
                                sample_pred,
                                save_name=f"{sample_name}_error_analysis",
                                norm_stats=self.norm_stats,
                            )
                            images_generated += 1

                        except Exception as e:
                            self.logger.warning(f"Sequence visualization failed: {e}")
                            import traceback

                            self.logger.warning(traceback.format_exc())

                    if images_generated < max_images_per_sample:
                        try:
                            obs_seq = None
                            try:
                                if ("observation_sequence" in sample) and (
                                    sample["observation_sequence"] is not None
                                ):
                                    obs_seq = sample["observation_sequence"]
                                elif ("observed_lr_sequence" in sample) and (
                                    sample["observed_lr_sequence"] is not None
                                ):
                                    obs_seq = sample["observed_lr_sequence"]
                            except Exception:
                                obs_seq = None
                            # Visualize multiple key timesteps (0, 20, 50, 70, etc.)
                            base_start_time = (
                                start_time
                                if start_time is not None
                                else int(
                                    self._cfg_select("data.time_step_start", default=0)
                                )
                            )
                            # Define timesteps to visualize (relative to sequence start)
                            # t=0 is always visualized (start_time)
                            # We also want t=20, t=50, t=70 if available
                            vis_steps = [0, 20, 50, 70]
                            # Add start_time if not 0 (though usually it's 0)
                            if (
                                base_start_time != 0
                                and base_start_time not in vis_steps
                            ):
                                vis_steps.insert(0, base_start_time)

                            for t_idx in vis_steps:
                                # Check if t_idx is within the sequence length
                                if t_idx < current_T_out:
                                    try:
                                        ar_visualizer.visualize_obs_gt_pred_error(
                                            sample_target,
                                            sample_pred,
                                            save_name=f"{sample_name}_obs_gt_pred_error_t{t_idx}",
                                            norm_stats=self.norm_stats,
                                            h_params=h_params,
                                            timestep_idx=t_idx,
                                            sample_idx=(
                                                sample_key
                                                if sample_key is not None
                                                else int(test_samples_visualized + 1)
                                            ),
                                            observation_seq=obs_seq,
                                        )
                                        # Only count as generated image once per sample to avoid hitting limit too early
                                        if t_idx == 0:
                                            images_generated += 1
                                    except Exception as _obs_err:
                                        self.logger.warning(
                                            f"Obs/GT/Pred/Error visualization failed for t={t_idx}: {_obs_err}"
                                        )

                            test_samples_visualized += 1
                            if test_samples_visualized >= max_test_samples:
                                break
                        except Exception as _obs_err:
                            self.logger.warning(
                                f"Obs/GT/Pred/Error visualization failed: {_obs_err}"
                            )

                    # 3. 时间分析（仅当T一致）
                    self.ensure_norm_stats()
                    try:
                        if (images_generated < max_images_per_sample) and (
                            sample_pred.shape[1] == sample_target.shape[1]
                        ):
                            ar_visualizer.create_temporal_analysis(
                                sample_pred,
                                sample_target,
                                save_name=f"{sample_name}_temporal_analysis",
                                norm_stats=self.norm_stats,
                            )
                            images_generated += 1
                            if images_generated >= max_images_per_sample:
                                test_samples_visualized += 1
                                if test_samples_visualized >= max_test_samples:
                                    break
                        else:
                            # 回退：仅分析最后帧（不做时序分析）
                            pass
                    except Exception:
                        pass

                    # 4. 边界带误差与频域RMSE诊断
                    try:
                        if images_generated < max_images_per_sample:
                            ar_visualizer.create_boundary_and_frequency_metrics(
                                sample_pred,
                                sample_target,
                                save_name=f"{sample_name}_boundary_frequency_metrics",
                                band_width=16,
                            )
                            images_generated += 1
                            if images_generated >= max_images_per_sample:
                                test_samples_visualized += 1
                                if test_samples_visualized >= max_test_samples:
                                    break
                    except Exception:
                        pass

                        try:
                            import numpy as np

                            self.ensure_norm_stats()
                            pred_last = sample_pred[:, -1]
                            tgt_last = sample_target[:, -1]
                            diff = pred_last - tgt_last
                            rel_l2 = np.linalg.norm(diff) / (
                                np.linalg.norm(tgt_last) + 1e-8
                            )
                            mae = float(np.mean(np.abs(diff)))
                            mse = float(np.mean(diff**2))
                            psnr = float(20.0 * np.log10(1.0 / (np.sqrt(mse) + 1e-8)))
                            metrics_list.append(
                                {
                                    "sample": int(test_samples_visualized + 1),
                                    "rel_l2": float(rel_l2),
                                    "mae": float(mae),
                                    "mse": float(mse),
                                    "psnr": float(psnr),
                                }
                            )
                        except Exception as _metrics_err:
                            self.logger.warning(
                                f"Metrics collection failed for {sample_name}: {_metrics_err}"
                            )

                        test_samples_visualized += 1

                        if test_samples_visualized >= max_test_samples:
                            break

            try:
                import json

                import numpy as np

                eval_dir = self.output_dir / "eval"
                eval_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = eval_dir / "metrics.jsonl"
                with open(metrics_path, "w") as f:
                    for m in metrics_list:
                        f.write(json.dumps(m) + "\n")
                summary = {}
                if metrics_list:
                    keys = [k for k in metrics_list[0].keys() if k != "sample"]
                    for k in keys:
                        vals = np.array([m[k] for m in metrics_list])
                        summary[k] = {
                            "mean": float(vals.mean()),
                            "std": float(vals.std()),
                            "min": float(vals.min()),
                            "max": float(vals.max()),
                            "median": float(np.median(vals)),
                            "count": int(len(vals)),
                        }
                else:
                    # Fallback: compute one metrics entry from last frames if available
                    try:
                        if last_pred_np is not None and last_tgt_np is not None:
                            diff = last_pred_np - last_tgt_np
                            rel_l2 = float(
                                np.linalg.norm(diff)
                                / (np.linalg.norm(last_tgt_np) + 1e-8)
                            )
                            mae = float(np.mean(np.abs(diff)))
                            mse = float(np.mean(diff**2))
                            psnr = float(20.0 * np.log10(1.0 / (np.sqrt(mse) + 1e-8)))
                            m = {
                                "sample": 1,
                                "rel_l2": rel_l2,
                                "mae": mae,
                                "mse": mse,
                                "psnr": psnr,
                            }
                            with open(metrics_path, "w") as f:
                                f.write(json.dumps(m) + "\n")
                            summary = {
                                "rel_l2": {
                                    "mean": rel_l2,
                                    "std": 0.0,
                                    "min": rel_l2,
                                    "max": rel_l2,
                                    "median": rel_l2,
                                    "count": 1,
                                },
                                "mae": {
                                    "mean": mae,
                                    "std": 0.0,
                                    "min": mae,
                                    "max": mae,
                                    "median": mae,
                                    "count": 1,
                                },
                                "mse": {
                                    "mean": mse,
                                    "std": 0.0,
                                    "min": mse,
                                    "max": mse,
                                    "median": mse,
                                    "count": 1,
                                },
                                "psnr": {
                                    "mean": psnr,
                                    "std": 0.0,
                                    "min": psnr,
                                    "max": psnr,
                                    "median": psnr,
                                    "count": 1,
                                },
                            }
                    except Exception:
                        pass

                # Write H consistency quick check
                try:
                    from ops.degradation import apply_degradation_operator

                    mean_v = self.norm_stats.get(
                        "u_mean", self.norm_stats.get("mean", 0.0)
                    )
                    std_v = self.norm_stats.get(
                        "u_std", self.norm_stats.get("std", 1.0)
                    )
                    mean_v = float(
                        mean_v
                        if isinstance(mean_v, (float, int))
                        else np.array(mean_v).reshape(-1)[0]
                    )
                    std_v = float(
                        std_v
                        if isinstance(std_v, (float, int))
                        else np.array(std_v).reshape(-1)[0]
                    )
                    if last_tgt_np is not None and last_obs_np is not None:
                        tgt_orig = last_tgt_np * std_v + mean_v
                        obs_orig = last_obs_np * std_v + mean_v
                        h_gt = apply_degradation_operator(
                            torch.from_numpy(tgt_orig).to(self.device), h_params
                        )
                        h_gt_err = torch.norm(
                            h_gt - torch.from_numpy(obs_orig).to(self.device), p=2
                        ).item()
                        h_gt_rel = h_gt_err / (np.linalg.norm(obs_orig) + 1e-8)
                        with open(
                            self.output_dir / "eval" / "h_consistency.json", "w"
                        ) as f:
                            json.dump(
                                {"h_gt_error": h_gt_err, "h_gt_rel_error": h_gt_rel},
                                f,
                                indent=2,
                            )
                except Exception:
                    pass
                with open(eval_dir / "summary_stats.json", "w") as f:
                    json.dump(summary, f, indent=2)
                md_lines = [
                    "# Evaluation Results",
                    "| Metric | Mean ± Std | Min | Max | Median |",
                    "|--------|------------|-----|-----|--------|",
                ]
                for k, stats in summary.items():
                    md_lines.append(
                        f"| {k} | {stats['mean']:.4f} ± {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} |"
                    )
                with open(eval_dir / "results_table.md", "w") as f:
                    f.write("\n".join(md_lines))
            except Exception:
                pass

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
                for file_path in list(test_viz_dir.glob("*.svg")) + list(
                    test_viz_dir.glob("*.png")
                ):
                    shutil.copy2(file_path, paper_test_dir)
                for file_path in test_viz_dir.glob("*.html"):
                    shutil.copy2(file_path, paper_test_dir)

                self.logger.info(
                    f"📋 Test visualization files copied to {paper_test_dir}"
                )

            self.logger.info(
                f"✅ Test visualizations completed, saved to {test_viz_dir} and {paper_test_dir}"
            )
            self._test_viz_done = True

        except Exception as e:
            self.logger.error(f"❌ Failed to generate test visualizations: {e}")
            import traceback

            traceback.print_exc()

    def _create_test_metrics_visualization(
        self, test_metrics: dict[str, float], output_dir: Path
    ):
        """Create test metrics visualization (English labels)."""
        try:
            import matplotlib.pyplot as plt

            # 准备指标数据
            metrics_names = list(test_metrics.keys())
            metrics_values = list(test_metrics.values())

            # 创建指标柱状图
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(metrics_names, metrics_values, color="skyblue", alpha=0.7)

            # 添加数值标签
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Test Metrics Results", fontsize=16, fontweight="bold")
            ax.set_ylabel("Metric Value", fontsize=12)
            ax.set_xlabel("Metrics", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # 保存图像
            plt.savefig(
                output_dir / "test_metrics.svg",
                dpi=300,
                bbox_inches="tight",
                format="svg",
            )
            plt.close()

            self.logger.info("📊 Test metrics visualization generated")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate test metrics visualization: {e}")

    def _create_test_html_report(
        self, test_metrics: dict[str, float], viz_dir: Path, paper_dir: Path
    ):
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
                "mse": "Mean Squared Error",
                "mae": "Mean Absolute Error",
                "rel_l2": "Relative L2 Error",
                "psnr": "Peak Signal-to-Noise Ratio",
                "ssim": "Structural Similarity Index",
                "temporal_mse": "Temporal MSE (temporal consistency error)",
                "long_term_stability": "Long-term Stability",
            }

            for metric_name, metric_value in test_metrics.items():
                description = metric_descriptions.get(metric_name, "Test Metric")
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
                <img src="test_metrics.svg" alt="Metrics Overview">
            </div>
        </div>
        
        <h2>🎯 Test Samples Visualization</h2>
        <div class="image-grid">
"""

            # 添加测试样本可视化
            for i in range(1, 6):  # 最多5个测试样本
                sample_files = [
                    f"test_sample_{i}_ar_predictions.svg",
                    f"test_sample_{i}_error_analysis.svg",
                    f"test_sample_{i}_temporal_analysis.svg",
                    f"test_sample_{i}_ar_predictions.png",
                    f"test_sample_{i}_error_analysis.png",
                    f"test_sample_{i}_temporal_analysis.png",
                    f"test_sample_{i}_boundary_frequency_metrics.svg",
                    f"test_sample_{i}_boundary_frequency_metrics.png",
                ]

                for file_name in sample_files:
                    if (viz_dir / file_name).exists():
                        title = (
                            file_name.replace(".svg", "")
                            .replace(".png", "")
                            .replace("_", " ")
                            .title()
                        )
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
            report_path = viz_dir / "test_report.html"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # 也保存到paper_package目录
            paper_report_path = paper_dir / "test_report.html"
            with open(paper_report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            self.logger.info(f"📄 Test HTML report generated: {report_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate test HTML report: {e}")

    def create_final_report(self):
        """Create final visualization report (English logs)."""
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning(
                "Visualization module unavailable, skipping final report generation"
            )
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
                self.logger.info("📋 Visualization files copied to paper_package")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate final report: {e}")
            import traceback

            traceback.print_exc()

    def smoke_test(self):
        """冒烟测试：运行一个Batch以检测显存泄漏或立即崩溃"""
        self.logger.info("🚬 Running SMOKE TEST (1 batch)...")
        self.get_model().train()

        # 临时覆盖配置
        debug_cuda = self._cfg_select("training.debug_cuda", default=False)

        try:
            # 取一个batch
            batch = next(iter(self.train_loader))
            if batch is None:
                self.logger.warning("Smoke test batch is None, skipping.")
                return

            # 记录初始显存
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                mem_start = torch.cuda.memory_allocated()

            # Forward
            input_seq = batch["input_sequence"].to(self.device)
            target_seq = batch["target_sequence"].to(self.device)
            current_T_out = 1  # 最简模式

            # 数据适配：如果是空间-only模型且输入是5D，取第一帧
            model = self.get_model()
            model_input = input_seq

            # 简单判断是否为纯空间模型（无时间维度处理能力的模型通常期望4D输入）
            # 注意：ARWrapper 等会自动处理，这里主要针对裸空间模型
            if input_seq.dim() == 5:
                # 尝试推断是否需要降维
                is_seq_model = (
                    hasattr(model, "temporal_forward")
                    or hasattr(model, "autoregressive_predict")
                    or (hasattr(model, "is_temporal") and model.is_temporal)
                    or (
                        hasattr(model, "_unified_interface")
                        and model._unified_interface
                    )
                )  # ARWrapper 标志

                if not is_seq_model:
                    # 假设是空间模型，只取第一帧 [B, T, C, H, W] -> [B, C, H, W]
                    model_input = input_seq[:, 0]

                    # 自动适配输入通道数（模拟 train_epoch 中的 concat 逻辑）
                    raw_model = model.module if hasattr(model, "module") else model
                    if hasattr(raw_model, "in_channels"):
                        expected_in = raw_model.in_channels
                    else:
                        expected_in = model_input.shape[1]

                    current_in = model_input.shape[1]
                    if current_in < expected_in:
                        pad_c = expected_in - current_in
                        # Pad with zeros: [B, pad_c, H, W]
                        padding = torch.zeros(
                            model_input.size(0),
                            pad_c,
                            model_input.size(2),
                            model_input.size(3),
                            device=model_input.device,
                            dtype=model_input.dtype,
                        )
                        model_input = torch.cat([model_input, padding], dim=1)
                        # self.logger.info(f"SmokeTest: Auto-padded input from {current_in} to {expected_in} channels")
                else:
                    # 如果是序列模型（如 ARWrapper），它可能需要 T_out 等参数
                    # 尝试调用 autoregressive_predict 或者 forward
                    if hasattr(model, "autoregressive_predict"):
                        # 这是一个 smoke test，我们手动构建一次 forward 调用
                        # 为了避免 forward(x) 抛出 4D input error，我们直接用单帧输入测试基本通路
                        # 或者显式调用 autoregressive_predict
                        pass  # 保持 model_input 为 5D，但下面调用可能需要 trick

            # 使用混合精度上下文
            amp_cfg = getattr(getattr(self.config, "training", None), "amp", None)
            amp_enabled = (
                bool(getattr(amp_cfg, "enabled", False))
                if amp_cfg is not None
                else False
            )
            autocast_dtype = getattr(self, "autocast_dtype", torch.bfloat16)

            with autocast(
                device_type=self.device.type, dtype=autocast_dtype, enabled=amp_enabled
            ):
                # 简单调用，不涉及复杂逻辑
                try:
                    # 针对 ARWrapper 特殊处理：如果是 5D 输入，forward 会报错
                    if (
                        hasattr(model, "_unified_interface")
                        and model._unified_interface
                        and model_input.dim() == 5
                    ):
                        # 方案A: 仅测试单帧 forward 能力
                        out = model(model_input[:, 0])
                        target_to_compare = target_seq[:, 0]
                        # 方案B: 如果非要测序列，需调用 autoregressive_predict
                        # out = model.autoregressive_predict(model_input, T_out=2, teacher=target_seq[:,:2])
                    else:
                        out = model(model_input, target_seq)
                except TypeError:
                    # 再次尝试单参数调用
                    if (
                        hasattr(model, "_unified_interface")
                        and model._unified_interface
                        and model_input.dim() == 5
                    ):
                        out = model(model_input[:, 0])
                        target_to_compare = target_seq[:, 0]
                    else:
                        out = model(model_input)

                # Loss
                from ops.losses import rel_l2

                if isinstance(out, dict):
                    out = out["final_pred"]

                # 简单的形状适配
                if out.ndim == 4 and target_seq.ndim == 5:
                    # [B, C, H, W] vs [B, T, C, H, W] -> 取target第一帧
                    target_to_compare = target_seq[:, 0]
                elif out.ndim == 5 and target_seq.ndim == 5:
                    # 都是5D，切片对齐T维度
                    min_t = min(out.shape[1], target_seq.shape[1])
                    out = out[:, :min_t]
                    target_to_compare = target_seq[:, :min_t]
                else:
                    target_to_compare = target_seq

                loss = rel_l2(out, target_to_compare)

            # Backward
            self.scaler.scale(loss).backward() if self.scaler else loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # 显存报告
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                mem_peak = torch.cuda.max_memory_allocated()
                mem_end = torch.cuda.memory_allocated()
                self.logger.info(
                    f"🚬 Smoke Test Memory: Start={mem_start/1e9:.2f}GB, Peak={mem_peak/1e9:.2f}GB, End={mem_end/1e9:.2f}GB"
                )

            self.logger.info("✅ Smoke test passed.")

        except Exception as e:
            self.logger.error(f"❌ Smoke test failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise e
        finally:
            self.cleanup_cuda()

    def train(self):
        """Main training loop"""
        # 可选：启动前进行Smoke Test
        if self._cfg_select("training.smoke_test", default=True):
            self.smoke_test()

        self._training_aborted = False
        self.logger.info("🚀 Starting training...")
        # 明确记录当前模式：空间-only 或 AR
        try:
            ar_cfg = getattr(self.config, "ar", None)
            ar_enabled = (
                bool(getattr(ar_cfg, "enabled", True)) if ar_cfg is not None else True
            )
        except Exception:
            ar_enabled = True
        if not ar_enabled:
            # 当禁用AR且 T_in=T_out=1 时，进一步标注空间-only
            try:
                t_in = int(getattr(self.config.data, "T_in", 1))
                t_out = int(getattr(self.config.data, "T_out", 1))
            except Exception:
                t_in, t_out = 1, 1
            if t_in == 1 and t_out == 1:
                self.logger.info(
                    "🌐 当前训练模式：空间-only（禁用时间预测，T_in=T_out=1）"
                )
            else:
                self.logger.info(
                    f"🌐 当前训练模式：部分时间禁用（AR禁用，T_in={t_in}, T_out={t_out}）"
                )
        else:
            self.logger.info("🕒 当前训练模式：自回归（启用时间预测）")

        start_time = time.time()
        start_epoch = self.current_epoch

        resource_monitor = None

        # 自适应资源调优配置与工具
        # 自适应监控配置：健壮读取，避免缺失键导致异常
        perf_cfg = getattr(self.config, "performance_monitoring", None)

        def _perf_bool(key: str, default: bool) -> bool:
            try:
                return (
                    bool(getattr(perf_cfg, key, default))
                    if perf_cfg is not None
                    else default
                )
            except Exception:
                return default

        def _perf_float(key: str, default: float) -> float:
            try:
                val = (
                    getattr(perf_cfg, key, default) if perf_cfg is not None else default
                )
                return float(val)
            except Exception:
                return default

        def _perf_int(key: str, default: int) -> int:
            try:
                val = (
                    getattr(perf_cfg, key, default) if perf_cfg is not None else default
                )
                return int(val)
            except Exception:
                return default

        adaptive_enabled = _perf_bool("enabled", True)
        gpu_low_threshold = _perf_float("gpu_low_threshold", 0.90)
        iowait_high_threshold = _perf_float("iowait_high_threshold", 0.12)
        cpu_low_threshold = _perf_float("cpu_low_threshold", 0.80)
        nw_step = _perf_int("num_workers_step", 4)
        pf_step = _perf_int("prefetch_factor_step", 2)
        bs_step = _perf_int("batch_size_step", 8)

        def _read_last_resource_record() -> dict | None:
            try:
                metrics_file = self.output_dir / "resource_metrics.jsonl"
                if not metrics_file.exists():
                    return None
                with open(metrics_file, "rb") as f:
                    try:
                        f.seek(-4096, os.SEEK_END)
                    except Exception:
                        pass
                    lines = (
                        f.read().decode("utf-8", errors="ignore").strip().splitlines()
                    )
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
                gpus = rec.get("gpus", []) or []
                avg_gpu_util = (
                    float(np.mean([g.get("util", 0.0) for g in gpus])) / 100.0
                    if gpus
                    else 0.0
                )
                # 计算平均显存占用比例（用于批次增长的安全阈值控制）
                try:
                    mem_ratios = []
                    for g in gpus:
                        used = float(g.get("mem_used_mib", 0.0))
                        total = float(g.get("mem_total_mib", 1.0))
                        if total > 0:
                            mem_ratios.append(used / total)
                    avg_mem_ratio = float(np.mean(mem_ratios)) if mem_ratios else 0.0
                except Exception:
                    avg_mem_ratio = 0.0
                cpu_pct = float(rec.get("cpu", {}).get("percent", 0.0)) / 100.0
                iowait_pct = (
                    float(rec.get("cpu_times_percent", {}).get("iowait", 0.0)) / 100.0
                )

                dl_cfg = getattr(self.config.data, "dataloader", None)
                if dl_cfg is None:
                    return
                cur_nw = int(getattr(dl_cfg, "num_workers", 0) or 0)
                cur_pf = int(getattr(dl_cfg, "prefetch_factor", 0) or 0)
                cur_bs = int(
                    getattr(
                        dl_cfg,
                        "batch_size",
                        getattr(self.config.training, "batch_size", 32),
                    )
                )

                changed = False

                # GPU低利用率且IO等待不高：增加workers/prefetch/batch
                if avg_gpu_util < gpu_low_threshold and iowait_pct < (
                    iowait_high_threshold * 0.8
                ):
                    new_nw = min(cur_nw + nw_step, os.cpu_count() or 96)
                    new_pf = cur_pf + pf_step if new_nw > 0 else cur_pf
                    # 批次增长需考虑显存阈值，避免OOM
                    vram_threshold = float(
                        getattr(
                            getattr(self.config, "hardware", {}), "vram_threshold", 0.94
                        )
                        or 0.94
                    )
                    if avg_mem_ratio < (vram_threshold * 0.92):
                        new_bs = cur_bs + bs_step
                    else:
                        new_bs = cur_bs
                    if new_nw != cur_nw:
                        dl_cfg.num_workers = new_nw
                        changed = True
                    if new_pf != cur_pf and new_nw > 0:
                        dl_cfg.prefetch_factor = new_pf
                        changed = True
                    if new_bs != cur_bs:
                        dl_cfg.batch_size = new_bs
                        self.config.training.batch_size = new_bs
                        changed = True
                        self.logger.info(
                            f"⚙️ 自适应↑ GPU低利用率: workers {cur_nw}->{new_nw}, prefetch {cur_pf}->{new_pf}, batch {cur_bs}->{new_bs} (avg_mem_ratio={avg_mem_ratio:.3f} < {vram_threshold*0.90:.3f})"
                        )
                    else:
                        self.logger.info(
                            f"⚖️ 批次未增长：avg_mem_ratio={avg_mem_ratio:.3f} 接近阈值，避免显存溢出"
                        )

                # IO等待偏高：下调workers，提升prefetch以缓冲IO
                if iowait_pct > iowait_high_threshold:
                    new_nw = max(cur_nw - max(1, nw_step // 2), 0)
                    new_pf = max(cur_pf, pf_step)
                    if new_nw != cur_nw:
                        dl_cfg.num_workers = new_nw
                        changed = True
                    if new_pf != cur_pf and new_nw > 0:
                        dl_cfg.prefetch_factor = new_pf
                        changed = True
                    self.logger.info(
                        f"⚙️ 自适应↓ IO等待偏高: workers {cur_nw}->{new_nw}, prefetch {cur_pf}->{new_pf}"
                    )

                # CPU低利用率且GPU也低：增加workers
                if cpu_pct < cpu_low_threshold and avg_gpu_util < gpu_low_threshold:
                    new_nw = min(cur_nw + nw_step, os.cpu_count() or 96)
                    if new_nw != cur_nw:
                        dl_cfg.num_workers = new_nw
                        changed = True
                        self.logger.info(
                            f"⚙️ 自适应↑ CPU低利用率: workers {cur_nw}->{new_nw}"
                        )

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
            bm = getattr(self.config, "benchmark", None)
            if (
                bm is not None
                and bool(getattr(bm, "enabled", False))
                and bool(getattr(bm, "run_before_training", True))
            ):
                nb = int(getattr(bm, "num_batches", 50) or 50)
                try:
                    self.run_quick_benchmark(nb)
                except Exception as _bm_err:
                    self.logger.debug(f"基准测试跳过: {_bm_err}")
        except Exception:
            pass

        try:
            for epoch in range(start_epoch, self.config.training.epochs):
                epoch_start_time = time.time()
                try:
                    self._update_phase(epoch)
                except Exception as e:
                    # Fix 2: Curriculum update failure is critical unless ignored
                    ignore_error = self._cfg_select(
                        "debug.ignore_phase_update_error", default=False
                    )
                    if ignore_error:
                        self.logger.warning(
                            f"⚠️ 课程学习阶段更新失败 (已忽略): {e}", exc_info=True
                        )
                    else:
                        self.logger.exception(f"❌ 课程学习阶段更新失败: {e}")
                        raise

                # 训练
                train_loss = self.train_epoch(epoch)

                # 验证（每个epoch都执行，确保history包含验证项）
                val_loss, val_metrics, sample_batch = self.validate_epoch(epoch)

                # 记录历史
                self.training_history["train_losses"].append(train_loss)
                self.training_history["val_losses"].append(val_loss)
                self.training_history["val_metrics"].append(val_metrics)
                self.training_history["learning_rates"].append(
                    self.optimizer.param_groups[0]["lr"]
                )
                self.training_history["epochs"].append(epoch)

                # 记录课程学习阶段
                current_T_out = self.get_current_T_out(epoch)
                self.training_history["curriculum_stages"].append(
                    {
                        "epoch": epoch,
                        "T_out": current_T_out,
                        "stage": self.current_stage,
                    }
                )

                # 检查是否为最佳模型（支持min_delta避免微小噪声触发）
                min_delta = 0.0
                try:
                    es_cfg = getattr(self.config.training, "early_stopping", None)
                    if es_cfg is not None:
                        min_delta = float(getattr(es_cfg, "min_delta", 0.0) or 0.0)
                except Exception:
                    min_delta = 0.0

                # Fix B: Validate val_loss before updating best
                import math

                is_valid = (val_loss is not None) and math.isfinite(val_loss)
                is_best = is_valid and (val_loss < (self.best_val_loss - min_delta))

                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                elif not is_valid:
                    if dist.is_available() and dist.is_initialized():
                        if dist.get_rank() == 0:
                            self.logger.warning(
                                f"⚠️ Invalid val_loss ({val_loss}), skipping best update and patience count"
                            )
                    else:
                        self.logger.warning(
                            f"⚠️ Invalid val_loss ({val_loss}), skipping best update and patience count"
                        )
                else:
                    self.patience_counter += 1

                # 保存检查点
                self.save_checkpoint(epoch, is_best)

                # 生成可视化（遵循配置开关，降低额外开销）
                viz_enabled = False
                try:
                    viz_enabled = bool(
                        getattr(self.config.visualization, "enabled", False)
                    )
                except Exception:
                    viz_enabled = False
                try:
                    viz_every = int(
                        self._cfg_select(
                            "logging.visualization.save_samples_every_n_epochs",
                            "visualization.save_samples_every_n_epochs",
                            default=10,
                        )
                        or 10
                    )
                except Exception:
                    viz_every = 10
                if viz_enabled and (((epoch + 1) % max(1, viz_every) == 0) or is_best):
                    self.create_visualizations(sample_batch, epoch)
                    try:
                        from tools.validation.validate_position_encoding_alignment import (
                            validate_alignment,
                        )

                        hr_h = int(
                            self._cfg_select("model.img_size", default=128) or 128
                        )
                        hr_w = hr_h
                        scale = int(
                            self._cfg_select(
                                "data.observation.scale_factor",
                                "observation.sr.scale_factor",
                                default=4,
                            )
                            or 4
                        )
                        ok, err = validate_alignment(hr_h, hr_w, scale)
                        pe_report = {
                            "epoch": int(epoch),
                            "aligned": bool(ok),
                            "max_abs_error": float(err),
                            "hr_size": [hr_h, hr_w],
                            "scale": int(scale),
                        }
                        out_path = (
                            self.output_dir
                            / "test_visualizations"
                            / "position_encoding_alignment.json"
                        )
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        import json as _json

                        with open(out_path, "w") as f:
                            _json.dump(pe_report, f, indent=2)
                        self.logger.info(
                            f"📐 Position encoding alignment: aligned={ok}, max_abs_error={err:.2e}"
                        )
                    except Exception as _pe_err:
                        self.logger.warning(
                            f"Position encoding alignment check failed: {_pe_err}"
                        )

                # 记录到TensorBoard
                self.writer.add_scalar("Val/Loss", val_loss, epoch)
                try:
                    if self.device.type == "cuda":
                        self.writer.add_scalar(
                            "Resources/GPU_PeakAllocated_GB",
                            float(torch.cuda.max_memory_allocated() / 1024**3),
                            epoch,
                        )
                        self.writer.add_scalar(
                            "Resources/GPU_PeakReserved_GB",
                            float(torch.cuda.max_memory_reserved() / 1024**3),
                            epoch,
                        )
                    if getattr(self, "_process", None) is not None:
                        self.writer.add_scalar(
                            "Resources/CPU_Percent",
                            float(self._process.cpu_percent(interval=None)),
                            epoch,
                        )
                        self.writer.add_scalar(
                            "Resources/SystemMem_Percent",
                            float(psutil.virtual_memory().percent),
                            epoch,
                        )
                except Exception:
                    pass
                # 分量记录：若存在则写入
                try:
                    if isinstance(val_metrics, dict):
                        if "dc_loss" in val_metrics:
                            self.writer.add_scalar(
                                "Val/DC_Loss", float(val_metrics["dc_loss"]), epoch
                            )
                        if "spectral_loss" in val_metrics:
                            self.writer.add_scalar(
                                "Val/Spectral_Loss",
                                float(val_metrics["spectral_loss"]),
                                epoch,
                            )
                        if "rel_l2" in val_metrics:
                            self.writer.add_scalar(
                                "Val/RelL2", float(val_metrics["rel_l2"]), epoch
                            )
                        if "mae" in val_metrics:
                            self.writer.add_scalar(
                                "Val/MAE", float(val_metrics["mae"]), epoch
                            )
                except Exception:
                    pass

                epoch_time = time.time() - epoch_start_time
                try:
                    rec = {
                        "epoch": int(epoch),
                        "time_sec": float(epoch_time),
                        "throughput_samples_per_sec": float(
                            self._perf_samples
                            / max(
                                1e-6,
                                (
                                    self._perf_fetch_time
                                    + self._perf_data_time
                                    + self._perf_compute_time
                                ),
                            )
                        ),
                        "gpu_peak_allocated_gb": 0.0,
                        "gpu_peak_reserved_gb": 0.0,
                        "cpu_percent": 0.0,
                        "system_memory_percent": 0.0,
                        "iowait_percent": 0.0,
                    }
                    if self.device.type == "cuda":
                        try:
                            rec["gpu_peak_allocated_gb"] = float(
                                torch.cuda.max_memory_allocated() / 1024**3
                            )
                            rec["gpu_peak_reserved_gb"] = float(
                                torch.cuda.max_memory_reserved() / 1024**3
                            )
                        except Exception:
                            pass
                    if getattr(self, "_process", None) is not None:
                        try:
                            rec["cpu_percent"] = float(
                                self._process.cpu_percent(interval=None)
                            )
                            vm = psutil.virtual_memory()
                            rec["system_memory_percent"] = float(vm.percent)
                            ctp = psutil.cpu_times_percent(interval=None)
                            rec["iowait_percent"] = float(getattr(ctp, "iowait", 0.0))
                        except Exception:
                            pass
                    with open(self.output_dir / "resources_epoch.jsonl", "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception as _ep_err:
                    self.logger.debug(f"资源记录写入失败: {_ep_err}")
                try:
                    tl_unscaled = float(
                        getattr(self, "_last_train_loss_unscaled", float("nan"))
                    )
                except Exception:
                    tl_unscaled = float("nan")
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{self.config.training.epochs} | "
                    f"Train Loss(scaled): {train_loss:.6f} | "
                    f"Train Loss(unscaled): {tl_unscaled:.6f} | "
                    f"Val Loss(NAR): {val_loss:.6f} | "
                    f"Val Loss(TF): {val_metrics.get('val_loss_tf', float('nan')):.6f} | "
                    f"Best: {self.best_val_loss:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                # 早停：patience 与 min_delta
                try:
                    es_cfg = getattr(self.config.training, "early_stopping", None)
                    if es_cfg and bool(getattr(es_cfg, "enabled", False)):
                        patience = int(getattr(es_cfg, "patience", 50))
                        if self.patience_counter >= patience:
                            self.logger.info(
                                f"⏹️ 早停触发: patience={patience}, best_val_loss={self.best_val_loss:.6f}, last_val_loss={val_loss:.6f}"
                            )
                            break
                except Exception as _es_err:
                    self.logger.debug(f"早停检查失败，已跳过: {_es_err}")

                # 写入每epoch资源JSONL

                # 资源监控指标写入与自适应调优

                # 更新学习率（仅当本epoch中发生过optimizer.step时才步进）
                try:
                    if hasattr(self, "scheduler") and self.scheduler is not None:
                        if int(getattr(self, "_epoch_opt_steps", 0) or 0) > 0:
                            self.scheduler.step()
                        else:
                            # 避免PyTorch关于步进顺序的警告：若未发生优化步则跳过调度器步进
                            self.logger.debug(
                                "本epoch未发生optimizer.step，跳过scheduler.step() 以避免警告"
                            )
                except Exception as _sch_err:
                    self.logger.warning(f"学习率调度器步进失败，已跳过: {_sch_err}")

                # 保存训练历史
                with open(self.output_dir / "training_history.json", "w") as f:
                    json.dump(self.training_history, f, indent=2)

        except KeyboardInterrupt:
            self.logger.info("⚠️ 训练被用户中断")
            self._training_aborted = True
        except Exception as e:
            self.logger.error(f"❌ 训练过程中出现错误: {e}")
            traceback.print_exc()
            self._training_aborted = True
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
            self.logger.info(
                f"🏁 Training finished, total time: {total_time/3600:.2f} hours"
            )

            # 在训练完成后，根据配置决定是否进行最终测试
            try:
                testing_enabled = bool(getattr(self.config.testing, "enabled", True))
                run_final_test = bool(
                    getattr(self.config.testing, "run_final_test", True)
                )
            except Exception:
                testing_enabled, run_final_test = True, True

            # 如果训练被中止，强制跳过测试
            if getattr(self, "_training_aborted", False):
                self.logger.warning("⚠️ 训练异常终止，跳过最终测试以保护环境")
                testing_enabled = False
                run_final_test = False

            if testing_enabled and run_final_test:
                # 兜底：确保模型已初始化
                try:
                    if not hasattr(self, "model") or self.model is None:
                        self.logger.info(
                            "ℹ️ Model not initialized before final test; initializing now"
                        )
                        self.setup_model()
                except Exception as _m_init_err:
                    self.logger.warning(
                        f"⚠️ Model initialization before final test failed: {_m_init_err}"
                    )

                best_ckpt_path = self.output_dir / "best.ckpt"
                if best_ckpt_path.exists():
                    self.logger.info("📊 Using best model for final test evaluation...")
                    ok = False
                    try:
                        ok = bool(self.load_checkpoint(str(best_ckpt_path)))
                    except Exception as _load_err:
                        self.logger.warning(
                            f"Failed to load best checkpoint: {_load_err}"
                        )
                        ok = False
                    if not ok:
                        self.logger.info(
                            "ℹ️ Fallback to current model for final test evaluation"
                        )
                    final_test_metrics = self.test_epoch()

                    # 保存测试结果
                    test_results = {
                        "final_test_metrics": final_test_metrics,
                        "test_time": time.time(),
                        "model_path": str(best_ckpt_path),
                    }

                    # 转换numpy类型为JSON可序列化类型
                    test_results = convert_numpy_types(test_results)

                    with open(self.output_dir / "test_results.json", "w") as f:
                        json.dump(test_results, f, indent=2)

                    self.logger.info("✅ Final test results saved to test_results.json")

                    # 生成测试阶段可视化
                    self.logger.info("🎨 Generating test-phase visualizations...")
                    self.create_test_visualizations(final_test_metrics)
                else:
                    self.logger.info(
                        "ℹ️ Best checkpoint not found, using current model for final test evaluation"
                    )
                    final_test_metrics = self.test_epoch()
                    test_results = {
                        "final_test_metrics": final_test_metrics,
                        "test_time": time.time(),
                        "model_path": "current_model",
                    }
                    test_results = convert_numpy_types(test_results)
                    with open(self.output_dir / "test_results.json", "w") as f:
                        json.dump(test_results, f, indent=2)
                    self.logger.info("✅ Final test results saved to test_results.json")
                    self.logger.info("🎨 Generating test-phase visualizations...")
                    self.create_test_visualizations(final_test_metrics)
            else:
                self.logger.info(
                    "⏭️ testing.disabled; skip all test-phase visualizations"
                )

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
                generate_paper = bool(
                    getattr(self.config, "paper_package", {}).get("auto_generate", True)
                )
            except Exception:
                generate_paper = True
            if generate_paper:
                try:
                    # 直接调用生成器脚本入口
                    from tools.generate_paper_package import PaperPackageGenerator

                    # 合并后的配置快照写入 paper_package/configs
                    paper_root = Path("paper_package")
                    paper_root.mkdir(exist_ok=True, parents=True)
                    cfg_out = paper_root / "configs" / "config_merged.yaml"
                    cfg_out.parent.mkdir(parents=True, exist_ok=True)
                    with open(cfg_out, "w") as f:
                        yaml_dump = OmegaConf.to_yaml(self.config)
                        f.write(yaml_dump)
                    generator = PaperPackageGenerator(self.config, paper_root)
                    generator.generate_package()
                    self.logger.info("📦 已自动生成论文材料包")
                except Exception as _pp_err:
                    self.logger.warning(f"论文材料自动生成失败: {_pp_err}")

            # 在分布式环境下，显式销毁进程组避免资源泄漏
            try:
                if (
                    getattr(self, "distributed", False)
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.destroy_process_group()
                    self.logger.info("🧹 已销毁分布式进程组")
            except Exception as e:
                self.logger.warning(f"⚠️ 销毁分布式进程组失败: {e}")

            self.writer.close()
            # 显式清理 DataLoader 以避免解释器关闭阶段线程创建错误
            try:
                if hasattr(self, "train_loader"):
                    self.train_loader = None
                if hasattr(self, "val_loader"):
                    self.val_loader = None
                if hasattr(self, "test_loader"):
                    self.test_loader = None
                import gc

                gc.collect()
            except Exception as _dl_err:
                self.logger.debug(f"DataLoader cleanup skipped: {_dl_err}")


# 注意：convert_numpy_types 已在文件顶部定义，避免重复定义


def main():
    """主函数"""
    import argparse
    import os as _os
    import traceback as _tb
    from datetime import datetime as _dt

    parser = argparse.ArgumentParser(description="真实扩散-反应数据AR训练")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "thesis_paper"
            / "configs"
            / "ar_paper_aligned.yaml"
        ),
        help="配置文件路径",
    )
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument(
        "--seeds", type=str, default=None, help="逗号分隔的随机种子列表，如 42,123,456"
    )

    # 新增模型选择参数
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型架构名称（如 swin_unet, unet, fno2d, segformer 等）",
    )
    parser.add_argument("--list-models", action="store_true", help="列出所有可用模型")
    parser.add_argument(
        "--smoke-all", action="store_true", help="对所有空间模型进行冒烟测试"
    )
    parser.add_argument(
        "--target-params-m", type=float, default=None, help="目标参数量(百万)，如10.0"
    )
    parser.add_argument(
        "--tolerance-m", type=float, default=0.5, help="参数量容差(百万)"
    )
    parser.add_argument(
        "--use_liif_decoder", action="store_true", help="强制启用LIIF解码器增强坐标消费"
    )
    parser.add_argument(
        "--test-only",
        "--test_only",
        dest="test_only",
        action="store_true",
        help="Only run test using checkpoint and exit",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="运行模式: train 或 test (默认 train)",
    )
    parser.add_argument(
        "--ckpt",
        "--ckpt_path",
        dest="ckpt",
        type=str,
        default="best",
        help="测试模式下加载的检查点: best/last/PATH (默认 best)",
    )

    # 使用 parse_known_args 允许接收 Hydra 风格的 overrides (key=value)
    args, unknown = parser.parse_known_args()

    # 收集 overrides
    overrides = []
    for arg in unknown:
        if arg.startswith("--"):
            print(f"Warning: Unknown flag {arg} ignored.")
        elif "=" in arg:
            overrides.append(arg)
        else:
            print(f"Warning: Unknown argument {arg} ignored.")

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

    if args.smoke_all:
        base_cfg = OmegaConf.load(args.config) if args.config else None
        if base_cfg is None:
            tmp_trainer = RealDataARTrainer(None)
            base_cfg = tmp_trainer.config
        models = list_models()
        to_run = []
        for m in models:
            info = get_model_info(m)
            fp = info.get("file_path") if info else None
            if fp and ("/models/spatial/" in fp.replace("\\", "/")):
                to_run.append(m)
        results = []
        for m in to_run:
            try:
                cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
                # Apply overrides to cfg
                if overrides:
                    try:
                        override_conf = OmegaConf.from_dotlist(overrides)
                        cfg = OmegaConf.merge(cfg, override_conf)
                    except Exception:
                        pass

                try:
                    cfg.model.name = m
                except Exception:
                    pass
                if args.target_params_m is not None:
                    cfg.model_budget = {
                        "target_params_m": float(args.target_params_m),
                        "tolerance_m": float(args.tolerance_m),
                        "auto_tune": True,
                    }
                try:
                    old = str(getattr(cfg.experiment, "name", "AR-DR2D-Smoke"))
                except Exception:
                    old = "AR-DR2D-Smoke"
                new_name = f"{old}-smoke-{m}"
                cfg.experiment.name = new_name
                tmp_dir = Path("runs") / "tmp_configs"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_cfg_path = tmp_dir / f"{new_name}.yaml"
                with open(tmp_cfg_path, "w") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                trainer = RealDataARTrainer(str(tmp_cfg_path), model_name=m)
                trainer.run_quick_benchmark(
                    num_batches=3, outfile=f"benchmark_{m}.json"
                )
                model_for_params = trainer.get_model()
                if hasattr(model_for_params, "module"):
                    model_for_params = model_for_params.module
                pc = sum(p.numel() for p in model_for_params.parameters())
                results.append({"model": m, "params_m": pc / 1e6, "status": "success"})
                print(f"✅ Smoke test passed for {m}: {pc/1e6:.2f}M")
            except Exception as e:
                print(f"❌ Smoke test failed for {m}: {e}")
                results.append({"model": m, "status": "failed", "error": str(e)})
        out_path = Path("runs") / "smoke_all_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ 冒烟测试完成，共 {len(results)} 个模型，结果写入 {out_path}")
        return

    # 如果提供了多种子列表，则顺序运行并聚合结果
    env_snapshot = {}
    try:
        # 将关键环境变量记录到标准输出，便于分布式诊断
        env_snapshot = {
            "WORLD_SIZE": _os.environ.get("WORLD_SIZE"),
            "RANK": _os.environ.get("RANK"),
            "LOCAL_RANK": _os.environ.get("LOCAL_RANK"),
            "CUDA_VISIBLE_DEVICES": _os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        print(f"[Env] DDP 环境变量快照: {env_snapshot}")

        if args.seeds:
            try:
                seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            except Exception:
                seed_list = []
            if len(seed_list) < 1:
                # 回退到配置中的 seeds
                base_cfg = OmegaConf.load(args.config) if args.config else None
                if base_cfg is not None and hasattr(base_cfg.experiment, "seeds"):
                    seed_list = list(base_cfg.experiment.seeds)
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

                # Apply overrides to cfg before writing
                if overrides:
                    try:
                        print(f"🔧 [Seed {s}] 应用命令行覆盖配置: {overrides}")
                        override_conf = OmegaConf.from_dotlist(overrides)
                        cfg = OmegaConf.merge(cfg, override_conf)
                    except Exception as e:
                        print(f"⚠️ [Seed {s}] 应用命令行覆盖配置失败: {e}")

                # 更新种子与实验名
                try:
                    old_name = str(cfg.experiment.name)
                except Exception:
                    old_name = "Real-DR2D-AR"
                if base_name is None:
                    base_name = old_name.split("-s")[0]
                new_name = f"{base_name}-s{s}"
                cfg.experiment.name = new_name
                cfg.experiment.seed = int(s)
                # 写入临时配置文件
                tmp_dir = Path("runs") / "tmp_configs"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_cfg_path = tmp_dir / f"{new_name}.yaml"
                with open(tmp_cfg_path, "w") as f:
                    f.write(OmegaConf.to_yaml(cfg))

                # 运行该种子训练
                trainer = RealDataARTrainer(str(tmp_cfg_path), model_name=args.model)
                if args.resume:
                    trainer.load_checkpoint(args.resume)
                trainer.train()

                try:
                    test_json = Path(trainer.output_dir) / "test_results.json"
                    if test_json.exists():
                        with open(test_json) as f:
                            res = json.load(f)
                            per_seed_results.append(
                                {
                                    "seed": s,
                                    "metrics": res.get("final_test_metrics", {}),
                                }
                            )
                except Exception:
                    pass

            # 聚合均值±标准差
            try:
                # 统一所有指标键
                keys = set()
                for item in per_seed_results:
                    keys.update(item.get("metrics", {}).keys())
                summary = {}
                for k in keys:
                    vals = [
                        float(item["metrics"].get(k, float("nan")))
                        for item in per_seed_results
                    ]
                    vals = [v for v in vals if not (np.isnan(v) or np.isinf(v))]
                    if len(vals) >= 1:
                        mean_v = float(np.mean(vals))
                        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        summary[k] = {"mean": mean_v, "std": std_v, "n": len(vals)}
                out_summary = {
                    "experiment_group": base_name or "Real-DR2D-AR",
                    "seeds": seed_list,
                    "metrics": summary,
                    "timestamp": time.time(),
                }
                out_path = (
                    Path("runs")
                    / f"{(base_name or 'Real-DR2D-AR')}_multi_seed_summary.json"
                )
                with open(out_path, "w") as f:
                    json.dump(out_summary, f, indent=2)
                print(f"✅ 多种子汇总已保存: {out_path}")
            except Exception as _agg_err:
                print(f"⚠️ 多种子汇总失败: {_agg_err}")
        else:
            # 单次训练

            # 1. 解析模式与路径
            is_test_mode = getattr(args, "test_only", False) or (
                getattr(args, "mode", "train") == "test"
            )
            output_dir_override = None
            ckpt_to_load = None

            if is_test_mode:
                resume_path = args.resume
                ckpt_arg = getattr(args, "ckpt", "best")

                # 尝试推断 run_dir 和 ckpt_path
                cand_run_dir = None

                if resume_path:
                    rp = Path(resume_path)
                    if rp.is_dir():
                        cand_run_dir = rp
                    elif rp.is_file():
                        cand_run_dir = rp.parent
                        ckpt_to_load = rp

                # 如果还没确定 ckpt_to_load，但在 run_dir 下找
                if cand_run_dir and not ckpt_to_load:
                    if ckpt_arg == "best":
                        ckpt_to_load = cand_run_dir / "best.ckpt"
                    elif ckpt_arg == "last":
                        ckpt_to_load = cand_run_dir / "last.ckpt"
                    else:
                        # 假设是相对路径或文件名
                        if (cand_run_dir / ckpt_arg).exists():
                            ckpt_to_load = cand_run_dir / ckpt_arg
                        else:
                            ckpt_to_load = Path(ckpt_arg)

                # 如果没有 run_dir，但有 ckpt_arg 是路径
                if not cand_run_dir and not ckpt_to_load:
                    if Path(ckpt_arg).exists():
                        ckpt_to_load = Path(ckpt_arg)
                        cand_run_dir = ckpt_to_load.parent  # 假设

                # 设置 override
                if cand_run_dir:
                    output_dir_override = cand_run_dir

            # 2. 初始化 Trainer
            trainer = RealDataARTrainer(
                args.config,
                model_name=args.model,
                use_liif_decoder=args.use_liif_decoder,
                output_dir_override=output_dir_override if is_test_mode else None,
                skip_optimizer=is_test_mode,
                skip_monitoring=is_test_mode,
                overrides=overrides,
            )

            # 3. 执行测试或训练
            if is_test_mode:
                # 最后的兜底：如果还没找到ckpt，试试 trainer.output_dir 下的 best.ckpt
                if not ckpt_to_load or not Path(ckpt_to_load).exists():
                    cand = trainer.output_dir / "best.ckpt"
                    if cand.exists():
                        ckpt_to_load = cand

                if not ckpt_to_load or not Path(ckpt_to_load).exists():
                    print(
                        "❌ Error: Test mode requested but no valid checkpoint found."
                    )
                    print(f"  --resume: {args.resume}")
                    print(f"  --ckpt: {args.ckpt}")
                    print(f"  Inferred path: {ckpt_to_load}")
                    sys.exit(1)

                print(f"🔍 Loading checkpoint for testing: {ckpt_to_load}")
                if not trainer.load_checkpoint(str(ckpt_to_load)):
                    print(f"❌ Failed to load checkpoint: {ckpt_to_load}")
                    sys.exit(1)

                print("🚀 Starting test-only evaluation...")
                # Ensure model is in eval mode
                if hasattr(trainer, "model") and trainer.model is not None:
                    trainer.model.eval()

                try:
                    final_test_metrics = trainer.test_epoch()

                    # Save results
                    test_results = {
                        "final_test_metrics": final_test_metrics,
                        "test_time": time.time(),
                        "model_path": str(ckpt_to_load),
                    }
                    test_results = convert_numpy_types(test_results)

                    out_json = trainer.output_dir / "test_results.json"
                    with open(out_json, "w") as f:
                        json.dump(test_results, f, indent=2)
                    print(f"✅ Test results saved to {out_json}")

                    # Visualizations
                    print("🎨 Generating visualizations...")
                    trainer.create_test_visualizations(final_test_metrics)

                    # HTML Report
                    trainer.create_final_report()

                    print("🏁 Test-only run completed successfully.")
                    return

                except Exception as e:
                    print(f"❌ Test execution failed: {e}")
                    _tb.print_exc()
                    sys.exit(1)

            # Train Mode
            if args.resume:
                trainer.load_checkpoint(args.resume)
            trainer.train()
    except Exception as _main_err:
        # 捕获顶层异常，按rank写入独立错误日志
        try:
            rank_val = int(_os.environ.get("LOCAL_RANK", _os.environ.get("RANK", "0")))
        except Exception:
            rank_val = 0
        exp_name = None
        try:
            # 尝试从配置文件读取实验名，构造输出目录
            if args.config:
                cfg = OmegaConf.load(args.config)
                exp_name = str(getattr(cfg.experiment, "name", "AR-DR2D-Unknown"))
        except Exception:
            exp_name = "AR-DR2D-Unknown"
        err_dir = Path("runs") / (exp_name or "AR-DR2D-Unknown")
        err_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        err_file = err_dir / f"error_rank{rank_val}_{ts}.log"
        with open(err_file, "w") as f:
            f.write("Top-level exception captured in main()\n")
            f.write("Environment snapshot:\n")
            f.write(json.dumps(env_snapshot, indent=2) + "\n")
            f.write("Traceback:\n")
            f.write(
                "".join(
                    _tb.format_exception(
                        type(_main_err), _main_err, _main_err.__traceback__
                    )
                )
            )
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
    try:
        mp.set_start_method("spawn", force=True)
    except Exception as e:
        print(f"Warning: Failed to set multiprocessing start method: {e}")
    main()
