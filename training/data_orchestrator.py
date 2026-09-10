"""数据编排与一致性验证模块 (Data Orchestrator)

负责解析复杂 Hydra 数据配置、装配空间与时序 DataModule、提取归一化统计量，
以及执行退化算子与观测数据一致性验收。
"""

import json
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from datasets import PDEBenchDataModule
from ops.degradation import verify_degradation_consistency


class DataOrchestrator:
    """数据模块加载与退化一致性验证管理器"""

    @staticmethod
    def setup_data_module(
        config: DictConfig, 
        logger: Optional[logging.Logger] = None
    ) -> Tuple[Any, DataLoader, DataLoader, Optional[DataLoader], Optional[Dict[str, torch.Tensor]]]:
        """解析配置并初始化数据模块与 DataLoader"""
        logger = logger or logging.getLogger(__name__)
        logger.info("Initializing data module via DataOrchestrator...")

        # 处理多级配置嵌套结构 (兼容根 data 字典或空字符串键)
        if hasattr(config, 'data'):
            data_config = config.data
        elif '' in config and 'data' in config['']:
            data_config = config['']['data']
        else:
            data_config = config.get('', config)

        # 检查是否有 datasets.data 配置
        if hasattr(data_config, 'datasets') and hasattr(data_config.datasets, 'data'):
            actual_data_config = data_config.datasets.data
            if hasattr(actual_data_config, '_target_') and 'temporal' in str(actual_data_config._target_):
                from datasets.temporal_pdebench import TemporalPDEBenchDataModule
                data_module = TemporalPDEBenchDataModule(actual_data_config.config)
            else:
                data_module = PDEBenchDataModule(actual_data_config)
        elif hasattr(data_config, 'data'):
            actual_data_config = data_config.data
            if hasattr(actual_data_config, '_target_') and 'temporal' in str(actual_data_config._target_):
                from datasets.temporal_pdebench import TemporalPDEBenchDataModule
                data_module = TemporalPDEBenchDataModule(actual_data_config.config)
            else:
                data_module = PDEBenchDataModule(actual_data_config)
        elif hasattr(data_config, '_target_'):
            if 'temporal' in str(data_config._target_):
                from datasets.temporal_pdebench import TemporalPDEBenchDataModule
                data_module = TemporalPDEBenchDataModule(data_config.config)
            else:
                data_module = PDEBenchDataModule(data_config)
        else:
            data_module = PDEBenchDataModule(data_config)

        # PDEBenchDataModule 具备 setup 方法
        if hasattr(data_module, 'setup'):
            data_module.setup()

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader() if hasattr(data_module, 'test_dataloader') else None
        norm_stats = data_module.get_norm_stats() if hasattr(data_module, 'get_norm_stats') else None

        logger.info(
            f"Data loaded successfully: train_batches={len(train_loader)}, "
            f"val_batches={len(val_loader)}, test_batches={len(test_loader) if test_loader else 0}"
        )
        return data_module, train_loader, val_loader, test_loader, norm_stats

    @staticmethod
    def verify_consistency(
        val_loader: DataLoader,
        config: DictConfig,
        output_dir: Path,
        logger: Optional[logging.Logger] = None
    ) -> bool:
        """验证验证集样本观测退化一致性并落盘报告"""
        logger = logger or logging.getLogger(__name__)
        logger.info("Verifying data consistency...")

        try:
            sample_batch = next(iter(val_loader))
            target = sample_batch['target']
            observation = sample_batch.get(
                'original_observation', 
                sample_batch.get('lr_observation', sample_batch.get('observation'))
            )

            obs_cfg_root = getattr(config, 'observation', {})
            obs_cfg_data = getattr(getattr(config, 'data', {}), 'observation', {})
            obs_cfg = obs_cfg_root if obs_cfg_root else obs_cfg_data

            h_params = sample_batch.get('h_params', {
                'task': 'SR',
                'scale': obs_cfg.get('scale_factor', 2) if hasattr(obs_cfg, 'get') else 2,
                'sigma': obs_cfg.get('blur_sigma', 1.0) if hasattr(obs_cfg, 'get') else 1.0,
                'blur_kernel': obs_cfg.get('kernel_size', 5) if hasattr(obs_cfg, 'get') else 5,
                'boundary': obs_cfg.get('boundary', 'mirror') if hasattr(obs_cfg, 'get') else 'mirror',
                'downsample_interpolation': obs_cfg.get('downsample_interpolation', 'area') if hasattr(obs_cfg, 'get') else 'area',
                'noise_std': obs_cfg.get('noise_std', 0.0) if hasattr(obs_cfg, 'get') else 0.0
            })

            # 处理时序维度
            if target.dim() == 5:
                target_sample = target[:, 0]
                observation_sample = observation[:, 0] if observation is not None else None
            else:
                target_sample = target
                observation_sample = observation

            if observation_sample is None:
                logger.warning("No observation found for consistency check; skipping")
                return True

            consistency_result = verify_degradation_consistency(
                target_sample, observation_sample, h_params
            )
            consistency_error = float(consistency_result.get('mse', 0.0))

            tr_cfg = getattr(config, 'training', getattr(config, 'train', {}))
            tolerance = float(tr_cfg.get('consistency_tolerance', 1e-6) if hasattr(tr_cfg, 'get') else 1e-6)
            passed = consistency_error < tolerance

            if passed:
                logger.info(f"Data consistency verified: MSE = {consistency_error:.2e}")
            else:
                logger.warning(f"Data consistency check failed: MSE = {consistency_error:.2e} (tolerance={tolerance:.2e})")

            # 落盘 consistency_report.json
            report = {
                'mse': consistency_error,
                'tolerance': tolerance,
                'passed': passed,
                'timestamp': time.time()
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'consistency_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)

            return passed

        except Exception as e:
            logger.error(f"Data consistency verification encountered exception: {e}")
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    'mse': None,
                    'tolerance': 1e-6,
                    'passed': False,
                    'skipped': True,
                    'error': str(e),
                    'timestamp': time.time()
                }
                with open(output_dir / 'consistency_report.json', 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
            except Exception:
                pass
            return False
