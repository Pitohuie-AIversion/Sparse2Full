"""训练成果与审计资产管理器 (Training Artifact Manager)

负责运行环境指纹提取、Checkpoint 状态持久化、TensorBoard/WandB 度量推送、
物理场样本可视化渲染、指标 JSONL 追加以及论文交付包 (paper_package) 自动化脚手架。
"""

import os
import sys
import json
import time
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from shutil import copyfile
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

from utils.checkpoint import CheckpointManager
from utils.visualization import ARVisualizer
from src.monitoring import TensorBoardLogger


class TrainingArtifactManager:
    """训练生命周期中的资产、日志、图表与检查点管理器"""

    def __init__(self, output_dir: Path, config: DictConfig, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 检查点管理器
        tr_cfg = getattr(config, 'training', getattr(config, 'train', {}))
        ckpt_cfg = tr_cfg.get('checkpoint', {}) if hasattr(tr_cfg, 'get') else {}
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.output_dir / 'checkpoints',
            max_checkpoints=int(ckpt_cfg.get('max_keep', 3) if hasattr(ckpt_cfg, 'get') else 3),
            save_best=bool(ckpt_cfg.get('save_best', True) if hasattr(ckpt_cfg, 'get') else True)
        )

        # 2. TensorBoard 日志器
        log_cfg = getattr(config, 'logging', {})
        use_tb = bool(log_cfg.get('use_tensorboard', True) if hasattr(log_cfg, 'get') else True)
        self.tb_logger = TensorBoardLogger(self.output_dir / 'tensorboard', enabled=use_tb)
        self.tb_writer = self.tb_logger.writer if self.tb_logger.enabled else None

        # 3. Weights & Biases
        self.use_wandb = False
        if hasattr(log_cfg, 'get') and log_cfg.get('use_wandb', False):
            try:
                import wandb
                wandb.init(
                    project=log_cfg.get('wandb_project', 'pdebench-sparse2full'),
                    name=getattr(getattr(config, 'experiment', None), 'name', 'sparse2full_exp'),
                    config=OmegaConf.to_container(config, resolve=True),
                    dir=str(self.output_dir)
                )
                self.use_wandb = True
            except Exception as e:
                self.logger.warning(f"Wandb initialization failed: {e}")

    def save_env_fingerprint(self) -> None:
        """保存环境指纹（Methodology 3.6 可审计证据）"""
        exp_cfg = getattr(self.config, 'experiment', {})
        seed = exp_cfg.get('seed', None) if hasattr(exp_cfg, 'get') else getattr(exp_cfg, 'seed', None)

        fingerprint = {
            'timestamp': datetime.now().isoformat(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cwd': os.getcwd(),
            'seed': seed
        }

        # Git commit
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            fingerprint['git_commit'] = commit_hash
            status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            fingerprint['git_dirty'] = bool(status)
        except Exception:
            fingerprint['git_commit'] = None
            fingerprint['git_dirty'] = None

        # Pip freeze
        try:
            pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.DEVNULL).decode('utf-8')
            fingerprint['pip_packages'] = pip_freeze.splitlines()
        except Exception:
            fingerprint['pip_packages'] = None

        try:
            with open(self.output_dir / 'env_fingerprint.json', 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, indent=2)
            self.logger.info(f"Environment fingerprint saved to {self.output_dir / 'env_fingerprint.json'}")
        except Exception as e:
            self.logger.warning(f"Failed to save env_fingerprint.json: {e}")

    def log_epoch_results(
        self, 
        epoch: int, 
        train_results: Dict[str, Any], 
        val_results: Dict[str, Any], 
        lr: float
    ) -> None:
        """记录每轮指标到日志、TensorBoard 与 metrics.jsonl"""
        t_loss_obj = train_results.get('total_loss', train_results.get('loss', 0.0))
        v_loss_obj = val_results.get('total_loss', val_results.get('loss', 0.0))
        train_loss = float(t_loss_obj.item() if hasattr(t_loss_obj, 'item') else t_loss_obj)
        val_loss = float(v_loss_obj.item() if hasattr(v_loss_obj, 'item') else v_loss_obj)
        
        rel_l2_raw = val_results.get('rel_l2', 0.0)
        val_rel_l2 = float(rel_l2_raw.mean().item() if hasattr(rel_l2_raw, 'mean') else (rel_l2_raw.item() if hasattr(rel_l2_raw, 'item') else rel_l2_raw))

        self.logger.info(
            f"Epoch {epoch:3d} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Rel-L2: {val_rel_l2:.6f}"
        )

        # TensorBoard
        if self.tb_logger is not None and self.tb_logger.enabled:
            train_scalars = {k: float(v.mean().item() if hasattr(v, 'mean') else (v.item() if hasattr(v, 'item') else v)) for k, v in train_results.items()}
            val_scalars = {k: float(v.mean().item() if hasattr(v, 'mean') else (v.item() if hasattr(v, 'item') else v)) for k, v in val_results.items()}
            self.tb_logger.log_scalars(train_scalars, epoch, prefix='epoch_train')
            self.tb_logger.log_scalars(val_scalars, epoch, prefix='epoch_val')
            self.tb_logger.log_scalars({'lr': lr}, epoch, prefix='epoch_train')
            self.tb_logger.flush()

        # Wandb
        if self.use_wandb:
            try:
                import wandb
                log_dict = {f'train/{k}': float(v) for k, v in train_results.items() if isinstance(v, (int, float))}
                log_dict.update({f'val/{k}': float(v) for k, v in val_results.items() if isinstance(v, (int, float))})
                log_dict['epoch'] = epoch
                log_dict['lr'] = lr
                wandb.log(log_dict)
            except Exception:
                pass

        # 追加写入 metrics.jsonl
        try:
            exp_name = str(getattr(getattr(self.config, 'experiment', None), 'name', 'unnamed'))
            record = {
                'epoch': int(epoch),
                'experiment': exp_name,
                'train': {k: float(v.mean().item() if hasattr(v, 'mean') else (v.item() if hasattr(v, 'item') else v)) for k, v in train_results.items()},
                'val': {k: float(v.mean().item() if hasattr(v, 'mean') else (v.item() if hasattr(v, 'item') else v)) for k, v in val_results.items()},
                'lr': float(lr),
                'timestamp': time.time()
            }
            with open(self.output_dir / 'metrics.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to append metrics.jsonl: {e}")

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: Optional[Any],
        val_results: Dict[str, Any],
        best_val_loss: float,
        is_best: bool,
        global_step: int = 0
    ) -> None:
        """保存检查点"""
        model_to_save = getattr(model, 'module', model)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'val_results': val_results,
            'config': self.config,
            'global_step': global_step
        }
        if scaler is not None and hasattr(scaler, 'state_dict'):
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        self.checkpoint_manager.save_checkpoint(checkpoint, is_best, epoch)

    def save_training_samples(
        self, 
        epoch: int, 
        val_batch: Dict[str, Any], 
        model: nn.Module, 
        batch_processor: Any,
        device: torch.device
    ) -> None:
        """渲染并落盘标准四栏物理场可视化图 (obs_gt_pred_err.png)"""
        tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        if not bool(tr_cfg.get('save_samples', True) if hasattr(tr_cfg, 'get') else True):
            return

        plot_interval = int(tr_cfg.get('plot_interval', 50) if hasattr(tr_cfg, 'get') else 50) or 50
        if epoch % plot_interval != 0:
            return

        try:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in val_batch.items()}
            model_input = batch_processor.build_model_input(batch, model)

            with torch.no_grad():
                pred = model(model_input)

            target = batch.get('target', None)
            if target is None:
                return
            target = batch_processor.prepare_target(target, pred.shape)

            save_dir = self.output_dir / 'samples' / f'epoch_{epoch:04d}'
            save_dir.mkdir(parents=True, exist_ok=True)
            viz = ARVisualizer(save_dir)

            observation = batch.get('observation', batch.get('baseline'))
            if observation is None:
                return
            if observation.dim() == 5:
                observation = observation[:, -1]

            max_samples = int(tr_cfg.get('max_samples', 4) if hasattr(tr_cfg, 'get') else 4) or 4
            out_path = save_dir / "obs_gt_pred_err.png"
            viz_path = viz.plot_obs_gt_pred_err_horizontal(
                observation=observation.detach().cpu(),
                targets=target.detach().cpu(),
                predictions=pred.detach().cpu(),
                save_path=str(out_path),
                num_samples=max_samples,
                channel=int(tr_cfg.get('viz_channel', 0) if hasattr(tr_cfg, 'get') else 0) or 0
            )
            self.logger.info(f"Saved standardized 4-column viz to {viz_path}")

            # TensorBoard 进阶可视化
            if self.tb_logger is not None and self.tb_logger.enabled:
                self.tb_logger.log_flow_field_grid(
                    gt_field=target[0],
                    pred_field=pred[0],
                    input_sparse=observation[0],
                    step=epoch,
                    tag="Validation/FlowField_Grid"
                )
                self.tb_logger.log_error_histogram(
                    gt=target,
                    pred=pred,
                    step=epoch,
                    tag="Validation/Error_Histogram"
                )
                self.tb_logger.log_energy_spectrum(
                    gt_field=target[0],
                    pred_field=pred[0],
                    step=epoch,
                    tag="Validation/Energy_Spectrum"
                )
        except Exception as e:
            self.logger.warning(f"Failed to save training samples: {e}")

    def close(
        self,
        best_val_loss: float,
        best_val_metrics: Dict[str, Any],
        train_time: float,
        val_time: float,
        model: Optional[nn.Module] = None
    ) -> None:
        """收尾：关闭写入器、写入资源摘要与生成论文成果交付包 (paper_package)"""
        if hasattr(self, 'tb_logger') and self.tb_logger is not None:
            self.tb_logger.close()
        elif self.tb_writer is not None:
            self.tb_writer.close()

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        # 1. 资源统计 (resource_stats.json)
        resource = {}
        try:
            model_unwrapped = getattr(model, 'module', model) if model is not None else None
            param_count = sum(p.numel() for p in model_unwrapped.parameters()) if model_unwrapped is not None else 0
            flops_g = float(model_unwrapped.compute_flops() / 1e9) if (model_unwrapped and hasattr(model_unwrapped, 'compute_flops')) else None
            max_cuda_mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

            resource = {
                'params': int(param_count),
                'flops_g': flops_g,
                'max_cuda_mem_bytes': max_cuda_mem,
                'train_time_sec': float(train_time),
                'val_time_sec': float(val_time)
            }
            with open(self.output_dir / 'resource_stats.json', 'w', encoding='utf-8') as f:
                json.dump(resource, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to write resource_stats.json: {e}")

        # 2. 论文材料包骨架 (paper_package)
        try:
            paper_dir = self.output_dir / 'paper_package'
            (paper_dir / 'configs').mkdir(parents=True, exist_ok=True)
            (paper_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (paper_dir / 'metrics').mkdir(parents=True, exist_ok=True)
            (paper_dir / 'figs').mkdir(parents=True, exist_ok=True)
            (paper_dir / 'scripts').mkdir(parents=True, exist_ok=True)

            # 配置快照
            config_merged = self.output_dir / 'config_merged.yaml'
            if config_merged.exists():
                copyfile(config_merged, paper_dir / 'configs' / 'config.yaml')

            # 写入指标摘要
            exp_name = str(getattr(getattr(self.config, 'experiment', None), 'name', 'unnamed'))
            summary = {
                'best_val_loss': float(best_val_loss),
                'best_val_metrics': {k: float(v) for k, v in best_val_metrics.items() if isinstance(v, (int, float))},
                'resource': resource,
                'experiment': exp_name
            }
            with open(paper_dir / 'metrics' / 'experiment_metrics.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

            metrics_jsonl = self.output_dir / 'metrics.jsonl'
            dc_report = self.output_dir / 'consistency_report.json'
            if metrics_jsonl.exists():
                copyfile(metrics_jsonl, paper_dir / 'metrics' / 'metrics.jsonl')
            if dc_report.exists():
                copyfile(dc_report, paper_dir / 'metrics' / 'consistency_report.json')

            reproduce_sh = paper_dir / 'scripts' / 'reproduce.sh'
            if not reproduce_sh.exists():
                reproduce_sh.write_text('#!/usr/bin/env bash\nset -e\npython train.py +experiment.output_dir="runs/reproduce"\n', encoding='utf-8')
        except Exception as e:
            self.logger.warning(f"Failed to scaffold paper_package: {e}")
