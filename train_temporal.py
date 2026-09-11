#!/usr/bin/env python3
"""时序物理场预测训练脚本 (Temporal PDE Trainer Facade)

遵循 Sparse2Full 解耦训练架构规范：
- 核心流水线解耦至 `training/` 正交子系统
- 优化器与混合精度构建托管至 `EngineBuilder`
- 产物归档、环境指纹与交付包托管至 `TrainingArtifactManager`
- 课程学习与损失权重调度托管至 `CurriculumScheduler`
- 保持对既有时序实验配置与数据流 100% 向后兼容
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.temporal_pdebench import TemporalPDEBenchDataModule
from models.registry import create_model as registry_create_model
from ops.losses import ARLoss, SpectralLoss, DCLoss
from utils.metrics import compute_metrics, MetricsCalculator
from utils.visualization import TemporalVisualizer
from utils.logger import setup_logger
from training import (
    BatchProcessor,
    CurriculumScheduler,
    EngineBuilder,
    TrainingArtifactManager
)


class TemporalTrainer:
    """时序 PDE 解耦训练器 (Facade)"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        exp_cfg = getattr(config, 'experiment', config)
        device_str = getattr(exp_cfg, 'device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        
        # 设置随机种子与确定性
        seed = int(getattr(exp_cfg, 'seed', 42))
        self._set_random_seed(seed)
        
        # 创建输出目录与基础日志
        output_base = Path(getattr(exp_cfg, 'output_dir', 'runs'))
        exp_name = str(getattr(exp_cfg, 'name', 'temporal_pde_experiment'))
        self.output_dir = output_base / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('train_temporal', self.output_dir / 'train.log')
        self.logger.info(f"Temporal training started with config:\n{OmegaConf.to_yaml(config)}")
        
        # 1. 初始化核心解耦子系统
        self.batch_processor = BatchProcessor(config, self.logger)
        self.curriculum_scheduler = CurriculumScheduler(config)
        self.artifact_manager = TrainingArtifactManager(self.output_dir, config, self.logger)
        self.artifact_manager.save_env_fingerprint()
        
        # 2. 初始化数据模块
        self._init_data()
        
        # 3. 初始化模型
        self._init_model()
        
        # 4. 初始化优化器、调度器与混合精度
        self._init_optimizer()
        self._init_scheduler()
        self._init_amp()
        
        # 5. 初始化损失函数、评估度量与可视化器
        self._init_losses()
        self._init_metrics()
        self._init_visualizer()
        
        # 训练状态管理
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.curriculum_stage = 0
        self._init_curriculum()
        
        self.logger.info(f"TemporalTrainer initialized successfully. Output dir: {self.output_dir}")

    def _set_random_seed(self, seed: int) -> None:
        """设置全局确定性随机种子"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _init_data(self) -> None:
        """初始化时序数据模块与 Loader"""
        self.logger.info("Initializing temporal data module...")
        data_cfg = getattr(self.config, 'data', self.config)
        target = getattr(data_cfg, '_target_', None) or (data_cfg.get('_target_') if isinstance(data_cfg, (dict, DictConfig)) else None)
        
        if target and "RealDiffusionReactionDataModule" in str(target):
            from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
            self.data_module = RealDiffusionReactionDataModule(self.config)
            self.data_module.setup(None)
            self.train_loader = self.data_module.train_dataloader()
            self.val_loader = self.data_module.val_dataloader()
            try:
                self.test_loader = self.data_module.test_dataloader()
            except Exception:
                self.test_loader = self.val_loader
        else:
            self.data_module = TemporalPDEBenchDataModule(data_cfg)
            self.train_loader = self.data_module.train_dataloader()
            self.val_loader = self.data_module.val_dataloader()
            self.test_loader = self.data_module.test_dataloader()
        self.logger.info(
            f"Data loaded: Train={len(self.train_loader)}, "
            f"Val={len(self.val_loader)}, Test={len(self.test_loader)}"
        )

    def _init_model(self) -> None:
        """从注册表安全加载时序模型"""
        self.logger.info("Initializing temporal model...")
        model_cfg = self.config.model
        model_name = getattr(model_cfg, 'name', 'SwinTemporalNAR')
        
        # 提取模型参数
        params = {}
        if hasattr(model_cfg, 'params'):
            params = dict(model_cfg.params)
        elif isinstance(model_cfg, (dict, DictConfig)):
            params = {k: v for k, v in dict(model_cfg).items() if k != 'name'}
        
        # 注入时序配置
        if hasattr(self.config, 'temporal') and hasattr(self.config.temporal, 'ar'):
            params['ar_config'] = self.config.temporal.ar
        
        if 'img_size' in params and isinstance(params['img_size'], (list, tuple)):
            params['img_size'] = params['img_size'][0]
            
        try:
            self.model = registry_create_model(model_name, **params)
        except Exception:
            # 兼容旧版基于字典配置对象的模式
            from models.base import create_model as base_create_model
            merged_model_cfg = model_cfg.copy()
            if hasattr(self.config, 'temporal') and hasattr(self.config.temporal, 'ar'):
                merged_model_cfg.ar_config = self.config.temporal.ar
            self.model = base_create_model(merged_model_cfg)
            
        self.model = self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model created: {model_name} (Params: total={total_params:,}, trainable={trainable_params:,})")

    def _init_optimizer(self) -> None:
        """使用统一 EngineBuilder 初始化优化器"""
        tr_cfg = getattr(self.config, 'train', getattr(self.config, 'training', {}))
        self.optimizer = EngineBuilder.build_optimizer(self.model, tr_cfg, self.logger)

    def _init_scheduler(self) -> None:
        """使用统一 EngineBuilder 初始化调度器"""
        tr_cfg = getattr(self.config, 'train', getattr(self.config, 'training', {}))
        self.scheduler, self.warmup_scheduler = EngineBuilder.build_scheduler(
            self.optimizer, tr_cfg, self.train_loader, self.logger
        )

    def _init_amp(self) -> None:
        """使用统一 EngineBuilder 初始化混合精度 Scaler"""
        tr_cfg = getattr(self.config, 'train', getattr(self.config, 'training', {}))
        model_name = getattr(self.config.model, 'name', '')
        self.scaler, self.use_amp = EngineBuilder.build_amp_scaler(
            tr_cfg, model_name, self.logger
        )

    def _init_losses(self) -> None:
        """初始化 AR 时序损失、频域能谱损失与 DC 一致性损失"""
        loss_cfg = getattr(self.config, 'loss', {})
        
        ar_cfg = getattr(loss_cfg, 'ar_loss', {'weight': 1.0, 'reduction': 'mean'})
        self.ar_loss = ARLoss(config=ar_cfg)
        
        spectral_cfg = getattr(loss_cfg, 'spectral_loss', getattr(loss_cfg, 'spectral', {'weight': 0.1}))
        self.spectral_loss = SpectralLoss(config=spectral_cfg)
        
        dc_cfg = getattr(loss_cfg, 'dc_loss', getattr(loss_cfg, 'degradation_consistency', {'weight': 0.5}))
        self.dc_loss = DCLoss(config=dc_cfg)
        self.logger.info("Physics loss functions initialized (ARLoss, SpectralLoss, DCLoss)")

    def _init_metrics(self) -> None:
        """初始化指标历史记录与轻量计算器"""
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'train_rel_l2': [], 'val_rel_l2': [],
            'train_mae': [], 'val_mae': [],
            'learning_rate': []
        }
        self.metric_calc = MetricsCalculator(image_size=(256, 256))

    def _init_visualizer(self) -> None:
        """初始化时序可视化器"""
        viz_cfg = getattr(self.config, 'visualization', {})
        if bool(getattr(viz_cfg, 'enabled', True)):
            save_dir = self.output_dir / getattr(viz_cfg, 'save_dir', 'visualizations')
            self.visualizer = TemporalVisualizer(save_dir=str(save_dir))
        else:
            self.visualizer = None

    def _init_curriculum(self) -> None:
        """初始化课程学习阶段"""
        cur_cfg = getattr(self.config, 'curriculum', {})
        if bool(getattr(cur_cfg, 'enabled', False)):
            self.curriculum_stages = getattr(cur_cfg, 'stages', None)
            self.logger.info(f"Curriculum learning enabled with {len(self.curriculum_stages)} stages")
        else:
            self.curriculum_stages = None

    def _update_curriculum(self) -> None:
        """更新课程学习阶段与模型展开步长"""
        if not self.curriculum_stages:
            return
        current_stage = self.curriculum_stages[self.curriculum_stage]
        if self.current_epoch >= current_stage.epochs:
            if self.curriculum_stage < len(self.curriculum_stages) - 1:
                self.curriculum_stage += 1
                new_stage = self.curriculum_stages[self.curriculum_stage]
                if hasattr(self.model, 'update_ar_config'):
                    self.model.update_ar_config({
                        'T_out': new_stage.T_out,
                        'teacher_forcing_ratio': new_stage.teacher_forcing_ratio
                    })
                self.logger.info(
                    f"Switched to curriculum stage {self.curriculum_stage + 1}: "
                    f"T_out={new_stage.T_out}, TF_ratio={new_stage.teacher_forcing_ratio}"
                )

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """统一规范化时序批次数据为 (input_seq, target_seq)"""
        if 'input_sequence' in batch:
            input_seq = batch['input_sequence'].to(self.device)
            target_seq = batch['target_sequence'].to(self.device)
            if 'observation_sequence' in batch:
                input_seq = batch['observation_sequence'].to(self.device)
        else:
            model_in = self.batch_processor.build_model_input(batch, self.model)
            input_seq = model_in.to(self.device)
            target_seq = batch['target'].to(self.device)
            if input_seq.dim() == 4:
                input_seq = input_seq.unsqueeze(1)
            if target_seq.dim() == 4:
                target_seq = target_seq.unsqueeze(1)
        return input_seq, target_seq

    def _forward_model(self, input_seq: torch.Tensor, target_seq: Optional[torch.Tensor] = None) -> Any:
        """优先尝试 5D 时序输入 [B,T,C,H,W] 并支持多步自回归推演，单帧输入则安全回退"""
        if target_seq is not None and target_seq.ndim == 5:
            T_out = target_seq.shape[1]
        else:
            T_out = int(getattr(getattr(self.config, 'temporal', {}), 'T_out', 1) or 1)
        tf_ratio = float(getattr(getattr(getattr(self.config, 'temporal', {}), 'ar', {}), 'teacher_forcing_ratio', 0.0) or 0.0)
        
        if input_seq.ndim == 5:
            try:
                return self.model(
                    input_seq,
                    T_out=T_out,
                    teacher_seq=target_seq if self.model.training else None,
                    teacher_forcing_ratio=tf_ratio if self.model.training else 0.0
                )
            except TypeError:
                try:
                    return self.model(input_seq)
                except Exception:
                    return self.model(input_seq[:, -1])
        return self.model(input_seq)

    def _compute_light_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """训练轻量指标计算 (Rel-L2, MAE)"""
        if pred.ndim == 5:
            pred = pred[:, -1]
        if target.ndim == 5:
            target = target[:, -1]
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
            
        rel_l2_bc = self.metric_calc.compute_rel_l2(pred, target)
        mae_bc = self.metric_calc.compute_mae(pred, target)
        return float(rel_l2_bc.mean().item()), float(mae_bc.mean().item())

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], target_seq: torch.Tensor, batch: Dict) -> torch.Tensor:
        """计算 AR 时序、频域能谱与 DC 物理综合损失"""
        predictions = outputs['predictions']
        
        # 统一成 5D: [B, T, C, H, W]
        if predictions.ndim == 4:
            predictions = predictions.unsqueeze(1)
            if target_seq.ndim == 5:
                target_seq = target_seq[:, -1:].contiguous()
            else:
                target_seq = target_seq.unsqueeze(1)
        elif predictions.ndim == 5:
            if target_seq.ndim == 4:
                target_seq = target_seq.unsqueeze(1)
            if predictions.shape[1] != target_seq.shape[1]:
                T = min(predictions.shape[1], target_seq.shape[1])
                predictions = predictions[:, :T]
                target_seq = target_seq[:, :T]
        else:
            raise ValueError(f"Unsupported predictions.ndim={predictions.ndim}")
            
        if predictions.shape[-2:] != target_seq.shape[-2:]:
            target_size = target_seq.shape[-2:]
            B, T, C = predictions.shape[:3]
            pred_bt = predictions.reshape(B * T, C, *predictions.shape[-2:])
            pred_bt = F.interpolate(pred_bt, size=target_size, mode="bilinear", align_corners=False)
            predictions = pred_bt.reshape(B, T, C, *target_size)
            
        ar_loss_res = self.ar_loss(predictions, target_seq)
        ar_loss = ar_loss_res['total_loss'] if isinstance(ar_loss_res, dict) else ar_loss_res
        
        spectral_loss_res = self.spectral_loss(predictions, target_seq)
        spectral_loss = spectral_loss_res['total_loss'] if isinstance(spectral_loss_res, dict) else spectral_loss_res
        
        dc_loss = torch.tensor(0.0, device=self.device)
        dc_weight = float(getattr(getattr(getattr(self.config, 'loss', {}), 'dc_loss', {}), 'weight', 1.0) or 0.0)
        if dc_weight > 0 and "h_params" in batch:
            target_obs = batch.get('lr_observation', batch.get('original_observation', None))
            if target_obs is not None:
                dc_loss = self.dc_loss(predictions[:, -1], target_obs.to(self.device), batch["h_params"])
            elif "observation" in batch:
                dc_loss = self.dc_loss(predictions[:, -1], batch["observation"].to(self.device), None)
            
        return ar_loss + spectral_loss + dc_loss

    def train_epoch(self) -> Dict[str, float]:
        """单轮训练循环"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'rel_l2': [], 'mae': []}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        grad_clip = float(getattr(getattr(self.config, 'train', {}), 'gradient_clip_val', 0.0) or 0.0)
        limit_train = int(getattr(getattr(self.config, 'train', {}), 'limit_train_batches', 0) or 0)
        
        for batch_idx, batch in enumerate(pbar):
            if limit_train > 0 and batch_idx >= limit_train:
                break
            input_seq, target_seq = self._prepare_batch(batch)
            self.optimizer.zero_grad()
            
            with autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu', enabled=self.use_amp):
                outputs = self._forward_model(input_seq, target_seq)
                predictions = outputs if isinstance(outputs, dict) else {'predictions': outputs}
                loss = self._compute_loss(predictions, target_seq, batch)
                
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
            with torch.no_grad():
                rel_l2, mae = self._compute_light_metrics(predictions['predictions'], target_seq)
                epoch_metrics['rel_l2'].append(rel_l2)
                epoch_metrics['mae'].append(mae)
                
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "rel_l2": f"{rel_l2:.4f}", "lr": f"{lr:.2e}"})
            
            # 定期可视化与日志
            log_interval = int(getattr(getattr(self.config, 'experiment', {}), 'log_every_n_steps', 50) or 50)
            if self.global_step % log_interval == 0:
                self.logger.info(f"Step {self.global_step}: loss={loss.item():.4f}, rel_l2={rel_l2:.4f}, lr={lr:.2e}")
                
        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        return {
            'loss': mean_train_loss,
            'total_loss': mean_train_loss,
            'rel_l2': float(np.mean(epoch_metrics['rel_l2'])) if epoch_metrics['rel_l2'] else 0.0,
            'mae': float(np.mean(epoch_metrics['mae'])) if epoch_metrics['mae'] else 0.0
        }

    def evaluate(self, loader: DataLoader, desc: str = "Evaluating") -> Dict[str, float]:
        """通用评估循环"""
        self.model.eval()
        losses = []
        metrics_acc = {"rel_l2": [], "mae": [], "psnr": [], "ssim": []}
        limit_val = int(getattr(getattr(self.config, 'train', {}), 'limit_val_batches', 0) or 0)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
                if limit_val > 0 and batch_idx >= limit_val:
                    break
                input_seq, target_seq = self._prepare_batch(batch)
                outputs = self._forward_model(input_seq, target_seq)
                predictions = outputs if isinstance(outputs, dict) else {"predictions": outputs}
                loss = self._compute_loss(predictions, target_seq, batch)
                losses.append(loss.item())
                
                pred_for_m = predictions["predictions"]
                target_for_m = target_seq[:, -1] if (pred_for_m.ndim == 4 and target_seq.ndim == 5) else target_seq
                m = compute_metrics(pred_for_m, target_for_m)
                for k in metrics_acc:
                    if k in m:
                        v = m[k]
                        if isinstance(v, torch.Tensor):
                            v = v.mean().detach().cpu().item() if v.numel() > 1 else v.detach().cpu().item()
                        metrics_acc[k].append(v)
                        
        mean_eval_loss = float(np.mean(losses)) if losses else 0.0
        out = {"loss": mean_eval_loss, "total_loss": mean_eval_loss}
        for k, vals in metrics_acc.items():
            out[k] = float(np.mean(vals)) if vals else 0.0
        return out

    def validate(self) -> Dict[str, float]:
        """验证集评估"""
        return self.evaluate(self.val_loader, desc="Validating")

    def save_checkpoint(self, is_best: bool = False) -> None:
        """委托给 TrainingArtifactManager 保存检查点"""
        self.artifact_manager.save_checkpoint(
            epoch=self.current_epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            val_results={'loss': self.best_val_loss},
            best_val_loss=self.best_val_loss,
            is_best=is_best,
            global_step=self.global_step
        )
        # 兼容保留旧版 last.ckpt / best.ckpt 接口
        ckpt_data = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(ckpt_data, self.output_dir / "last.ckpt")
        if is_best:
            torch.save(ckpt_data, self.output_dir / "best.ckpt")

    def train(self) -> None:
        """执行完整时序训练控制流"""
        self.logger.info("Starting temporal training lifecycle...")
        start_time = time.time()
        train_cfg = getattr(self.config, 'train', {})
        epochs_val = train_cfg.get('epochs') if isinstance(train_cfg, (dict, DictConfig)) else getattr(train_cfg, 'epochs', None)
        max_epochs_val = train_cfg.get('max_epochs') if isinstance(train_cfg, (dict, DictConfig)) else getattr(train_cfg, 'max_epochs', None)
        if epochs_val is not None and max_epochs_val is not None:
            max_epochs = min(int(epochs_val), int(max_epochs_val))
        else:
            max_epochs = int(epochs_val or max_epochs_val or 100)
        val_interval = int(getattr(getattr(self.config, 'experiment', {}), 'val_check_interval', 1) or 1)
        patience = int(getattr(getattr(self.config.experiment, 'early_stopping', {}), 'patience', 10) or 10)
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            self._update_curriculum()
            
            train_metrics = self.train_epoch()
            
            if epoch % val_interval == 0:
                val_metrics = self.validate()
                self.metrics_history['train_loss'].append(train_metrics['loss'])
                self.metrics_history['val_loss'].append(val_metrics['loss'])
                self.metrics_history['train_rel_l2'].append(train_metrics['rel_l2'])
                self.metrics_history['val_rel_l2'].append(val_metrics['rel_l2'])
                self.metrics_history['train_mae'].append(train_metrics['mae'])
                self.metrics_history['val_mae'].append(val_metrics['mae'])
                self.metrics_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                # 写入 metrics.jsonl
                self.artifact_manager.log_epoch_results(
                    epoch=epoch,
                    train_results=train_metrics,
                    val_results=val_metrics,
                    lr=self.optimizer.param_groups[0]['lr']
                )
                
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    
                self.save_checkpoint(is_best)
                self.logger.info(
                    f"Epoch {epoch:3d}: train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, val_rel_l2={val_metrics['rel_l2']:.4f} (best={self.best_val_loss:.4f})"
                )
                
                if self.early_stopping_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                    
            if self.scheduler:
                self.scheduler.step()
                
        total_time = time.time() - start_time
        self.logger.info(f"Training finished in {total_time:.2f}s")


@hydra.main(version_base=None, config_path="configs/experiment", config_name="temporal_training")
def main(config: DictConfig) -> None:
    """Hydra CLI 主控入口"""
    try:
        trainer = TemporalTrainer(config)
        trainer.train()
        print("✅ 训练完成!")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise


if __name__ == "__main__":
    main()
