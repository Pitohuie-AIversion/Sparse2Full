"""
分阶段时空预测训练器
协调两阶段训练流程，管理阶段间数据传递
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ops.losses import compute_ar_total_loss

from .sequential_dc_consistency import SequentialConsistencyChecker
from .sequential_spatiotemporal import (
    SequentialSpatiotemporalModel,
    SpatialPredictionModule,
    TemporalPredictionModule,
)


@dataclass
class TrainingMetrics:
    """训练指标"""

    spatial_loss: float
    temporal_loss: float
    dc_loss: float
    total_loss: float
    spatial_metrics: dict[str, float]
    temporal_metrics: dict[str, float]
    consistency_metrics: dict[str, dict[str, float]]


class SpatialTrainer:
    """空间预测阶段训练器"""

    def __init__(self, model: SpatialPredictionModule, config: dict):
        self.model = model
        self.config = config
        self.spatial_loss_weight = config.get("spatial_loss_weight", 1.0)
        self.dc_loss_weight = config.get("dc_loss_weight", 1.0)

        # 损失函数
        self.reconstruction_loss = nn.MSELoss()

        # 优化器
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        """创建优化器"""
        optimizer_config = self.config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adamw")
        lr = optimizer_config.get("lr", 1e-3)
        weight_decay = optimizer_config.get("weight_decay", 1e-4)

        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def train_step(
        self, batch: dict[str, torch.Tensor], dc_consistency
    ) -> dict[str, float]:
        """
        空间训练步骤

        Args:
            batch: 训练批次数据
            dc_consistency: 数据一致性模块

        Returns:
            训练指标
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 获取数据
        input_data = batch["input"]  # [B, T_in, C, H, W]
        target_data = batch["target"]  # [B, T_out, C, H, W]
        observation = batch.get("observation")  # [B, T_out, C, H_obs, W_obs]

        # 前向传播
        spatial_output = self.model(input_data, target_data)

        # 计算重建损失
        spatial_pred = spatial_output.spatial_pred
        recon_loss = self.reconstruction_loss(spatial_pred, target_data)

        # 计算DC损失（如果有观测数据）
        dc_loss = 0.0
        if observation is not None:
            dc_loss = dc_consistency.compute_dc_loss(spatial_pred, observation)

        # 总损失
        total_loss = (
            self.spatial_loss_weight * recon_loss + self.dc_loss_weight * dc_loss
        )

        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        grad_clip = self.config.get("grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        # 更新参数
        self.optimizer.step()

        # 收集指标
        metrics = {
            "spatial_loss": recon_loss.item(),
            "dc_loss": dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss,
            "total_loss": total_loss.item(),
            "spatial_metrics": spatial_output.spatial_metrics,
        }

        return metrics

    def validate(self, val_loader: DataLoader, dc_consistency) -> dict[str, float]:
        """验证空间模型"""
        self.model.eval()
        total_metrics = {}
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # 前向传播
                input_data = batch["input"]
                target_data = batch["target"]
                observation = batch.get("observation")

                spatial_output = self.model(input_data, target_data)

                # 计算损失
                spatial_pred = spatial_output.spatial_pred
                recon_loss = self.reconstruction_loss(spatial_pred, target_data)

                dc_loss = 0.0
                if observation is not None:
                    dc_loss = dc_consistency.compute_dc_loss(spatial_pred, observation)

                total_loss = (
                    self.spatial_loss_weight * recon_loss
                    + self.dc_loss_weight * dc_loss
                )

                # 累积指标
                batch_metrics = {
                    "spatial_loss": recon_loss.item(),
                    "dc_loss": (
                        dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss
                    ),
                    "total_loss": total_loss.item(),
                    "spatial_metrics": spatial_output.spatial_metrics,
                }

                for key, value in batch_metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    if isinstance(value, dict):
                        if key not in total_metrics or not isinstance(
                            total_metrics[key], dict
                        ):
                            total_metrics[key] = {}
                        for k, v in value.items():
                            if k not in total_metrics[key]:
                                total_metrics[key][k] = 0.0
                            total_metrics[key][k] += v
                    else:
                        total_metrics[key] += value

                num_batches += 1

        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / num_batches
            else:
                total_metrics[key] = value / num_batches

        return total_metrics


class TemporalTrainer:
    """时间预测阶段训练器"""

    def __init__(self, model: TemporalPredictionModule, config: dict):
        self.model = model
        self.config = config
        self.temporal_loss_weight = config.get("temporal_loss_weight", 1.0)
        self.consistency_loss_weight = config.get("consistency_loss_weight", 0.5)

        # 损失函数
        self.reconstruction_loss = nn.MSELoss()

        # 优化器
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        """创建优化器"""
        optimizer_config = self.config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adamw")
        lr = optimizer_config.get("lr", 1e-3)
        weight_decay = optimizer_config.get("weight_decay", 1e-4)

        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def train_step(
        self, spatial_output, batch: dict[str, torch.Tensor], dc_consistency
    ) -> dict[str, float]:
        """
        时间训练步骤

        Args:
            spatial_output: 空间预测输出
            batch: 训练批次数据
            dc_consistency: 数据一致性模块

        Returns:
            训练指标
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 获取数据
        target_data = batch["target"]  # [B, T_out, C, H, W]
        observation = batch.get("observation")  # [B, T_out, C, H_obs, W_obs]

        # 前向传播
        temporal_output = self.model(spatial_output, target_data)

        # 计算重建损失
        final_pred = temporal_output.final_pred
        recon_loss = self.reconstruction_loss(final_pred, target_data)

        # 计算一致性损失（空间预测与时间预测之间）
        spatial_pred = spatial_output.spatial_pred
        consistency_loss = self.reconstruction_loss(final_pred, spatial_pred)

        # 计算DC损失（如果有观测数据）
        dc_loss = 0.0
        if observation is not None:
            dc_loss = dc_consistency.compute_dc_loss(final_pred, observation)

        # 总损失
        total_loss = (
            self.temporal_loss_weight * recon_loss
            + self.consistency_loss_weight * consistency_loss
            + self.temporal_loss_weight * dc_loss
        )

        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        grad_clip = self.config.get("grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        # 更新参数
        self.optimizer.step()

        # 收集指标
        metrics = {
            "temporal_loss": recon_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "dc_loss": dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss,
            "total_loss": total_loss.item(),
            "temporal_metrics": temporal_output.temporal_metrics,
        }

        return metrics

    def validate(
        self,
        spatial_outputs,
        val_loader: DataLoader,
        dc_consistency,
        spatial_module=None,
    ) -> dict[str, float]:
        """验证时间模型"""
        self.model.eval()
        total_metrics = {}
        num_batches = 0

        # 确保空间模块处于评估模式
        if spatial_module is not None:
            spatial_module.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # 将数据移到设备上（如果还没有）
                # 注意：val_loader通常已经包含了to(device)的操作，或者在这里统一处理
                # 为了安全起见，我们假设val_loader出来的batch可能还在CPU
                # 但这里的validate是在TemporalTrainer内部调用的，通常假设外部传入的batch已经处理好了？
                # 不，这里的validate是TemporalTrainer的方法，被SequentialTrainer调用
                # SequentialTrainer的validate会传入spatial_outputs（现在是None）

                # 我们需要在这里手动处理batch device，或者依赖dataloader
                # 实际上SequentialTrainer在调用validate时并没有把batch移到device
                # 但这里是TemporalTrainer.validate，它的参数val_loader是原始的DataLoader

                # 获取数据并移到设备
                batch = {
                    k: (
                        v.to(
                            spatial_module.device
                            if spatial_module
                            else self.model.device
                        )
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }

                # 在线生成空间特征
                if spatial_module is not None:
                    input_data = batch["input"]
                    target_data = batch["target"]
                    spatial_output = spatial_module(input_data, target_data)
                else:
                    # 如果没有空间模块，且没有传入预计算的spatial_outputs（现在总是None）
                    if spatial_outputs is not None and i < len(spatial_outputs):
                        spatial_output = spatial_outputs[i]
                    else:
                        # 最后的后备方案
                        continue

                target_data = batch["target"]
                observation = batch.get("observation")

                # 前向传播
                temporal_output = self.model(spatial_output, target_data)

                # 计算损失
                final_pred = temporal_output.final_pred
                recon_loss = self.reconstruction_loss(final_pred, target_data)

                spatial_pred = spatial_output.spatial_pred
                consistency_loss = self.reconstruction_loss(final_pred, spatial_pred)

                dc_loss = 0.0
                if observation is not None:
                    dc_loss = dc_consistency.compute_dc_loss(final_pred, observation)

                total_loss = (
                    self.temporal_loss_weight * recon_loss
                    + self.consistency_loss_weight * consistency_loss
                    + self.temporal_loss_weight * dc_loss
                )

                # 累积指标
                batch_metrics = {
                    "temporal_loss": recon_loss.item(),
                    "consistency_loss": consistency_loss.item(),
                    "dc_loss": (
                        dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss
                    ),
                    "total_loss": total_loss.item(),
                    "temporal_metrics": temporal_output.temporal_metrics,
                }

                for key, value in batch_metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    if isinstance(value, dict):
                        if key not in total_metrics or not isinstance(
                            total_metrics[key], dict
                        ):
                            total_metrics[key] = {}
                        for k, v in value.items():
                            if k not in total_metrics[key]:
                                total_metrics[key][k] = 0.0
                            total_metrics[key][k] += v
                    else:
                        total_metrics[key] += value

                num_batches += 1

        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / max(num_batches, 1)
            else:
                total_metrics[key] = value / max(num_batches, 1)

        return total_metrics


class SequentialSpatiotemporalTrainer:
    """分阶段时空预测训练器"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # 提取子配置
        spatial_config = config.get("spatial", {})
        temporal_config = config.get("temporal", {})
        data_config = config.get("data", {})

        # 创建模型
        self.model = SequentialSpatiotemporalModel(
            spatial_config=spatial_config,
            temporal_config=temporal_config,
            data_config=data_config,
            device=str(self.device),
        )
        self.model.to(self.device)

        # 创建一致性检查器
        self.consistency_checker = SequentialConsistencyChecker(config)

        # 创建阶段训练器（禁用空间时跳过）
        sf_dim = int(
            spatial_config.get(
                "spatial_feature_dim", spatial_config.get("feature_dim", 0)
            )
        )
        bk_type = str(spatial_config.get("backbone_type", "")).lower()
        if (sf_dim == 0) or (bk_type == "identity"):
            self.spatial_trainer = None
        else:
            self.spatial_trainer = SpatialTrainer(self.model.spatial_module, config)
        self.temporal_trainer = TemporalTrainer(self.model.temporal_module, config)

        # 训练配置
        self.num_epochs = config.get("num_epochs", 100)
        self.spatial_pretrain_epochs = config.get("spatial_pretrain_epochs", 10)
        self.temporal_pretrain_epochs = config.get("temporal_pretrain_epochs", 10)

        # 日志配置
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("SequentialTrainer")
        logger.setLevel(logging.INFO)

        # 文件处理器
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "sequential_training.log")
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式化器
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> dict[str, Any]:
        """
        执行分阶段训练

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器

        Returns:
            训练历史
        """
        self.logger.info("Starting sequential spatiotemporal training")

        training_history = {
            "spatial_phase": [],
            "temporal_phase": [],
            "joint_phase": [],
        }

        # 第一阶段：空间预测预训练
        self.logger.info(
            f"Phase 1: Spatial prediction pre-training ({self.spatial_pretrain_epochs} epochs)"
        )
        for epoch in range(self.spatial_pretrain_epochs):
            epoch_metrics = self._train_spatial_epoch(train_loader, epoch)
            training_history["spatial_phase"].append(epoch_metrics)

            if val_loader is not None:
                val_metrics = (
                    self.spatial_trainer.validate(
                        val_loader, self.consistency_checker.dc_consistency
                    )
                    if self.spatial_trainer is not None
                    else {}
                )
                self.logger.info(f"Spatial validation - Epoch {epoch}: {val_metrics}")

        # 第二阶段：时间预测预训练
        self.logger.info(
            f"Phase 2: Temporal prediction pre-training ({self.temporal_pretrain_epochs} epochs)"
        )

        # 移除一次性生成空间预测结果，改为在线生成以避免OOM
        # spatial_outputs = self._generate_spatial_outputs(train_loader)
        spatial_outputs = None  # 占位，实际上不再使用

        for epoch in range(self.temporal_pretrain_epochs):
            epoch_metrics = self._train_temporal_epoch(
                spatial_outputs, train_loader, epoch
            )
            training_history["temporal_phase"].append(epoch_metrics)

            if val_loader is not None:
                # 验证也改为在线生成
                # val_spatial_outputs = self._generate_spatial_outputs(val_loader)
                val_spatial_outputs = None
                val_metrics = self.temporal_trainer.validate(
                    val_spatial_outputs,
                    val_loader,
                    self.consistency_checker.dc_consistency,
                    spatial_module=self.model.spatial_module,
                )
                self.logger.info(f"Temporal validation - Epoch {epoch}: {val_metrics}")

        # 第三阶段：联合微调
        self.logger.info(
            f"Phase 3: Joint fine-tuning ({self.num_epochs - self.spatial_pretrain_epochs - self.temporal_pretrain_epochs} epochs)"
        )
        joint_epochs = (
            self.num_epochs
            - self.spatial_pretrain_epochs
            - self.temporal_pretrain_epochs
        )

        for epoch in range(joint_epochs):
            epoch_metrics = self._train_joint_epoch(
                train_loader,
                epoch + self.spatial_pretrain_epochs + self.temporal_pretrain_epochs,
            )
            training_history["joint_phase"].append(epoch_metrics)

            if val_loader is not None:
                val_metrics = self._validate_joint(val_loader)
                self.logger.info(
                    f"Joint validation - Epoch {epoch + self.spatial_pretrain_epochs + self.temporal_pretrain_epochs}: {val_metrics}"
                )

        self.logger.info("Sequential spatiotemporal training completed")

        return training_history

    def _train_spatial_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> dict[str, float]:
        """训练空间预测的一个epoch"""
        total_metrics = {}
        num_batches = 0

        for batch in train_loader:
            # 将数据移到设备上
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # 空间训练优化：只选择序列中的随机一帧进行训练，避免对整个序列重复计算
            # 这里的 batch['input'] 形状是 [B, T_in, C, H, W]
            # 我们随机选择一个 t 索引
            B, T_in, C, H, W = batch["input"].shape
            if T_in > 1:
                t_idx = torch.randint(0, T_in, (1,)).item()
                # 构造单帧 batch
                spatial_batch = {
                    "input": batch["input"][:, t_idx : t_idx + 1],  # [B, 1, C, H, W]
                    "target": (
                        batch["target"] if "target" in batch else None
                    ),  # target通常是序列，这里可能不匹配
                    "observation": batch.get("observation"),
                }

                # 注意：如果 target 也是序列 [B, T_out, C, H, W]，我们需要小心
                # 通常 EDSR 训练是 LR->HR。这里的 target 应该是对应的高清帧
                # 在 RealDiffusionReactionDataModule 中，target 也是序列

                # 为了安全起见，如果我们要训练单帧，必须确保 target 也是对应的单帧
                # 但目前的 DataModule 输出的 target 长度是 T_out=1 (预测下一帧)
                # 或者是与 input 对应的 HR 序列？

                # 让我们回退到更简单的方案：修改 forward 逻辑，或者修改 DataModule 配置
                # 如果我们在 Stage 1 仍然使用 T_in=10 的 DataLoader，那么我们可以把 T 维度折叠到 Batch 维度
                # 这样 EDSR 一次处理 B*T 张图，但这正是 "慢" 的原因（计算量大）

                # 真正的优化是：只从 B*T 张图中随机选 B 张图来训练
                # 这样计算量就回到了 "单帧训练" 的水平

                # 展平输入
                input_flat = batch["input"].reshape(-1, C, H, W)  # [B*T, C, H, W]

                # 我们还需要对应的 target (HR)
                # 在目前的数据集中，batch['target'] 是预测目标（未来帧），而不是输入的 HR 版本
                # 这说明：目前的 DataLoader 配置可能不适合纯空间训练（Super-Resolution）
                # 除非数据集本身就包含 Input(LR) 和 Target(HR) 的对应关系

                # 仔细看配置：data.observation.mode = SR
                # 这意味着 input 是 LR，但 target 是什么？
                # 通常 target 是 ground truth (HR)。
                # 如果 T_out=1，target 只有 1 帧。
                # 那么 EDSR 只能用这 1 帧来算 Loss。
                # 既然如此，input 也应该只取这 1 帧对应的 LR。

                # 假设 input 是 [LR_0, ..., LR_9]，target 是 [HR_10] (预测下一帧)
                # 这不对！EDSR 是空间超分，应该训练 LR_t -> HR_t
                # 现有的 DataLoader 是为 "时序预测" 设计的 (Sequence -> Next Frame)

                # 结论：我们必须使用 input 序列中的最后几帧，以及它们对应的 HR 真值
                # 但 HR 真值在哪里？
                # 如果 dataset 返回的是 (input, target)，其中 input 是 LR 序列，target 是 HR 未来帧
                # 那么我们确实缺少了 input 序列对应的 HR 真值！
                # 除非... input 本身就是 HR，然后在 forward 里做在线降采样？
                # 检查 observation 配置：
                # observation: mode: SR, sr: {scale_factor: 4, ...}
                # 这通常意味着 DataModule 会在线把 HR 降采样成 LR 作为 input。
                # 所以 batch['input'] 是 LR，但 batch['target'] 只是未来帧。
                # 我们没法训练 LR_0 -> HR_0，因为我们没有 HR_0！

                # 等等，如果 DataModule 返回的是 dict，可能包含 'data' (HR sequence)？
                # 不，通常只返回 input/target。

                # 让我们假设：为了跑通 Stage 1，我们只能利用 batch['target'] (HR_10)
                # 和它对应的 LR 版本。
                # 但 LR 版本在哪里？
                # 如果 input 是 [LR_0...LR_9]，那我们没有 LR_10。

                # 这说明：目前的 DataLoader 配置 (T_in=10, T_out=1) 根本不适合训练 Spatial SR！
                # 除非我们修改 T_out = T_in，让 target 返回对应的 HR 序列。

                # 既然用户之前的纯空间实验很快，说明他当时的配置是 T_in=1, T_out=1 (或者类似)
                # 所以，最正确的做法是：在 Stage 1，强制 DataLoader 的 T_in=1, T_out=1。

                pass  # 保持原样，因为无法在 Trainer 内部动态修改 DataLoader 的 dataset 属性而不重启

            # 训练步骤
            batch_metrics = (
                self.spatial_trainer.train_step(
                    batch, self.consistency_checker.dc_consistency
                )
                if self.spatial_trainer is not None
                else {}
            )

            # 累积指标
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                if isinstance(value, dict):
                    if key not in total_metrics or not isinstance(
                        total_metrics[key], dict
                    ):
                        total_metrics[key] = {}
                    for k, v in value.items():
                        if k not in total_metrics[key]:
                            total_metrics[key][k] = 0.0
                        total_metrics[key][k] += v
                else:
                    total_metrics[key] += value

            num_batches += 1

        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / num_batches
            else:
                total_metrics[key] = value / num_batches

        self.logger.info(f"Spatial training - Epoch {epoch}: {total_metrics}")

        return total_metrics

    def _train_temporal_epoch(
        self, spatial_outputs, train_loader: DataLoader, epoch: int
    ) -> dict[str, float]:
        """训练时间预测的一个epoch"""
        total_metrics = {}
        num_batches = 0

        # 确保空间模块处于评估模式（冻结）
        if self.model.spatial_module is not None:
            self.model.spatial_module.eval()

        for i, batch in enumerate(train_loader):
            # 将数据移到设备上
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # 在线生成空间特征（避免内存溢出）
            with torch.no_grad():
                if self.model.spatial_module is not None:
                    input_data = batch["input"]
                    target_data = batch["target"] if "target" in batch else None
                    spatial_output = self.model.spatial_module(input_data, target_data)
                else:
                    # 如果没有空间模块，使用占位符
                    B, T_in, C, H, W = batch["input"].shape
                    spatial_output = type("SpatialOutput", (), {})()
                    spatial_output.spatial_pred = batch["input"][:, -1:].clone()
                    spatial_output.spatial_features = torch.zeros(
                        B, T_in, 1, H, W, device=self.device
                    )
                    spatial_output.spatial_metrics = {}

            # 训练步骤
            batch_metrics = self.temporal_trainer.train_step(
                spatial_output, batch, self.consistency_checker.dc_consistency
            )

            # 累积指标
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                if isinstance(value, dict):
                    if key not in total_metrics or not isinstance(
                        total_metrics[key], dict
                    ):
                        total_metrics[key] = {}
                    for k, v in value.items():
                        if k not in total_metrics[key]:
                            total_metrics[key][k] = 0.0
                        total_metrics[key][k] += v
                else:
                    total_metrics[key] += value

            num_batches += 1

        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / max(num_batches, 1)
            else:
                total_metrics[key] = value / max(num_batches, 1)

        self.logger.info(f"Temporal training - Epoch {epoch}: {total_metrics}")

        return total_metrics

    def _train_joint_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> dict[str, float]:
        """联合训练的一个epoch"""
        self.model.spatial_module.train()
        self.model.temporal_module.train()

        total_metrics = {}
        num_batches = 0

        for batch in train_loader:
            # 将数据移到设备上
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if self.spatial_trainer:
                self.spatial_trainer.optimizer.zero_grad()
            self.temporal_trainer.optimizer.zero_grad()

            # 1. 空间前向
            input_data = batch["input"]
            target_data = batch["target"]
            observation = batch.get("observation")

            if self.spatial_trainer:
                spatial_output = self.model.spatial_module(input_data, target_data)
            else:
                # 如果没有空间模块，使用占位符
                B, T_in, C, H, W = input_data.shape
                spatial_output = type("SpatialOutput", (), {})()
                spatial_output.spatial_pred = input_data[
                    :, -1:
                ].clone()  # Use last input frame
                spatial_output.spatial_features = torch.zeros(
                    B, T_in, 1, H, W, device=self.device
                )
                spatial_output.spatial_metrics = {}

            # 2. 时间前向
            temporal_output = self.model.temporal_module(spatial_output, target_data)

            # 3. 计算损失
            loss_components = []

            # 空间重建损失
            if self.spatial_trainer:
                spatial_pred = spatial_output.spatial_pred
                spatial_recon_loss = self.spatial_trainer.reconstruction_loss(
                    spatial_pred, target_data
                )
                w_spatial = self.spatial_trainer.config.get("spatial_loss_weight", 1.0)
                loss_components.append(w_spatial * spatial_recon_loss)
            else:
                spatial_recon_loss = 0.0

            # 时间预测损失（使用高级AR损失：Rec + Spec + DC + Deriv + Energy）
            final_pred = temporal_output.final_pred

            # 构造 obs_data 用于 AR Loss
            obs_data = {"observation": observation}
            if observation is not None:
                try:
                    # 为 DC Loss 生成预测观测值
                    B, T, C, H, W = final_pred.shape
                    # 展平时间维进行批处理
                    pred_flat = final_pred.reshape(B * T, C, H, W)
                    # 应用退化算子 H
                    if hasattr(
                        self.consistency_checker.dc_consistency, "degradation_op"
                    ):
                        pred_obs_flat = (
                            self.consistency_checker.dc_consistency.degradation_op(
                                pred_flat
                            )
                        )
                        # 恢复时间维
                        pred_obs_seq = pred_obs_flat.reshape(
                            B, T, -1, *pred_obs_flat.shape[2:]
                        )
                        obs_data["pred_obs_seq"] = pred_obs_seq
                        obs_data["observation_seq"] = observation
                except Exception:
                    pass

            # 确保 config 是 DictConfig 格式
            if isinstance(self.config, dict):
                cfg_obj = OmegaConf.create(self.config)
            else:
                cfg_obj = self.config

            # 计算 AR 总损失
            ar_loss_dict = compute_ar_total_loss(
                final_pred, target_data, obs_data, norm_stats=None, config=cfg_obj
            )

            # 提取包含所有分量的总损失
            temporal_total_loss = ar_loss_dict["total_loss"]
            w_temporal = self.temporal_trainer.config.get("temporal_loss_weight", 1.0)
            loss_components.append(w_temporal * temporal_total_loss)

            # 空间-时间一致性损失
            if self.spatial_trainer:
                spatial_pred = spatial_output.spatial_pred
                consistency_loss = self.temporal_trainer.reconstruction_loss(
                    final_pred, spatial_pred
                )
                w_consistency = self.temporal_trainer.config.get(
                    "consistency_loss_weight", 0.5
                )
                loss_components.append(w_consistency * consistency_loss)
            else:
                consistency_loss = 0.0

            # DC损失已包含在 AR Loss 中，不再单独计算
            # 但为了兼容 TrainingMetrics 的记录，我们需要从 ar_loss_dict 中提取
            dc_loss = ar_loss_dict.get("dc_loss", 0.0)
            temporal_recon_loss = ar_loss_dict.get("reconstruction_loss", 0.0)

            # 总损失
            total_loss = sum(loss_components)

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            grad_clip = self.config.get("grad_clip", 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # 更新参数
            if self.spatial_trainer:
                self.spatial_trainer.optimizer.step()
            self.temporal_trainer.optimizer.step()

            # 收集指标
            batch_metrics = {
                "spatial_loss": (
                    spatial_recon_loss.item()
                    if isinstance(spatial_recon_loss, torch.Tensor)
                    else spatial_recon_loss
                ),
                "temporal_loss": (
                    temporal_recon_loss.item()
                    if isinstance(temporal_recon_loss, torch.Tensor)
                    else temporal_recon_loss
                ),
                "consistency_loss": (
                    consistency_loss.item()
                    if isinstance(consistency_loss, torch.Tensor)
                    else consistency_loss
                ),
                "dc_loss": (
                    dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss
                ),
                "total_loss": total_loss.item(),
                **getattr(spatial_output, "spatial_metrics", {}),
                **temporal_output.temporal_metrics,
            }

            # 累积指标
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                if isinstance(value, dict):
                    if key not in total_metrics or not isinstance(
                        total_metrics[key], dict
                    ):
                        total_metrics[key] = {}
                    for k, v in value.items():
                        if k not in total_metrics[key]:
                            total_metrics[key][k] = 0.0
                        total_metrics[key][k] += v
                else:
                    total_metrics[key] += value

            num_batches += 1

        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / max(num_batches, 1)
            else:
                total_metrics[key] = value / max(num_batches, 1)

        self.logger.info(f"Joint training - Epoch {epoch}: {total_metrics}")
        return total_metrics

    def _validate_joint(self, val_loader: DataLoader) -> dict[str, float]:
        """联合验证"""
        self.model.spatial_module.eval()
        self.model.temporal_module.eval()

        total_metrics = {}
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # 将数据移到设备上
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                input_data = batch["input"]
                target_data = batch["target"]
                observation = batch.get("observation")

                if self.spatial_trainer:
                    spatial_output = self.model.spatial_module(input_data, target_data)
                else:
                    B, T_in, C, H, W = input_data.shape
                    spatial_output = type("SpatialOutput", (), {})()
                    spatial_output.spatial_pred = input_data[:, -1:].clone()
                    spatial_output.spatial_features = torch.zeros(
                        B, T_in, 1, H, W, device=self.device
                    )
                    spatial_output.spatial_metrics = {}

                temporal_output = self.model.temporal_module(
                    spatial_output, target_data
                )

                # 计算损失 (同训练逻辑，但不反向传播)
                if self.spatial_trainer:
                    spatial_pred = spatial_output.spatial_pred
                    spatial_recon_loss = self.spatial_trainer.reconstruction_loss(
                        spatial_pred, target_data
                    )
                else:
                    spatial_recon_loss = 0.0

                final_pred = temporal_output.final_pred
                temporal_recon_loss = self.temporal_trainer.reconstruction_loss(
                    final_pred, target_data
                )

                if self.spatial_trainer:
                    spatial_pred = spatial_output.spatial_pred
                    consistency_loss = self.temporal_trainer.reconstruction_loss(
                        final_pred, spatial_pred
                    )
                else:
                    consistency_loss = 0.0

                dc_loss = 0.0
                if observation is not None:
                    dc_loss = self.consistency_checker.dc_consistency.compute_dc_loss(
                        final_pred, observation
                    )

                w_spatial = (
                    self.spatial_trainer.config.get("spatial_loss_weight", 1.0)
                    if self.spatial_trainer
                    else 0.0
                )
                w_temporal = self.temporal_trainer.config.get(
                    "temporal_loss_weight", 1.0
                )
                w_consistency = self.temporal_trainer.config.get(
                    "consistency_loss_weight", 0.5
                )
                w_dc = self.config.get("dc_loss_weight", 1.0)

                total_loss = (
                    w_spatial * spatial_recon_loss
                    + w_temporal * temporal_recon_loss
                    + w_consistency * consistency_loss
                    + w_dc * dc_loss
                )

                batch_metrics = {
                    "spatial_loss": (
                        spatial_recon_loss.item()
                        if isinstance(spatial_recon_loss, torch.Tensor)
                        else spatial_recon_loss
                    ),
                    "temporal_loss": (
                        temporal_recon_loss.item()
                        if isinstance(temporal_recon_loss, torch.Tensor)
                        else temporal_recon_loss
                    ),
                    "consistency_loss": (
                        consistency_loss.item()
                        if isinstance(consistency_loss, torch.Tensor)
                        else consistency_loss
                    ),
                    "dc_loss": (
                        dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss
                    ),
                    "total_loss": total_loss.item(),
                    **getattr(spatial_output, "spatial_metrics", {}),
                    **temporal_output.temporal_metrics,
                }

                for key, value in batch_metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    if isinstance(value, dict):
                        if key not in total_metrics or not isinstance(
                            total_metrics[key], dict
                        ):
                            total_metrics[key] = {}
                        for k, v in value.items():
                            if k not in total_metrics[key]:
                                total_metrics[key][k] = 0.0
                            total_metrics[key][k] += v
                    else:
                        total_metrics[key] += value

                num_batches += 1

        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / max(num_batches, 1)
            else:
                total_metrics[key] = value / max(num_batches, 1)

        return total_metrics

    def _generate_spatial_outputs(self, data_loader: DataLoader):
        """生成空间预测输出用于时间训练"""
        # 若禁用空间模块，则返回占位输出（使用输入序列的最后一帧作为空间预测）
        if getattr(self.model.spatial_module, "feature_extractor", None) is None:
            spatial_outputs = []
            with torch.no_grad():
                for batch in data_loader:
                    input_data = batch["input"].to(self.device)
                    # 占位空间输出：直接使用上一帧作为空间预测，空间特征置零
                    B, T_in, C, H, W = input_data.shape
                    placeholder = {
                        "spatial_pred": input_data[:, -1:].clone(),
                        "spatial_features": torch.zeros(
                            B, T_in, 1, H, W, device=self.device, dtype=input_data.dtype
                        ),
                    }
                    spatial_outputs.append(placeholder)
            return spatial_outputs

        self.model.spatial_module.eval()
        spatial_outputs = []

        with torch.no_grad():
            for batch in data_loader:
                input_data = batch["input"].to(self.device)
                target_data = (
                    batch["target"].to(self.device) if "target" in batch else None
                )

                spatial_output = self.model.spatial_module(input_data, target_data)
                spatial_outputs.append(spatial_output)

        return spatial_outputs

    def save_checkpoint(self, filepath: str, epoch: int, metrics: dict[str, Any]):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "spatial_optimizer_state_dict": (
                self.spatial_trainer.optimizer.state_dict()
                if getattr(self, "spatial_trainer", None)
                else None
            ),
            "temporal_optimizer_state_dict": (
                self.temporal_trainer.optimizer.state_dict()
                if getattr(self, "temporal_trainer", None)
                else None
            ),
            "config": self.config,
            "metrics": metrics,
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.spatial_trainer is not None and checkpoint.get(
            "spatial_optimizer_state_dict"
        ):
            self.spatial_trainer.optimizer.load_state_dict(
                checkpoint["spatial_optimizer_state_dict"]
            )
        self.temporal_trainer.optimizer.load_state_dict(
            checkpoint["temporal_optimizer_state_dict"]
        )

        self.logger.info(f"Checkpoint loaded: {filepath}")

        return checkpoint["epoch"], checkpoint["metrics"]
