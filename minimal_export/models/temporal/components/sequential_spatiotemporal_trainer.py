"""
分阶段时空预测训练器 - Sequential Spatiotemporal Trainer
基于技术架构文档实现联合微调阶段的时空序列预测
"""

import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.data_consistency import DataConsistencyChecker, DegradationEquivalenceChecker

# from utils.visualization import ARTrainingVisualizer  # 暂时注释掉，避免导入错误
from utils.logging_utils import setup_logger
from utils.metrics import psnr_metric, ssim_metric


class SpatialPredictionModule(nn.Module):
    """空间预测模块 - 负责空间特征提取和空间预测"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = self._build_feature_extractor()
        self.prediction_head = self._build_prediction_head()
        self.feature_normalizer = self._build_feature_normalizer()

    def _build_feature_extractor(self) -> nn.Module:
        """构建空间特征提取器"""
        swin_cls = None
        try:
            swin_cls = getattr(
                import_module("models.sequential_spatiotemporal_trainer"),
                "SwinUNet",
                None,
            )
        except Exception:
            swin_cls = None
        if swin_cls is None:
            from models.swin_unet import SwinUNet as swin_cls
        return swin_cls(
            in_channels=self.config.data.T_in * self.config.data.channels,
            out_channels=self.config.spatial.feature_dim,
            img_size=self.config.data.img_size,
            patch_size=self.config.model.patch_size,
            window_size=self.config.model.window_size,
            depths=self.config.model.depths,
            num_heads=self.config.model.num_heads,
            embed_dim=self.config.model.embed_dim,
        )

    def _build_prediction_head(self) -> nn.Module:
        """构建空间预测头"""
        return nn.Sequential(
            nn.Conv2d(
                self.config.spatial.feature_dim, self.config.data.channels, 3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.data.channels, self.config.data.channels, 3, padding=1
            ),
        )

    def _build_feature_normalizer(self) -> nn.Module:
        """构建特征标准化器"""
        return nn.GroupNorm(num_groups=8, num_channels=self.config.spatial.feature_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播"""
        B, T_in, C, H, W = x.shape

        # 重塑输入: [B, T_in, C, H, W] -> [B, T_in*C, H, W]
        x_reshaped = x.reshape(B, T_in * C, H, W)

        # 提取空间特征
        spatial_features = self.feature_extractor(x_reshaped)

        # 标准化特征
        normalized_features = self.feature_normalizer(spatial_features)

        # 生成空间预测
        spatial_pred = self.prediction_head(spatial_features)

        # 扩展时间维度: [B, C, H, W] -> [B, T_out, C, H, W]
        T_out = self.config.data.T_out
        spatial_pred_expanded = spatial_pred.unsqueeze(1).expand(B, T_out, C, H, W)
        normalized_features_expanded = normalized_features.unsqueeze(1).expand(
            B, T_out, -1, H, W
        )

        return {
            "spatial_pred": spatial_pred_expanded,  # [B, T_out, C, H, W]
            "spatial_features": normalized_features_expanded,  # [B, T_out, C_feat, H, W]
            "raw_features": spatial_features,  # [B, C_feat, H, W]
        }


class TemporalPredictionModule(nn.Module):
    """时间预测模块 - 负责时间特征提取和时间预测"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.temporal_encoder = self._build_temporal_encoder()
        self.temporal_decoder = self._build_temporal_decoder()
        self.prediction_head = self._build_prediction_head()

    def _build_temporal_encoder(self) -> nn.Module:
        """构建时序编码器"""
        if self.config.temporal.encoder_type == "transformer":
            return TemporalTransformerEncoder(self.config)
        elif self.config.temporal.encoder_type == "conv1d":
            return TemporalConv1DEncoder(self.config)
        else:
            raise ValueError(
                f"Unknown encoder type: {self.config.temporal.encoder_type}"
            )

    def _build_temporal_decoder(self) -> nn.Module:
        """构建时序解码器"""
        if self.config.temporal.encoder_type == "transformer":
            return TemporalTransformerDecoder(self.config)
        elif self.config.temporal.encoder_type == "conv1d":
            return TemporalConv1DDecoder(self.config)
        else:
            raise ValueError(
                f"Unknown encoder type: {self.config.temporal.encoder_type}"
            )

    def _build_prediction_head(self) -> nn.Module:
        """构建预测头"""
        return nn.Sequential(
            nn.Linear(self.config.temporal.d_model, self.config.data.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.data.channels, self.config.data.channels),
        )

    def forward(
        self, spatial_results: dict[str, torch.Tensor], x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """前向传播"""
        spatial_pred = spatial_results["spatial_pred"]  # [B, T_out, C, H, W]
        spatial_features = spatial_results[
            "spatial_features"
        ]  # [B, T_out, C_feat, H, W]

        B, T_out, C, H, W = spatial_pred.shape

        spatial_pred_pooled = spatial_pred.mean(dim=(-1, -2))  # [B, T_out, C]
        if self.config.temporal.use_spatial_features:
            spatial_feat_pooled = spatial_features.mean(
                dim=(-1, -2)
            )  # [B, T_out, C_feat]
            temporal_input = torch.cat(
                [spatial_feat_pooled, spatial_pred_pooled], dim=-1
            )  # [B, T_out, C_feat+C]
        else:
            temporal_input = spatial_pred_pooled  # [B, T_out, C]

        temporal_encoded = self.temporal_encoder(temporal_input)  # [B, T_out, d_model]
        _ = self.temporal_decoder(
            temporal_encoded
        )  # [B, T_out, C_feat+C] or [B, T_out, C]

        delta = self.prediction_head(temporal_encoded)  # [B, T_out, C]
        final_pred = spatial_pred + delta[:, :, :, None, None].expand(B, T_out, C, H, W)

        return {
            "final_pred": final_pred,  # [B, T_out, C, H, W]
            "temporal_features": temporal_encoded,  # [B, T_out, d_model]
            "spatial_features": spatial_features,  # [B, T_out, C_feat, H, W]
        }


class TemporalTransformerEncoder(nn.Module):
    """时序Transformer编码器"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.positional_encoding = PositionalEncoding(
            d_model=config.temporal.d_model,
            dropout=config.temporal.dropout,
            max_len=config.data.T_out,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.temporal.d_model,
            nhead=config.temporal.nhead,
            dim_feedforward=config.temporal.dim_feedforward,
            dropout=config.temporal.dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.temporal.num_layers
        )

        in_dim = (
            int(config.spatial.feature_dim + config.data.channels)
            if bool(config.temporal.use_spatial_features)
            else int(config.data.channels)
        )
        self.input_projection = nn.Linear(in_dim, config.temporal.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x: [B, T_out, C_feat+C] or [B, T_out, C]
        x_projected = self.input_projection(x)  # [B, T_out, d_model]

        # 添加位置编码
        x_pos = self.positional_encoding(x_projected.transpose(0, 1)).transpose(
            0, 1
        )  # [B, T_out, d_model]

        # Transformer编码
        encoded = self.transformer_encoder(x_pos)  # [B, T_out, d_model]
        return encoded  # [B, T_out, d_model]


class TemporalTransformerDecoder(nn.Module):
    """时序Transformer解码器"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.temporal.d_model,
            nhead=config.temporal.nhead,
            dim_feedforward=config.temporal.dim_feedforward,
            dropout=config.temporal.dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.temporal.num_layers
        )

        out_dim = (
            int(config.spatial.feature_dim + config.data.channels)
            if bool(config.temporal.use_spatial_features)
            else int(config.data.channels)
        )
        self.output_projection = nn.Linear(config.temporal.d_model, out_dim)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # encoded: [B, T_out, d_model]
        decoded = self.transformer_decoder(encoded, encoded)  # [B, T_out, d_model]

        # 投影回原始维度
        return self.output_projection(decoded)  # [B, T_out, C_feat+C]


class TemporalConv1DEncoder(nn.Module):
    """1D卷积时序编码器"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        in_dim = (
            int(config.spatial.feature_dim + config.data.channels)
            if bool(config.temporal.use_spatial_features)
            else int(config.data.channels)
        )
        self.input_projection = nn.Linear(in_dim, config.temporal.d_model)

        # 1D卷积层
        layers = []
        in_channels = config.temporal.d_model
        current_channels = in_channels

        for i, out_channels in enumerate(config.temporal.conv_channels):
            layers.append(
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=config.temporal.kernel_size,
                    padding=config.temporal.kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            if i < len(config.temporal.conv_channels) - 1:
                layers.append(nn.Dropout(config.temporal.dropout))

            current_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x: [B, T_out, C_feat+C] or [B, T_out, C]
        x_projected = self.input_projection(x)  # [B, T_out, d_model]
        x_reshaped = x_projected.transpose(1, 2)  # [B, d_model, T_out]

        # 1D卷积编码
        encoded = self.conv_layers(x_reshaped)  # [B, d_model, T_out]
        return encoded.transpose(1, 2)  # [B, T_out, d_model]


class TemporalConv1DDecoder(nn.Module):
    """1D卷积时序解码器"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # 1D反卷积层
        layers = []
        conv_channels = config.temporal.conv_channels[::-1]  # 反转通道顺序
        in_channels = conv_channels[0] if conv_channels else config.temporal.d_model
        current_channels = in_channels

        out_dim = (
            int(config.spatial.feature_dim + config.data.channels)
            if bool(config.temporal.use_spatial_features)
            else int(config.data.channels)
        )
        for i, out_channels in enumerate(conv_channels[1:] + [out_dim]):
            layers.append(
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=config.temporal.kernel_size,
                    padding=config.temporal.kernel_size // 2,
                )
            )

            if i < len(conv_channels) - 1:
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(config.temporal.dropout))

            current_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # encoded: [B, T_out, d_model]
        encoded_reshaped = encoded.transpose(1, 2)  # [B, d_model, T_out]

        # 1D反卷积解码
        decoded = self.conv_layers(encoded_reshaped)  # [B, C_feat+C, T_out]
        return decoded.transpose(1, 2)  # [B, T_out, C_feat+C]


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class SequentialSpatiotemporalTrainer:
    """分阶段时空预测训练器 - 支持联合微调"""

    def __init__(self, config: Any):
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        self.config = config
        self._normalize_config()
        self.device = torch.device(
            self.config.device
            if hasattr(self.config, "device")
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not hasattr(self.config, "logging"):
            self.config.logging = OmegaConf.create(
                {"log_interval": 10, "checkpoint_interval": 50}
            )
        if not hasattr(self.config, "data_consistency"):
            self.config.data_consistency = OmegaConf.create({"check_interval": 100})
        if not hasattr(self.config, "output_dir"):
            self.config.output_dir = Path(tempfile.mkdtemp(prefix="s2f_run_"))
        else:
            self.config.output_dir = Path(self.config.output_dir)

        # 初始化模块
        self.spatial_module = SpatialPredictionModule(config)
        self.temporal_module = TemporalPredictionModule(config)

        # 移动到设备
        self.spatial_module.to(self.device)
        self.temporal_module.to(self.device)

        # 初始化优化器
        self.spatial_optimizer = self._build_spatial_optimizer()
        self.temporal_optimizer = self._build_temporal_optimizer()

        # 初始化学习率调度器
        self.spatial_scheduler = self._build_spatial_scheduler()
        self.temporal_scheduler = self._build_temporal_scheduler()

        # 初始化损失函数
        self.spatial_loss_fn = self._build_spatial_loss()
        self.temporal_loss_fn = self._build_temporal_loss()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # 数据一致性检查器
        self.data_consistency_checker = DataConsistencyChecker(config)
        self.degradation_checker = DegradationEquivalenceChecker()

        # 日志记录
        self.logger = setup_logger("sequential_trainer")

    def _normalize_config(self) -> None:
        if not hasattr(self.config, "data"):
            channels = 1
            img_size = 64
            if hasattr(self.config, "model") and hasattr(self.config.model, "spatial"):
                channels = int(
                    getattr(self.config.model.spatial, "in_channels", channels)
                )
                img_size = int(getattr(self.config.model.spatial, "img_size", img_size))
            self.config.data = OmegaConf.create(
                {"T_in": 1, "T_out": 1, "channels": channels, "img_size": img_size}
            )
        else:
            if not hasattr(self.config.data, "T_in"):
                self.config.data.T_in = 1
            if not hasattr(self.config.data, "T_out"):
                self.config.data.T_out = 1
            if not hasattr(self.config.data, "channels"):
                self.config.data.channels = 1
            if not hasattr(self.config.data, "img_size"):
                self.config.data.img_size = 64

        if not hasattr(self.config, "spatial"):
            feature_dim = 128
            if hasattr(self.config, "model") and hasattr(self.config.model, "spatial"):
                feature_dim = int(
                    getattr(self.config.model.spatial, "hidden_dim", feature_dim)
                )
            self.config.spatial = OmegaConf.create({"feature_dim": feature_dim})
        else:
            if not hasattr(self.config.spatial, "feature_dim"):
                self.config.spatial.feature_dim = 128

        if not hasattr(self.config, "temporal"):
            self.config.temporal = OmegaConf.create(
                {
                    "encoder_type": "transformer",
                    "d_model": 256,
                    "nhead": 8,
                    "num_layers": 4,
                    "dim_feedforward": 1024,
                    "dropout": 0.1,
                    "conv_channels": [256, 512, 256],
                    "kernel_size": 3,
                    "use_spatial_features": True,
                }
            )

        if not hasattr(self.config, "model"):
            self.config.model = OmegaConf.create({})
        for k, v in {
            "patch_size": 4,
            "window_size": 8,
            "depths": [2, 2, 2],
            "num_heads": [4, 8, 16],
            "embed_dim": 96,
        }.items():
            if not hasattr(self.config.model, k):
                setattr(self.config.model, k, v)

        if not hasattr(self.config, "training"):
            self.config.training = OmegaConf.create({})
        if not hasattr(self.config.training, "spatial_lr"):
            self.config.training.spatial_lr = float(
                getattr(self.config.training, "lr", 1e-4)
            )
        if not hasattr(self.config.training, "temporal_lr"):
            self.config.training.temporal_lr = float(
                getattr(self.config.training, "lr", 1e-4)
            )
        if not hasattr(self.config.training, "spatial_weight_decay"):
            self.config.training.spatial_weight_decay = float(
                getattr(self.config.training, "weight_decay", 0.0)
            )
        if not hasattr(self.config.training, "temporal_weight_decay"):
            self.config.training.temporal_weight_decay = float(
                getattr(self.config.training, "weight_decay", 0.0)
            )
        if not hasattr(self.config.training, "spatial_scheduler"):
            if hasattr(self.config.training, "scheduler") and hasattr(
                self.config.training.scheduler, "type"
            ):
                self.config.training.spatial_scheduler = (
                    self.config.training.scheduler.type
                )
                self.config.training.temporal_scheduler = (
                    self.config.training.scheduler.type
                )
            else:
                self.config.training.spatial_scheduler = "cosine"
                self.config.training.temporal_scheduler = "cosine"
        if not hasattr(self.config.training, "temporal_scheduler"):
            self.config.training.temporal_scheduler = (
                self.config.training.spatial_scheduler
            )
        if not hasattr(self.config.training, "epochs"):
            self.config.training.epochs = 1

    def _build_spatial_optimizer(self):
        """构建空间优化器"""
        return AdamW(
            self.spatial_module.parameters(),
            lr=self.config.training.spatial_lr,
            weight_decay=self.config.training.spatial_weight_decay,
        )

    def _build_temporal_optimizer(self):
        """构建时序优化器"""
        return AdamW(
            self.temporal_module.parameters(),
            lr=self.config.training.temporal_lr,
            weight_decay=self.config.training.temporal_weight_decay,
        )

    def _build_spatial_scheduler(self):
        """构建空间学习率调度器"""
        if self.config.training.spatial_scheduler == "cosine":
            return CosineAnnealingLR(
                self.spatial_optimizer, T_max=self.config.training.epochs, eta_min=1e-6
            )
        return None

    def _build_temporal_scheduler(self):
        """构建时序学习率调度器"""
        if self.config.training.temporal_scheduler == "cosine":
            return CosineAnnealingLR(
                self.temporal_optimizer, T_max=self.config.training.epochs, eta_min=1e-6
            )
        return None

    def _build_spatial_loss(self):
        """构建空间损失函数"""
        return nn.MSELoss()

    def _build_temporal_loss(self):
        """构建时序损失函数"""
        return nn.MSELoss()

    def _ensure_time_dim(self, tensor: torch.Tensor, target_t: int) -> torch.Tensor:
        """确保张量具有正确的时间维度"""
        # [B, C, H, W] -> [B, 1, C, H, W]
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)

        # 截断或填充
        if tensor.shape[1] > target_t:
            tensor = tensor[:, :target_t]
        elif tensor.shape[1] < target_t:
            # 这种情况比较少见，通常需要报错或特殊处理
            # 这里简单重复最后一帧（仅作防御）
            repeats = target_t - tensor.shape[1]
            last_frame = tensor[:, -1:].repeat(1, repeats, 1, 1, 1)
            tensor = torch.cat([tensor, last_frame], dim=1)

        return tensor

    def _get_loss_weight(self, key: str, default: float = 1.0) -> float:
        try:
            if key == "spatial":
                if hasattr(self.config, "spatial") and hasattr(
                    self.config.spatial, "loss_weight"
                ):
                    return float(self.config.spatial.loss_weight)
                if hasattr(self.config, "loss") and hasattr(
                    self.config.loss, "spatial_weight"
                ):
                    return float(self.config.loss.spatial_weight)
            if key == "temporal":
                if hasattr(self.config, "temporal") and hasattr(
                    self.config.temporal, "loss_weight"
                ):
                    return float(self.config.temporal.loss_weight)
                if hasattr(self.config, "loss") and hasattr(
                    self.config.loss, "temporal_weight"
                ):
                    return float(self.config.loss.temporal_weight)
        except Exception:
            pass
        return float(default)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int = 0
    ) -> dict[str, float]:
        """训练步骤 - 支持联合微调和数据一致性检查"""
        x = self._ensure_time_dim(
            batch["input"].to(self.device), int(self.config.data.T_in)
        )
        y = self._ensure_time_dim(
            batch["target"].to(self.device), int(self.config.data.T_out)
        )

        # 数据一致性检查（每N个batch检查一次）
        if (
            batch_idx % getattr(self.config.data_consistency, "check_interval", 100)
            == 0
        ):
            self._perform_data_consistency_check(batch)

        total_loss = 0.0
        spatial_loss_val = 0.0
        temporal_loss_val = 0.0

        # 1. 空间阶段训练 (仅当 spatial_module 未冻结时)
        run_spatial = any(p.requires_grad for p in self.spatial_module.parameters())
        # 2. 时序阶段训练 (仅当 temporal_module 未冻结时)
        run_temporal = any(p.requires_grad for p in self.temporal_module.parameters())

        # 特殊情况：如果两个都未冻结（联合微调），则需要端到端计算
        # 如果只有空间未冻结（Phase 1），则不需要跑 Temporal
        # 如果只有时序未冻结（Phase 2），则需要跑 Spatial (no_grad) 然后跑 Temporal

        # 阶段1: 空间预测
        if run_spatial:
            spatial_results = self.spatial_module(x)
            spatial_loss = self._calculate_spatial_loss(spatial_results, y)
            total_loss += spatial_loss
            spatial_loss_val = spatial_loss.item()
        else:
            with torch.no_grad():
                spatial_results = self.spatial_module(x)

        # 阶段2: 时间预测
        if run_temporal:
            # 在 Phase 2 (Temporal Only) 中，spatial_results 必须 detach 以防止梯度回传到冻结的 Spatial
            # 但在 Phase 3 (Joint) 中，需要保持梯度链
            if not run_spatial:
                # Detach spatial outputs if spatial is frozen
                spatial_results_detached = {
                    k: v.detach() if isinstance(v, torch.Tensor) else v
                    for k, v in spatial_results.items()
                }
                temporal_results = self.temporal_module(spatial_results_detached, x)
            else:
                temporal_results = self.temporal_module(spatial_results, x)

            temporal_loss = self._calculate_temporal_loss(temporal_results, y)
            total_loss += temporal_loss
            temporal_loss_val = temporal_loss.item()
            final_pred = temporal_results["final_pred"]
        else:
            # 仅 Spatial 训练时，不跑 Temporal
            final_pred = spatial_results["spatial_pred"]

        # 反向传播
        if run_spatial:
            self.spatial_optimizer.zero_grad()
        if run_temporal:
            self.temporal_optimizer.zero_grad()

        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()

            # 梯度裁剪
            if hasattr(self.config.training, "grad_clip"):
                if run_temporal:
                    torch.nn.utils.clip_grad_norm_(
                        self.temporal_module.parameters(),
                        self.config.training.grad_clip,
                    )
                if run_spatial:
                    torch.nn.utils.clip_grad_norm_(
                        self.spatial_module.parameters(), self.config.training.grad_clip
                    )

            if run_spatial:
                self.spatial_optimizer.step()
            if run_temporal:
                self.temporal_optimizer.step()

            total_loss_val = total_loss.item()
        else:
            total_loss_val = 0.0

        # 计算指标
        with torch.no_grad():
            spatial_metrics = self._calculate_spatial_metrics(
                spatial_results["spatial_pred"], y
            )
            if run_temporal:
                temporal_metrics = self._calculate_temporal_metrics(final_pred, y)
            else:
                temporal_metrics = {}

        return {
            "joint_loss": total_loss_val,
            "spatial_loss": spatial_loss_val,
            "temporal_loss": temporal_loss_val,
            "total_loss": total_loss_val,
            **spatial_metrics,
            **temporal_metrics,
        }

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int = 0
    ) -> dict[str, float]:
        """验证步骤"""
        x = self._ensure_time_dim(
            batch["input"].to(self.device), int(self.config.data.T_in)
        )
        y = self._ensure_time_dim(
            batch["target"].to(self.device), int(self.config.data.T_out)
        )

        with torch.no_grad():
            # 空间预测
            spatial_results = self.spatial_module(x)

            # 时间预测
            temporal_results = self.temporal_module(spatial_results, x)

            # 计算验证损失
            spatial_loss = self._calculate_spatial_loss(spatial_results, y)
            temporal_loss = self._calculate_temporal_loss(temporal_results, y)

            # 计算验证指标
            spatial_metrics = self._calculate_spatial_metrics(
                spatial_results["spatial_pred"], y
            )
            temporal_metrics = self._calculate_temporal_metrics(
                temporal_results["final_pred"], y
            )

            # 总体指标
            overall_rel_l2 = torch.norm(
                temporal_results["final_pred"] - y
            ) / torch.norm(y)

            total_loss = spatial_loss + temporal_loss
            return {
                "val_joint_loss": total_loss.item(),
                "val_spatial_loss": spatial_loss.item(),
                "val_temporal_loss": temporal_loss.item(),
                "val_total_loss": total_loss.item(),
                "val_spatial_rel_l2": spatial_metrics.get("spatial_rel_l2", 0.0),
                "val_temporal_rel_l2": temporal_metrics.get("temporal_rel_l2", 0.0),
                "val_overall_rel_l2": overall_rel_l2.item(),
            }

    def staged_validation(self, val_dataloader, epoch: int) -> dict[str, float]:
        """分阶段验证

        1. 空间阶段验证：仅评估空间模块性能
        2. 时序阶段验证：仅评估时序模块性能
        3. 联合阶段验证：评估联合模型性能
        """
        self.logger.info(f"Performing staged validation for epoch {epoch}")

        # 阶段1：空间模块验证
        spatial_metrics = self._validate_spatial_stage(val_dataloader)

        # 阶段2：时序模块验证
        temporal_metrics = self._validate_temporal_stage(val_dataloader)

        # 阶段3：联合模型验证
        joint_metrics = self._validate_joint_stage(val_dataloader)

        # 合并所有阶段的指标
        all_metrics = {}
        for key, value in spatial_metrics.items():
            all_metrics[f"spatial_{key}"] = value
        for key, value in temporal_metrics.items():
            all_metrics[f"temporal_{key}"] = value
        for key, value in joint_metrics.items():
            all_metrics[f"joint_{key}"] = value

        self.logger.info("Staged validation completed:")
        self.logger.info(
            f"  Spatial stage - Loss: {spatial_metrics.get('loss', 0):.6f}"
        )
        self.logger.info(
            f"  Temporal stage - Loss: {temporal_metrics.get('loss', 0):.6f}"
        )
        self.logger.info(f"  Joint stage - Loss: {joint_metrics.get('loss', 0):.6f}")

        return all_metrics

    def _validate_spatial_stage(self, dataloader) -> dict[str, float]:
        """验证空间阶段"""
        self.spatial_module.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = self._ensure_time_dim(
                    batch["input"].to(self.device), int(self.config.data.T_in)
                )
                y = self._ensure_time_dim(
                    batch["target"].to(self.device), int(self.config.data.T_out)
                )

                spatial_results = self.spatial_module(x)
                loss = self._calculate_spatial_loss(spatial_results, y)

                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    def _validate_temporal_stage(self, dataloader) -> dict[str, float]:
        """验证时序阶段"""
        self.spatial_module.eval()
        self.temporal_module.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = self._ensure_time_dim(
                    batch["input"].to(self.device), int(self.config.data.T_in)
                )
                y = self._ensure_time_dim(
                    batch["target"].to(self.device), int(self.config.data.T_out)
                )

                # 在线生成空间特征，避免 OOM
                spatial_results = self.spatial_module(x)
                temporal_results = self.temporal_module(spatial_results, x)
                loss = self._calculate_temporal_loss(temporal_results, y)

                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    def _validate_joint_stage(self, dataloader) -> dict[str, float]:
        """验证联合阶段"""
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                metrics = self.validation_step(batch, batch_idx)
                total_loss += metrics["val_joint_loss"]
                num_batches += 1

        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    def _calculate_spatial_loss(
        self, spatial_results: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """计算空间损失"""
        spatial_pred = spatial_results["spatial_pred"]
        return self.spatial_loss_fn(spatial_pred, target) * self._get_loss_weight(
            "spatial", default=1.0
        )

    def _calculate_temporal_loss(
        self, temporal_results: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """计算时序损失"""
        final_pred = temporal_results["final_pred"]
        return self.temporal_loss_fn(final_pred, target) * self._get_loss_weight(
            "temporal", default=1.0
        )

    def _calculate_spatial_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """计算空间指标"""
        metrics = {}

        # 基础精度指标
        diff = pred - target
        metrics["spatial_rel_l2"] = (torch.norm(diff) / torch.norm(target)).item()
        metrics["spatial_mae"] = F.l1_loss(pred, target).item()
        metrics["spatial_rmse"] = torch.sqrt(F.mse_loss(pred, target)).item()
        try:
            pred_img = pred[:, -1, ...] if pred.dim() == 5 else pred
            target_img = target[:, -1, ...] if target.dim() == 5 else target
            metrics["spatial_ssim"] = float(ssim_metric(pred_img, target_img))
            metrics["spatial_psnr"] = float(psnr_metric(pred_img, target_img))
        except Exception:
            metrics["spatial_ssim"] = 0.0
            metrics["spatial_psnr"] = 0.0

        return metrics

    def _calculate_temporal_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """计算时序指标"""
        metrics = {}

        # 时序精度指标
        diff = pred - target
        metrics["temporal_rel_l2"] = (torch.norm(diff) / torch.norm(target)).item()
        metrics["temporal_mae"] = F.l1_loss(pred, target).item()
        metrics["temporal_rmse"] = torch.sqrt(F.mse_loss(pred, target)).item()

        # 时序相关性（简化版）
        B, T, C, H, W = pred.shape
        pred_temporal = pred.reshape(B, T, -1)
        target_temporal = target.reshape(B, T, -1)

        correlations = []
        for b in range(B):
            for i in range(pred_temporal.shape[2]):
                pred_seq = pred_temporal[b, :, i]
                target_seq = target_temporal[b, :, i]

                pred_mean = pred_seq.mean()
                target_mean = target_seq.mean()

                pred_centered = pred_seq - pred_mean
                target_centered = target_seq - target_mean

                # 避免除零
                pred_norm = torch.sqrt(torch.sum(pred_centered**2))
                target_norm = torch.sqrt(torch.sum(target_centered**2))

                if pred_norm > 1e-8 and target_norm > 1e-8:
                    correlation = torch.sum(pred_centered * target_centered) / (
                        pred_norm * target_norm
                    )
                    correlations.append(correlation.item())

        metrics["temporal_correlation"] = np.mean(correlations) if correlations else 0.0

        return metrics

    def _perform_data_consistency_check(self, batch: dict[str, torch.Tensor]):
        """执行数据一致性检查"""
        try:
            x = self._ensure_time_dim(batch["input"], int(self.config.data.T_in))
            y = self._ensure_time_dim(batch["target"], int(self.config.data.T_out))

            # 检查数据管道一致性
            pipeline_check = (
                self.data_consistency_checker.check_data_pipeline_consistency(
                    raw_data=x,
                    processed_data=x,  # 这里可以传入原始数据作为对比
                    data_pipeline=None,
                    check_normalization=True,
                )
            )

            # 检查时序一致性
            temporal_check = self.data_consistency_checker.check_temporal_consistency(
                pred_sequence=x,  # 使用输入序列作为预测序列的代理
                target_sequence=y,
                temporal_smoothness_threshold=0.1,
            )

            if not pipeline_check["consistent"]:
                self.logger.warning("Data pipeline consistency check failed")

            if not temporal_check["consistent"]:
                self.logger.warning("Temporal consistency check failed")

        except Exception as e:
            self.logger.error(f"Data consistency check error: {e}")

    def check_degradation_equivalence(
        self,
        degradation_op1: nn.Module,
        degradation_op2: nn.Module,
        test_data: torch.Tensor,
    ) -> dict[str, Any]:
        """检查降质算子等价性"""
        return self.degradation_checker.check_equivalence(
            degradation_op1=degradation_op1,
            degradation_op2=degradation_op2,
            test_data=test_data,
            num_samples=100,
        )

    def train_epoch(self, dataloader, epoch: int) -> dict[str, float]:
        """训练一个epoch"""
        # 根据冻结状态设置模式
        if all(not p.requires_grad for p in self.spatial_module.parameters()):
            self.spatial_module.eval()
        else:
            self.spatial_module.train()

        if all(not p.requires_grad for p in self.temporal_module.parameters()):
            self.temporal_module.eval()
        else:
            self.temporal_module.train()

        self.current_epoch = epoch

        epoch_metrics = []

        for batch_idx, batch in enumerate(dataloader):
            # 自动处理不同阶段的训练逻辑
            step_metrics = self.training_step(batch, batch_idx)
            epoch_metrics.append(step_metrics)
            self.global_step += 1

            # 日志记录
            if batch_idx % self.config.logging.log_interval == 0:
                self.logger.info(
                    f"Epoch [{epoch}] Step [{batch_idx}/{len(dataloader)}] - "
                    f"Spatial Loss: {step_metrics['spatial_loss']:.6f}, "
                    f"Temporal Loss: {step_metrics['temporal_loss']:.6f}, "
                    f"Total Loss: {step_metrics['total_loss']:.6f}"
                )

        # 聚合epoch指标
        avg_metrics = {}
        if epoch_metrics:
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        avg_metrics["train_loss"] = float(
            avg_metrics.get("total_loss", avg_metrics.get("joint_loss", 0.0))
        )
        return avg_metrics

    def validate_epoch(self, dataloader, epoch: int = 0) -> dict[str, float]:
        """验证一个epoch"""
        staged_metrics = self.staged_validation(dataloader, epoch)
        return {
            "val_loss": staged_metrics.get("joint_loss", 0.0),
            "val_spatial_loss": staged_metrics.get("spatial_loss", 0.0),
            "val_temporal_loss": staged_metrics.get("temporal_loss", 0.0),
        }

    def train(self, train_dataloader, val_dataloader, num_epochs: int):
        """主训练循环"""
        self.logger.info("Starting sequential spatiotemporal training...")

        best_model_path = None
        training_history = []

        for epoch in range(num_epochs):
            # 训练阶段
            train_metrics = self.train_epoch(train_dataloader, epoch)

            # 验证阶段
            val_metrics = self.validate_epoch(val_dataloader)

            # 更新学习率调度器
            if self.spatial_scheduler:
                self.spatial_scheduler.step()
            if self.temporal_scheduler:
                self.temporal_scheduler.step()

            # 记录历史
            epoch_history = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "spatial_lr": self.spatial_optimizer.param_groups[0]["lr"],
                "temporal_lr": self.temporal_optimizer.param_groups[0]["lr"],
            }
            training_history.append(epoch_history)

            # 日志记录
            self.logger.info(
                f"Epoch [{epoch}/{num_epochs}] - "
                f"Train Total Loss: {train_metrics['total_loss']:.6f}, "
                f"Val Total Loss: {val_metrics['val_total_loss']:.6f}, "
                f"Val Overall Rel-L2: {val_metrics['val_overall_rel_l2']:.6f}"
            )

            # 保存最佳模型
            if val_metrics["val_total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_total_loss"]
                best_model_path = self.save_checkpoint(epoch, val_metrics)
                self.logger.info(f"New best model saved at epoch {epoch}")

        self.logger.info("Training completed!")
        return training_history, best_model_path

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> str:
        """保存检查点"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"

        checkpoint = {
            "epoch": epoch,
            "spatial_module_state_dict": self.spatial_module.state_dict(),
            "temporal_module_state_dict": self.temporal_module.state_dict(),
            "spatial_optimizer_state_dict": self.spatial_optimizer.state_dict(),
            "temporal_optimizer_state_dict": self.temporal_optimizer.state_dict(),
            "spatial_scheduler_state_dict": (
                self.spatial_scheduler.state_dict() if self.spatial_scheduler else None
            ),
            "temporal_scheduler_state_dict": (
                self.temporal_scheduler.state_dict()
                if self.temporal_scheduler
                else None
            ),
            "metrics": metrics,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Any):
        """加载检查点"""
        checkpoint = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )

        if "config" in checkpoint and checkpoint["config"] is not None:
            self.config = checkpoint["config"]
            if not isinstance(self.config, DictConfig):
                self.config = OmegaConf.create(self.config)
            self._normalize_config()
            if not hasattr(self.config, "output_dir"):
                self.config.output_dir = Path(tempfile.mkdtemp(prefix="s2f_run_"))
            else:
                self.config.output_dir = Path(self.config.output_dir)
            self.device = torch.device(
                self.config.device
                if hasattr(self.config, "device")
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.spatial_module = SpatialPredictionModule(self.config).to(self.device)
            self.temporal_module = TemporalPredictionModule(self.config).to(self.device)
            self.spatial_optimizer = self._build_spatial_optimizer()
            self.temporal_optimizer = self._build_temporal_optimizer()
            self.spatial_scheduler = self._build_spatial_scheduler()
            self.temporal_scheduler = self._build_temporal_scheduler()
            self.spatial_loss_fn = self._build_spatial_loss()
            self.temporal_loss_fn = self._build_temporal_loss()

        self.spatial_module.load_state_dict(checkpoint["spatial_module_state_dict"])
        self.temporal_module.load_state_dict(checkpoint["temporal_module_state_dict"])
        try:
            self.spatial_optimizer.load_state_dict(
                checkpoint["spatial_optimizer_state_dict"]
            )
            self.temporal_optimizer.load_state_dict(
                checkpoint["temporal_optimizer_state_dict"]
            )
        except Exception:
            pass

        try:
            if self.spatial_scheduler and checkpoint.get(
                "spatial_scheduler_state_dict"
            ):
                self.spatial_scheduler.load_state_dict(
                    checkpoint["spatial_scheduler_state_dict"]
                )
            if self.temporal_scheduler and checkpoint.get(
                "temporal_scheduler_state_dict"
            ):
                self.temporal_scheduler.load_state_dict(
                    checkpoint["temporal_scheduler_state_dict"]
                )
        except Exception:
            pass

        self.current_epoch = int(checkpoint["epoch"])
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        metrics = checkpoint.get("metrics", {})

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return self.current_epoch, metrics

    def test(self, test_dataloader) -> dict[str, Any]:
        """测试模型"""
        self.spatial_module.eval()
        self.temporal_module.eval()

        test_metrics = []
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                x = batch["input"].to(self.device)
                y = batch["target"].to(self.device)

                # 前向传播
                spatial_results = self.spatial_module(x)
                temporal_results = self.temporal_module(spatial_results, x)

                # 计算指标
                metrics = self._calculate_temporal_metrics(
                    temporal_results["final_pred"], y
                )
                test_metrics.append(metrics)

                # 收集预测和真实值
                predictions.append(temporal_results["final_pred"].cpu())
                targets.append(y.cpu())

        # 聚合指标
        avg_metrics = {}
        for key in test_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in test_metrics])

        # 收集所有预测和真实值
        all_predictions = torch.cat(predictions, dim=0)
        all_targets = torch.cat(targets, dim=0)

        # 计算总体指标
        overall_rel_l2 = torch.norm(all_predictions - all_targets) / torch.norm(
            all_targets
        )
        avg_metrics["test_overall_rel_l2"] = overall_rel_l2.item()

        return {
            "metrics": avg_metrics,
            "predictions": all_predictions,
            "targets": all_targets,
        }
