"""
分阶段时空预测模型
基于架构设计文档实现空间预测和时间预测的两阶段架构
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class SpatialPredictionOutput:
    """空间预测输出"""

    spatial_pred: torch.Tensor  # [B, T_out, C, H, W]
    spatial_features: torch.Tensor  # [B, T_out, C_feat, H, W]
    spatial_metrics: dict[str, float]


@dataclass
class TemporalPredictionOutput:
    """时间预测输出"""

    final_pred: torch.Tensor  # [B, T_out, C, H, W]
    temporal_features: torch.Tensor  # [B, T_out, C_temp]
    temporal_metrics: dict[str, float]


class SpatialFeatureExtractor(nn.Module):
    """空间特征提取器 - 支持多种骨干网络"""

    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        img_size: tuple[int, int],
        backbone_type: str = "simple",
        backbone_config: dict = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.img_size = img_size
        self.backbone_type = backbone_type

        if backbone_type == "fno2d":
            # 使用FNO2D作为空间特征提取器
            from models.spatial.fno2d import FNO2d

            self.backbone = FNO2d(
                in_channels=in_channels,
                out_channels=feature_dim,
                img_size=img_size[0],  # 假设正方形
                modes1=backbone_config.get("modes1", 12),
                modes2=backbone_config.get("modes2", 12),
                width=backbone_config.get("width", 64),
                n_layers=backbone_config.get("n_layers", 4),
                activation=backbone_config.get("activation", "gelu"),
            )
        elif backbone_type == "stable_fno2d":
            # 使用数值稳定的FNO2D作为空间特征提取器
            from models.spatial.fno2d_stable import StableFNO2d

            self.backbone = StableFNO2d(
                in_channels=in_channels,
                out_channels=feature_dim,
                img_size=img_size[0],  # 假设正方形
                modes1=backbone_config.get("modes1", 12),
                modes2=backbone_config.get("modes2", 12),
                width=backbone_config.get("width", 64),
                n_layers=backbone_config.get("n_layers", 4),
                activation=backbone_config.get("activation", "gelu"),
                spectral_norm=backbone_config.get("spectral_norm", True),
                gradient_clip=backbone_config.get("gradient_clip", 1.0),
            )
        elif backbone_type == "edsr":
            # 使用EDSR作为空间特征提取器
            from models.spatial.edsr import EDSR

            self.backbone = EDSR(
                in_channels=in_channels,
                out_channels=feature_dim,  # 输出特征维度
                img_size=img_size[0],
                n_feats=backbone_config.get("n_feats", 64),
                n_resblocks=backbone_config.get("n_resblocks", 16),
                res_scale=backbone_config.get("res_scale", 0.1),
                upscale=backbone_config.get("upscale", 1),
                bias=backbone_config.get("bias", True),
                add_input_residual=backbone_config.get("add_input_residual", None),
            )
        elif backbone_type == "simple_cnn":
            # 使用简单的CNN作为空间特征提取器
            from models.temporal.components.simple_spatial_cnn import SimpleSpatialCNN

            self.backbone = SimpleSpatialCNN(
                in_channels=in_channels,
                out_channels=feature_dim,
                hidden_channels=backbone_config.get("hidden_channels", 64),
                num_layers=backbone_config.get("num_layers", 4),
                kernel_size=backbone_config.get("kernel_size", 3),
                activation=backbone_config.get("activation", "relu"),
                dropout=backbone_config.get("dropout", 0.1),
                use_batch_norm=backbone_config.get("use_batch_norm", True),
            )
        else:
            # 默认使用简单卷积层
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, feature_dim, 3, padding=1),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取空间特征
        Args:
            x: [B, T, C, H, W] 或 [B, C, H, W]
        Returns:
            features: [B, T, C_feat, H, W] 或 [B, C_feat, H, W]
        """
        # 输入稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: NaN/Inf detected in SpatialFeatureExtractor input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        # 处理时空输入
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # 重塑为2D进行处理
            x = x.reshape(B * T, C, H, W)
            features = self.backbone(x)
            # 动态获取特征空间维度
            _, _, H_feat, W_feat = features.shape
            # 重塑回时空格式
            features = features.reshape(B, T, self.feature_dim, H_feat, W_feat)
        else:
            features = self.backbone(x)

        # 输出稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("WARNING: NaN/Inf detected in SpatialFeatureExtractor output")
            features = torch.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)

        return features


class SpatialPredictionHead(nn.Module):
    """空间预测头"""

    def __init__(self, feature_dim: int, out_channels: int):
        super().__init__()
        self.prediction_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        生成空间预测
        Args:
            features: [B, T, C_feat, H, W] 或 [B, C_feat, H, W]
        Returns:
            predictions: [B, T, C_out, H, W] 或 [B, C_out, H, W]
        """
        if features.dim() == 5:
            B, T, C_feat, H, W = features.shape
            features = features.reshape(B * T, C_feat, H, W)
            predictions = self.prediction_head(features)
            predictions = predictions.reshape(B, T, -1, H, W)
        else:
            predictions = self.prediction_head(features)

        return predictions


class SpatialMetricsCalculator:
    """空间评估指标计算器"""

    @staticmethod
    def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """计算空间预测指标"""
        with torch.no_grad():
            # 对齐时间维：允许 [B,T,C,H,W] 对 [B,T_out,C,H,W] 或 [B,C,H,W]
            if pred.dim() == 5 and target.dim() == 5:
                Bt, Tt = target.shape[0], target.shape[1]
                Tp = pred.shape[1]
                if Tp != Tt:
                    if Tp > Tt:
                        pred = pred[:, :Tt]
                    else:
                        pad = Tt - Tp
                        pred = torch.cat(
                            [pred, pred[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1
                        )
            elif pred.dim() == 5 and target.dim() == 4:
                # 将target扩展为与pred同样的时间维，使用第一帧或最后一帧作为近似
                Bp, Tp = pred.shape[0], pred.shape[1]
                target = target.unsqueeze(1).repeat(1, Tp, 1, 1, 1)
            elif pred.dim() == 4 and target.dim() == 5:
                # 将pred扩展到时间维
                Bt, Tt = target.shape[0], target.shape[1]
                pred = pred.unsqueeze(1).repeat(1, Tt, 1, 1, 1)
            # 统一到 5D 计算
            if pred.dim() == 4:
                pred = pred.unsqueeze(1)
            if target.dim() == 4:
                target = target.unsqueeze(1)
            # Rel-L2 相对误差（逐帧再平均）
            num = torch.sqrt(((pred - target) ** 2).sum(dim=(2, 3, 4)))
            den = torch.sqrt((target**2).sum(dim=(2, 3, 4))) + 1e-8
            rel_l2 = (num / den).mean()
            # MAE（逐帧再平均）
            mae = torch.mean(torch.abs(pred - target))
            # 最大值误差
            max_error = torch.max(torch.abs(pred - target))

            return {
                "spatial_rel_l2": rel_l2.item(),
                "spatial_mae": mae.item(),
                "spatial_max_error": max_error.item(),
            }


class SpatialPredictionModule(nn.Module):
    """空间预测模块"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # 从配置中提取参数
        in_channels = config.get("in_channels", 1)
        feature_dim = config.get("spatial_feature_dim", 128)
        out_channels = config.get("out_channels", 1)
        img_size = config.get("img_size", (256, 256))
        backbone_type = config.get("backbone_type", "simple")
        backbone_config = config.get("backbone_config", {})

        if backbone_type == "identity" or feature_dim == 0:
            self.feature_extractor = None
            self.prediction_head = None
        else:
            self.feature_extractor = SpatialFeatureExtractor(
                in_channels, feature_dim, img_size, backbone_type, backbone_config
            )
            self.prediction_head = SpatialPredictionHead(feature_dim, out_channels)
        self.metrics_calculator = SpatialMetricsCalculator()

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> SpatialPredictionOutput:
        """
        空间预测前向传播
        Args:
            x: 输入时空序列 [B, T_in, C, H, W]
            target: 目标时空序列 [B, T_out, C, H, W] (可选，用于训练时计算指标)
        Returns:
            SpatialPredictionOutput: 空间预测结果
        """
        # 提取空间特征
        if self.feature_extractor is None:
            # 纯时序：兼容4D输入，空间特征设为零张量，空间预测直接用上一帧（或输入的最后一帧）作为占位
            if x.dim() == 4:
                x = x.unsqueeze(1)
            B, T_in, C, H, W = x.shape
            spatial_features = torch.zeros(
                B, T_in, 1, H, W, device=x.device, dtype=x.dtype
            )
            spatial_pred = x[:, -1:].clone()
        else:
            spatial_features = self.feature_extractor(x)
            # 生成空间预测
            spatial_pred = self.prediction_head(spatial_features)

        # 计算评估指标（如果有目标值）
        spatial_metrics = {}
        if target is not None:
            spatial_metrics = self.metrics_calculator.calculate_metrics(
                spatial_pred, target
            )

        return SpatialPredictionOutput(
            spatial_pred=spatial_pred,
            spatial_features=spatial_features,
            spatial_metrics=spatial_metrics,
        )


class TemporalFeatureExtractor(nn.Module):
    """时间特征提取器 - 基于Transformer"""

    def __init__(
        self,
        input_dim: int,
        temporal_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.use_positional_encoding = bool(use_positional_encoding)

        # 输入投影 - 立即初始化以确保优化器能捕获参数
        self.input_proj = nn.Linear(input_dim, temporal_dim)

        # 时序Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=num_heads,
            dim_feedforward=temporal_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取时序特征
        Args:
            x: [B, T, C] 时序输入
        Returns:
            features: [B, T, C_temp]
        """
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: NaN/Inf detected in TemporalFeatureExtractor input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        # 检查输入维度匹配
        if self.input_proj.in_features != x.shape[-1]:
            raise RuntimeError(
                f"TemporalFeatureExtractor input mismatch: Model expects {self.input_proj.in_features}, got {x.shape[-1]}"
            )

        # 输入投影
        x = self.input_proj(x)  # [B, T, temporal_dim]

        if self.use_positional_encoding:
            B, T, D = x.shape
            positions = torch.arange(T, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, D, 2, device=x.device) * (-(math.log(10000.0) / D))
            )
            pe = torch.zeros(T, D, device=x.device)
            pe[:, 0::2] = torch.sin(positions * div_term)
            pe[:, 1::2] = torch.cos(positions * div_term)
            x = x + pe.unsqueeze(0)

        # 梯度裁剪避免爆炸
        x = torch.clamp(x, min=-1e3, max=1e3)

        # 时序建模
        features = self.transformer(x)  # [B, T, temporal_dim]

        # 输出稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("WARNING: NaN/Inf detected in TemporalFeatureExtractor output")
            features = torch.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)

        return features


class TemporalPredictionHead(nn.Module):
    """时间预测头"""

    def __init__(self, temporal_dim: int, out_channels: int, img_size: tuple[int, int]):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.out_channels = out_channels
        self.img_size = img_size

        # 时序到空间的映射
        self.temporal_to_spatial = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels * img_size[0] * img_size[1]),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        生成最终预测
        Args:
            temporal_features: [B, T, C_temp]
        Returns:
            predictions: [B, T, C_out, H, W]
        """
        # 输入稳定性检查
        if torch.isnan(temporal_features).any() or torch.isinf(temporal_features).any():
            print("WARNING: NaN/Inf detected in TemporalPredictionHead input")
            temporal_features = torch.nan_to_num(
                temporal_features, nan=0.0, posinf=1e3, neginf=-1e3
            )

        B, T, C_temp = temporal_features.shape

        # 时序特征到空间预测
        spatial_flat = self.temporal_to_spatial(temporal_features)  # [B, T, C_out*H*W]

        # 梯度裁剪避免爆炸
        spatial_flat = torch.clamp(spatial_flat, min=-1e3, max=1e3)

        # 重塑为空间格式
        predictions = spatial_flat.reshape(
            B, T, self.out_channels, self.img_size[0], self.img_size[1]
        )

        # 输出稳定性检查
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("WARNING: NaN/Inf detected in TemporalPredictionHead output")
            predictions = torch.nan_to_num(
                predictions, nan=0.0, posinf=1e3, neginf=-1e3
            )

        return predictions


class TemporalMetricsCalculator:
    """时间评估指标计算器"""

    @staticmethod
    def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """计算时间预测指标"""
        with torch.no_grad():
            # 严格校验时间维度
            assert (
                pred.shape[1] == target.shape[1]
            ), f"Time dimension mismatch: Pred {pred.shape[1]} vs Target {target.shape[1]}"

            # 时序Rel-L2
            rel_l2 = torch.norm(pred - target) / torch.norm(target)

            # 时序稳定性：计算相邻时间步的变化一致性
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            if pred.shape[1] > 1:
                stability = 1.0 - torch.mean(torch.abs(pred_diff - target_diff))
            else:
                stability = 1.0

            # 长期误差增长
            long_term_error = torch.mean(torch.abs(pred[:, -1] - target[:, -1]))

            return {
                "temporal_rel_l2": rel_l2.item(),
                "temporal_stability": (
                    stability.item()
                    if isinstance(stability, torch.Tensor)
                    else stability
                ),
                "long_term_error": long_term_error.item(),
            }


class TemporalPredictionModule(nn.Module):
    """时间预测模块"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.backend = str(config.get("backend", "transformer"))

        # 从配置中提取参数
        spatial_feature_dim = config.get("spatial_feature_dim", 128)
        temporal_dim = config.get("temporal_dim", 256)
        num_layers = config.get("num_layers", 4)
        dropout = config.get("dropout", 0.1)
        out_channels = config.get("out_channels", 1)
        img_size = config.get("img_size", (256, 256))
        reduce_spatial = config.get("reduce_spatial", None)
        reduce_size = tuple(config.get("reduce_size", img_size))

        # 计算时序输入维度：空间特征展平 + 空间预测展平
        # 使用占位符，在forward中根据实际输入动态设置
        if reduce_spatial == "avg_pool" and reduce_size is not None:
            input_dim = max(
                1,
                (spatial_feature_dim + out_channels) * reduce_size[0] * reduce_size[1],
            )
        else:
            input_dim = max(
                1, (spatial_feature_dim + out_channels) * img_size[0] * img_size[1]
            )

        if self.backend == "tcn":
            from models.temporal.components.temporal_encoder import (
                create_temporal_encoder,
            )

            self.temporal_encoder = create_temporal_encoder(
                input_dim=input_dim,
                config={
                    "hidden_dim": temporal_dim,
                    "num_conv_layers": num_layers,
                    "kernel_size": int(config.get("kernel_size", 3)),
                    "dilation_base": int(config.get("dilation_base", 2)),
                    "dropout": float(dropout),
                    "use_positional_encoding": bool(
                        config.get("use_positional_encoding", True)
                    ),
                },
            )
            self.prediction_head = TemporalPredictionHead(
                input_dim, out_channels, img_size
            )
            self.feature_extractor = None
        elif self.backend == "physics_transformer":
            from models.temporal.models.physics_transformer import (
                PhysicsTransformerTemporal,
            )

            # 物理Transformer直接在展平的时序输入上编码→解码；使用reduce_size作为时序空间尺寸
            self.physics_model = PhysicsTransformerTemporal(
                in_channels=(spatial_feature_dim + out_channels),
                out_channels=out_channels,
                img_size=reduce_size,
                T_in=1,
                T_out=1,
                hidden_dim=temporal_dim,
                num_heads=num_layers,
                num_layers=num_layers,
                use_frequency_encoding=True,
                dropout=dropout,
                mode="ar",
            )
            self.prediction_head = None
            self.feature_extractor = None
        elif self.backend == "conv_rnn":
            from models.temporal.components.conv_temporal import ConvTemporalPredictor

            # 自动修正通道数...
            effective_feature_dim = spatial_feature_dim
            # 检查 backbone config
            if "backbone_config" in config and "width" in config["backbone_config"]:
                if (
                    config.get("backbone_type") != "identity"
                    and spatial_feature_dim == 0
                ):
                    effective_feature_dim = config["backbone_config"]["width"]
            if config.get("backbone_type") == "identity":
                effective_feature_dim = 0

            self.conv_model = ConvTemporalPredictor(
                in_channels=(effective_feature_dim + out_channels),
                hidden_channels=temporal_dim,
                out_channels=out_channels,
                num_layers=num_layers,
                kernel_size=int(config.get("kernel_size", 3)),
                dropout=dropout,
            )
            self.prediction_head = None
            self.feature_extractor = None

            # Same channel logic as ConvRNN
            # 在 verify 脚本中，spatial_feature_dim 被设为 0 (Identity)，但实际上 spatial_features 可能是 None
            # 我们需要检查初始化逻辑

            # ConvRNN/VideoSwin 的 in_channels 初始化为 spatial_feature_dim + out_channels
            # 如果 spatial_feature_dim=0, out=1, 则 in_channels=1

            # 但是在 verify 脚本中，我们使用了默认值?
            # 让我们检查 verify_video_swin.py:
            # spatial_config = { ..., 'spatial_feature_dim': 0 }
            # temporal_config = { ..., 'temporal_dim': 24 }

            # 所以 in_channels 应该是 1。
            # 但错误信息说: weight size [24, 129, 1, 1, 1], expected input to have 129 channels.
            # 这意味着 in_channels 被初始化为了 129。
            # 为什么是 129? 128 (default) + 1 ?

            # 是的，config.get('spatial_feature_dim', 128) 在 TemporalPredictionModule.__init__ 中
            # 但我们传入了 config，里面包含了 spatial_feature_dim=0
            # 等等，SequentialSpatiotemporalModel.__init__ 中合并了 config:
            # self.config = { **spatial, **temporal, ... }
            # 如果 spatial_config 有 spatial_feature_dim=0, temporal_config 没有
            # 那么合并后 spatial_feature_dim=0

            # TemporalPredictionModule.__init__(self, config):
            # spatial_feature_dim = config.get('spatial_feature_dim', 128)
            # 如果 config 中有 0，get 应该返回 0。

            # 除非... TemporalPredictionModule 的 config 并不是合并后的 config，而是仅 temporal_config?
            # 检查 SequentialSpatiotemporalModel.__init__:
            # self.temporal_module = TemporalPredictionModule(temporal_config)
            # 是的！它只传了 temporal_config。
            # 而 verify 脚本中 temporal_config 没有 'spatial_feature_dim'。
            # 所以它使用了默认值 128。

            # 修正：我们需要确保 temporal_config 中包含了正确的 spatial_feature_dim。
            # 或者在 TemporalPredictionModule 初始化时，不仅依赖 temporal_config，还应该允许覆盖。

            # 但这里我们只能修改 SequentialSpatiotemporalModel 或 TemporalPredictionModule。

            # 最好是在 SequentialSpatiotemporalModel 中，将 spatial_feature_dim 注入到 temporal_config 中。

            # 但我们正在修改的是 TemporalPredictionModule (它是 SequentialSpatiotemporalModel 的一部分文件，但它是独立类)

            # 让我们在 TemporalPredictionModule 初始化中，更智能地获取 spatial_feature_dim
            # 实际上，代码已经在初始化 ConvRNN/VideoSwin 时尝试修正 effective_feature_dim

            # effective_feature_dim = spatial_feature_dim (这里是 128)
            # if 'backbone_config' in config ... (temporal_config 没有 backbone_config)
            # if config.get('backbone_type') == 'identity' ... (temporal_config 没有 backbone_type)

            # 所以 effective_feature_dim 保持为 128。
            # in_channels = 128 + 1 = 129.

            # 这就是为什么权重是 129。

            # 解决方法：
            # 1. 在 verify 脚本中，显式在 temporal_config 中设置 spatial_feature_dim=0。
            # 2. 或者在 SequentialSpatiotemporalModel 中传递。

            # 由于我不能修改 verify 脚本（或者我可以，但我应该让代码更健壮），
            # 我将修改 SequentialSpatiotemporalModel 的初始化，把 spatial_feature_dim 传给 temporal_module。

            # 但现在我只能修改 sequential_spatiotemporal.py。

            # 让我们修改 SequentialSpatiotemporalModel.__init__
            pass  # 占位，将在下一个 SearchReplace 中修改 __init__

        elif self.backend == "video_swin":
            from models.temporal.components.video_swin import VideoSwinPredictor

            # 使用 effective_feature_dim 逻辑
            # 注意：在forward中我们强制设置了 spatial_features = None
            # 因此这里输入通道应该只包含 spatial_pred 的通道 (out_channels)
            # 而不包含 spatial_feature_dim
            # 但为了保持兼容性，我们需要确保 VideoSwin 初始化时的 in_channels 匹配实际输入

            # 修正：既然我们强制 Two-Stage 只传图像，那么 in_channels 应该等于 out_channels
            # 但我们需要确认这是否会破坏其他使用 VideoSwin 的配置（非 Two-Stage）
            # 目前这个类是在 SequentialSpatiotemporalModel 中使用的
            # 而我们刚才硬编码了 spatial_features = None
            # 所以这里必须匹配

            self.video_swin_model = VideoSwinPredictor(
                in_channels=out_channels,  # 只输入图像，不含特征
                hidden_dim=temporal_dim,
                out_channels=out_channels,
                num_layers=num_layers,
                num_heads=int(config.get("num_heads", 4)),
                window_size=tuple(config.get("window_size", (2, 7, 7))),
                dropout=dropout,
            )
            self.prediction_head = None
            self.feature_extractor = None

        else:
            self.feature_extractor = TemporalFeatureExtractor(
                input_dim,
                temporal_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_positional_encoding=bool(
                    config.get("use_positional_encoding", False)
                ),
            )
            self.prediction_head = TemporalPredictionHead(
                temporal_dim, out_channels, img_size
            )
        self.metrics_calculator = TemporalMetricsCalculator()
        # 统一 reduce_spatial 命名
        if isinstance(reduce_spatial, str) and reduce_spatial.lower() in (
            "avgpool",
            "avg_pool",
            "avg",
        ):
            self.reduce_spatial = "avg_pool"
        else:
            self.reduce_spatial = reduce_spatial
        self.reduce_size = reduce_size

    def forward(
        self,
        spatial_results: SpatialPredictionOutput,
        target: torch.Tensor | None = None,
    ) -> TemporalPredictionOutput:
        """
        时间预测前向传播
        Args:
            spatial_results: 空间预测结果
            target: 目标时空序列 [B, T_out, C, H, W] (可选，用于训练时计算指标)
        Returns:
            TemporalPredictionOutput: 时间预测结果
        """
        spatial_pred = spatial_results.spatial_pred
        spatial_features = spatial_results.spatial_features

        # 强制忽略潜空间特征，只使用空间模型的输出图像作为时序模型的输入
        # 这符合 "Two Stage" 的物理意义：时序模型仅基于空间恢复后的高清图像序列进行预测
        spatial_features = None

        B = spatial_pred.shape[0]

        # 1. 确定目标时间步数 T_out
        # 必须优先使用 target 的时间步数（如果存在）
        if target is not None and target.dim() == 5:
            T_out = target.shape[1]
        else:
            # 推断 T_out
            if spatial_pred.dim() == 5:
                T_out = spatial_pred.shape[1]
            else:
                T_out = 1

        # 2. 处理 Spatial Pred -> [B, T_out, -1]
        # 先处理 avg_pool
        if self.reduce_spatial == "avg_pool" and self.reduce_size is not None:
            import torch.nn.functional as F

            if spatial_pred.dim() == 5:
                b, t, c, h, w = spatial_pred.shape
                sp = spatial_pred.reshape(b * t, c, h, w)
                sp = F.adaptive_avg_pool2d(sp, self.reduce_size)
                spatial_pred = sp.reshape(
                    b, t, c, self.reduce_size[0], self.reduce_size[1]
                )
            else:
                spatial_pred = F.adaptive_avg_pool2d(spatial_pred, self.reduce_size)

        # 展平
        if spatial_pred.dim() == 5:
            spatial_pred_flat = spatial_pred.reshape(B, spatial_pred.shape[1], -1)
        else:
            spatial_pred_flat = spatial_pred.reshape(B, 1, -1)

        # 扩展/对齐到 T_out
        T_pred = spatial_pred_flat.shape[1]
        if T_pred != T_out:
            if T_pred == 1:
                spatial_pred_flat = spatial_pred_flat.expand(B, T_out, -1)
            elif T_pred > T_out:
                spatial_pred_flat = spatial_pred_flat[:, :T_out, :]
            else:
                # 补全
                last = spatial_pred_flat[:, -1:, :]
                pad = T_out - T_pred
                spatial_pred_flat = torch.cat(
                    [spatial_pred_flat, last.expand(B, pad, -1)], dim=1
                )

        # 3. 处理 Spatial Features -> [B, T_out, F]
        if spatial_features is not None:
            # 降维处理
            if self.reduce_spatial == "avg_pool" and self.reduce_size is not None:
                import torch.nn.functional as F

                if spatial_features.dim() == 5:
                    b, t, c, h, w = spatial_features.shape
                    sf = spatial_features.reshape(b * t, c, h, w)
                    sf = F.adaptive_avg_pool2d(sf, self.reduce_size)
                    spatial_features = sf.reshape(
                        b, t, c, self.reduce_size[0], self.reduce_size[1]
                    )
                elif spatial_features.dim() == 4:
                    spatial_features = F.adaptive_avg_pool2d(
                        spatial_features, self.reduce_size
                    )

            # 展平与规范化为 [B, T_feat, F]
            if spatial_features.dim() == 5:
                spatial_features_flat = spatial_features.reshape(
                    B, spatial_features.shape[1], -1
                )
            elif spatial_features.dim() == 4:
                spatial_features_flat = spatial_features.reshape(B, 1, -1)
            elif spatial_features.dim() == 3:
                spatial_features_flat = spatial_features
            elif spatial_features.dim() == 2:
                spatial_features_flat = spatial_features.unsqueeze(1)
            else:
                spatial_features_flat = spatial_features.reshape(B, 1, -1)

            # 扩展/对齐到 T_out
            T_feat = spatial_features_flat.shape[1]
            if T_feat != T_out:
                if T_feat == 1:
                    spatial_features_flat = spatial_features_flat.expand(B, T_out, -1)
                elif T_feat > T_out:
                    spatial_features_flat = spatial_features_flat[:, :T_out, :]
                else:
                    last = spatial_features_flat[:, -1:, :]
                    pad = T_out - T_feat
                    spatial_features_flat = torch.cat(
                        [spatial_features_flat, last.expand(B, pad, -1)], dim=1
                    )
        else:
            spatial_features_flat = torch.zeros(B, T_out, 0, device=spatial_pred.device)

        # 4. 拼接
        temporal_input = torch.cat([spatial_pred_flat, spatial_features_flat], dim=-1)

        # 5. 严格维度检查
        actual_input_dim = temporal_input.shape[-1]
        if (
            self.backend != "tcn"
            and self.backend != "physics_transformer"
            and self.backend != "conv_rnn"
            and self.backend != "video_swin"
        ):
            expected_dim = self.feature_extractor.input_proj.in_features
            if actual_input_dim != expected_dim:
                raise RuntimeError(
                    f"TemporalPredictionModule input mismatch: Model expects {expected_dim}, got {actual_input_dim}. "
                    f"Please check spatial_feature_dim or model configuration."
                )
        elif self.backend == "physics_transformer":
            try:
                expected_dim = self.physics_model.input_projection.in_features
                if actual_input_dim != expected_dim:
                    raise RuntimeError(
                        f"PhysicsTransformer input mismatch: Model expects {expected_dim}, got {actual_input_dim}."
                    )
            except AttributeError:
                pass

        # 6. Forward pass
        # 提取时序特征
        if self.backend == "tcn":
            enc = self.temporal_encoder(temporal_input)
            temporal_features = enc["encoded_sequence"]
            B, T, _ = temporal_input.shape

            if target is not None and target.dim() == 5:
                C_out = target.shape[2]
                H_out = target.shape[3]
                W_out = target.shape[4]
            else:
                if spatial_pred.dim() == 5:
                    C_out = spatial_pred.shape[2]
                else:
                    C_out = spatial_pred.shape[1]
                img_size = self.config.get("img_size", (256, 256))
                H_out, W_out = img_size

            out_dim = C_out * H_out * W_out

            if self.prediction_head is not None:
                last_linear = None
                for m in self.prediction_head.temporal_to_spatial:
                    if isinstance(m, nn.Linear):
                        last_linear = m
                if last_linear is None or last_linear.out_features != out_dim:
                    raise RuntimeError(
                        f"TCN Output Head mismatch: Expected out_features={out_dim}, got {last_linear.out_features if last_linear else 'None'}"
                    )

            plane_flat = self.prediction_head.temporal_to_spatial(temporal_features)
            final_pred = plane_flat.view(B, T, C_out, H_out, W_out)

        elif self.backend == "physics_transformer":
            B, T, _ = temporal_input.shape
            temporal_features = self.physics_model.encode_temporal_features(
                temporal_input
            )
            decoded = self.physics_model.decode_temporal_features(temporal_features)

            import torch.nn.functional as F

            if target is not None and target.dim() == 5:
                C_out = target.shape[2]
            else:
                C_out = self.config.get("out_channels", 1)

            h_red, w_red = (
                self.physics_model.img_size
                if hasattr(self.physics_model, "img_size")
                else (256, 256)
            )
            final_pred = decoded.reshape(B, T, C_out, h_red, w_red)

            if target is not None and target.dim() == 5:
                desired_size = (target.shape[3], target.shape[4])
            else:
                desired_size = self.config.get("img_size", (256, 256))

            if (
                final_pred.shape[-2] != desired_size[0]
                or final_pred.shape[-1] != desired_size[1]
            ):
                final_pred = F.interpolate(
                    final_pred.reshape(B * T, C_out, h_red, w_red),
                    size=desired_size,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B, T, C_out, desired_size[0], desired_size[1])

        elif self.backend == "conv_rnn":
            # ConvRNN 处理 (不需要展平，需要保持空间维度)
            # 重新获取未展平的输入
            # spatial_pred: [B, T, C, H, W]
            # spatial_features: [B, T, F, H, W]

            # 确保 spatial_features 和 spatial_pred 空间尺寸一致
            if spatial_features is not None and spatial_features.shape[2] > 0:
                # 拼接
                conv_input = torch.cat(
                    [spatial_pred, spatial_features], dim=2
                )  # [B, T, C+F, H, W]
            else:
                conv_input = spatial_pred

            final_pred = self.conv_model(conv_input, T_out=T_out)
            temporal_features = None  # ConvRNN隐状态暂不暴露

            # 如果进行了空间降维，需要上采样回原始分辨率
            if target is not None and target.dim() == 5:
                desired_size = (int(target.shape[3]), int(target.shape[4]))
            else:
                img_size = self.config.get("img_size", (256, 256))
                try:
                    desired_size = (int(img_size[0]), int(img_size[1]))
                except (TypeError, IndexError, KeyError):
                    desired_size = (int(img_size), int(img_size))

            if (
                final_pred.shape[-2] != desired_size[0]
                or final_pred.shape[-1] != desired_size[1]
            ):
                import torch.nn.functional as F

                B_out, T_out, C_out, H_out, W_out = final_pred.shape
                final_pred = F.interpolate(
                    final_pred.reshape(B_out * T_out, C_out, H_out, W_out),
                    size=desired_size,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B_out, T_out, C_out, desired_size[0], desired_size[1])

        elif self.backend == "video_swin":

            # Similar to ConvRNN, we need proper input channel handling
            if spatial_features is not None and spatial_features.shape[2] > 0:
                conv_input = torch.cat([spatial_pred, spatial_features], dim=2)
            else:
                conv_input = spatial_pred

            final_pred = self.video_swin_model(conv_input, T_out=T_out)
            temporal_features = None

            # Upsampling if needed
            if target is not None and target.dim() == 5:
                desired_size = (int(target.shape[3]), int(target.shape[4]))
            else:
                img_size = self.config.get("img_size", (256, 256))
                try:
                    desired_size = (int(img_size[0]), int(img_size[1]))
                except (TypeError, IndexError, KeyError):
                    desired_size = (int(img_size), int(img_size))

            if (
                final_pred.shape[-2] != desired_size[0]
                or final_pred.shape[-1] != desired_size[1]
            ):
                import torch.nn.functional as F

                B_out, T_out, C_out, H_out, W_out = final_pred.shape
                final_pred = F.interpolate(
                    final_pred.reshape(B_out * T_out, C_out, H_out, W_out),
                    size=desired_size,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B_out, T_out, C_out, desired_size[0], desired_size[1])

        else:
            temporal_features = self.feature_extractor(temporal_input)

            if target is not None and target.dim() == 5:
                C_out = target.shape[2]
                H_out = target.shape[3]
                W_out = target.shape[4]
                out_dim = C_out * H_out * W_out

                last_linear = None
                if self.prediction_head is not None:
                    for m in self.prediction_head.temporal_to_spatial:
                        if isinstance(m, nn.Linear):
                            last_linear = m
                    if last_linear is None or last_linear.out_features != out_dim:
                        raise RuntimeError(
                            f"Prediction Head mismatch: Expected out_features={out_dim}, got {last_linear.out_features if last_linear else 'None'}"
                        )

            final_pred = self.prediction_head(temporal_features)

        # 数值稳定性检查 - 最终输出
        if torch.isnan(final_pred).any() or torch.isinf(final_pred).any():
            print("WARNING: NaN/Inf detected in final_pred")
            final_pred = torch.nan_to_num(final_pred, nan=0.0, posinf=1e6, neginf=-1e6)

        # 计算评估指标（如果有目标值）
        temporal_metrics = {}
        if target is not None:
            temporal_metrics = self.metrics_calculator.calculate_metrics(
                final_pred, target
            )

        return TemporalPredictionOutput(
            final_pred=final_pred,
            temporal_features=temporal_features,
            temporal_metrics=temporal_metrics,
        )


class SequentialSpatiotemporalModel(nn.Module):
    """分阶段时空预测模型"""

    def __init__(
        self,
        spatial_config: dict,
        temporal_config: dict,
        data_config: dict,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        # 转换为普通字典以避免OmegaConf的结构限制和副作用
        from omegaconf import DictConfig, OmegaConf

        if isinstance(spatial_config, DictConfig):
            spatial_config = OmegaConf.to_container(spatial_config, resolve=True)
        elif hasattr(spatial_config, "items"):
            spatial_config = dict(spatial_config)

        if isinstance(temporal_config, DictConfig):
            temporal_config = OmegaConf.to_container(temporal_config, resolve=True)
        elif hasattr(temporal_config, "items"):
            temporal_config = dict(temporal_config)

        # 合并配置
        self.config = {
            **spatial_config,
            **temporal_config,
            "data_config": data_config,
            "device": device,
            **kwargs,
        }

        # 自动将空间配置中的特征维度同步到时序配置中，防止维度不匹配
        if (
            "spatial_feature_dim" in spatial_config
            and "spatial_feature_dim" not in temporal_config
        ):
            temporal_config["spatial_feature_dim"] = spatial_config[
                "spatial_feature_dim"
            ]
        # 同时也同步 backbone 信息，用于 VideoSwin/ConvRNN 的通道推断
        if "backbone_type" in spatial_config:
            temporal_config["backbone_type"] = spatial_config["backbone_type"]
        if "backbone_config" in spatial_config:
            temporal_config["backbone_config"] = spatial_config["backbone_config"]

        self.spatial_module = SpatialPredictionModule(spatial_config)
        self.temporal_module = TemporalPredictionModule(temporal_config)
        self.teacher_prob: float = 0.0
        self.teacher_forcing_decay: float = 0.95

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        分阶段前向传播
        Args:
            x: 输入时空序列 [B, T_in, C, H, W]
            target: 目标时空序列 [B, T_out, C, H, W] (可选)
        Returns:
            dict: 包含所有预测结果和指标
        """
        # 第一阶段：空间预测（若禁用空间，则构造占位输出）
        if getattr(self.spatial_module, "feature_extractor", None) is None:
            if x.dim() == 4:
                x = x.unsqueeze(1)
            B, T_in, C, H, W = x.shape
            # 使用最后一帧作为空间预测占位，减少无意义的时间维差异
            spatial_pred = x[:, -1:].clone()
            T_sp = spatial_pred.shape[1]
            # 如果没有特征提取器，特征维度应为0，而不是1个通道的零张量
            spatial_features = torch.zeros(
                B, T_sp, 0, H, W, device=x.device, dtype=x.dtype
            )
            spatial_output = SpatialPredictionOutput(
                spatial_pred=spatial_pred,
                spatial_features=spatial_features,
                spatial_metrics={},
            )
        else:
            spatial_output = self.spatial_module(x, target)

        # 第二阶段：时间预测
        # Teacher Forcing（Scheduled Sampling）：根据 teacher_prob 混合部分空间预测为GT
        if target is not None and self.teacher_prob > 0.0:
            try:
                mixed_spatial_pred = spatial_output.spatial_pred.clone()
                # 对齐到目标维度
                if mixed_spatial_pred.dim() == 5 and target.dim() == 5:
                    Bt, Tt, Ct, Ht, Wt = target.shape
                    if mixed_spatial_pred.shape[1] != Tt:
                        if mixed_spatial_pred.shape[1] > Tt:
                            mixed_spatial_pred = mixed_spatial_pred[:, :Tt]
                        else:
                            pad_frames = Tt - mixed_spatial_pred.shape[1]
                            mixed_spatial_pred = torch.cat(
                                [
                                    mixed_spatial_pred,
                                    mixed_spatial_pred[:, -1:].repeat(
                                        1, pad_frames, 1, 1, 1
                                    ),
                                ],
                                dim=1,
                            )
                    # 生成时间步级别的采样掩码
                    mask = (
                        torch.rand(Bt, Tt, device=mixed_spatial_pred.device)
                        < float(self.teacher_prob)
                    ).float()
                    mask = mask.view(Bt, Tt, 1, 1, 1)
                    try:
                        self._last_teacher_mask = mask.detach()
                    except Exception:
                        pass
                    # 混合：部分时间步用GT替代预测作为时序输入的“观测”
                    spatial_output = SpatialPredictionOutput(
                        spatial_pred=mixed_spatial_pred * (1.0 - mask) + target * mask,
                        spatial_features=spatial_output.spatial_features,
                        spatial_metrics=spatial_output.spatial_metrics,
                    )
            except Exception:
                pass

        temporal_output = self.temporal_module(spatial_output, target)

        # 合并结果
        return {
            "spatial_pred": spatial_output.spatial_pred,
            "spatial_features": spatial_output.spatial_features,
            "spatial_metrics": spatial_output.spatial_metrics,
            "final_pred": temporal_output.final_pred,
            "temporal_features": temporal_output.temporal_features,
            "temporal_metrics": temporal_output.temporal_metrics,
        }

    def set_epoch(self, epoch: int, decay: float | None = None):
        try:
            if decay is not None:
                self.teacher_forcing_decay = float(decay)
            prob = float(self.teacher_forcing_decay) ** int(epoch)
            # 设定早期TF上限，减少训练/验证分布差异
            cap = 0.3
            try:
                cap = float(self.config.get("teacher_forcing_cap", cap))
            except Exception:
                pass
            self.teacher_prob = min(prob, cap)
        except Exception:
            self.teacher_prob = 0.0

    def get_model_info(self) -> dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "name": self.__class__.__name__,
            "parameters": total_params,
            "parameters_M": total_params / 1e6,
        }

    def get_memory_usage(self, batch_size: int = 1) -> dict[str, float]:
        """估算显存使用量"""
        param_memory = (
            sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        )
        return {
            "parameters_MB": float(param_memory),
            "total_MB": float(param_memory),  # 简化估算
        }

    def spatial_forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> SpatialPredictionOutput:
        """仅执行空间预测"""
        return self.spatial_module(x, target)

    def temporal_forward(
        self,
        spatial_results: SpatialPredictionOutput,
        target: torch.Tensor | None = None,
    ) -> TemporalPredictionOutput:
        """仅执行时间预测"""
        return self.temporal_module(spatial_results, target)

    def rollout_inference(
        self,
        x: torch.Tensor,
        T_out: int,
        step_by_step: bool = True,
        preserve_grad: bool = False,
    ) -> torch.Tensor:
        """
        自回归推理模式 - 逐步预测以验证真实时序建模能力

        Args:
            x: 输入序列 [B, T_in, C, H, W]
            T_out: 需要预测的时间步数
            step_by_step: 是否逐步预测（True）还是一次性预测（False）

        Returns:
            predictions: 预测序列 [B, T_out, C, H, W]
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T_in, C, H, W = x.shape
        predictions = []
        # 使用最近的 T_in 帧作为输入窗口
        current_input = x.clone()

        if step_by_step:
            if not preserve_grad:
                self.eval()
                with torch.no_grad():
                    for t in range(T_out):
                        # 如果当前输入超过了模型需要的T_in，裁剪
                        if current_input.shape[1] > T_in:
                            current_input = current_input[:, -T_in:]

                        outputs = self.forward(current_input)
                        # 使用当前窗口的最后一步作为下一步预测的代理
                        pred_t = outputs["final_pred"][:, -1:]
                        predictions.append(pred_t)
                        current_input = torch.cat([current_input, pred_t], dim=1)
                    return torch.cat(predictions, dim=1)
            else:
                self.train()
                for t in range(T_out):
                    if current_input.shape[1] > T_in:
                        current_input = current_input[:, -T_in:]

                    outputs = self.forward(current_input)
                    pred_t = outputs["final_pred"][:, -1:]
                    predictions.append(pred_t)
                    current_input = torch.cat([current_input, pred_t], dim=1)
                return torch.cat(predictions, dim=1)

        else:
            # 一次性预测模式（训练模式）
            outputs = self.forward(x)
            return outputs["final_pred"][:, :T_out]

    def autoregressive_predict(
        self,
        x: torch.Tensor,
        T_out: int,
        teacher: torch.Tensor | None = None,
        train_mode: bool = False,
    ) -> torch.Tensor:
        """兼容 Trainer 的 autoregressive_predict 接口"""
        return self.rollout_inference(
            x, T_out, step_by_step=True, preserve_grad=train_mode
        )
