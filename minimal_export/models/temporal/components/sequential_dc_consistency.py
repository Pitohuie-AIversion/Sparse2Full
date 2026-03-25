"""
分阶段预测的数据一致性模块
确保观测算子H和训练DC的算子一致性，遵循黄金法则
"""

import torch
import torch.nn as nn

from ops.degradation import (
    CropOperator,
    SuperResolutionOperator,
    apply_degradation_operator,
    verify_degradation_consistency,
)


class SequentialDCConsistency(nn.Module):
    """
    分阶段预测的数据一致性模块
    确保空间预测和时间预测阶段使用一致的观测算子
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # 从配置中读取H算子参数
        self.h_params = config.get("h_params", {})
        self.consistency_tolerance = config.get("consistency_tolerance", 1e-8)

        # 初始化H算子
        self._init_h_operators()

    def _init_h_operators(self):
        """初始化H算子"""
        task = self.h_params.get("task", "sr")

        if task == "sr":
            scale = self.h_params.get("scale", 2)
            sigma = self.h_params.get("sigma", 1.0)
            kernel_size = self.h_params.get("kernel_size", 5)
            boundary = self.h_params.get("boundary", "mirror")

            self.h_operator = SuperResolutionOperator(
                scale=scale, sigma=sigma, kernel_size=kernel_size, boundary=boundary
            )

        elif task == "crop":
            crop_size = self.h_params.get("crop_size", (128, 128))
            crop_box = self.h_params.get("crop_box", None)
            boundary = self.h_params.get("boundary", "mirror")

            self.h_operator = CropOperator(
                crop_size=crop_size, crop_box=crop_box, boundary=boundary
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def apply_h_operator(
        self, x: torch.Tensor, params: dict | None = None
    ) -> torch.Tensor:
        """
        应用H算子到输入数据

        Args:
            x: 输入张量 [B, C, H, W] 或 [B, T, C, H, W]
            params: 可选的H算子参数

        Returns:
            应用H算子后的张量
        """
        # 处理时空输入
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # 重塑为2D进行处理
            x_2d = x.view(B * T, C, H, W)

            # 应用H算子
            if params is not None:
                result_2d = apply_degradation_operator(x_2d, params)
            else:
                result_2d = self.h_operator(x_2d)

            # 重塑回时空格式
            _, C_out, H_out, W_out = result_2d.shape
            result = result_2d.view(B, T, C_out, H_out, W_out)

        else:
            # 2D输入直接处理
            if params is not None:
                result = apply_degradation_operator(x, params)
            else:
                result = self.h_operator(x)

        return result

    def verify_spatial_consistency(
        self, spatial_pred: torch.Tensor, observation: torch.Tensor
    ) -> dict[str, float]:
        """
        验证空间预测的一致性

        Args:
            spatial_pred: 空间预测结果 [B, T_out, C, H, W]
            observation: 观测数据 [B, T_out, C, H_obs, W_obs]

        Returns:
            一致性验证结果
        """
        # 应用H算子到空间预测
        h_spatial_pred = self.apply_h_operator(spatial_pred)

        # 验证一致性
        consistency_result = verify_degradation_consistency(
            spatial_pred, observation, self.h_params, self.consistency_tolerance
        )

        return consistency_result

    def verify_temporal_consistency(
        self, temporal_pred: torch.Tensor, observation: torch.Tensor
    ) -> dict[str, float]:
        """
        验证时间预测的一致性

        Args:
            temporal_pred: 时间预测结果 [B, T_out, C, H, W]
            observation: 观测数据 [B, T_out, C, H_obs, W_obs]

        Returns:
            一致性验证结果
        """
        # 应用H算子到时间预测
        h_temporal_pred = self.apply_h_operator(temporal_pred)

        # 验证一致性
        consistency_result = verify_degradation_consistency(
            temporal_pred, observation, self.h_params, self.consistency_tolerance
        )

        return consistency_result

    def compute_dc_loss(
        self, pred: torch.Tensor, observation: torch.Tensor
    ) -> torch.Tensor:
        """
        计算数据一致性损失

        Args:
            pred: 预测结果 [B, T, C, H, W]
            observation: 观测数据 [B, T, C, H_obs, W_obs]

        Returns:
            DC损失标量
        """
        # 应用H算子到预测结果
        h_pred = self.apply_h_operator(pred)

        # 确保尺寸匹配
        if h_pred.shape != observation.shape:
            # 调整观测数据尺寸以匹配H算子输出
            if h_pred.numel() == observation.numel():
                observation = observation.view(h_pred.shape)
            else:
                # 使用插值调整尺寸
                observation = torch.nn.functional.interpolate(
                    observation.view(-1, *observation.shape[-3:]),
                    size=h_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).view(h_pred.shape)

        # 计算MSE损失
        dc_loss = torch.nn.functional.mse_loss(h_pred, observation)

        return dc_loss


class SequentialConsistencyChecker:
    """
    分阶段一致性检查器
    确保两阶段预测的一致性
    """

    def __init__(self, config: dict):
        self.config = config
        self.dc_consistency = SequentialDCConsistency(config)

    def check_stage_consistency(
        self, spatial_output, temporal_output, observation
    ) -> dict[str, dict[str, float]]:
        """
        检查两阶段的一致性

        Args:
            spatial_output: 空间预测输出
            temporal_output: 时间预测输出
            observation: 观测数据

        Returns:
            一致性检查结果
        """
        results = {}

        # 检查空间预测一致性
        if hasattr(spatial_output, "spatial_pred"):
            results["spatial_consistency"] = (
                self.dc_consistency.verify_spatial_consistency(
                    spatial_output.spatial_pred, observation
                )
            )

        # 检查时间预测一致性
        if hasattr(temporal_output, "final_pred"):
            results["temporal_consistency"] = (
                self.dc_consistency.verify_temporal_consistency(
                    temporal_output.final_pred, observation
                )
            )

        # 检查两阶段间的一致性
        if hasattr(spatial_output, "spatial_pred") and hasattr(
            temporal_output, "final_pred"
        ):
            results["stage_transition"] = self._check_stage_transition(
                spatial_output.spatial_pred, temporal_output.final_pred
            )

        return results

    def _check_stage_transition(
        self, spatial_pred: torch.Tensor, temporal_pred: torch.Tensor
    ) -> dict[str, float]:
        """
        检查两阶段间的转换一致性

        Args:
            spatial_pred: 空间预测结果
            temporal_pred: 时间预测结果

        Returns:
            转换一致性结果
        """
        with torch.no_grad():
            # 计算两阶段预测的差异
            pred_diff = torch.abs(temporal_pred - spatial_pred)

            # 计算相对变化
            relative_change = torch.norm(pred_diff) / torch.norm(spatial_pred)

            # 计算最大差异
            max_diff = torch.max(pred_diff)

            return {
                "relative_change": relative_change.item(),
                "max_difference": max_diff.item(),
                "mean_difference": torch.mean(pred_diff).item(),
            }

    def compute_total_consistency_loss(
        self,
        spatial_pred: torch.Tensor,
        temporal_pred: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算总的一致性损失

        Args:
            spatial_pred: 空间预测结果
            temporal_pred: 时间预测结果
            observation: 观测数据

        Returns:
            总一致性损失
        """
        # 空间预测DC损失
        spatial_dc_loss = self.dc_consistency.compute_dc_loss(spatial_pred, observation)

        # 时间预测DC损失
        temporal_dc_loss = self.dc_consistency.compute_dc_loss(
            temporal_pred, observation
        )

        # 两阶段一致性损失
        stage_consistency_loss = torch.nn.functional.mse_loss(
            temporal_pred, spatial_pred
        )

        # 总损失
        total_loss = spatial_dc_loss + temporal_dc_loss + 0.5 * stage_consistency_loss

        return total_loss
