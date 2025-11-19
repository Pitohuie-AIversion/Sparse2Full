"""
混合精度训练优化器
基于技术方案实现自动混合精度训练，支持Tensor Core优化和动态损失缩放
"""

import os
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import time
import numpy as np
from pathlib import Path
import json

# 导入已实现的优化器
from src.optimizers.gpu_optimizer import GPUOptimizer
from src.optimizers.numa_manager import NUMAMemoryManager

logger = logging.getLogger(__name__)

@dataclass
class MixedPrecisionConfig:
    """混合精度训练配置"""
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    loss_scale: Optional[float] = None
    loss_scale_window: int = 1000
    min_loss_scale: float = 1.0
    max_loss_scale: float = 2**24
    hysteresis: int = 2
    dynamic_loss_scale: bool = True
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = None
    optimize_for_tensor_cores: bool = True
    memory_efficient: bool = True
    
class MixedPrecisionTrainer:
    """
    混合精度训练优化器
    实现自动混合精度训练，支持Tensor Core优化和动态损失缩放
    """
    
    def __init__(self, config: MixedPrecisionConfig, 
                 gpu_optimizer: Optional[GPUOptimizer] = None):
        self.config = config
        self.gpu_optimizer = gpu_optimizer or GPUOptimizer()
        self.scaler = None
        self._initialized = False
        self._training_stats = {
            "loss_scale_history": [],
            "gradient_norms": [],
            "overflow_count": 0,
            "total_steps": 0,
            "skipped_steps": 0,
        }
        
        self._initialize_scaler()
        logger.info("混合精度训练优化器初始化完成")
    
    def _initialize_scaler(self):
        """初始化梯度缩放器"""
        try:
            if self.config.enabled:
                if self.config.dynamic_loss_scale:
                    # 动态损失缩放
                    self.scaler = GradScaler(
                        init_scale=self.config.loss_scale or 2**16,
                        growth_factor=2.0,
                        backoff_factor=0.5,
                        growth_interval=self.config.loss_scale_window,
                        enabled=self.config.enabled
                    )
                    logger.info(f"动态梯度缩放器初始化完成，初始缩放: {self.scaler.get_scale()}")
                else:
                    # 静态损失缩放
                    if self.config.loss_scale is not None:
                        self.scaler = GradScaler(
                            init_scale=self.config.loss_scale,
                            growth_factor=1.0,
                            backoff_factor=1.0,
                            growth_interval=1000000,  # 很大，基本不调整
                            enabled=self.config.enabled
                        )
                        logger.info(f"静态梯度缩放器初始化完成，缩放: {self.config.loss_scale}")
                    else:
                        logger.warning("未指定静态损失缩放值，使用动态缩放")
                        self.scaler = GradScaler(enabled=self.config.enabled)
            else:
                logger.info("混合精度训练已禁用")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"梯度缩放器初始化失败: {e}")
            self.scaler = GradScaler(enabled=False)
    
    @contextmanager
    def autocast_context(self, dtype: Optional[torch.dtype] = None):
        """
        自动混合精度上下文管理器
        
        Args:
            dtype: 数据类型，如果为None则使用配置中的dtype
        """
        if not self.config.enabled:
            yield
            return
        
        try:
            target_dtype = dtype or self.config.dtype
            with autocast(dtype=target_dtype):
                yield
        except Exception as e:
            logger.error(f"自动混合精度上下文失败: {e}")
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        缩放损失值
        
        Args:
            loss: 原始损失值
            
        Returns:
            缩放后的损失值
        """
        if not self.config.enabled or self.scaler is None:
            return loss
        
        try:
            return self.scaler.scale(loss)
        except Exception as e:
            logger.error(f"损失缩放失败: {e}")
            return loss
    
    def backward(self, loss: torch.Tensor, 
                optimizer: torch.optim.Optimizer,
                retain_graph: bool = False,
                create_graph: bool = False) -> bool:
        """
        执行反向传播
        
        Args:
            loss: 损失值
            optimizer: 优化器
            retain_graph: 是否保留计算图
            create_graph: 是否创建计算图
            
        Returns:
            是否成功执行反向传播
        """
        if not self.config.enabled or self.scaler is None:
            # 标准反向传播
            try:
                loss.backward(retain_graph=retain_graph, create_graph=create_graph)
                return True
            except Exception as e:
                logger.error(f"标准反向传播失败: {e}")
                return False
        
        try:
            # 混合精度反向传播
            self.scaler.scale(loss).backward(retain_graph=retain_graph, create_graph=create_graph)
            return True
            
        except Exception as e:
            logger.error(f"混合精度反向传播失败: {e}")
            return False
    
    def step(self, optimizer: torch.optim.Optimizer,
             max_grad_norm: Optional[float] = None) -> bool:
        """
        执行优化器步骤
        
        Args:
            optimizer: 优化器
            max_grad_norm: 梯度裁剪范数
            
        Returns:
            是否成功执行优化器步骤
        """
        if not self.config.enabled or self.scaler is None:
            # 标准优化器步骤
            try:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_grad_norm)
                optimizer.step()
                return True
            except Exception as e:
                logger.error(f"标准优化器步骤失败: {e}")
                return False
        
        try:
            # 记录当前损失缩放值
            current_scale = self.scaler.get_scale()
            self._training_stats["loss_scale_history"].append(current_scale)
            self._training_stats["total_steps"] += 1
            
            # 梯度裁剪（可选）
            if max_grad_norm is not None or self.config.clip_grad_norm is not None:
                grad_norm = self._clip_gradients(optimizer, max_grad_norm or self.config.clip_grad_norm)
                self._training_stats["gradient_norms"].append(grad_norm)
            
            # 执行优化器步骤
            self.scaler.step(optimizer)
            
            # 更新损失缩放
            self.scaler.update()
            
            # 检查是否有溢出
            new_scale = self.scaler.get_scale()
            if new_scale < current_scale:
                self._training_stats["overflow_count"] += 1
                self._training_stats["skipped_steps"] += 1
                logger.debug(f"检测到梯度溢出，损失缩放从 {current_scale} 调整到 {new_scale}")
            
            return True
            
        except Exception as e:
            logger.error(f"混合精度优化器步骤失败: {e}")
            return False
    
    def _clip_gradients(self, optimizer: torch.optim.Optimizer, 
                       max_norm: float) -> float:
        """
        裁剪梯度
        
        Args:
            optimizer: 优化器
            max_norm: 最大梯度范数
            
        Returns:
            梯度范数
        """
        try:
            # 获取所有参数
            params = []
            for group in optimizer.param_groups:
                params.extend(group['params'])
            
            # 计算梯度范数
            if self.config.enabled and self.scaler is not None:
                # 使用缩放后的梯度
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, max_norm, error_if_nonfinite=True
                )
            else:
                # 使用原始梯度
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, max_norm
                )
            
            return grad_norm.item()
            
        except Exception as e:
            logger.error(f"梯度裁剪失败: {e}")
            return 0.0
    
    def optimize_model_for_mixed_precision(self, model: nn.Module) -> nn.Module:
        """
        优化模型以最大化混合精度训练效果
        
        Args:
            model: PyTorch模型
            
        Returns:
            优化后的模型
        """
        logger.info("优化模型以支持混合精度训练...")
        
        try:
            # 应用GPU优化（包括Tensor Core优化）
            if self.gpu_optimizer:
                model = self.gpu_optimizer.optimize_for_tensor_cores(model)
            
            # 确保模型参数适合混合精度
            model = self._optimize_model_parameters(model)
            
            # 优化批量归一化层
            model = self._optimize_batch_norm_for_fp16(model)
            
            logger.info("混合精度模型优化完成")
            return model
            
        except Exception as e:
            logger.error(f"混合精度模型优化失败: {e}")
            return model
    
    def _optimize_model_parameters(self, model: nn.Module) -> nn.Module:
        """优化模型参数以适应混合精度"""
        logger.info("优化模型参数...")
        
        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 确保参数是连续的
                    if not param.data.is_contiguous():
                        param.data = param.data.contiguous()
                    
                    # 如果有梯度，也确保连续性
                    if param.grad is not None and not param.grad.is_contiguous():
                        param.grad.data = param.grad.data.contiguous()
            
            logger.info("模型参数优化完成")
            return model
            
        except Exception as e:
            logger.error(f"模型参数优化失败: {e}")
            return model
    
    def _optimize_batch_norm_for_fp16(self, model: nn.Module) -> nn.Module:
        """优化批量归一化层以适应FP16"""
        logger.info("优化批量归一化层...")
        
        try:
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    # 确保批量归一化层在FP32下运行
                    module.float()
                    logger.debug(f"批量归一化层 {name} 已转换为FP32")
            
            logger.info("批量归一化层优化完成")
            return model
            
        except Exception as e:
            logger.error(f"批量归一化层优化失败: {e}")
            return model
    
    def create_optimizer_for_mixed_precision(self, model: nn.Module,
                                           optimizer_class: type = torch.optim.AdamW,
                                           **optimizer_kwargs) -> torch.optim.Optimizer:
        """
        创建适合混合精度训练的优化器
        
        Args:
            model: PyTorch模型
            optimizer_class: 优化器类
            **optimizer_kwargs: 优化器参数
            
        Returns:
            优化器实例
        """
        try:
            # 分离需要不同处理的参数
            fp16_params = []
            fp32_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 检查参数是否适合FP16
                    if self._should_use_fp16_for_param(name, param):
                        fp16_params.append(param)
                    else:
                        fp32_params.append(param)
            
            # 创建参数组
            param_groups = []
            if fp16_params:
                param_groups.append({
                    'params': fp16_params,
                    'lr': optimizer_kwargs.get('lr', 1e-3),
                    'weight_decay': optimizer_kwargs.get('weight_decay', 0.01),
                })
            
            if fp32_params:
                param_groups.append({
                    'params': fp32_params,
                    'lr': optimizer_kwargs.get('lr', 1e-3),
                    'weight_decay': optimizer_kwargs.get('weight_decay', 0.01),
                })
            
            # 创建优化器
            optimizer = optimizer_class(param_groups, **optimizer_kwargs)
            
            logger.info(f"混合精度优化器创建完成: FP16参数 {len(fp16_params)}, FP32参数 {len(fp32_params)}")
            return optimizer
            
        except Exception as e:
            logger.error(f"混合精度优化器创建失败: {e}")
            # 返回标准优化器
            return optimizer_class(model.parameters(), **optimizer_kwargs)
    
    def _should_use_fp16_for_param(self, name: str, param: torch.Tensor) -> bool:
        """判断参数是否适合使用FP16"""
        try:
            # 批量归一化参数使用FP32
            if 'bn' in name.lower() or 'batch_norm' in name.lower():
                return False
            
            # 小的参数使用FP32
            if param.numel() < 1000:
                return False
            
            # 嵌入层参数使用FP32
            if 'embed' in name.lower():
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"参数FP16判断失败 {name}: {e}")
            return True
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = {
            "config": {
                "enabled": self.config.enabled,
                "dtype": str(self.config.dtype),
                "dynamic_loss_scale": self.config.dynamic_loss_scale,
                "loss_scale": self.config.loss_scale,
            },
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "training_stats": self._training_stats,
        }
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        try:
            if self.scaler and state_dict.get("scaler_state"):
                self.scaler.load_state_dict(state_dict["scaler_state"])
            
            if "training_stats" in state_dict:
                self._training_stats.update(state_dict["training_stats"])
            
            logger.info("混合精度训练状态加载完成")
            
        except Exception as e:
            logger.error(f"状态加载失败: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        stats = self._training_stats.copy()
        
        # 计算额外的统计信息
        if stats["total_steps"] > 0:
            stats["overflow_rate"] = stats["overflow_count"] / stats["total_steps"]
            stats["skip_rate"] = stats["skipped_steps"] / stats["total_steps"]
        else:
            stats["overflow_rate"] = 0.0
            stats["skip_rate"] = 0.0
        
        # 当前损失缩放值
        if self.scaler:
            stats["current_loss_scale"] = self.scaler.get_scale()
        else:
            stats["current_loss_scale"] = 1.0
        
        return stats
    
    def export_stats(self, filepath: Union[str, Path]):
        """导出训练统计信息"""
        try:
            stats = self.get_training_stats()
            
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"训练统计信息已导出到: {filepath}")
            
        except Exception as e:
            logger.error(f"统计信息导出失败: {e}")
    
    def reset_stats(self):
        """重置训练统计信息"""
        self._training_stats = {
            "loss_scale_history": [],
            "gradient_norms": [],
            "overflow_count": 0,
            "total_steps": 0,
            "skipped_steps": 0,
        }
        logger.info("训练统计信息已重置")
    
    def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理混合精度训练资源...")
            
            if self.scaler:
                # 清空梯度缩放器状态
                self.scaler._per_optimizer_states.clear()
            
            logger.info("混合精度训练资源清理完成")
            
        except Exception as e:
            logger.error(f"清理失败: {e}")

# 全局混合精度训练器实例
_mixed_precision_trainer = None

def get_mixed_precision_trainer(config: Optional[MixedPrecisionConfig] = None,
                              gpu_optimizer: Optional[GPUOptimizer] = None) -> MixedPrecisionTrainer:
    """获取全局混合精度训练器实例"""
    global _mixed_precision_trainer
    
    if _mixed_precision_trainer is None:
        if config is None:
            config = MixedPrecisionConfig()
        _mixed_precision_trainer = MixedPrecisionTrainer(config, gpu_optimizer)
    
    return _mixed_precision_trainer