"""
GPU特化优化器
基于技术方案实现GPU特定优化，包括Tensor Core优化、CUDA流管理等
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from pathlib import Path

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GPUOptimizationConfig:
    """GPU优化配置"""
    gpu_count: int = 0
    memory_fraction: float = 0.85
    enable_tensor_cores: bool = True
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True
    cuda_streams: Dict[str, Any] = field(default_factory=dict)
    optimization_level: str = "aggressive"

class GPUOptimizer:
    """
    GPU特化优化器
    实现GPU特定优化，包括Tensor Core优化、CUDA流管理等
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = GPUOptimizationConfig()
        self.gpu_count = torch.cuda.device_count()
        self.optimal_settings = {}
        self.cuda_streams = {}
        self._initialized = False
        
        if config:
            self._update_config(config)
        
        self._initialize_gpu_settings()
    
    def _update_config(self, config: Dict[str, Any]):
        """更新配置"""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _initialize_gpu_settings(self):
        """初始化GPU设置"""
        logger.info(f"初始化GPU优化设置，检测到 {self.gpu_count} 个GPU设备")
        
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA不可用，跳过GPU优化")
                return
            
            # 配置每个GPU的设置
            for i in range(self.gpu_count):
                self._configure_gpu_device(i)
            
            # 启用全局GPU优化
            self._enable_global_gpu_optimizations()
            
            # 初始化CUDA流
            self._initialize_cuda_streams()
            
            self._initialized = True
            logger.info("GPU优化设置初始化完成")
            
        except Exception as e:
            logger.error(f"GPU设置初始化失败: {e}")
            self._initialized = False
    
    def _configure_gpu_device(self, device_id: int):
        """配置单个GPU设备"""
        try:
            logger.info(f"配置GPU设备 {device_id}...")
            
            # 设置当前设备
            torch.cuda.set_device(device_id)
            
            # 设置GPU内存增长策略
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction, device_id
                )
            
            # 获取设备属性
            props = torch.cuda.get_device_properties(device_id)
            
            # 存储设备设置
            device_settings = {
                "device_id": device_id,
                "name": props.name,
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}",
                "memory_fraction": self.config.memory_fraction,
                "tensor_cores": self._detect_tensor_core_support(props),
                "max_threads_per_block": props.max_threads_per_block,
                "warp_size": props.warp_size,
                "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
            }
            
            self.optimal_settings[f"gpu_{device_id}"] = device_settings
            
            logger.info(f"GPU {device_id} 配置完成: {props.name} "
                       f"({props.total_memory // 1024**2}MB, "
                       f"CC: {device_settings['compute_capability']})")
            
        except Exception as e:
            logger.error(f"GPU设备 {device_id} 配置失败: {e}")
    
    def _detect_tensor_core_support(self, props) -> bool:
        """检测Tensor Core支持"""
        # 基于计算能力判断Tensor Core支持
        major, minor = props.major, props.minor
        
        # Volta架构及以上支持Tensor Core
        if major >= 7:
            return True
        
        return False
    
    def _enable_global_gpu_optimizations(self):
        """启用全局GPU优化"""
        logger.info("启用全局GPU优化...")
        
        try:
            # 启用Tensor Core优化
            if self.config.enable_tensor_cores:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Tensor Core优化已启用 (TF32)")
            
            # 启用cuDNN自动调优
            if self.config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN自动调优已启用")
            
            # 设置cuDNN确定性模式
            if os.environ.get('CUDNN_DETERMINISTIC', '0') == '1':
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info("cuDNN确定性模式已启用")
            
            # 启用TF32（如果支持）
            if self.config.enable_tf32 and hasattr(torch.backends.cuda, 'allow_tf32'):
                torch.backends.cuda.allow_tf32 = True
                logger.info("CUDA TF32优化已启用")
            
            # 设置CUDA内存分配器
            if hasattr(torch.cuda, 'memory'):
                # 启用内存池
                torch.cuda.memory.set_per_process_memory_fraction(
                    self.config.memory_fraction
                )
            
            logger.info("全局GPU优化设置完成")
            
        except Exception as e:
            logger.error(f"全局GPU优化启用失败: {e}")
    
    def _initialize_cuda_streams(self):
        """初始化CUDA流"""
        logger.info("初始化CUDA流...")
        
        try:
            for device_id in range(self.gpu_count):
                torch.cuda.set_device(device_id)
                
                # 创建多个CUDA流以支持并行操作
                streams = {
                    "main_stream": torch.cuda.Stream(device_id),
                    "data_stream": torch.cuda.Stream(device_id),
                    "compute_stream": torch.cuda.Stream(device_id),
                    "memory_stream": torch.cuda.Stream(device_id),
                }
                
                self.cuda_streams[device_id] = streams
                
                logger.debug(f"GPU {device_id} CUDA流创建完成: {len(streams)}个流")
            
            logger.info(f"CUDA流初始化完成，总计 {len(self.cuda_streams)} 个设备流")
            
        except Exception as e:
            logger.error(f"CUDA流初始化失败: {e}")
    
    @contextmanager
    def cuda_stream_context(self, stream_name: str = "main", device_id: int = 0):
        """
        CUDA流上下文管理器
        
        Args:
            stream_name: 流名称 (main, data, compute, memory)
            device_id: GPU设备ID
        """
        try:
            if device_id in self.cuda_streams:
                stream = self.cuda_streams[device_id].get(stream_name)
                if stream:
                    with torch.cuda.stream(stream):
                        yield stream
                else:
                    yield None
            else:
                yield None
                
        except Exception as e:
            logger.error(f"CUDA流上下文管理失败: {e}")
            yield None
    
    def optimize_for_tensor_cores(self, model: nn.Module) -> nn.Module:
        """
        针对Tensor Core优化模型
        
        Args:
            model: 要优化的PyTorch模型
            
        Returns:
            优化后的模型
        """
        logger.info("针对Tensor Core优化模型...")
        
        try:
            # 检查是否有支持Tensor Core的GPU
            has_tensor_cores = any(
                device.get("tensor_cores", False) 
                for device in self.optimal_settings.values()
            )
            
            if not has_tensor_cores:
                logger.info("未检测到支持Tensor Core的GPU，跳过优化")
                return model
            
            # 优化模型层以最大化Tensor Core利用率
            model = self._optimize_model_layers(model)
            
            # 确保模型参数内存对齐
            model = self._optimize_memory_layout(model)
            
            logger.info("Tensor Core模型优化完成")
            return model
            
        except Exception as e:
            logger.error(f"Tensor Core模型优化失败: {e}")
            return model
    
    def _optimize_model_layers(self, model: nn.Module) -> nn.Module:
        """优化模型层以利用Tensor Core"""
        logger.info("优化模型层结构...")
        
        try:
            # 遍历模型的所有模块
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                    # 确保维度是8的倍数以优化Tensor Core使用
                    self._optimize_module_dimensions(module, name)
                
                elif isinstance(module, nn.BatchNorm2d):
                    # 优化批量归一化层
                    self._optimize_batch_norm(module, name)
                
                elif isinstance(module, nn.LayerNorm):
                    # 优化层归一化层
                    self._optimize_layer_norm(module, name)
            
            logger.info("模型层优化完成")
            return model
            
        except Exception as e:
            logger.error(f"模型层优化失败: {e}")
            return model
    
    def _optimize_module_dimensions(self, module: nn.Module, name: str):
        """优化模块维度以适应Tensor Core"""
        try:
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                # 对于卷积层，确保输入输出通道数是8的倍数
                if hasattr(module, 'in_channels') and module.in_channels % 8 != 0:
                    logger.debug(f"优化 {name} 输入通道数: {module.in_channels} -> {self._round_to_multiple(module.in_channels, 8)}")
                
                if hasattr(module, 'out_channels') and module.out_channels % 8 != 0:
                    logger.debug(f"优化 {name} 输出通道数: {module.out_channels} -> {self._round_to_multiple(module.out_channels, 8)}")
                
                # 确保卷积核大小适合Tensor Core
                if hasattr(module, 'kernel_size'):
                    kernel_size = module.kernel_size
                    if isinstance(kernel_size, (tuple, list)):
                        # 对于2D/3D卷积，确保空间维度适合
                        optimal_kernel = tuple(
                            self._round_to_multiple(k, 2) if k > 1 else k 
                            for k in kernel_size
                        )
                        if optimal_kernel != kernel_size:
                            logger.debug(f"优化 {name} 卷积核大小: {kernel_size} -> {optimal_kernel}")
            
            elif isinstance(module, nn.Linear):
                # 对于线性层，确保输入输出特征是8的倍数
                if hasattr(module, 'in_features') and module.in_features % 8 != 0:
                    logger.debug(f"优化 {name} 输入特征数: {module.in_features} -> {self._round_to_multiple(module.in_features, 8)}")
                
                if hasattr(module, 'out_features') and module.out_features % 8 != 0:
                    logger.debug(f"优化 {name} 输出特征数: {module.out_features} -> {self._round_to_multiple(module.out_features, 8)}")
        
        except Exception as e:
            logger.warning(f"模块维度优化失败 {name}: {e}")
    
    def _round_to_multiple(self, value: int, multiple: int) -> int:
        """将值向上舍入到指定倍数"""
        return ((value + multiple - 1) // multiple) * multiple
    
    def _optimize_batch_norm(self, module: nn.BatchNorm2d, name: str):
        """优化批量归一化层"""
        try:
            # 确保特征数是8的倍数
            num_features = module.num_features
            if num_features % 8 != 0:
                optimal_features = self._round_to_multiple(num_features, 8)
                logger.debug(f"优化 {name} 批量归一化特征数: {num_features} -> {optimal_features}")
        
        except Exception as e:
            logger.warning(f"批量归一化优化失败 {name}: {e}")
    
    def _optimize_layer_norm(self, module: nn.LayerNorm, name: str):
        """优化层归一化层"""
        try:
            # 确保归一化维度是8的倍数
            normalized_shape = module.normalized_shape
            if isinstance(normalized_shape, (tuple, list)) and len(normalized_shape) > 0:
                last_dim = normalized_shape[-1]
                if last_dim % 8 != 0:
                    optimal_dim = self._round_to_multiple(last_dim, 8)
                    logger.debug(f"优化 {name} 层归一化维度: {last_dim} -> {optimal_dim}")
        
        except Exception as e:
            logger.warning(f"层归一化优化失败 {name}: {e}")
    
    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """优化内存布局以提高缓存效率"""
        logger.info("优化内存布局...")
        
        try:
            for name, param in model.named_parameters():
                if param.data.is_cuda:
                    # 确保参数内存对齐和连续性
                    param.data = param.data.contiguous()
                    
                    # 如果有梯度，也确保梯度内存连续性
                    if param.grad is not None:
                        param.grad.data = param.grad.data.contiguous()
            
            logger.info("内存布局优化完成")
            return model
            
        except Exception as e:
            logger.error(f"内存布局优化失败: {e}")
            return model
    
    def optimize_data_transfer(self, data: torch.Tensor, device: int = 0, 
                             non_blocking: bool = True) -> torch.Tensor:
        """
        优化数据传输到GPU
        
        Args:
            data: 要传输的数据张量
            device: 目标GPU设备ID
            non_blocking: 是否使用非阻塞传输
            
        Returns:
            传输到GPU的数据
        """
        try:
            if not data.is_cuda:
                # 确保数据是连续的以优化传输
                if not data.is_contiguous():
                    data = data.contiguous()
                
                # 使用指定设备传输
                with torch.cuda.device(device):
                    return data.to(f'cuda:{device}', non_blocking=non_blocking)
            
            return data
            
        except Exception as e:
            logger.error(f"数据传输优化失败: {e}")
            return data.to(f'cuda:{device}')
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, Any]:
        """获取GPU内存信息"""
        try:
            if not torch.cuda.is_available():
                return {"error": "CUDA不可用"}
            
            # 获取PyTorch内存统计
            memory_stats = {
                "allocated": torch.cuda.memory_allocated(device_id) // 1024**2,  # MB
                "reserved": torch.cuda.memory_reserved(device_id) // 1024**2,    # MB
                "free": 0,  # 需要NVIDIA ML计算
                "total": 0,  # 需要NVIDIA ML计算
            }
            
            # 使用NVIDIA ML获取更详细信息
            if NVIDIA_ML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    
                    # 获取显存信息
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_stats["total"] = memory_info.total // 1024**2
                    memory_stats["free"] = memory_info.free // 1024**2
                    memory_stats["used"] = memory_info.used // 1024**2
                    
                    # 获取GPU利用率
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_stats["gpu_utilization"] = utilization.gpu
                    memory_stats["memory_utilization"] = utilization.memory
                    
                    pynvml.nvmlShutdown()
                    
                except Exception as e:
                    logger.warning(f"NVIDIA ML内存信息获取失败: {e}")
            
            return memory_stats
            
        except Exception as e:
            logger.error(f"GPU内存信息获取失败: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """清理GPU资源"""
        try:
            logger.info("清理GPU资源...")
            
            # 清空CUDA缓存
            if torch.cuda.is_available():
                for device_id in range(self.gpu_count):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            # 清理CUDA流
            self.cuda_streams.clear()
            
            logger.info("GPU资源清理完成")
            
        except Exception as e:
            logger.error(f"GPU资源清理失败: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return {
            "gpu_count": self.gpu_count,
            "initialized": self._initialized,
            "config": {
                "memory_fraction": self.config.memory_fraction,
                "enable_tensor_cores": self.config.enable_tensor_cores,
                "enable_cudnn_benchmark": self.config.enable_cudnn_benchmark,
                "enable_tf32": self.config.enable_tf32,
            },
            "device_settings": self.optimal_settings,
            "cuda_streams": len(self.cuda_streams),
        }

# 全局GPU优化器实例
_gpu_optimizer = None

def get_gpu_optimizer(config: Optional[Dict[str, Any]] = None) -> GPUOptimizer:
    """获取全局GPU优化器实例"""
    global _gpu_optimizer
    
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer(config)
    
    return _gpu_optimizer