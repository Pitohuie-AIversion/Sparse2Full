"""
硬件自动检测与资源分配模块
基于技术方案实现智能硬件配置检测和最优参数计算
"""

import os
import psutil
import torch
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import subprocess
import json
from pathlib import Path

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

try:
    import cpuinfo
    CPU_INFO_AVAILABLE = True
except ImportError:
    CPU_INFO_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class HardwareConfig:
    """硬件配置数据类（兼容测试期望的训练配置字段）"""
    # 硬件检测信息
    cpu_info: Dict[str, Any] = field(default_factory=dict)
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    memory_info: Dict[str, Any] = field(default_factory=dict)

    # 训练相关最优配置（测试期望）
    batch_size: int = 32
    num_workers: int = 4
    mixed_precision: bool = True
    compile_model: bool = False
    gradient_accumulation_steps: int = 1
    numa_optimization_level: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class HardwareProfiler:
    """
    硬件配置自动检测与优化器
    基于技术方案实现智能硬件检测和最优配置计算
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/hardware_profile.json"
        self.hardware_config = HardwareConfig()
        self._initialize_hardware_detection()
    
    def _initialize_hardware_detection(self):
        """初始化硬件检测"""
        logger.info("开始硬件自动检测与配置...")
        
        # 检测CPU配置
        self.hardware_config.cpu_info = self._detect_cpu()
        
        # 检测GPU配置
        self.hardware_config.gpu_info = self._detect_gpu()
        
        # 检测内存配置
        self.hardware_config.memory_info = self._detect_memory()
        
        # 计算最优配置（数据类）
        optimal_cfg = self._calculate_optimal_config()
        # 将检测信息填入数据类
        optimal_cfg.cpu_info = self.hardware_config.cpu_info
        optimal_cfg.gpu_info = self.hardware_config.gpu_info
        optimal_cfg.memory_info = self.hardware_config.memory_info
        self.hardware_config = optimal_cfg
        
        logger.info("硬件检测与配置完成")
        self._log_hardware_info()
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """检测CPU配置"""
        logger.info("检测CPU配置...")
        
        cpu_info = {}
        
        try:
            # 获取CPU基本信息
            if CPU_INFO_AVAILABLE:
                info = cpuinfo.get_cpu_info()
                cpu_info["model"] = info.get("brand_raw", "Unknown")
                cpu_info["arch"] = info.get("arch_string_raw", "Unknown")
                cpu_info["bits"] = info.get("bits", 64)
                cpu_info["flags"] = info.get("flags", [])
                cpu_info["avx512_support"] = "avx512f" in cpu_info["flags"]
            else:
                # 备用检测方法
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    if "AMD EPYC" in content:
                        cpu_info["model"] = "AMD EPYC 9654"
                    else:
                        cpu_info["model"] = "Unknown CPU"
                    cpu_info["avx512_support"] = "avx512f" in content
            
            # 获取CPU核心信息
            cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
            cpu_info["logical_cores"] = psutil.cpu_count(logical=True)
            cpu_info["frequency"] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
            
            # 检测NUMA拓扑
            cpu_info["numa_nodes"] = self._detect_numa_topology()
            
            # 获取缓存信息
            cpu_info["l3_cache"] = self._get_l3_cache_size()
            
            logger.info(f"CPU检测完成: {cpu_info['model']} ({cpu_info['logical_cores']}核)")
            
        except Exception as e:
            logger.error(f"CPU检测失败: {e}")
            cpu_info = {
                "model": "Unknown CPU",
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "numa_nodes": 1,
                "avx512_support": False
            }
        
        return cpu_info
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """检测GPU配置"""
        logger.info("检测GPU配置...")
        
        gpu_info = {}
        
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA不可用，使用CPU模式")
                gpu_info["available"] = False
                gpu_info["count"] = 0
                return gpu_info
            
            # 获取GPU数量
            gpu_count = torch.cuda.device_count()
            gpu_info["available"] = True
            gpu_info["count"] = gpu_count
            
            if gpu_count == 0:
                logger.warning("未检测到GPU设备")
                return gpu_info
            
            # 检测每个GPU的详细信息
            gpu_info["devices"] = []
            
            if NVIDIA_ML_AVAILABLE:
                pynvml.nvmlInit()
                
                for i in range(gpu_count):
                    device_info = {}
                    
                    # 获取设备句柄
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # 获取设备名称
                    device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    device_info["name"] = device_name
                    
                    # 获取显存信息
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    device_info["total_memory"] = memory_info.total // 1024**2  # MB
                    device_info["free_memory"] = memory_info.free // 1024**2
                    device_info["used_memory"] = memory_info.used // 1024**2
                    
                    # 获取CUDA计算能力
                    cuda_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    device_info["compute_capability"] = f"{cuda_capability[0]}.{cuda_capability[1]}"
                    
                    # 获取驱动信息
                    driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    device_info["driver_version"] = driver_version
                    
                    # 获取CUDA核心数
                    try:
                        device_info["cuda_cores"] = pynvml.nvmlDeviceGetNumGpuCores(handle)
                    except:
                        device_info["cuda_cores"] = "Unknown"
                    
                    # Tensor Core支持检测
                    device_info["tensor_cores"] = self._detect_tensor_cores(device_name)
                    
                    # 获取设备属性
                    props = torch.cuda.get_device_properties(i)
                    device_info["multi_processor_count"] = props.multi_processor_count
                    device_info["max_threads_per_block"] = props.max_threads_per_block
                    device_info["warp_size"] = props.warp_size
                    
                    gpu_info["devices"].append(device_info)
                
                pynvml.nvmlShutdown()
            
            else:
                # 备用检测方法（仅使用PyTorch）
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    device_info = {
                        "name": props.name,
                        "total_memory": props.total_memory // 1024**2,
                        "multi_processor_count": props.multi_processor_count,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "tensor_cores": props.major >= 7,  # Turing架构及以上支持Tensor Core
                        "cuda_cores": "Unknown (pynvml not available)"
                    }
                    gpu_info["devices"].append(device_info)
            
            logger.info(f"GPU检测完成: {gpu_count}个GPU设备")
            
        except Exception as e:
            logger.error(f"GPU检测失败: {e}")
            gpu_info = {
                "available": torch.cuda.is_available(),
                "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "devices": []
            }
        
        return gpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """检测内存配置"""
        logger.info("检测内存配置...")
        
        memory_info = {}
        
        try:
            # 获取系统内存信息
            vm = psutil.virtual_memory()
            memory_info["total"] = vm.total // 1024**3  # GB
            memory_info["available"] = vm.available // 1024**3
            memory_info["used"] = vm.used // 1024**3
            memory_info["percentage"] = vm.percent
            
            # 获取交换分区信息
            swap = psutil.swap_memory()
            memory_info["swap_total"] = swap.total // 1024**3
            memory_info["swap_used"] = swap.used // 1024**3
            memory_info["swap_percentage"] = swap.percent
            
            # 获取内存带宽信息（如果可用）
            memory_info["bandwidth"] = self._estimate_memory_bandwidth()
            
            logger.info(f"内存检测完成: 总计{memory_info['total']}GB，可用{memory_info['available']}GB")
            
        except Exception as e:
            logger.error(f"内存检测失败: {e}")
            memory_info = {
                "total": 0,
                "available": 0,
                "used": 0,
                "percentage": 0
            }
        
        return memory_info
    
    def _detect_numa_topology(self) -> int:
        """检测NUMA拓扑结构"""
        try:
            # 尝试使用numactl命令
            result = subprocess.run(['numactl', '--hardware'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # 解析NUMA节点数
                for line in result.stdout.split('\n'):
                    if 'available' in line and 'nodes' in line:
                        import re
                        match = re.search(r'(\d+) nodes', line)
                        if match:
                            return int(match.group(1))
            
            # 备用方法：检查/sys/devices/system/node/
            node_path = Path("/sys/devices/system/node")
            if node_path.exists():
                node_dirs = list(node_path.glob("node*"))
                return len(node_dirs)
            
            # 默认返回1个NUMA节点
            return 1
            
        except Exception as e:
            logger.warning(f"NUMA拓扑检测失败: {e}，使用默认值1")
            return 1
    
    def _get_l3_cache_size(self) -> str:
        """获取L3缓存大小"""
        try:
            # 尝试从/sys/devices/system/cpu/获取
            cache_path = Path("/sys/devices/system/cpu/cpu0/cache")
            if cache_path.exists():
                for cache_dir in cache_path.glob("index*"):
                    level_file = cache_dir / "level"
                    if level_file.exists():
                        with open(level_file, 'r') as f:
                            level = f.read().strip()
                            if level == "3":
                                size_file = cache_dir / "size"
                                if size_file.exists():
                                    with open(size_file, 'r') as f:
                                        return f.read().strip()
            
            # 备用方法：使用getconf
            result = subprocess.run(['getconf', 'LEVEL3_CACHE_SIZE'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                size_bytes = int(result.stdout.strip())
                if size_bytes > 0:
                    return f"{size_bytes // 1024 // 1024}MB"
            
            return "Unknown"
            
        except Exception as e:
            logger.warning(f"L3缓存检测失败: {e}")
            return "Unknown"
    
    def _detect_tensor_cores(self, device_name: str) -> bool:
        """检测Tensor Core支持"""
        # NVIDIA GPU架构支持Tensor Core的情况
        tensor_core_architectures = [
            "V100", "T4", "TITAN V",  # Volta架构
            "2080", "2070", "2060", "TITAN RTX",  # Turing架构
            "A100", "A40", "A30", "A10",  # Ampere架构
            "L40", "L4", "H100", "H40"  # Ada Lovelace和Hopper架构
        ]
        
        return any(arch in device_name for arch in tensor_core_architectures)
    
    def _estimate_memory_bandwidth(self) -> str:
        """估算内存带宽"""
        try:
            # 基于CPU型号估算内存带宽
            cpu_model = self.hardware_config.cpu_info.get("model", "")
            
            if "AMD EPYC" in cpu_model:
                # AMD EPYC 9654支持8通道DDR5-4800
                # 理论带宽：8 × 4800 MHz × 8 bytes/cycle = 307.2 GB/s
                return "307.2 GB/s"
            elif "Intel" in cpu_model:
                # Intel Xeon类似配置
                return "200+ GB/s"
            else:
                return "Unknown"
                
        except Exception as e:
            logger.warning(f"内存带宽估算失败: {e}")
            return "Unknown"
    
    def _calculate_optimal_config(self) -> HardwareConfig:
        """计算最优配置参数，返回 HardwareConfig 数据类"""
        logger.info("计算最优训练配置...")
        config_kv: Dict[str, Any] = {}
        
        try:
            # 基础批次大小计算
            config_kv["batch_size"] = self._optimal_batch_size()
            
            # 工作进程数计算
            config_kv["num_workers"] = self._optimal_workers()
            
            # GPU内存使用比例
            config_kv["gpu_memory_fraction"] = 0.85  # 保守策略，避免OOM
            
            # 混合精度训练
            config_kv["mixed_precision"] = self._should_use_mixed_precision()
            
            # 模型编译优化
            config_kv["compile_model"] = self._should_compile_model()
            
            # 梯度累积步数
            config_kv["gradient_accumulation_steps"] = self._optimal_gradient_accumulation()
            
            # NUMA优化
            numa_nodes = (self.hardware_config.cpu_info or {}).get("numa_nodes", 1)
            config_kv["numa_optimization_level"] = int(numa_nodes)
            
            # 分布式训练
            config_kv["distributed"] = (self.hardware_config.gpu_info or {}).get("count", 0) > 1
            
            logger.info(f"最优配置计算完成: batch_size={config_kv['batch_size']}, workers={config_kv['num_workers']}")
            
        except Exception as e:
            logger.error(f"最优配置计算失败: {e}，使用默认配置")
            config_kv = {
                "batch_size": 16,
                "num_workers": 4,
                "gpu_memory_fraction": 0.8,
                "mixed_precision": True,
                "compile_model": False,
                "gradient_accumulation_steps": 1,
                "numa_optimization_level": 1,
                "distributed": False
            }
        # 构造数据类
        return HardwareConfig(
            batch_size=int(config_kv.get("batch_size", 32)),
            num_workers=int(config_kv.get("num_workers", 4)),
            mixed_precision=bool(config_kv.get("mixed_precision", True)),
            compile_model=bool(config_kv.get("compile_model", False)),
            gradient_accumulation_steps=int(config_kv.get("gradient_accumulation_steps", 1)),
            numa_optimization_level=int(config_kv.get("numa_optimization_level", 1)),
        )
    
    def _optimal_batch_size(self) -> int:
        """计算最优批次大小"""
        gpu_info = self.hardware_config.gpu_info
        memory_info = self.hardware_config.memory_info
        
        if not gpu_info.get("available", False):
            # CPU模式，基于内存计算
            available_memory = memory_info.get("available", 16)  # GB
            return min(64, available_memory * 2)  # 保守估计
        
        # GPU模式，基于显存计算
        if gpu_info.get("devices"):
            # 取最小显存的GPU作为基准
            min_gpu_memory = min(device["total_memory"] for device in gpu_info["devices"])
            
            # 根据模型复杂度和显存大小估算
            # 假设每个样本需要约100MB显存（保守估计）
            base_batch_size = int(min_gpu_memory * 0.8 / 100)
            
            # 考虑Tensor Core优化，批次大小应该是8的倍数
            optimal_batch_size = (base_batch_size // 8) * 8
            
            # 限制在合理范围内
            return max(8, min(optimal_batch_size, 256))
        
        return 32  # 默认值
    
    def _optimal_workers(self) -> int:
        """计算最优工作进程数"""
        cpu_info = self.hardware_config.cpu_info
        numa_nodes = cpu_info.get("numa_nodes", 1)
        logical_cores = cpu_info.get("logical_cores", 8)
        
        # 基于NUMA架构优化
        # 每个NUMA节点分配8个工作进程
        numa_workers = numa_nodes * 8
        
        # 基于CPU核心数的限制
        core_based_workers = min(logical_cores // 2, 32)  # 不超过32个工作进程
        
        # 取两者中的较小值
        return min(numa_workers, core_based_workers)
    
    def _should_use_mixed_precision(self) -> bool:
        """判断是否应使用混合精度训练"""
        gpu_info = self.hardware_config.gpu_info
        
        if not gpu_info.get("available", False):
            return False
        
        # 检查GPU架构是否支持Tensor Core
        if gpu_info.get("devices"):
            for device in gpu_info["devices"]:
                if device.get("tensor_cores", False):
                    return True
        
        return False
    
    def _should_compile_model(self) -> bool:
        """判断是否应编译模型"""
        # 检查PyTorch版本是否支持编译
        if hasattr(torch, 'compile'):
            # 对于支持的硬件配置启用编译
            return True
        
        return False
    
    def _optimal_gradient_accumulation(self) -> int:
        """计算最优梯度累积步数"""
        # 基于GPU数量和显存大小
        gpu_info = self.hardware_config.gpu_info
        
        if gpu_info.get("count", 0) <= 1:
            return 1  # 单GPU不需要梯度累积
        
        # 多GPU情况下，根据显存大小调整
        if gpu_info.get("devices"):
            min_gpu_memory = min(device["total_memory"] for device in gpu_info["devices"])
            if min_gpu_memory < 16000:  # 小于16GB显存
                return 2
        
        return 1
    
    def _log_hardware_info(self):
        """记录硬件信息"""
        logger.info("=== 硬件配置摘要 ===")
        
        # CPU信息
        cpu_info = self.hardware_config.cpu_info
        logger.info(f"CPU: {cpu_info.get('model', 'Unknown')} "
                   f"({cpu_info.get('logical_cores', 0)}核, "
                   f"{cpu_info.get('numa_nodes', 1)} NUMA节点)")
        
        # GPU信息
        gpu_info = self.hardware_config.gpu_info
        if gpu_info.get("available", False):
            logger.info(f"GPU: {gpu_info.get('count', 0)}个设备")
            for i, device in enumerate(gpu_info.get("devices", [])):
                logger.info(f"  GPU {i}: {device.get('name', 'Unknown')} "
                           f"({device.get('total_memory', 0)}MB, "
                           f"CC: {device.get('compute_capability', 'Unknown')})")
        else:
            logger.info("GPU: 不可用，使用CPU模式")
        
        # 内存信息
        memory_info = self.hardware_config.memory_info
        logger.info(f"内存: {memory_info.get('total', 0)}GB总计, "
                   f"{memory_info.get('available', 0)}GB可用")
        
        # 最优配置（直接读取数据类字段）
        logger.info(
            f"最优配置: batch_size={getattr(self.hardware_config, 'batch_size', 'Unknown')}, "
            f"workers={getattr(self.hardware_config, 'num_workers', 'Unknown')}, "
            f"mixed_precision={getattr(self.hardware_config, 'mixed_precision', False)}"
        )
        
        logger.info("===================")
    
    def get_optimal_config(self) -> HardwareConfig:
        """获取计算的最优配置（数据类）"""
        return self.hardware_config
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """获取硬件配置摘要"""
        return {
            "cpu": self.hardware_config.cpu_info,
            "gpu": self.hardware_config.gpu_info,
            "memory": self.hardware_config.memory_info,
            "optimal_config": self.hardware_config.to_dict()
        }
    
    def save_profile(self, filepath: Optional[str] = None):
        """保存硬件配置文件"""
        save_path = filepath or self.config_path
        
        try:
            profile_data = self.get_hardware_summary()
            
            # 确保目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"硬件配置文件已保存: {save_path}")
            
        except Exception as e:
            logger.error(f"保存硬件配置文件失败: {e}")
    
    def load_profile(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """加载硬件配置文件"""
        load_path = filepath or self.config_path
        
        try:
            if Path(load_path).exists():
                with open(load_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                
                logger.info(f"硬件配置文件已加载: {load_path}")
                return profile_data
            else:
                logger.warning(f"硬件配置文件不存在: {load_path}")
                return self.get_hardware_summary()
                
        except Exception as e:
            logger.error(f"加载硬件配置文件失败: {e}")
            return self.get_hardware_summary()

    def estimate_training_performance(self, config: HardwareConfig) -> Dict[str, float]:
        """估算训练性能（兼容测试期望的键）"""
        try:
            batch_size = max(1, int(config.batch_size))
            workers = max(1, int(config.num_workers))
            mp_factor = 1.2 if config.mixed_precision else 1.0
            compile_factor = 1.1 if config.compile_model else 1.0

            # 简化估算：样本/秒 ~ 基础系数 * workers * mp/compile
            base_sp = 50.0  # 基础CPU/GPU无关系数
            samples_per_second = base_sp * workers * mp_factor * compile_factor

            # 每epoch时间（假设1000样本）
            estimated_epoch_time = 1000.0 / samples_per_second

            # 内存使用估算（GB）
            mem_gb = (batch_size * 0.1)  # 每样本约0.1GB的保守估计

            return {
                'samples_per_second': float(samples_per_second),
                'estimated_epoch_time': float(estimated_epoch_time),
                'memory_usage_gb': float(mem_gb)
            }
        except Exception as e:
            logger.error(f"性能估算失败: {e}")
            return {
                'samples_per_second': 0.0,
                'estimated_epoch_time': 0.0,
                'memory_usage_gb': 0.0
            }

# 全局硬件分析器实例
_hardware_profiler = None

def get_hardware_profiler(config_path: Optional[str] = None) -> HardwareProfiler:
    """获取全局硬件分析器实例"""
    global _hardware_profiler
    
    if _hardware_profiler is None:
        _hardware_profiler = HardwareProfiler(config_path)
    
    return _hardware_profiler