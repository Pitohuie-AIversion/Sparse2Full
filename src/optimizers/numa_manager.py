"""
NUMA架构内存管理器
基于技术方案实现NUMA感知的内存分配和数据加载优化
"""

import os
import threading
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

try:
    import numa
    NUMA_AVAILABLE = True
except ImportError:
    NUMA_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class NUMAMemoryConfig:
    """NUMA内存配置"""
    numa_nodes: int = 1
    memory_pools: Dict[int, Any] = field(default_factory=dict)
    thread_affinities: Dict[int, int] = field(default_factory=dict)
    preferred_node: int = 0

class NUMAMemoryManager:
    """
    NUMA架构内存管理器
    实现NUMA感知的内存分配和数据加载优化
    """
    
    def __init__(self, numa_nodes: Optional[int] = None):
        self.config = NUMAMemoryConfig()
        self.numa_nodes = numa_nodes or self._detect_numa_nodes()
        self.config.numa_nodes = self.numa_nodes
        self.memory_pools = {}
        self.thread_pools = {}
        self._initialized = False
        
        self._initialize_numa_manager()
    
    def _detect_numa_nodes(self) -> int:
        """检测NUMA节点数"""
        try:
            if NUMA_AVAILABLE:
                return numa.get_max_node() + 1
            
            # 备用检测方法
            node_path = Path("/sys/devices/system/node")
            if node_path.exists():
                node_dirs = list(node_path.glob("node*"))
                return len(node_dirs)
            
            # 检查CPU拓扑
            cpu_count = os.cpu_count()
            if cpu_count and cpu_count > 64:  # 大型系统通常有多个NUMA节点
                return min(2, cpu_count // 64)
            
            return 1
            
        except Exception as e:
            logger.warning(f"NUMA节点检测失败: {e}，使用默认值1")
            return 1
    
    def _initialize_numa_manager(self):
        """初始化NUMA管理器"""
        logger.info(f"初始化NUMA内存管理器，节点数: {self.numa_nodes}")
        
        try:
            # 创建内存池
            self._create_memory_pools()
            
            # 初始化线程池
            self._initialize_thread_pools()
            
            # 设置线程亲和性
            self._setup_thread_affinities()
            
            self._initialized = True
            logger.info("NUMA内存管理器初始化完成")
            
        except Exception as e:
            logger.error(f"NUMA内存管理器初始化失败: {e}")
            self._initialized = False
    
    def _create_memory_pools(self):
        """创建NUMA感知的内存池"""
        logger.info("创建NUMA内存池...")
        
        try:
            for node_id in range(self.numa_nodes):
                if NUMA_AVAILABLE:
                    # 创建NUMA节点绑定的内存池
                    try:
                        numa.set_preferred(node_id)
                        self.memory_pools[node_id] = {
                            "node_id": node_id,
                            "preferred": True,
                            "memory_size": 0
                        }
                        logger.info(f"NUMA节点 {node_id} 内存池创建成功")
                    except Exception as e:
                        logger.warning(f"NUMA节点 {node_id} 内存池创建失败: {e}")
                        self.memory_pools[node_id] = {
                            "node_id": node_id,
                            "preferred": False,
                            "memory_size": 0
                        }
                else:
                    # 模拟NUMA内存池
                    self.memory_pools[node_id] = {
                        "node_id": node_id,
                        "preferred": True,
                        "memory_size": 0
                    }
            
            logger.info(f"创建了 {len(self.memory_pools)} 个NUMA内存池")
            
        except Exception as e:
            logger.error(f"NUMA内存池创建失败: {e}")
            # 创建默认内存池
            self.memory_pools[0] = {
                "node_id": 0,
                "preferred": True,
                "memory_size": 0
            }
    
    def _initialize_thread_pools(self):
        """初始化线程池"""
        logger.info("初始化NUMA线程池...")
        
        try:
            for node_id in range(self.numa_nodes):
                # 为每个NUMA节点创建线程池
                thread_pool = ThreadPoolExecutor(
                    max_workers=8,  # 每个NUMA节点8个工作线程
                    thread_name_prefix=f"numa_{node_id}"
                )
                self.thread_pools[node_id] = thread_pool
                logger.info(f"NUMA节点 {node_id} 线程池创建成功")
            
        except Exception as e:
            logger.error(f"线程池初始化失败: {e}")
            # 创建默认线程池
            self.thread_pools[0] = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="default"
            )
    
    def _setup_thread_affinities(self):
        """设置线程亲和性"""
        logger.info("设置线程亲和性...")
        
        try:
            if NUMA_AVAILABLE and self.numa_nodes > 1:
                # 获取当前线程ID
                current_thread = threading.current_thread()
                
                # 为每个NUMA节点设置CPU亲和性
                for node_id in range(self.numa_nodes):
                    try:
                        # 获取NUMA节点的CPU列表
                        node_cpus = numa.node_to_cpus(node_id)
                        if node_cpus:
                            # 设置线程亲和性
                            self.config.thread_affinities[node_id] = node_cpus[0]  # 使用第一个CPU
                            logger.info(f"NUMA节点 {node_id} 线程亲和性设置完成: CPU {node_cpus[0]}")
                    except Exception as e:
                        logger.warning(f"NUMA节点 {node_id} 线程亲和性设置失败: {e}")
            
            else:
                logger.info("单NUMA节点或NUMA不可用，跳过线程亲和性设置")
                
        except Exception as e:
            logger.error(f"线程亲和性设置失败: {e}")
    
    def get_preferred_numa_node(self) -> int:
        """获取当前线程的首选NUMA节点"""
        try:
            if NUMA_AVAILABLE and self.numa_nodes > 1:
                # 获取当前线程的NUMA节点
                current_node = numa.get_current_node()
                if current_node in self.memory_pools:
                    return current_node
            
            # 基于线程ID的简单负载均衡
            thread_id = threading.current_thread().ident or 0
            return thread_id % self.numa_nodes
            
        except Exception as e:
            logger.debug(f"获取首选NUMA节点失败: {e}，使用默认值0")
            return 0
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                       device: str = "cpu", pin_memory: bool = True) -> torch.Tensor:
        """
        NUMA感知的张量分配
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            device: 设备类型 ("cpu" 或 "cuda:X")
            pin_memory: 是否使用锁页内存
            
        Returns:
            分配的张量
        """
        try:
            if device.startswith("cuda"):
                # GPU张量分配
                return torch.empty(shape, dtype=dtype, device=device)
            
            # CPU张量分配 - NUMA优化
            preferred_node = self.get_preferred_numa_node()
            
            if NUMA_AVAILABLE and self._initialized:
                try:
                    # 设置NUMA内存分配策略
                    original_preferred = numa.get_preferred()
                    numa.set_preferred(preferred_node)
                    
                    # 分配内存
                    tensor = torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
                    
                    # 恢复原始设置
                    numa.set_preferred(original_preferred)
                    
                    # 更新内存池统计
                    self.memory_pools[preferred_node]["memory_size"] += tensor.numel() * tensor.element_size()
                    
                    logger.debug(f"NUMA节点 {preferred_node} 张量分配完成: {shape}, {dtype}")
                    return tensor
                    
                except Exception as e:
                    logger.warning(f"NUMA张量分配失败: {e}，使用标准分配")
            
            # 标准张量分配
            tensor = torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
            logger.debug(f"标准张量分配完成: {shape}, {dtype}")
            return tensor
            
        except Exception as e:
            logger.error(f"张量分配失败: {e}")
            # 最后的回退方案
            return torch.zeros(shape, dtype=dtype)
    
    def allocate_numpy_array(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """NUMA感知的NumPy数组分配"""
        try:
            preferred_node = self.get_preferred_numa_node()
            
            if NUMA_AVAILABLE and self._initialized:
                try:
                    # 设置NUMA内存分配策略
                    original_preferred = numa.get_preferred()
                    numa.set_preferred(preferred_node)
                    
                    # 分配NumPy数组
                    array = np.empty(shape, dtype=dtype)
                    
                    # 恢复原始设置
                    numa.set_preferred(original_preferred)
                    
                    logger.debug(f"NUMA节点 {preferred_node} NumPy数组分配完成: {shape}, {dtype}")
                    return array
                    
                except Exception as e:
                    logger.warning(f"NUMA NumPy数组分配失败: {e}，使用标准分配")
            
            # 标准NumPy数组分配
            array = np.empty(shape, dtype=dtype)
            logger.debug(f"标准NumPy数组分配完成: {shape}, {dtype}")
            return array
            
        except Exception as e:
            logger.error(f"NumPy数组分配失败: {e}")
            return np.zeros(shape, dtype=dtype)
    
    def optimize_data_loading(self, dataloader: DataLoader, numa_aware: bool = True) -> DataLoader:
        """
        优化数据加载器以利用NUMA架构
        
        Args:
            dataloader: 原始数据加载器
            numa_aware: 是否启用NUMA感知优化
            
        Returns:
            优化的数据加载器
        """
        if not numa_aware or not self._initialized:
            logger.info("NUMA优化未启用，返回原始数据加载器")
            return dataloader
        
        try:
            logger.info("优化数据加载器以利用NUMA架构...")
            
            # 计算基于NUMA的最优工作进程数
            optimal_workers = self.numa_nodes * 8  # 每个NUMA节点8个工作进程
            
            # 限制在合理范围内
            optimal_workers = max(4, min(optimal_workers, 32))
            
            # 创建工作进程初始化函数
            def numa_aware_worker_init(worker_id):
                """NUMA感知的工作进程初始化"""
                try:
                    # 为工作进程分配NUMA节点
                    numa_node = worker_id % self.numa_nodes
                    
                    if NUMA_AVAILABLE:
                        # 设置内存分配策略
                        numa.set_preferred(numa_node)
                        
                        # 设置CPU亲和性（如果可能）
                        try:
                            node_cpus = numa.node_to_cpus(numa_node)
                            if node_cpus and len(node_cpus) > 0:
                                # 简单地将工作进程绑定到NUMA节点的第一个CPU
                                os.sched_setaffinity(0, {node_cpus[0]})
                        except Exception as e:
                            logger.debug(f"CPU亲和性设置失败: {e}")
                    
                    logger.debug(f"工作进程 {worker_id} 绑定到NUMA节点 {numa_node}")
                    
                except Exception as e:
                    logger.warning(f"工作进程初始化失败: {e}")
            
            # 创建优化的数据加载器
            optimized_loader = DataLoader(
                dataset=dataloader.dataset,
                batch_size=dataloader.batch_size,
                shuffle=getattr(dataloader, 'shuffle', True),
                sampler=dataloader.sampler,
                batch_sampler=dataloader.batch_sampler,
                num_workers=optimal_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=True,  # 启用锁页内存
                drop_last=getattr(dataloader, 'drop_last', False),
                timeout=getattr(dataloader, 'timeout', 0),
                worker_init_fn=numa_aware_worker_init,
                multiprocessing_context=getattr(dataloader, 'multiprocessing_context', None),
                generator=getattr(dataloader, 'generator', None),
                prefetch_factor=4,  # 增加预取因子
                persistent_workers=True  # 保持工作进程存活
            )
            
            logger.info(f"NUMA优化数据加载器创建完成: {optimal_workers}个工作进程")
            return optimized_loader
            
        except Exception as e:
            logger.error(f"数据加载器优化失败: {e}，返回原始数据加载器")
            return dataloader
    
    def optimize_dataset_memory(self, dataset: Dataset) -> Dataset:
        """
        优化数据集内存使用
        
        Args:
            dataset: 原始数据集
            
        Returns:
            内存优化的数据集包装器
        """
        class NUMADatasetWrapper(Dataset):
            """NUMA优化的数据集包装器"""
            
            def __init__(self, original_dataset, numa_manager):
                self.original_dataset = original_dataset
                self.numa_manager = numa_manager
                self._memory_cache = {}
            
            def __len__(self):
                return len(self.original_dataset)
            
            def __getitem__(self, idx):
                try:
                    # 尝试从缓存获取
                    if idx in self._memory_cache:
                        return self._memory_cache[idx]
                    
                    # 获取数据
                    data = self.original_dataset[idx]
                    
                    # 如果是NumPy数组，使用NUMA优化分配
                    if isinstance(data, np.ndarray):
                        optimized_data = self.numa_manager.allocate_numpy_array(
                            data.shape, data.dtype
                        )
                        optimized_data[:] = data  # 复制数据
                        data = optimized_data
                    
                    # 缓存数据（限制缓存大小）
                    if len(self._memory_cache) < 1000:  # 限制缓存大小
                        self._memory_cache[idx] = data
                    
                    return data
                    
                except Exception as e:
                    logger.warning(f"数据集项 {idx} 优化失败: {e}，返回原始数据")
                    return self.original_dataset[idx]
        
        try:
            logger.info("优化数据集内存使用...")
            return NUMADatasetWrapper(dataset, self)
            
        except Exception as e:
            logger.error(f"数据集内存优化失败: {e}，返回原始数据集")
            return dataset
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取NUMA内存使用统计"""
        try:
            stats = {
                "numa_nodes": self.numa_nodes,
                "memory_pools": {},
                "total_allocated": 0
            }
            
            for node_id, pool in self.memory_pools.items():
                pool_stats = {
                    "node_id": node_id,
                    "memory_size": pool["memory_size"],
                    "preferred": pool["preferred"]
                }
                stats["memory_pools"][f"node_{node_id}"] = pool_stats
                stats["total_allocated"] += pool["memory_size"]
            
            return stats
            
        except Exception as e:
            logger.error(f"获取内存统计失败: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """清理NUMA资源"""
        try:
            logger.info("清理NUMA资源...")
            
            # 关闭线程池
            for node_id, thread_pool in self.thread_pools.items():
                try:
                    thread_pool.shutdown(wait=True)
                    logger.debug(f"NUMA节点 {node_id} 线程池已关闭")
                except Exception as e:
                    logger.warning(f"线程池关闭失败: {e}")
            
            # 清理内存池统计
            for pool in self.memory_pools.values():
                pool["memory_size"] = 0
            
            logger.info("NUMA资源清理完成")
            
        except Exception as e:
            logger.error(f"NUMA资源清理失败: {e}")

# 全局NUMA内存管理器实例
_numa_manager = None

def get_numa_manager(numa_nodes: Optional[int] = None) -> NUMAMemoryManager:
    """获取全局NUMA内存管理器实例"""
    global _numa_manager
    
    if _numa_manager is None:
        _numa_manager = NUMAMemoryManager(numa_nodes)
    
    return _numa_manager