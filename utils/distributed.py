#!/usr/bin/env python3
"""
分布式训练支持模块

实现多GPU训练和分布式数据并行，支持：
1. 分布式环境初始化和清理
2. 数据并行和模型同步
3. 梯度聚合和通信优化
4. 检查点保存和加载
5. 日志和监控

Author: PDEBench Team
Date: 2025-01-11
"""

import os
import sys
import time
import socket
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_logger(name: str, rank: int = 0) -> logging.Logger:
    """设置分布式日志记录器"""
    logger = logging.getLogger(name)
    
    # 只在主进程中输出日志
    if rank == 0:
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [Rank %(rank)d] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    else:
        logger.setLevel(logging.WARNING)
    
    return logger


class DistributedManager:
    """分布式训练管理器
    
    负责分布式环境的初始化、模型包装、数据分发等核心功能。
    支持多种后端（NCCL、Gloo）和启动方式。
    """
    
    def __init__(self, 
                 backend: str = "nccl",
                 init_method: str = "env://",
                 timeout_minutes: int = 30):
        """初始化分布式管理器
        
        Args:
            backend: 通信后端 ("nccl", "gloo", "mpi")
            init_method: 初始化方法
            timeout_minutes: 超时时间（分钟）
        """
        self.backend = backend
        self.init_method = init_method
        self.timeout = timeout_minutes * 60
        
        # 分布式状态
        self.is_distributed = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = None
        
        # 日志记录器
        self.logger = setup_logger("DistributedManager", 0)  # 初始化时使用rank 0
        
        # 性能统计
        self.comm_stats = {
            'allreduce_time': 0.0,
            'allreduce_count': 0,
            'broadcast_time': 0.0,
            'broadcast_count': 0
        }
    
    def setup(self) -> bool:
        """初始化分布式环境"""
        try:
            # 检查环境变量
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
                self.is_distributed = self.world_size > 1
            else:
                print("未检测到分布式环境变量，使用单GPU模式")
                self.is_distributed = False
                self.rank = 0
                self.local_rank = 0
                self.world_size = 1
            
            # 设置设备
            if torch.cuda.is_available():
                if self.is_distributed:
                    torch.cuda.set_device(self.local_rank)
                    self.device = torch.device(f"cuda:{self.local_rank}")
                else:
                    self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
                if self.is_distributed:
                    self.backend = "gloo"  # CPU只支持gloo
            
            # 更新日志记录器的rank
            self.logger = setup_logger("DistributedManager", self.rank)
            
            # 初始化进程组
            if self.is_distributed:
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    world_size=self.world_size,
                    rank=self.rank,
                    timeout=torch.distributed.default_pg_timeout
                )
                
                # 验证初始化
                if not dist.is_initialized():
                    raise RuntimeError("分布式初始化失败")
                
                # 同步所有进程
                dist.barrier()
            

            
            if self.is_distributed:
                self.logger.info(f"分布式训练已初始化")
                self.logger.info(f"  后端: {self.backend}")
                self.logger.info(f"  世界大小: {self.world_size}")
                self.logger.info(f"  当前排名: {self.rank}")
                self.logger.info(f"  本地排名: {self.local_rank}")
                self.logger.info(f"  设备: {self.device}")
            else:
                self.logger.info(f"单GPU训练模式，设备: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ 分布式初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """清理分布式环境"""
        if self.is_distributed and dist.is_initialized():
            self.logger.info("清理分布式环境...")
            
            # 打印通信统计
            if self.rank == 0:
                self.print_communication_stats()
            
            dist.destroy_process_group()
            self.logger.info("分布式环境已清理")
    
    def wrap_model(self, model: nn.Module, 
                   find_unused_parameters: bool = False,
                   gradient_as_bucket_view: bool = True) -> nn.Module:
        """包装模型为分布式模型"""
        model = model.to(self.device)
        
        if self.is_distributed:
            # 使用DistributedDataParallel包装
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=gradient_as_bucket_view
            )
            
            self.logger.info(f"模型已包装为DDP，设备: {self.device}")
        else:
            self.logger.info(f"单GPU模式，模型已移至设备: {self.device}")
        
        return model
    
    def create_dataloader(self, dataset, 
                         batch_size: int,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         pin_memory: bool = True,
                         drop_last: bool = True,
                         **kwargs) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        """创建分布式数据加载器"""
        sampler = None
        
        if self.is_distributed:
            # 创建分布式采样器
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=drop_last
            )
            shuffle = False  # 使用sampler时不能shuffle
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            **kwargs
        )
        
        self.logger.info(f"数据加载器已创建，批大小: {batch_size}, 工作进程: {num_workers}")
        
        return dataloader, sampler
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """全局归约操作"""
        if not self.is_distributed:
            return tensor
        
        start_time = time.time()
        
        # 确保tensor在正确设备上
        tensor = tensor.to(self.device)
        
        # 执行全局归约
        dist.all_reduce(tensor, op=op)
        
        # 统计通信时间
        self.comm_stats['allreduce_time'] += time.time() - start_time
        self.comm_stats['allreduce_count'] += 1
        
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """全局收集操作"""
        if not self.is_distributed:
            return tensor.unsqueeze(0)
        
        # 准备输出张量列表
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        
        # 执行全局收集
        dist.all_gather(tensor_list, tensor)
        
        # 拼接结果
        return torch.stack(tensor_list, dim=0)
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """广播操作"""
        if not self.is_distributed:
            return tensor
        
        start_time = time.time()
        
        # 确保tensor在正确设备上
        tensor = tensor.to(self.device)
        
        # 执行广播
        dist.broadcast(tensor, src=src)
        
        # 统计通信时间
        self.comm_stats['broadcast_time'] += time.time() - start_time
        self.comm_stats['broadcast_count'] += 1
        
        return tensor
    
    def reduce_dict(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """归约字典中的所有张量"""
        if not self.is_distributed:
            return input_dict
        
        reduced_dict = {}
        for key, tensor in input_dict.items():
            if isinstance(tensor, torch.Tensor):
                reduced_tensor = self.all_reduce(tensor.clone())
                reduced_dict[key] = reduced_tensor / self.world_size
            else:
                reduced_dict[key] = tensor
        
        return reduced_dict
    
    def average_gradients(self, model: nn.Module):
        """平均梯度（手动实现，通常DDP会自动处理）"""
        if not self.is_distributed:
            return
        
        for param in model.parameters():
            if param.grad is not None:
                self.all_reduce(param.grad.data)
                param.grad.data /= self.world_size
    
    def save_checkpoint(self, state_dict: Dict[str, Any], 
                       filepath: str,
                       is_best: bool = False,
                       only_main_process: bool = True):
        """保存检查点"""
        if only_main_process and self.rank != 0:
            return
        
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 添加分布式信息
        state_dict['distributed'] = {
            'world_size': self.world_size,
            'rank': self.rank,
            'backend': self.backend
        }
        
        # 保存检查点
        torch.save(state_dict, filepath)
        
        if is_best:
            best_path = str(Path(filepath).parent / "best_model.pth")
            torch.save(state_dict, best_path)
        
        self.logger.info(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str, 
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       map_location: Optional[str] = None) -> Dict[str, Any]:
        """加载检查点"""
        if map_location is None:
            map_location = str(self.device)
        
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # 加载模型状态
        if hasattr(model, 'module'):
            # DDP模型
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"检查点已加载: {filepath}")
        
        return checkpoint
    
    def barrier(self):
        """同步所有进程"""
        if self.is_distributed:
            dist.barrier()
    
    def print_communication_stats(self):
        """打印通信统计信息"""
        if self.rank != 0:
            return
        
        stats = self.comm_stats
        
        print("\n📊 分布式通信统计:")
        print(f"  AllReduce调用次数: {stats['allreduce_count']}")
        print(f"  AllReduce总时间: {stats['allreduce_time']:.3f}s")
        if stats['allreduce_count'] > 0:
            print(f"  AllReduce平均时间: {stats['allreduce_time']/stats['allreduce_count']*1000:.2f}ms")
        
        print(f"  Broadcast调用次数: {stats['broadcast_count']}")
        print(f"  Broadcast总时间: {stats['broadcast_time']:.3f}s")
        if stats['broadcast_count'] > 0:
            print(f"  Broadcast平均时间: {stats['broadcast_time']/stats['broadcast_count']*1000:.2f}ms")
    
    @property
    def is_main_process(self) -> bool:
        """是否为主进程"""
        return self.rank == 0
    
    def get_world_size(self) -> int:
        """获取世界大小"""
        return self.world_size
    
    def get_rank(self) -> int:
        """获取当前排名"""
        return self.rank


def get_world_size() -> int:
    """获取世界大小（全局函数）"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """获取当前排名（全局函数）"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def find_free_port() -> int:
    """查找可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed_environment(master_addr: str = "localhost",
                                 master_port: Optional[int] = None,
                                 backend: str = "nccl") -> DistributedManager:
    """设置分布式环境的便捷函数"""
    
    # 设置环境变量
    if master_port is None:
        master_port = find_free_port()
    
    os.environ.setdefault('MASTER_ADDR', master_addr)
    os.environ.setdefault('MASTER_PORT', str(master_port))
    
    # 创建分布式管理器
    dist_manager = DistributedManager(backend=backend)
    
    # 初始化
    if dist_manager.setup():
        return dist_manager
    else:
        raise RuntimeError("分布式环境初始化失败")


class DistributedMetrics:
    """分布式指标聚合器"""
    
    def __init__(self, dist_manager: DistributedManager):
        self.dist_manager = dist_manager
        self.metrics_buffer = {}
    
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []
            self.metrics_buffer[key].append(value)
    
    def compute_and_reduce(self) -> Dict[str, float]:
        """计算并归约指标"""
        reduced_metrics = {}
        
        for key, values in self.metrics_buffer.items():
            if values:
                # 计算本地平均值
                local_avg = sum(values) / len(values)
                local_tensor = torch.tensor(local_avg, device=self.dist_manager.device)
                
                # 全局归约
                global_tensor = self.dist_manager.all_reduce(local_tensor)
                global_avg = global_tensor.item() / self.dist_manager.world_size
                
                reduced_metrics[key] = global_avg
        
        # 清空缓冲区
        self.metrics_buffer.clear()
        
        return reduced_metrics


def launch_distributed_training(train_fn, 
                               world_size: int,
                               master_addr: str = "localhost",
                               master_port: Optional[int] = None,
                               backend: str = "nccl",
                               **kwargs):
    """启动分布式训练的便捷函数"""
    
    if master_port is None:
        master_port = find_free_port()
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # 使用torch.multiprocessing启动
    import torch.multiprocessing as mp
    
    def worker(rank, world_size, train_fn, **kwargs):
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        
        # 创建分布式管理器
        dist_manager = DistributedManager(backend=backend)
        
        try:
            if dist_manager.setup():
                # 调用训练函数
                train_fn(dist_manager, **kwargs)
            else:
                print(f"Rank {rank}: 分布式初始化失败")
        finally:
            dist_manager.cleanup()
    
    # 启动多进程
    mp.spawn(worker, args=(world_size, train_fn), kwargs=kwargs, nprocs=world_size, join=True)


# 使用示例和测试函数
def test_distributed_setup():
    """测试分布式设置"""
    print("🧪 测试分布式环境设置...")
    
    # 创建分布式管理器
    dist_manager = DistributedManager()
    
    try:
        # 初始化
        if dist_manager.setup():
            print("✓ 分布式环境初始化成功")
            
            # 测试基本功能
            test_tensor = torch.randn(10, device=dist_manager.device)
            
            # 测试全局归约
            reduced_tensor = dist_manager.all_reduce(test_tensor.clone())
            print(f"✓ AllReduce测试完成，张量形状: {reduced_tensor.shape}")
            
            # 测试广播
            broadcast_tensor = dist_manager.broadcast(test_tensor.clone())
            print(f"✓ Broadcast测试完成，张量形状: {broadcast_tensor.shape}")
            
            # 测试指标归约
            metrics = {'loss': 1.5, 'accuracy': 0.85}
            tensor_metrics = {k: torch.tensor(v, device=dist_manager.device) for k, v in metrics.items()}
            reduced_metrics = dist_manager.reduce_dict(tensor_metrics)
            print(f"✓ 指标归约测试完成: {reduced_metrics}")
            
            print("✅ 所有分布式功能测试通过")
            
        else:
            print("❌ 分布式环境初始化失败")
            
    finally:
        dist_manager.cleanup()


if __name__ == "__main__":
    test_distributed_setup()