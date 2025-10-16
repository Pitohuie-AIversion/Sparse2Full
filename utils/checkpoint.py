"""检查点管理模块

提供模型检查点的保存、加载和管理功能
支持最佳模型保存和检查点清理
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


class CheckpointManager:
    """检查点管理器
    
    负责模型检查点的保存、加载和管理
    """
    
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5, 
                 save_best: bool = True):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保存检查点数量
            save_best: 是否保存最佳模型
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], is_best: bool = False, 
                       epoch: Optional[int] = None) -> str:
        """保存检查点
        
        Args:
            checkpoint: 检查点数据
            is_best: 是否为最佳模型
            epoch: 训练轮次
            
        Returns:
            checkpoint_path: 保存的检查点路径
        """
        # 生成检查点文件名
        if epoch is not None:
            checkpoint_name = f'checkpoint_epoch_{epoch:04d}.pth'
        else:
            checkpoint_name = 'checkpoint_latest.pth'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # 保存最佳模型
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / 'best.pth'
            shutil.copy2(checkpoint_path, best_path)
            self.logger.info(f"Best model saved: {best_path}")
        
        # 清理旧检查点
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            device: 目标设备
            
        Returns:
            checkpoint: 检查点数据
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint
    
    def _cleanup_checkpoints(self) -> None:
        """清理旧检查点"""
        if self.max_checkpoints <= 0:
            return
        
        # 获取所有检查点文件
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # 按修改时间排序
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        
        # 删除最旧的检查点
        files_to_delete = checkpoint_files[:-self.max_checkpoints]
        for file_path in files_to_delete:
            file_path.unlink()
            self.logger.info(f"Old checkpoint deleted: {file_path}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点路径"""
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        if latest_path.exists():
            return str(latest_path)
        
        # 查找最新的epoch检查点
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoint_files:
            latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            return str(latest_file)
        
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点路径"""
        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            return str(best_path)
        return None
    
    def list_checkpoints(self) -> List[str]:
        """列出所有检查点"""
        checkpoint_files = list(self.checkpoint_dir.glob('*.pth'))
        return [str(f) for f in sorted(checkpoint_files)]


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """便捷函数：加载检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def save_checkpoint(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """便捷函数：保存检查点"""
    # 确保目录存在
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path)