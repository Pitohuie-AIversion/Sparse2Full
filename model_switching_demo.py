#!/usr/bin/env python3
"""
模型切换训练演示
展示如何在UNet和SwinUNet之间切换进行训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Dict, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 简化的数据集类
class SyntheticDataset(Dataset):
    """合成PDE数据用于演示"""
    def __init__(self, num_samples=100, image_size=128, channels=1):
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels
        
        # 生成合成数据
        np.random.seed(42)
        self.data = []
        for i in range(num_samples):
            # 创建简单的波动模式
            x = np.linspace(0, 4*np.pi, image_size)
            y = np.linspace(0, 4*np.pi, image_size)
            X, Y = np.meshgrid(x, y)
            
            # 添加一些随机频率和相位
            freq_x = np.random.uniform(1, 3)
            freq_y = np.random.uniform(1, 3)
            phase = np.random.uniform(0, 2*np.pi)
            
            field = np.sin(freq_x * X + phase) * np.cos(freq_y * Y + phase)
            field = field[np.newaxis, :, :]  # 添加通道维度
            
            self.data.append(torch.FloatTensor(field))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # 输入和目标相同

# 简化的UNet模型
class SimpleUNet(nn.Module):
    """简化的UNet模型用于演示"""
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()
        self.encoder1 = self._conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._conv_block(features, features*2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self._conv_block(features*2, features*4)
        
        self.decoder3 = nn.ConvTranspose2d(features*4, features*2, 2, 2)
        self.decoder2 = nn.ConvTranspose2d(features*2, features, 2, 2)
        self.final = nn.Conv2d(features, out_channels, 1)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # 解码器
        dec3 = self.decoder3(enc3)
        dec2 = self.decoder2(dec3 + enc2)
        out = self.final(dec2 + enc1)
        
        return out

# 训练配置
@dataclass
class TrainingConfig:
    model_name: str = "unet"
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-3
    image_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples: int = 100

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建数据集
        self.train_dataset = SyntheticDataset(
            num_samples=config.num_samples,
            image_size=config.image_size
        )
        self.val_dataset = SyntheticDataset(
            num_samples=20,
            image_size=config.image_size
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False
        )
        
        # 创建模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def _create_model(self):
        """根据配置创建模型"""
        if self.config.model_name.lower() == "unet":
            logger.info("创建UNet模型")
            return SimpleUNet(in_channels=1, out_channels=1)
        elif self.config.model_name.lower() == "swin_unet":
            logger.info("创建SwinUNet模型")
            # 这里使用我们测试过的SwinUNet模型
            from models.swin_unet import SwinUNet
            return SwinUNet(
                in_chans=1,
                out_chans=1,
                img_size=self.config.image_size,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                window_size=8,
                patch_size=4,
                in_channels=1  # 显式设置输入通道数
            )
        else:
            raise ValueError(f"不支持的模型: {self.config.model_name}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"批次 {batch_idx}/{len(self.train_loader)}, 损失: {loss.item():.6f}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """完整训练流程"""
        logger.info(f"开始训练 {self.config.model_name} 模型")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练样本: {len(self.train_dataset)}")
        logger.info(f"验证样本: {len(self.val_dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            logger.info(f"  训练损失: {train_loss:.6f}")
            logger.info(f"  验证损失: {val_loss:.6f}")
            logger.info(f"  学习率: {self.scheduler.get_last_lr()[0]:.6f}")
            logger.info(f"  用时: {epoch_time:.2f}s")
            
            # 检查点
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"runs/{self.config.model_name}_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.info(f"保存检查点: {checkpoint_path}")
        
        total_time = time.time() - start_time
        logger.info(f"训练完成！总用时: {total_time:.2f}s")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'total_time': total_time,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1]
        }

def compare_models():
    """比较不同模型的训练性能"""
    
    # 确保输出目录存在
    os.makedirs("runs", exist_ok=True)
    
    results = {}
    
    # 测试的模型
    models_to_test = ["unet", "swin_unet"]
    
    for model_name in models_to_test:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试模型: {model_name}")
        logger.info(f"{'='*50}")
        
        config = TrainingConfig(
            model_name=model_name,
            epochs=5,
            batch_size=4,
            learning_rate=1e-3
        )
        
        try:
            trainer = ModelTrainer(config)
            result = trainer.train()
            results[model_name] = result
            
            logger.info(f"✅ {model_name} 训练成功!")
            logger.info(f"最终训练损失: {result['final_train_loss']:.6f}")
            logger.info(f"最终验证损失: {result['final_val_loss']:.6f}")
            
        except Exception as e:
            logger.error(f"❌ {model_name} 训练失败: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    # 生成对比报告
    logger.info(f"\n{'='*60}")
    logger.info("模型性能对比报告")
    logger.info(f"{'='*60}")
    
    for model_name, result in results.items():
        if "error" in result:
            logger.info(f"{model_name}: 训练失败 - {result['error']}")
        else:
            logger.info(f"{model_name}:")
            logger.info(f"  最终训练损失: {result['final_train_loss']:.6f}")
            logger.info(f"  最终验证损失: {result['final_val_loss']:.6f}")
            logger.info(f"  总训练时间: {result['total_time']:.2f}s")
    
    return results

if __name__ == "__main__":
    logger.info("🚀 开始模型切换训练演示")
    results = compare_models()
    logger.info("✅ 演示完成！")