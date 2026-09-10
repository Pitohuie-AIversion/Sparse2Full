#!/usr/bin/env python3
"""
简单模型切换演示脚本
直接使用我们测试通过的核心模型进行训练和验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import logging
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入我们测试过的模型
from models import UNet, SwinUNet

class SimpleTrainer:
    """简化版训练器，专注于模型切换演示"""
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model()
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        
    def _create_model(self):
        """创建模型实例"""
        logger.info(f"创建 {self.model_name} 模型...")
        
        if self.model_name == 'UNet':
            model = UNet(
                in_channels=1,
                out_channels=1, 
                img_size=128,
                features=[64, 128, 256, 512],
                bilinear=True
            )
        elif self.model_name == 'SwinUNet':
            model = SwinUNet(
                in_chans=1,
                num_classes=1,
                img_size=128,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=8
            )
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
            
        model = model.to(self.device)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"模型创建完成！参数量: {param_count:,}")
        
        return model
    
    def _create_optimizer(self, lr: float = 1e-3):
        """创建优化器"""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
    
    def _generate_synthetic_data(self, num_samples: int = 100):
        """生成合成训练数据"""
        logger.info("生成合成训练数据...")
        
        # 创建简单的2D函数作为训练数据
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        
        data = []
        for i in range(num_samples):
            # 生成不同的波动模式
            freq_x = np.random.uniform(1, 3)
            freq_y = np.random.uniform(1, 3)
            phase = np.random.uniform(0, 2*np.pi)
            
            wave = np.sin(freq_x * X) * np.cos(freq_y * Y + phase)
            # 添加一些噪声
            noise = np.random.normal(0, 0.1, wave.shape)
            wave += noise
            
            data.append(wave[None, :, :])  # 添加通道维度
            
        data = np.array(data)
        return torch.FloatTensor(data)
    
    def train_epoch(self, data_loader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, data in enumerate(data_loader):
            # 输入和目标相同（自编码器模式）
            inputs = data.to(self.device)
            targets = data.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        self.scheduler.step()
        
        return avg_loss
    
    def validate(self, data_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data in data_loader:
                inputs = data.to(self.device)
                targets = data.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, epochs: int = 5, batch_size: int = 4):
        """完整训练流程"""
        logger.info(f"开始训练 {self.model_name}， epochs: {epochs}, batch_size: {batch_size}")
        
        # 创建优化器
        self._create_optimizer()
        
        # 生成数据
        train_data = self._generate_synthetic_data(100)
        val_data = self._generate_synthetic_data(20)
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # 训练循环
        best_val_loss = float('inf')
        training_history = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 记录历史
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            logger.info(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.scheduler.get_last_lr()[0]:.2e}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
        
        training_time = time.time() - start_time
        
        logger.info(f"\n{'='*50}")
        logger.info(f"训练完成！总耗时: {training_time:.2f}秒")
        logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        
        return training_history
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_name': self.model_name
        }
        
        checkpoint_path = f"runs/{self.model_name}_best_checkpoint.pth"
        Path(checkpoint_path).parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存最佳模型: {checkpoint_path}")
    
    def test_inference_speed(self, num_runs: int = 100):
        """测试推理速度"""
        logger.info(f"测试 {self.model_name} 推理速度...")
        
        self.model.eval()
        dummy_input = torch.randn(1, 1, 128, 128).to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # 正式测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        logger.info(f"平均推理时间: {avg_time*1000:.2f}ms")
        logger.info(f"推理速度: {fps:.1f} FPS")
        
        return avg_time, fps

def compare_models():
    """比较不同模型的性能"""
    logger.info("开始模型性能对比测试...")
    
    models = ['UNet', 'SwinUNet']
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试模型: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # 创建训练器
            trainer = SimpleTrainer(model_name)
            
            # 测试推理速度
            inference_time, fps = trainer.test_inference_speed()
            
            # 快速训练测试
            logger.info(f"开始快速训练测试...")
            history = trainer.train(epochs=3, batch_size=2)
            
            # 记录结果
            results[model_name] = {
                'inference_time_ms': inference_time * 1000,
                'fps': fps,
                'final_train_loss': history[-1]['train_loss'],
                'final_val_loss': history[-1]['val_loss'],
                'param_count': sum(p.numel() for p in trainer.model.parameters())
            }
            
            logger.info(f"✅ {model_name} 测试完成！")
            
        except Exception as e:
            logger.error(f"❌ {model_name} 测试失败: {e}")
            results[model_name] = {'error': str(e)}
    
    # 打印对比结果
    logger.info(f"\n{'='*60}")
    logger.info("模型性能对比结果")
    logger.info(f"{'='*60}")
    
    for model_name, result in results.items():
        if 'error' in result:
            logger.info(f"{model_name}: 测试失败 - {result['error']}")
        else:
            logger.info(f"\n{model_name}:")
            logger.info(f"  参数量: {result['param_count']:,}")
            logger.info(f"  推理时间: {result['inference_time_ms']:.2f}ms")
            logger.info(f"  推理速度: {result['fps']:.1f} FPS")
            logger.info(f"  最终训练损失: {result['final_train_loss']:.6f}")
            logger.info(f"  最终验证损失: {result['final_val_loss']:.6f}")
    
    return results

def main():
    """主函数"""
    print("🚀 简单模型切换演示")
    print("="*60)
    
    # 运行模型对比测试
    results = compare_models()
    
    print("\n✅ 模型切换演示完成！")
    print("您现在可以根据这些测试结果选择合适的模型进行正式训练。")

if __name__ == "__main__":
    main()
