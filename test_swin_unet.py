#!/usr/bin/env python3
"""
单独测试SwinUNet模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入模型
from models import SwinUNet

def test_swin_unet():
    """测试SwinUNet模型"""
    logger.info("测试SwinUNet模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SwinUNet(
        in_channels=1,
        out_channels=1,
        img_size=128,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        patch_size=4,
        embed_dim=96
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"SwinUNet参数量: {param_count:,}")
    
    # 测试前向传播
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    logger.info(f"输入形状: {dummy_input.shape}")
    logger.info(f"输出形状: {output.shape}")
    
    # 测试训练
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    target = torch.randn_like(output)
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    logger.info(f"训练测试 - 损失: {loss.item():.6f}")
    logger.info("✅ SwinUNet测试通过！")
    
    # 测试推理速度
    model.eval()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    avg_time = (time.time() - start_time) / 50
    
    logger.info(f"平均推理时间: {avg_time*1000:.2f}ms")
    logger.info(f"推理速度: {1.0/avg_time:.1f} FPS")

if __name__ == "__main__":
    test_swin_unet()