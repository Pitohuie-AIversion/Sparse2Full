"""
单元测试：SwinTemporalNAR模型
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json

# 导入测试模块
from src.models.swin_temporal_nar import (
    SwinTemporalNAR, SwinTemporalConfig, 
    PatchEmbedding, WindowAttention, SwinTransformerBlock,
    TemporalSwinBlock, SwinTemporalStage, SwinTemporalEncoder,
    TemporalNARHead
)

class TestSwinTemporalConfig(unittest.TestCase):
    """SwinTemporal配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SwinTemporalConfig()
        
        self.assertEqual(config.input_channels, 1)
        self.assertEqual(config.hidden_dim, 96)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.window_size, 7)
        self.assertEqual(config.time_steps, 10)
        self.assertEqual(config.prediction_steps, 5)
        self.assertEqual(config.spatial_resolution, (64, 64))
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SwinTemporalConfig(
            input_channels=3,
            hidden_dim=128,
            num_layers=6,
            num_heads=16,
            window_size=8,
            time_steps=20,
            prediction_steps=10,
            spatial_resolution=(128, 128)
        )
        
        self.assertEqual(config.input_channels, 3)
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.window_size, 8)
        self.assertEqual(config.time_steps, 20)
        self.assertEqual(config.prediction_steps, 10)
        self.assertEqual(config.spatial_resolution, (128, 128))
    
    def test_config_serialization(self):
        """测试配置序列化"""
        config = SwinTemporalConfig(
            input_channels=2,
            hidden_dim=64,
            num_layers=2
        )
        
        # 转换为字典
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['input_channels'], 2)
        self.assertEqual(config_dict['hidden_dim'], 64)
        
        # 从字典创建
        new_config = SwinTemporalConfig.from_dict(config_dict)
        self.assertEqual(new_config.input_channels, 2)
        self.assertEqual(new_config.hidden_dim, 64)
        self.assertEqual(new_config.num_layers, 2)

class TestPatchEmbedding(unittest.TestCase):
    """PatchEmbedding模块测试"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.time_steps = 10
        self.channels = 3
        self.height = 64
        self.width = 64
        self.patch_size = 4
        self.embed_dim = 96
        
        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建测试输入 (B, T, C, H, W)
        x = torch.randn(self.batch_size, self.time_steps, self.channels, self.height, self.width)
        
        # 前向传播
        output = self.patch_embedding(x)
        
        # 验证输出形状
        expected_height = self.height // self.patch_size
        expected_width = self.width // self.patch_size
        
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.time_steps)
        self.assertEqual(output.shape[2], expected_height * expected_width)
        self.assertEqual(output.shape[3], self.embed_dim)
    
    def test_patch_size_compatibility(self):
        """测试不同patch size的兼容性"""
        patch_sizes = [2, 4, 8]
        
        for patch_size in patch_sizes:
            patch_embedding = PatchEmbedding(
                patch_size=patch_size,
                embed_dim=self.embed_dim
            )
            
            # 调整输入大小以确保能被patch size整除
            adjusted_height = (self.height // patch_size) * patch_size
            adjusted_width = (self.width // patch_size) * patch_size
            
            x = torch.randn(self.batch_size, self.time_steps, self.channels, adjusted_height, adjusted_width)
            output = patch_embedding(x)
            
            expected_patches = (adjusted_height // patch_size) * (adjusted_width // patch_size)
            self.assertEqual(output.shape[2], expected_patches)

class TestWindowAttention(unittest.TestCase):
    """WindowAttention模块测试"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.time_steps = 10
        self.num_patches = 16
        self.embed_dim = 96
        self.num_heads = 8
        self.window_size = 7
        
        self.attention = WindowAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size
        )
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建测试输入 (B, T, N, C)
        x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
        
        # 前向传播
        output = self.attention(x)
        
        # 验证输出形状
        self.assertEqual(output.shape, x.shape)
    
    def test_attention_weights(self):
        """测试注意力权重"""
        x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
        
        # 前向传播并获取注意力权重
        output, attention_weights = self.attention(x, return_attention=True)
        
        # 验证注意力权重形状
        expected_shape = (self.batch_size, self.time_steps, self.num_heads, self.num_patches, self.num_patches)
        self.assertEqual(attention_weights.shape, expected_shape)
        
        # 验证注意力权重是有效的概率分布
        attention_sum = attention_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5))

class TestSwinTransformerBlock(unittest.TestCase):
    """SwinTransformerBlock模块测试"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.time_steps = 10
        self.num_patches = 16
        self.embed_dim = 96
        self.num_heads = 8
        self.window_size = 7
        self.mlp_ratio = 4.0
        
        self.block = SwinTransformerBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio
        )
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建测试输入 (B, T, N, C)
        x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
        
        # 前向传播
        output = self.block(x)
        
        # 验证输出形状
        self.assertEqual(output.shape, x.shape)
    
    def test_residual_connection(self):
        """测试残差连接"""
        x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
        
        # 前向传播
        output = self.block(x)
        
        # 残差连接应该保持形状不变
        self.assertEqual(output.shape, x.shape)

class TestTemporalSwinBlock(unittest.TestCase):
    """TemporalSwinBlock模块测试"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.time_steps = 10
        self.num_patches = 16
        self.embed_dim = 96
        self.num_heads = 8
        self.window_size = 7
        self.temporal_encoding_type = 'sinusoidal'
        
        self.temporal_block = TemporalSwinBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            temporal_encoding_type=self.temporal_encoding_type
        )
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建测试输入 (B, T, N, C)
        x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
        
        # 前向传播
        output = self.temporal_block(x)
        
        # 验证输出形状
        self.assertEqual(output.shape, x.shape)
    
    def test_different_temporal_encodings(self):
        """测试不同的时间编码类型"""
        encoding_types = ['sinusoidal', 'learnable', 'relative', 'rope']
        
        for encoding_type in encoding_types:
            temporal_block = TemporalSwinBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                temporal_encoding_type=encoding_type
            )
            
            x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
            output = temporal_block(x)
            
            # 验证输出形状
            self.assertEqual(output.shape, x.shape)

class TestSwinTemporalStage(unittest.TestCase):
    """SwinTemporalStage模块测试"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.time_steps = 10
        self.num_patches = 16
        self.embed_dim = 96
        self.num_heads = 8
        self.window_size = 7
        self.depth = 2
        self.downsample = False
        
        self.stage = SwinTemporalStage(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            depth=self.depth,
            downsample=self.downsample
        )
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建测试输入 (B, T, N, C)
        x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
        
        # 前向传播
        output = self.stage(x)
        
        # 验证输出形状
        if self.downsample:
            # 如果下采样，patch数量应该减少
            self.assertEqual(output.shape[0], self.batch_size)
            self.assertEqual(output.shape[1], self.time_steps)
            self.assertEqual(output.shape[3], self.embed_dim * 2)  # 通道数翻倍
        else:
            self.assertEqual(output.shape, x.shape)
    
    def test_downsample_functionality(self):
        """测试下采样功能"""
        stage_with_downsample = SwinTemporalStage(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            depth=self.depth,
            downsample=True
        )
        
        x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
        output = stage_with_downsample(x)
        
        # 验证下采样后的特征维度
        self.assertEqual(output.shape[3], self.embed_dim * 2)

class TestSwinTemporalEncoder(unittest.TestCase):
    """SwinTemporalEncoder模块测试"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.time_steps = 10
        self.channels = 3
        self.height = 64
        self.width = 64
        self.embed_dim = 96
        self.num_heads = [8, 16, 32, 64]
        self.depths = [2, 2, 6, 2]
        self.window_size = 7
        self.patch_size = 4
        
        self.encoder = SwinTemporalEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            depths=self.depths,
            window_size=self.window_size,
            patch_size=self.patch_size
        )
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建测试输入 (B, T, C, H, W)
        x = torch.randn(self.batch_size, self.time_steps, self.channels, self.height, self.width)
        
        # 前向传播
        output = self.encoder(x)
        
        # 验证输出形状
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.time_steps)
        self.assertEqual(output.shape[2], self.embed_dim * 8)  # 最终的特征维度
    
    def test_feature_extraction(self):
        """测试特征提取功能"""
        x = torch.randn(self.batch_size, self.time_steps, self.channels, self.height, self.width)
        
        # 获取多尺度特征
        features = self.encoder.forward_features(x)
        
        # 验证特征数量
        self.assertEqual(len(features), 4)  # 4个阶段的特征
        
        # 验证每个阶段的特征维度
        expected_dims = [self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8]
        for i, (feature, expected_dim) in enumerate(zip(features, expected_dims)):
            self.assertEqual(feature.shape[-1], expected_dim)

class TestTemporalNARHead(unittest.TestCase):
    """TemporalNARHead模块测试"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.time_steps = 10
        self.num_patches = 16
        self.embed_dim = 96
        self.prediction_steps = 5
        self.output_channels = 1
        self.patch_size = 4
        self.spatial_resolution = (64, 64)
        
        self.head = TemporalNARHead(
            embed_dim=self.embed_dim,
            prediction_steps=self.prediction_steps,
            output_channels=self.output_channels,
            patch_size=self.patch_size,
            spatial_resolution=self.spatial_resolution
        )
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建测试输入 (B, T, N, C)
        x = torch.randn(self.batch_size, self.time_steps, self.num_patches, self.embed_dim)
        
        # 前向传播
        output = self.head(x)
        
        # 验证输出形状 (B, prediction_steps, output_channels, H, W)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.prediction_steps)
        self.assertEqual(output.shape[2], self.output_channels)
        self.assertEqual(output.shape[3], self.spatial_resolution[0])
        self.assertEqual(output.shape[4], self.spatial_resolution[1])

class TestSwinTemporalNAR(unittest.TestCase):
    """SwinTemporalNAR模型测试"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.time_steps = 10
        self.prediction_steps = 5
        self.channels = 3
        self.height = 64
        self.width = 64
        
        self.config = SwinTemporalConfig(
            input_channels=self.channels,
            time_steps=self.time_steps,
            prediction_steps=self.prediction_steps,
            spatial_resolution=(self.height, self.width)
        )
        
        self.model = SwinTemporalNAR(self.config)
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建测试输入 (B, T, C, H, W)
        x = torch.randn(self.batch_size, self.time_steps, self.channels, self.height, self.width)
        
        # 前向传播
        output = self.model(x)
        
        # 验证输出形状 (B, prediction_steps, C, H, W)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.prediction_steps)
        self.assertEqual(output.shape[2], self.channels)
        self.assertEqual(output.shape[3], self.height)
        self.assertEqual(output.shape[4], self.width)
    
    def test_different_input_sizes(self):
        """测试不同输入大小的兼容性"""
        input_sizes = [(32, 32), (64, 64), (128, 128)]
        
        for height, width in input_sizes:
            config = SwinTemporalConfig(
                input_channels=self.channels,
                time_steps=self.time_steps,
                prediction_steps=self.prediction_steps,
                spatial_resolution=(height, width)
            )
            
            model = SwinTemporalNAR(config)
            
            x = torch.randn(self.batch_size, self.time_steps, self.channels, height, width)
            output = model(x)
            
            # 验证输出形状
            self.assertEqual(output.shape[3], height)
            self.assertEqual(output.shape[4], width)
    
    def test_model_parameters(self):
        """测试模型参数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        self.assertGreater(total_params, 0)
        self.assertIsInstance(total_params, int)
        
        # 验证可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total_params, trainable_params)
    
    def test_gradient_flow(self):
        """测试梯度流"""
        x = torch.randn(self.batch_size, self.time_steps, self.channels, self.height, self.width)
        target = torch.randn(self.batch_size, self.prediction_steps, self.channels, self.height, self.width)
        
        # 前向传播
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        
        # 反向传播
        loss.backward()
        
        # 验证梯度存在
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存模型
            save_path = Path(temp_dir) / 'test_model.pth'
            self.model.save_model(str(save_path))
            
            self.assertTrue(save_path.exists())
            
            # 创建新模型并加载
            new_model = SwinTemporalNAR(self.config)
            new_model.load_model(str(save_path))
            
            # 验证参数一致性
            for (name1, param1), (name2, param2) in zip(
                self.model.named_parameters(), new_model.named_parameters()
            ):
                self.assertTrue(torch.allclose(param1, param2))
    
    def test_model_info(self):
        """测试模型信息"""
        info = self.model.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('total_parameters', info)
        self.assertIn('trainable_parameters', info)
        self.assertIn('model_size_mb', info)
        self.assertIn('config', info)
        
        self.assertGreater(info['total_parameters'], 0)
        self.assertGreater(info['model_size_mb'], 0)

if __name__ == '__main__':
    # 设置测试日志
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # 运行测试
    unittest.main(verbosity=2)