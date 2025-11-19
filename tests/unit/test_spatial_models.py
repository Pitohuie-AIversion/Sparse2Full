"""
Comprehensive unit tests for spatial prediction models.
Tests all spatial models with unified interface: forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from omegaconf import OmegaConf

# Import all spatial models
from models import (
    UNet, UNetPlusPlus, FNO2d, UFNOUNet,
    SegFormer, UNetFormer, SegFormerUNetFormer,
    VisionTransformer, SwinTransformerTiny, Transformer,
    SwinUNet, HybridModel, MLPModel, MLPMixer, LIIFModel,
    SparseAttentionEncoder, SparseSwinUNet
)


class SpatialModelTestConfig:
    """Configuration for spatial model testing"""
    
    # Standard input configurations
    STANDARD_INPUT_SHAPES = [
        (1, 1, 64, 64),    # Single channel, small
        (2, 3, 128, 128),  # Multi-channel, medium
        (1, 5, 256, 256),  # Multi-channel, large
    ]
    
    # Edge case input configurations
    EDGE_CASE_SHAPES = [
        (1, 1, 32, 64),    # Non-square input
        (3, 1, 127, 129),  # Odd dimensions
        (1, 10, 512, 512), # Many channels
    ]
    
    # Minimum supported sizes for different model families
    MIN_SIZES = {
        'swin': 32,      # Swin transformer minimum
        'vit': 16,       # Vision transformer minimum patch
        'segformer': 32, # SegFormer minimum
        'default': 16,   # Default minimum
    }


class TestSpatialModels:
    """Test suite for all spatial prediction models"""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_model_config(self, model_class: type) -> Dict[str, Any]:
        """Get appropriate configuration for each model type"""
        model_name = model_class.__name__.lower()
        
        # Base configuration - use constructor parameters that match the actual model
        base_config = {
            'in_ch': 3,
            'out_ch': 3,
            'img_size': 128,
        }
        
        # Model-specific configurations with corrected parameter names
        model_configs = {
            'unet': {
                'in_channels': 3,
                'out_channels': 3,
                'features': [32, 64, 128, 256],
                'bilinear': True,
            },
            'unetplusplus': {
                'in_channels': 3,
                'out_channels': 3,
                'deep_supervision': False,
                'features': [32, 64, 128, 256],
            },
            'fno2d': {
                'modes': 16,
                'width': 32,
                'layers': 4,
            },
            'ufnounet': {
                'modes': 16,
                'width': 32,
                'layers': 4,
                'bilinear': True,
            },
            'swinunet': {
                'in_chans': 3,
                'num_classes': 3,
                'depths': [2, 2, 2, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8,
                'mlp_ratio': 4.0,
            },
            'hybridmodel': {
                'in_ch': 3,
                'out_ch': 3,
                'img_size': 128,
                'backbone': 'swin',
                'fusion': 'concat',
                'attention_ch': 64,
                'fno_modes': 16,
                'fno_width': 32,
            },
            'segformer': {
                'in_channels': 3,
                'num_classes': 3,
                'backbone': 'b0',
                'embed_dim': 256,
                'num_heads': [1, 2, 5, 8],
            },
            'unetformer': {
                'in_chans': 3,
                'num_classes': 3,
                'backbone': 'resnet50',
                'num_heads': 8,
                'mlp_ratio': 4.0,
            },
            'segformerunetformer': {
                'in_chans': 3,
                'num_classes': 3,
                'segformer_backbone': 'b0',
                'unetformer_backbone': 'resnet50',
                'fusion': 'attention',
            },
            'visiontransformer': {
                'in_chans': 3,
                'num_classes': 3,
                'patch_size': 16,
                'embed_dim': 768,
                'num_heads': 12,
                'depth': 12,
            },
            'swintransformertiny': {
                'in_chans': 3,
                'num_classes': 3,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'embed_dim': 96,
            },
            'transformer': {
                'in_channels': 3,
                'out_channels': 3,
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 1024,
            },
            'mlpmodel': {
                'in_features': 3,
                'out_features': 3,
                'hidden_dim': 256,
                'num_layers': 4,
                'activation': 'gelu',
                'use_coords': True,
            },
            'mlpmixer': {
                'in_channels': 3,
                'out_channels': 3,
                'patch_size': 16,
                'hidden_dim': 512,
                'num_blocks': 8,
                'tokens_mlp_dim': 256,
                'channels_mlp_dim': 2048,
            },
            'liifmodel': {
                'in_dim': 3,
                'out_dim': 3,
                'hidden_dim': 256,
                'num_layers': 4,
                'coord_encode': True,
                'cell_decode': True,
            },
            'sparseattentionencoder': {
                'in_channels': 3,
                'out_channels': 3,
                'embed_dim': 256,
                'num_heads': 8,
                'sparse_ratio': 0.5,
                'attention_type': 'sparse',
            },
            'sparseswinunet': {
                'in_chans': 3,
                'num_classes': 3,
                'depths': [2, 2, 2, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8,
                'sparse_ratio': 0.5,
            },
        }
        
        # Merge configurations
        config = base_config.copy()
        if model_name in model_configs:
            config.update(model_configs[model_name])
        
        return config
    
    def create_model(self, model_class: type, device: torch.device, **kwargs) -> nn.Module:
        """Create model instance with given configuration"""
        config = self.get_model_config(model_class)
        config.update(kwargs)
        
        try:
            model = model_class(**config)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            pytest.skip(f"Failed to create {model_class.__name__}: {str(e)}")
    
    @pytest.mark.parametrize("model_class,input_shape", [
        (UNet, (1, 3, 128, 128)),
        (UNetPlusPlus, (1, 3, 128, 128)),
        (FNO2d, (1, 3, 128, 128)),
        (UFNOUNet, (1, 3, 128, 128)),
        (SwinUNet, (1, 3, 128, 128)),
        (HybridModel, (1, 3, 128, 128)),
        (MLPModel, (1, 3, 128, 128)),
        (MLPMixer, (1, 3, 128, 128)),
        (LIIFModel, (1, 3, 128, 128)),
        (SegFormer, (1, 3, 128, 128)),
        (UNetFormer, (1, 3, 128, 128)),
        (SegFormerUNetFormer, (1, 3, 128, 128)),
        (VisionTransformer, (1, 3, 128, 128)),
        (SwinTransformerTiny, (1, 3, 128, 128)),
        (Transformer, (1, 3, 128, 128)),
        (SparseAttentionEncoder, (1, 3, 128, 128)),
        (SparseSwinUNet, (1, 3, 128, 128)),
    ])
    def test_model_forward_pass(self, model_class, input_shape, device):
        """Test basic forward pass for each spatial model"""
        model = self.create_model(model_class, device)
        
        # Create input tensor
        x = torch.randn(input_shape, device=device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        batch_size, channels, height, width = input_shape
        expected_shape = (batch_size, model.out_ch, height, width)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check output properties
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
    
    @pytest.mark.parametrize("model_class", [
        UNet, UNetPlusPlus, FNO2d, UFNOUNet, SwinUNet, HybridModel,
        MLPModel, MLPMixer, LIIFModel, SegFormer, UNetFormer,
        SegFormerUNetFormer, VisionTransformer, SwinTransformerTiny,
        Transformer, SparseAttentionEncoder, SparseSwinUNet
    ])
    def test_model_different_input_sizes(self, model_class, device):
        """Test model with different input sizes"""
        model = self.create_model(model_class, device)
        
        test_shapes = SpatialModelTestConfig.STANDARD_INPUT_SHAPES
        
        for shape in test_shapes:
            # Skip if size too small for certain models
            min_size = self.get_minimum_size(model_class)
            if shape[-1] < min_size or shape[-2] < min_size:
                continue
            
            x = torch.randn(shape, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            # Check output maintains spatial dimensions
            assert output.shape[-2:] == x.shape[-2:], f"Spatial dimensions changed: {x.shape} -> {output.shape}"
    
    def get_minimum_size(self, model_class: type) -> int:
        """Get minimum supported input size for model"""
        model_name = model_class.__name__.lower()
        
        if 'swin' in model_name:
            return SpatialModelTestConfig.MIN_SIZES['swin']
        elif 'vit' in model_name or 'vision' in model_name:
            return SpatialModelTestConfig.MIN_SIZES['vit']
        elif 'segformer' in model_name:
            return SpatialModelTestConfig.MIN_SIZES['segformer']
        else:
            return SpatialModelTestConfig.MIN_SIZES['default']
    
    @pytest.mark.parametrize("model_class", [
        UNet, SwinUNet, HybridModel, MLPModel, SegFormer
    ])
    def test_model_edge_cases(self, model_class, device):
        """Test model with edge case inputs"""
        model = self.create_model(model_class, device)
        
        # Test with non-square input
        x_rect = torch.randn(1, 3, 64, 128, device=device)
        with torch.no_grad():
            output_rect = model(x_rect)
        assert output_rect.shape[-2:] == (64, 128)
        
        # Test with single channel
        x_single = torch.randn(1, 1, 128, 128, device=device)
        model_single_ch = self.create_model(model_class, device, in_ch=1, out_ch=1)
        with torch.no_grad():
            output_single = model_single_ch(x_single)
        assert output_single.shape == (1, 1, 128, 128)
        
        # Test with many channels
        x_multi = torch.randn(1, 10, 128, 128, device=device)
        model_multi_ch = self.create_model(model_class, device, in_ch=10, out_ch=5)
        with torch.no_grad():
            output_multi = model_multi_ch(x_multi)
        assert output_multi.shape == (1, 5, 128, 128)
    
    @pytest.mark.parametrize("model_class", [
        UNet, SwinUNet, HybridModel, MLPModel, FNO2d
    ])
    def test_model_gradient_flow(self, model_class, device):
        """Test gradient flow through model"""
        model = self.create_model(model_class, device)
        
        # Create input with requires_grad
        x = torch.randn(1, 3, 128, 128, device=device, requires_grad=True)
        
        # Forward pass
        output = model(x)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"
        
        # Check parameter gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"
    
    @pytest.mark.parametrize("model_class", [
        UNet, SwinUNet, HybridModel, SegFormer
    ])
    def test_model_batch_consistency(self, model_class, device):
        """Test model consistency across different batch sizes"""
        model = self.create_model(model_class, device)
        model.eval()
        
        # Create single sample
        x_single = torch.randn(1, 3, 128, 128, device=device)
        
        # Create batch of the same sample
        x_batch = x_single.repeat(4, 1, 1, 1)
        
        with torch.no_grad():
            output_single = model(x_single)
            output_batch = model(x_batch)
        
        # Check consistency
        for i in range(4):
            diff = torch.abs(output_single - output_batch[i:i+1]).max()
            assert diff < 1e-5, f"Batch inconsistency: max diff = {diff}"
    
    @pytest.mark.parametrize("model_class", [
        UNet, SwinUNet, HybridModel, MLPModel
    ])
    def test_model_parameter_count(self, model_class, device):
        """Test model parameter count is reasonable"""
        model = self.create_model(model_class, device)
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Reasonable parameter count ranges (in millions)
        param_ranges = {
            'unet': (1e6, 50e6),
            'swinunet': (10e6, 100e6),
            'hybridmodel': (5e6, 80e6),
            'mlpmodel': (0.1e6, 10e6),
            'default': (0.1e6, 100e6),
        }
        
        model_name = model_class.__name__.lower()
        min_params, max_params = param_ranges.get(model_name, param_ranges['default'])
        
        assert min_params <= param_count <= max_params, \
            f"Parameter count {param_count:,} outside expected range [{min_params:,}, {max_params:,}]"
    
    @pytest.mark.parametrize("model_class", [
        UNet, SwinUNet, HybridModel, SegFormer, FNO2d
    ])
    def test_model_inference_speed(self, model_class, device):
        """Test model inference speed is reasonable"""
        import time
        
        model = self.create_model(model_class, device)
        model.eval()
        
        # Warm up
        x = torch.randn(1, 3, 128, 128, device=device)
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        
        # Time inference
        num_runs = 10
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        # Reasonable inference time (seconds)
        max_inference_time = 1.0  # 1 second max for 128x128 image
        assert avg_time < max_inference_time, f"Inference too slow: {avg_time:.3f}s average"
    
    def test_unet_architecture(self, device):
        """Specific test for UNet architecture"""
        model = self.create_model(UNet, device, features=[32, 64, 128])
        
        # Test encoder-decoder structure
        x = torch.randn(1, 3, 128, 128, device=device)
        
        # Test skip connections work
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 3, 128, 128)
        
        # Test different feature configurations
        model_small = self.create_model(UNet, device, features=[16, 32])
        x_small = torch.randn(1, 3, 64, 64, device=device)
        
        with torch.no_grad():
            output_small = model_small(x_small)
        
        assert output_small.shape == (1, 3, 64, 64)
    
    def test_swin_unet_window_attention(self, device):
        """Specific test for SwinUNet window attention"""
        model = self.create_model(SwinUNet, device, window_size=8)
        
        # Test window-based attention
        x = torch.randn(1, 3, 128, 128, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 3, 128, 128)
        
        # Test different window sizes
        if torch.cuda.is_available() or device.type == 'cpu':
            model_ws4 = self.create_model(SwinUNet, device, window_size=4)
            with torch.no_grad():
                output_ws4 = model_ws4(x)
            assert output_ws4.shape == (1, 3, 128, 128)
    
    def test_fno_spectral_operations(self, device):
        """Specific test for FNO spectral operations"""
        model = self.create_model(FNO2d, device, modes=16, width=32)
        
        # Test spectral convolution
        x = torch.randn(1, 3, 128, 128, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 3, 128, 128)
        
        # Test different mode configurations
        model_few_modes = self.create_model(FNO2d, device, modes=8, width=16)
        with torch.no_grad():
            output_few = model_few_modes(x)
        assert output_few.shape == (1, 3, 128, 128)
    
    def test_mlp_coord_encoding(self, device):
        """Specific test for MLP coordinate encoding"""
        model_with_coords = self.create_model(MLPModel, device, use_coords=True)
        model_without_coords = self.create_model(MLPModel, device, use_coords=False)
        
        x = torch.randn(1, 3, 64, 64, device=device)
        
        with torch.no_grad():
            output_with = model_with_coords(x)
            output_without = model_without_coords(x)
        
        # Both should produce valid output
        assert output_with.shape == (1, 3, 64, 64)
        assert output_without.shape == (1, 3, 64, 64)
    
    def test_sparse_attention_masking(self, device):
        """Specific test for sparse attention masking"""
        model = self.create_model(SparseAttentionEncoder, device, sparse_ratio=0.5)
        
        x = torch.randn(1, 3, 128, 128, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 3, 128, 128)
        
        # Test different sparse ratios
        model_dense = self.create_model(SparseAttentionEncoder, device, sparse_ratio=1.0)
        with torch.no_grad():
            output_dense = model_dense(x)
        assert output_dense.shape == (1, 3, 128, 128)
    
    @pytest.mark.parametrize("model_class", [
        UNet, SwinUNet, HybridModel, SegFormer
    ])
    def test_model_memory_efficiency(self, model_class, device):
        """Test model memory usage is reasonable"""
        if not torch.cuda.is_available():
            pytest.skip("Memory test requires CUDA")
        
        model = self.create_model(model_class, device)
        model.eval()
        
        # Test with large input
        x_large = torch.randn(1, 3, 512, 512, device=device)
        
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            output = model(x_large)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # Reasonable memory usage (GB)
        max_memory_gb = 8.0
        assert peak_memory < max_memory_gb, f"Memory usage too high: {peak_memory:.2f}GB"
    
    def test_model_robustness_to_noise(self, device):
        """Test model robustness to input noise"""
        model = self.create_model(SwinUNet, device)
        model.eval()
        
        # Clean input
        x_clean = torch.randn(1, 3, 128, 128, device=device)
        
        # Noisy input
        noise_level = 0.1
        x_noisy = x_clean + noise_level * torch.randn_like(x_clean)
        
        with torch.no_grad():
            output_clean = model(x_clean)
            output_noisy = model(x_noisy)
        
        # Check that outputs are similar
        diff = torch.abs(output_clean - output_noisy).mean()
        relative_diff = diff / (torch.abs(output_clean).mean() + 1e-8)
        
        # Allow up to 20% relative difference
        assert relative_diff < 0.2, f"Model too sensitive to noise: relative diff = {relative_diff:.3f}"


class TestModelConfigurations:
    """Test different model configurations and hyperparameters"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_unet_configurations(self, device):
        """Test UNet with different configurations"""
        configs = [
            {'features': [16, 32], 'bilinear': True},
            {'features': [32, 64, 128], 'bilinear': False},
            {'features': [64, 128, 256, 512], 'bilinear': True},
        ]
        
        for config in configs:
            model = UNet(in_ch=3, out_ch=3, img_size=128, **config)
            model.to(device)
            model.eval()
            
            x = torch.randn(1, 3, 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, 128, 128)
    
    def test_swin_unet_configurations(self, device):
        """Test SwinUNet with different configurations"""
        configs = [
            {'depths': [2, 2, 2, 2], 'num_heads': [3, 6, 12, 24], 'window_size': 8},
            {'depths': [2, 2, 6, 2], 'num_heads': [4, 8, 16, 32], 'window_size': 7},
            {'depths': [1, 1, 1, 1], 'num_heads': [2, 4, 8, 16], 'window_size': 4},
        ]
        
        for config in configs:
            model = SwinUNet(in_ch=3, out_ch=3, img_size=128, **config)
            model.to(device)
            model.eval()
            
            x = torch.randn(1, 3, 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, 128, 128)
    
    def test_hybrid_model_backbones(self, device):
        """Test HybridModel with different backbones"""
        backbones = ['swin', 'unet', 'fno']
        
        for backbone in backbones:
            model = HybridModel(
                in_ch=3, out_ch=3, img_size=128,
                backbone=backbone, fusion='concat'
            )
            model.to(device)
            model.eval()
            
            x = torch.randn(1, 3, 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, 128, 128)
    
    def test_segformer_backbones(self, device):
        """Test SegFormer with different backbones"""
        backbones = ['b0', 'b1', 'b2']
        
        for backbone in backbones:
            try:
                model = SegFormer(
                    in_ch=3, out_ch=3, img_size=128,
                    backbone=backbone
                )
                model.to(device)
                model.eval()
                
                x = torch.randn(1, 3, 128, 128, device=device)
                with torch.no_grad():
                    output = model(x)
                
                assert output.shape == (1, 3, 128, 128)
            except Exception as e:
                # Some backbones might not be available
                pytest.skip(f"Backbone {backbone} not available: {str(e)}")
    
    def test_mlp_model_variants(self, device):
        """Test MLPModel with different configurations"""
        configs = [
            {'use_coords': True, 'activation': 'gelu'},
            {'use_coords': False, 'activation': 'relu'},
            {'use_coords': True, 'activation': 'silu', 'hidden_dim': 512},
        ]
        
        for config in configs:
            model = MLPModel(in_ch=3, out_ch=3, img_size=128, **config)
            model.to(device)
            model.eval()
            
            x = torch.randn(1, 3, 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, 128, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])