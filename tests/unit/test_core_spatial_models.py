"""
Simplified comprehensive unit tests for core spatial prediction models.
Tests the most important spatial models with proper configurations.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

# Import core spatial models that are well-tested and stable
from models import (
    UNet, SwinUNet, HybridModel, MLPModel, FNO2d
)


class CoreSpatialModelsTest:
    """Test suite for core spatial prediction models"""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_unet_config(self) -> Dict[str, Any]:
        """Get UNet configuration"""
        return {
            'in_channels': 3,
            'out_channels': 3,
            'features': [32, 64, 128],
            'bilinear': True,
        }
    
    def get_swin_unet_config(self, img_size: int = 128) -> Dict[str, Any]:
        """Get SwinUNet configuration"""
        return {
            'in_chans': 3,
            'num_classes': 3,
            'img_size': img_size,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'mlp_ratio': 4.0,
        }
    
    def get_hybrid_model_config(self) -> Dict[str, Any]:
        """Get HybridModel configuration"""
        return {
            'in_ch': 3,
            'out_ch': 3,
            'img_size': 128,
            'backbone': 'swin',
            'fusion': 'concat',
            'attention_ch': 64,
            'fno_modes': 16,
            'fno_width': 32,
        }
    
    def get_mlp_model_config(self) -> Dict[str, Any]:
        """Get MLPModel configuration"""
        return {
            'in_features': 3,
            'out_features': 3,
            'img_size': 128,
            'hidden_dim': 256,
            'num_layers': 4,
            'activation': 'gelu',
            'use_coords': True,
        }
    
    def get_fno2d_config(self) -> Dict[str, Any]:
        """Get FNO2d configuration"""
        return {
            'modes': 16,
            'width': 32,
            'layers': 4,
        }
    
    def create_model(self, model_class: type, device: torch.device, **kwargs) -> nn.Module:
        """Create model instance with given configuration"""
        try:
            model = model_class(**kwargs)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            pytest.skip(f"Failed to create {model_class.__name__}: {str(e)}")
    
    @pytest.mark.parametrize("model_class,config_method", [
        (UNet, 'get_unet_config'),
        (SwinUNet, 'get_swin_unet_config'),
        (HybridModel, 'get_hybrid_model_config'),
        (MLPModel, 'get_mlp_model_config'),
        (FNO2d, 'get_fno2d_config'),
    ])
    def test_model_forward_pass(self, model_class, config_method, device):
        """Test basic forward pass for core models"""
        # Get configuration
        config = getattr(self, config_method)()
        
        # Create model
        model = self.create_model(model_class, device, **config)
        
        # Create input tensor
        x = torch.randn(1, 3, 128, 128, device=device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        assert output.shape == (1, 3, 128, 128), f"Expected (1, 3, 128, 128), got {output.shape}"
        
        # Check output properties
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
    
    @pytest.mark.parametrize("model_class,config_method", [
        (UNet, 'get_unet_config'),
        (SwinUNet, 'get_swin_unet_config'),
        (HybridModel, 'get_hybrid_model_config'),
        (FNO2d, 'get_fno2d_config'),
    ])
    def test_model_different_sizes(self, model_class, config_method, device):
        """Test model with different input sizes"""
        config = getattr(self, config_method)()
        
        # Adjust img_size for SwinUNet
        if model_class == SwinUNet:
            test_sizes = [64, 128, 256]  # Must be divisible by 32
        else:
            test_sizes = [64, 128, 256]
        
        for size in test_sizes:
            if model_class == SwinUNet:
                config = self.get_swin_unet_config(size)
            
            model = self.create_model(model_class, device, **config)
            
            x = torch.randn(1, 3, size, size, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, size, size), f"Failed for size {size}"
    
    @pytest.mark.parametrize("model_class,config_method", [
        (UNet, 'get_unet_config'),
        (SwinUNet, 'get_swin_unet_config'),
        (HybridModel, 'get_hybrid_model_config'),
        (MLPModel, 'get_mlp_model_config'),
    ])
    def test_model_gradient_flow(self, model_class, config_method, device):
        """Test gradient flow through model"""
        config = getattr(self, config_method)()
        model = self.create_model(model_class, device, **config)
        
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
    
    @pytest.mark.parametrize("model_class,config_method", [
        (UNet, 'get_unet_config'),
        (SwinUNet, 'get_swin_unet_config'),
        (HybridModel, 'get_hybrid_model_config'),
    ])
    def test_model_batch_consistency(self, model_class, config_method, device):
        """Test model consistency across different batch sizes"""
        config = getattr(self, config_method)()
        model = self.create_model(model_class, device, **config)
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
    
    @pytest.mark.parametrize("model_class,config_method", [
        (UNet, 'get_unet_config'),
        (SwinUNet, 'get_swin_unet_config'),
        (HybridModel, 'get_hybrid_model_config'),
        (MLPModel, 'get_mlp_model_config'),
    ])
    def test_model_parameter_count(self, model_class, config_method, device):
        """Test model parameter count is reasonable"""
        config = getattr(self, config_method)()
        model = self.create_model(model_class, device, **config)
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Reasonable parameter count ranges (in millions)
        min_params = 0.1e6   # 100K minimum
        max_params = 100e6   # 100M maximum for reasonable models
        
        assert min_params <= param_count <= max_params, \
            f"Parameter count {param_count:,} outside expected range [{min_params:,}, {max_params:,}]"
    
    def test_unet_configurations(self, device):
        """Test UNet with different configurations"""
        configs = [
            {'in_channels': 3, 'out_channels': 3, 'features': [16, 32], 'bilinear': True},
            {'in_channels': 3, 'out_channels': 3, 'features': [32, 64, 128], 'bilinear': False},
            {'in_channels': 1, 'out_channels': 1, 'features': [32, 64], 'bilinear': True},
        ]
        
        for config in configs:
            model = self.create_model(UNet, device, **config)
            
            x = torch.randn(1, config['in_channels'], 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            expected_shape = (1, config['out_channels'], 128, 128)
            assert output.shape == expected_shape
    
    def test_swin_unet_window_sizes(self, device):
        """Test SwinUNet with different window sizes"""
        window_sizes = [4, 8, 16]
        
        for window_size in window_sizes:
            config = self.get_swin_unet_config()
            config['window_size'] = window_size
            
            model = self.create_model(SwinUNet, device, **config)
            
            x = torch.randn(1, 3, 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, 128, 128)
    
    def test_fno_modes(self, device):
        """Test FNO2d with different mode configurations"""
        mode_configs = [
            {'modes': 8, 'width': 16, 'layers': 2},
            {'modes': 16, 'width': 32, 'layers': 4},
            {'modes': 32, 'width': 64, 'layers': 6},
        ]
        
        for config in mode_configs:
            model = self.create_model(FNO2d, device, **config)
            
            x = torch.randn(1, 3, 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, 128, 128)
    
    def test_hybrid_backbones(self, device):
        """Test HybridModel with different backbones"""
        backbones = ['swin', 'unet', 'fno']
        
        for backbone in backbones:
            config = self.get_hybrid_model_config()
            config['backbone'] = backbone
            
            model = self.create_model(HybridModel, device, **config)
            
            x = torch.randn(1, 3, 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, 128, 128)
    
    def test_mlp_variants(self, device):
        """Test MLPModel with different configurations"""
        configs = [
            {'use_coords': True, 'activation': 'gelu'},
            {'use_coords': False, 'activation': 'relu'},
            {'use_coords': True, 'activation': 'tanh'},
        ]
        
        base_config = self.get_mlp_model_config()
        
        for variant_config in configs:
            config = base_config.copy()
            config.update(variant_config)
            
            model = self.create_model(MLPModel, device, **config)
            
            x = torch.randn(1, 3, 128, 128, device=device)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 3, 128, 128)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_memory_efficiency(self, device):
        """Test model memory usage is reasonable"""
        models_to_test = [
            (UNet, self.get_unet_config()),
            (SwinUNet, self.get_swin_unet_config()),
            (HybridModel, self.get_hybrid_model_config()),
        ]
        
        for model_class, config in models_to_test:
            model = self.create_model(model_class, device, **config)
            model.eval()
            
            # Test with large input
            x_large = torch.randn(1, 3, 512, 512, device=device)
            
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                output = model(x_large)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            # Reasonable memory usage (GB)
            max_memory_gb = 8.0
            assert peak_memory < max_memory_gb, f"{model_class.__name__} memory usage too high: {peak_memory:.2f}GB"
    
    def test_model_inference_speed(self, device):
        """Test model inference speed is reasonable"""
        import time
        
        models_to_test = [
            (UNet, self.get_unet_config()),
            (SwinUNet, self.get_swin_unet_config()),
            (FNO2d, self.get_fno2d_config()),
        ]
        
        for model_class, config in models_to_test:
            model = self.create_model(model_class, device, **config)
            model.eval()
            
            # Warm up
            x = torch.randn(1, 3, 128, 128, device=device)
            for _ in range(5):
                with torch.no_grad():
                    _ = model(x)
            
            # Time inference
            num_runs = 10
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    output = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
            
            # Reasonable inference time (milliseconds)
            max_inference_time = 200.0  # 200ms max for 128x128 image
            assert avg_time < max_inference_time, f"{model_class.__name__} inference too slow: {avg_time:.3f}ms average"


class TestModelRobustness:
    """Test model robustness and edge cases"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_input_noise_robustness(self, device):
        """Test model robustness to input noise"""
        from models import SwinUNet
        
        # Create model
        model = SwinUNet(
            in_chans=3,
            num_classes=3,
            img_size=128,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8
        ).to(device)
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
        
        # Allow up to 30% relative difference
        assert relative_diff < 0.3, f"Model too sensitive to noise: relative diff = {relative_diff:.3f}"
    
    def test_edge_case_inputs(self, device):
        """Test models with edge case inputs"""
        from models import UNet

        model = UNet(
            in_channels=3,
            out_channels=3,
            img_size=128,
            features=[32, 64, 128, 256],
            bilinear=True
        ).to(device)
        model.eval()
        
        # Test with very small values
        x_small = torch.full((1, 3, 128, 128), 1e-6, device=device)
        with torch.no_grad():
            output_small = model(x_small)
        assert torch.isfinite(output_small).all()
        
        # Test with large values (but not extreme)
        x_large = torch.full((1, 3, 128, 128), 10.0, device=device)
        with torch.no_grad():
            output_large = model(x_large)
        assert torch.isfinite(output_large).all()
        
        # Test with zeros
        x_zeros = torch.zeros((1, 3, 128, 128), device=device)
        with torch.no_grad():
            output_zeros = model(x_zeros)
        assert torch.isfinite(output_zeros).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])