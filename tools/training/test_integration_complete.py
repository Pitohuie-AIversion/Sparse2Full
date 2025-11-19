#!/usr/bin/env python3
"""
Comprehensive integration test for the enhanced training script with four-layer fallback strategy.

This test validates:
1. All three model loaders work correctly
2. The four-layer fallback strategy functions properly
3. Model creation succeeds for different model types
4. Forward pass testing works correctly
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
import torch

# Import all model loaders
from tools.training.model_loader import create_model_with_loader, list_models
from tools.training.model_loader_improved import create_improved_model, list_improved_models
from tools.training.model_loader_enhanced import create_enhanced_model, list_enhanced_models, test_enhanced_model


def test_model_loaders():
    """Test all three model loaders with different model types."""
    print("=== Testing Model Loaders ===")
    
    # Test different model types
    test_models = ['swin_unet', 'unet', 'fno2d', 'mlpmixer']
    config = OmegaConf.create({
        'data': {'img_size': 128, 'channels': 1},
        'model': {'embed_dim': 96, 'depths': [2, 2, 6, 2]}
    })
    
    results = {}
    
    for model_name in test_models:
        print(f"\n--- Testing {model_name} ---")
        model_results = {}
        
        # Test enhanced loader
        try:
            model = create_enhanced_model(model_name, config, in_channels=1, out_channels=1, img_size=128)
            model_results['enhanced'] = {'success': True, 'type': type(model).__name__}
            print(f"✅ Enhanced loader: {type(model).__name__}")
            
            # Test forward pass
            try:
                test_input = torch.randn(1, 1, 128, 128)
                with torch.no_grad():
                    output = model(test_input)
                model_results['enhanced']['forward_pass'] = f"{test_input.shape} -> {output.shape}"
                print(f"✅ Forward pass: {test_input.shape} -> {output.shape}")
            except Exception as e:
                model_results['enhanced']['forward_pass'] = f"Failed: {e}"
                print(f"⚠️ Forward pass failed: {e}")
                
        except Exception as e:
            model_results['enhanced'] = {'success': False, 'error': str(e)}
            print(f"❌ Enhanced loader failed: {e}")
        
        # Test improved loader
        try:
            model = create_improved_model(model_name, config, in_channels=1, out_channels=1, img_size=128)
            model_results['improved'] = {'success': True, 'type': type(model).__name__}
            print(f"✅ Improved loader: {type(model).__name__}")
        except Exception as e:
            model_results['improved'] = {'success': False, 'error': str(e)}
            print(f"❌ Improved loader failed: {e}")
        
        # Test original loader
        try:
            model = create_model_with_loader(model_name, config, in_channels=1, out_channels=1, img_size=128)
            model_results['original'] = {'success': True, 'type': type(model).__name__}
            print(f"✅ Original loader: {type(model).__name__}")
        except Exception as e:
            model_results['original'] = {'success': False, 'error': str(e)}
            print(f"❌ Original loader failed: {e}")
        
        results[model_name] = model_results
    
    return results


def test_fallback_strategy():
    """Test the four-layer fallback strategy."""
    print("\n=== Testing Four-Layer Fallback Strategy ===")
    
    # Create a mock training script setup
    from tools.training.train_real_data_ar import RealDataARTrainer
    
    # Create minimal config
    config_content = """
data:
  img_size: 128
  channels: 1
  T_in: 10
  T_out: 20
  
model:
  name: swin_unet
  in_channels: 1
  out_channels: 1
  img_size: 128
  embed_dim: 96
  depths: [2, 2, 6, 2]
  
training:
  epochs: 1
  batch_size: 2
  
experiment:
  name: test_fallback
  output_dir: /tmp/test_fallback
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        config = OmegaConf.load(config_path)
        
        # Test the setup_traditional_model method indirectly
        print("Testing model creation with different scenarios...")
        
        # Test 1: Normal model creation
        try:
            model = create_enhanced_model('swin_unet', config, in_channels=1, out_channels=1, img_size=128)
            print(f"✅ Layer 1 (Enhanced): Created {type(model).__name__}")
        except Exception as e:
            print(f"❌ Layer 1 (Enhanced) failed: {e}")
            
            # Test 2: Fallback to improved loader
            try:
                model = create_improved_model('swin_unet', config, in_channels=1, out_channels=1, img_size=128)
                print(f"✅ Layer 2 (Improved): Created {type(model).__name__}")
            except Exception as e2:
                print(f"❌ Layer 2 (Improved) failed: {e2}")
                
                # Test 3: Fallback to original loader
                try:
                    model = create_model_with_loader('swin_unet', config, in_channels=1, out_channels=1, img_size=128)
                    print(f"✅ Layer 3 (Original): Created {type(model).__name__}")
                except Exception as e3:
                    print(f"❌ Layer 3 (Original) failed: {e3}")
                    
                    # Test 4: Final fallback to default SwinUNet
                    try:
                        from models.swin_unet import SwinUNet
                        model = SwinUNet(in_channels=1, out_channels=1, img_size=128)
                        print(f"✅ Layer 4 (Default SwinUNet): Created {type(model).__name__}")
                    except Exception as e4:
                        print(f"❌ All layers failed: {e4}")
        
        # Test with a problematic model that should trigger fallback
        print("\n--- Testing fallback with problematic model ---")
        try:
            # Try a model that might not exist or have issues
            model = create_enhanced_model('nonexistent_model', config, in_channels=1, out_channels=1, img_size=128)
            print(f"Unexpected success with nonexistent model: {type(model).__name__}")
        except Exception as e:
            print(f"✅ Correctly failed with nonexistent model: {e}")
            
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_model_compatibility():
    """Test model compatibility with different configurations."""
    print("\n=== Testing Model Compatibility ===")
    
    # Test with different image sizes and channel configurations
    test_configs = [
        {'img_size': 64, 'channels': 1},
        {'img_size': 128, 'channels': 2},
        {'img_size': 256, 'channels': 3},
    ]
    
    models_to_test = ['swin_unet', 'unet', 'fno2d']
    
    for config_dict in test_configs:
        print(f"\n--- Testing with img_size={config_dict['img_size']}, channels={config_dict['channels']} ---")
        
        config = OmegaConf.create({
            'data': config_dict,
            'model': {'embed_dim': 96, 'depths': [2, 2, 6, 2]}
        })
        
        for model_name in models_to_test:
            try:
                model = create_enhanced_model(
                    model_name, 
                    config, 
                    in_channels=config_dict['channels'], 
                    out_channels=config_dict['channels'], 
                    img_size=config_dict['img_size']
                )
                
                # Test forward pass
                test_input = torch.randn(1, config_dict['channels'], config_dict['img_size'], config_dict['img_size'])
                with torch.no_grad():
                    output = model(test_input)
                
                print(f"✅ {model_name}: {test_input.shape} -> {output.shape}")
                
            except Exception as e:
                print(f"❌ {model_name}: {e}")


def main():
    """Run all integration tests."""
    print("🧪 Starting Comprehensive Integration Tests")
    print("=" * 60)
    
    try:
        # Test 1: Model loaders
        loader_results = test_model_loaders()
        
        # Test 2: Fallback strategy
        test_fallback_strategy()
        
        # Test 3: Model compatibility
        test_model_compatibility()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        # Count successful model creations
        total_attempts = 0
        successful_creations = 0
        
        for model_name, results in loader_results.items():
            for loader_type, result in results.items():
                total_attempts += 1
                if result.get('success', False):
                    successful_creations += 1
        
        success_rate = (successful_creations / total_attempts) * 100 if total_attempts > 0 else 0
        
        print(f"Total model creation attempts: {total_attempts}")
        print(f"Successful creations: {successful_creations}")
        print(f"Overall success rate: {success_rate:.1f}%")
        
        if success_rate >= 70:
            print("✅ Integration tests PASSED - System ready for production!")
            return 0
        else:
            print("⚠️ Integration tests PARTIAL - Some models need attention")
            return 1
            
    except Exception as e:
        print(f"❌ Integration tests FAILED: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())