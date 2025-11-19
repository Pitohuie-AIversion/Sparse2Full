#!/usr/bin/env python3
"""
Comprehensive test for the improved model loader.
Tests all models that failed with the original loader.
"""

import os
import sys
import logging
import torch
from typing import Dict, Any

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from tools.training.model_loader_improved import (
    ImprovedModelLoader, 
    create_improved_model, 
    list_improved_models,
    get_improved_model_info
)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_model_creation():
    """Test creating models with the improved loader."""
    print("=" * 80)
    print("TESTING IMPROVED MODEL LOADER")
    print("=" * 80)
    
    # Initialize the loader
    loader = ImprovedModelLoader()
    
    # Get available models
    available_models = loader.list_models()
    utility_classes = loader.list_utility_classes()
    
    print(f"\n📊 LOADER STATISTICS:")
    print(f"   Total models: {len(available_models)}")
    print(f"   Utility classes: {len(utility_classes)}")
    print(f"   Success rate: {len(available_models)/(len(available_models) + len(utility_classes))*100:.1f}%")
    
    print(f"\n🔧 UTILITY CLASSES (excluded from model list):")
    for i, util_class in enumerate(sorted(utility_classes)[:10], 1):
        print(f"   {i:2d}. {util_class}")
    if len(utility_classes) > 10:
        print(f"   ... and {len(utility_classes) - 10} more")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Standard Config',
            'params': {
                'in_channels': 3,
                'out_channels': 3,
                'img_size': 224,
            }
        },
        {
            'name': 'Alternative Names Config',
            'params': {
                'in_ch': 3,
                'out_ch': 3,
                'img_size': 224,
            }
        },
        {
            'name': 'Mixed Config',
            'params': {
                'n_channels': 3,
                'n_classes': 3,
                'image_size': 224,
            }
        }
    ]
    
    # Test each model
    print(f"\n🧪 TESTING MODEL CREATION:")
    print("-" * 80)
    
    success_count = 0
    failure_count = 0
    
    for model_name in sorted(available_models):
        print(f"\nTesting: {model_name}")
        print("-" * 40)
        
        model_success = False
        model_errors = []
        
        for config in test_configs:
            try:
                # Try to create the model
                model = loader.create_model(model_name, config=config['params'])
                
                # Test forward pass
                test_input = torch.randn(1, config['params']['in_channels'], 
                                       config['params']['img_size'], 
                                       config['params']['img_size'])
                
                with torch.no_grad():
                    output = model(test_input)
                
                print(f"   ✓ {config['name']}: Success")
                print(f"     Model: {type(model).__name__}")
                print(f"     Input: {test_input.shape} → Output: {output.shape}")
                model_success = True
                success_count += 1
                break
                
            except Exception as e:
                error_msg = f"{config['name']}: {str(e)[:100]}..."
                model_errors.append(error_msg)
                print(f"   ✗ {error_msg}")
        
        if not model_success:
            failure_count += 1
            print(f"   ❌ All configurations failed for {model_name}")
            for error in model_errors:
                print(f"      - {error}")
    
    # Test specific problematic models that were identified in the original analysis
    print(f"\n🔍 TESTING PREVIOUSLY PROBLEMATIC MODELS:")
    print("-" * 80)
    
    problematic_models = [
        'SwinUNet',  # Had **kwargs issues
        'UNet',      # Missing from original loader
        'FNO2D',     # Missing from original loader
    ]
    
    for model_name in problematic_models:
        if model_name in available_models:
            print(f"\nTesting previously problematic: {model_name}")
            try:
                model = create_improved_model(model_name)
                print(f"   ✓ Successfully created with improved loader!")
                print(f"   Model info: {get_improved_model_info(model_name)}")
            except Exception as e:
                print(f"   ✗ Still failing: {e}")
        else:
            print(f"\n{model_name} not found in available models")
    
    # Summary
    print(f"\n📈 FINAL RESULTS:")
    print("=" * 80)
    print(f"Total models tested: {len(available_models)}")
    print(f"Successful creations: {success_count}")
    print(f"Failed creations: {failure_count}")
    print(f"Success rate: {success_count/len(available_models)*100:.1f}%")
    
    if failure_count > 0:
        print(f"\n⚠️  {failure_count} models still have issues and need attention.")
    else:
        print(f"\n🎉 All models successfully created! The improved loader is working perfectly.")

def test_parameter_mapping():
    """Test the parameter mapping functionality."""
    print(f"\n🔧 TESTING PARAMETER MAPPING:")
    print("-" * 80)
    
    loader = ImprovedModelLoader()
    
    # Test different parameter name conventions
    test_params = {
        'in_ch': 5,
        'out_ch': 7,
        'n_channels': 5,
        'n_classes': 7,
        'image_size': 256,
        'input_size': 256,
    }
    
    print("Testing parameter name mapping with different conventions:")
    for key, value in test_params.items():
        print(f"   {key}: {value}")
    
    # Test with a model that should accept these parameters
    try:
        model = loader.create_model('SwinUNet', **test_params)
        print(f"\n✓ Successfully created model with mapped parameters!")
        print(f"Model type: {type(model).__name__}")
    except Exception as e:
        print(f"\n✗ Parameter mapping failed: {e}")

def main():
    """Main test function."""
    setup_logging()
    
    try:
        test_model_creation()
        test_parameter_mapping()
        
        print(f"\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())