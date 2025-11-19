#!/usr/bin/env python3
"""
Detailed analysis of model creation failures to identify specific fix patterns.
"""

import os
import sys
import inspect
import torch.nn as nn

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from tools.training.model_loader_improved import ImprovedModelLoader

def analyze_model_failures():
    """Analyze why each model failed and identify fix patterns."""
    
    loader = ImprovedModelLoader()
    available_models = loader.list_models()
    
    print("🔍 DETAILED MODEL FAILURE ANALYSIS")
    print("=" * 80)
    
    failure_patterns = {
        'missing_positional_args': [],
        'dimension_mismatch': [],
        'tensor_shape_issues': [],
        'missing_required_params': [],
        'other_errors': []
    }
    
    for model_name in sorted(available_models):
        print(f"\n📋 Analyzing: {model_name}")
        print("-" * 50)
        
        # Get model info
        model_info = loader.get_model_info(model_name)
        if 'error' in model_info:
            print(f"   ❌ Info error: {model_info['error']}")
            continue
            
        model_class = model_info['class']
        parameters = model_info['parameters']
        
        print(f"   📊 Constructor parameters:")
        for param_name, param_info in parameters.items():
            if param_name != 'self' and param_name != 'kwargs':
                print(f"      - {param_name}: {param_info}")
        
        # Try to create with different parameter sets
        test_configs = [
            {
                'name': 'Minimal Standard',
                'params': {
                    'in_channels': 3,
                    'out_channels': 3,
                    'img_size': 224,
                }
            },
            {
                'name': 'With Embed Dim',
                'params': {
                    'in_channels': 3,
                    'out_channels': 3,
                    'img_size': 224,
                    'embed_dim': 96,
                }
            },
            {
                'name': 'With Depths',
                'params': {
                    'in_channels': 3,
                    'out_channels': 3,
                    'img_size': 224,
                    'embed_dim': 96,
                    'depths': [2, 2, 6, 2],
                }
            }
        ]
        
        for config in test_configs:
            try:
                model = loader.create_model(model_name, **config['params'])
                
                # Test forward pass
                test_input = torch.randn(1, config['params']['in_channels'], 
                                       config['params']['img_size'], 
                                       config['params']['img_size'])
                
                with torch.no_grad():
                    output = model(test_input)
                
                print(f"   ✅ {config['name']}: SUCCESS")
                print(f"      Input: {test_input.shape} → Output: {output.shape}")
                break
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ {config['name']}: {error_msg[:100]}...")
                
                # Categorize the error
                if "missing" in error_msg and "required positional" in error_msg:
                    failure_patterns['missing_positional_args'].append((model_name, error_msg))
                elif "dimension" in error_msg or "size" in error_msg:
                    failure_patterns['dimension_mismatch'].append((model_name, error_msg))
                elif "tensor" in error_msg and "shape" in error_msg:
                    failure_patterns['tensor_shape_issues'].append((model_name, error_msg))
                elif "required" in error_msg:
                    failure_patterns['missing_required_params'].append((model_name, error_msg))
                else:
                    failure_patterns['other_errors'].append((model_name, error_msg))
    
    # Print failure pattern summary
    print(f"\n📈 FAILURE PATTERN SUMMARY")
    print("=" * 80)
    
    for pattern, failures in failure_patterns.items():
        if failures:
            print(f"\n🔸 {pattern.replace('_', ' ').title()}: {len(failures)} models")
            for model_name, error in failures[:3]:  # Show first 3
                print(f"   - {model_name}: {error[:80]}...")
            if len(failures) > 3:
                print(f"   ... and {len(failures) - 3} more")
    
    return failure_patterns

def identify_model_categories():
    """Identify what types of models we have and their requirements."""
    
    loader = ImprovedModelLoader()
    available_models = loader.list_models()
    
    categories = {
        'complete_architectures': [],
        'temporal_components': [],
        'spatial_components': [],
        'decoder_components': [],
        'attention_components': [],
        'other_components': []
    }
    
    print(f"\n🏗️ MODEL CATEGORIZATION")
    print("=" * 80)
    
    for model_name in sorted(available_models):
        model_info = loader.get_model_info(model_name)
        if 'error' in model_info:
            continue
            
        model_class = model_info['class']
        parameters = model_info['parameters']
        
        # Analyze parameter patterns to categorize
        param_names = set(parameters.keys())
        
        # Check for temporal-specific parameters
        temporal_params = {'temporal_dim', 'input_dim', 'd_model', 'sequence_length'}
        has_temporal = any(p in param_names for p in temporal_params)
        
        # Check for decoder-specific parameters  
        decoder_params = {'encoder_channels', 'decoder_channels', 'skip_connections'}
        has_decoder = any(p in param_names for p in decoder_params)
        
        # Check for attention-specific parameters
        attention_params = {'num_heads', 'attention_dim', 'qkv_bias'}
        has_attention = any(p in param_names for p in attention_params)
        
        # Check for standard architecture parameters
        standard_params = {'in_channels', 'out_channels', 'img_size', 'embed_dim'}
        has_standard = any(p in param_names for p in standard_params)
        
        # Categorize based on patterns
        if has_temporal and not has_standard:
            categories['temporal_components'].append(model_name)
        elif has_decoder and not has_standard:
            categories['decoder_components'].append(model_name)
        elif has_attention and not has_standard:
            categories['attention_components'].append(model_name)
        elif has_standard and not has_temporal:
            categories['complete_architectures'].append(model_name)
        elif has_standard and has_temporal:
            categories['complete_architectures'].append(model_name)  # Hybrid models
        else:
            categories['other_components'].append(model_name)
    
    # Print categorization
    for category, models in categories.items():
        if models:
            print(f"\n📂 {category.replace('_', ' ').title()}: {len(models)} models")
            for model in models[:5]:  # Show first 5
                print(f"   - {model}")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")
    
    return categories

def suggest_model_specific_params():
    """Suggest model-specific parameter configurations."""
    
    print(f"\n💡 MODEL-SPECIFIC PARAMETER SUGGESTIONS")
    print("=" * 80)
    
    suggestions = {
        'SwinUNet': {
            'params': {
                'in_channels': 3, 'out_channels': 3, 'img_size': 224,
                'embed_dim': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24],
                'window_size': 8, 'patch_size': 4
            },
            'notes': 'Standard SwinUNet configuration'
        },
        'SparseSwinUNet': {
            'params': {
                'in_channels': 3, 'out_channels': 3, 'img_size': 224,
                'embed_dim': 96, 'depths': [2, 2, 6, 2]
            },
            'notes': 'Sparse variant with reduced parameters'
        },
        'TemporalConv1D': {
            'params': {
                'in_channels': 3, 'hidden_channels': 64, 'num_layers': 4,
                'kernel_size': 3, 'dilation_base': 2
            },
            'notes': '1D temporal convolution, needs sequence input'
        },
        'SwinTransformerBlock': {
            'params': {
                'dim': 96, 'input_resolution': (56, 56), 'num_heads': 6,
                'window_size': 8
            },
            'notes': 'Single transformer block, not complete architecture'
        }
    }
    
    for model_name, config in suggestions.items():
        print(f"\n🔧 {model_name}:")
        print(f"   Parameters: {config['params']}")
        print(f"   Notes: {config['notes']}")
    
    return suggestions

def main():
    """Main analysis function."""
    
    failure_patterns = analyze_model_failures()
    categories = identify_model_categories()
    suggestions = suggest_model_specific_params()
    
    print(f"\n🎯 RECOMMENDATIONS")
    print("=" * 80)
    print("1. Focus on 'complete_architectures' category for training")
    print("2. Use model-specific parameter configurations")
    print("3. Add intelligent parameter inference for temporal models")
    print("4. Create wrapper functions for component models")
    print("5. Implement model capability detection")
    
    return failure_patterns, categories, suggestions

if __name__ == "__main__":
    main()