"""
Core Spatial Models Test Report Generator
"""

import torch
from models import UNet, SwinUNet, HybridModel, MLPModel, FNO2d
import time
import json
import os

def test_model_performance():
    """Test core spatial model performance and functionality"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Testing on device: {device}')
    
    # Test configurations for core models
    configs = {
        'UNet': {
            'model': UNet,
            'config': {'in_channels': 3, 'out_channels': 3, 'img_size': 128, 'features': [32, 64, 128, 256], 'bilinear': True}
        },
        'SwinUNet': {
            'model': SwinUNet,
            'config': {'in_chans': 3, 'num_classes': 3, 'img_size': 128, 'depths': [2, 2, 2, 2], 'num_heads': [3, 6, 12, 24], 'window_size': 8}
        },
        'HybridModel': {
            'model': HybridModel,
            'config': {'in_ch': 3, 'out_ch': 3, 'img_size': 128, 'backbone': 'swin', 'fusion': 'concat', 'attention_ch': 64, 'fno_modes': 16, 'fno_width': 32}
        },
        'FNO2d': {
            'model': FNO2d,
            'config': {'modes': 16, 'width': 32, 'layers': 4, 'in_channels': 3, 'out_channels': 3}
        }
    }
    
    results = {}
    
    for name, info in configs.items():
        try:
            print(f'\nTesting {name}...')
            
            # Create model
            model = info['model'](**info['config']).to(device)
            model.eval()
            
            # Test input
            x = torch.randn(1, 3, 128, 128, device=device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)
            
            # Measure inference time
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            num_runs = 10
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            # Memory usage (if CUDA)
            memory_usage = 0
            if device.type == 'cuda':
                memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # Output shape validation
            with torch.no_grad():
                test_output = model(x)
            
            results[name] = {
                'status': 'PASS',
                'inference_time_ms': avg_time,
                'parameters': param_count,
                'memory_usage_mb': memory_usage,
                'output_shape': list(test_output.shape),
                'output_valid': torch.isfinite(test_output).all().item()
            }
            
            print(f'  Inference time: {avg_time:.2f}ms')
            print(f'  Parameters: {param_count:,}')
            print(f'  Memory usage: {memory_usage:.1f}MB')
            print(f'  Output shape: {test_output.shape}')
            
        except Exception as e:
            results[name] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f'  Failed: {str(e)}')
    
    return results

def generate_test_report():
    """Generate comprehensive test report"""
    print('='*60)
    print('CORE SPATIAL MODELS TEST REPORT')
    print('='*60)
    print('Testing spatial prediction models for functionality and performance')
    print('='*60)
    
    # Run performance tests
    results = test_model_performance()
    
    # Calculate statistics
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    total = len(results)
    
    print('\nTEST SUMMARY')
    print('-' * 40)
    print(f'Total models tested: {total}')
    print(f'Passed: {passed}')
    print(f'Failed: {total - passed}')
    print(f'Success rate: {passed/total*100:.1f}%')
    
    if passed > 0:
        print('\nPERFORMANCE METRICS')
        print('-' * 40)
        
        # Sort by inference time
        sorted_results = [(name, result) for name, result in results.items() 
                         if result['status'] == 'PASS']
        sorted_results.sort(key=lambda x: x[1]['inference_time_ms'])
        
        for name, result in sorted_results:
            print(f'{name:15} | {result["inference_time_ms"]:6.2f}ms | {result["parameters"]:8,} params | {result["memory_usage_mb"]:6.1f}MB')
        
        # Calculate average performance
        avg_time = sum(r['inference_time_ms'] for r in results.values() if r['status'] == 'PASS') / passed
        avg_params = sum(r['parameters'] for r in results.values() if r['status'] == 'PASS') / passed
        avg_memory = sum(r['memory_usage_mb'] for r in results.values() if r['status'] == 'PASS') / passed
        
        print(f'\nAVERAGE PERFORMANCE:')
        print(f'  Inference time: {avg_time:.2f}ms')
        print(f'  Parameters: {avg_params:,.0f}')
        print(f'  Memory usage: {avg_memory:.1f}MB')
    
    if total - passed > 0:
        print('\nFAILED MODELS')
        print('-' * 40)
        for name, result in results.items():
            if result['status'] == 'FAIL':
                print(f'{name}: {result["error"]}')
    
    print('\n' + '='*60)
    print('TEST EXECUTION COMPLETED')
    print('='*60)
    
    # Save results to JSON
    with open('core_models_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_models': total,
                'passed': passed,
                'failed': total - passed,
                'success_rate': passed/total*100
            },
            'detailed_results': results
        }, f, indent=2)
    
    print('\nDetailed results saved to: core_models_test_results.json')
    
    return results

if __name__ == "__main__":
    generate_test_report()