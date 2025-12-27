"""
Performance monitoring and benchmarking configuration for spatial models.
Defines performance standards, thresholds, and monitoring metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch


@dataclass
class PerformanceThresholds:
    """Performance thresholds for spatial models"""
    
    # Inference speed thresholds (milliseconds)
    max_inference_time_128: float = 50.0    # Max time for 128x128 image
    max_inference_time_256: float = 200.0   # Max time for 256x256 image
    max_inference_time_512: float = 800.0   # Max time for 512x512 image
    
    # Memory usage thresholds (MB)
    max_memory_128: float = 1024.0          # Max memory for 128x128 image
    max_memory_256: float = 2048.0          # Max memory for 256x256 image
    max_memory_512: float = 4096.0          # Max memory for 512x512 image
    
    # Throughput requirements (FPS)
    min_throughput_128: float = 20.0        # Min FPS for 128x128 image
    min_throughput_256: float = 5.0         # Min FPS for 256x256 image
    min_throughput_512: float = 1.0           # Min FPS for 512x512 image
    
    # Model size thresholds (millions of parameters)
    max_params_lightweight: float = 10.0      # Lightweight models
    max_params_standard: float = 50.0         # Standard models
    max_params_heavy: float = 100.0           # Heavy models
    
    # Accuracy/Quality thresholds (task-dependent)
    min_psnr_reconstruction: float = 25.0      # Minimum PSNR for reconstruction
    min_ssim_reconstruction: float = 0.8       # Minimum SSIM for reconstruction
    min_accuracy_segmentation: float = 0.85   # Minimum accuracy for segmentation
    min_iou_segmentation: float = 0.7         # Minimum IoU for segmentation
    
    # Training efficiency thresholds
    max_training_time_per_epoch: float = 300.0  # Max seconds per epoch (100 samples)
    min_convergence_improvement: float = 0.1    # Minimum loss improvement ratio


@dataclass
class ModelPerformanceProfile:
    """Performance profile for a specific model type"""
    
    model_name: str
    model_category: str  # 'lightweight', 'standard', 'heavy'
    
    # Expected performance ranges
    expected_inference_time_128: Tuple[float, float]  # (min, max) ms
    expected_memory_128: Tuple[float, float]          # (min, max) MB
    expected_throughput_128: Tuple[float, float]    # (min, max) FPS
    
    # Quality expectations
    expected_psnr_range: Tuple[float, float]          # (min, max) dB
    expected_ssim_range: Tuple[float, float]         # (min, max)
    
    # Resource utilization
    gpu_memory_efficiency: float                      # 0-1, higher is better
    cpu_efficiency: float                             # 0-1, higher is better


class PerformanceStandards:
    """Performance standards for different spatial model categories"""
    
    def __init__(self):
        self.thresholds = PerformanceThresholds()
        self.model_profiles = self._initialize_model_profiles()
    
    def _initialize_model_profiles(self) -> Dict[str, ModelPerformanceProfile]:
        """Initialize performance profiles for different model types"""
        
        profiles = {
            # Lightweight models (fast inference, low memory)
            'MLPModel': ModelPerformanceProfile(
                model_name='MLPModel',
                model_category='lightweight',
                expected_inference_time_128=(5.0, 20.0),
                expected_memory_128=(100.0, 300.0),
                expected_throughput_128=(50.0, 200.0),
                expected_psnr_range=(20.0, 35.0),
                expected_ssim_range=(0.6, 0.9),
                gpu_memory_efficiency=0.9,
                cpu_efficiency=0.8
            ),
            
            'UNet_small': ModelPerformanceProfile(
                model_name='UNet_small',
                model_category='lightweight',
                expected_inference_time_128=(10.0, 30.0),
                expected_memory_128=(200.0, 500.0),
                expected_throughput_128=(30.0, 100.0),
                expected_psnr_range=(25.0, 40.0),
                expected_ssim_range=(0.7, 0.95),
                gpu_memory_efficiency=0.85,
                cpu_efficiency=0.75
            ),
            
            # Standard models (balanced performance)
            'UNet': ModelPerformanceProfile(
                model_name='UNet',
                model_category='standard',
                expected_inference_time_128=(15.0, 40.0),
                expected_memory_128=(300.0, 800.0),
                expected_throughput_128=(25.0, 70.0),
                expected_psnr_range=(28.0, 42.0),
                expected_ssim_range=(0.8, 0.96),
                gpu_memory_efficiency=0.8,
                cpu_efficiency=0.7
            ),
            
            'SwinUNet': ModelPerformanceProfile(
                model_name='SwinUNet',
                model_category='standard',
                expected_inference_time_128=(20.0, 50.0),
                expected_memory_128=(400.0, 1000.0),
                expected_throughput_128=(20.0, 50.0),
                expected_psnr_range=(30.0, 45.0),
                expected_ssim_range=(0.85, 0.98),
                gpu_memory_efficiency=0.75,
                cpu_efficiency=0.65
            ),
            
            'HybridModel': ModelPerformanceProfile(
                model_name='HybridModel',
                model_category='standard',
                expected_inference_time_128=(25.0, 60.0),
                expected_memory_128=(500.0, 1200.0),
                expected_throughput_128=(15.0, 40.0),
                expected_psnr_range=(32.0, 46.0),
                expected_ssim_range=(0.87, 0.98),
                gpu_memory_efficiency=0.7,
                cpu_efficiency=0.6
            ),
            
            'FNO2d': ModelPerformanceProfile(
                model_name='FNO2d',
                model_category='standard',
                expected_inference_time_128=(20.0, 45.0),
                expected_memory_128=(350.0, 900.0),
                expected_throughput_128=(20.0, 60.0),
                expected_psnr_range=(26.0, 40.0),
                expected_ssim_range=(0.75, 0.94),
                gpu_memory_efficiency=0.8,
                cpu_efficiency=0.7
            ),
            
            # Heavy models (high quality, resource intensive)
            'SegFormer': ModelPerformanceProfile(
                model_name='SegFormer',
                model_category='heavy',
                expected_inference_time_128=(30.0, 70.0),
                expected_memory_128=(600.0, 1500.0),
                expected_throughput_128=(10.0, 30.0),
                expected_psnr_range=(32.0, 48.0),
                expected_ssim_range=(0.88, 0.99),
                gpu_memory_efficiency=0.65,
                cpu_efficiency=0.55
            ),
            
            'VisionTransformer': ModelPerformanceProfile(
                model_name='VisionTransformer',
                model_category='heavy',
                expected_inference_time_128=(35.0, 80.0),
                expected_memory_128=(800.0, 2000.0),
                expected_throughput_128=(8.0, 25.0),
                expected_psnr_range=(34.0, 50.0),
                expected_ssim_range=(0.9, 0.995),
                gpu_memory_efficiency=0.6,
                cpu_efficiency=0.5
            ),
            
            'UNetFormer': ModelPerformanceProfile(
                model_name='UNetFormer',
                model_category='heavy',
                expected_inference_time_128=(40.0, 90.0),
                expected_memory_128=(700.0, 1800.0),
                expected_throughput_128=(8.0, 20.0),
                expected_psnr_range=(33.0, 47.0),
                expected_ssim_range=(0.89, 0.99),
                gpu_memory_efficiency=0.62,
                cpu_efficiency=0.52
            ),
        }
        
        return profiles
    
    def evaluate_model_performance(self, model_name: str, metrics: Dict[str, float], 
                                input_size: int = 128) -> Dict[str, any]:
        """Evaluate model performance against standards"""
        
        # Get model profile
        profile = self.model_profiles.get(model_name)
        if not profile:
            return {
                'status': 'unknown_model',
                'message': f'No performance profile for {model_name}'
            }
        
        # Get appropriate thresholds based on input size
        thresholds = self._get_thresholds_for_size(input_size)
        
        # Performance evaluation
        evaluation = {
            'model_name': model_name,
            'model_category': profile.model_category,
            'input_size': input_size,
            'overall_status': 'pass',
            'metrics': {},
            'warnings': [],
            'failures': []
        }
        
        # Check inference time
        inference_key = f'inference_time_ms_{input_size}'
        if inference_key in metrics:
            inference_time = metrics[inference_key]
            max_allowed = getattr(thresholds, f'max_inference_time_{input_size}')
            
            if inference_time <= max_allowed:
                evaluation['metrics']['inference_time'] = {
                    'status': 'pass',
                    'value': inference_time,
                    'threshold': max_allowed,
                    'margin': max_allowed - inference_time
                }
            else:
                evaluation['metrics']['inference_time'] = {
                    'status': 'fail',
                    'value': inference_time,
                    'threshold': max_allowed,
                    'excess': inference_time - max_allowed
                }
                evaluation['failures'].append(f"Inference time {inference_time:.1f}ms exceeds limit {max_allowed:.1f}ms")
                evaluation['overall_status'] = 'fail'
        
        # Check memory usage
        memory_key = f'memory_usage_mb_{input_size}'
        if memory_key in metrics:
            memory_usage = metrics[memory_key]
            max_allowed = getattr(thresholds, f'max_memory_{input_size}')
            
            if memory_usage <= max_allowed:
                evaluation['metrics']['memory_usage'] = {
                    'status': 'pass',
                    'value': memory_usage,
                    'threshold': max_allowed,
                    'margin': max_allowed - memory_usage
                }
            else:
                evaluation['metrics']['memory_usage'] = {
                    'status': 'fail',
                    'value': memory_usage,
                    'threshold': max_allowed,
                    'excess': memory_usage - max_allowed
                }
                evaluation['failures'].append(f"Memory usage {memory_usage:.1f}MB exceeds limit {max_allowed:.1f}MB")
                evaluation['overall_status'] = 'fail'
        
        # Check throughput
        throughput_key = f'throughput_fps_{input_size}'
        if throughput_key in metrics:
            throughput = metrics[throughput_key]
            min_required = getattr(thresholds, f'min_throughput_{input_size}')
            
            if throughput >= min_required:
                evaluation['metrics']['throughput'] = {
                    'status': 'pass',
                    'value': throughput,
                    'threshold': min_required,
                    'margin': throughput - min_required
                }
            else:
                evaluation['metrics']['throughput'] = {
                    'status': 'fail',
                    'value': throughput,
                    'threshold': min_required,
                    'deficit': min_required - throughput
                }
                evaluation['failures'].append(f"Throughput {throughput:.1f}FPS below requirement {min_required:.1f}FPS")
                evaluation['overall_status'] = 'fail'
        
        # Check quality metrics if available
        if 'psnr' in metrics:
            psnr = metrics['psnr']
            min_psnr = thresholds.min_psnr_reconstruction
            
            if psnr >= min_psnr:
                evaluation['metrics']['psnr'] = {
                    'status': 'pass',
                    'value': psnr,
                    'threshold': min_psnr,
                    'margin': psnr - min_psnr
                }
            else:
                evaluation['metrics']['psnr'] = {
                    'status': 'warning',
                    'value': psnr,
                    'threshold': min_psnr,
                    'deficit': min_psnr - psnr
                }
                evaluation['warnings'].append(f"PSNR {psnr:.1f}dB below recommended {min_psnr:.1f}dB")
        
        # Compare with expected profile
        self._compare_with_profile(evaluation, profile, metrics)
        
        return evaluation
    
    def _get_thresholds_for_size(self, input_size: int) -> PerformanceThresholds:
        """Get performance thresholds adjusted for input size"""
        # Scale factors based on typical computational complexity
        size_factors = {
            64: 0.5,   # Half the computational cost
            128: 1.0,  # Base reference
            256: 2.5,  # 2.5x computational cost (quadratic scaling)
            512: 6.0,  # 6x computational cost
        }
        
        factor = size_factors.get(input_size, 1.0)
        
        # Create scaled thresholds
        thresholds = PerformanceThresholds()
        
        # Scale time thresholds
        thresholds.max_inference_time_128 *= factor
        thresholds.max_inference_time_256 *= factor
        thresholds.max_inference_time_512 *= factor
        
        # Scale memory thresholds (less aggressive scaling)
        memory_factor = 0.5 + 0.5 * factor  # Memory scales less than compute
        thresholds.max_memory_128 *= memory_factor
        thresholds.max_memory_256 *= memory_factor
        thresholds.max_memory_512 *= memory_factor
        
        # Scale throughput requirements
        thresholds.min_throughput_128 /= factor
        thresholds.min_throughput_256 /= factor
        thresholds.min_throughput_512 /= factor
        
        return thresholds
    
    def _compare_with_profile(self, evaluation: Dict[str, any], 
                            profile: ModelPerformanceProfile, metrics: Dict[str, float]):
        """Compare performance with expected profile"""
        
        # Inference time comparison
        if 'inference_time_ms_128' in metrics:
            actual_time = metrics['inference_time_ms_128']
            expected_min, expected_max = profile.expected_inference_time_128
            
            if actual_time < expected_min:
                evaluation['warnings'].append(
                    f"Inference time {actual_time:.1f}ms faster than expected range [{expected_min:.1f}, {expected_max:.1f}]ms"
                )
            elif actual_time > expected_max:
                evaluation['warnings'].append(
                    f"Inference time {actual_time:.1f}ms slower than expected range [{expected_min:.1f}, {expected_max:.1f}]ms"
                )
        
        # Memory usage comparison
        if 'memory_usage_mb_128' in metrics:
            actual_memory = metrics['memory_usage_mb_128']
            expected_min, expected_max = profile.expected_memory_128
            
            if actual_memory < expected_min:
                evaluation['warnings'].append(
                    f"Memory usage {actual_memory:.1f}MB lower than expected range [{expected_min:.1f}, {expected_max:.1f}]MB"
                )
            elif actual_memory > expected_max:
                evaluation['warnings'].append(
                    f"Memory usage {actual_memory:.1f}MB higher than expected range [{expected_min:.1f}, {expected_max:.1f}]MB"
                )
        
        # Throughput comparison
        if 'throughput_fps_128' in metrics:
            actual_throughput = metrics['throughput_fps_128']
            expected_min, expected_max = profile.expected_throughput_128
            
            if actual_throughput > expected_max:
                evaluation['warnings'].append(
                    f"Throughput {actual_throughput:.1f}FPS higher than expected range [{expected_min:.1f}, {expected_max:.1f}]FPS"
                )
            elif actual_throughput < expected_min:
                evaluation['warnings'].append(
                    f"Throughput {actual_throughput:.1f}FPS lower than expected range [{expected_min:.1f}, {expected_max:.1f}]FPS"
                )
    
    def get_recommendations(self, evaluation: Dict[str, any]) -> List[str]:
        """Get performance improvement recommendations"""
        recommendations = []
        
        if evaluation['overall_status'] == 'fail':
            if 'inference_time' in evaluation['metrics'] and evaluation['metrics']['inference_time']['status'] == 'fail':
                recommendations.append("Consider model optimization techniques: quantization, pruning, or knowledge distillation")
                recommendations.append("Evaluate if smaller model architecture can achieve similar quality")
            
            if 'memory_usage' in evaluation['metrics'] and evaluation['metrics']['memory_usage']['status'] == 'fail':
                recommendations.append("Implement gradient checkpointing to reduce memory usage")
                recommendations.append("Consider mixed precision training (FP16) to reduce memory footprint")
            
            if 'throughput' in evaluation['metrics'] and evaluation['metrics']['throughput']['status'] == 'fail':
                recommendations.append("Optimize batch processing and implement efficient data loading")
                recommendations.append("Consider model parallelism for large models")
        
        if 'warnings' in evaluation and evaluation['warnings']:
            if any('PSNR' in warning for warning in evaluation['warnings']):
                recommendations.append("Improve model architecture or training strategy to enhance reconstruction quality")
            
            if any('faster than expected' in warning for warning in evaluation['warnings']):
                recommendations.append("Model is performing better than expected - consider using this model for production")
            
            if any('slower than expected' in warning for warning in evaluation['warnings']):
                recommendations.append("Model is underperforming - investigate optimization opportunities")
        
        return recommendations


# Global performance standards instance
performance_standards = PerformanceStandards()


def evaluate_spatial_model_performance(model_name: str, metrics: Dict[str, float], 
                                     input_size: int = 128) -> Dict[str, any]:
    """Convenience function to evaluate spatial model performance"""
    return performance_standards.evaluate_model_performance(model_name, metrics, input_size)


def get_model_performance_recommendations(evaluation: Dict[str, any]) -> List[str]:
    """Get performance improvement recommendations"""
    return performance_standards.get_recommendations(evaluation)
