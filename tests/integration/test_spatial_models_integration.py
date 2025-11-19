"""
Integration and end-to-end tests for spatial prediction models.
Tests model integration with data pipelines, training workflows, and real-world scenarios.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import os
from pathlib import Path
import json
from omegaconf import OmegaConf

# Import spatial models
from models import (
    UNet, UNetPlusPlus, FNO2d, UFNOUNet,
    SegFormer, UNetFormer, SegFormerUNetFormer,
    VisionTransformer, SwinTransformerTiny, Transformer,
    SwinUNet, HybridModel, MLPModel, MLPMixer, LIIFModel,
    SparseAttentionEncoder, SparseSwinUNet
)


class SyntheticSpatialDataset(Dataset):
    """Synthetic dataset for spatial model testing"""
    
    def __init__(self, num_samples: int = 100, input_shape: Tuple[int, int, int, int] = (3, 128, 128), 
                 task_type: str = 'reconstruction', noise_level: float = 0.1):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.task_type = task_type
        self.noise_level = noise_level
        
        # Generate synthetic data
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate synthetic spatial data"""
        data = []
        
        for i in range(self.num_samples):
            # Create base pattern
            if self.task_type == 'reconstruction':
                # Clean input, noisy target (denoising task)
                clean = self._create_pattern(i)
                noisy = clean + self.noise_level * torch.randn_like(clean)
                data.append((noisy, clean))
            
            elif self.task_type == 'super_resolution':
                # High-res target, low-res input
                high_res = self._create_pattern(i, scale=1.0)
                low_res = torch.nn.functional.interpolate(
                    high_res.unsqueeze(0), 
                    scale_factor=0.5, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                # Upsample low-res to match high-res size for training
                low_res_upsampled = torch.nn.functional.interpolate(
                    low_res.unsqueeze(0), 
                    scale_factor=2.0, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                data.append((low_res_upsampled, high_res))
            
            elif self.task_type == 'segmentation':
                # Input image, segmentation mask
                input_img = self._create_pattern(i)
                mask = (input_img > 0.5).float()
                data.append((input_img, mask))
            
            else:
                # Default: identity mapping
                pattern = self._create_pattern(i)
                data.append((pattern, pattern))
        
        return data
    
    def _create_pattern(self, seed: int, scale: float = 1.0) -> torch.Tensor:
        """Create a synthetic spatial pattern"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        channels, height, width = self.input_shape
        
        # Scale dimensions
        scaled_height = int(height * scale)
        scaled_width = int(width * scale)
        
        # Create base pattern
        pattern = torch.zeros(channels, scaled_height, scaled_width)
        
        # Add some structured patterns
        for c in range(channels):
            # Create checkerboard pattern
            checker = torch.zeros(scaled_height, scaled_width)
            for i in range(0, scaled_height, 8):
                for j in range(0, scaled_width, 8):
                    if (i // 8 + j // 8) % 2 == 0:
                        checker[i:i+8, j:j+8] = 1.0
            
            # Add some circles
            center_y, center_x = scaled_height // 2, scaled_width // 2
            y, x = torch.meshgrid(torch.arange(scaled_height), torch.arange(scaled_width))
            circle = ((y - center_y)**2 + (x - center_x)**2 < (min(scaled_height, scaled_width) // 4)**2).float()
            
            # Combine patterns
            pattern[c] = 0.3 * checker + 0.7 * circle + 0.1 * torch.randn_like(checker)
            pattern[c] = torch.clamp(pattern[c], 0, 1)
        
        return pattern
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


class SpatialModelIntegrationTest:
    """Integration test suite for spatial models"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.test_results = []
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def create_model(self, model_class: type, config: Dict[str, Any]) -> nn.Module:
        """Create model with given configuration"""
        try:
            model = model_class(**config)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Failed to create {model_class.__name__}: {str(e)}")
            return None
    
    def create_optimizer(self, model: nn.Module, lr: float = 1e-3) -> optim.Optimizer:
        """Create optimizer for model"""
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    def create_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    def create_loss_function(self, task_type: str) -> nn.Module:
        """Create loss function based on task type"""
        if task_type == 'segmentation':
            return nn.BCEWithLogitsLoss()
        else:
            return nn.MSELoss()
    
    def train_model_for_integration(self, model: nn.Module, train_loader: DataLoader, 
                                   val_loader: DataLoader, num_epochs: int = 3,
                                   task_type: str = 'reconstruction') -> Dict[str, List[float]]:
        """Train model for integration testing"""
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer, num_epochs)
        criterion = self.create_loss_function(task_type)
        
        model.train()
        training_history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            # Record metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['lr'].append(current_lr)
            
            # Step scheduler
            scheduler.step()
            model.train()
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        return training_history
    
    def evaluate_model_integration(self, model: nn.Module, test_loader: DataLoader, 
                                 task_type: str = 'reconstruction') -> Dict[str, float]:
        """Evaluate model integration"""
        criterion = self.create_loss_function(task_type)
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                predictions.append(outputs.cpu())
                targets_list.append(targets.cpu())
        
        avg_loss = total_loss / total_samples
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(predictions, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        # Calculate additional metrics based on task type
        metrics = {'test_loss': avg_loss}
        
        if task_type == 'reconstruction':
            # Calculate PSNR and SSIM-like metrics
            mse = torch.mean((all_predictions - all_targets) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            metrics['psnr'] = psnr.item()
            
            # Simple SSIM approximation
            mu_pred = torch.mean(all_predictions)
            mu_target = torch.mean(all_targets)
            sigma_pred = torch.std(all_predictions)
            sigma_target = torch.std(all_targets)
            sigma_pred_target = torch.mean((all_predictions - mu_pred) * (all_targets - mu_target))
            
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
                   ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred**2 + sigma_target**2 + c2))
            metrics['ssim'] = ssim.item()
        
        elif task_type == 'segmentation':
            # Calculate accuracy and IoU
            pred_binary = (torch.sigmoid(all_predictions) > 0.5).float()
            target_binary = (all_targets > 0.5).float()
            
            accuracy = (pred_binary == target_binary).float().mean()
            metrics['accuracy'] = accuracy.item()
            
            # IoU calculation
            intersection = (pred_binary * target_binary).sum()
            union = pred_binary.sum() + target_binary.sum() - intersection
            iou = intersection / (union + 1e-8)
            metrics['iou'] = iou.item()
        
        return metrics
    
    def test_model_data_pipeline(self, model_class: type, task_type: str = 'reconstruction') -> Dict[str, Any]:
        """Test model integration with data pipeline"""
        print(f"\nTesting {model_class.__name__} with {task_type} task...")
        
        # Create synthetic dataset
        train_dataset = SyntheticSpatialDataset(num_samples=50, task_type=task_type)
        val_dataset = SyntheticSpatialDataset(num_samples=20, task_type=task_type)
        test_dataset = SyntheticSpatialDataset(num_samples=30, task_type=task_type)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # Create model configuration
        config = self._get_model_config(model_class, task_type)
        model = self.create_model(model_class, config)
        
        if model is None:
            return {'status': 'failed', 'error': 'Model creation failed'}
        
        try:
            # Train model
            training_history = self.train_model_for_integration(
                model, train_loader, val_loader, num_epochs=2, task_type=task_type
            )
            
            # Evaluate model
            test_metrics = self.evaluate_model_integration(model, test_loader, task_type)
            
            # Check if training was successful
            training_success = self._check_training_success(training_history, task_type)
            
            result = {
                'status': 'success' if training_success else 'partial_success',
                'model_name': model_class.__name__,
                'task_type': task_type,
                'training_history': training_history,
                'test_metrics': test_metrics,
                'final_train_loss': training_history['train_loss'][-1],
                'final_val_loss': training_history['val_loss'][-1],
                'test_loss': test_metrics['test_loss'],
            }
            
            if task_type == 'reconstruction':
                result['psnr'] = test_metrics.get('psnr', 0.0)
                result['ssim'] = test_metrics.get('ssim', 0.0)
            elif task_type == 'segmentation':
                result['accuracy'] = test_metrics.get('accuracy', 0.0)
                result['iou'] = test_metrics.get('iou', 0.0)
            
            return result
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'model_name': model_class.__name__,
                'task_type': task_type
            }
    
    def _get_model_config(self, model_class: type, task_type: str) -> Dict[str, Any]:
        """Get appropriate model configuration"""
        base_config = {
            'in_ch': 3,
            'out_ch': 3,
            'img_size': 128,
        }
        
        # Model-specific configurations
        model_configs = {
            UNet: {'features': [32, 64, 128], 'bilinear': True},
            UNetPlusPlus: {'features': [32, 64, 128], 'deep_supervision': False},
            FNO2d: {'modes': 16, 'width': 32, 'layers': 4},
            UFNOUNet: {'modes': 16, 'width': 32, 'layers': 4, 'bilinear': True},
            SwinUNet: {'depths': [2, 2, 2], 'num_heads': [3, 6, 12], 'window_size': 8},
            HybridModel: {'backbone': 'swin', 'fusion': 'concat', 'attention_ch': 64},
            MLPModel: {'hidden_dim': 256, 'num_layers': 4, 'use_coords': True},
            MLPMixer: {'patch_size': 16, 'hidden_dim': 256, 'num_blocks': 6},
            LIIFModel: {'hidden_dim': 256, 'num_layers': 4, 'coord_encode': True},
            SegFormer: {'backbone': 'b0', 'embed_dim': 256},
            UNetFormer: {'backbone': 'resnet34', 'num_heads': 8},
            SegFormerUNetFormer: {'segformer_backbone': 'b0', 'unetformer_backbone': 'resnet34', 'fusion': 'concat'},
            VisionTransformer: {'patch_size': 16, 'embed_dim': 384, 'depth': 6},
            SwinTransformerTiny: {'depths': [2, 2, 6], 'num_heads': [3, 6, 12], 'embed_dim': 96},
            Transformer: {'d_model': 256, 'nhead': 8, 'num_layers': 4},
            SparseAttentionEncoder: {'embed_dim': 256, 'num_heads': 8, 'sparse_ratio': 0.5},
            SparseSwinUNet: {'depths': [2, 2, 2], 'num_heads': [3, 6, 12], 'sparse_ratio': 0.5},
        }
        
        config = base_config.copy()
        if model_class in model_configs:
            config.update(model_configs[model_class])
        
        return config
    
    def _check_training_success(self, training_history: Dict[str, List[float]], task_type: str) -> bool:
        """Check if training was successful"""
        if not training_history['train_loss'] or not training_history['val_loss']:
            return False
        
        # Check if loss decreased
        initial_train_loss = training_history['train_loss'][0]
        final_train_loss = training_history['train_loss'][-1]
        
        initial_val_loss = training_history['val_loss'][0]
        final_val_loss = training_history['val_loss'][-1]
        
        # Training is successful if loss decreased significantly
        train_loss_improved = final_train_loss < initial_train_loss * 0.95
        val_loss_improved = final_val_loss < initial_val_loss * 0.95
        
        # Check for reasonable loss values
        reasonable_loss = final_train_loss < 1.0 and final_val_loss < 1.0
        
        return train_loss_improved and val_loss_improved and reasonable_loss
    
    def run_integration_tests(self, model_classes: List[type], task_types: List[str]) -> Dict[str, Any]:
        """Run complete integration test suite"""
        results = {
            'summary': {
                'total_models': len(model_classes),
                'total_tasks': len(task_types),
                'successful': 0,
                'partial_success': 0,
                'failed': 0,
                'start_time': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
                'end_time': None
            },
            'detailed_results': []
        }
        
        if results['summary']['start_time']:
            results['summary']['start_time'].record()
        
        for model_class in model_classes:
            for task_type in task_types:
                try:
                    result = self.test_model_data_pipeline(model_class, task_type)
                    results['detailed_results'].append(result)
                    
                    if result['status'] == 'success':
                        results['summary']['successful'] += 1
                    elif result['status'] == 'partial_success':
                        results['summary']['partial_success'] += 1
                    else:
                        results['summary']['failed'] += 1
                        
                except Exception as e:
                    error_result = {
                        'status': 'failed',
                        'error': str(e),
                        'model_name': model_class.__name__,
                        'task_type': task_type
                    }
                    results['detailed_results'].append(error_result)
                    results['summary']['failed'] += 1
        
        if results['summary']['start_time']:
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            results['summary']['end_time'] = end_time
        
        # Calculate success rate
        total_tests = len(results['detailed_results'])
        success_rate = (results['summary']['successful'] / total_tests * 100) if total_tests > 0 else 0
        results['summary']['success_rate'] = success_rate
        
        return results
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save integration test results"""
        # Convert to JSON-serializable format
        serializable_results = {
            'summary': results['summary'].copy(),
            'detailed_results': []
        }
        
        # Convert detailed results
        for result in results['detailed_results']:
            serializable_result = result.copy()
            
            # Convert training history lists to JSON-serializable format
            if 'training_history' in result:
                serializable_result['training_history'] = {
                    key: [float(x) for x in values] 
                    for key, values in result['training_history'].items()
                }
            
            serializable_results['detailed_results'].append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable test report"""
        summary = results['summary']
        
        report = f"""
SPATIAL MODEL INTEGRATION TEST REPORT
=====================================

Test Summary:
- Total Models Tested: {summary['total_models']}
- Total Tasks: {summary['total_tasks']}
- Successful Tests: {summary['successful']}
- Partial Success: {summary['partial_success']}
- Failed Tests: {summary['failed']}
- Success Rate: {summary['success_rate']:.1f}%

Detailed Results by Model:
"""
        
        # Group results by model
        model_results = {}
        for result in results['detailed_results']:
            model_name = result['model_name']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        for model_name, model_res in model_results.items():
            report += f"\n{model_name}:\n"
            for res in model_res:
                status = res['status']
                task_type = res['task_type']
                
                if status == 'success':
                    final_loss = res['final_val_loss']
                    test_loss = res['test_loss']
                    report += f"  ✓ {task_type}: Val Loss: {final_loss:.4f}, Test Loss: {test_loss:.4f}"
                    
                    if 'psnr' in res:
                        report += f", PSNR: {res['psnr']:.2f}dB"
                    if 'accuracy' in res:
                        report += f", Accuracy: {res['accuracy']:.3f}"
                    
                    report += "\n"
                
                elif status == 'partial_success':
                    report += f"  ⚠ {task_type}: Partial success (check logs)\n"
                
                else:
                    error = res.get('error', 'Unknown error')
                    report += f"  ✗ {task_type}: Failed - {error}\n"
        
        return report


# Pytest integration
@pytest.mark.integration
class TestSpatialModelIntegration:
    """Integration tests for spatial models"""
    
    @pytest.fixture
    def integration_tester(self):
        """Create integration tester"""
        return SpatialModelIntegrationTest()
    
    @pytest.fixture
    def core_models(self):
        """Core spatial models for integration testing"""
        return [
            UNet, SwinUNet, HybridModel, MLPModel, FNO2d
        ]
    
    @pytest.fixture
    def all_spatial_models(self):
        """All spatial models for comprehensive testing"""
        return [
            UNet, UNetPlusPlus, FNO2d, UFNOUNet,
            SegFormer, UNetFormer, SegFormerUNetFormer,
            VisionTransformer, SwinTransformerTiny, Transformer,
            SwinUNet, HybridModel, MLPModel, MLPMixer, LIIFModel,
            SparseAttentionEncoder, SparseSwinUNet
        ]
    
    def test_core_models_reconstruction(self, integration_tester, core_models):
        """Test core models with reconstruction task"""
        results = integration_tester.run_integration_tests(
            core_models, ['reconstruction']
        )
        
        # Check that most models succeeded
        success_rate = results['summary']['success_rate']
        assert success_rate >= 80.0, f"Core models reconstruction success rate too low: {success_rate:.1f}%"
        
        # Save results
        results_dir = Path('tests/results')
        results_dir.mkdir(exist_ok=True)
        integration_tester.save_results(results, str(results_dir / 'core_models_integration.json'))
        
        print(integration_tester.generate_report(results))
    
    def test_core_models_segmentation(self, integration_tester, core_models):
        """Test core models with segmentation task"""
        results = integration_tester.run_integration_tests(
            core_models, ['segmentation']
        )
        
        # Check that most models succeeded
        success_rate = results['summary']['success_rate']
        assert success_rate >= 70.0, f"Core models segmentation success rate too low: {success_rate:.1f}%"
        
        # Save results
        results_dir = Path('tests/results')
        results_dir.mkdir(exist_ok=True)
        integration_tester.save_results(results, str(results_dir / 'core_models_segmentation.json'))
        
        print(integration_tester.generate_report(results))
    
    @pytest.mark.slow
    def test_all_models_comprehensive(self, integration_tester, all_spatial_models):
        """Comprehensive test of all spatial models with multiple tasks"""
        task_types = ['reconstruction', 'segmentation', 'super_resolution']
        
        results = integration_tester.run_integration_tests(
            all_spatial_models, task_types
        )
        
        # Check overall success rate
        success_rate = results['summary']['success_rate']
        assert success_rate >= 60.0, f"Overall success rate too low: {success_rate:.1f}%"
        
        # Save comprehensive results
        results_dir = Path('tests/results')
        results_dir.mkdir(exist_ok=True)
        integration_tester.save_results(results, str(results_dir / 'all_models_comprehensive.json'))
        
        # Generate and save report
        report = integration_tester.generate_report(results)
        with open(results_dir / 'comprehensive_test_report.txt', 'w') as f:
            f.write(report)
        
        print(report)


# End-to-end training pipeline test
class EndToEndTrainingPipeline:
    """End-to-end test of complete training pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config_path) if config_path else self._get_default_config()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return OmegaConf.load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model': {
                'name': 'SwinUNet',
                'in_ch': 3,
                'out_ch': 3,
                'img_size': 128,
                'depths': [2, 2, 2, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8
            },
            'training': {
                'num_epochs': 5,
                'batch_size': 4,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'data': {
                'num_train_samples': 100,
                'num_val_samples': 20,
                'num_test_samples': 30,
                'task_type': 'reconstruction'
            }
        }
    
    def run_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Run complete end-to-end training pipeline"""
        print("Running end-to-end training pipeline test...")
        
        try:
            # Create datasets
            train_dataset = SyntheticSpatialDataset(
                num_samples=self.config['data']['num_train_samples'],
                task_type=self.config['data']['task_type']
            )
            val_dataset = SyntheticSpatialDataset(
                num_samples=self.config['data']['num_val_samples'],
                task_type=self.config['data']['task_type']
            )
            test_dataset = SyntheticSpatialDataset(
                num_samples=self.config['data']['num_test_samples'],
                task_type=self.config['data']['task_type']
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
            
            # Create model
            model_class = globals()[self.config['model']['name']]
            model_config = {k: v for k, v in self.config['model'].items() if k != 'name'}
            model = model_class(**model_config).to(self.device)
            
            # Create optimizer and scheduler
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=self.config['training']['learning_rate'],
                                  weight_decay=self.config['training']['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['training']['num_epochs'])
            
            # Training loop
            criterion = nn.MSELoss()
            training_history = {'train_loss': [], 'val_loss': [], 'lr': []}
            
            print(f"Training {self.config['model']['name']} for {self.config['training']['num_epochs']} epochs...")
            
            for epoch in range(self.config['training']['num_epochs']):
                # Training
                model.train()
                train_loss = 0.0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, targets).item()
                
                # Record metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                current_lr = optimizer.param_groups[0]['lr']
                
                training_history['train_loss'].append(avg_train_loss)
                training_history['val_loss'].append(avg_val_loss)
                training_history['lr'].append(current_lr)
                
                scheduler.step()
                
                print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}: "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Final evaluation
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
            
            avg_test_loss = test_loss / len(test_loader)
            
            # Check success criteria
            training_success = self._evaluate_training_success(training_history, avg_test_loss)
            
            result = {
                'status': 'success' if training_success else 'partial_success',
                'model_name': self.config['model']['name'],
                'training_history': training_history,
                'final_train_loss': training_history['train_loss'][-1],
                'final_val_loss': training_history['val_loss'][-1],
                'test_loss': avg_test_loss,
                'training_success': training_success
            }
            
            print(f"End-to-end pipeline completed. Final test loss: {avg_test_loss:.4f}")
            return result
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'model_name': self.config['model']['name']
            }
    
    def _evaluate_training_success(self, training_history: Dict[str, List[float]], 
                                 final_test_loss: float) -> bool:
        """Evaluate if training was successful"""
        if not training_history['train_loss'] or not training_history['val_loss']:
            return False
        
        # Check loss improvement
        initial_train = training_history['train_loss'][0]
        final_train = training_history['train_loss'][-1]
        initial_val = training_history['val_loss'][0]
        final_val = training_history['val_loss'][-1]
        
        # Success criteria
        train_improved = final_train < initial_train * 0.8
        val_improved = final_val < initial_val * 0.8
        reasonable_final_loss = final_test_loss < 0.1
        
        return train_improved and val_improved and reasonable_final_loss


@pytest.mark.e2e
class TestEndToEndPipeline:
    """End-to-end tests for complete training pipeline"""
    
    def test_swin_unet_end_to_end(self):
        """Test SwinUNet end-to-end training pipeline"""
        pipeline = EndToEndTrainingPipeline()
        result = pipeline.run_end_to_end_pipeline()
        
        assert result['status'] != 'failed', f"End-to-end pipeline failed: {result.get('error', 'Unknown error')}"
        assert result['training_success'], "Training did not meet success criteria"
        
        # Save results
        results_dir = Path('tests/results')
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'swin_unet_e2e.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"End-to-end test completed. Status: {result['status']}")


if __name__ == "__main__":
    # Run integration tests
    tester = SpatialModelIntegrationTest()
    
    # Test core models
    core_models = [UNet, SwinUNet, HybridModel, MLPModel, FNO2d]
    task_types = ['reconstruction', 'segmentation']
    
    print("Running spatial model integration tests...")
    results = tester.run_integration_tests(core_models, task_types)
    
    # Save and display results
    results_dir = Path('tests/results')
    results_dir.mkdir(exist_ok=True)
    
    tester.save_results(results, str(results_dir / 'integration_test_results.json'))
    report = tester.generate_report(results)
    
    with open(results_dir / 'integration_test_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    
    # Run end-to-end test
    print("\nRunning end-to-end pipeline test...")
    pipeline = EndToEndTrainingPipeline()
    e2e_result = pipeline.run_end_to_end_pipeline()
    
    with open(results_dir / 'e2e_test_result.json', 'w') as f:
        json.dump(e2e_result, f, indent=2)
    
    print(f"End-to-end test status: {e2e_result['status']}")
    
    print(f"\nAll test results saved to {results_dir}")