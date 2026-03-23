import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from omegaconf import DictConfig
import json

from ..config.manager import SpatiotemporalConfigManager
from ..data.module import RealDiffusionReactionDataModule
from ..models.factory import ModelFactory
from ..utils.common import DeviceManager, LogManager, convert_numpy_types

class SpatiotemporalTrainer:
    """Spatiotemporal decomposition trainer implementing the three-stage training process.
    
    Stages:
    1. Spatial Pretraining: Train spatial reconstruction module.
    2. Temporal Pretraining: Train temporal prediction module.
    3. Joint Finetuning: Train the entire model end-to-end.
    """
    
    def __init__(self, config: DictConfig):
        self.config = SpatiotemporalConfigManager.validate_config(config)
        self.device_manager = DeviceManager(self.config)
        self.device = self.device_manager.setup_device()
        
        self.output_dir = Path(self.config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Logger
        self.log_manager = LogManager(self.config, self.output_dir)
        self.logger = self.log_manager.setup_logging()
        
        # Setup Data
        self.data_module = RealDiffusionReactionDataModule(self.config)
        
        # Setup Model
        self.model = ModelFactory.create_spatiotemporal_model(self.config, self.device)
        
        # Training state
        self.current_stage = 'spatial'
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        
    def setup(self) -> bool:
        """Setup training environment."""
        try:
            self.data_module.setup()
            self.logger.info("Data module setup complete.")
            return True
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            return False

    def train(self) -> Dict[str, Any]:
        """Execute the full three-stage training process."""
        self.logger.info("Starting Spatiotemporal Decomposition Training")
        
        # Stage 1: Spatial Pretraining
        spatial_history = []
        if self.config.training.spatial_stage.enabled:
            spatial_history = self.train_spatial_stage()
            
        # Stage 2: Temporal Pretraining
        temporal_history = []
        if self.config.training.temporal_stage.enabled:
            temporal_history = self.train_temporal_stage()
            
        # Stage 3: Joint Finetuning
        joint_history = []
        if self.config.training.joint_stage.enabled:
            joint_history = self.train_joint_stage()
            
        full_history = {
            'spatial': spatial_history,
            'temporal': temporal_history,
            'joint': joint_history
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(convert_numpy_types(full_history), f, indent=2)
            
        self.logger.info("Training completed!")
        return full_history

    def train_spatial_stage(self):
        """Execute spatial pretraining stage."""
        self.logger.info("=== Starting Spatial Pretraining Stage ===")
        self.current_stage = 'spatial'
        
        train_loader = self.data_module.spatial_train_loader()
        val_loader = self.data_module.val_dataloader()
        
        cfg = self.config.training.spatial_stage
        optimizer = torch.optim.AdamW(
            self.model.spatial_module.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        history = []
        for epoch in range(cfg.epochs):
            self.current_epoch = epoch
            
            # Train
            self.model.spatial_module.train()
            train_loss = 0.0
            batches = 0
            for batch in train_loader:
                x = batch['input_sequence'].to(self.device)
                y = batch['target_sequence'].to(self.device)
                
                # Spatial module usually takes [B, T, C, H, W] or [B, C, H, W]
                # For spatial pretraining, we might want to flatten time
                if x.dim() == 5:
                    b, t, c, h, w = x.shape
                    x_flat = x.reshape(b*t, c, h, w)
                    y_flat = y.reshape(b*t, c, h, w)
                else:
                    x_flat, y_flat = x, y

                pred = self.model.spatial_module(x_flat)
                # If prediction returns a dict/object, extract tensor
                if hasattr(pred, 'spatial_pred'):
                    pred = pred.spatial_pred
                
                loss = nn.MSELoss()(pred, y_flat)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batches += 1
            
            avg_train_loss = train_loss / batches if batches > 0 else 0.0
            
            # Validation (Simplified)
            avg_val_loss = self._validate_spatial(val_loader)
            
            self.logger.info(f"Spatial Epoch {epoch}: Train Loss {avg_train_loss:.6f}, Val Loss {avg_val_loss:.6f}")
            history.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
            
        return history

    def _validate_spatial(self, loader) -> float:
        self.model.spatial_module.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in loader:
                x = batch['input_sequence'].to(self.device)
                y = batch['target_sequence'].to(self.device)
                if x.dim() == 5:
                    b, t, c, h, w = x.shape
                    x_flat = x.reshape(b*t, c, h, w)
                    y_flat = y.reshape(b*t, c, h, w)
                else:
                    x_flat, y_flat = x, y
                
                pred = self.model.spatial_module(x_flat)
                if hasattr(pred, 'spatial_pred'):
                    pred = pred.spatial_pred
                loss = nn.MSELoss()(pred, y_flat)
                total_loss += loss.item()
                batches += 1
        return total_loss / batches if batches > 0 else 0.0

    def train_temporal_stage(self):
        """Execute temporal pretraining stage."""
        self.logger.info("=== Starting Temporal Pretraining Stage ===")
        self.current_stage = 'temporal'
        
        train_loader = self.data_module.temporal_train_loader()
        val_loader = self.data_module.val_dataloader()
        
        cfg = self.config.training.temporal_stage
        optimizer = torch.optim.AdamW(
            self.model.temporal_module.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        history = []
        for epoch in range(cfg.epochs):
            self.current_epoch = epoch
            self.model.spatial_module.eval() # Freeze spatial
            self.model.temporal_module.train()
            
            train_loss = 0.0
            batches = 0
            for batch in train_loader:
                x = batch['input_sequence'].to(self.device)
                y = batch['target_sequence'].to(self.device)
                
                with torch.no_grad():
                    # Get spatial features
                    spatial_out = self.model.spatial_module(x)
                
                # Temporal prediction
                temp_out = self.model.temporal_module(spatial_out, target=y) # Pass target for TF if needed
                pred = temp_out.final_pred
                
                loss = nn.MSELoss()(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batches += 1
                
            avg_train_loss = train_loss / batches if batches > 0 else 0.0
            avg_val_loss = self._validate_temporal(val_loader)
            
            self.logger.info(f"Temporal Epoch {epoch}: Train Loss {avg_train_loss:.6f}, Val Loss {avg_val_loss:.6f}")
            history.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
            
        return history

    def _validate_temporal(self, loader) -> float:
        self.model.temporal_module.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in loader:
                x = batch['input_sequence'].to(self.device)
                y = batch['target_sequence'].to(self.device)
                spatial_out = self.model.spatial_module(x)
                temp_out = self.model.temporal_module(spatial_out)
                pred = temp_out.final_pred
                loss = nn.MSELoss()(pred, y)
                total_loss += loss.item()
                batches += 1
        return total_loss / batches if batches > 0 else 0.0

    def train_joint_stage(self):
        """Execute joint finetuning stage."""
        self.logger.info("=== Starting Joint Finetuning Stage ===")
        self.current_stage = 'joint'
        
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        cfg = self.config.training.joint_stage
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        history = []
        for epoch in range(cfg.epochs):
            self.current_epoch = epoch
            self.model.train()
            
            # Update scheduled sampling / teacher forcing
            if hasattr(self.model, 'set_epoch'):
                self.model.set_epoch(epoch)
            
            train_loss = 0.0
            batches = 0
            for batch in train_loader:
                x = batch['input_sequence'].to(self.device)
                y = batch['target_sequence'].to(self.device)
                
                out = self.model(x, target=y)
                pred = out['final_pred']
                
                loss = nn.MSELoss()(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batches += 1
                
            avg_train_loss = train_loss / batches if batches > 0 else 0.0
            avg_val_loss = self._validate_joint(val_loader)
            
            self.logger.info(f"Joint Epoch {epoch}: Train Loss {avg_train_loss:.6f}, Val Loss {avg_val_loss:.6f}")
            history.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
            
        return history

    def _validate_joint(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in loader:
                x = batch['input_sequence'].to(self.device)
                y = batch['target_sequence'].to(self.device)
                out = self.model(x)
                pred = out['final_pred']
                loss = nn.MSELoss()(pred, y)
                total_loss += loss.item()
                batches += 1
        return total_loss / batches if batches > 0 else 0.0
    
    def test(self) -> Dict[str, float]:
        """Test the model."""
        self.logger.info("Starting Testing")
        loader = self.data_module.test_dataloader()
        return {'test_loss': self._validate_joint(loader)}
