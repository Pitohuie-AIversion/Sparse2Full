import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

# Try to import project models
try:
    from models.sequential_spatiotemporal import SequentialSpatiotemporalModel
    from models.swin_unet import SwinUNet
    from models.ar.wrapper import ARWrapper
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logger.warning("Project models not found. Using fallback/dummy implementations.")

    class SequentialSpatiotemporalModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.spatial_module = nn.Identity()
            self.temporal_module = nn.Identity()
        def forward(self, x, target=None):
            return {'final_pred': x, 'spatial_pred': x}
        def set_epoch(self, epoch): pass

    class SwinUNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x): return x

    class ARWrapper(nn.Module):
        def __init__(self, model, **kwargs):
            super().__init__()
            self.model = model
        def forward(self, x): return self.model(x)

class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_spatiotemporal_model(config: Union[DictConfig, Dict[str, Any]], device: torch.device) -> nn.Module:
        """Create a SequentialSpatiotemporalModel."""
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
            
        spatial_config = config.get('spatial', {})
        temporal_config = config.get('temporal', {})
        data_config = config.get('data', {})
        
        model = SequentialSpatiotemporalModel(
            spatial_config=spatial_config,
            temporal_config=temporal_config,
            data_config=data_config,
            device=str(device)
        )
        return model.to(device)

    @staticmethod
    def create_ar_model(config: Union[DictConfig, Dict[str, Any]], device: torch.device) -> nn.Module:
        """Create a SwinUNet wrapped in ARWrapper."""
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
            
        model_cfg = config.get('model', {})
        
        base_model = SwinUNet(
            in_channels=model_cfg.get('in_channels', 2),
            out_channels=model_cfg.get('out_channels', 2),
            img_size=model_cfg.get('img_size', 256),
            patch_size=model_cfg.get('patch_size', 4),
            window_size=model_cfg.get('window_size', 8),
            depths=model_cfg.get('depths', [2, 2, 2, 2]),
            num_heads=model_cfg.get('num_heads', [3, 6, 12, 24]),
            embed_dim=model_cfg.get('embed_dim', 48),
            mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            drop_rate=model_cfg.get('drop_rate', 0.1),
            attn_drop_rate=model_cfg.get('attn_drop_rate', 0.1),
            drop_path_rate=model_cfg.get('drop_path_rate', 0.1),
        )
        
        model = ARWrapper(base_model, T_out=config.get('data', {}).get('T_out', 5))
        return model.to(device)
