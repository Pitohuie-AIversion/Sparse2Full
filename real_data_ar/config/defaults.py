from omegaconf import DictConfig

DEFAULT_CONFIG = DictConfig({
    'experiment': {
        'name': 'spatiotemporal_decomposition', 'seed': 42, 
        'output_dir': 'runs/spatiotemporal', 'device': 'cuda',
        'precision': '32', 'log_every_n_steps': 10,
        'save_config_snapshot': True
    },
    'data': {
        'data_path': 'data/real_diffusion_reaction.h5', 'T_in': 1, 'T_out': 5,
        'img_size': 256, 'channels': 2, 'train_ratio': 0.7,
        'val_ratio': 0.15, 'test_ratio': 0.15, 'normalize': True,
        'augmentation': {'enabled': False}, 'keys': ['u', 'v'],
        'dataloader': {
            'batch_size': 4, 'val_batch_size': 4, 'test_batch_size': 2,
            'num_workers': 4, 'pin_memory': True, 'persistent_workers': True,
            'drop_last': True, 'shuffle': True, 'prefetch_factor': 2
        }
    },
    'spatial': {
        'feature_dim': 128, 'pretrain_epochs': 20, 'lr': 1e-4,
        'weight_decay': 1e-4, 'loss_weight': 1.0
    },
    'temporal': {
        'd_model': 256, 'nhead': 8, 'num_layers': 6,
        'dim_feedforward': 1024, 'dropout': 0.1,
        'encoder_type': 'transformer', 'use_spatial_features': True,
        'pretrain_epochs': 20, 'lr': 1e-4, 'weight_decay': 1e-4,
        'loss_weight': 1.0
    },
    'joint': {
        'epochs': 50, 'lr': 5e-5, 'weight_decay': 1e-4,
        'spatial_lr_ratio': 0.1, 'temporal_lr_ratio': 1.0,
        'loss_weights': {'spatial': 0.5, 'temporal': 1.0, 'consistency': 0.1}
    },
    'model': {
        'name': 'SequentialSpatiotemporalModel',
        'in_channels': 2, 'out_channels': 2, 'img_size': 256,
        'patch_size': 4, 'window_size': 8, 'depths': [2, 2, 2, 2],
        'num_heads': [3, 6, 12, 24], 'embed_dim': 48, 'mlp_ratio': 4.0,
        'drop_rate': 0.1, 'attn_drop_rate': 0.1, 'drop_path_rate': 0.1
    },
    'training': {
        'spatial_lr': 1e-4, 'temporal_lr': 1e-4, 'joint_lr': 5e-5,
        'spatial_weight_decay': 1e-4, 'temporal_weight_decay': 1e-4, 'joint_weight_decay': 1e-4,
        'spatial_epochs': 20, 'temporal_epochs': 20, 'joint_epochs': 50,
        'spatial_batch_size': 4, 'temporal_batch_size': 4, 'joint_batch_size': 4,
        'spatial_lr_ratio': 0.1, 'temporal_lr_ratio': 1.0,
        'spatial_scheduler': {'name': 'CosineAnnealingLR', 'T_max': 20, 'eta_min': 1e-6},
        'temporal_scheduler': {'name': 'CosineAnnealingLR', 'T_max': 20, 'eta_min': 1e-6},
        'joint_scheduler': {'name': 'CosineAnnealingLR', 'T_max': 50, 'eta_min': 1e-6},
        'spatial_stage': {
            'enabled': True, 'epochs': 20, 'batch_size': 4,
            'learning_rate': 1e-4, 'weight_decay': 1e-4
        },
        'temporal_stage': {
            'enabled': True, 'epochs': 20, 'batch_size': 4,
            'learning_rate': 1e-4, 'weight_decay': 1e-4
        },
        'joint_stage': {
            'enabled': True, 'epochs': 50, 'batch_size': 4,
            'learning_rate': 5e-5, 'weight_decay': 1e-4
        },
        'gradient_clip_val': 1.0, 'accumulate_grad_batches': 1,
        'scheduler': {'name': 'CosineAnnealingLR', 'T_max': 50, 'eta_min': 1e-6}
    },
    'loss': {
        'reconstruction': {'weight': 1.0}, 'spectral': {'weight': 0.5},
        'data_consistency': {'weight': 1.0}, 'degradation_consistency': {'weight': 0.0},
        'gradient_weight': 0.0, 'temporal_consistency': {'weight': 0.1}
    },
    'validation': {
        'check_val_every_n_epoch': 1, 'val_check_interval': 1.0,
        'metrics': ['rel_l2', 'mae', 'mse']
    },
    'observation': {
        'mode': 'identity', 'scale_factor': 1, 'blur_sigma': 0.0,
        'kernel_size': 1, 'boundary': 'mirror', 'downsample_interpolation': 'area'
    }
})
