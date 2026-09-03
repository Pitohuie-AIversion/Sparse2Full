import sys
sys.path.append("/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full")
from omegaconf import OmegaConf
from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporalModel
import torch
from thop import profile

spatial_config = {
  "in_channels": 1,
  "spatial_feature_dim": 64,
  "out_channels": 1,
  "img_size": [128, 128],
  "backbone_type": "edsr",
  "backbone_config": {
    "n_feats": 64,
    "n_resblocks": 32,
    "res_scale": 0.1,
    "upscale": 1
  }
}
temporal_config = {
  "spatial_feature_dim": 64,
  "temporal_dim": 96,
  "out_channels": 1,
  "img_size": [128, 128],
  "backend": "video_swin",
  "num_heads": 4,
  "num_layers": 2,
  "window_size": [2, 8, 8],
  "dropout": 0.1
}
data_config = {
  "t_in": 10,
  "t_out": 10
}
from omegaconf import DictConfig
cfg_s = DictConfig(spatial_config)
cfg_t = DictConfig(temporal_config)
cfg_d = DictConfig(data_config)
model = SequentialSpatiotemporalModel(spatial_config=cfg_s, temporal_config=cfg_t, data_config=cfg_d)
print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
x = torch.randn(1, 10, 1, 128, 128)
macs, params = profile(model, inputs=(x,), verbose=False)
print(f"FLOPs: {macs * 2 / 1e9 / 10:.2f}G per step")  # Divide by 10 steps
