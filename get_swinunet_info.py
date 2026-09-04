import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
from omegaconf import OmegaConf
from models.temporal.factory import create_model
import torch
from thop import profile

cfg_dict = {
  "model_name": "SwinUNet",
  "in_channels": 1,
  "out_channels": 1,
  "img_size": [128, 128]
}
from omegaconf import DictConfig
cfg = DictConfig(cfg_dict)
model = create_model(**cfg)
print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
x = torch.randn(1, 10, 1, 128, 128)
macs, params = profile(model, inputs=(x,), verbose=False)
print(f"FLOPs: {macs * 2 / 1e9 / 10:.2f}G per step")
