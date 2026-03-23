import torch
import torch.nn as nn
from .encoders.sparse_input_encoder import SparseInputEncoder
from .spatial.factory import create_model as create_backbone
from .decoders.bilinear3x3 import Bilinear3x3Decoder

class ModularSRModel(nn.Module):
    def __init__(self, encoder_cfg, backbone_cfg, decoder_cfg, img_size: int, out_channels: int):
        super().__init__()
        self.encoder = SparseInputEncoder(
            in_img_channels=encoder_cfg.get('in_img_channels', 1),
            out_channels=encoder_cfg.get('out_channels', backbone_cfg.get('in_channels', 64)),
            use_coords=encoder_cfg.get('use_coords', True),
            use_mask=encoder_cfg.get('use_mask', True),
            use_pe=encoder_cfg.get('use_pe', False),
        )
        bb_args = dict(backbone_cfg)
        bb_args['in_channels'] = encoder_cfg.get('out_channels', bb_args.get('in_channels', 64))
        bb_args['out_channels'] = out_channels
        bb_args['img_size'] = img_size
        self.backbone = create_backbone(backbone_cfg['name'], **bb_args)
        self.decoder = Bilinear3x3Decoder(in_channels=out_channels, out_channels=out_channels)
        self.img_size = img_size

    def forward(self, x_img: torch.Tensor, coords: torch.Tensor = None, mask: torch.Tensor = None, fourier_pe: torch.Tensor = None):
        x_enc = self.encoder(x_img, coords=coords, mask=mask, fourier_pe=fourier_pe)
        y = self.backbone(x_enc)
        y = self.decoder(y, target_size=(self.img_size, self.img_size))
        return y

