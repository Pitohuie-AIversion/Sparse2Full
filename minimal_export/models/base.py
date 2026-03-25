import torch.nn as nn

class BaseModel(nn.Module):
    """Base model class for all spatial and temporal models."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Many models expect these to be set by the base class
        if len(args) >= 1:
            self.in_channels = args[0]
        if len(args) >= 2:
            self.out_channels = args[1]
        if len(args) >= 3:
            self.img_size = args[2]
            
        if 'in_channels' in kwargs:
            self.in_channels = kwargs['in_channels']
        if 'out_channels' in kwargs:
            self.out_channels = kwargs['out_channels']
        if 'img_size' in kwargs:
            self.img_size = kwargs['img_size']
