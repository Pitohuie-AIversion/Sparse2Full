"""LIIF (Learning Implicit Image Function) 模型

LIIF是一种基于隐式神经表示的图像超分辨率方法，通过学习连续的图像函数来实现任意倍数的超分辨率。

Reference:
    Learning Continuous Image Representation with Local Implicit Image Function
    https://arxiv.org/abs/2012.09161
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .base import BaseModel


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_list: list = [256, 256, 256, 256]
    ):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LIIF(nn.Module):
    """LIIF核心模块
    
    学习局部隐式图像函数，将坐标映射到像素值
    """
    
    def __init__(
        self, 
        encoder_spec: dict, 
        imnet_spec: dict = None,
        local_ensemble: bool = True, 
        feat_unfold: bool = True, 
        cell_decode: bool = True
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        # 编码器
        self.encoder = self._make_encoder(encoder_spec)
        
        # 隐式网络
        if imnet_spec is None:
            imnet_spec = {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256, 256, 256, 256]}}
        
        if self.feat_unfold:
            imnet_in_dim = self.encoder.out_dim * 9
        else:
            imnet_in_dim = self.encoder.out_dim
        
        if self.cell_decode:
            imnet_in_dim += 2
            
        imnet_spec['args']['in_dim'] = imnet_in_dim
        self.imnet = MLP(**imnet_spec['args'])
    
    def _make_encoder(self, encoder_spec):
        """创建编码器"""
        if encoder_spec['name'] == 'simple_cnn':
            return SimpleCNNEncoder(**encoder_spec['args'])
        else:
            raise ValueError(f"Unknown encoder: {encoder_spec['name']}")
    
    def gen_feat(self, inp):
        """生成特征"""
        return self.encoder(inp)
    
    def query_rgb(self, feat, coord, cell=None):
        """查询RGB值"""
        # 如果输入是原始图像，先通过编码器生成特征
        if feat.dim() == 4 and feat.shape[1] == self.encoder.in_channels:
            feat = self.gen_feat(feat)
        
        # 确保feat是特征张量而不是原始输入
        
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0
        
        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        
        feat_coord = self.make_coord(feat.shape[-2:], feat.device)
        # 确保feat_coord的批次维度与feat匹配
        if feat_coord.shape[0] != feat.shape[0]:
            feat_coord = feat_coord.expand(feat.shape[0], -1, -1, -1)
        
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                # 确保grid和input的批次大小一致
                grid = coord_.flip(-1).unsqueeze(1)
                if grid.shape[0] != feat.shape[0]:
                    grid = grid.expand(feat.shape[0], -1, -1, -1)
                
                q_feat = F.grid_sample(
                    feat, grid,
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                
                q_coord = F.grid_sample(
                    feat_coord, grid,
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                inp = q_feat
                
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)
                
                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)
                
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)
        
        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret
    
    def make_coord(self, shape, device):
        """生成坐标网格"""
        h, w = shape
        coord = torch.ones(1, 2, h, w, device=device)
        coord[:, 0] = torch.linspace(-1, 1, h, device=device).view(-1, 1).expand(-1, w)
        coord[:, 1] = torch.linspace(-1, 1, w, device=device).view(1, -1).expand(h, -1)
        return coord
    
    def forward(self, inp, coord, cell):
        """前向传播"""
        return self.query_rgb(inp, coord, cell)


class SimpleCNNEncoder(nn.Module):
    """简单CNN编码器"""
    
    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, out_dim, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x


class LIIFModel(BaseModel):
    """LIIF模型
    
    基于局部隐式图像函数的超分辨率模型
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        encoder_dim: int = 256,
        imnet_hidden: list = [256, 256, 256, 256],
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.encoder_dim = encoder_dim
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        # LIIF核心
        encoder_spec = {
            'name': 'simple_cnn',
            'args': {'in_channels': in_channels, 'out_dim': encoder_dim}
        }
        
        imnet_spec = {
            'name': 'mlp',
            'args': {'out_dim': out_channels, 'hidden_list': imnet_hidden}
        }
        
        self.liif = LIIF(
            encoder_spec=encoder_spec,
            imnet_spec=imnet_spec,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def make_coord(self, shape, device):
        """生成坐标网格"""
        h, w = shape
        coord = torch.ones(1, h * w, 2, device=device)
        coord[:, :, 0] = torch.linspace(-1, 1, h, device=device).view(-1, 1).expand(-1, w).contiguous().view(-1)
        coord[:, :, 1] = torch.linspace(-1, 1, w, device=device).view(1, -1).expand(h, -1).contiguous().view(-1)
        return coord
    
    def make_cell(self, shape, device):
        """生成cell"""
        h, w = shape
        cell = torch.ones(1, h * w, 2, device=device)
        cell[:, :, 0] = 2 / h
        cell[:, :, 1] = 2 / w
        return cell
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        B, C, H, W = x.shape
        
        # 生成目标坐标和cell - 修复批次维度处理
        coord = self.make_coord((H, W), x.device)
        cell = self.make_cell((H, W), x.device)
        
        # 确保coord和cell的批次维度正确
        coord = coord.expand(B, -1, -1)
        cell = cell.expand(B, -1, -1)
        
        # LIIF查询 - 修复方法调用
        pred = self.liif.query_rgb(x, coord, cell)  # [B, H*W, C_out]
        
        # 重塑为图像格式
        pred = pred.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
        
        return pred