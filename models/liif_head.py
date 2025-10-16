"""LIIF-Head模型

Local Implicit Image Function (LIIF) 头部模型，用于任意分辨率的图像重建。
LIIF通过学习连续的隐式表示来实现超分辨率和图像重建。

Reference:
    Learning Continuous Image Representation with Local Implicit Image Function
    https://arxiv.org/abs/2012.09161
"""

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


def make_coord(shape: Tuple[int, int], ranges: Optional[List[List[float]]] = None, flatten: bool = True) -> torch.Tensor:
    """生成坐标网格
    
    Args:
        shape: 网格形状 (H, W)
        ranges: 坐标范围，默认为[[-1, 1], [-1, 1]]
        flatten: 是否展平为序列
        
    Returns:
        坐标张量 [H*W, 2] 或 [H, W, 2]
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_list: List[int],
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = []
        lastv = in_dim
        
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            lastv = hidden
        
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LIIFHead(nn.Module):
    """LIIF头部
    
    将特征图和查询坐标映射到像素值的隐式函数
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 3,
        hidden_list: List[int] = None,
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        if hidden_list is None:
            hidden_list = [256, 256, 256, 256]
        
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        # 计算输入维度
        mlp_in_dim = in_dim
        
        if self.feat_unfold:
            mlp_in_dim *= 9  # 3x3邻域
        
        mlp_in_dim += 2  # 坐标
        
        if self.cell_decode:
            mlp_in_dim += 2  # 单元格大小
        
        # MLP网络
        self.imnet = MLP(mlp_in_dim, out_dim, hidden_list, activation, dropout)
    
    def query_rgb(
        self, 
        feat: torch.Tensor, 
        coord: torch.Tensor, 
        cell: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """查询RGB值
        
        Args:
            feat: 特征图 [B, C, H, W]
            coord: 查询坐标 [B, N, 2]，范围[-1, 1]
            cell: 单元格大小 [B, N, 2]，可选
            
        Returns:
            RGB值 [B, N, out_dim]
        """
        feat = feat.contiguous()
        coord = coord.contiguous()
        
        # 特征采样
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0
        
        # 局部集成
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            if feat.is_cuda else make_coord(feat.shape[-2:], flatten=False)
        feat_coord = feat_coord.permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        
        preds = []
        areas = []
        
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                
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
    
    def forward(
        self, 
        feat: torch.Tensor, 
        coord: torch.Tensor, 
        cell: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            feat: 特征图 [B, C, H, W]
            coord: 查询坐标 [B, N, 2]
            cell: 单元格大小 [B, N, 2]，可选
            
        Returns:
            输出 [B, N, out_dim]
        """
        return self.query_rgb(feat, coord, cell)


class SimpleEncoder(nn.Module):
    """简单的编码器
    
    用于将输入图像编码为特征图
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            if i == 0:
                # 第一层不下采样
                layers.extend([
                    nn.Conv2d(current_channels, out_channels, 3, 1, 1),
                    nn.ReLU(inplace=True)
                ])
                current_channels = out_channels
            else:
                # 后续层逐步增加通道数
                next_channels = min(out_channels * (2 ** i), 512)
                layers.extend([
                    nn.Conv2d(current_channels, next_channels, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(next_channels, next_channels, 3, 1, 1),
                    nn.ReLU(inplace=True)
                ])
                current_channels = next_channels
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = current_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LIIFModel(BaseModel):
    """LIIF模型
    
    基于Local Implicit Image Function的图像重建模型。
    能够处理任意分辨率的输出，特别适用于超分辨率任务。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 图像尺寸（正方形）
        encoder_channels: 编码器输出通道数，默认64
        encoder_layers: 编码器层数，默认4
        hidden_list: LIIF头部隐藏层列表，默认[256, 256, 256, 256]
        local_ensemble: 是否使用局部集成，默认True
        feat_unfold: 是否使用特征展开，默认True
        cell_decode: 是否使用单元格解码，默认True
        activation: 激活函数，默认'relu'
        dropout: Dropout概率，默认0.0
        **kwargs: 其他参数
    
    Examples:
        >>> model = LIIFModel(in_channels=3, out_channels=3, img_size=256)
        >>> x = torch.randn(1, 3, 64, 64)  # 低分辨率输入
        >>> coord = make_coord((256, 256)).unsqueeze(0)  # 目标坐标
        >>> cell = torch.ones(1, 256*256, 2) * (2/256)  # 单元格大小
        >>> y = model(x, coord=coord, cell=cell)
        >>> print(y.shape)  # torch.Size([1, 65536, 3])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        encoder_channels: int = 64,
        encoder_layers: int = 4,
        hidden_list: List[int] = None,
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        activation: str = 'relu',
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        if hidden_list is None:
            hidden_list = [256, 256, 256, 256]
        
        self.encoder_channels = encoder_channels
        self.encoder_layers = encoder_layers
        self.hidden_list = hidden_list
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.activation = activation
        self.dropout = dropout
        
        # 编码器
        self.encoder = SimpleEncoder(
            in_channels=in_channels,
            out_channels=encoder_channels,
            num_layers=encoder_layers
        )
        
        # LIIF头部
        self.liif_head = LIIFHead(
            in_dim=self.encoder.out_channels,
            out_dim=out_channels,
            hidden_list=hidden_list,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
            activation=activation,
            dropout=dropout
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        coord: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H_in, W_in]
            coord: 查询坐标 [B, N, 2]，如果为None则生成默认坐标
            cell: 单元格大小 [B, N, 2]，如果为None则自动计算
            **kwargs: 其他参数
            
        Returns:
            如果提供coord：输出张量 [B, N, C_out]
            如果未提供coord：输出张量 [B, C_out, img_size, img_size]
        """
        # 编码
        feat = self.encoder(x)
        
        # 如果没有提供坐标，生成默认坐标（用于标准重建）
        if coord is None:
            coord = make_coord((self.img_size, self.img_size), flatten=True)
            coord = coord.unsqueeze(0).expand(x.shape[0], -1, -1)
            if x.is_cuda:
                coord = coord.cuda()
            
            # 自动计算单元格大小
            if cell is None:
                cell = torch.ones_like(coord)
                cell[:, :, 0] *= 2 / self.img_size
                cell[:, :, 1] *= 2 / self.img_size
            
            # LIIF解码
            pred = self.liif_head(feat, coord, cell)
            
            # 重塑为图像格式
            pred = pred.view(x.shape[0], self.img_size, self.img_size, self.out_channels)
            pred = pred.permute(0, 3, 1, 2).contiguous()
            
            return pred
        else:
            # 使用提供的坐标
            if cell is None and self.cell_decode:
                # 如果需要单元格解码但没有提供cell，使用默认值
                cell = torch.ones_like(coord)
                cell[:, :, 0] *= 2 / feat.shape[-2]
                cell[:, :, 1] *= 2 / feat.shape[-1]
            
            return self.liif_head(feat, coord, cell)
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> int:
        """计算FLOPs（简化估算）
        
        Args:
            input_shape: 输入形状，默认为(1, in_channels, img_size//4, img_size//4)
            
        Returns:
            FLOPs数量
        """
        if input_shape is None:
            # 假设输入是目标分辨率的1/4
            input_shape = (1, self.in_channels, self.img_size // 4, self.img_size // 4)
        
        batch_size, _, height, width = input_shape
        
        # 编码器FLOPs
        encoder_flops = 0
        current_channels = self.in_channels
        h, w = height, width
        
        for i in range(self.encoder_layers):
            if i == 0:
                next_channels = self.encoder_channels
                encoder_flops += current_channels * next_channels * 3 * 3 * h * w
            else:
                next_channels = min(self.encoder_channels * (2 ** i), 512)
                # 两个3x3卷积
                encoder_flops += current_channels * next_channels * 3 * 3 * h * w
                encoder_flops += next_channels * next_channels * 3 * 3 * h * w
            current_channels = next_channels
        
        # LIIF头部FLOPs
        num_queries = self.img_size * self.img_size
        liif_in_dim = current_channels
        if self.feat_unfold:
            liif_in_dim *= 9
        liif_in_dim += 2  # 坐标
        if self.cell_decode:
            liif_in_dim += 2  # 单元格
        
        liif_flops = 0
        prev_dim = liif_in_dim
        for hidden_dim in self.hidden_list:
            liif_flops += prev_dim * hidden_dim * num_queries
            prev_dim = hidden_dim
        liif_flops += prev_dim * self.out_channels * num_queries
        
        # 局部集成的额外计算（如果启用）
        if self.local_ensemble:
            liif_flops *= 4  # 4个邻域点
        
        total_flops = encoder_flops + liif_flops
        self._flops = total_flops * batch_size
        return self._flops
    
    def freeze_encoder(self) -> None:
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_liif_head(self) -> None:
        """冻结LIIF头部参数"""
        for param in self.liif_head.parameters():
            param.requires_grad = False
    
    def super_resolve(
        self, 
        x: torch.Tensor, 
        scale_factor: float = 2.0
    ) -> torch.Tensor:
        """超分辨率推理
        
        Args:
            x: 低分辨率输入 [B, C_in, H, W]
            scale_factor: 放大倍数
            
        Returns:
            高分辨率输出 [B, C_out, H*scale, W*scale]
        """
        _, _, h, w = x.shape
        target_h, target_w = int(h * scale_factor), int(w * scale_factor)
        
        # 生成目标坐标
        coord = make_coord((target_h, target_w), flatten=True)
        coord = coord.unsqueeze(0).expand(x.shape[0], -1, -1)
        if x.is_cuda:
            coord = coord.cuda()
        
        # 计算单元格大小
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / target_h
        cell[:, :, 1] *= 2 / target_w
        
        # 编码
        feat = self.encoder(x)
        
        # LIIF解码
        pred = self.liif_head(feat, coord, cell)
        
        # 重塑为图像格式
        pred = pred.view(x.shape[0], target_h, target_w, self.out_channels)
        pred = pred.permute(0, 3, 1, 2).contiguous()
        
        return pred
    
    def arbitrary_scale_forward(
        self,
        x: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """任意尺寸前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            target_size: 目标尺寸 (H, W)
            
        Returns:
            输出张量 [B, C_out, target_H, target_W]
        """
        target_h, target_w = target_size
        
        # 生成目标坐标
        coord = make_coord((target_h, target_w), flatten=True)
        coord = coord.unsqueeze(0).expand(x.shape[0], -1, -1)
        if x.is_cuda:
            coord = coord.cuda()
        
        # 计算单元格大小
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / target_h
        cell[:, :, 1] *= 2 / target_w
        
        # 编码
        feat = self.encoder(x)
        
        # LIIF解码
        pred = self.liif_head(feat, coord, cell)
        
        # 重塑为图像格式
        pred = pred.view(x.shape[0], target_h, target_w, self.out_channels)
        pred = pred.permute(0, 3, 1, 2).contiguous()
        
        return pred


# 别名，保持向后兼容
LIIF = LIIFModel