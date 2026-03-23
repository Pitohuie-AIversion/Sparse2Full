"""
LIIF-Head 模型（带出处注释 + 工程化修正）

本实现的核心思想（implicit function head + local ensemble + feat unfold + cell decoding）
来自 LIIF 论文与作者官方实现：

- Paper (CVPR 2021 Oral): "Learning Continuous Image Representation with Local Implicit Image Function"
  arXiv: https://arxiv.org/abs/2012.09161
- Official GitHub (author): https://github.com/yinboc/liif

说明（避免“名称欺骗”）：
- 你这里的 SimpleEncoder 是为了自包含而写的轻量 backbone，并非论文中的 EDSR/RDN 等配置；
  因此本文件应被视为“LIIF head + 简化 backbone”的实现，用于工程集成/消融，而非复现论文指标。
"""

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


def make_coord(
    shape: Tuple[int, int],
    ranges: Optional[List[List[float]]] = None,
    flatten: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """生成坐标网格（LIIF 官方实现同类函数的工程化版本）

    Args:
        shape: (H, W)
        ranges: 坐标范围，默认 [-1, 1] × [-1, 1]
        flatten: True -> [H*W, 2]；False -> [H, W, 2]
        device/dtype: 显式指定坐标张量设备与精度（避免 .cuda() 硬编码）

    Returns:
        坐标张量
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1.0, 1.0
        else:
            v0, v1 = float(ranges[i][0]), float(ranges[i][1])
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device, dtype=dtype)
        coord_seqs.append(seq)

    # PyTorch>=1.10: indexing='ij'
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)  # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # [H*W, 2]
    return ret


class MLP(nn.Module):
    """多层感知机（用于隐式函数 imnet）"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_list: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            lastv = hidden

        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LIIFHead(nn.Module):
    """LIIF 头部（Local Implicit Image Function）

    参考：
    - LIIF 论文与官方实现中的隐式函数头（imnet）、local ensemble、feat unfold、cell decode。
      https://arxiv.org/abs/2012.09161
      https://github.com/yinboc/liif
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 3,
        hidden_list: Optional[List[int]] = None,
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_list is None:
            hidden_list = [256, 256, 256, 256]

        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # LIIF 官方实现的常见做法：可选 feat unfold（3x3 邻域）来增强局部表达能力
        mlp_in_dim = in_dim
        if self.feat_unfold:
            mlp_in_dim *= 9  # 3x3 neighborhood

        mlp_in_dim += 2  # rel_coord (dx, dy)
        if self.cell_decode:
            mlp_in_dim += 2  # rel_cell (sx, sy)

        self.imnet = MLP(mlp_in_dim, out_dim, hidden_list, activation, dropout)

    def query_rgb(
        self,
        feat: torch.Tensor,          # [B, C, H, W]
        coord: torch.Tensor,         # [B, N, 2] in [-1, 1]
        cell: Optional[torch.Tensor] = None,  # [B, N, 2]
    ) -> torch.Tensor:
        """查询像素值（核心逻辑来自 LIIF 官方实现的 query 过程）"""

        feat = feat.contiguous()
        coord = coord.contiguous().to(device=feat.device, dtype=feat.dtype)

        if self.cell_decode:
            if cell is None:
                raise ValueError("LIIFHead: cell_decode=True but `cell` is None. Please pass `cell`.")
            cell = cell.contiguous().to(device=feat.device, dtype=feat.dtype)

        # 1) 特征展开（官方实现常见做法）
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
            )

        # 2) local ensemble（官方实现：对四个相邻采样点做集成）
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0.0

        # NOTE：官方实现中 rx/ry 的写法常见为 2/H/2 与 2/W/2（等价于 1/H, 1/W）
        rx = 2.0 / feat.shape[-2] / 2.0
        ry = 2.0 / feat.shape[-1] / 2.0

        # 3) 构建特征网格坐标（与 feat 同 device/dtype，替代 .cuda()）
        feat_coord = make_coord(
            feat.shape[-2:],
            flatten=False,
            device=feat.device,
            dtype=feat.dtype,
        )  # [H, W, 2]
        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, feat.shape[-2], feat.shape[-1]
        )  # [B, 2, H, W]

        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # 4) grid_sample 采样特征（官方实现常用 nearest）
                q_feat = F.grid_sample(
                    feat,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)  # [B, N, C]

                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

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

                # 5) 面积权重（官方实现中用于 local ensemble 加权）
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            # 官方实现中的交换顺序（保证四象限权重对应关系）
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0.0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

    def forward(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.query_rgb(feat, coord, cell)


class SimpleEncoder(nn.Module):
    """简化 backbone（工程自包含版本）

    注意：论文/官方实现常用 EDSR/RDN 等作为特征提取网络；
    这里提供一个轻量 encoder，便于你在统一框架里先跑通实验与消融。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Conv2d(current_channels, out_channels, 3, 1, 1),
                    nn.ReLU(inplace=True),
                ])
                current_channels = out_channels
            else:
                next_channels = min(out_channels * (2 ** i), 512)
                layers.extend([
                    nn.Conv2d(current_channels, next_channels, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(next_channels, next_channels, 3, 1, 1),
                    nn.ReLU(inplace=True),
                ])
                current_channels = next_channels

        self.encoder = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


@register_model(name="liif", aliases=["LIIF", "LIIFModel", "liif_model"])
class LIIFModel(BaseModel):
    """LIIF 模型（简化 backbone + LIIF head）

    Reference:
      - https://arxiv.org/abs/2012.09161
      - https://github.com/yinboc/liif
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        encoder_channels: int = 64,
        encoder_layers: int = 4,
        hidden_list: Optional[List[int]] = None,
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
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

        self.encoder = SimpleEncoder(
            in_channels=in_channels,
            out_channels=encoder_channels,
            num_layers=encoder_layers,
        )

        self.liif_head = LIIFHead(
            in_dim=self.encoder.out_channels,
            out_dim=out_channels,
            hidden_list=hidden_list,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
            activation=activation,
            dropout=dropout,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        **kwargs,
    ) -> torch.Tensor:
        """统一接口：
        - 若 coord is None：默认输出 [B, C_out, img_size, img_size]
        - 若 coord 给定：输出 [B, N, C_out]
        """
        feat = self.encoder(x)

        # 默认生成整幅图坐标（标准重建）
        if coord is None:
            coord = make_coord(
                (self.img_size, self.img_size),
                flatten=True,
                device=feat.device,
                dtype=feat.dtype,
            ).unsqueeze(0).expand(x.shape[0], -1, -1)  # [B, H*W, 2]

            if cell is None and self.cell_decode:
                cell = torch.ones_like(coord)
                cell[:, :, 0] *= 2.0 / self.img_size
                cell[:, :, 1] *= 2.0 / self.img_size

            pred = self.liif_head(feat, coord, cell)  # [B, H*W, C_out]
            pred = pred.view(x.shape[0], self.img_size, self.img_size, self.out_channels)
            pred = pred.permute(0, 3, 1, 2).contiguous()
            return pred

        # 使用外部提供坐标（任意分辨率查询）
        coord = coord.to(device=feat.device, dtype=feat.dtype)
        if cell is not None:
            cell = cell.to(device=feat.device, dtype=feat.dtype)
        else:
            # 这里无法从 coord 反推目标分辨率；如果你需要 cell_decode，请在调用方显式传入 cell
            if self.cell_decode:
                raise ValueError("LIIFModel: cell_decode=True but `cell` is None. Please pass `cell` for arbitrary queries.")

        return self.liif_head(feat, coord, cell)

    def super_resolve(self, x: torch.Tensor, scale_factor: float = 2.0) -> torch.Tensor:
        """便捷超分推理：根据 scale_factor 生成目标坐标与 cell"""
        _, _, h, w = x.shape
        target_h, target_w = int(h * scale_factor), int(w * scale_factor)

        coord = make_coord(
            (target_h, target_w),
            flatten=True,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0).expand(x.shape[0], -1, -1)

        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2.0 / target_h
        cell[:, :, 1] *= 2.0 / target_w

        feat = self.encoder(x)
        pred = self.liif_head(feat, coord, cell)
        pred = pred.view(x.shape[0], target_h, target_w, self.out_channels)
        pred = pred.permute(0, 3, 1, 2).contiguous()
        return pred


# 向后兼容别名（如你的工程里有直接 from xxx import LIIF）
LIIF = LIIFModel
