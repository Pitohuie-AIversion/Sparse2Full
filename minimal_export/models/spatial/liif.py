"""
LIIF (Learning Implicit Image Function) 模型（工程化版本）

用途：
- 基于隐式连续表示的任意分辨率重建/超分辨
- 保持统一接口：forward(x[B,C_in,H,W], **kwargs)

Reference (Method):
- Chen et al., "Learning Continuous Image Representation with Local Implicit Image Function"
  arXiv:2012.09161  https://arxiv.org/abs/2012.09161

Reference (Official implementation by authors):
- https://github.com/yinboc/liif

实现说明（与论文/官方实现对齐的核心点）：
- local_ensemble：四邻域局部集成（论文/官方实现的 query_rgb 关键策略）
- feat_unfold：对特征做 3x3 unfold 提升局部表达
- cell_decode：把 cell（像素覆盖范围）编码进隐式网络
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


def make_coord(
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    ranges: list[list[float]] | None = None,
    flatten: bool = True,
) -> torch.Tensor:
    """
    生成坐标网格，默认范围 [-1, 1]×[-1, 1]。

    Args:
        shape: (H, W)
        device/dtype: 坐标张量设备与类型
        ranges: [[x0, x1], [y0, y1]]
        flatten: True -> [H*W, 2]；False -> [H, W, 2]
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1.0, 1.0
        else:
            v0, v1 = float(ranges[i][0]), float(ranges[i][1])
        r = (v1 - v0) / (2.0 * n)
        seq = v0 + r + (2.0 * r) * torch.arange(n, device=device, dtype=dtype)
        coord_seqs.append(seq)

    # meshgrid: (H,W,2)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    """
    简单 MLP，避免默认可变 list 参数。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_list: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()
        if hidden_list is None:
            hidden_list = [256, 256, 256, 256]

        act: nn.Module
        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "leaky_relu":
            act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        lastv = in_dim
        for h in hidden_list:
            layers.append(nn.Linear(lastv, h))
            layers.append(act)
            lastv = h
        layers.append(nn.Linear(lastv, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNNEncoder(nn.Module):
    """
    轻量编码器（工程简化版）。
    论文与官方实现允许替换 encoder（如 EDSR），这里保留可用的 CNN baseline。
    """

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, out_dim, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        return x


class LIIFCore(nn.Module):
    """
    LIIF 核心：query_rgb(feat, coord, cell) -> [B, N, C_out]

    关键逻辑参考：
    - arXiv:2012.09161
    - github.com/yinboc/liif 的 query_rgb 实现范式（局部集成/相对坐标/cell 编码）
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        out_dim: int,
        imnet_hidden: list[int] | None = None,
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        mlp_activation: str = "relu",
    ):
        super().__init__()
        self.encoder = encoder

        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        enc_out_dim = getattr(encoder, "out_dim", None)
        if enc_out_dim is None:
            raise ValueError("encoder 必须提供 out_dim 属性。")

        imnet_in_dim = enc_out_dim * (9 if feat_unfold else 1)  # unfold: 3x3
        imnet_in_dim += 2  # relative coord
        if cell_decode:
            imnet_in_dim += 2  # cell size

        self.imnet = MLP(
            in_dim=imnet_in_dim,
            out_dim=out_dim,
            hidden_list=imnet_hidden,
            activation=mlp_activation,
        )

    @torch.no_grad()
    def _feat_coord(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # [H,W,2] -> [1,2,H,W]
        coord = make_coord((h, w), device=device, dtype=dtype, flatten=False)  # [H,W,2]
        coord = coord.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,2,H,W]
        return coord

    def gen_feat(self, inp: torch.Tensor) -> torch.Tensor:
        return self.encoder(inp)

    def query_rgb(
        self,
        feat_or_inp: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            feat_or_inp:
                - 特征图 [B, C, H, W]，或
                - 原始输入 [B, C_in, H_in, W_in]（自动编码）
            coord: [B, N, 2]，范围 [-1, 1]
            cell: [B, N, 2]，范围以 [-1,1] 坐标系定义的像素覆盖大小

        Returns:
            [B, N, C_out]
        """
        if feat_or_inp.dim() != 4:
            raise ValueError("feat_or_inp 需要是 4D 张量。")

        # 若传入原图，自动编码
        if feat_or_inp.shape[1] == getattr(self.encoder, "in_channels", -999):
            feat = self.gen_feat(feat_or_inp)
        else:
            feat = feat_or_inp

        b, c, h, w = feat.shape
        if coord.dim() != 3 or coord.shape[0] != b or coord.shape[-1] != 2:
            raise ValueError("coord 需要是 [B, N, 2]，且 B 与 feat 批次一致。")

        if self.cell_decode:
            if cell is None:
                raise ValueError("cell_decode=True 时，cell 不能为空。")
            if (
                cell.dim() != 3
                or cell.shape[:2] != coord.shape[:2]
                or cell.shape[-1] != 2
            ):
                raise ValueError("cell 需要是 [B, N, 2]，并与 coord 的 [B, N] 对齐。")

        feat = feat.contiguous()
        coord = coord.contiguous()
        if cell is not None:
            cell = cell.contiguous()

        # unfold: [B, C*9, H, W]
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(b, c * 9, h, w)

        if self.local_ensemble:
            vx_lst, vy_lst = [-1, 1], [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst = [0], [0]
            eps_shift = 0.0

        # field radius (global: [-1, 1])
        rx = 1.0 / h
        ry = 1.0 / w

        # feat_coord: [B,2,H,W]
        feat_coord = self._feat_coord(h, w, feat.device, feat.dtype).expand(
            b, -1, -1, -1
        )

        preds: list[torch.Tensor] = []
        areas: list[torch.Tensor] = []

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1.0 + 1e-6, 1.0 - 1e-6)

                grid = coord_.flip(-1).unsqueeze(1)  # [B,1,N,2]

                # q_feat: [B,N,C]
                q_feat = (
                    F.grid_sample(feat, grid, mode="nearest", align_corners=False)[
                        :, :, 0, :
                    ]
                    .permute(0, 2, 1)
                    .contiguous()
                )
                # q_coord: [B,N,2]
                q_coord = (
                    F.grid_sample(
                        feat_coord, grid, mode="nearest", align_corners=False
                    )[:, :, 0, :]
                    .permute(0, 2, 1)
                    .contiguous()
                )

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode and cell is not None:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= h
                    rel_cell[:, :, 1] *= w
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, qn = coord.shape[:2]
                pred = self.imnet(inp.view(bs * qn, -1)).view(bs, qn, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas, dim=0).sum(dim=0)

        # 官方实现中的面积重排（保持加权一致性）
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]

        ret = 0.0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret


@register_model(name="liif", aliases=["LIIF", "LIIFModel"])
class LIIFModel(BaseModel):
    """
    LIIF 模型（统一接口版）

    forward 逻辑：
    - 默认：输出分辨率 = (img_size, img_size)，返回 [B, C_out, img_size, img_size]
    - 若传入 coord/cell：返回 [B, N, C_out]（与论文定义一致，便于任意采样）
      - 可选传入 target_size=(H,W) 以便外部自行 reshape
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        encoder_dim: int = 256,
        imnet_hidden: list[int] | None = None,
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        mlp_activation: str = "relu",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        encoder = SimpleCNNEncoder(in_channels=in_channels, out_dim=encoder_dim)

        self.core = LIIFCore(
            encoder=encoder,
            out_dim=out_channels,
            imnet_hidden=imnet_hidden,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
            mlp_activation=mlp_activation,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_cell(
        self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # cell 大小：在 [-1,1] 坐标系中每个像素覆盖范围约为 (2/H, 2/W)
        n = h * w
        cell = torch.empty((b, n, 2), device=device, dtype=dtype)
        cell[:, :, 0] = 2.0 / float(h)
        cell[:, :, 1] = 2.0 / float(w)
        return cell

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
            kwargs:
                coord: Optional[Tensor[B, N, 2]]
                cell:  Optional[Tensor[B, N, 2]]
                target_size: Optional[Tuple[int,int]] 仅用于用户侧 reshape（本函数不强制使用）

        Returns:
            - 默认：[B, C_out, img_size, img_size]
            - 提供 coord：返回 [B, N, C_out]
        """
        b = x.shape[0]
        device = x.device
        dtype = (
            x.dtype
            if x.dtype in (torch.float16, torch.float32, torch.float64)
            else torch.float32
        )

        coord: torch.Tensor | None = kwargs.get("coord", None)
        cell: torch.Tensor | None = kwargs.get("cell", None)

        feat = self.core.gen_feat(x)

        # 用户提供 coord：返回序列输出，符合 LIIF 定义
        if coord is not None:
            if cell is None and self.core.cell_decode:
                # 默认 cell：按 feat 空间大小定义（若用户希望按目标采样密度定义，建议显式传入 cell）
                # 工程上更常用：按目标输出分辨率定义 cell；建议在调用侧提供 cell
                # 这里保守采用 coord 采样对应的输出密度推断较困难，因此采用 feat 尺寸作为默认
                cell = self._make_cell(b, feat.shape[-2], feat.shape[-1], device, dtype)
            return self.core.query_rgb(feat, coord, cell)

        # 默认输出：按 img_size 生成整网格坐标
        out_h = int(self.img_size)
        out_w = int(self.img_size)

        coord_grid = make_coord(
            (out_h, out_w), device=device, dtype=dtype, flatten=True
        )  # [N,2]
        coord_grid = coord_grid.unsqueeze(0).expand(b, -1, -1).contiguous()  # [B,N,2]

        if self.core.cell_decode:
            cell = self._make_cell(b, out_h, out_w, device, dtype)
        else:
            cell = None

        pred = self.core.query_rgb(feat, coord_grid, cell)  # [B,N,C_out]
        pred = (
            pred.view(b, out_h, out_w, self.out_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return pred
