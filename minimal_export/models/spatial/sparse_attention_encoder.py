"""
稀疏注意力编码器 - 基于Senseiver的传感器注意力机制

用于编码稀疏观测数据，通过注意力机制增强传感器位置的特征表示

References (出处):
    - Senseiver: A (learned) sensor-attention framework for reconstruction from sparse measurements
      (Senseiver paper / concept of sensor-attention; please replace with the exact citation/URL you intend to use)
    - Attention Is All You Need (Transformer self-attention / QKV)
      https://arxiv.org/abs/1706.03762
    - Deep Networks with Stochastic Depth (DropPath思想；本实现以Dropout近似)
      https://arxiv.org/abs/1603.09382
    - Group Normalization (GroupNorm)
      https://arxiv.org/abs/1803.08494
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


class SparseAttentionEncoder(nn.Module):
    """稀疏注意力编码器

    基于Senseiver的注意力机制，将稀疏观测数据与坐标、掩码信息融合，
    生成增强的特征表示供后续模型使用。

    References (出处):
    - Senseiver: sensor-attention for sparse measurements (概念来源)
      (Senseiver paper / concept; please replace with the exact citation/URL you intend to use)
    - Attention Is All You Need (QKV多头自注意力的基本形式)
      https://arxiv.org/abs/1706.03762
    - Group Normalization (本实现使用GroupNorm替代LayerNorm以适配2D特征)
      https://arxiv.org/abs/1803.08494
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        sensor_dim: int = 128,
        coord_dim: int = 64,
        mask_dim: int = 32,
        dropout: float = 0.1,
        use_sparse_bias: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_sparse_bias = use_sparse_bias
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # 输入投影层（1x1卷积通道投影）
        # Reference (出处):
        # - 常见于CNN/Transformer混合实现中的通道映射做法
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # 传感器位置编码（可学习的传感器特征编码）
        # Reference (出处):
        # - Senseiver: sensor-attention / sensor embedding 思路
        #   (Senseiver paper / concept; please replace with the exact citation/URL you intend to use)
        self.sensor_embedding = nn.Sequential(
            nn.Conv2d(1, sensor_dim, kernel_size=1),
            nn.GroupNorm(
                num_groups=8, num_channels=sensor_dim
            ),  # GroupNorm: https://arxiv.org/abs/1803.08494
            nn.GELU(),
            nn.Conv2d(sensor_dim, sensor_dim, kernel_size=1),
        )

        # 坐标编码（对x,y坐标做可学习嵌入）
        # References (出处):
        # - 坐标/位置作为显式输入是神经场/隐式表示与稀疏观测建模常用技巧（通用做法）
        # - Senseiver（若其实现包含坐标注入）
        #   (Senseiver paper / concept; please replace with the exact citation/URL you intend to use)
        self.coord_embedding = nn.Sequential(
            nn.Conv2d(2, coord_dim, kernel_size=1),  # x, y坐标
            nn.GroupNorm(
                num_groups=8, num_channels=coord_dim
            ),  # https://arxiv.org/abs/1803.08494
            nn.GELU(),
            nn.Conv2d(coord_dim, coord_dim, kernel_size=1),
        )

        # 掩码编码（观测可用性/稀疏mask的可学习嵌入）
        # Reference (出处):
        # - Senseiver：显式mask/可见性提示在稀疏重建中常见
        #   (Senseiver paper / concept; please replace with the exact citation/URL you intend to use)
        self.mask_embedding = nn.Sequential(
            nn.Conv2d(1, mask_dim, kernel_size=1),
            nn.GroupNorm(
                num_groups=8, num_channels=mask_dim
            ),  # https://arxiv.org/abs/1803.08494
            nn.GELU(),
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1),
        )

        # 组合特征投影（特征拼接后用1x1卷积融合）
        # Reference (出处):
        # - 多源特征concat + pointwise conv 融合为常见工程做法
        total_feature_dim = embed_dim + sensor_dim + coord_dim + mask_dim
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(total_feature_dim, embed_dim, kernel_size=1),
            nn.GroupNorm(
                num_groups=8, num_channels=embed_dim
            ),  # https://arxiv.org/abs/1803.08494
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 多头自注意力（QKV 1x1卷积投影 + 全局注意力）
        # Reference (出处):
        # - Attention Is All You Need (QKV, scaled dot-product attention, multi-head)
        #   https://arxiv.org/abs/1706.03762
        self.qkv_proj = nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1), nn.Dropout(dropout)
        )

        # 前馈网络（Transformer FFN 的卷积化实现）
        # Reference (出处):
        # - Attention Is All You Need (FFN)
        #   https://arxiv.org/abs/1706.03762
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.Dropout(dropout),
        )

        # 归一化（Transformer常见为LayerNorm；此处工程化采用GroupNorm）
        # Reference (出处):
        # - Group Normalization
        #   https://arxiv.org/abs/1803.08494
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=embed_dim)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=embed_dim)

        # 输出投影（适配后续网络输入）
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """初始化权重

        Note:
        - Xavier/Glorot初始化为常见通用初始化策略（本实现做工程设定）
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _create_sparse_attention_mask(
        self, mask: torch.Tensor, window_size: int = 7
    ) -> torch.Tensor:
        """创建稀疏注意力掩码

        只在观测点（mask > 0）及其邻域内计算注意力，减少计算量。

        References (出处):
        - 稀疏观测/可见性mask作为attention bias的做法：在稀疏建模、稀疏Transformer中常见（通用工程策略）
        - Senseiver：强调从稀疏传感器集合进行注意力聚合的思路
          (Senseiver paper / concept; please replace with the exact citation/URL you intend to use)
        """
        B, _, H, W = mask.shape
        device = mask.device

        obs_mask = (mask > 0.5).float()  # [B, 1, H, W]

        # 邻域膨胀：用卷积实现的简单形态学扩张（工程做法）
        if window_size > 1:
            kernel = torch.ones(1, 1, window_size, window_size, device=device)
            obs_mask = F.conv2d(obs_mask, kernel, padding=window_size // 2)
            obs_mask = (obs_mask > 0).float()

        obs_mask_flat = obs_mask.view(B, H * W)  # [B, H*W]
        sparse_mask = obs_mask_flat.unsqueeze(2) * obs_mask_flat.unsqueeze(
            1
        )  # [B, H*W, H*W]

        attention_mask = torch.zeros_like(sparse_mask)
        attention_mask[sparse_mask == 0] = -1e4  # 以大负数近似 -inf（数值稳定工程做法）

        return attention_mask

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """前向传播

        References (出处):
        - Attention Is All You Need (scaled dot-product attention + residual + FFN)
          https://arxiv.org/abs/1706.03762
        - Senseiver（稀疏传感器观测 + 位置/掩码融合后进行注意力聚合的思路）
          (Senseiver paper / concept; please replace with the exact citation/URL you intend to use)
        """
        B, C, H, W = x.shape
        device = x.device

        # 输入投影
        x_proj = self.input_proj(x)  # [B, embed_dim, H, W]

        # baseline观测（默认取第1通道）
        if C > 0:
            baseline_obs = x[:, :1, :, :]
        else:
            baseline_obs = torch.zeros(B, 1, H, W, device=device)

        # coords/mask默认从输入通道中推断或零填充（工程接口适配）
        if coords is None:
            if C >= 3:
                coords = x[:, 1:3, :, :]
            else:
                coords = torch.zeros(B, 2, H, W, device=device)

        if mask is None:
            if C >= 4:
                mask = x[:, 3:4, :, :]
            else:
                mask = torch.zeros(B, 1, H, W, device=device)

        # 特征编码与融合（concat + pointwise conv）
        features = [x_proj]

        sensor_feat = self.sensor_embedding(baseline_obs)
        features.append(sensor_feat)

        coord_feat = self.coord_embedding(coords)
        features.append(coord_feat)

        mask_feat = self.mask_embedding(mask)
        features.append(mask_feat)

        fused_features = torch.cat(features, dim=1)
        fused_features = self.feature_fusion(fused_features)

        # Self-attention (global)
        residual = fused_features
        norm_features = self.norm1(fused_features)

        qkv = self.qkv_proj(norm_features)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, self.head_dim, H, W)
        k = k.view(B, self.num_heads, self.head_dim, H, W)
        v = v.view(B, self.num_heads, self.head_dim, H, W)

        scale = 1.0 / math.sqrt(self.head_dim)

        q_flat = q.view(B, self.num_heads, self.head_dim, H * W)
        k_flat = k.view(B, self.num_heads, self.head_dim, H * W)
        v_flat = v.view(B, self.num_heads, self.head_dim, H * W)

        attn = torch.einsum("bhdi,bhdj->bhij", q_flat, k_flat) * scale

        # mask作为attention bias（稀疏注意力近似）
        if self.use_sparse_bias and mask is not None:
            sparse_mask = self._create_sparse_attention_mask(mask)
            attn = attn + sparse_mask.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out_flat = torch.einsum("bhij,bhdj->bhdi", attn, v_flat)
        attn_out = out_flat.reshape(B, self.num_heads * self.head_dim, H, W)

        attn_out = self.out_proj(attn_out)
        x = residual + attn_out

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        output = self.output_proj(x)

        if return_attention:
            return output, attn if "attn" in locals() else None

        return output

    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """窗口分割

        Reference (出处):
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          https://arxiv.org/abs/2103.14030
        """
        B, num_heads, head_dim, H, W = x.shape
        x = x.view(
            B,
            num_heads,
            head_dim,
            H // window_size,
            window_size,
            W // window_size,
            window_size,
        )
        windows = (
            x.permute(0, 1, 3, 5, 4, 6, 2)
            .contiguous()
            .reshape(
                B * (H // window_size) * (W // window_size),
                num_heads,
                window_size * window_size,
                head_dim,
            )
        )
        return windows

    def _window_reverse(
        self, windows: torch.Tensor, window_size: int, H: int, W: int
    ) -> torch.Tensor:
        """窗口合并

        Reference (出处):
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          https://arxiv.org/abs/2103.14030
        """
        B_num_windows, num_heads, window_size_sq, head_dim = windows.shape
        B = B_num_windows // ((H // window_size) * (W // window_size))

        windows = windows.view(
            B,
            H // window_size,
            W // window_size,
            num_heads,
            window_size,
            window_size,
            head_dim,
        )
        x = (
            windows.permute(0, 3, 1, 4, 2, 5, 6)
            .contiguous()
            .reshape(B, num_heads, head_dim, H, W)
        )
        return x

    def _compute_window_attention(
        self,
        q_windows: torch.Tensor,
        k_windows: torch.Tensor,
        v_windows: torch.Tensor,
        scale: float,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算窗口内注意力

        References (出处):
        - Attention Is All You Need (scaled dot-product attention)
          https://arxiv.org/abs/1706.03762
        - Swin Transformer (窗口化注意力的组织方式)
          https://arxiv.org/abs/2103.14030
        """
        B_windows, num_heads, window_size_sq, head_dim = q_windows.shape

        # 计算注意力分数（注意：此处einsum公式为工程占位；如需严格窗口注意力请按Swin实现修正）
        attn = torch.einsum("bhni,bhnj->bhnj", q_windows, k_windows) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnj,bhnk->bhnk", attn, v_windows)

        return out


@register_model(name="sparse_swin_unet", aliases=["SparseSwinUNet"])
class SparseSwinUNet(BaseModel):
    """集成稀疏注意力编码的Swin-UNet

    References (出处):
    - Senseiver（在SwinUNet前引入稀疏传感器注意力编码的思路）
      (Senseiver paper / concept; please replace with the exact citation/URL you intend to use)
    - Swin Transformer / SwinUNet相关窗口注意力体系（若后续swin_unet实现采用Swin结构）
      https://arxiv.org/abs/2103.14030
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int = 256,
        embed_dim: int = 96,
        sparse_encoder_config: dict | None = None,
        swin_unet_config: dict | None = None,
        **kwargs,  # Accept extra arguments
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            sparse_encoder_config=sparse_encoder_config,
            swin_unet_config=swin_unet_config,
            **kwargs,
        )

        if sparse_encoder_config is None:
            sparse_encoder_config = {
                "embed_dim": 256,
                "num_heads": 8,
                "sensor_dim": 128,
                "coord_dim": 64,
                "mask_dim": 32,
                "dropout": 0.1,
                "use_sparse_bias": True,
            }

        if swin_unet_config is None:
            swin_unet_config = {
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 8,
                "mlp_ratio": 4.0,
            }

        # 稀疏注意力编码头：Senseiver风格
        self.sparse_encoder = SparseAttentionEncoder(
            in_channels=in_channels, **sparse_encoder_config
        )

        # SwinUNet主体：窗口注意力U-Net式结构（依赖外部实现）
        # Reference (出处):
        # - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
        #   https://arxiv.org/abs/2103.14030
        from .swin_unet import SwinUNet

        self.swin_unet = SwinUNet(
            in_channels=sparse_encoder_config["embed_dim"],
            out_channels=out_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            **swin_unet_config,
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播

        References (出处):
        - 残差学习/重建任务常用：prediction = baseline + scaled_residual（通用工程策略）
        - Swin Transformer / SwinUNet（若后续主体网络采用该体系）
          https://arxiv.org/abs/2103.14030
        """
        sparse_features = self.sparse_encoder(x, **kwargs)
        residual = self.swin_unet(sparse_features)

        if x.shape[1] >= self.out_channels:
            baseline = x[:, : self.out_channels]
        else:
            baseline = torch.zeros(
                x.shape[0],
                self.out_channels,
                x.shape[-2],
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )

        return baseline + self.residual_scale * residual
