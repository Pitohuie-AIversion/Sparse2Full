"""
经典Transformer模型实现（batch-first，适配2D图像输入输出）

基于"Attention is All You Need"的标准Encoder-Decoder Transformer，
用于PDEBench稀疏观测重建：输入/输出均为2D场（B,C,H,W）。

严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# =========================================================
# Positional Encoding (batch-first)
# =========================================================
class PositionalEncoding(nn.Module):
    """标准正弦位置编码（batch-first版）
    输入:  x [B, N, D]
    输出:  x + pe[:, :N, :]
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)  # [max_len, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(
            1
        )  # [max_len,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"PositionalEncoding expects [B,N,D], got shape={tuple(x.shape)}"
            )
        n = x.size(1)
        if n > self.pe.size(1):
            raise ValueError(f"Sequence length N={n} exceeds max_len={self.pe.size(1)}")
        return x + self.pe[:, :n, :]


# =========================================================
# Multi-Head Attention (batch-first)
# =========================================================
class MultiHeadAttention(nn.Module):
    """多头注意力（batch-first）"""

    def __init__(
        self, d_model: int, num_heads, dropout: float = 0.1, use_sdpa: bool = True
    ):
        super().__init__()
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]
        num_heads = int(num_heads)

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = int(d_model)
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.use_sdpa = bool(use_sdpa) and hasattr(F, "scaled_dot_product_attention")

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, B: int, N: int) -> torch.Tensor:
        # [B, N, D] -> [B, heads, N, d_k]
        return x.view(B, N, self.num_heads, self.d_k).transpose(1, 2).contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query/key/value: [B, N, D]
            attn_mask: 可选
                - bool mask:  True表示允许注意力，False表示屏蔽
                - 或 float/additive mask: 形状可广播到 [B, heads, N, N]
        Returns:
            out: [B, N, D]
        """
        if query.ndim != 3:
            raise ValueError("query must be [B,N,D].")
        B, N, D = query.shape
        if D != self.d_model:
            raise ValueError(f"query last dim D={D} != d_model={self.d_model}")

        Q = self._shape(self.w_q(query), B, N)
        K = self._shape(self.w_k(key), B, key.shape[1])
        V = self._shape(self.w_v(value), B, value.shape[1])

        # SDPA（更快更稳）
        if self.use_sdpa:
            # F.scaled_dot_product_attention expects:
            # q,k,v: [B, heads, N, d_k]
            # attn_mask: bool or float additive, broadcastable to [B, heads, N, S]
            out = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # 手写 attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
                self.d_k
            )  # [B,heads,N,S]
            if attn_mask is not None:
                # bool mask: False -> -inf
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, V)  # [B,heads,N,d_k]

        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)  # [B,N,D]
        out = self.w_o(out)
        return out


# =========================================================
# FFN
# =========================================================
class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            x = F.gelu(self.linear1(x))
        else:
            x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# =========================================================
# Encoder / Decoder layers (post-LN,与你原版一致)
# =========================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads,
        d_ff: int,
        dropout: float = 0.1,
        use_sdpa: bool = True,
    ):
        super().__init__()
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]
        self.self_attn = MultiHeadAttention(
            d_model, int(num_heads), dropout, use_sdpa=use_sdpa
        )
        self.ffn = FeedForward(d_model, d_ff, dropout, activation="relu")
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.drop(attn))
        ffn = self.ffn(x)
        x = self.norm2(x + self.drop(ffn))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads,
        d_ff: int,
        dropout: float = 0.1,
        use_sdpa: bool = True,
    ):
        super().__init__()
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]
        self.self_attn = MultiHeadAttention(
            d_model, int(num_heads), dropout, use_sdpa=use_sdpa
        )
        self.cross_attn = MultiHeadAttention(
            d_model, int(num_heads), dropout, use_sdpa=use_sdpa
        )
        self.ffn = FeedForward(d_model, d_ff, dropout, activation="relu")
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.drop(self_attn))
        cross = self.cross_attn(x, memory, memory, attn_mask=mem_mask)
        x = self.norm2(x + self.drop(cross))
        ffn = self.ffn(x)
        x = self.norm3(x + self.drop(ffn))
        return x


# =========================================================
# Patch embedding / reconstruction (supports any H,W)
# =========================================================
class PatchEmbedding(nn.Module):
    """将[B,C,H,W] -> tokens [B, N, D]，并返回padding信息与patch网格大小"""

    def __init__(self, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int]]:
        """
        Returns:
            tokens: [B, N, D]
            pad_hw: (H0, W0, Hp, Wp) 原始与padding后的尺寸
            grid_hw: (Gh, Gw) patch网格
        """
        B, C, H0, W0 = x.shape
        P = self.patch_size
        pad_h = (P - H0 % P) % P
        pad_w = (P - W0 % P) % P
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H0 + pad_h, W0 + pad_w

        feat = self.proj(x)  # [B, D, Gh, Gw]
        Gh, Gw = feat.shape[-2], feat.shape[-1]
        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]
        return tokens, (H0, W0, Hp, Wp), (Gh, Gw)


class PatchReconstruction(nn.Module):
    """将 tokens [B,N,D] -> [B,C_out,H,W]（先重建到padding后尺寸，再裁剪）"""

    def __init__(self, d_model: int, patch_size: int, out_channels: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.proj = nn.Linear(
            d_model, self.out_channels * self.patch_size * self.patch_size
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: tuple[int, int],
        pad_hw: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            grid_hw: (Gh, Gw)
            pad_hw: (H0, W0, Hp, Wp)
        """
        B, N, D = x.shape
        Gh, Gw = grid_hw
        H0, W0, Hp, Wp = pad_hw
        P = self.patch_size

        if N != Gh * Gw:
            raise ValueError(f"N={N} mismatch grid_hw={grid_hw} (Gh*Gw={Gh*Gw}).")

        x = self.proj(x)  # [B, N, C_out*P*P]
        x = x.view(B, Gh, Gw, self.out_channels, P, P)  # [B,Gh,Gw,C,P,P]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B,C,Gh,P,Gw,P]
        x = x.view(B, self.out_channels, Gh * P, Gw * P)  # [B,C,Hp,Wp]

        # 裁剪回原始尺寸
        x = x[:, :, :H0, :W0].contiguous()
        return x


# =========================================================
# Transformer Model
# =========================================================
@register_model(name="transformer", aliases=["Transformer", "TransformerModel"])
class Transformer(BaseModel):
    """经典Encoder-Decoder Transformer（用于2D场重建）"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,  # 仅用于BaseModel登记；forward支持任意H,W
        patch_size: int = 16,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads=8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 20000,  # 位置编码最大长度（>= 最大 patch 数）
        use_sdpa: bool = True,
        final_activation: str | None = None,  # None | "tanh" | "sigmoid"
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]
        num_heads = int(num_heads)

        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.num_heads = num_heads

        self.patch_embedding = PatchEmbedding(
            self.patch_size, in_channels, self.d_model
        )
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=max_len)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.d_model, num_heads, d_ff, dropout, use_sdpa=use_sdpa
                )
                for _ in range(int(num_encoder_layers))
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    self.d_model, num_heads, d_ff, dropout, use_sdpa=use_sdpa
                )
                for _ in range(int(num_decoder_layers))
            ]
        )

        # learned query embedding：长度在forward里由N动态决定（避免强绑定img_size）
        self.query_embed = nn.Embedding(max_len, self.d_model)

        self.patch_reconstruction = PatchReconstruction(
            self.d_model, self.patch_size, out_channels
        )
        self.drop = nn.Dropout(dropout)

        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int]]:
        tokens, pad_hw, grid_hw = self.patch_embedding(x)  # [B,N,D]
        tokens = self.pos_encoding(tokens)
        tokens = self.drop(tokens)
        for layer in self.encoder_layers:
            tokens = layer(tokens, src_mask=src_mask)
        return tokens, pad_hw, grid_hw

    def decode(
        self,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = memory.shape
        if N > self.query_embed.num_embeddings:
            raise ValueError(
                f"N={N} exceeds query_embed.max_len={self.query_embed.num_embeddings}"
            )

        # learned queries: [B,N,D]
        idx = torch.arange(N, device=memory.device)
        tgt = self.query_embed(idx).unsqueeze(0).expand(B, -1, -1).contiguous()
        tgt = self.pos_encoding(tgt)
        tgt = self.drop(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return tgt  # [B,N,D]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W]
        """
        # 可选mask从kwargs取（通常你这里不需要mask）
        src_mask = kwargs.get("src_mask", None)
        tgt_mask = kwargs.get("tgt_mask", None)
        mem_mask = kwargs.get("mem_mask", None)

        memory, pad_hw, grid_hw = self.encode(x, src_mask=src_mask)
        dec = self.decode(memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        y = self.patch_reconstruction(dec, grid_hw=grid_hw, pad_hw=pad_hw)
        y = self.final_activation(y)
        return y

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "patch_size": self.patch_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "encoder_layers": len(self.encoder_layers),
                "decoder_layers": len(self.decoder_layers),
                "query_embed_max_len": self.query_embed.num_embeddings,
            }
        )
        return info


def create_transformer(**kwargs) -> Transformer:
    return Transformer(**kwargs)
