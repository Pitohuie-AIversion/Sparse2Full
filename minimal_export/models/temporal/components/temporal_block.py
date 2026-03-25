"""时序模块实现

轻量级时序处理模块，仅在时间维度进行操作，保持空间维度不变。
支持因果卷积、非因果卷积和Transformer编码器三种模式。
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalConv1D(nn.Module):
    """轻时序卷积模块

    在时间维度进行1D卷积，聚合历史信息为单帧输出。
    设计原则：轻量、稳定、高效。

    Args:
        c_in: 输入通道数
        c_out: 输出通道数，默认等于c_in
        k: 卷积核大小，默认3
        causal: 是否使用因果卷积，默认True
        activation: 激活函数，默认None
        dropout: dropout概率，默认0.0
    """

    def __init__(
        self,
        c_in: int,
        c_out: int | None = None,
        k: int = 3,
        causal: bool = True,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out or c_in
        self.k = k
        self.causal = causal

        # 计算padding
        if causal:
            self.padding = k - 1  # 因果卷积：只看过去
        else:
            self.padding = (k - 1) // 2  # 非因果卷积：看过去和未来

        # 1D卷积层
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=self.c_out,
            kernel_size=k,
            padding=self.padding,
            bias=True,
        )

        # 可选的激活函数和dropout
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # 初始化权重
        self._init_weights()

        logger.info(f"TemporalConv1D: {c_in}->{self.c_out}, k={k}, causal={causal}")

    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (B, T, C, H, W)

        Returns:
            输出张量 (B, C_out, H, W)
        """
        B, T, C, H, W = x.shape

        # 重排维度：(B, T, C, H, W) -> (B*H*W, C, T)
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, C, T)
        x = x.view(B * H * W, C, T)  # (B*H*W, C, T)

        # 1D卷积
        x = self.conv(x)  # (B*H*W, C_out, T_out)

        # 应用激活函数
        if self.activation is not None:
            x = self.activation(x)

        # 应用dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # 时间聚合
        if self.causal:
            # 因果卷积：取最后一个时间步
            x = x[..., -1]  # (B*H*W, C_out)
        else:
            # 非因果卷积：取平均
            x = x.mean(dim=-1)  # (B*H*W, C_out)

        # 恢复空间维度：(B*H*W, C_out) -> (B, C_out, H, W)
        x = x.view(B, H, W, self.c_out)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C_out, H, W)

        return x

    def get_output_channels(self) -> int:
        """获取输出通道数"""
        return self.c_out

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "module_type": "TemporalConv1D",
            "input_channels": self.c_in,
            "output_channels": self.c_out,
            "kernel_size": self.k,
            "causal": self.causal,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


class TemporalTransformerEncoder(nn.Module):
    """时序Transformer编码器

    使用多头自注意力机制处理时序信息，支持因果掩码。
    适合捕捉长程时序依赖关系。

    Args:
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout概率
        causal: 是否使用因果掩码
        max_seq_len: 最大序列长度
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        causal: bool = True,
        max_seq_len: int = 64,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.causal = causal
        self.max_seq_len = max_seq_len

        # 位置编码
        self.pos_encoding = nn.Parameter(
            self._generate_positional_encoding(max_seq_len, d_model),
            requires_grad=False,
        )

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)

        logger.info(
            f"TemporalTransformerEncoder: d_model={d_model}, nhead={nhead}, "
            f"layers={num_layers}, causal={causal}"
        )

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """生成正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (B, T, C, H, W)

        Returns:
            输出张量 (B, C, H, W) - 聚合后的单帧输出
        """
        B, T, C, H, W = x.shape

        # 重排维度并展平空间维度：(B, T, C, H, W) -> (B*H*W, T, C)
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, C)
        x = x.view(B * H * W, T, C)  # (B*H*W, T, C)

        # 检查维度匹配
        if C != self.d_model:
            raise ValueError(f"Input channel {C} doesn't match d_model {self.d_model}")

        # 添加位置编码
        if T <= self.max_seq_len:
            pos_enc = self.pos_encoding[:, :T, :]  # (1, T, d_model)
            x = x + pos_enc
        else:
            logger.warning(
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )
            # 截断或循环使用位置编码
            pos_enc = self.pos_encoding[:, : self.max_seq_len, :].repeat(
                1, (T // self.max_seq_len) + 1, 1
            )
            x = x + pos_enc[:, :T, :]

        # 生成注意力掩码
        attn_mask = None
        if self.causal:
            attn_mask = self._generate_causal_mask(T).to(x.device)

        # Transformer编码
        x = self.transformer_encoder(x, mask=attn_mask)  # (B*H*W, T, d_model)

        # 输出投影
        x = self.output_proj(x)  # (B*H*W, T, d_model)

        # 时间聚合：取最后一个时间步（因果）或平均（非因果）
        if self.causal:
            x = x[:, -1, :]  # (B*H*W, d_model)
        else:
            x = x.mean(dim=1)  # (B*H*W, d_model)

        # 恢复空间维度：(B*H*W, d_model) -> (B, d_model, H, W)
        x = x.view(B, H, W, self.d_model)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, d_model, H, W)

        return x

    def get_output_channels(self) -> int:
        """获取输出通道数"""
        return self.d_model

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "module_type": "TemporalTransformerEncoder",
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "causal": self.causal,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


class FiLMTemporalBlock(nn.Module):
    """FiLM时序模块

    使用Feature-wise Linear Modulation进行时序信息融合。
    相比TemporalConv1D更轻量，适合资源受限场景。

    Args:
        c_in: 输入通道数
        c_out: 输出通道数，默认等于c_in
        hidden_dim: 隐藏层维度，默认为c_in//4
        activation: 激活函数，默认ReLU
    """

    def __init__(
        self,
        c_in: int,
        c_out: int | None = None,
        hidden_dim: int | None = None,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out or c_in
        self.hidden_dim = hidden_dim or max(c_in // 4, 8)

        # 时间编码器：将时间维度编码为调制参数
        self.time_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局时间池化
            nn.Flatten(),
            nn.Linear(c_in, self.hidden_dim),
            activation,
            nn.Linear(self.hidden_dim, self.c_out * 2),  # 输出gamma和beta
        )

        # 如果输入输出通道数不同，添加投影层
        if self.c_in != self.c_out:
            self.projection = nn.Conv2d(c_in, self.c_out, 1, bias=False)
        else:
            self.projection = None

        logger.info(
            f"FiLMTemporalBlock: {c_in}->{self.c_out}, hidden={self.hidden_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (B, T, C, H, W)

        Returns:
            输出张量 (B, C, H, W)
        """
        B, T, C, H, W = x.shape

        # 使用最后一帧作为基础
        base_frame = x[:, -1]  # (B, C, H, W)

        # 投影到输出通道数
        if self.projection is not None:
            base_frame = self.projection(base_frame)

        if T == 1:
            # 只有一帧，直接返回
            return base_frame

        # 计算时间调制参数
        # 重排维度进行时间编码：(B, T, C, H, W) -> (B, C, T*H*W)
        x_temp = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        x_temp = x_temp.view(B, C, T * H * W)  # (B, C, T*H*W)

        # 时间编码
        modulation = self.time_encoder(x_temp)  # (B, 2*C_out)
        modulation = modulation.view(B, 2, self.c_out)  # (B, 2, C_out)

        gamma = modulation[:, 0, :].unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1)
        beta = modulation[:, 1, :].unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1)

        # FiLM调制
        output = gamma * base_frame + beta

        return output

    def get_output_channels(self) -> int:
        """获取输出通道数"""
        return self.c_out

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "module_type": "FiLMTemporalBlock",
            "input_channels": self.c_in,
            "output_channels": self.c_out,
            "hidden_dim": self.hidden_dim,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def create_temporal_module(temporal_type: str, c_in: int, **kwargs) -> nn.Module:
    """时序模块工厂函数

    Args:
        temporal_type: 时序模块类型 ('conv1d' | 'film' | 'transformer')
        c_in: 输入通道数
        **kwargs: 其他参数

    Returns:
        时序模块实例
    """
    if temporal_type == "conv1d":
        return TemporalConv1D(c_in=c_in, **kwargs)
    elif temporal_type == "film":
        return FiLMTemporalBlock(c_in=c_in, **kwargs)
    elif temporal_type == "transformer":
        # 过滤掉Transformer不需要的参数
        transformer_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "nhead",
                "num_layers",
                "dim_feedforward",
                "dropout",
                "causal",
                "max_seq_len",
            ]
        }

        # 对于Transformer，使用d_model参数而不是c_in
        d_model = kwargs.get("d_model", c_in)
        nhead = transformer_kwargs.get("nhead", 8)

        # 确保d_model能被nhead整除
        if d_model % nhead != 0:
            # 调整nhead为d_model的因子
            valid_nheads = [i for i in range(1, d_model + 1) if d_model % i == 0]
            nhead = min(valid_nheads, key=lambda x: abs(x - nhead))
            transformer_kwargs["nhead"] = nhead
            logger.warning(
                f"Adjusted nhead from {kwargs.get('nhead', 8)} to {nhead} to match d_model={d_model}"
            )

        return TemporalTransformerEncoder(d_model=d_model, **transformer_kwargs)
    else:
        raise ValueError(f"Unsupported temporal type: {temporal_type}")


# 导出接口
__all__ = [
    "TemporalConv1D",
    "TemporalTransformerEncoder",
    "FiLMTemporalBlock",
    "create_temporal_module",
]
