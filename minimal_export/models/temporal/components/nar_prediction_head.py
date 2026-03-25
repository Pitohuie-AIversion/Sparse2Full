"""
NAR (Non-AutoRegressive) 多步预测头模块

该模块实现了基于交叉注意力的时间查询头，支持并行输出多个时间步的预测结果。
主要特性：
1. CrossAttnTimeQueryHead: 基于交叉注意力的时间查询机制
2. 支持可学习的时间查询向量
3. 支持位置编码和时间编码
4. 支持多尺度特征融合
5. 支持不同的输出投影策略

作者: SOLO Coding
日期: 2025-01-11
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTimeQueries(nn.Module):
    """可学习的时间查询向量

    为每个预测时间步生成可学习的查询向量，用于交叉注意力机制。
    """

    def __init__(
        self,
        T_out: int,
        hidden_dim: int,
        init_strategy: str = "normal",
        temperature: float = 1.0,
    ):
        """初始化时间查询向量

        Args:
            T_out: 输出时间步数
            hidden_dim: 隐藏维度
            init_strategy: 初始化策略 ("normal", "uniform", "xavier", "temporal")
            temperature: 温度参数，用于调节查询向量的尺度
        """
        super().__init__()
        self.T_out = T_out
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # 可学习的时间查询向量 [T_out, hidden_dim]
        self.time_queries = nn.Parameter(torch.empty(T_out, hidden_dim))

        # 初始化查询向量
        self._init_queries(init_strategy)

    def _init_queries(self, strategy: str):
        """初始化查询向量"""
        if strategy == "normal":
            nn.init.normal_(self.time_queries, std=0.02)
        elif strategy == "uniform":
            nn.init.uniform_(self.time_queries, -0.1, 0.1)
        elif strategy == "xavier":
            nn.init.xavier_uniform_(self.time_queries)
        elif strategy == "temporal":
            # 基于时间位置的初始化
            for t in range(self.T_out):
                # 使用正弦/余弦位置编码的思想
                for d in range(self.hidden_dim):
                    if d % 2 == 0:
                        self.time_queries.data[t, d] = math.sin(
                            t / (10000 ** (d / self.hidden_dim))
                        )
                    else:
                        self.time_queries.data[t, d] = math.cos(
                            t / (10000 ** ((d - 1) / self.hidden_dim))
                        )
        else:
            raise ValueError(f"未知的初始化策略: {strategy}")

    def forward(self, batch_size: int) -> torch.Tensor:
        """前向传播

        Args:
            batch_size: 批次大小

        Returns:
            时间查询向量 [B, T_out, hidden_dim]
        """
        # 扩展到批次维度
        queries = self.time_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # 应用温度缩放
        if self.temperature != 1.0:
            queries = queries / self.temperature

        return queries


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力模块

    实现查询向量与输入特征之间的交叉注意力计算。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """初始化多头交叉注意力

        Args:
            hidden_dim: 隐藏维度
            num_heads: 注意力头数
            dropout: Dropout概率
            bias: 是否使用偏置
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # 线性投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            query: 查询向量 [B, T_out, hidden_dim]
            key: 键向量 [B, T_in, hidden_dim]
            value: 值向量 [B, T_in, hidden_dim]
            mask: 注意力掩码 [B, T_out, T_in] 或 [B, num_heads, T_out, T_in]

        Returns:
            输出特征 [B, T_out, hidden_dim]
        """
        B, T_out, _ = query.shape
        _, T_in, _ = key.shape

        # 线性投影
        Q = self.q_proj(query)  # [B, T_out, hidden_dim]
        K = self.k_proj(key)  # [B, T_in, hidden_dim]
        V = self.v_proj(value)  # [B, T_in, hidden_dim]

        # 重塑为多头格式
        Q = Q.view(B, T_out, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, T_out, head_dim]
        K = K.view(B, T_in, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, T_in, head_dim]
        V = V.view(B, T_in, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, T_in, head_dim]

        # 计算注意力分数
        attn_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        )  # [B, num_heads, T_out, T_in]

        # 应用掩码
        if mask is not None:
            if mask.dim() == 3:  # [B, T_out, T_in]
                mask = mask.unsqueeze(1)  # [B, 1, T_out, T_in]
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, T_out, head_dim]

        # 重塑回原始格式
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, T_out, self.hidden_dim)
        )

        # 输出投影
        output = self.out_proj(attn_output)

        return output


class CrossAttnTimeQueryHead(nn.Module):
    """基于交叉注意力的时间查询预测头

    该模块使用可学习的时间查询向量，通过交叉注意力机制从输入序列中提取信息，
    并行生成多个时间步的预测结果。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        T_out: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
        query_init_strategy: str = "temporal",
        temperature: float = 1.0,
        output_projection: str = "linear",  # "linear", "conv", "mlp"
    ):
        """初始化交叉注意力时间查询头

        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            T_out: 输出时间步数
            hidden_dim: 隐藏维度
            num_heads: 注意力头数
            num_layers: 交叉注意力层数
            dropout: Dropout概率
            use_pos_encoding: 是否使用位置编码
            query_init_strategy: 查询向量初始化策略
            temperature: 温度参数
            output_projection: 输出投影类型
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T_out = T_out
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_pos_encoding = use_pos_encoding
        self.output_projection = output_projection

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 可学习的时间查询向量
        self.time_queries = LearnableTimeQueries(
            T_out=T_out,
            hidden_dim=hidden_dim,
            init_strategy=query_init_strategy,
            temperature=temperature,
        )

        # 位置编码（可选）
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1000, hidden_dim) * 0.02
            )  # 支持最多1000个时间步

        # 多层交叉注意力
        self.cross_attn_layers = nn.ModuleList(
            [
                MultiHeadCrossAttention(
                    hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # 层归一化
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # 前馈网络
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        # 输出投影
        self._build_output_projection()

    def _build_output_projection(self):
        """构建输出投影层"""
        if self.output_projection == "linear":
            self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        elif self.output_projection == "conv":
            # 使用1D卷积进行输出投影
            self.output_proj = nn.Sequential(
                nn.Conv1d(
                    self.hidden_dim, self.hidden_dim // 2, kernel_size=3, padding=1
                ),
                nn.GELU(),
                nn.Conv1d(self.hidden_dim // 2, self.output_dim, kernel_size=1),
            )
        elif self.output_projection == "mlp":
            # 使用多层感知机进行输出投影
            self.output_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.output_dim),
            )
        else:
            raise ValueError(f"未知的输出投影类型: {self.output_projection}")

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 [B, T_in, input_dim] 或 [B, T_in, C, H, W]
            mask: 输入掩码 [B, T_in]

        Returns:
            预测结果 [B, T_out, output_dim] 或 [B, T_out, C, H, W]
        """
        # 处理不同输入格式
        if x.dim() == 5:  # [B, T_in, C, H, W]
            B, T_in, C, H, W = x.shape
            x = x.view(B, T_in, C * H * W)
            spatial_shape = (C, H, W)
        elif x.dim() == 3:  # [B, T_in, input_dim]
            B, T_in = x.shape[:2]
            spatial_shape = None
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")

        # 输入投影
        x = self.input_proj(x)  # [B, T_in, hidden_dim]

        # 添加位置编码
        if self.use_pos_encoding:
            pos_enc = self.pos_encoding[:T_in].unsqueeze(0).expand(B, -1, -1)
            x = x + pos_enc

        # 获取时间查询向量
        queries = self.time_queries(B)  # [B, T_out, hidden_dim]

        # 多层交叉注意力
        for i in range(self.num_layers):
            # 交叉注意力
            attn_output = self.cross_attn_layers[i](
                query=queries, key=x, value=x, mask=mask
            )

            # 残差连接和层归一化
            queries = self.layer_norms[i](queries + attn_output)

            # 前馈网络
            ffn_output = self.ffn_layers[i](queries)
            queries = queries + ffn_output

        # 输出投影
        if self.output_projection == "conv":
            # 对于卷积投影，需要转置维度
            queries = queries.transpose(1, 2)  # [B, hidden_dim, T_out]
            output = self.output_proj(queries)  # [B, output_dim, T_out]
            output = output.transpose(1, 2)  # [B, T_out, output_dim]
        else:
            output = self.output_proj(queries)  # [B, T_out, output_dim]

        # 恢复空间维度
        if spatial_shape is not None:
            C, H, W = spatial_shape
            output = output.view(B, self.T_out, C, H, W)

        return output

    def get_attention_weights(
        self, x: torch.Tensor, layer_idx: int = -1
    ) -> torch.Tensor:
        """获取注意力权重（用于可视化）

        Args:
            x: 输入特征 [B, T_in, input_dim]
            layer_idx: 层索引，-1表示最后一层

        Returns:
            注意力权重 [B, num_heads, T_out, T_in]
        """
        # 简化版前向传播，只返回指定层的注意力权重
        if x.dim() == 5:
            B, T_in, C, H, W = x.shape
            x = x.view(B, T_in, C * H * W)

        x = self.input_proj(x)

        if self.use_pos_encoding:
            pos_enc = (
                self.pos_encoding[: x.size(1)].unsqueeze(0).expand(x.size(0), -1, -1)
            )
            x = x + pos_enc

        queries = self.time_queries(x.size(0))

        # 执行到指定层
        target_layer = layer_idx if layer_idx >= 0 else self.num_layers + layer_idx

        for i in range(target_layer + 1):
            if i == target_layer:
                # 在目标层返回注意力权重
                attn_layer = self.cross_attn_layers[i]
                Q = attn_layer.q_proj(queries)
                K = attn_layer.k_proj(x)

                B, T_out, _ = Q.shape
                _, T_in, _ = K.shape

                Q = Q.view(
                    B, T_out, attn_layer.num_heads, attn_layer.head_dim
                ).transpose(1, 2)
                K = K.view(
                    B, T_in, attn_layer.num_heads, attn_layer.head_dim
                ).transpose(1, 2)

                attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * attn_layer.scale
                attn_weights = F.softmax(attn_scores, dim=-1)

                return attn_weights
            else:
                # 正常前向传播
                attn_output = self.cross_attn_layers[i](queries, x, x)
                queries = self.layer_norms[i](queries + attn_output)
                ffn_output = self.ffn_layers[i](queries)
                queries = queries + ffn_output

        return None


def create_nar_prediction_head(
    input_dim: int, output_dim: int, T_out: int, head_type: str = "cross_attn", **kwargs
) -> nn.Module:
    """创建NAR预测头的工厂函数

    Args:
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        T_out: 输出时间步数
        head_type: 预测头类型
        **kwargs: 其他参数

    Returns:
        NAR预测头模块
    """
    if head_type == "cross_attn":
        return CrossAttnTimeQueryHead(
            input_dim=input_dim, output_dim=output_dim, T_out=T_out, **kwargs
        )
    else:
        raise ValueError(f"未知的预测头类型: {head_type}")


# 单元测试
if __name__ == "__main__":
    print("🧪 测试NAR多步预测头模块...")

    # 测试参数
    batch_size = 2
    T_in = 5
    T_out = 10
    input_dim = 64
    output_dim = 32
    hidden_dim = 128

    # 测试可学习时间查询向量
    print("🧪 测试可学习时间查询向量...")
    time_queries = LearnableTimeQueries(T_out=T_out, hidden_dim=hidden_dim)
    queries = time_queries(batch_size)
    print(f"时间查询向量形状: {queries.shape}")
    assert queries.shape == (batch_size, T_out, hidden_dim)

    # 测试多头交叉注意力
    print("🧪 测试多头交叉注意力...")
    cross_attn = MultiHeadCrossAttention(hidden_dim=hidden_dim, num_heads=8)
    query = torch.randn(batch_size, T_out, hidden_dim)
    key = torch.randn(batch_size, T_in, hidden_dim)
    value = torch.randn(batch_size, T_in, hidden_dim)
    attn_output = cross_attn(query, key, value)
    print(f"交叉注意力输出形状: {attn_output.shape}")
    assert attn_output.shape == (batch_size, T_out, hidden_dim)

    # 测试完整的NAR预测头 - 3D输入
    print("🧪 测试NAR预测头 (3D输入)...")
    nar_head = CrossAttnTimeQueryHead(
        input_dim=input_dim,
        output_dim=output_dim,
        T_out=T_out,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=2,
    )

    x_3d = torch.randn(batch_size, T_in, input_dim)
    output_3d = nar_head(x_3d)
    print(f"3D输入形状: {x_3d.shape}")
    print(f"3D输出形状: {output_3d.shape}")
    assert output_3d.shape == (batch_size, T_out, output_dim)

    # 测试5D输入
    print("🧪 测试NAR预测头 (5D输入)...")
    C, H, W = 3, 32, 32
    x_5d = torch.randn(batch_size, T_in, C, H, W)

    nar_head_5d = CrossAttnTimeQueryHead(
        input_dim=C * H * W,
        output_dim=C * H * W,
        T_out=T_out,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=2,
    )

    output_5d = nar_head_5d(x_5d)
    print(f"5D输入形状: {x_5d.shape}")
    print(f"5D输出形状: {output_5d.shape}")
    assert output_5d.shape == (batch_size, T_out, C, H, W)

    # 测试注意力权重获取
    print("🧪 测试注意力权重获取...")
    attn_weights = nar_head.get_attention_weights(x_3d, layer_idx=-1)
    print(f"注意力权重形状: {attn_weights.shape}")
    assert attn_weights.shape == (batch_size, 8, T_out, T_in)  # num_heads=8

    # 测试不同的输出投影类型
    print("🧪 测试不同输出投影类型...")
    for proj_type in ["linear", "conv", "mlp"]:
        nar_head_proj = CrossAttnTimeQueryHead(
            input_dim=input_dim,
            output_dim=output_dim,
            T_out=T_out,
            hidden_dim=hidden_dim,
            output_projection=proj_type,
        )
        output_proj = nar_head_proj(x_3d)
        print(f"{proj_type}投影输出形状: {output_proj.shape}")
        assert output_proj.shape == (batch_size, T_out, output_dim)

    # 测试工厂函数
    print("🧪 测试工厂函数...")
    factory_head = create_nar_prediction_head(
        input_dim=input_dim,
        output_dim=output_dim,
        T_out=T_out,
        head_type="cross_attn",
        hidden_dim=hidden_dim,
    )
    factory_output = factory_head(x_3d)
    print(f"工厂函数输出形状: {factory_output.shape}")
    assert factory_output.shape == (batch_size, T_out, output_dim)

    print("✅ NAR多步预测头模块测试完成！")
