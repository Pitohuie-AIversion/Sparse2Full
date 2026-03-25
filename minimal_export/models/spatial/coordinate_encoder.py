import math

import torch
import torch.nn as nn


class CoordinateEncoder(nn.Module):
    """
    坐标编码器：输出严格为 [B, encoding_dim, H, W]

    References (出处):
    - Positional (sin/cos 多频位置编码思路，广泛用于隐式神经表示/坐标网络):
      NeRF: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
      https://arxiv.org/abs/2003.08934

    - Random Fourier Features / Fourier Features（随机傅里叶特征，用于增强高频表达）:
      "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
      https://arxiv.org/abs/2006.10739

    - Random Fourier Features 的经典理论来源（核近似）:
      Rahimi & Recht, "Random Features for Large-Scale Kernel Machines" (NIPS 2007)
      https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
    """

    def __init__(
        self,
        encoding_type: str = "positional",
        encoding_dim: int = 64,
        max_freq: float = 10.0,
        include_pi: bool = True,
    ):
        super().__init__()
        self.encoding_type = encoding_type
        self.encoding_dim = int(encoding_dim)
        self.max_freq = float(max_freq)
        self.include_pi = bool(include_pi)

        if self.encoding_type == "positional":
            # NeRF-style 多频 sin/cos 编码（对坐标 x,y 使用一组 log-space 频率）:
            # https://arxiv.org/abs/2003.08934
            freq_dim = math.ceil(self.encoding_dim / 4)
            self.register_buffer(
                "freqs",
                torch.exp(torch.linspace(0.0, math.log(self.max_freq), freq_dim)),
                persistent=False,
            )
            raw_dim = 4 * freq_dim  # x: sin/cos + y: sin/cos
            self.proj = (
                nn.Identity()
                if raw_dim == self.encoding_dim
                else nn.Conv2d(raw_dim, self.encoding_dim, 1)
            )

        elif self.encoding_type == "fourier":
            # Random Fourier Features / Fourier Features（Tancik et al.）:
            # https://arxiv.org/abs/2006.10739
            # 经典理论：Rahimi & Recht (NIPS 2007)
            half = math.ceil(self.encoding_dim / 2)
            B = torch.randn(half, 2) * self.max_freq
            self.register_buffer("B", B, persistent=False)
            raw_dim = 2 * half  # sin + cos
            self.proj = (
                nn.Identity()
                if raw_dim == self.encoding_dim
                else nn.Conv2d(raw_dim, self.encoding_dim, 1)
            )

        else:
            self.proj = None  # 'none'：直接返回 coords（2通道）

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [B, 2, H, W]，建议已归一化到 [-1, 1]
        """
        if self.encoding_type == "positional":
            return self._positional(coords)
        if self.encoding_type == "fourier":
            return self._fourier(coords)
        return coords  # [B,2,H,W]

    def _positional(self, coords: torch.Tensor) -> torch.Tensor:
        """
        NeRF-style positional encoding:
        - 对每个坐标维度，用一组频率做 sin/cos，得到多频特征
        Reference: https://arxiv.org/abs/2003.08934
        """
        B, _, H, W = coords.shape
        x = coords[:, 0:1]  # [B,1,H,W]
        y = coords[:, 1:2]

        freqs = self.freqs.view(1, 1, 1, 1, -1)  # [1,1,1,1,f]
        scale = math.pi if self.include_pi else 1.0

        xw = x.unsqueeze(-1) * freqs * scale  # [B,1,H,W,f]
        yw = y.unsqueeze(-1) * freqs * scale

        # 拼到通道维： [B,4,H,W,f] -> [B,4f,H,W]
        enc = torch.cat(
            [torch.sin(xw), torch.cos(xw), torch.sin(yw), torch.cos(yw)], dim=1
        )
        enc = enc.permute(0, 1, 4, 2, 3).contiguous().view(B, -1, H, W)
        return self.proj(enc)

    def _fourier(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Fourier Features / Random Fourier Features:
        - 使用随机矩阵 B 将坐标投影到高维，再做 sin/cos
        Reference: https://arxiv.org/abs/2006.10739
        Theory: Rahimi & Recht (NIPS 2007)
        """
        B, _, H, W = coords.shape
        c = coords.permute(0, 2, 3, 1).contiguous().view(-1, 2)  # [BHW,2]
        proj = c @ self.B.t()  # [BHW,half]
        scale = 2 * math.pi if self.include_pi else 1.0
        proj = proj * scale
        enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [BHW, 2*half]
        enc = enc.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B,2*half,H,W]
        return self.proj(enc)
