下面这份是**针对你当前 GitHub「Sparse2Full」仓库**、在已完成 **AR 基线** 上的**下一步开发手册**。所有改动都尽量低侵入，保持你现有 **Swin-UNet 单帧接口**与训练脚本兼容。

---

# Sparse2Full｜下一步开发手册（基于已完成 AR）

## 0. 范围与目标

- **范围**：在现有 `models/swin_unet.py` + AR 包装器基础上，加入**轻时序模块（Temporal）与非自回归（NAR）头**，统一时序 I/O 与评测。
- **目标**：
    1. 让模型“看见历史”（T_in>1），提升短地平线预测；
    2. 一次性并行预测多步（NAR），降低误差累积、加速推理；
    3. 保持与现数据/损失/评测管线兼容。

---

## 1. 目录增量与文件改动

```
models/
  temporal_block.py        # ← 新增：TemporalConv1D / FiLM（只看时间维）
  decoder/
    query_head.py          # ← 新增：TimeQueryHead（NAR 一次性多步输出）
  wrappers/
    swin_temporal.py       # ← 新增：把 Temporal 接到 Swin 前端（保持单帧接口）
    ar_nar_wrapper.py      # ← 新增：组合 AR 与 NAR 双头（训练用）
# 现有：
models/swin_unet.py        # 不改或仅加一个“返回中间特征”开关（可选）
ops/losses.py              # 复用；新增/封装多步损失聚合
ops/metrics.py             # 确认支持 (B,T,C,H,W) 的 mean/last/worst
configs/train.yaml         # 新增时序/NAR配置键
train.py                   # 根据 arch/flags 构建相应包装器

```

> 不破坏：SwinUNet.forward(B,C,H,W)->(B,C_out,H,W) 保持不变；有序在外层完成时序与多步。
> 

---

## 2. 接口与配置规范

### 2.1 张量与 batch 键

- 训练输入：
    - `baseline_seq: [B, T_in, C, H, W]`（若暂时没有，`baseline[:,None]` 兼容）
    - `target_seq: [B, T_out, C, H, W]`
- 仍可用：`baseline: [B,C,H,W]`, `target: [B,C,H,W]`（单帧/回归测试）
- 稀疏信息（选用）：`coords[2,H,W]`, `mask[1,H,W]`, `h_params`（DC 用）

### 2.2 Hydra 配置新增

```yaml
model:
  arch: swin_temporal_nar         # swin | swin_ar | swin_temporal | swin_temporal_nar
  temporal:
    enabled: true
    type: conv1d                  # conv1d | film
    k: 3
    causal: true
  heads:
    use_ar:  true                  # 训练期并行，推理可只用 NAR
    use_nar: true
    nar: { d_model: 96 }          # 对齐 Swin 最浅层 embed_dim
data:
  temporal: { enabled: true, T_in: 4, T_out: 5, mode: forecast }
loss:
  rec_nar_weight: 1.0
  rec_ar_weight:  0.3
  # 频谱/DC/梯度先关，稳定后再开
train:
  amp: true
  grad_clip: 1.0
  warmup_steps: 1000
  schedule_sampling: { p_start: 0.0, p_end: 0.5, warm_epochs: 5 }

```

---

## 3. 模块最小实现（可直接落地）

### 3.1 Temporal（轻时序）

`models/temporal_block.py`

```python
import torch, torch.nn as nn

class TemporalConv1D(nn.Module):
    def __init__(self, c_in, c_out=None, k=3, causal=True):
        super().__init__()
        self.c_out = c_out or c_in
        pad = (k-1) if causal else (k-1)//2
        self.conv = nn.Conv1d(c_in, self.c_out, k, padding=pad)
        self.causal = causal
    def forward(self, x):  # x: (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.permute(0,3,4,2,1).contiguous().view(B*H*W, C, T)
        x = self.conv(x)
        x = x[..., -1] if self.causal else x.mean(-1)
        return x.view(B,H,W,-1).permute(0,3,1,2).contiguous()

```

### 3.2 NAR 时间查询头（简版）

`models/decoder/query_head.py`

```python
import torch, torch.nn as nn, math

def sinusoid_time_embed(t, d):
    pe = torch.zeros(t.shape[0], d, device=t.device)
    div = torch.exp(torch.arange(0,d,2,device=t.device)*-(math.log(10000.0)/d))
    pe[:,0::2] = torch.sin(t[:,None]*div); pe[:,1::2] = torch.cos(t[:,None]*div)
    return pe  # (T_out, d)

class TimeQueryHead(nn.Module):
    def __init__(self, d_model, c_out):
        super().__init__()
        self.to_kv = nn.Conv2d(d_model, 2*d_model, 1)
        self.proj  = nn.Conv2d(d_model, c_out, 1)
    def forward(self, mem, T_out):  # mem: (B,D,H,W)
        B,D,H,W = mem.shape
        K,V = self.to_kv(mem).split(D, dim=1)
        t = torch.arange(1, T_out+1, device=mem.device, dtype=torch.float32)
        Qt = sinusoid_time_embed(t, D).view(T_out, D, 1, 1)
        outs=[]
        for i in range(T_out):
            z = V * Qt[i]                  # 时间条件化（轻量稳定）
            outs.append(self.proj(z).unsqueeze(1))
        return torch.cat(outs, 1)          # (B,T_out,C,H,W)

```

### 3.3 包装器（Temporal 接入 + 双头）

`models/wrappers/swin_temporal.py`

```python
import torch, torch.nn as nn
from models.temporal_block import TemporalConv1D
from models.decoder.query_head import TimeQueryHead
from models.swin_unet import SwinUNet

class SwinTemporalNAR(nn.Module):
    def __init__(self, base_kwargs, temporal_cfg, nar_cfg, use_ar=True):
        super().__init__()
        self.backbone = SwinUNet(**base_kwargs)         # 你的主干不改
        self.use_ar = use_ar
        # temporal
        self.temporal = None
        if temporal_cfg.get("enabled", False):
            self.temporal = TemporalConv1D(
                c_in=base_kwargs["in_channels"],
                k=temporal_cfg.get("k",3),
                causal=temporal_cfg.get("causal",True)
            )
        # NAR head
        d_model = nar_cfg.get("d_model", self.backbone.embed_dim)
        self.nar = TimeQueryHead(d_model=d_model, c_out=base_kwargs["out_channels"])

    def _to_mem(self, x):  # 从 Swin 打一处 2D 特征作为 memory
        # 简便：使用 patch_embed 输出 reshape 成 (B,D,H,W)
        pe = self.backbone.patch_embed(x)        # (B,N,D)
        B,N,D = pe.shape; H=W=int(N**0.5)
        return pe.transpose(1,2).view(B,D,H,W)

    def forward(self, x_seq, T_out=1, teacher_seq=None, train_mode=True):
        # 1) 聚合历史为单帧
        x = x_seq[:, -1] if x_seq.dim()==5 else x_seq
        if self.temporal is not None and x_seq.dim()==5:
            x = self.temporal(x_seq)             # (B,C,H,W)
        # 2) NAR 一次性多步
        mem = self._to_mem(x)
        nar_out = self.nar(mem, T_out=T_out)     # (B,T_out,C,H,W)
        # 3) 可选 AR 对照（单帧复制成T_out步；或用你已有 ARWrapper）
        ar_out = None
        if self.use_ar:
            y1 = self.backbone(x).unsqueeze(1)
            ar_out = y1.repeat(1, T_out, 1, 1, 1)
        return ar_out, nar_out

```

---

## 4. 训练与损失（双头并行，先稳再强）

### 4.1 训练步伪码（与 train.py 对齐）

```python
# 取 batch
x_seq = batch.get("baseline_seq", batch["baseline"].unsqueeze(1))  # (B,T_in,C,H,W) or (B,1,...)
y_seq = batch.get("target_seq")                                    # (B,T_out,C,H,W)
T_out = cfg.data.temporal.T_out

# 前向
ar_out, nar_out = model(x_seq, T_out=T_out, teacher_seq=y_seq, train_mode=True)

# 损失
loss_rec_nar = (nar_out - y_seq).abs().mean() * cfg.loss.rec_nar_weight
loss_rec_ar  = 0.0
if ar_out is not None:
    loss_rec_ar = (ar_out - y_seq).abs().mean() * cfg.loss.rec_ar_weight
loss = loss_rec_nar + loss_rec_ar

# 反向与日志
loss.backward(); optimizer.step(); optimizer.zero_grad()
log = compute_all_metrics(nar_out, y_seq)  # 统一评测 NAR；可另记 AR

```

### 4.2 Scheduled Sampling（若仍使用 AR teacher forcing 训练）

在你的 ARWrapper 中加入概率替换真值为预测，线性从 `p_start→p_end`。

---

## 5. 验收清单（每阶段 3 条）

### P1（AR 打磨）

- [ ]  `T_out=3` roll-out 稳定，无 NaN；
- [ ]  训练/验证曲线平滑；
- [ ]  记录 `Rel2_mean/Rel2_last/Rel2_worst` 与 latency vs T_out（线性增长）。

### P2（Temporal 接入）

- [ ]  同等 `T_in/T_out` 下，相比纯 AR，`Rel2_last` 下降 ≥3–5%；
- [ ]  额外显存/时延开销 ≤ +15%；
- [ ]  仍无 NaN，学习率无需明显下调。

### P3（NAR 头上线）

- [ ]  `T_out=5/10`，**NAR 推理延迟**基本不随步数增长；
- [ ]  `Rel2_last` 长地平线优于 AR；
- [ ]  双头联合训练稳定（AR 权重 0.2–0.3 即可）。

---

## 6. 常见坑与快速修复

- **显存暴涨**：确保 Temporal 只在**时间维**做 1D 卷积；NAR 头用条件调制版，先不做全注意力。
- **数值震荡**：AMP + grad clip=1.0 + 线性 warmup；先只开重建损失（标准化域）。
- **维度不整齐**：`patch_embed.num_patches` 必须是平方数；否则 NAR mem reshape 时先插值对齐。
- **评测不一致**：统一 `(B,T,C,H,W)` 评测 API；报表写入 mean/last/worst 三列。

---

## 7. 后续增强（等你通过 P3 再做）

- **更强 NAR**：把 TimeQueryHead 换成 **Cross-Attention**（Q=时间查询, KV=mem 特征）；
- **Temporal Transformer**：把 TemporalConv1D 换成**仅时间维**的 Transformer-Encoder（1–2 层，低秩/线性注意）；
- **FNO 并联/瓶颈**：修正你现有 FNO 权重为 `cfloat`，并与 Swin 并联后 1×1 融合；
- **Graph 模块**：在编码深层插入稀疏感知注意力（读取 `coords/mask`）。

---

### 一句话路线

先把**轻时序（Temporal）接上 → 再用 NAR 头一次性多步输出 → 以 双头训练稳定收敛与对照 → 通过后再升级为 时间注意/跨注意 的强版 NAR。
这套改动都在外层包装**完成，**不动你的 Swin-UNet 主干**，最快速、最安全。