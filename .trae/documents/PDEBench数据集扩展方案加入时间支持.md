好的！下面这份是**围绕你当前 Swin-UNet 代码**的**自回归（AR）开发路线**。目标是：**零/低侵入地**把多步预测跑通，先当强基线，并给出训练/推理/评测的一整套落地方案。

---

# 🧭 总目标

- 利用你现有 `SwinUNet.forward(x[B,C,H,W])→y[B,C_out,H,W]`，
- 在**外层**加一个极薄的 AR 包装器：
    - **训练**：teacher forcing（喂真值）
    - **推理**：roll-out（喂模型上一步输出）
- 输出统一为 `[B, T_out, C_out, H, W]`，与后续 NAR/时序分支共用指标与可视化。

---

# 1) 目录与文件改动（最小集）

```
models/
  swin_unet.py                # 你现有主干，不改或仅加返回中间特征的开关
  ar/
    __init__.py
    wrapper.py                # ← 新增：AR 包装器（训练/推理）
ops/
  losses.py                   # 现有，新增/复用多步损失聚合函数
ops/
  metrics.py                  # 现有，确保支持多步版本
configs/
  train.yaml                  # 新增 AR 相关开关
train.py                      # Hydra 主入口，小改：根据 arch 选择模型

```

---

# 2) AR 包装器（可直接用）

```python
# models/ar/wrapper.py
import torch
import torch.nn as nn
from typing import Optional

class ARWrapper(nn.Module):
    """
    将单帧模型包装成多步自回归预测：
    - 训练：teacher forcing（每步用真值作为下一步输入）
    - 推理：roll-out（每步用上一步预测作为下一步输入）
    兼容：baseline/target 在标准化域；外层训练脚本已做反归一化等。
    """
    def __init__(self, single_frame_model: nn.Module, detach_rollout: bool = True):
        super().__init__()
        self.m = single_frame_model
        self.detach_rollout = detach_rollout  # 推理/评估阶段避免梯度累积

    @torch.no_grad()
    def _rollout(self, x0: torch.Tensor, T_out: int) -> torch.Tensor:
        """推理：以 x0 作为第1步输入，串行滚动输出 T_out 帧"""
        self.m.eval()
        last = x0
        outs = []
        for _ in range(T_out):
            y = self.m(last)                  # (B,C,H,W)
            outs.append(y.unsqueeze(1))
            last = y if not self.detach_rollout else y.detach()
        return torch.cat(outs, dim=1)         # (B,T_out,C,H,W)

    def forward(
        self,
        x_in: torch.Tensor,                 # (B,T_in,C,H,W) 或 (B,C,H,W)
        T_out: int = 1,
        teacher: Optional[torch.Tensor] = None,  # (B,T_out,C,H,W), 训练时可传
        train_mode: bool = True
    ) -> torch.Tensor:
        if x_in.dim() == 4:  # (B,C,H,W)
            last = x_in
        else:
            last = x_in[:, -1]  # 以最后可用帧作为第1步输入

        if (not train_mode) or (teacher is None):
            return self._rollout(last, T_out)

        # 训练：teacher forcing
        outs = []
        for t in range(T_out):
            y = self.m(last)
            outs.append(y.unsqueeze(1))
            last = teacher[:, t]  # 用真值替代，稳定训练
        return torch.cat(outs, dim=1)

```

> 放置：models/ar/wrapper.py。
> 
> 
> 这不触碰你的 Swin-UNet 主干，随时可撤回。
> 

---

# 3) 训练脚本对接（Hydra 主入口 `train.py`）

**模型构建**（示例伪码）：

```python
# train.py 片段
from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper

def build_model(cfg):
    base = SwinUNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        img_size=cfg.data.img_size,
        # ...其余超参从 cfg 传入
        use_fno_bottleneck=cfg.model.fno.use_fno_bottleneck,
        fno_modes=cfg.model.fno.fno_modes,
    )
    if cfg.model.arch == "swin_ar":
        return ARWrapper(base, detach_rollout=True)
    return base

```

**前向与损失**（训练步骤伪码）：

```python
# 训练 step
# batch: dict，已含 baseline[B,C,H,W] / target[B,C,H,W] / (可选) target_seq[B,T_out,C,H,W]
x_in = batch.get("baseline_seq", None)  # 若暂时无多帧输入，可用 None
if x_in is None:
    # 先用最后一帧观测当输入（或 baseline 单帧）
    x_in = batch["baseline"].unsqueeze(1)  # (B,1,C,H,W)

teacher = batch.get("target_seq", None)    # (B,T_out,C,H,W)
pred_seq = model(
    x_in, T_out=cfg.data.temporal.T_out,
    teacher=teacher, train_mode=model.training
)  # (B,T_out,C,H,W)

loss, loss_items = compute_ar_loss(pred_seq, teacher, cfg.loss)  # 见下一节

```

---

# 4) 损失与指标（AR 版）

**损失聚合**（与现有 `compute_total_loss` 保持风格一致；在标准化域先只开重建项）：

```python
# ops/losses.py 片段
import torch
import torch.nn.functional as F

def l1_mae(x, y):  return (x - y).abs().mean()
def rel_l2(x, y, eps=1e-8):
    num = torch.sqrt(((x-y)**2).sum(dim=(2,3,4)))
    den = torch.sqrt((y**2).sum(dim=(2,3,4))) + eps
    return (num/den).mean()

def compute_ar_loss(pred_seq, gt_seq, cfg_loss):
    # pred_seq, gt_seq: (B,T_out,C,H,W)
    assert gt_seq is not None, "AR 训练需要 teacher（target_seq）"
    w_rel2 = cfg_loss.get("rel2_weight", 1.0)
    w_mae  = cfg_loss.get("mae_weight", 0.1)
    # 按时间步平均（也可加权）
    rel2 = rel_l2(pred_seq, gt_seq)
    mae  = l1_mae(pred_seq, gt_seq)
    loss = w_rel2*rel2 + w_mae*mae
    return loss, {"rel2": rel2.item(), "mae": mae.item()}

```

**指标评估**（共用你的 `ops/metrics.py`，确保支持 `[B,T,...]`；若当前只支持单帧，评估时在时间维取均值再传入）。

---

# 5) 配置（Hydra/YAML）

```yaml
# configs/train.yaml
model:
  arch: "swin_ar"                  # "swin" | "swin_ar"
  in_channels:  C_in
  out_channels: C_out
  fno:
    use_fno_bottleneck: false      # 若要对比 FNO 版，设 true
    fno_modes: 16

data:
  img_size: 256
  temporal:
    enabled: true
    T_in:  1                        # 先 1（只用最后帧），后续接多帧也可
    T_out: 3
    mode: "forecast"

loss:
  rel2_weight: 1.0
  mae_weight:  0.1
  # freq/DC/grad 先关，稳定后再开

```

---

# 6) 阶段性验证（建议顺序）

1. **M0-AR(1→1)**：`T_in=1, T_out=1`
    - 用最后一帧 baseline 作为输入；
    - 目标是下一帧（或当前帧先等价性测试）；
    - 检查与单帧版数值一致性（当 teacher 即当前帧时）。
2. **M0-AR(1→3)**：`T_in=1, T_out=3`
    - 训练：teacher forcing；
    - 验证/推理：roll-out；
    - 记录 MAE/Rel2 随步数的退化曲线与时延（AR 会线性增长）。
3. **M2-AR(FNO)**：打开 `use_fno_bottleneck`
    - 对比长地平线时的稳定性变化；
    - 高频/低频的误差分解（你已有频谱指标时再开）。
4. **噪声/掩码消融**（SR 模式掩码恒 1，可跳；Crop 模式可测）
    - 观测退化参数 `h_params` 保持一致，仅改 AR 结构。

---

# 7) 训练技巧（让 AR 更稳）

- **scheduled sampling（选做）**：
    
    训练后期以概率 (p) 用模型上一步输出替代真值（从 0 → 0.5 线性提升），缓解训练/推理分布差异。
    
    在 `ARWrapper.forward` 的训练分支中，把
    
    `last = teacher[:, t]`
    
    改为
    
    `last = teacher[:, t] if torch.rand(1)<(1-p) else y.detach()`。
    
- **梯度/稳定**：AMP + grad clip 1.0；学习率先和单帧配置一致；
- **缓存（后续）**：若加了时序模块再考虑缓存中间特征；当前单帧 Swin 没必要。

---

# 8) 日志与可视化

- 统一把输出存为 `[B,T_out,C,H,W]`，便于与你现有 `compute_all_metrics` 适配；
- 每个 epoch 画一张 **时间步对比图**：`(input last) / step1 / step2 / step3 / target step3 / error step3`；
- 记录 **latency vs T_out**：AR 串行应近似线性增长（后面和 NAR 做对照）。

---

# 9) 与现有管线对齐的小注意

- **标准化域**：AR 训练与评测都在标准化域先对齐（你数据层的 z-score），频谱/DC 在反归一化后再开；
- **键名**：Hydra 主入口用 `baseline_seq` / `target_seq` 更清晰；若暂时没有多帧 `baseline_seq`，就把 `baseline` 扩成 `[:,None]`；
- **SR/Crop 模式**：AR 与模式无关，输入始终走 `baseline`（或其时间序列），评估 target 对齐到同一分辨率即可。

---

## ✅ 交付物清单（本轮完成即具备 AR 能力）

1. `models/ar/wrapper.py`（如上代码）
2. `train.py` 中 `build_model` 的小分支（`arch: swin_ar`）
3. `ops/losses.py` 中 `compute_ar_loss`（如上）
4. `configs/train.yaml` 新增字段
5. 两个实验脚本/配置：
    - `T_out=1` 等价性跑通
    - `T_out=3` 多步 roll-out

---

需要的话，我可以把 **完整可直接运行的训练 step**（含从 `batch` 取键、喂 ARWrapper、落日志）的代码片段也写好，和你当前 Hydra 入口的键名对齐。你要不要我顺手把 `train.py` 的关键 30 行示例也贴出来？