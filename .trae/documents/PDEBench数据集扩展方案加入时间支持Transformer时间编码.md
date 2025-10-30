### 开发流程：提升多步预测能力和引入时间编码器

---

## 1. **提升多步预测能力（NAR 头扩展 T_out）**

### **目标**

提升 **NAR 头** 的多步预测能力，支持 **更长的预测地平线**，例如 **T_out=5** 或 **T_out=10**。

### **开发步骤**

1. **扩展 `SwinTemporalNAR` 包装器中的 T_out**
    - **修改 `SwinTemporalNAR` 中的代码**：
        
        将 **T_out** 设置从原本的 3 扩展到 5 或 10。
        
        具体来说，需要确保 **NAR 头** 能够并行输出多个时间步，修改 **`nar_out` 输出维度**。
        
    - **检查 `TimeQueryHead` 输出**：
        
        确保 `TimeQueryHead` 被调整为 **一次性生成多个时步**，例如 T_out=5 或 T_out=10，并能够支持并行输出。
        
2. **修改 `TimeQueryHead` 以支持更长的 T_out**
    - **调整模型架构**：
        
        修改 `TimeQueryHead` 的结构，使其能够处理更大的输出维度。例如，修改 `nar_out` 部分的 `reshape` 操作，支持生成 T_out=5 或 T_out=10 时步。
        
        - **当前代码示例**：
            
            ```python
            nar_out = self.nar(mem, T_out=5)  # T_out=5
            
            ```
            
3. **测试与验证**
    - **测试 `T_out=5/10` 的稳定性**：
        - 确认 **T_out=5/10** 下的预测性能，重点查看 **Rel2_last** 是否显著优于 `T_out=3`。
        - 验证 **NAR** 模型推理时延（`Latency vs T_out`）是否稳定增长，不会因为 T_out 扩展而出现显著性能下降。
    - **对比 AR 和 NAR 预测效果**：
        - 测试 NAR 在 T_out=5 和 T_out=10 时的效果，并与 AR 模型进行对比，确保 **NAR** 的性能优于 **AR**。

### **验收标准**

- `T_out=5` 下，**`Rel2_last` 相较于 `T_out=3` 显著优于 AR**。
- 推理时延（`Latency vs T_out`）在 **T_out 增加时基本平稳**，不会随着时步的增加显著增加。

---

## 2. **引入时间编码器（Temporal Transformer 或 Conv1D）**

### **目标**

引入 **Temporal Transformer Encoder** 或 **Conv1D** 来增强 **时间建模能力**。

### **开发步骤**

1. **在 `SwinTemporalNAR` 中加入 Temporal Transformer Encoder 或 TemporalConv1D**
    - **引入 Temporal 模块**：
        
        选择 **Temporal Transformer Encoder** 或 **TemporalConv1D** 模块，接入 `SwinTemporalNAR`，使得模型能够处理时间维度的长程依赖。
        
        - **Temporal Transformer Encoder**：使用标准的 Transformer 编码器，应用在时间序列的 token 上。
        - **TemporalConv1D**：使用卷积网络的方式处理时间序列特征，计算时序信息。
2. **连接 Temporal 模块**
    - **在 `wrapper` 中连接 Temporal 模块**，确保能够处理多帧输入（例如 `T_in > 1`），让模型能对过去的多个帧进行时序建模。
    - 在 `SwinTemporalNAR` 中加入 **causal mask**（如果使用 Transformer），确保推理时只依赖过去的时刻，避免泄露未来的信息。
3. **验证模型效果**
    - **评估时间维度的长程依赖捕捉能力**：
        
        测试在 **T_out=5** 下，比较引入 **Temporal 模块** 后，模型的 **Rel2_last** 是否提升。
        
        - 确保 **`T_out=5`** 时，推理误差稳定，不会因为 **Temporal Encoder** 或 **TemporalConv1D** 的引入而显著增加。

### **验收标准**

- 在 **T_out=5** 下，与 **T_out=3** 对比，`Rel2_last` **提升至少 2%**，推理误差稳定，且不受时间维扩展的显著影响。
- 使用 **Temporal Transformer 或 Conv1D** 后，模型能够有效处理多帧输入，并增强时间维度的建模能力。

---

### 参考代码（用于 `TimeQueryHead` 和 Temporal 编码器）

### **NAR 头扩展 T_out**

```python
class TimeQueryHead(nn.Module):
    def __init__(self, d_model, c_out, T_out=5, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.T_out = T_out
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Conv2d(d_model, d_model, 1)
        self.v_proj = nn.Conv2d(d_model, d_model, 1)
        self.out = nn.Conv2d(d_model, c_out, 1)
        self.nhead = nhead

    def forward(self, mem, t_embed):
        B, D, H, W = mem.shape
        K = self.k_proj(mem).flatten(2).transpose(1,2)
        V = self.v_proj(mem).flatten(2).transpose(1,2)
        Q = self.q_proj(t_embed).unsqueeze(0).expand(B,-1,-1)
        attn = torch.softmax((Q @ K.transpose(1,2)) / (D**0.5), dim=-1)
        Z = attn @ V
        Z = Z.transpose(1,2).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,H,W)
        return self.out(Z.view(B*self.T_out, D, H, W)).view(B, -1, -1, H, W)

```

### **Temporal Transformer Encoder**

```python
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(TemporalTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, T, D) where T is the number of time steps
        return self.transformer_encoder(x)

```

### **TemporalConv1D**

```python
class TemporalConv1D(nn.Module):
    def __init__(self, c_in, k=3, causal=True):
        super(TemporalConv1D, self).__init__()
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=k, padding=k//2 if not causal else k-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T, C)
        return self.relu(self.conv(x.transpose(1, 2)).transpose(1, 2))

```

---

### 结论：

通过扩展 **T_out** 支持更多时步预测、引入 **Temporal Transformer Encoder** 或 **Conv1D** 来增强时间建模能力，你将能够进一步提升模型在 **长序列预测** 上的性能，并确保其稳定性与效率。