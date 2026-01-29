# 实现自适应数据集归一化 Loss 缩放 (Adaptive Dataset Normalization)

## 互联网搜索确认
搜索结果强烈支持您的观点：在 Physics-Informed Neural Networks (PINN) 训练中，**Loss 量级平衡至关重要**。
*   文献指出，当物理约束 Loss（如 PDE/边界条件）与数据 Loss 量级差异巨大时（通常由于物理单位导致），优化器会产生严重偏差，导致模型无法收敛或忽略数据拟合。
*   **结论**：必须对物理 Loss 进行缩放。相比于手动调参，**基于数据统计量的自适应归一化**是更稳健、更符合“Scaling Law”的做法。

## 执行计划
我将修改代码以实现自动缩放，而非修改配置文件中的硬编码权重。

### 1. 修改 `ops/enhanced_losses.py`
修改 `compute_enhanced_total_loss` 函数，引入基于 `norm_stats` 的自适应缩放机制：

*   **逻辑变更**：
    在计算 Spectral Loss 和 DC Loss（物理域）后，利用数据集的方差 $\sigma^2$ 进行归一化。
    $$ \text{Loss}_{\text{scaled}} = \text{Loss}_{\text{phys}} \times \frac{1}{\sigma^2 + \epsilon} $$
    这在数学上等价于将物理域误差投影回 Z-score 域的量级，同时保留了物理约束的相对梯度方向。

*   **代码实现**：
    1.  从 `norm_stats` 中提取 `sigma`。
    2.  计算缩放因子 `scale_factor = 1.0 / (sigma.pow(2).mean() + 1e-8)`（取通道平均方差）。
    3.  将 `spectral_loss` 和 `dc_loss` 乘以该因子。

### 2. 预期效果
*   **自动平衡**：无论数据集物理数值是 100 还是 0.001，Loss 都会自动对齐到 ~1.0 量级。
*   **配置解耦**：配置文件中的权重（1.0/0.5）将重新生效，无需针对每个数据集手动微调。
*   **解决现状**：Train Loss 将从 ~11221 立即降低到正常范围（~2-3）。

请确认执行此代码修改。