# 补充实验建议清单 (Supplementary Experiment Recommendations)

## 1. 实验缺口分析 (Gap Analysis)

基于对第4章实验结果的深度审查，当前实验体系已非常完备，但在以下三个维度存在进一步提升的空间，可显著增强论文的说服力与理论深度。

### 1.1 鲁棒性边界 (Robustness)
*   **现状**：已在 4.4.3 节测试了 $\sigma_n \in [0.01, 0.10]$ 的噪声敏感性，且发现了模型在低噪下的敏感性（Rel-L2 翻倍）。
*   **缺口**：缺乏针对“观测位置扰动”的测试。实际传感器可能存在位置漂移（Jitter），固定网格假设可能过于理想化。
*   **建议**：增加 **Grid Jitter Robustness** 实验。

### 1.2 泛化能力 (Generalization)
*   **现状**：主要在 SWE 和 DRD 的标准参数下测试。
*   **缺口**：缺乏对物理参数变化的泛化测试（Out-of-Distribution, OOD）。例如，在 $Re=100$ 上训练，能否在 $Re=500$ 上泛化？
*   **建议**：增加 **Parameter Extrapolation** 实验。

### 1.3 统计显著性 (Statistical Rigor)
*   **现状**：已执行 3 种子重复与 Paired t-test。
*   **缺口**：对于深度学习论文，3 次重复是及格线。若能提升至 5 次，将更具说服力。
*   **建议**：可视计算资源情况，将主实验（Table 4-2）的种子数提升至 5。

---

## 2. 建议补充的实验清单 (Prioritized List)

以下实验按优先级排序，请根据剩余时间和算力资源酌情选择。

### P0: 必须补充 (Critical) - 直接回应审稿人常见质疑

#### 实验 1: 物理参数外推泛化 (OOD Generalization)
*   **目的**：验证模型是否学到了物理规律，而非仅拟合特定数据集分布。
*   **方法**：
    *   **训练集**：使用 SWE (Shallow Water Equation) 的标准参数配置。
    *   **测试集**：使用具有不同初始条件或雷诺数的 SWE 样本（PDEBench 中有现成数据）。
    *   **指标**：Rel-L2 (In-Distribution) vs Rel-L2 (OOD)。
*   **预期成果**：证明 Consistency-First 框架在物理参数变化时，性能衰减显著小于 Baseline（如 UNet），体现“物理一致性”带来的泛化优势。

### P1: 强烈建议 (High Value) - 增强论文厚度

#### 实验 2: 网格位置扰动测试 (Grid Jitter Robustness)
*   **目的**：模拟真实传感器的安装误差或位置漂移。
*   **方法**：
    *   在测试阶段，对规则观测网格 $(x_i, y_i)$ 施加随机偏移：$(x_i + \delta_x, y_i + \delta_y)$，其中 $\delta \sim U(-\epsilon, \epsilon)$。
    *   **对比**：Ours (Unified Operator) vs Baseline (Standard Interpolation)。
*   **预期成果**：展示统一算子对采样位置微小偏差的容忍度，进一步佐证 $DC \equiv H$ 的鲁棒性。

### P2: 锦上添花 (Nice to Have) - 提升数据置信度

#### 实验 3: 种子数扩增 (Seed Expansion)
*   **目的**：降低偶然性，提升统计检验的效力（Power）。
*   **方法**：将 Table 4-2（主架构扫描）的随机种子从 3 个增加到 5 个（例如添加 Seed 123, 999）。
*   **预期成果**：标准差（Std）进一步降低，t-test 的 p-value 更显著（$p < 0.01$）。

---

## 3. 执行建议

*   **时间预算**：
    *   实验 1 (OOD)：约需 12 GPU-Hours（仅需推理，无需重训）。
    *   实验 2 (Jitter)：约需 4 GPU-Hours（仅需推理）。
    *   实验 3 (Seeds)：约需 48 GPU-Hours（需重训）。
*   **推荐路径**：优先完成 **实验 1** 和 **实验 2**，这两者仅需使用现有模型在新的测试集上推理，成本极低但论证价值极高。

## 4. 衍生分析建议

*   **频谱误差的可视化**：在 4.2.5 节中，目前的功率谱是 1D 曲线。建议补充 **2D 误差谱 (Error Spectrum)** 热图，直观展示误差在波数空间 $(k_x, k_y)$ 的分布，这能更清晰地揭示“各向异性”误差。
