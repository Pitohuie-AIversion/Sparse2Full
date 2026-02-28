# 移至第4章 (实验设置) 的内容

以下内容已从第3章移除，建议插入至 **第4章 实验 (Experiments)** 的 **4.1 实验设置** 小节。

## 4.1.x 训练与环境设置 (Training & Environmental Setup)

为确保实验结果的公平性与可复现性，所有实验均在统一的软硬件环境下执行，并遵循严格的参数配置协议。

### 1. 优化器与超参数 (Optimization)

*   **优化器 (Optimizer)**：采用 AdamW 优化器，以实现权重衰减与梯度更新的解耦。参数配置如下：
    *   动量参数：$\beta_1=0.9, \beta_2=0.999$
    *   权重衰减 (Weight Decay)：$1\times 10^{-4}$
*   **学习率调度 (Learning Rate Schedule)**：
    *   策略：余弦退火 (Cosine Annealing)
    *   初始学习率：$1\times 10^{-3}$
    *   预热策略 (Warmup)：前 5 个 epoch 采用线性预热，以缓解训练初期的梯度震荡。
*   **混合精度训练 (AMP)**：启用自动混合精度 (Automatic Mixed Precision, float16/bfloat16)，在保证数值稳定性的前提下，显著降低显存占用并提升计算吞吐量。

### 2. 可复现性控制 (Reproducibility & Audit)

针对深度学习实验中常见的“随机性黑盒”问题，本研究实施了以下工程级控制措施：

*   **全局随机种子 (Global Seed)**：统一固定 Python、NumPy 及 PyTorch 的随机种子 (Seed=42)，确保数据划分与模型初始化的确定性。
*   **确定性算法 (Deterministic Algorithms)**：在 PyTorch 中开启 `torch.use_deterministic_algorithms(True)`，强制使用确定性卷积算法，消除 GPU 并行计算引入的微小数值扰动。
*   **环境指纹 (Environment Fingerprint)**：每次实验启动时，系统自动抓取并记录 `env_fingerprint.json`。该文件包含：
    *   CUDA 驱动与 Toolkit 版本
    *   PyTorch 及核心科学计算库 (NumPy, SciPy) 版本
    *   GPU 硬件型号与拓扑信息
    这确保了跨时间、跨平台的实验结果具有严格的可比性基准。

### 3. 评测指标定义 (Evaluation Metrics)

本研究采用多维度指标全面评估重建质量与物理一致性。

*   **相对重建误差 (Rel-L2)**：
    $$
    \mathrm{Rel}\text{-}L_2=\frac{\lVert \tilde{\mathbf{U}}-\mathbf{U}\rVert_F}{\lVert \mathbf{U}\rVert_F}
    $$
    衡量全场逼近精度的核心指标，反映了重建场 $\tilde{\mathbf{U}}$ 与真实场 $\mathbf{U}$ 在 Frobenius 范数下的相对偏差。

*   **口径一致性误差 ($H_{\mathrm{err}}$)**：
    $$
    H_{\mathrm{err}}=\lVert H(\tilde{\mathbf{U}})-\mathbf{y}\rVert_F
    $$
    该指标衡量重建结果再次经过观测算子 $H$ 投影后，是否能回退到原始观测数据 $\mathbf{y}$。它是验证“观测一致性”假设的关键依据，直接反映了重建结果的物理可信度。
