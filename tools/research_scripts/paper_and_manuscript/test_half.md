## 摘 要

在计算物理、环境监测与工业诊断等场景中，高分辨率时空物理场重建对下游预测、控制与决策具有重要意义。受限于传感器部署成本、数据采集带宽与复杂环境约束，实际观测往往呈现稀疏、非均匀与含噪等退化特征；同时，真实观测过程通常包含抗混叠滤波、边界裁剪与对齐规则，而许多学习方法在训练阶段采用理想化退化过程，导致训练与评测/部署口径不一致，从而引发泛化性能下降与结论可复现性不足。面向稀疏观测条件（例如全域覆盖率低于 $20\%$ 且伴随混叠与测量噪声），本文提出一种“评测口径一致性优先”的时空场重建框架。方法上，首先构建统一观测算子 $H$，并约束训练端退化过程 $DC$ 在插值核、抗混叠预滤、边界处理与对齐规则上与评测口径严格一致，以削弱由算子不匹配引入的隐性偏差；其次提出序列化时空训练策略，将任务分解为“空间重构预训练—时序演化预训练—时空联合微调”三阶段递进优化，以提升联合优化稳定性；最后设计由重建损失、低频谱一致性损失与原值域观测一致性损失构成的三元损失函数，在提升逼近精度的同时显式约束观测口径一致性。基于 PDEBench 的 Shallow Water（SWE）与 Diffusion–Reaction（DRD）子集的实验表明：在 SWE 全域重建设置中，相比轻量级基线（ResNetLite），PSNR 从 $46.52\,\mathrm{dB}$ 提升至 $71.05\,\mathrm{dB}$（提升 $24.53\,\mathrm{dB}$）；在 DRD 时空预测设置中，端到端融合空间重建与时序建模可避免稀疏观测下的模型崩溃风险，$\mathrm{Rel}\text{-}L_2$ 从 $0.9336$ 稳定至 $0.1783$；相较两阶段训练基线，端到端联合优化将高频频谱误差 $\mathrm{fRMSE}\text{-}\mathrm{High}$ 从 $4.4524$ 降至 $1.9236$（降低 $56.8\%$），更有利于恢复陡峭梯度与细尺度结构。在极度稀疏裁剪观测任务中，$16\times16$ 观测窗口仅占 $128\times128$ 全域的 $1.56\%$，引入全局注意力的 Transformer 仍能推断全局结构并保持可用重建质量。资源成本分析（FLOPs、显存占用与推理延迟）进一步验证了所提框架在工程应用中的可行性。

**关键词**：时空场重建；稀疏观测；观测口径一致性；Transformer；序列化训练；科学机器学习

---

## ABSTRACT

High-resolution spatiotemporal field reconstruction is crucial for downstream prediction, control, and decision-making in computational physics and engineering. Practical measurements are often sparse, non-uniform, and noisy due to constraints in sensing cost, bandwidth, and environmental complexity. Moreover, real observation processes typically involve anti-aliasing filtering, boundary cropping, and alignment rules, whereas many learning-based methods rely on idealized degradations during training. Such inconsistency between training degradations and evaluation/deployment observations leads to degraded generalization and weak reproducibility. Under sparse observations (e.g., $<20\%$ spatial coverage) with aliasing effects and measurement noise, this thesis proposes a consistency-first reconstruction framework. A Unified Observation Operator $H$ is defined, and the training-time degradation process $DC$ is constrained to strictly match $H$ in interpolation kernels, anti-aliasing pre-filtering, boundary handling, and alignment rules, reducing operator-induced bias. A sequential spatiotemporal training strategy is further introduced, progressively optimizing the model via spatial reconstruction pretraining, temporal evolution pretraining, and joint spatiotemporal fine-tuning to improve optimization stability. In addition, a tri-component loss is designed by combining reconstruction loss, low-frequency spectral consistency, and observation-domain consistency, enforcing both approximation accuracy and evaluation consistency. Experiments on the PDEBench Shallow Water (SWE) and Diffusion–Reaction (DRD) subsets demonstrate that, on SWE full-field reconstruction, PSNR increases from $46.52\,\mathrm{dB}$ (ResNetLite baseline) to $71.05\,\mathrm{dB}$, yielding a $24.53\,\mathrm{dB}$ gain. On DRD spatiotemporal prediction, end-to-end integration of spatial reconstruction and temporal modeling prevents collapse under sparse observations, stabilizing $\mathrm{Rel}\text{-}L_2$ from $0.9336$ to $0.1783$. Compared with a two-stage baseline, end-to-end joint optimization reduces the high-frequency spectral error $\mathrm{fRMSE}\text{-}\mathrm{High}$ from $4.4524$ to $1.9236$, corresponding to a $56.8\%$ reduction, which better preserves steep gradients and fine-scale structures. Under extreme cropped observations, a $16\times16$ window covers only $1.56\%$ of a $128\times128$ domain, while a global-attention Transformer still infers coherent global structures with usable reconstruction quality. Resource-cost analyses in FLOPs, memory footprint, and inference latency further support practical feasibility.

**Keywords**: spatiotemporal field reconstruction; sparse observation; observation-operator consistency; Transformer; sequential training; scientific machine learning



# 符号说明表 (Notation Table)

为确保论文叙述的严谨性与一致性，本文主要数学符号及其含义约定如下。除特殊说明外，全文遵循此表定义。

| 符号 (Symbol) | 类型 | 含义与说明 (Description) |
| :--- | :--- | :--- |
| **基础变量** |  |  |
| $u(\mathbf{x},t)$ | 连续场 | 真实物理场，定义在时空域 $\Omega\times[0,T]$ |
| $\mathbf{U}$ | 张量 | 真实场的高分辨率离散表示（原值域），维度通常为 $B\times T\times C\times H\times W$ |
| $\mathbf{U}^{(z)}$ | 张量 | $\mathbf{U}$ 的 z-score 标准化表示，用于训练 |
| $\mathbf{y}$ | 张量 | 稀疏观测数据：$\mathbf{y}=H(\mathbf{U})+\boldsymbol{\varepsilon}$ |
| $\tilde{\mathbf{U}}$ | 张量 | 模型重建的预测场（原值域），用于最终评测，与 $\mathbf{U}$ 同分辨率与维度 |
| $\hat{\mathbf{U}}^{(z)}$ | 张量 | 网络直接输出的预测场（z-score 域），反归一化后得到 $\tilde{\mathbf{U}}$ |
| $\mathbf{x}$ | 向量 | 空间坐标向量，$\mathbf{x}\in\mathbb{R}^d$（本文取 $d=2$） |
| $t$ | 标量 | 时间变量，$t\in[0,T]$ |
| $\boldsymbol{\varepsilon}$ | 张量 | 观测噪声，常设 $\boldsymbol{\varepsilon}\sim\mathcal{N}(0,\sigma_n^2)$ |
| $\sigma_{\mathrm{blur}}$ | 标量 | 观测算子 $H$ 中高斯抗混叠滤波器的标准差 |
| $\sigma_n$ | 标量 | 观测噪声标准差 |
| $\boldsymbol{\mu}_z$ | 向量 | 逐通道 z-score 标准化均值 |
| $\boldsymbol{\sigma}_z$ | 向量 | 逐通道 z-score 标准化标准差 |
| $\odot$ | 运算 | 逐元素乘（Hadamard 乘积） |
|  |  | 反归一化：$\tilde{\mathbf{U}}=\hat{\mathbf{U}}^{(z)}\odot\boldsymbol{\sigma}_z+\boldsymbol{\mu}_z$ |
| **算子与映射** |  |  |
| $H(\cdot)$ | 算子 | **观测算子 (Observation Operator)**。从高分辨率原值域离散场到稀疏观测的映射，包含抗混叠、降采样、裁剪、掩码/采样等过程 |
| $DC(\cdot)$ | 算子 | **训练退化算子 (Training Degradation Operator)**。训练阶段用于模拟观测生成，本文约束 $DC\equiv H$（同参数、同实现） |
| $G_{\sigma_{\mathrm{blur}}}(\cdot)$ | 算子 | 高斯低通（抗混叠）滤波算子，参数为 $\sigma_{\mathrm{blur}}$ |
| $D_s(\cdot)$ | 算子 | 下采样算子，降采样倍率为 $s$ |
| $C_{h_c,w_c}(\cdot)$ | 算子 | 裁剪算子，输出窗口大小为 $(h_c,w_c)$（常用中心对齐） |
| $M(\cdot)$ | 算子 | 掩码/采样算子：将全域场映射到稀疏观测位置或执行缺失掩码 |
|  |  | 常用组合写法：$H=M\circ C_{h_c,w_c}\circ D_s\circ G_{\sigma_{\mathrm{blur}}}$（按本文实现可删减某些环节） |
| $f_{\boldsymbol{\theta}}(\cdot)$ | 函数 | 深度神经网络重建模型，参数为 $\boldsymbol{\theta}$，输入为 $\mathbf{y}$（及辅助信息），输出为 $\hat{\mathbf{U}}^{(z)}$ |
| $\mathcal{F}(\cdot)$ | 变换 | 傅里叶变换 (Fourier Transform)，用于频域分析与谱损失计算 |
| **损失函数** |  |  |
| $\mathcal{L}_{\mathrm{total}}$ | 标量 | 总损失函数，用于反向传播优化 |
