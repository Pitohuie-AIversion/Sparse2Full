# 摘要

**背景与意义**：在计算物理、环境监测与工业诊断等领域，高分辨率时空物理场的重建对于下游的预测、控制与决策至关重要。然而，受限于传感器成本、采集带宽与环境约束，实际工程中的观测数据往往呈现稀疏、非均匀且伴随噪声的退化形态。虽然近年来以物理约束神经网络（PINN）和神经算子（Neural Operator）为代表的科学机器学习方法在求解偏微分方程（PDE）反问题上取得了显著进展，但在面向真实观测的重建任务中，普遍存在“训练退化模型与评测观测口径不一致”的现象，导致模型在实验指标上表现优异，但在实际部署中性能显著下降，且实验结论难以在跨研究间精确复现。

**问题定义**：本文的核心研究问题是：如何在观测极度稀疏（如仅覆盖全域 <1%）且存在混叠与噪声的条件下，构建一个既能恢复高频物理细节，又能保证评测口径一致性的时空场重建框架。其中的关键挑战在于消除训练阶段的模拟退化算子（Degradation Operator）与测试阶段的真实观测算子（Observation Operator）之间的“语义断裂”，并解决时空联合优化中的收敛不稳定性与频谱偏差问题。

**研究方法**：针对上述挑战，本文提出了一种“评测口径一致性优先”的稀疏观测时空场重建方法论。首先，构建了**统一观测算子（Unified Observation Operator, $H$）**框架，强制训练端的退化算子（$DC$）在插值核、抗混叠预滤、边界策略与对齐规则上与数据侧的观测算子（$H$）保持镜像复用，从根本上消除了隐性域偏差。其次，提出了**序列化时空训练策略（Sequential Spatiotemporal Training）**，基于课程学习思想，将复杂的时空重建任务分解为“空间重构预训练 $\to$ 时序演化预训练 $\to$ 时空联合微调”三个阶段，有效规避了直接端到端训练的局部极小值。最后，设计了包含重建损失、低频谱一致性损失与原值域观测一致性损失的**三元损失函数**，在保证数学逼近精度的同时，显式约束模型输出符合观测口径的物理一致性。

**实验结果**：基于 PDEBench 基准数据集（涵盖 Navier-Stokes、浅水方程等典型流体动力学场景）的广泛实验表明，本文提出的物理驱动稀疏重建框架具有显著的通用性与鲁棒性。消融实验揭示，高质量的空间重建是准确时序预测的先决条件：相比于低分辨率输入，采用高保真重建使时序预测的相对误差（Rel-L2）降低了一个数量级（从 11.67% 降至 0.88%），信噪比（PSNR）提升超过 22 dB。在轻量级模型（如 U-Net）上，引入三元损失约束使相对 $L_2$ 误差降低了 **11.5%**，且评测口径误差（$H_{\mathrm{err}}$）降低 **21%**。对于高容量 SOTA 模型（如 Video Swin Transformer），该框架在保持极高精度（Rel-L2 < 1%）的同时，完美恢复了波纹的高频物理细节。

**结论与贡献**：本文不仅在算法层面提升了稀疏观测重建的精度，更在方法论层面建立了一套可复现、可审计的评测协议。通过严格的显著性检验（Paired t-test）与资源成本分析（FLOPs/显存/延迟），验证了所提框架在工程落地的可行性。本文的研究为 AI4Science 领域中“从模拟到实验”的闭环验证提供了具有参考价值的标准范式。

**关键词**：时空场重建；稀疏观测；评测口径一致性；神经算子；序列化训练；科学机器学习

---

**Keywords**: Sparse Observation; Spatiotemporal Field Reconstruction; Physics-Informed Machine Learning; Observation Consistency; Spectral Loss

**Abstract**:
**Background**: Reconstructing high-resolution spatiotemporal physical fields from sparse observations is a fundamental problem in experimental fluid dynamics and environmental monitoring. Existing data-driven methods often overlook the "consistency gap" between the ideal degradation operators used during training and the complex observation operators in real-world scenarios (e.g., anti-aliasing filtering, irregular boundary cropping), leading to poor generalization in practical deployments.
**Methods**: To address this, we propose a "Consistency-First" reconstruction framework. First, we define a **Unified Observation Operator ($H$)** to ensure strict alignment between the training degradation process ($DC$) and the evaluation observation process ($H$), eliminating the "operator aliasing" effect. Second, we design a **Sequential Spatiotemporal Training Strategy**, which decouples spatial structural recovery from temporal dynamic evolution, progressively optimizing the model. Third, we introduce a **Triple Loss Function** incorporating reconstruction loss, low-frequency spectral consistency loss, and observation consistency loss, explicitly constraining the model to satisfy physical observation data.
**Results**: Extensive experiments on the PDEBench benchmark (covering Navier-Stokes, Shallow Water equations, etc.) demonstrate that the proposed improved Swin-UNet architecture and sequential training strategy achieve a 15%~25% reduction in relative $L_2$ error compared to baselines like FNO and U-Net under extreme sparsity (e.g., 16x downsampling). Crucially, the evaluation consistency error $H_{\mathrm{err}}$ decreases synchronously with the reconstruction error. Furthermore, the introduction of low-frequency spectral constraints significantly enhances the robustness of large-scale structure recovery.
