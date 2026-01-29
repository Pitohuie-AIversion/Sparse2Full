# 摘要

**背景与意义**：在计算物理、环境监测与工业诊断等领域，高分辨率时空物理场的重建对于下游的预测、控制与决策具有重要意义。然而，受限于传感器部署成本、数据采集带宽及复杂环境约束，实际工程场景中的观测数据普遍呈现出稀疏性、非均匀性及噪声干扰等退化特征。尽管近年来以物理约束神经网络（PINN）和神经算子（Neural Operator）为代表的科学机器学习（SciML）方法在求解偏微分方程（PDE）反问题上取得了显著进展，但在面向真实观测的重建任务中，普遍存在“训练退化模型与评测观测口径不一致”的方法论不足。这一问题导致模型虽然在理想实验条件下表现优异，但在实际部署中性能显著衰退，且实验结论在不同研究之间缺乏精确的可复现性。

**问题定义**：本文聚焦的核心科学问题是：在观测稀疏（如全域覆盖率低于 20%）且伴随混叠效应与测量噪声的约束条件下，如何构建一套既能有效恢复高频物理细节，又能确保评测口径一致性的时空场重建框架。其中的关键挑战在于消除训练阶段模拟退化算子（Degradation Operator）与测试阶段真实观测算子（Observation Operator）之间的“语义断裂”，并解决时空联合优化过程中的收敛不稳定性与频谱偏差问题。

**研究方法**：针对上述挑战，本文提出了一种坚持“评测口径一致性优先”原则的稀疏观测时空场重建方法论。首先，构建了**统一观测算子（Unified Observation Operator, $H$）**框架，强制训练端的退化算子（$DC$）在插值核、抗混叠预滤、边界处理策略与对齐规则上与数据侧的观测算子（$H$）保持严格的一致性，从根本上消除了隐性域偏差。其次，提出了**序列化时空训练策略（Sequential Spatiotemporal Training）**，基于课程学习（Curriculum Learning）思想，将复杂的时空重建任务分解为“空间重构预训练 $\to$ 时序演化预训练 $\to$ 时空联合微调”三个递进阶段，有效规避了端到端训练易陷入局部极小值的难题。最后，设计了包含重建损失、低频谱一致性损失与原值域观测一致性损失的**三元损失函数**，在保证数学逼近精度的同时，显式约束模型输出符合观测口径的物理一致性。

**实验结果**：基于 PDEBench 基准数据集（涵盖 Navier-Stokes、浅水方程等典型流体动力学场景）的系统性实验表明，本文提出的物理驱动稀疏重建框架具备显著的通用性与鲁棒性。消融实验揭示，高质量的空间重建是实现准确时序预测的先决条件：相比于低分辨率输入，采用高保真重建使得时序预测的相对误差（Rel-L2）降低了一个数量级（从 11.67% 降至 0.88%），峰值信噪比（PSNR）提升超过 22 dB。在极度稀疏的裁剪观测任务中，实验表明即使观测面积仅占 **1.5%**（$16\times 16$ 窗口），引入全局注意力机制的 Transformer 模型仍能有效推断全局物理结构，性能优于局部卷积网络。此外，端到端（End-to-End）联合优化策略相比分阶段基线，进一步将高频频谱误差（fRMSE-High）降低了 **14.6%**，有效捕捉了激波等关键物理微结构。

**结论与贡献**：本文不仅在算法层面显著提升了稀疏观测重建的精度，更在方法论层面建立了一套可复现、可审计的标准化评测协议。通过多维度的定量评估与资源成本分析（FLOPs/显存/延迟），验证了所提框架在工程落地的可行性与有效性。本研究为 AI4Science 领域中“从模拟到实验”的闭环验证提供了具有重要参考价值的标准范式。

**关键词**：时空场重建；稀疏观测；评测口径一致性；神经算子；序列化训练；科学机器学习

---

**Keywords**: Spatiotemporal Field Reconstruction; Sparse Observation; Evaluation Consistency; Neural Operator; Sequential Training; Scientific Machine Learning

**Abstract**:
**Background**: Reconstructing high-resolution spatiotemporal physical fields from sparse observations is a fundamental problem in computational physics and environmental monitoring. Existing data-driven methods often overlook the "consistency discrepancy" between the ideal degradation operators used during training and the complex observation operators in real-world scenarios (e.g., anti-aliasing filtering, irregular boundary cropping), leading to poor generalization in practical deployments.

**Methods**: To address this, we propose a "Consistency-First" reconstruction framework. First, we define a **Unified Observation Operator ($H$)** to ensure strict alignment between the training degradation process ($DC$) and the evaluation observation process ($H$), eliminating the "operator discrepancy". Second, we design a **Sequential Spatiotemporal Training Strategy**, which decouples spatial structural recovery from temporal dynamic evolution, progressively optimizing the model through curriculum learning. Third, we introduce a **Tri-component Loss Function** incorporating reconstruction loss, low-frequency spectral consistency loss, and observation consistency loss, explicitly constraining the model to satisfy physical observation constraints.

**Results**: Extensive experiments on the PDEBench benchmark (specifically covering 2D Diffusion-Reaction, Darcy Flow, etc.) demonstrate that the proposed framework achieves significant robustness. Compared to low-resolution baselines, our method reduces the relative $L_2$ error by an order of magnitude (from 11.67% to 0.88%) in sequential prediction tasks and improves the Peak Signal-to-Noise Ratio (PSNR) by over 22 dB. Even under extreme sparsity (e.g., 1.5% observation ratio), the framework effectively recovers global physical structures. Furthermore, the End-to-End joint optimization strategy reduces the high-frequency spectral error (fRMSE-High) by **14.6%**, effectively capturing critical micro-structures like shock waves.

**Conclusion**: This study establishes a reproducible and auditable evaluation protocol for sparse reconstruction. By enforcing evaluation consistency, we bridge the gap between simulation-based training and real-world deployment, providing a standard paradigm for AI4Science research.
