## 摘 要

在计算物理、环境监测与工业数字孪生等前沿领域，基于稀疏传感器观测重建高分辨率时空物理场是连接物理世界与数字模型的关键环节。然而，受限于部署成本、通信带宽及环境约束，实际观测数据常呈现极度稀疏（覆盖率 $< 5\%$）、非均匀采样与强噪声干扰等退化特征。此外，真实物理观测过程涉及抗混叠滤波、积分效应与边界裁剪等复杂机制，而现有深度学习方法多基于理想化退化假设训练。这种训练与评测之间的**观测算子错配（Operator Mismatch）**，导致模型在真实稀疏场景下泛化性能显著下降，且实验结论难以在不同物理工况间复现。因此，在稀疏观测条件下实现高分辨率且满足物理一致性的时空场重建，具有重要的工程意义与理论价值。

针对上述挑战，本文提出一种“评测口径一致性优先”的时空物理场重建框架（Consistency-First Reconstruction Framework）。**主要创新与工作如下**：

第一，构建了物理一致的**统一观测算子（Unified Observation Operator, $H$）**，并将训练阶段的退化算子 $DC$ 显式约束为 $DC \equiv H$。该算子集成了抗混叠高斯预滤、非均匀采样与边界对齐规则，有效规避了由算子近似引入的隐性偏差，确保了训练与评测的一致性。

第二，针对稀疏数据下端到端优化困难的问题，提出**序列化时空课程学习策略（Sequential Spatiotemporal Curriculum）**。将复杂重建任务解耦为“空间结构重构 $\to$ 时序演化预测 $\to$ 时空联合微调”三个渐进阶段，有效解决了极度欠定条件下直接训练导致的收敛效率低下与局部极值问题。

第三，设计了包含空间重建损失、低频加权谱一致性损失（Spectral Consistency Loss）与原值域观测一致性损失的**三元混合损失函数**。该函数在保证数据保真度的同时，强化了模型对物理场低频主模态与守恒量的捕捉能力。

在国际标准基准 **PDEBench** 的浅水波方程（SWE）与反应扩散方程（DRD）子集上的实验表明：(1) **精度提升显著**：在 SWE 全域重建任务中，本文方法相比轻量级基线（ResNetLite）将 PSNR 从 $46.52\,\mathrm{dB}$ 提升至 $71.05\,\mathrm{dB}$，且参数量仅为对比大模型的 $1/10$；(2) **稀疏鲁棒性强**：在 $16\times16$ 极度稀疏观测（全域占比 $1.56\%$）的 DRD 任务中，本文框架将相对误差 $\mathrm{Rel}\text{-}L_2$ 稳定在 $0.1787$ 水平，有效避免了模型崩塌；(3) **工程可行性高**：序列化学习策略将训练收敛速度提升了 **2.3 倍**，且推理延迟与显存占用满足边缘计算设备的部署需求。

本文研究证实，通过严格约束观测口径一致性并结合序列化物理先验，深度学习模型能够在极度稀疏观测下实现高保真的物理场重建，为构建低成本、高精度的工业监测系统提供了新的理论视角与技术路径。

**关键词**：时空场重建；稀疏观测；观测算子一致性；科学机器学习；序列化训练；Transformer

---

## ABSTRACT

In computational physics, environmental monitoring, and industrial digital twins, reconstructing high-resolution spatiotemporal fields from sparse sensor observations is a critical link between the physical world and digital models. However, constrained by deployment costs, communication bandwidth, and environmental complexities, practical observations are often characterized by extreme sparsity (coverage $< 5\%$), non-uniform sampling, and significant noise. Crucially, real-world observation processes involve complex physical degradations such as anti-aliasing filtering, integration effects, and boundary cropping. In contrast, existing deep learning methods often rely on idealized or simplified degradation assumptions during training. This **"Operator Mismatch"** between training and evaluation leads to poor generalization in real-world sparse scenarios and hinders the reproducibility of scientific conclusions.

To address these challenges, this thesis proposes a **Consistency-First Spatiotemporal Field Reconstruction Framework**. The core innovations and contributions are as follows:

**First**, a **Unified Observation Operator ($H$)** is constructed, with the training-time degradation operator $DC$ strictly constrained as $DC \equiv H$. This operator integrates anti-aliasing pre-filtering, non-uniform sampling, and boundary alignment rules, fundamentally eliminating implicit biases introduced by operator approximations.

**Second**, to overcome optimization difficulties under sparse data, a **Sequential Spatiotemporal Curriculum Learning** strategy is proposed. The complex reconstruction task is decoupled into three progressive stages: "Spatial Structure Reconstruction $\to$ Temporal Evolution Prediction $\to$ Joint Spatiotemporal Fine-tuning." This approach effectively circumvents the risk of convergence failure or model collapse often encountered when directly training on severely ill-posed problems.

**Third**, a **Tri-Component Hybrid Loss** is designed, incorporating spatial reconstruction loss, low-frequency weighted spectral consistency loss, and observation-domain consistency loss. This objective function enforces both data fidelity and the preservation of physical conservation laws and dominant low-frequency modes.

Extensive experiments on the **PDEBench** Shallow Water Equation (SWE) and Diffusion-Reaction (DRD) subsets demonstrate:
1.  **Accuracy Breakthrough**: On the SWE full-field reconstruction task, the proposed method increases PSNR from $46.52\,\mathrm{dB}$ (ResNetLite baseline) to $71.05\,\mathrm{dB}$, achieving a **$24.53\,\mathrm{dB}$ gain** with only $1/10$ the parameters of comparable large models.
2.  **Sparse Robustness Verification**: In DRD spatiotemporal prediction under extremely sparse observations ($16\times16$ window, covering only $1.56\%$ of the domain), the framework prevents the model collapse observed in baselines, stabilizing the relative error ($\mathrm{Rel}\text{-}L_2$) at $0.1787$. Compared to end-to-end joint training strategies, the sequential learning approach accelerates training convergence by **2.3 times** while maintaining comparable accuracy, significantly improving engineering feasibility.
3.  **Engineering Feasibility**: Resource analysis confirms that the proposed lightweight Transformer variants achieve SOTA accuracy while maintaining inference latency and memory usage demonstrating potential for deployment on edge computing devices.

This study demonstrates that by strictly enforcing the observation operator consistency and incorporating sequential physical priors, deep learning models can achieve high-fidelity physical field reconstruction even under extremely sparse observations, offering new theoretical perspectives and technical pathways for low-cost, high-precision industrial monitoring systems.

**Keywords**: Spatiotemporal Field Reconstruction; Sparse Observation; Observation Operator Consistency; Scientific Machine Learning (SciML); Sequential Training; Transformer
