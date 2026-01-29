# 第3章 方法论

## 3.1 问题定义与数学建模

### 3.1.1 离散时空场与学习目标

设物理过程的空间定义域为 $\Omega\subset\mathbb{R}^2$，离散化后的网格空间为 $\Omega_h$，其分辨率为 $N_x\times N_y$。时间维度为离散序列，索引 $t\in\{1,\dots,T\}$。目标物理场通常为标量或多通道张量场，记为：
$$
u_t:\Omega_h\rightarrow \mathbb{R}^{C},\qquad u_{1:T}=\{u_t\}_{t=1}^{T}.
$$
观测数据由观测算子 $H$ 与加性噪声项 $n_t$ 生成：
$$
y_t = H(u_t)+n_t,\qquad y_{1:T}=\{y_t\}_{t=1}^{T}.
$$
本研究的学习目标是构建一个参数化映射 $\Phi_{\omega}$（其中 $\omega$ 为可学习参数），利用观测序列及相关辅助信息恢复全时空高分辨率场：
$$
\tilde{u}_{1:T}=\Phi_{\omega}\big(y_{1:T},\, m_{1:T},\, p\big),
$$
其中 $m_t$ 为观测掩码（指示观测位置或缺失区域），$p$ 为显式坐标编码（如 Fourier 特征编码）。

### 3.1.2 评价指标的双重性：重建域与观测域

为克服“训练指标改善但评测口径误差未降”的断裂现象，本研究提出并采用双重误差评价体系：
1.  **重建域误差（Reconstruction Error）**：衡量预测场 $\tilde{u}$ 对真实场 $u$ 的逼近程度，通常采用相对 $L_2$ 范数：
    $$
    \mathrm{Rel\text{-}L2}=\frac{\lVert \tilde{u}-u\rVert_2}{\lVert u\rVert_2}.
    $$
    该指标反映了模型在数学意义上对真值的还原能力。

2.  **观测口径误差（Observation Consistency Error, $H_{\mathrm{err}}$）**：衡量预测场经观测算子 $H$ 作用后与原始观测 $y$ 的一致性：
    $$
    H_{\mathrm{err}} \triangleq \big\| H(\tilde{u})-y \big\|_2,
    $$
    其中 $\tilde{u}$ 为反标准化后的预测场。该指标反映了模型输出在观测意义上是否符合真实的物理测量口径。只有当二者同步下降时，模型的改进才具有实际的工程部署价值。

## 3.2 统一观测算子 $H$ 的构建与规范

观测算子 $H$ 不仅决定了数据的生成方式，更是模型评测的基准。因此，本研究确立 $H$ 为**唯一口径入口**，要求数据生成、训练退化、一致性损失计算及测试评测均基于同一 $H$ 实现。

### 3.2.1 超分辨（SR）观测口径

针对超分辨或降采样观测任务，遵循抗混叠（Anti-aliasing）原则，采用“先低通滤波、再降采样”的工程流程。SR 观测算子形式化为：
$$
y^{\mathrm{SR}}_t = D_s\!\left(G_{\sigma_{\mathrm{blur}}}\ast u_t\right)+n_t,
$$
其中：
-   $G_{\sigma_{\mathrm{blur}}}$ 为高斯低通滤波器，$\sigma_{\mathrm{blur}}$ 为模糊尺度；
-   $D_s$ 为降采样算子，采样倍率为 $s$。
为确保可复现性，本研究在实现中固定采用 `INTER_AREA` 插值算法进行降采样，并显式声明边界处理策略（如 `reflect`）与坐标对齐规则。

### 3.2.2 裁剪（Crop）观测口径

针对局部观测任务，采用中心对齐的裁剪策略：
$$
y^{\mathrm{Crop}}_t = C_{h_c,w_c}(u_t)+n_t,
$$
其中 $C_{h_c,w_c}$ 为裁剪算子，输出窗口大小为 $h_c \times w_c$。
为避免对齐偏差，本研究强制约束裁剪窗口尺寸为网络 Patch 尺寸的整数倍，并严格定义中心点与像素网格的对应关系。同时，掩码 $m_t$ 必须与裁剪操作同步更新，以保证输入与标签在几何口径上的一致性。

## 3.3 训练退化算子 $DC$ 的同源复用机制

### 3.3.1 硬约束定义

训练阶段的退化算子 $DC$ 用于合成训练输入及计算一致性损失。为消除“训练端自造口径”带来的隐性域偏差，本研究引入硬性约束：
$$
DC \equiv H \quad \text{（同一实现、同一参数、同一边界与对齐策略）}.
$$
该约束确保了训练过程中的退化模型与测试阶段的观测模型在数学与工程实现上完全等价。

### 3.3.2 阻断式等价性审计

为保证口径一致性的严格执行，本研究在实验流程中引入了阻断式审计机制。在训练启动前，随机抽取 $N$ 个样本（$N \ge 100$），验证以下等价性条件：
$$
\mathrm{MSE}\big(H(u^{(i)}),\,DC(u^{(i)})\big) < \varepsilon,
$$
其中 $\varepsilon$ 为数值容差（默认取 $10^{-8}$）。若验证失败，实验将自动终止，并输出差异诊断报告。这一机制从源头上杜绝了因口径不一致导致的实验偏差。

## 3.4 模型架构与统一接口

### 3.4.1 统一输入接口

为保证不同模型架构的可比性，本研究定义了统一的输入张量构造方式。单帧输入 $x_t$ 由以下分量按通道拼接而成：
$$
x_t=\mathrm{Concat}\big(\mathrm{baseline}(y_t),\,m_t,\,\mathrm{coords},\,\mathrm{PE}_{\mathrm{Fourier}}\big).
$$
各分量定义如下：
-   `baseline`：基础重建结果（如双线性插值），提供初始解；
-   $m_t$：观测掩码，指示观测数据的有效区域；
-   `coords`：归一化坐标网格 $(x,y)\in[0,1]^2$；
-   $\mathrm{PE}_{\mathrm{Fourier}}$：Fourier 特征编码，旨在提升网络对高频细节的感知能力。

### 3.4.2 序列化时空训练策略

针对时空耦合模型端到端训练收敛困难的问题，本研究提出了**三阶段序列化训练策略（Sequential Training Strategy）**，将空间重建与时序演化任务解耦：

1.  **阶段一：空间预训练（Spatial Pretraining）**
    冻结时序模块，仅优化空间编码器与解码器。训练目标聚焦于单帧空间重建，利用 $L_{\mathrm{rec}} + L_{\mathrm{spec}} + L_{\mathrm{dc}}$ 损失函数，确保模型优先具备从稀疏观测恢复高频细节的能力。

2.  **阶段二：时序预训练（Temporal Pretraining）**
    冻结已训练的空间模块，仅优化时序演化模块（如 LSTM 或 Transformer）。采用 Teacher Forcing 策略，输入真实的历史特征，迫使模型学习潜在空间的动力学演化规律。

3.  **阶段三：联合微调（Joint Fine-tuning）**
    解冻所有参数，进行端到端的自回归滚动预测（Autoregressive Rollout）。引入 **Teacher Forcing Decay** 机制，随训练进程逐步减少真值引导，平滑过渡到完全自回归模式，以缓解 Exposure Bias 问题并提升长时预测的稳定性。

### 3.4.3 空间重建模型分类

本研究涵盖了多种具有代表性的空间重建模型，主要可分为以下四类：

1.  **基于 CNN 的重建模型（CNN-based Reconstruction Models）**
    此类模型以卷积神经网络为基础，擅长提取局部特征并利用平移不变性。典型代表包括经典的 **U-Net** 及其变体（如 UNet++），以及在超分辨率领域表现优异的 **EDSR**、**RCAN** 和 **RDN**。这些模型通过堆叠卷积层、残差块或密集连接块，有效地从稀疏观测中恢复高频细节。

2.  **基于 Transformer 的模型（Transformer-based Models）**
    利用自注意力机制（Self-Attention）捕获全局长程依赖。代表性模型包括 **Vision Transformer (ViT)** 及其层次化变体 **Swin Transformer**。针对密集预测任务优化的 **Swin-UNet** 和 **U-NetFormer** 结合了 U-Net 的多尺度结构与 Transformer 的全局建模能力。此外，**Restormer** 和 **SegFormer** 等架构也在特定任务中展现了强大的特征表达能力。

3.  **算子学习模型（Operator Learning Models）**
    此类模型旨在学习函数空间之间的映射，具有分辨率无关（Resolution-invariant）的特性。**Fourier Neural Operator (FNO)** 通过在频域进行全局卷积来求解偏微分方程；**DeepONet** 利用分支网络（Branch Net）和主干网络（Trunk Net）的内积来逼近算子。**U-FNO** 则尝试结合 U-Net 的多尺度编码与 FNO 的频谱处理能力。

4.  **隐式神经表示与 MLP 模型（Implicit Neural Representations / Others）**
    **LIIF (Local Implicit Image Function)** 通过学习连续的图像表示来实现任意分辨率的重建。**MLP-Mixer** 则完全摒弃了卷积和注意力机制，仅通过多层感知机（MLP）在空间和通道维度上进行混合，展示了纯 MLP 架构的潜力。

### 3.4.4 时序演化模型

为了处理动态物理场的时空预测任务，本研究引入了时序演化模块，重点关注以下两种主流架构：

1.  **ConvLSTM (Convolutional LSTM)**
    ConvLSTM 是将卷积操作引入 LSTM 单元的经典架构。它在输入到状态、状态到状态的转换中均采用卷积运算，从而在提取时序依赖的同时保留了空间结构信息。这使其特别适合于降水临近预报等时空序列预测任务，能够有效地捕捉物理场的局部动态变化。

2.  **Video Swin Transformer (VideoSwin)**
    Video Swin Transformer 是 Swin Transformer 在视频领域的扩展。它将 2D 移位窗口机制（Shifted Window）扩展为 3D 时空窗口，仅在此时空窗口内计算自注意力。这种设计不仅大大降低了计算复杂度，还实现了时空特征的联合建模。VideoSwin 通过层级化的结构，能够有效地捕获不同尺度的时空演化规律，尤其在处理长程时空依赖方面表现出色。

## 3.5 三元损失函数设计

为实现物理一致性与观测一致性的协同优化，本研究设计了包含三部分的复合损失函数：
$$
\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{spec}}\mathcal{L}_{\mathrm{spec}} + \lambda_{\mathrm{dc}}\mathcal{L}_{\mathrm{dc}}.
$$

### 3.5.1 重建损失 $\mathcal{L}_{\mathrm{rec}}$

在标准化（z-score）域计算，直接约束预测值对真值的逼近程度：
$$
\mathcal{L}_{\mathrm{rec}}=\left\|\hat{u}^{(z)}-u^{(z)}\right\|_2^2.
$$

### 3.5.2 低频谱一致性损失 $\mathcal{L}_{\mathrm{spec}}$

针对大尺度物理结构的稳定性，在频域引入低频约束。对二维傅里叶变换后的低频系数集合 $\mathcal{K}_{\mathrm{low}}$ 计算误差：
$$
\mathcal{L}_{\mathrm{spec}}=
\sum_{k_x,k_y\in\mathcal{K}_{\mathrm{low}}}
\left|\mathcal{F}(\hat{u}^{(z)})_{k_x,k_y}
-\mathcal{F}(u^{(z)})_{k_x,k_y}\right|^2.
$$
该项旨在缓解深度网络常见的频谱偏置问题，防止大尺度结构的漂移。

### 3.5.3 原值域观测一致性损失 $\mathcal{L}_{\mathrm{dc}}$

在反标准化后的原值域计算，显式约束预测结果符合观测口径：
$$
\mathcal{L}_{\mathrm{dc}}=\left\|H(\tilde{u})-y\right\|_2^2.
$$
该项将评测口径误差 $H_{\mathrm{err}}$ 内生化为训练目标，从机制上保证了模型输出的物理可解释性与观测一致性。

### 3.5.4 时序一致性正则化

针对长时预测任务，引入导数一致性（$\mathcal{L}_{\mathrm{deriv}}$）与能量一致性（$\mathcal{L}_{\mathrm{energy}}$）作为辅助正则项，分别约束时序变化率与系统总能量的演化轨迹，以抑制误差累积与非物理耗散。

## 3.6 本章小结

本章构建了一套以“评测口径一致性”为核心的方法论框架。通过确立 $H$ 与 $DC$ 的同源复用机制，消除了训练与评测之间的语义断裂；通过统一的输入接口与序列化训练策略，解决了时空耦合优化的稳定性难题；通过三元损失函数的设计，实现了数学精度、物理结构与观测一致性的多维约束。这一框架为后续的实验验证与理论分析奠定了坚实的基础。
