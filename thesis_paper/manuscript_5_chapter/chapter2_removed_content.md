# 从第2章移出的内容 (待整合至第3章)

以下内容原属于 `chapter2_problem_framework.md`，但根据写作大纲 `writing_checklist.md`，它们应属于 **第3章 算法设计与实现 (Methodology)** 或 **第4章 实验 (Experiments)**。为防止内容丢失，暂存如下：

## 1. 模型架构与统一接口 (原 2.4)

### 统一输入接口 (原 2.4.1)

为保证不同模型架构的可比性，本研究定义了统一的输入张量构造方式。单帧输入 $x_t$ 由以下分量按通道拼接而成：
$$
x_t=\mathrm{Concat}\big(\mathrm{baseline}(y_t),\,m_t,\,\mathrm{coords},\,\mathrm{PE}_{\mathrm{Fourier}}\big).
$$
各分量定义如下：
- `baseline`：基础重建结果（如双线性插值/最近邻填充等），提供初始解；在稀疏掩码输入下，`baseline` 需与 $m_t$ 同步使用以避免引入几何偏差；
- $m_t$：观测掩码，指示观测数据的有效区域；
- `coords`：归一化坐标网格 $(x,y)\in[0,1]^2$；
- $\mathrm{PE}_{\mathrm{Fourier}}$：Fourier 特征编码，旨在提升网络对高频细节的感知能力（通常为 $\mathrm{PE}_{\mathrm{Fourier}}=\gamma(\mathrm{coords})$）。

### 序列化时空训练策略 (原 2.4.2)

针对时空耦合模型端到端训练收敛困难的问题，本研究提出**三阶段序列化训练策略（Sequential Training Strategy）**，将空间重建与时序演化任务解耦：

1. **阶段一：空间预训练（Spatial Pretraining）**  
   冻结时序模块，仅优化空间编码器与解码器。训练目标聚焦于单帧空间重建，使用 $\mathcal{L}_{\mathrm{rec}}+\lambda_{\mathrm{spec}}\mathcal{L}_{\mathrm{spec}}+\lambda_{\mathrm{dc}}\mathcal{L}_{\mathrm{dc}}$ 作为目标函数，确保模型优先具备从稀疏观测恢复高频细节的能力。

2. **阶段二：时序预训练（Temporal Pretraining）**  
   冻结已训练的空间模块，仅优化时序演化模块（如 ConvLSTM 或 Transformer）。采用 Teacher Forcing 策略，输入真实历史特征，迫使模型学习潜在空间的动力学演化规律。

3. **阶段三：联合微调（Joint Fine-tuning）**  
   解冻所有参数，进行端到端的自回归滚动预测（Autoregressive Rollout）。引入 **Teacher Forcing Decay** 机制，随训练进程逐步减少真值引导，平滑过渡到完全自回归模式，以缓解 Exposure Bias 并提升长时预测的稳定性。

### 空间重建模型分类 (原 2.4.3)

本研究涵盖多种具有代表性的空间重建模型，主要可分为以下四类：

1. **基于 CNN 的重建模型（CNN-based Reconstruction Models）**  
   此类模型以卷积神经网络为基础，擅长提取局部特征并利用平移不变性。典型代表包括经典的 **U-Net** 及其变体（如 UNet++），以及在超分辨率领域表现优异的 **EDSR**、**RCAN** 和 **RDN**。这些模型通过堆叠卷积层、残差块或密集连接块，从稀疏观测中恢复高频细节。

2. **基于 Transformer 的模型（Transformer-based Models）**  
   利用自注意力机制（Self-Attention）捕获全局长程依赖。代表性模型包括 **Vision Transformer (ViT)** 及其层次化变体 **Swin Transformer**。针对密集预测任务优化的 **Swin-UNet** 和 **U-NetFormer** 结合了 U-Net 的多尺度结构与 Transformer 的全局建模能力。此外，**Restormer** 和 **SegFormer** 等架构也在特定任务中展现了强特征表达能力。

3. **算子学习模型（Operator Learning Models）**  
   此类模型旨在学习函数空间之间的映射，具有一定的分辨率无关（Resolution-invariant）特性。**Fourier Neural Operator (FNO)** 通过在频域进行全局卷积来逼近 PDE 解算子；**DeepONet** 利用分支网络（Branch Net）和主干网络（Trunk Net）的内积形式逼近算子。**U-FNO** 则尝试结合 U-Net 的多尺度编码与 FNO 的频谱处理能力。

4. **隐式神经表示与 MLP 模型（Implicit Neural Representations / Others）**  
   **LIIF (Local Implicit Image Function)** 通过学习连续表示以支持任意分辨率重建。**MLP-Mixer** 则完全摒弃卷积与注意力机制，仅通过多层感知机（MLP）在空间与通道维度上进行混合，展示了纯 MLP 架构的潜力。

### 时序演化模型 (原 2.4.4)

为处理动态物理场的时空预测任务，本研究引入时序演化模块，重点关注以下两种主流架构：

1. **ConvLSTM (Convolutional LSTM)**  
   ConvLSTM 将卷积操作引入 LSTM 单元，在输入到状态、状态到状态的转换中均采用卷积运算，从而在提取时序依赖的同时保留空间结构信息。其适用于多类时空序列预测任务，能够捕捉物理场的局部动态变化。

2. **Video Swin Transformer (VideoSwin)**  
   Video Swin Transformer 将 Swin Transformer 的移位窗口机制扩展为 3D 时空窗口，仅在局部时空窗口内计算自注意力，从而降低复杂度并实现时空特征的联合建模。其层级结构有助于捕获不同尺度的时空演化规律，尤其适用于长程依赖建模。

## 2. 三元损失函数设计 (原 2.5)

为实现物理一致性与观测一致性的协同优化，本研究设计包含三部分的复合损失函数：
$$
\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{spec}}\mathcal{L}_{\mathrm{spec}} + \lambda_{\mathrm{dc}}\mathcal{L}_{\mathrm{dc}}.
$$

### 重建损失 $\mathcal{L}_{\mathrm{rec}}$ (原 2.5.1)

在标准化（z-score）域计算，直接约束预测值对真值的逼近程度：
$$
\mathcal{L}_{\mathrm{rec}}=\left\|\hat{u}^{(z)}-u^{(z)}\right\|_2^2.
$$

### 低频谱一致性损失 $\mathcal{L}_{\mathrm{spec}}$ (原 2.5.2)

针对大尺度物理结构的稳定性，在频域引入低频约束。对二维傅里叶变换后的低频系数集合 $\mathcal{K}_{\mathrm{low}}$ 计算误差：
$$
\mathcal{L}_{\mathrm{spec}}=
\sum_{k_x,k_y\in\mathcal{K}_{\mathrm{low}}}
\left|\mathcal{F}(\hat{u}^{(z)})_{k_x,k_y}
-\mathcal{F}(u^{(z)})_{k_x,k_y}\right|^2.
$$
该项旨在缓解深度网络常见的频谱偏置问题，防止大尺度结构漂移；在实现中通常对每一帧/每一通道分别计算 FFT，再在时间与通道维度做求和或取均值。

### 原值域观测一致性损失 $\mathcal{L}_{\mathrm{dc}}$ (原 2.5.3)

在反标准化后的原值域计算，显式约束预测结果符合观测口径：
$$
\mathcal{L}_{\mathrm{dc}}=\left\|H(\tilde{u})-y\right\|_2^2.
$$
该项将评测口径误差 $H_{\mathrm{err}}$ 内生化为训练目标，从机制上保证模型输出的物理可解释性与观测一致性。

## 3. 观测算子 $H$ 的物理建模与实现 (原 1.3.1 详细版)

### 抗混叠观测口径

真实观测算子 $H$ 需显式建模抗混叠预滤与降采样过程。其形式化定义为：
$$
\mathbf{y} = H(\mathbf{U}) = D_{\downarrow s}\big(G_{\sigma} * \mathbf{U}\big) + \boldsymbol{\varepsilon}.
$$
其中：
- $G_{\sigma}$ 为高斯低通滤波器，核大小通常设为 $k=5$ 或 $k=7$，$\sigma$ 根据降采样因子 $s$ 确定（通常 $\sigma \approx s/2$），以抑制高于奈奎斯特频率的高频成分。
- $D_{\downarrow s}$ 为降采样算子，工程上采用面积插值（Area Interpolation, 对应 OpenCV `INTER_AREA`）以模拟传感器的空间积分效应。
- $\boldsymbol{\varepsilon}$ 为加性噪声项。

### 边界与对齐策略

为避免边界伪影（Boundary Artifacts）沿时间传播，必须明确边界处理策略：
- **Padding**：对于非周期边界，通常采用镜像填充（Reflection Padding）或复制填充（Replication Padding）以保持边界连续性。
- **Alignment**：对于局部裁剪（Crop），窗口需与网格对齐（例如 Patch Size 的整数倍），并显式定义中心对齐或角点对齐规则，确保 $H$ 的输出与训练输入严格对应。

## 4. 评测指标定义 (原 1.7.2 详细版)

### 相对重建误差 (Rel-L2)
$$
\mathrm{Rel}\text{-}L_2=\frac{\lVert \tilde{\mathbf{U}}-\mathbf{U}\rVert_F}{\lVert \mathbf{U}\rVert_F}.
$$
这是衡量全场逼近精度的核心指标。

### 口径一致性误差 (H_err)
$$
H_{\mathrm{err}}=\lVert H(\tilde{\mathbf{U}})-\mathbf{y}\rVert_F.
$$
该指标衡量重建结果再次经过观测过程后，是否能回退到原始观测数据，直接反映了重建结果的物理可信度与口径一致性。
