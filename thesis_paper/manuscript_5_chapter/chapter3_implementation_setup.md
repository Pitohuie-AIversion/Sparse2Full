# 第3章 算法设计与实现 (Methodology)

## 3.0 引言

本章详细阐述面向稀疏观测的物理场重建算法（Consistency-First Framework）的系统设计与工程实现。针对第 2 章提出的“算子错配”与“欠定优化”挑战，本章提出了一套以**观测一致性**为核心约束的端到端重建方案。核心设计包含三个层面：
1.  **统一观测算子模块**：在物理层消除训练与评测的口径偏差；
2.  **序列化课程学习**：在时空层解耦优化难度，实现从稳态到动态的渐进式收敛；
3.  **三元混合损失**：在优化层兼顾数据保真度、物理守恒性与观测一致性。

本章将依次介绍总体框架、核心模块实现、网络架构设计、训练策略及损失函数定义，并给出关键的工程实现细节以确保研究的可复现性。

---

## 3.1 总体框架 (Consistency-First Framework)

本研究提出的“观测一致性优先（Consistency-First）”框架是一个端到端的深度学习重建系统。其核心思想是将物理观测过程显式嵌入到训练回路中，强制模型输出在观测算子 $H$ 的作用下回归到输入观测 $y$，从而构成闭环约束。

### 3.1.1 端到端流程

系统的数据流与处理逻辑如图 3-1 所示（见下文描述），主要包含以下四个阶段：

![图 3-1: “评测口径一致性优先”的时空场重建网络总体架构。模型接收稀疏观测、几何掩码与时空坐标作为输入，通过空间编码器与时序演化模块重建高分辨率物理场，并引入观测一致性闭环约束（Consistency Loop）以消除算子错配带来的系统性偏差。](images/fig3-1_framework.png)

1.  **输入构造 (Input Construction)**：
    将稀疏观测 $y$、观测掩码 $m$（指示数据缺失位置）、归一化时空坐标 $\mathbf{x}_{\text{grid}}$ 以及可选的傅里叶特征编码 $\text{PE}$ 进行通道级拼接，形成高维输入张量 $X_{\text{in}}$。
    $$ X_{\text{in}} = \text{Concat}(y', m, \mathbf{x}_{\text{grid}}, \text{PE}) $$
    其中 $y'$ 为经过基础插值（如双线性插值）初始化的粗糙场，为模型提供低频基准。

2.  **特征编码与演化 (Encoding & Evolution)**：
    利用编码器提取多尺度空间特征，并通过时序模块（如 VideoSwin 或 ConvLSTM）在潜在空间（Latent Space）模拟物理场的动力学演化过程，捕捉时空相关性。

3.  **解码与重建 (Decoding)**：
    解码器将演化后的特征映射回物理空间，输出高分辨率的重建场 $\hat{u}$。

4.  **一致性校验 (Consistency Check)**：
    重建场 $\hat{u}$ 经过与数据生成阶段完全一致的**退化算子** $DC$（Degradation Operator），生成重投影观测 $\hat{y}$。训练过程不仅最小化 $\hat{u}$ 与真值 $u$ 的误差，同时强制 $\hat{y} \approx y$。

---

## 3.2 统一观测算子模块 (Unified Operator Module)

为解决“算子错配”问题，本研究在工程实现上严格遵循“单一入口（Single Entry Point）”原则。即：**训练阶段的退化算子 $DC$ 与数据生成阶段的观测算子 $H$ 必须复用同一套代码实现与配置参数**。

### 3.2.1 算子实现机制

基于 PyTorch 的 `Function` 或 `nn.Module` 实现可微算子，确保梯度能够通过观测算子回传至重建网络。图 3-2 展示了两种任务（SR 与 Crop）下的统一算子生成逻辑。

![图 3-2: 统一观测算子模块与三元混合损失函数示意图。算子模块 $H$ 集成了物理降质与几何采样过程，确保训练与评测口径严格对齐；三元损失分别在空间域、频域与观测域施加多维约束，以兼顾重建精度与物理守恒性。](images/fig3-2_operator.png)

**伪代码 3-1：统一观测算子实现逻辑**

```python
class DegradationOperator(nn.Module):
    def __init__(self, task_cfg):
        super().__init__()
        # 统一配置入口，确保 H 与 DC 参数一致
        self.task = task_cfg.name
        self.params = task_cfg.params
        
        # 预构建静态核（如高斯核），避免训练中重复计算
        if self.task == 'SR':
            self.register_buffer('kernel', build_gaussian_kernel(self.params))

    def forward(self, u_hr):
        """
        Input:  u_hr [B, C, H, W] (High-Res Ground Truth or Prediction)
        Output: y_lr [B, C, h, w] (Low-Res Observation)
        """
        if self.task == 'SR':
            # 1. 抗混叠滤波 (Anti-aliasing)
            u_blur = functional.conv2d(u_hr, self.kernel, padding='reflect')
            # 2. 下采样 (Downsampling)
            # 强制使用 area 插值以模拟物理积分效应
            y_out = functional.interpolate(u_blur, scale_factor=1/self.params.scale, mode='area')
            
        elif self.task == 'Crop':
            # 1. 中心裁剪 (Center Crop)
            h, w = self.params.crop_size
            y_out = center_crop(u_hr, (h, w))
            # 2. 掩码生成 (Mask Generation)
            # 同步更新掩码，指示有效区域
            
        return y_out
```

### 3.2.2 关键配置协议

为确保实验的可比性，针对两类典型任务定义了严格的配置协议：

1.  **超分辨 (SR)**：
    *   **滤波**：使用固定尺寸（如 $k=5$）和标准差（$\sigma$）的高斯核进行预处理，模拟传感器的空间积分效应。工程上，采用 **Separable Filter**（可分离卷积）实现二维高斯核，以降低计算复杂度至 $O(N)$。
    *   **边界**：采用镜像填充（Reflection Padding）处理边界，避免零填充引入的人工高频伪影（Boundary Artifacts）。
    *   **采样**：严格使用 `INTER_AREA`（区域插值）而非 `Nearest` 或 `Bilinear`，以符合物理上的能量守恒特性，确保下采样过程中的光通量（Flux）不变。

2.  **稀疏裁剪 (Crop)**：
    *   **几何对齐**：裁剪窗口严格以图像中心为基准，尺寸 $(h_c, w_c)$ 设为网络 Patch Size 的整数倍，避免边缘对齐误差。
    *   **掩码同步**：算子同步输出二值掩码 $M$，其中 $M_{i,j}=1$ 表示观测有效，$M_{i,j}=0$ 表示缺失，确保网络能精确感知数据边界。这种设计兼容了 **Masked Image Modeling (MIM)** 的预训练范式。

---

## 3.3 网络架构设计

本研究采用模块化设计，将网络解耦为**空间特征提取**、**时序演化建模**与**物理映射解码**三个子模块。这种设计允许针对不同物理场景灵活替换骨干网络（Backbone），符合 **Model-Agnostic Meta-Learning** 的设计思想。

### 3.3.1 空间特征提取 (Spatial Encoder)

该模块负责从单帧或多帧输入中提取多尺度空间特征。为验证框架的通用性，本研究支持多种主流骨干网络。其中，**EDSR (Enhanced Deep Super-Resolution)** 因其去除 Batch Normalization 的设计更适合物理场数值回归，被选为本研究的核心空间骨干网络，其架构如图 3-4 所示。

![图 3-4: EDSR 空间特征提取网络架构图。该架构去除了 Batch Normalization 层以适应物理场数值回归任务，并利用深层残差块（ResBlocks）提取多尺度空间特征。](../figures_nn/build_export_j2/edsr/fig_edsr_auto.svg)

1.  **基于 CNN 的模型**：
    以卷积神经网络为基础，擅长提取局部特征并利用平移不变性。
    *   **轻量化 CNN**：对于计算资源受限场景，采用类似 **EDSR** 或 **ConvUNet** 的全卷积结构，利用局部感受野快速捕捉高频纹理。为了适应物理场重建，移除了 Batch Normalization 层，以保留物理量的绝对数值分布信息。

2.  **基于 Transformer 的模型**：
    利用自注意力机制捕获全局长程依赖。
    *   **Swin Transformer**：对于强非局部相关性场（如湍流），采用 Swin Transformer Block。利用移动窗口注意力机制（Shifted Window Attention）在保持线性计算复杂度的同时建立长程依赖，有效解决了全局 Attention 的显存瓶颈。
    *   **U-NetFormer**：结合 U-Net 的多尺度结构与 Transformer 的全局建模能力，利用跳跃连接（Skip Connection）融合深层语义特征与浅层空间细节。

3.  **算子学习与隐式表示**：
    *   **Fourier Neural Operator (FNO)**：通过频域全局卷积逼近 PDE 解算子，具有分辨率无关特性。
    *   **MLP-Mixer**：摒弃卷积与注意力，仅通过 MLP 在空间与通道维度混合，探索极简架构的潜力。

### 3.3.2 时序演化建模 (Temporal Modeling)

针对时变物理场，在潜在特征空间引入时序模块，重点支持以下两种架构以应对不同动力学特性。其中，**Video Swin Transformer** 因其在长程依赖建模上的优势，被选为处理复杂湍流场景的核心时序模块，其架构如图 3-5 所示。

![图 3-5: Video Swin Transformer 时序演化模块架构图。该模块利用 3D 移位窗口注意力机制（3D Shifted Window Attention）在降低计算复杂度的同时捕捉时空长程依赖。](../figures_nn/build_export_j2/videoswin/fig_videoswin_auto.svg)

1.  **ConvLSTM (Convolutional LSTM)**：
    将卷积操作引入 LSTM 单元，在状态转换中保留空间结构信息。适用于捕捉局部动态变化，能够有效处理具有明确对流特征的物理过程。

2.  **Video Swin Transformer (VideoSwin)**：
    将 Swin Transformer 的移位窗口机制扩展为 3D 时空窗口，仅在局部时空窗口内计算自注意力。该架构能够同时降低计算复杂度并实现时空特征的联合建模，尤其适用于需要长程依赖建模的复杂湍流场景。

### 3.3.3 解码器 (Decoder)

解码器将深层特征映射回物理空间。为抑制传统转置卷积（Transposed Conv）易产生的棋盘格伪影（Checkerboard Artifacts），本文优先采用**“双线性上采样 + 卷积层”**的组合策略，确保重建结果在空间上的平滑性与物理合理性。

---

## 3.4 训练策略：序列化课程学习

物理场重建是一个典型的病态（Ill-posed）反问题。直接进行端到端训练往往面临收敛困难或陷入局部极小值。为此，本研究设计了“空间 $\to$ 时序 $\to$ 联合”的三阶段序列化课程学习（Sequential Curriculum Learning）策略，如图 3-3 所示。

![图 3-3: 序列化时空课程学习（Sequential Spatiotemporal Curriculum Learning）策略流程图。通过将复杂的时空重建任务解耦为“空间结构重构 $\to$ 动力学演化预测 $\to$ 联合微调”三个渐进阶段，有效解决了极度欠定条件下的优化收敛难题。](images/fig3-3_sequential.png)

### 阶段一：空间重构预训练 (Spatial Pretraining)
*   **目标**：让网络首先学会从稀疏观测中恢复静态的空间结构。
*   **策略**：
    *   冻结时序模块，视输入为独立的单帧样本。
    *   仅利用空间相关性进行插值与超分。
    *   **验证重点**：静态场的 SSIM 与 PSNR 指标。

### 阶段二：时序演化预训练 (Temporal Pretraining)
*   **目标**：在空间特征稳定的基础上，学习流体的动力学演化规律。
*   **策略**：
    *   冻结空间编码器与解码器，仅训练时序模块（如 LSTM 权重）。
    *   引入 **Teacher Forcing**：在训练初期输入真实的上一帧特征，引导模型捕捉正确的时间导数。
    *   **动态衰减**：Teacher Forcing Ratio 随训练进程从 1.0 线性衰减至 0.0，平滑过渡至自回归模式。

### 阶段三：时空联合微调 (Joint Fine-tuning)
*   **目标**：协同优化空间与时序模块，消除模块间的特征不对齐，实现全局最优。
*   **策略**：
    *   解冻所有参数。
    *   执行多步自回归滚动预测（Autoregressive Rollout），计算长时序累积误差。
    *   引入观测一致性损失，进行端到端的物理一致性微调。

---

## 3.5 损失函数设计

为了实现前文所述的“广义物理一致性”（Generalized Physics Consistency），本研究并未单纯依赖难以优化的高阶 PDE 残差，而是构建了一个包含**数据保真项**与**物理正则项**的混合损失函数体系。该体系通过显式约束观测算子投影（$L_{dc}$）与频域统计分布（$L_{spec}$），在数据驱动框架下实现了对物理守恒律的高效逼近：

$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{spec}} \mathcal{L}_{\text{spec}} + \lambda_{\text{dc}} \mathcal{L}_{\text{dc}} $$

### 3.5.1 重建损失 ($\mathcal{L}_{\text{rec}}$)
基础的像素级保真度约束，采用 $L_1$ 或 $L_2$ 范数：
$$ \mathcal{L}_{\text{rec}} = \frac{1}{T \cdot N} \sum_{t=1}^T || \hat{u}_t - u_t ||_2^2 $$
其中 $\hat{u}_t$ 为重建场，$u_t$ 为真实场（Ground Truth）。

### 3.5.2 谱一致性损失 ($\mathcal{L}_{\text{spec}}$)
物理场的能量主要集中在低频模态。为避免高频噪声干扰并确保大尺度结构的准确性，在频域施加约束：
$$ \mathcal{L}_{\text{spec}} = || \mathcal{F}(\hat{u}) \cdot W_{\text{low}} - \mathcal{F}(u) \cdot W_{\text{low}} ||_2^2 $$
其中 $\mathcal{F}(\cdot)$ 表示二维快速傅里叶变换 (FFT)，$W_{\text{low}}$ 为低通滤波器掩码，仅关注波数 $k \le k_{\text{cutoff}}$ 的低频分量。

### 3.5.3 观测一致性损失 ($\mathcal{L}_{\text{dc}}$)
这是本框架的核心创新点。强制重建结果在经过观测算子投影后，能够复现原始观测数据：
$$ \mathcal{L}_{\text{dc}} = || DC(\hat{u}) - y ||_2^2 $$
该损失项相当于在优化过程中引入了一个“物理护栏”，确保解空间始终受限于观测数据的约束流形内，有效防止产生非物理的虚假纹理（Hallucination）。

### 3.5.4 辅助物理约束 ($\mathcal{L}_{\text{pde}}$)
除了上述核心损失外，本框架在代码实现层支持引入 PDE 残差（如 Navier-Stokes 或 Shallow-Water 方程残差）作为物理正则化项 $\mathcal{L}_{\text{pde}}$。然而，实验研究发现，在极度稀疏观测条件下，直接优化高阶 PDE 导数项往往导致训练不稳定。相比之下，基于统一算子的观测一致性损失 $\mathcal{L}_{\text{dc}}$ 与低频加权的谱损失 $\mathcal{L}_{\text{spec}}$ 已能提供足够且鲁棒的物理约束。为避免引入额外的超参数敏感性，本研究的主实验中并未强制启用 PDE 残差项，而是将其作为可选的辅助约束，重点依靠谱一致性与数据一致性来保证物理合理性。

---

## 3.6 本章小结

本章详细阐述了稀疏观测物理场重建的算法设计与工程实现。通过构建“观测一致性优先”的总体框架，利用统一观测算子消除了训练与评测的口径偏差；通过序列化课程学习策略解决了欠定反问题的优化难题；通过三元混合损失函数有效平衡了数据精度与物理守恒性。这些设计共同构成了一个闭环、鲁棒且可复现的科学机器学习系统，为第 4 章的实验验证奠定了坚实的技术基础。
