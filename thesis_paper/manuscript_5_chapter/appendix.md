# 附录 A：候选模型性能扫描全表

本附录提供了第 4 章 4.2 节中 28 个候选模型在 Shallow Water Equation (SWE) 数据集上的完整性能扫描结果。

**实验设置说明**：
- **任务**：Super-Resolution (SR) x4 / Crop (部分)
- **数据**：PDEBench - 2D Shallow Water Equation
- **训练**：10M 参数量级约束（部分模型除外），600 Epochs (默认)，统一损失函数。
- **指标**：Rel-L2 (相对 L2 误差), PSNR (峰值信噪比), SSIM (结构相似性), Inference Latency (推理延迟, ms), FLOPs (浮点运算量, G), Params (参数量, M)。

**表 A-1 28 个模型的详细性能扫描结果**

| 模型名称 (Model) | 类别 (Class) | Params (M) | FLOPs (G) | Latency (ms) | Rel-L2 $\downarrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | fRMSE (Low) $\downarrow$ | DC Error ($H_{\mathrm{err}}$) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **edsr** | CNN (Res) | 1.22 | 19.95 | 4.05 | **0.0023** | **71.05** | **1.0000** | **0.45** | **0.0001** |
| **edsrnet** | CNN (Res) | 1.22 | 19.95 | 19.89 | **0.0023** | **71.05** | **1.0000** | **0.45** | **0.0001** |
| **nafnet** | CNN (Gate) | 8.15 | 771.14 | 16.07 | 0.0193 | 52.19 | 0.9981 | 2.53 | 0.0017 |
| **UformerLite** | Trans (Lite) | 2.00 | 32.67 | **0.99** | 0.0243 | 50.18 | 0.9983 | 3.27 | 0.0030 |
| **uno** | Operator | 28.05 | **4.24** | 4.63 | 0.0314 | 48.77 | 0.9983 | 5.31 | 0.0012 |
| **unonet** | Operator | 28.05 | **4.24** | 4.59 | 0.0382 | 46.42 | 0.9977 | 5.98 | 0.0009 |
| **resnetlite** | CNN (Lite) | 9.99 | 163.62 | 6.15 | 0.0376 | 46.52 | 0.9963 | 5.73 | 0.0064 |
| **ConvUNetLite** | CNN (Lite) | 10.09 | 165.24 | 8.63 | 0.0432 | 45.23 | 0.9963 | 5.49 | 0.0040 |
| **SwinIRLite** | Trans (Lite) | 3.99 | 65.29 | 2.24 | 0.0451 | 45.29 | 0.9959 | 6.39 | 0.0064 |
| **partialconvunet** | CNN (Partial) | 9.77 | 159.95 | 3.14 | 0.0488 | 43.82 | 0.9963 | 8.24 | 0.0022 |
| **pconvunet** | CNN (Partial) | 9.77 | 159.95 | 3.15 | 0.0484 | 43.86 | 0.9963 | 7.35 | 0.0013 |
| **UNet** | CNN (Base) | 9.89 | 161.84 | 1.18 | 0.0484 | 43.58 | 0.9932 | 10.96 | 0.0079 |
| **SwinUNet** | Trans (U) | 55.67 | 0.08 | 11.99 | 0.1605 | 33.07 | 0.9395 | 48.88 | 0.0061 |
| **swinunet** | Trans (U) | 3.52 | 0.01 | 12.00 | 0.1830 | 31.96 | 0.9175 | 57.54 | 0.0176 |
| **swin_unet** | Trans (U) | 3.52 | 0.01 | 12.23 | 0.2072 | 30.88 | 0.9021 | 67.09 | 0.0166 |
| **deeponet** | Operator | 10.14 | 154.79 | 0.94 | 0.0629 | 41.35 | 0.9907 | 17.03 | 0.0084 |
| **deeponet2d** | Operator | 10.14 | 154.79 | 0.94 | 0.0622 | 41.46 | 0.9909 | 16.42 | 0.0077 |
| **NAFNetLite** | CNN (Lite) | 0.04 | 5.39 | **0.82** | 0.0683 | 40.85 | 0.9881 | 25.79 | 0.0075 |
| **bilinear3x3decoder**| CNN (Tiny) | 0.00 | 0.00 | 2.33 | 0.0746 | 39.74 | 0.9861 | 25.66 | 0.0028 |
| **mlpmodel** | MLP | 0.01 | 0.14 | **0.35** | 0.0770 | 39.52 | 0.9853 | 23.59 | 0.0014 |
| **stablefno2d** | Operator | 10.64 | 0.34 | 5.36 | 0.0800 | 39.08 | 0.9831 | 26.96 | 0.0186 |
| **stablefnomodel** | Operator | 10.64 | 0.34 | 5.34 | 0.0800 | 39.08 | 0.9831 | 26.96 | 0.0186 |
| **fno2d** | Operator | 10.67 | 0.60 | 1.40 | 0.1335 | 34.67 | 0.9091 | 35.92 | 0.0455 |
| **FNO2d** (Older) | Operator | 10.67 | 0.60 | 2.97 | 0.1346 | 34.66 | 0.9138 | 35.72 | 0.0371 |
| **fno** (Older) | Operator | 10.65 | 0.34 | 1.41 | 0.2798 | 28.58 | 0.7109 | 61.65 | 0.0954 |
| **MLPMixer** | MLP | 9.70 | 1.62 | 2.87 | 0.1569 | 33.28 | 0.9172 | 41.38 | 0.0186 |
| **ViT** | Trans | 10.17 | 0.87 | 10.54 | 0.1987 | 31.23 | 0.8780 | 65.56 | 0.0253 |
| **segformer** | Trans | 23.21 | 88.62 | 5.78 | 0.1740 | 32.36 | 0.9333 | 62.69 | 0.0729 |
| **SegFormer** (Old)| Trans | 23.21 | 88.62 | 5.72 | 0.3231 | 27.18 | 0.7503 | 146.00 | 0.1515 |
| **RestormerLite** | Trans (Lite) | 0.05 | 2.85 | 2.64 | 0.2772 | 28.30 | 0.9092 | 108.47 | 0.0030 |
| **UNetPlusPlus** | CNN (Base) | 10.11 | 152.19 | 4.02 | 0.1893 | 32.09 | 0.9409 | 9.37 | 0.0181 |

> **注**：
> 1.  **Params**: 模型参数量 (Million)。
> 2.  **FLOPs**: 浮点运算量 (Giga-FLOPs), 基于 256x256 输入估算。
> 3.  **Latency**: 单帧推理延迟 (ms), 在 NVIDIA L40 GPU 上测得。
> 4.  **DC Error ($H_{\mathrm{err}}$)**: 观测一致性误差 $\|H(\tilde{u})-y\|_2$，越低表示物理约束越好。
> 5.  表中包含了部分模型的变体（如 `fno2d` vs `stablefno2d`），展示了架构微调对稳定性的影响。

## 附录 B：实验配置快照 (Experiment Configuration)

（此处保持原有的 YAML 配置内容不变，省略以节省篇幅）

## 附录 C：完整模型架构图集 (Full Model Architecture Gallery)

本附录详细收录了本项目涉及的 36 个深度学习模型的架构图及技术解析。所有模型均已在 `models/` 目录下实现，并按照轻量化设计、Transformer 变体、神经算子（Neural Operators）及经典基线进行分类管理。

### 图例 (Legend)
以下为所有模型架构图中使用的统一图例说明，涵盖了 2D 空间操作与 3D 时空操作的核心图元。

#### 2D 网络模块图例
![Legend 2D](../figures_nn/build_export_j2/legend_2d.svg)

#### 3D 时空网络模块图例
![Legend 3D](../figures_nn/build_export_j2/legend_3d.svg)

---

### C.1 CNN Attention Lite
**架构解析**：本模型是针对低算力边缘设备定制的轻量化 CNN 基线。核心设计借鉴了 MobileNet 的**深度可分离卷积 (Depthwise Separable Convolution)** 思想，将标准卷积分解为深度卷积 (DW Conv) 和逐点卷积 (PW Conv)，大幅降低了参数量。在此基础上，引入了 **SE (Squeeze-and-Excitation)** 通道注意力模块，通过全局池化和全连接层自适应地重标定通道权重，增强了模型对物理场关键特征（如激波锋面）的捕捉能力。
**适用场景**：适用于对推理延迟极度敏感（<5ms）的实时监测场景。
**源码对应**：`models/spatial/cnn_attn_lite.py`
![CNN Attention Lite](../figures_nn/build_export_j2/cnn_attn_lite/fig_cnn_attn_lite_auto.svg)

### C.2 Conv Gate Lite
**架构解析**：该模型是 SOTA 图像复原网络 **NAFNet** 的工程简化版。它完全摒弃了传统的非线性激活函数（如 ReLU、GELU），转而利用**简单门控机制 (SimpleGate)** 来引入非线性。核心公式为 $Gate(X, Y) = X \odot Y$，其中 $X, Y$ 为特征图沿通道切分后的两部分。这种设计消除了复杂的指数/对数计算，极大提升了硬件亲和性与推理吞吐量。
**优缺点**：计算极快且显存占用低，但在处理极度复杂的湍流纹理时，表达能力略逊于带注意力的大模型。
**源码对应**：`models/spatial/conv_gate_lite.py`
![Conv Gate Lite](../figures_nn/build_export_j2/conv_gate_lite/fig_conv_gate_lite_auto.svg)

### C.3 ConvLSTM
**架构解析**：专为时空序列预测设计的循环神经网络 (RNN) 变体。传统的 LSTM 在处理图像序列时会破坏空间结构（因为使用了全连接层），而 ConvLSTM 将输入到状态、状态到状态的转换全部替换为**卷积操作**。公式描述为：
$$i_t = \sigma(W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + b_i)$$
通过这种方式，记忆单元 $\mathcal{C}_t$ 和隐藏状态 $\mathcal{H}_t$ 均保留了 3D 张量结构 $(C, H, W)$，能够同时捕捉局部空间特征与时间演化规律。
**源码对应**：`models/temporal/components/conv_temporal.py`
![ConvLSTM](../figures_nn/build_export_j2/convlstm/fig_convlstm_auto.svg)

### C.4 ConvUNet Lite
**架构解析**：这是经典 U-Net 的“瘦身版”实现。保留了对称的编码器-解码器结构和跳跃连接 (Skip Connections)，但对内部 Block 进行了极致精简：仅包含 **Conv3x3 -> GELU -> Conv3x3** 的残差结构，移除了所有复杂的注意力机制或密集连接。下采样采用 MaxPool，上采样采用双线性插值后接卷积。
**定位**：作为纯 CNN 架构在低参数量限制下的性能下界基准，用于验证“复杂结构是否真的必要”。
**源码对应**：`models/spatial/conv_unet_lite.py`
![ConvUNet Lite](../figures_nn/build_export_j2/conv_unet_lite/fig_conv_unet_lite_auto.svg)

### C.5 DeepONet
**架构解析**：基于通用逼近定理 (Universal Approximation Theorem) 的算子学习模型。架构包含两个独立子网：
1.  **Branch Net**: 编码离散的输入函数 $u$（如稀疏传感器读数），输出特征向量 $[b_1, \dots, b_p]$。
2.  **Trunk Net**: 编码查询坐标 $y$（即我们想知道物理量的位置），输出基函数值 $[t_1, \dots, t_p]$。
最终输出为两者的内积：$G(u)(y) = \sum_{k=1}^p b_k \cdot t_k$。这种解耦设计使得 DeepONet 特别适合处理非结构化网格或从极稀疏点重建连续场。
**源码对应**：`models/spatial/deeponet.py`
![DeepONet](../figures_nn/build_export_j2/deeponet/fig_deeponet_auto.svg)

### C.6 EDSR
**架构解析**：增强型深度超分辨率网络 (Enhanced Deep Super-Resolution)。该模型基于 ResNet，但做出了针对物理/图像重建的关键修改：**移除了 Batch Normalization (BN) 层**。
**理论依据**：BN 层在分类任务中用于归一化特征分布，但在超分辨率或物理场重建中，绝对数值分布（如温度值、流速值）包含重要物理信息，BN 的归一化会破坏这些信息并限制网络的数值范围灵活性。EDSR 通过堆叠大量宽残差块 (ResBlocks) 并引入残差缩放 (Residual Scaling) 来稳定训练，是公认的稳健基线。
**源码对应**：`models/spatial/edsr.py`
![EDSR](../figures_nn/build_export_j2/edsr/fig_edsr_auto.svg)

### C.7 FNO (Fourier Neural Operator)
**架构解析**：傅里叶神经算子，一种具有**分辨率无关性 (Resolution Invariance)** 的模型。其核心操作是在频域进行卷积：
$$v_{t+1}(x) = \sigma(W v_t(x) + \mathcal{F}^{-1}(R \cdot \mathcal{F}(v_t)(k)))$$
1.  **FFT**: 将特征变换到频域。
2.  **Filtering**: 仅保留低频模态（截断高频），并对其乘以可学习的复数权重矩阵 $R$。
3.  **IFFT**: 变换回空域。
这种设计使得 FNO 能够高效捕捉全局特征（全局感受野），且训练后可应用于任意分辨率的网格，非常适合求解偏微分方程 (PDE)。
**源码对应**：`models/spatial/fno2d.py`
![FNO](../figures_nn/build_export_j2/fno/fig_fno_auto.svg)

### C.8 Hybrid Model
**架构解析**：一种集成式“专家混合”架构，旨在结合不同范式的优势。采用并行三分支设计：
1.  **Attention Branch**: 利用 Transformer 捕捉长程依赖。
2.  **FNO Branch**: 利用频域卷积捕捉全局低频模式。
3.  **U-Net Branch**: 利用局部卷积捕捉高频边界细节。
三者的输出通过可学习的权重或注意力机制进行融合。这种设计显著提升了模型在复杂多尺度物理场（如湍流）中的泛化能力。
**源码对应**：`models/spatial/hybrid.py`
![Hybrid Model](../figures_nn/build_export_j2/hybrid/fig_hybrid_auto.svg)

### C.9 LIIF (Local Implicit Image Function)
**架构解析**：基于**隐式神经表示 (Implicit Neural Representation)** 的前沿模型。LIIF 不直接输出离散像素，而是学习一个连续函数 $f(z, x) \to s$。
**核心机制**：
1.  **Feature Unfold**: 对 Encoder 提取的特征图进行 3x3 邻域展开，丰富局部上下文。
2.  **Local Ensemble**: 对于任意查询坐标 $x_q$，找到其在特征网格中最近的 4 个潜在编码 $z_{00}, z_{01}, z_{10}, z_{11}$，分别预测 RGB 值，然后根据面积权重进行加权融合。
这使得 LIIF 能够实现**任意倍率的超分辨率 (Arbitrary-scale SR)**，即只需训练一次，即可在测试时以 2x, 4x, 甚至 30x 的分辨率重建物理场。
**源码对应**：`models/spatial/liif.py`
![LIIF](../figures_nn/build_export_j2/liif/fig_liif_auto.svg)

### C.10 Mixer
**架构解析**：MLP-Mixer 的一种通用实现变体。它挑战了“卷积或注意力是必须的”这一成见，仅使用全连接层 (Dense Layers) 和转置操作。
**机制**：输入被切分为 Patch，首先通过一个 MLP 混合不同 Patch 之间的信息（空间混合），然后转置，通过另一个 MLP 混合每个 Patch 内部通道的信息（通道混合）。这种简单的架构在数据量充足时表现出了惊人的竞争力。
**源码对应**：`models/spatial/mixer/mixer.py`
![Mixer](../figures_nn/build_export_j2/mixer/fig_mixer_auto.svg)

### C.11 MLP
**架构解析**：最基础的多层感知机网络 (Multi-Layer Perceptron)。
**应用**：在本项目中，主要用于**逐点 (Point-wise) 映射**任务，或者作为 DeepONet 的 Trunk Net 部分。虽然无法捕捉空间相关性，但常作为极简基线来评估空间特征提取模块（如 CNN/ViT）带来的增益到底有多少。
**源码对应**：`models/spatial/mlp.py`
![MLP](../figures_nn/build_export_j2/mlp/fig_mlp_auto.svg)

### C.12 MLP-Mixer
**架构解析**：Google 提出的全 MLP 视觉骨干网络。
**核心组件**：
- **Token-mixing MLP**: 作用于列（空间位置），允许不同位置的特征进行交互。
- **Channel-mixing MLP**: 作用于行（通道特征），允许同一位置的不同特征进行交互。
它证明了在拥有足够归纳偏置（如 Patch 切分）的情况下，简单的矩阵乘法也能学习到复杂的空间特征。
**源码对应**：`models/spatial/mlp_mixer.py`
![MLP-Mixer](../figures_nn/build_export_j2/mlp_mixer/fig_mlp_mixer_auto.svg)

### C.13 NAFNet
**架构解析**：非线性激活自由网络 (Nonlinear Activation Free Network)。该模型是图像复原领域的里程碑式工作。
**核心创新**：
1.  **移除非线性激活**：移除了 ReLU、GELU 等激活函数，避免了梯度消失/爆炸风险。
2.  **SimpleGate**: 仅通过 $X \odot Y$ 引入非线性，计算极其高效。
3.  **SCA (Simplified Channel Attention)**: 简化了传统的通道注意力，移除了其中的 Global Pooling 和复杂 FC 层，仅保留通道加权功能。
**源码对应**：`models/spatial/nafnet.py`
![NAFNet](../figures_nn/build_export_j2/nafnet/fig_nafnet_auto.svg)

### C.14 PartialConv UNet
**架构解析**：引入**部分卷积 (Partial Convolution)** 的 U-Net 变体，专为处理缺失数据设计。
**机制**：标准卷积对缺失值（通常填0）非常敏感，会导致模糊。PartialConv 在卷积时引入一个二值掩码 $M$，仅对 $M=1$ 的有效像素进行卷积运算，并在每一层自动更新掩码（膨胀）。
$$x' = \begin{cases} W^T (x \odot M) \frac{sum(1)}{sum(M)} + b & \text{if } sum(M) > 0 \\ 0 & \text{otherwise} \end{cases}$$
这种机制使其成为稀疏观测重建（Inpainting 任务）的强力候选者。
**源码对应**：`models/spatial/partialconv_unet.py`
![PartialConv UNet](../figures_nn/build_export_j2/partialconv_unet/fig_partialconv_unet_auto.svg)

### C.15 PerceiverIO
**架构解析**：DeepMind 提出的通用感知机架构。
**核心思想**：为了处理任意大小和模态的输入（如百万级像素或稀疏点云），PerceiverIO 引入了一组固定大小的**潜在向量 (Latent Array)**。
1.  **Cross-Attention (Encode)**: 将输入映射到 Latent Array（$O(N)$ 复杂度）。
2.  **Self-Attention (Process)**: 在深层 Latent Space 中进行处理（复杂度与输入大小无关）。
3.  **Cross-Attention (Decode)**: 将处理后的 Latent 映射回目标输出结构（如图像网格）。
**源码对应**：`models/spatial/perceiverio.py`
![PerceiverIO](../figures_nn/build_export_j2/perceiverio/fig_perceiverio_auto.svg)

### C.16 Physics-Informed
**架构解析**：物理感知神经网络 (PINN) 的变体实现。这不仅仅是一个架构，更是一种训练范式。
**机制**：在标准网络（如 MLP 或 ResNet）的输出端，通过自动微分 (Auto-grad) 计算输出场 $u$ 对时空的偏导数，并将其代入偏微分方程（如 Navier-Stokes）。产生的残差作为**物理损失 (Physics Loss)** 添加到总损失中：
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{PDE}$$
这强制网络不仅拟合数据，还必须符合物理定律（如质量守恒、动量守恒）。
**源码对应**：`models/temporal/components/physics_constraints.py`
![Physics](../figures_nn/build_export_j2/physics/fig_physics_auto.svg)

### C.17 RCAN
**架构解析**：残差通道注意力网络 (Residual Channel Attention Network)。
**核心设计**：采用了**“残差中的残差 (RIR)”** 结构，包含多个残差组 (Residual Group)，每个组内又包含多个残差块。
**Channel Attention (CA)**: 每个残差块内都嵌入了 CA 模块，通过全局平均池化压缩空间信息，学习通道间的相关性，自适应地强调高频特征通道。这种极深的网络结构使其在超分辨率任务中能够恢复极高频的纹理细节。
**源码对应**：`models/spatial/rcan.py`
![RCAN](../figures_nn/build_export_j2/rcan/fig_rcan_auto.svg)

### C.18 RDN
**架构解析**：残差密集网络 (Residual Dense Network)。
**核心设计**：结合了 ResNet 的残差连接与 DenseNet 的**密集连接 (Dense Connection)**。
1.  **RDB (Residual Dense Block)**: 块内的每一层都接收之前所有层的输出作为输入，实现了局部特征的极致复用。
2.  **GFF (Global Feature Fusion)**: 在网络末端，融合所有 RDB 的输出，确保低层特征也能直接辅助最终重建。
**源码对应**：`models/spatial/rdn.py`
![RDN](../figures_nn/build_export_j2/rdn/fig_rdn_auto.svg)

### C.19 ResNet Lite
**架构解析**：标准 ResNet 的回归版本，针对物理场重建进行了轻量化适配。
**修改点**：移除了用于分类的池化层和全连接层，保留了核心的残差块堆叠结构。相比于 EDSR，它可能保留了 BN 层（视具体配置而定）或使用了较少的通道数，旨在提供一个参数量适中、推理速度快的通用基线。
**源码对应**：`models/spatial/resnet.py`
![ResNet Lite](../figures_nn/build_export_j2/resnet_lite/fig_resnet_lite_auto.svg)

### C.20 Restormer
**架构解析**：专为图像复原任务优化的 Transformer 架构，CVPR 2022 最佳论文之一。
**核心创新**：
1.  **MDTA (Multi-DConv Head Transposed Attention)**: 传统的 Self-Attention 在空间维度计算（复杂度 $O(H^2 W^2)$），而 MDTA 在**通道维度**计算互协方差（复杂度 $O(C^2)$）。这使得网络能够处理高分辨率图像。
2.  **GDFN (Gated-DConv Feed-Forward Network)**: 在 FFN 中引入门控机制和深度卷积，增强了局部特征提取能力。
**源码对应**：`models/spatial/restormer.py`
![Restormer](../figures_nn/build_export_j2/restormer/fig_restormer_auto.svg)

### C.21 SegFormer
**架构解析**：原用于语义分割的高效 Transformer。
**核心组件**：
1.  **MiT (Mix Transformer) Encoder**: 采用分层结构（从 H/4 到 H/32），使用 Overlap Patch Embedding 保持局部连续性，并使用 Efficient Self-Attention 降低复杂度。
2.  **All-MLP Decoder**: 仅由线性层构成的解码器，简单高效地融合多尺度特征。
在本项目中，我们将其适配为回归任务，利用其强大的多尺度建模能力重建精细的物理场结构。
**源码对应**：`models/spatial/segformer.py`
![SegFormer](../figures_nn/build_export_j2/segformer/fig_segformer_auto.svg)

### C.22 Sequential Model
**架构解析**：这是一个通用的时序模型容器，而非单一网络。它通常指代基于 **RNN (LSTM/GRU)** 或 **Transformer (GPT-style)** 的自回归模型。
**工作流**：接收历史帧序列 $u_{t-k}, \dots, u_t$，通过内部状态更新，预测下一帧 $u_{t+1}$。在本项目中，它常作为外层包装器，内部可以嵌入 ConvLSTM、SwinTemporal 等具体的时间演化模块。
**源码对应**：`models/temporal/sequential_trainer.py`
![Sequential Model](../figures_nn/build_export_j2/sequential/fig_sequential_auto.svg)

### C.23 Sparse Swin UNet
**架构解析**：针对稀疏输入优化的 Swin Transformer U-Net。
**机制**：在标准 Swin UNet 的基础上，可能引入了以下优化之一：
1.  **Masked Attention**: 仅在有效像素（观测点）周围计算注意力。
2.  **Sparse Embedding**: 输入层仅处理非零位置，通过稀疏卷积或索引操作降低计算量。
旨在解决全图 Attention 在处理极稀疏数据时的计算浪费问题。
**源码对应**：`models/spatial/sparse/swin_unet.py`
![Sparse Swin UNet](../figures_nn/build_export_j2/sparse_swin_unet/fig_sparse_swin_unet_auto.svg)

### C.24 Swin Transformer
**架构解析**：基于**移动窗口自注意力 (Shifted Window Attention)** 的通用视觉骨干。
**机制**：
1.  **Window Attention**: 将图像切分为不重叠的窗口，仅在窗口内计算 Self-Attention，复杂度由平方级降为线性。
2.  **Shifted Window**: 在下一层，移动窗口划分位置，使得上一层独立的窗口之间能够进行信息交互（Cross-window Connection）。
这种设计完美兼顾了局部细节捕捉（窗口内）和全局长程依赖（跨窗口），是目前最强大的 Vision Transformer 骨干。
**源码对应**：`models/spatial/swin_t.py`
![Swin Transformer](../figures_nn/build_export_j2/swin/fig_swin_auto.svg)

### C.25 SwinIR
**架构解析**：专门用于图像复原的 Swin Transformer 变体。
**区别于 SwinT**：
1.  **无下采样**：保持特征图分辨率不变，以保留高频信息。
2.  **RSTB (Residual Swin Transformer Block)**: 采用了深层的残差结构，包含多个 STL (Swin Transformer Layer) 和卷积层。
3.  **ConvFFN**: 可能在 FFN 中引入卷积以增强局部性。
SwinIR 在超分辨率、去噪等任务上通常能取得优于 CNN 和传统 Transformer 的效果。
**源码对应**：`models/spatial/swinir.py`
![SwinIR](../figures_nn/build_export_j2/swinir/fig_swinir_auto.svg)

### C.26 SwinT
**架构解析**：Swin Transformer 的 **Tiny** 版本实现。
**参数设置**：通常指 `embed_dim=96`, `depths=[2, 2, 6, 2]`, `num_heads=[3, 6, 12, 24]` 的配置。
**定位**：作为轻量级 Transformer 基线，用于评估自注意力机制在较小参数规模（~28M）下的有效性，与 ResNet50 等量级模型进行公平对比。
**源码对应**：`models/spatial/swin_t.py`
![SwinT](../figures_nn/build_export_j2/swint/fig_swint_auto.svg)

### C.27 Swin Temporal
**架构解析**：Swin Transformer 的时序扩展版本。
**机制**：这通常指在 Swin Block 中引入时间维度的处理，例如：
1.  **3D Window**: 将窗口扩展为 $(T, H, W)$。
2.  **Temporal Attention**: 在空间 Attention 之后增加专门的时间 Attention 层。
用于捕捉物理场随时间的动态演化规律。
**源码对应**：`models/temporal/wrappers/swin_temporal.py`
![Swin Temporal](../figures_nn/build_export_j2/swin_temporal/fig_swin_temporal_auto.svg)

### C.28 Swin UNet
**架构解析**：纯 Transformer 的 U 型网络。
**设计**：完全利用 Swin Transformer Block 替代了 U-Net 中的卷积层。
- **Encoder**: Swin Blocks + Patch Merging (下采样)。
- **Decoder**: Swin Blocks + Patch Expanding (上采样)。
- **Skip Connection**: 将 Encoder 的多尺度特征直接传递给 Decoder。
是医学图像分割和物理场重建领域的 SOTA 模型之一。
**源码对应**：`models/spatial/swin_unet.py`
![Swin UNet](../figures_nn/build_export_j2/swin_unet/fig_swin_unet_auto.svg)

### C.29 SwinUNet (Variant)
**架构解析**：Swin UNet 的另一种实现变体。可能在具体的 Block 堆叠方式、瓶颈层设计（如是否加入 FNO）或跳跃连接的处理上（如是否加入卷积融合）有所不同，作为消融实验的一部分。
**源码对应**：`models/spatial/swinunet.py`
![SwinUNet](../figures_nn/build_export_j2/swinunet/fig_swinunet_auto.svg)

### C.30 U-FNO
**架构解析**：将 Fourier Neural Operator (FNO) 嵌入到 U-Net 瓶颈层的混合架构。
**动机**：U-Net 擅长提取局部细节，但感受野受限于卷积层数；FNO 拥有全局感受野，但对高频细节捕捉不足。
**设计**：利用 U-Net 的 Encoder 提取多尺度特征，在最深层（Bottleneck）使用 FNO 处理全局低频信息，再由 Decoder 重建。这种“取长补短”的设计在很多多尺度物理问题中表现优异。
**源码对应**：`models/spatial/ufno_unet_bottleneck.py`
![U-FNO](../figures_nn/build_export_j2/u-fno/fig_u-fno_auto.svg)

### C.31 UFNO
**架构解析**：U-FNO 的另一种命名或实现版本。通常指代同一类混合架构，可能在具体实现细节（如 FNO 层的模式数、融合方式）上略有差异。
**源码对应**：`models/spatial/ufno_unet_bottleneck.py`
![UFNO](../figures_nn/build_export_j2/ufno/fig_ufno_auto.svg)

### C.32 UNet
**架构解析**：深度学习历史上最经典的**全卷积编码器-解码器**网络。
**核心机制**：
1.  **U型结构**: 对称的收缩路径（Encoder）和扩张路径（Decoder）。
2.  **Skip Connection**: 将浅层的细粒度空间特征拼接到深层的语义特征上，解决了上采样过程中的信息丢失问题。
**地位**：物理场重建任务中最稳健、最通用的基线模型，几乎适用于所有像素级回归任务。
**源码对应**：`models/spatial/unet.py`
![UNet](../figures_nn/build_export_j2/unet/fig_unet_auto.svg)

### C.33 UNetFormer
**架构解析**：结合 U-Net 结构与 Transformer 模块的混合架构。
**设计**：通常保持 U-Net 的整体骨架，但在 Encoder 或 Decoder 的 Block 中引入 Transformer 模块（如 LeWin Block 或 Global Attention），或者仅在 Bottleneck 处使用 Transformer。
**目的**：旨在突破纯 CNN 在全局建模上的局限性，同时保留 U-Net 优秀的局部特征恢复能力。
**源码对应**：`models/spatial/unetformer.py`
![UNetFormer](../figures_nn/build_export_j2/unetformer/fig_unetformer_auto.svg)

### C.34 UNet++
**架构解析**：UNet 的改进版本，引入了**嵌套的密集跳跃连接 (Nested Dense Skip Pathways)**。
**动机**：标准的 U-Net 假设 Encoder 和 Decoder 对应层的语义是直接对齐的，但这并不总是成立。
**机制**：在 Encoder 和 Decoder 之间构建了一系列密集的卷积节点网络，使得特征在融合前经过了不同深度的处理。这显著提升了模型对不同尺度特征的整合能力，当然也增加了计算量。
**源码对应**：`models/spatial/unet_plus_plus.py`
![UNet++](../figures_nn/build_export_j2/unet_plus_plus/fig_unet_plus_plus_auto.svg)

### C.35 UNO
**架构解析**：U-shaped Neural Operator。
**设计**：一种多尺度的神经算子架构。它采用了类似于 U-Net 的多级结构，但每一层（包括下采样和上采样）都由**积分算子**（如 FNO 层）构成，而非普通卷积。
**优势**：相比于标准 FNO（通常只有一层分辨率），UNO 能够在不同分辨率下进行算子学习，从而更精细地捕捉多尺度的物理特征，性能通常优于标准 FNO。
**源码对应**：`models/spatial/uno.py`
![UNO](../figures_nn/build_export_j2/uno/fig_uno_auto.svg)

### C.36 Video Swin Transformer
**架构解析**：Swin Transformer 的 **3D 时空扩展版本**。
**核心机制**：将 2D 的移位窗口机制扩展为 **3D 时空窗口**。
- **3D Window Partition**: 将视频输入 $(T, H, W)$ 切分为 $(P_t, P_h, P_w)$ 大小的 3D 窗口。
- **3D Attention**: 在每个 3D 窗口内计算自注意力。
这使得模型能够同时捕捉空间特征（纹理、边界）和时间演化规律（速度、轨迹），是处理复杂时变物理场（如湍流演化）的核心模型。
**源码对应**：`models/temporal/components/video_swin.py`
![Video Swin Transformer](../figures_nn/build_export_j2/videoswin/fig_videoswin_auto.svg)
