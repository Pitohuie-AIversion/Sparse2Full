# 模型架构详细说明文档

**版本**: 1.1  
**生成日期**: 2026-01-03  
**维护者**: Sparse2Full Team  

本文档系统性地整理了 `Sparse2Full` 项目中所有核心模型架构的详细信息，包括空间重建模型、时序预测模型及自回归框架。

---

## 目录 (Index)

1.  [第一章：空间重建模型 (Spatial Reconstruction Models)](#第一章空间重建模型-spatial-reconstruction-models)
    *   [1.1 Swin-UNet (v1.0)](#11-swin-unet-v10)
    *   [1.2 FNO 2D - Fourier Neural Operator (v1.0)](#12-fno-2d---fourier-neural-operator-v10)
    *   [1.3 Classic U-Net (v2.0)](#13-classic-u-net-v20)
    *   [1.4 DeepONet 2D (v1.0)](#14-deeponet-2d-v10)
    *   [1.5 SegFormer (v1.0)](#15-segformer-v10)
    *   [1.6 UNetFormer (v1.0)](#16-unetformer-v10)
    *   [1.7 ModularSR (v1.0)](#17-modularsr-v10)
    *   [1.8 SparseAttentionEncoder (v1.0)](#18-sparseattentionencoder-v10)
2.  [第二章：时序预测模型 (Temporal Prediction Models)](#第二章时序预测模型-temporal-prediction-models)
    *   [2.1 SwinTemporal Wrapper (v1.0)](#21-swintemporal-wrapper-v10)
    *   [2.2 PhysicsTransformer (v1.0)](#22-physicstransformer-v10)
    *   [2.3 Temporal Components](#23-temporal-components)
3.  [第三章：自回归框架 (Autoregressive Framework)](#第三章自回归框架-autoregressive-framework)
    *   [3.1 ARWrapper (v1.0)](#31-arwrapper-v10)
4.  [第四章：混合与其他模型 (Hybrid & Other Models)](#第四章混合与其他模型-hybrid--other-models)
    *   [4.1 HybridModel (Attention∥FNO∥UNet)](#41-hybridmodel-attentionfnounet)
    *   [4.2 MLPModel](#42-mlpmodel)

---

## 第一章：空间重建模型 (Spatial Reconstruction Models)

本章包含用于单帧空间场重建的核心模型，适用于超分辨率 (SR)、缺失数据补全 (Inpainting) 等任务。

### 1.1 Swin-UNet (v1.0)

基于 Swin Transformer 的 U-Net 架构，集成了分层 Transformer 编码器与解码器，并支持可选的频域瓶颈层 (FNO Bottleneck)。

#### 1.1.1 核心组件与层级结构

```text
[Input] (B, C_in, H, W)
   │
   ▼
[PatchEmbed] -> Token序列
   │
   ▼
[Encoder] (Swin Transformer Blocks)
   ├── Stage 1: SwinBlock x2 -> PatchMerging (H/2, W/2)
   ├── Stage 2: SwinBlock x2 -> PatchMerging (H/4, W/4)
   ├── Stage 3: SwinBlock x6 -> PatchMerging (H/8, W/8)
   └── Stage 4: SwinBlock x2
   │
   ▼
[Bottleneck] (Optional FNO)
   ├── FFT2D -> 频域卷积 (SpectralConv) -> IFFT2D
   │
   ▼
[Decoder] (Symmetric Swin Blocks)
   ├── Stage 4: SwinBlock x2 -> PatchExpanding (H/4, W/4) + Skip Connect
   ├── Stage 3: SwinBlock x6 -> PatchExpanding (H/2, W/2) + Skip Connect
   ├── Stage 2: SwinBlock x2 -> PatchExpanding (H, W) + Skip Connect
   └── Stage 1: SwinBlock x2
   │
   ▼
[Output Head] Conv2d(1x1) -> (B, C_out, H, W)
```

#### 1.1.2 输入输出规格

*   **输入**: `[B, C_in, H, W]` - 归一化的观测场数据。
*   **输出**: `[B, C_out, H, W]` - 重建的物理场数据。

#### 1.1.3 关键参数配置表

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `in_channels` | 3 | 输入通道数 |
| `out_channels` | 3 | 输出通道数 |
| `img_size` | 256 | 输入图像尺寸 |
| `patch_size` | 4 | Patch 嵌入大小 |
| `embed_dim` | 96 | 基础嵌入维度 |
| `depths` | [2, 2, 6, 2] | 编码器各阶段层数 |
| `num_heads` | [3, 6, 12, 24] | 各阶段注意力头数 |
| `window_size` | 8 | 局部注意力窗口大小 |
| `use_fno_bottleneck` | False | 是否启用频域瓶颈 |

#### 1.1.4 适用场景与性能特点

*   **适用场景**: 高精度流场重建、多尺度结构恢复。
*   **性能特点**:
    *   **优点**: 强大的长距离依赖捕捉能力，通过层级结构有效处理多尺度特征。
    *   **缺点**: 计算量较大，推理延迟高于纯 CNN 模型。

---

### 1.2 FNO 2D - Fourier Neural Operator (v1.0)

基于频域卷积的神经算子，旨在学习函数空间之间的映射，具有分辨率无关性。

#### 1.2.1 核心组件与层级结构

```text
[Input] (B, C_in, H, W) + [Grid] (B, 2, H, W)
   │
   ▼
[Projector] Linear(C_in+2 -> width)
   │
   ▼
[Spectral Layers] x N_Layers
   ├── 分支1 (频域): FFT -> 截断高频 -> ComplexMul -> IFFT
   ├── 分支2 (空域): Conv1x1 (W)
   └── 融合: Activation(分支1 + 分支2)
   │
   ▼
[Projector] MLP(width -> 128 -> C_out)
   │
   ▼
[Output] (B, C_out, H, W)
```

#### 1.2.2 输入输出规格

*   **输入**: `[B, C_in, H, W]` - 任意分辨率的网格数据。
*   **输出**: `[B, C_out, H, W]` - 对应分辨率的输出场。

#### 1.2.3 关键参数配置表

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `modes1`, `modes2` | 12 | X/Y 方向保留的低频模态数 |
| `width` | 64 | 隐藏层通道宽度 |
| `n_layers` | 4 | 频域卷积层堆叠数 |
| `activation` | 'gelu' | 激活函数类型 |

#### 1.2.4 适用场景与性能特点

*   **适用场景**: PDE 求解、全局物理约束强的场重建。
*   **性能特点**:
    *   **优点**: 零样本超分辨率 (Zero-shot SR)，参数效率高，物理一致性好。
    *   **缺点**: 对高频细节（如激波、边界层）的捕捉能力较弱（受限于模态截断）。

---

### 1.3 Classic U-Net (v2.0)

经典的全卷积编码器-解码器架构，作为稳健的基线模型。

#### 1.3.1 核心组件与层级结构

```text
[Input]
   │
   ▼
[Encoder]
   ├── DoubleConv (64) -> MaxPool
   ├── DoubleConv (128) -> MaxPool
   ├── DoubleConv (256) -> MaxPool
   └── DoubleConv (512) -> MaxPool
   │
   ▼
[Bottleneck] DoubleConv (1024)
   │
   ▼
[Decoder]
   ├── UpSample -> Concat(Skip) -> DoubleConv (512)
   ├── UpSample -> Concat(Skip) -> DoubleConv (256)
   ├── UpSample -> Concat(Skip) -> DoubleConv (128)
   └── UpSample -> Concat(Skip) -> DoubleConv (64)
   │
   ▼
[Output] Conv1x1
```

#### 1.3.2 关键参数配置表

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `features` | [64, 128, 256, 512] | 各层特征通道数 |
| `bilinear` | True | 是否使用双线性插值上采样 |
| `dropout` | 0.0 | Dropout 概率 |

#### 1.3.3 适用场景

*   **适用场景**: 通用图像修复、基准测试。

---

### 1.4 DeepONet 2D (v1.0)

基于算子理论的 Deep Operator Network，适用于非规则网格或稀疏观测点到稠密场的映射。

#### 1.4.1 核心组件与层级结构

```text
[Sparse Input] (B, C, H, W)     [Query Coords] (B, H*W, 2)
      │                               │
      ▼                               ▼
[Branch Net] (CNN)             [Trunk Net] (MLP)
      │                               │
      ▼                               ▼
[Coefficients] (B, P)          [Basis Functions] (B, H*W, P)
      │                               │
      └───────────────┬───────────────┘
                      ▼
               [Dot Product]
                      │
                      ▼
               [Output] (B, H*W, 1) -> Reshape
```

#### 1.4.2 关键参数配置表

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `branch_channels` | [64, 128, 256] | Branch CNN 通道数 |
| `trunk_hidden` | [256, 256, 256] | Trunk MLP 隐藏层节点数 |
| `latent_dim` | 256 | 潜在空间维度 (P) |
| `use_fourier_features` | True | 是否使用 Fourier 特征映射坐标 |

---

### 1.5 SegFormer (v1.0)

轻量级 Transformer 架构，移除位置编码，支持多尺度特征融合。

#### 1.5.1 核心组件与层级结构

```text
[Input]
   │
   ▼
[MiT Encoder] (Mix Transformer)
   ├── Stage 1: Overlap Patch Embed -> Transformer Block
   ├── Stage 2: Overlap Patch Embed -> Transformer Block
   ├── Stage 3: Overlap Patch Embed -> Transformer Block
   └── Stage 4: Overlap Patch Embed -> Transformer Block
   │
   ▼
[MLP Decoder]
   ├── Linear Fuse: 融合所有层特征
   └── Upsample: 恢复分辨率
   │
   ▼
[Output]
```

#### 1.5.2 关键参数
*   `embed_dims`: `[64, 128, 320, 512]`
*   `sr_ratios`: `[8, 4, 2, 1]` (空间缩减比)

---

### 1.6 UNetFormer (v1.0)

结合 CNN 的局部特征提取与 Transformer 的全局建模能力。

#### 1.6.1 核心组件与层级结构

```text
[Input]
   │
   ▼
[Encoder]
   ├── Conv Block
   ├── GL Block (Global-Local Transformer)
   └── Downsample
   │
   ▼
[Bottleneck] GL Block
   │
   ▼
[Decoder]
   ├── Upsample
   ├── Concat(Skip from Encoder)
   └── GL Block
   │
   ▼
[Output] Conv1x1
```

---

### 1.7 ModularSR (v1.0)

模块化超分辨率模型，采用编码器-主干网络-解码器的流水线设计，专为稀疏观测输入优化。

#### 1.7.1 核心组件与层级结构

```text
[Sparse Input] [Coords] [Mask] [FourierPE]
      │           │       │         │
      └───────────┼───────┼─────────┘
                  ▼
         [SparseInputEncoder]
(Concat -> Conv -> Norm -> Act -> Conv)
                  │
                  ▼
              [Backbone]
    (Swin-UNet / FNO / ResNet / ...)
                  │
                  ▼
         [Bilinear3x3Decoder]
    (Upsample -> Bilinear + Conv3x3)
                  │
                  ▼
         [Output] (B, C, H, W)
```

#### 1.7.2 关键参数
*   `encoder_cfg`: 稀疏输入编码器配置
*   `backbone_cfg`: 核心重建网络配置
*   `decoder_cfg`: 解码器配置

---

### 1.8 SparseAttentionEncoder (v1.0)

基于 Senseiver 概念的稀疏注意力编码器，用于增强稀疏输入的特征表示。

#### 1.8.1 核心组件与层级结构

```text
[Sensor Obs]  [Coords]  [Mask]
     │           │        │
     ▼           ▼        ▼
[SensorEmb] [CoordEmb] [MaskEmb]
     │           │        │
     └───────────┼────────┘
                 ▼
         [Feature Fusion]
                 │
                 ▼
      [Sparse Self-Attention]
 (Masked Attention on Valid Points)
                 │
                 ▼
         [Output Features]
```

#### 1.8.2 关键特性
*   **稀疏优化**: 注意力计算仅在有效观测点及其邻域进行。
*   **多模态融合**: 融合观测值、位置信息和稀疏掩码。

---

## 第二章：时序预测模型 (Temporal Prediction Models)

本章包含处理时间序列数据的模型组件，通常作为空间模型的包装器或独立的时空模型。

### 2.1 SwinTemporal Wrapper (v1.0)

将静态的 Swin-UNet 扩展为时序模型，保持空间主干不变，在特征层插入时序聚合模块。

#### 2.1.1 核心组件与层级结构

```text
[Input Sequence] (B, T, C, H, W)
   │
   ▼
[Spatial Encoder] (Swin-UNet Encoder) -> (B*T, C_feat, H', W')
   │
   ▼
[Temporal Aggregation]
   ├── Reshape -> (B, T, C_feat, H', W')
   ├── TemporalConv1D / Transformer / FiLM
   └── Collapse T -> (B, C_feat, H', W')
   │
   ▼
[Spatial Decoder] (Swin-UNet Decoder)
   │
   ▼
[Output Heads]
   ├── NAR Head: 预测未来 k 帧
   └── AR Head: 预测下一帧
```

#### 2.1.2 关键参数配置表

| 参数名 | 说明 |
| :--- | :--- |
| `base_model` | 基础空间模型 (通常为 SwinUNet) |
| `temporal_method` | 聚合方法: 'conv', 'transformer', 'film' |
| `temporal_depth` | 时序模块层数 |
| `nar_prediction_steps` | 非自回归预测步数 |

---

### 2.2 PhysicsTransformer (v1.0)

物理感知 Transformer，嵌入物理约束与多尺度注意力。

#### 2.2.1 核心组件与层级结构

```text
[Input Sequence]
   │
   ▼
[Physics Embedding] (Coords + Params)
   │
   ▼
[Multi-Scale Attention Layers]
   ├── Temporal Attention
   ├── Spatial Attention
   └── Physics Constraint (PDE Residual)
   │
   ▼
[Feed Forward Network]
   │
   ▼
[Output]
```

---

### 2.3 Temporal Components

通用时序组件，可嵌入不同架构。

#### 2.3.1 TemporalBlock Variants

**TemporalConv1D**
```text
(B, T, C, H, W) -> (B*H*W, C, T) -> Conv1d(k) -> (B, C_out, H, W)
```

**TemporalTransformer**
```text
(B, T, C, H, W) -> (B*H*W, T, C) -> TransformerEncoder -> (B, C_out, H, W)
```

**FiLMTemporal**
```text
(B, T, C, H, W) -> GlobalPool -> Linear -> (Gamma, Beta) -> Modulate Last Frame
```

---

## 第三章：自回归框架 (Autoregressive Framework)

### 3.1 ARWrapper (v1.0)

通用的自回归训练与推理包装器，可将任何单帧输入输出模型 (`Image -> Image`) 转换为序列预测模型。

#### 3.1.1 核心组件与层级结构

```text
[Input History] [Ground Truth (Training)]
       │                 │
       ▼                 ▼
  [Single-Step Model] <--┤ (Teacher Forcing Switch)
       │                 │
       ▼                 │
  [Prediction t+1] ──────┘
       │
       ▼ (Loop for T steps)
  [Output Sequence]
```

#### 3.1.2 关键参数配置表

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `teacher_forcing_ratio` | 0.5 | 训练时使用真值的概率 |
| `scheduled_sampling` | False | 是否启用计划采样衰减 |
| `detach_rollout` | True | 推理时是否切断梯度回传 (节省显存) |

---

## 第四章：混合与其他模型 (Hybrid & Other Models)

### 4.1 HybridModel (Attention∥FNO∥UNet)

并行组合模型，旨在融合不同架构的归纳偏置。

#### 4.1.1 核心组件与层级结构

```text
       [Input]
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
 [UNet] [FNO] [Attn]
    │     │     │
    ▼     ▼     ▼
   [w1]  [w2]  [w3] (Learnable Weights)
    │     │     │
    └─────┼─────┘
          ▼
       [Sum] -> [Output]
```

### 4.2 MLPModel

纯多层感知机基线，用于验证极简架构在特定任务上的表现。

#### 4.2.1 核心组件与层级结构

```text
[Input] -> Flatten -> Linear -> ReLU -> ... -> Linear -> Reshape -> [Output]
```

---

**文档结束**
