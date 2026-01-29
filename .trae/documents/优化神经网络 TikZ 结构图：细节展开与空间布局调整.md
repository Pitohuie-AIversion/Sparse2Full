
# 优化模型结构图布局与细节展开计划

## 核心目标
1.  **完善细节展开**：对剩余模型（FNO2d, SegFormer）的关键模块进行内部结构展开，确保所有模型都展示核心细节。
2.  **优化空间布局**：调整模块间距，防止重叠，优化 Skip Connection 的路径，确保几何关系清晰。

## 具体实施步骤

### 1. 调整间距参数
-   增加主要模块（Stage/Block 之间）的默认水平间距，从 `1.0` 增加到 `1.5` 或更大。
-   保持内部组件（如 Conv-ReLU）的紧凑间距（`0.2` - `0.5`）。

### 2. 模型细节展开

#### FNO2d (Fourier Neural Operator)
-   **当前**：Lift -> Spectral Conv -> Projection（3 个块）。
-   **改进**：将 "Spectral Conv" 展开为：
    -   `FFT` (Box)
    -   `Spectral Weights` (Thin Box)
    -   `IFFT` (Box)
    -   添加并联的残差连接（$W x$）。

#### SegFormer
-   **当前**：Stage 1-4 均为单一 Box。
-   **改进**：将 **Stage 1** 展开为 MixTransformer Block 结构：
    -   `PatchEmbed` (Conv)
    -   `LayerNorm`
    -   `Attention` (Mix-FFN/Self-Attention)
    -   `LayerNorm`
    -   `MLP`

### 3. 几何关系与防重叠优化

#### EDSR
-   **Skip Connection**：调整 Skip Connection 的高度偏移 `++(0, 3, 0)`，确保不与展开后的 `Conv-ReLU-Conv` 结构发生视觉冲突。
-   **间距**：增加 `head` 到 `res1`，以及 `res1` 到 `res2` 的间距。

#### UNet
-   **Skip Connection**：检查 `enc2` 到 `dec2` 的贝塞尔曲线路径，调整控制点（`out=45, in=135`）以避免穿过中间的 Bottleneck 块。
-   **Input/Output**：将 2D 节点向外移动，避免被 3D 块遮挡。

#### SwinT
-   **Stage 1**：确保展开后的 LN/MSA/MLP 之间有足够间隙，且整体 Stage 1 与 Stage 2 的距离拉大，体现层级感。

### 4. 代码实现
-   修改 `thesis_paper/figures_nn/export_and_gen_tikz.py` 中的生成器函数。
-   重新运行批处理命令生成所有 PDF。
