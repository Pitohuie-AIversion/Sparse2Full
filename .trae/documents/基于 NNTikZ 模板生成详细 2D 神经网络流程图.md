
# 2D 平面神经网络流程图生成计划

## 核心目标
使用 `nntikz` 模板风格（基于 `transformer.tex` 的扁平化设计），生成包含详细内部结构的 2D 平面流程图，补充 3D 视图中省略的细节。

## 具体实施步骤

### 1. 创建生成脚本
新建 `thesis_paper/figures_nn/export_and_gen_tikz_2d.py`，专门负责 2D 图生成。
-   **Header 定义**：提取 `nntikz` 的核心样式（`block`, `layer`, `input`, `arrow`）作为通用头部。
-   **输出目录**：`thesis_paper/figures_nn/build_export/2d/`。

### 2. 模型生成器实现 (Detailed 2D)

#### EDSR
-   **主干**：水平排列。
-   **ResBlock 展开**：画出一个放大的 ResBlock 框，内部包含 `Conv` -> `ReLU` -> `Conv` 节点，以及残差连接。
-   **整体**：Head -> Body (N x ResBlock) -> Tail -> Upsample -> Output。

#### UNet
-   **U型结构**：使用 `positioning` 库实现经典的 U 形布局。
-   **DoubleConv 展开**：每个 Encoder/Decoder 块显示为两个连续的 `Conv-ReLU` 层。
-   **Skip Connection**：清晰的水平箭头连接 Encoder 和 Decoder。

#### SwinT
-   **Swin Block 展开**：详细画出 `LayerNorm` -> `W-MSA` -> `LayerNorm` -> `MLP` 的内部数据流，包括残差连接。
-   **Patch Merging**：显示为独立的层级变换模块。

#### SegFormer
-   **Mix-FFN 展开**：在 Transformer Block 中展示 `Conv` (Depth-wise) 在 FFN 中的位置。
-   **Overlap Patch Embed**：显示为卷积层。

#### FNO2d
-   **Fourier Layer 展开**：清晰展示 `FFT` (2D) -> `Spectral Transform` (R/I part) -> `IFFT` (2D) 的频域处理流程，以及并行的空间域 `1x1 Conv`。

### 3. 执行生成
-   运行脚本生成 `.tex` 文件。
-   调用 `tectonic` 编译生成 `.pdf`。
-   清理中间文件。

### 4. 目录结构
```
thesis_paper/figures_nn/build_export/2d/
├── EDSR/
├── UNet/
├── ...
```
每个模型一个子文件夹，保持整洁。
