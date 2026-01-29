# 《Sparse2Full》硕士论文配图策略指南

本指南基于 `experiment_plan.md` 与 `models_catalog.yaml` 制定，旨在为您的 5 章节硕士论文提供一套逻辑严密、视觉专业的配图方案。

## 0. 核心原则 (Golden Rules)

1.  **一图胜千言 (Visual First)**：每个核心观点（Method/Result）必须对应一张图。
2.  **证据链闭环**：Method 中的模块图 $\leftrightarrow$ Experiment 中的消融实验图，需一一对应。
3.  **风格统一**：
    *   **字体**：Times New Roman (英文) / 宋体 (中文) 或 Arial。
    *   **字号**：图内文字不小于 8pt (Caption 往往是 10pt)。
    *   **配色**：推荐学术蓝/红/灰配色，避免高饱和度“默认色”。
    *   **格式**：数据图表导出为 PDF/SVG (矢量)，照片/热图导出为 300dpi PNG。

---

## 1. 章节配图详单

### 第一章：绪论 (Introduction)
**目标**：直观展示“痛点”与“效果”，吸引审稿人/读者。

*   **图 1-1：任务定义与效果概览 (Teaser Figure)**
    *   **内容**：
        *   **左侧 (Input)**：极其稀疏的观测点（如 1% 的散点），叠加在底图轮廓上。
        *   **中间 (Ours)**：Sparse2Full 模型恢复出的高分辨率全流场。
        *   **右侧 (Ground Truth)**：真实的高分辨率流场。
        *   **下方 (Baseline)**：传统插值（Bicubic/Bilinear）的模糊结果，形成鲜明反差。
    *   **Caption**：从 1% 稀疏观测重建全流场示意图。相比传统插值（左下）的模糊结果，本文方法（中）能精确恢复涡旋结构，细节逼近真实值（右）。

*   **图 1-2：应用场景示意 (Application Context)**
    *   **内容**：绘制一个实际物理场景（如海洋浮标监测、气象站分布），说明“传感器稀疏”是客观存在的物理限制，引出 Sparse2Full 的现实意义。

### 第二章：相关工作 (Related Work)
**目标**：梳理领域脉络，定位本文位置。

*   **图 2-1：相关技术分类树 (Taxonomy)**
    *   **结构**：
        *   **传统方法**：插值 (Bilinear, Bicubic)、矩阵补全 (Matrix Completion)。
        *   **深度学习**：
            *   CNN (SRResNet, EDSR)
            *   Transformer (SwinIR, ViT)
            *   Operator Learning (FNO, DeepONet)
        *   **本文位置**：Hybrid (Swin+FNO) + Sequential Training。

### 第三章：方法 (Methodology) —— **核心章节**
**目标**：清晰展示系统架构与创新模块。

*   **图 3-1：Sparse2Full 整体框架图 (Overall Framework)** —— **全书最重要的图**
    *   **布局**：横向长图。
    *   **流程**：
        1.  **Input**: `Sparse Points (x, y, v)` + `Mask` + `Coords`。
        2.  **Embedding**: Coordinate Embedding / Patch Embedding。
        3.  **Encoder**: Swin-UNet Backbone (提取局部特征)。
        4.  **Bottleneck**: FNO / Spectral Block (提取全局/频域特征)。
        5.  **Decoder**: 上采样恢复全分辨率。
        6.  **Output**: `Dense Field`。
    *   **标注**：在图中明确标出 $L_{rec}, L_{spec}, L_{dc}$ 三个损失函数的作用位置。

*   **图 3-2：统一观测算子与数据一致性模块 (Unified Observation Operator)**
    *   **内容**：详细拆解 $H$ 算子的实现。
        *   `High-Res Field` $\xrightarrow{\text{Gaussian Blur}}$ `Filtered Field` $\xrightarrow{\text{Subsample}}$ `Sparse Observations`。
    *   **对比**：左边画“训练时的 H”，右边画“测试时的 H”，用等号连接，强调 **Consistency**（对应实验三）。

*   **图 3-3：序列化时空训练策略 (Sequential Training Curriculum)**
    *   **形式**：流程图或时间轴。
    *   **内容**：
        *   **Phase 1**: 预测 $T=1$ (Warmup)。
        *   **Phase 2**: 预测 $T=5$ (Short-term)。
        *   **Phase 3**: 预测 $T=10$ (Long-term)。
        *   箭头指示 Curriculum Learning 的进阶过程。

### 第四章：实验与分析 (Experiments) —— **证据章节**
**目标**：多维度证明方法的有效性。

*   **图 4-1：主性能定性对比 (Qualitative Comparison)**
    *   **布局**：矩阵排列 (Grid)。
    *   **行**：Methods (Bilinear, U-Net, FNO, Swin-UNet, **Ours**, GT)。
    *   **列**：Time Steps ($T=1, 5, 10$) 或 Different Cases。
    *   **关键点**：**必须包含 Error Map**（$|Pred - GT|$）。Error Map 使用 `seismic` 或 `bwr` (白-红) 色标，白色表示0误差，能直观展示你的误差图最干净。

*   **图 4-2：频谱分析 (Spectral Analysis)**
    *   **形式**：折线图 (Log-Log Plot)。
    *   **X轴**：Wavenumber (波数/频率)。
    *   **Y轴**：Power Spectral Density (能量谱密度)。
    *   **内容**：GT 是一条黑线。你的曲线（红线）应该在高频部分（右侧）紧贴 GT。Baseline（蓝/绿线）通常在高频部分快速衰减（Blurry 现象）。这直接证明 $L_{spec}$ 的有效性。

*   **图 4-3：消融实验可视化 (Ablation Visuals)**
    *   **内容**：选取一个局部纹理丰富的区域放大（Zoom-in）。
    *   **对比**：
        1.  Baseline (Blurry)
        2.  +Spec Loss (纹理出现，但位置可能偏)
        3.  +DC Loss (位置准，但可能噪)
        4.  **Full (Ours)** (既准又清晰)

*   **图 4-4：鲁棒性分析 (Robustness Curves)**
    *   **形式**：折线图。
    *   **子图 A (Sparsity)**：X轴 = 稀疏度 (0.1% $\to$ 10%)，Y轴 = Rel-L2 Error。展示在极稀疏端（左侧）你的曲线更平缓。
    *   **子图 B (Noise)**：X轴 = 噪声水平 ($\sigma=0.0 \to 0.2$)，Y轴 = Rel-L2 Error。

*   **图 4-5：效率-性能权衡 (Efficiency Pareto Frontier)**
    *   **形式**：散点图。
    *   **X轴**：FLOPs (G) 或 Params (M)。
    *   **Y轴**：Rel-L2 Error (越低越好)。
    *   **内容**：你的模型点应位于左下角（Pareto 前沿），或在同等计算量下误差最低。

### 第五章：总结与展望 (Conclusion)
*   （可选）**图 5-1：未来工作展望**。例如展示 3D 流场切片，或不规则网格的示意图，表明方法的扩展潜力。

---

## 2. 常用工具与资源

*   **绘图代码模板**：请参考 `thesis_paper/scripts/plot_templates.py` (即将生成)。
*   **架构图工具**：
    *   **PowerPoint**: 最快，适合组合现成素材。
    *   **Visio**: 传统的工程图标准。
    *   **Adobe Illustrator / Inkscape**: 出版级矢量编辑，适合精修。
    *   **TikZ (LaTeX)**: 纯代码生成，适合数学公式推导图，但学习曲线陡峭。
*   **配色网站**：
    *   [ColorBrewer](https://colorbrewer2.org/): 学术制图配色圣经。
    *   [Coolors](https://coolors.co/): 快速生成配色方案。

## 3. 检查清单 (DoD)

- [ ] 所有坐标轴是否有单位和标签？(e.g., "Time (s)", "Rel-L2 Error")
- [ ] 字号是否在论文中清晰可见？(建议打印出来看一眼)
- [ ] 引用是否正确？(e.g., "Compared with FNO [12]")
- [ ] 矢量图是否已转曲/嵌入字体？(防止在不同电脑上乱码)
- [ ] 热图是否有 Colorbar？Colorbar 的范围是否统一？
