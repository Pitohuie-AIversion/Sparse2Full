# 针对《Sparse2Full》硕士论文的配图策略建议

基于您提供的 `experiment_plan.md` 和 `models_catalog.yaml`，结合硕士论文的学术标准与叙事逻辑，为您定制了以下 **5章节全流程配图策略**。

## 核心原则 (Golden Rules)
1.  **一图胜千言 (Visual First)**：每个核心观点（Method/Result）必须对应一张图。
2.  **证据链闭环**：Method 中的模块图 $\leftrightarrow$ Experiment 中的消融实验图，需一一对应。
3.  **风格统一**：所有图表保持字体（Times New Roman/Arial）、配色（推荐学术蓝/红/灰）、线宽一致。

---

## 章节配图详单

### 第一章：绪论 (Introduction)
**目标**：直观展示“痛点”与“效果”，吸引审稿人/读者。

*   **图 1-1：任务定义与效果概览 (Teaser Figure)**
    *   **内容**：左侧展示极其稀疏的观测点（如 1% 的散点），中间展示你的模型，右侧展示恢复出的高分辨率全流场（GT vs Ours）。
    *   **亮点**：并在下方对比展示 Baseline（如插值）的模糊结果，形成鲜明反差。
    *   **Caption**：从 1% 稀疏观测重建全流场示意图。相比传统插值（左下），本文方法（右下）能精确恢复涡旋结构。

*   **图 1-2：应用场景示意 (Application Context)**
    *   **内容**：绘制一个实际物理场景（如海洋浮标监测、气象站分布），说明“传感器稀疏”是客观存在的物理限制，引出 Sparse2Full 的现实意义。

### 第二章：相关工作 (Related Work)
**目标**：梳理领域脉络，定位本文位置。

*   **图 2-1：相关技术分类树 (Taxonomy)**
    *   **结构**：
        *   传统方法（插值、矩阵补全）
        *   深度学习（CNN, Transformer, Operator Learning）
        *   **本文位置**：结合了 Transformer (Swin) 与 Operator Learning (FNO) 的混合架构，并引入序列化训练。

### 第三章：方法 (Methodology) —— **核心章节**
**目标**：清晰展示系统架构与创新模块。

*   **图 3-1：Sparse2Full 整体框架图 (Overall Framework)** —— **全书最重要的图**
    *   **布局**：横向长图。
    *   **流程**：`Sparse Input (x, y, v)` $\to$ `Embedding (Coord/Patch)` $\to$ `Encoder (Swin/Hybrid)` $\to$ `Bottleneck` $\to$ `Decoder` $\to$ `Dense Output`。
    *   **标注**：在图中明确标出 $L_{rec}, L_{spec}, L_{dc}$ 三个损失函数的作用位置。

*   **图 3-2：统一观测算子与数据一致性模块 (Unified Observation Operator)**
    *   **内容**：详细拆解 $H$ 算子的实现（Gaussian Blur $\to$ Subsample）。
    *   **对比**：左边画“训练时的 H”，右边画“测试时的 H”，用等号连接，强调 **Consistency**（这是你实验三的核心）。

*   **图 3-3：序列化时空训练策略 (Sequential Training Curriculum)**
    *   **形式**：流程图或时间轴。
    *   **内容**：展示 Phase 1 ($T=1$) $\to$ Phase 2 ($T=5$) $\to$ Phase 3 ($T=10$) 的渐进过程。可以用阶梯状示意图表示预测长度的增加。

### 第四章：实验与分析 (Experiments) —— **证据章节**
**目标**：多维度证明方法的有效性（对应 `experiment_plan.md`）。

*   **图 4-1：主性能定性对比 (Qualitative Comparison)**
    *   **布局**：矩阵排列。行 = 不同方法 (Bilinear, U-Net, FNO, Swin-UNet, Ours, GT)；列 = 不同时间步或不同案例。
    *   **技巧**：
        1.  使用统一的 `jet` 或 `viridis` 色标。
        2.  **必须包含 Error Map**（预测值 - GT 的绝对误差热图），通常用白-红或白-蓝单色标，能直观展示你的误差最接近全白（零）。
    
*   **图 4-2：频谱分析 (Spectral Analysis)**
    *   **形式**：折线图。X轴 = 波数 (Wavenumber)，Y轴 = 能量 (Power, Log scale)。
    *   **内容**：GT 是一条曲线，你的曲线应该紧贴 GT，而 Baseline (如 UNet) 在高频部分（X轴右侧）会掉下去（过平滑）。这直接证明 $L_{spec}$ 的有效性。

*   **图 4-3：消融实验可视化 (Ablation Visuals)**
    *   **内容**：选取一个局部纹理丰富的区域放大（Zoom-in）。
    *   **对比**：Baseline vs "+Spec" vs "+DC" vs "Full"。展示每个模块加上去后，细节是如何一步步变清晰、观测点是如何变准确的。

*   **图 4-4：鲁棒性分析 (Robustness Curves)**
    *   **形式**：折线图。
    *   **子图 A**：X轴 = 稀疏度 (0.1% - 10%)，Y轴 = Rel-L2。展示在极稀疏端你的优势。
    *   **子图 B**：X轴 = 噪声水平 ($\sigma$)，Y轴 = Rel-L2。

*   **图 4-5：效率-性能权衡 (Efficiency Pareto Frontier)**
    *   **形式**：散点图。X轴 = FLOPs 或 Params，Y轴 = Rel-L2 (越低越好)。
    *   **内容**：你的模型应该位于左下角（低耗能、低误差）或在同等误差下更靠左。

### 第五章：总结与展望 (Conclusion)
通常不需要图，但如果有余力，可以放一张“未来工作展望图”（如：应用到 3D 流场或不规则网格）。

---

## 执行工具推荐
*   **架构图/流程图**：PowerPoint (最快), Visio, 或 Adobe Illustrator (出版级)。
*   **数据图表**：Python (`matplotlib`, `seaborn`)。确保导出为 PDF 矢量格式，避免放大模糊。
*   **热图/可视化**：Python (`matplotlib.pyplot.imshow`)。确保色标 (Colorbar) 统一且有物理单位。

这个策略完全覆盖了您的 `experiment_plan.md` 中的所有实验点，并将其逻辑化地转化为论文中的视觉证据。是否需要我针对某个具体的图提供 Python 绘图代码？