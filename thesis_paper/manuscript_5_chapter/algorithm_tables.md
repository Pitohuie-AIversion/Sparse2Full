# 论文核心算法表 (Algorithm Tables - Markdown版)

本文档使用 Markdown 原生格式整理了论文核心算法流程，可直接在支持 Markdown 的编辑器（如 Typora, Obsidian）或在线平台渲染查看。

---

## 算法 1：统一观测算子生成 (Unified Observation Operator Generation)

**对应章节**：3.2 统一观测算子模块  
**核心逻辑**：确保训练退化算子 ($DC$) 与测试观测算子 ($H$) 的严格一致性 ($H \equiv DC$)，包含抗混叠滤波与边界对齐。

> **Algorithm 1: Unified Observation Operator Generation ($H \equiv DC$)**
>
> **输入 (Input)**: 
> *   高分辨率物理场 $U \in \mathbb{R}^{B \times C \times H \times W}$
> *   任务配置 $\mathcal{C}$ (Task Config)
> *   参数 $\Theta$ (缩放倍率 $s$, 高斯核 $\sigma$, 裁剪尺寸 $h_c, w_c$)
>
> **输出 (Output)**: 
> *   稀疏观测 $y$ (Sparse Observation)
> *   观测掩码 $M$ (Observation Mask)
>
> **过程 (Process)**:
> 1.  **初始化** $M \leftarrow \mathbf{1}^{B \times 1 \times H \times W}$ (全观测状态)
> 2.  **If** 任务类型 is **Super-Resolution (SR)** **Then**
>     3.  **生成核**: 根据 $\Theta$ 构建高斯核 $K$
>     4.  **抗混叠滤波**: $U_{\text{blur}} \leftarrow U * K$ (采用反射填充卷积)
>     5.  **物理下采样**: $y \leftarrow \text{Downsample}(U_{\text{blur}}, \text{scale}=1/s, \text{mode}='area')$
>     6.  **掩码更新**: $M \leftarrow \text{Downsample}(M, \text{scale}=1/s, \text{mode}='nearest')$
> 7.  **Else If** 任务类型 is **Crop (Limited FOV)** **Then**
>     8.  **几何对齐**: 计算中心坐标 $(c_x, c_y)$ 并对齐 Patch 网格
>     9.  **窗口定义**: 确定裁剪窗口 $\mathcal{W}$，尺寸为 $(h_c, w_c)$
>     10. **稀疏采样**: $y \leftarrow \text{Crop}(U, \mathcal{W})$
>     11. **掩码更新**: $M \leftarrow \text{UpdateMask}(M, \mathcal{W}, \text{value}=0)$ (未观测区域置0)
> 12. **End If**
> 13. **Return** $y, M$

---

## 算法 2：序列化时空课程学习 (Sequential Spatiotemporal Curriculum Learning)

**对应章节**：3.4 训练策略  
**核心逻辑**：将欠定反问题解耦为“空间重构 $\to$ 时序演化 $\to$ 联合微调”三个阶段，解决直接端到端训练收敛难的问题。

> **Algorithm 2: Sequential Spatiotemporal Curriculum Learning**
>
> **输入 (Input)**: 
> *   数据集 $\mathcal{D}$, 空间编码器 $E_s$, 时序模块 $E_t$, 解码器 $D$
> *   最大轮数 $T_{max}$, 阶段阈值 $T_1, T_2$
>
> **输出 (Output)**: 
> *   优化后的模型参数 $\theta^* = \{\theta_{E_s}, \theta_{E_t}, \theta_{D}\}$
>
> **过程 (Process)**:
>
> **// 第一阶段：空间重构 (Spatial Reconstruction)**
> 1.  **For** epoch $e = 1$ **to** $T_1$ **Do**
>     2.  **冻结 (Freeze)**: 时序模块 $\theta_{E_t}$
>     3.  **采样**: $(x, y) \sim \mathcal{D}_{\text{spatial}}$ (单帧样本)
>     4.  **前向**: $\hat{y} \leftarrow D(E_s(x))$
>     5.  **更新**: 优化 $\theta_{E_s}, \theta_{D}$ 最小化 $\mathcal{L}_{\text{rec}}(y, \hat{y})$
> 6.  **End For**
>
> **// 第二阶段：时序演化 (Temporal Evolution)**
> 7.  **For** epoch $e = T_1 + 1$ **to** $T_2$ **Do**
>     8.  **冻结 (Freeze)**: 空间模块 $\theta_{E_s}, \theta_{D}$
>     9.  **采样**: $(x_{1:T}, y_{1:T}) \sim \mathcal{D}_{\text{seq}}$ (序列样本)
>     10. **特征提取**: $z_{1:T} \leftarrow E_s(x_{1:T})$ (提取空间特征)
>     11. **时序演化**: $\hat{z}_{1:T} \leftarrow E_t(z_{1:T})$ (学习动力学)
>     12. **解码**: $\hat{y}_{1:T} \leftarrow D(\hat{z}_{1:T})$
>     13. **更新**: 优化 $\theta_{E_t}$ 最小化 $\mathcal{L}_{\text{rec}} + \lambda \mathcal{L}_{\text{temp}}$
> 14. **End For**
>
> **// 第三阶段：联合微调 (Joint Fine-tuning)**
> 15. **For** epoch $e = T_2 + 1$ **to** $T_{max}$ **Do**
>     16. **解冻 (Unfreeze)**: 所有参数 $\theta_{E_s}, \theta_{E_t}, \theta_{D}$
>     17. **前向**: 计算全序列预测 $\hat{y}_{1:T}$
>     18. **计算物理一致性损失**: 
>         $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_s \mathcal{L}_{\text{spec}} + \lambda_{dc} \mathcal{L}_{\text{dc}}$
>     19. **更新**: 联合优化 $\theta^*$ 最小化 $\mathcal{L}_{\text{total}}$
> 20. **End For**
> 21. **Return** $\theta^*$

---

## 算法 3：基于一致性校验的训练闭环 (Consistency-Aware Training Step)

**对应章节**：3.1 总体框架 & 3.5 损失函数  
**核心逻辑**：展示单步训练中，数据如何经过统一算子生成观测，并通过三元混合损失（重建+谱+观测一致性）进行闭环优化。

> **Algorithm 3: Consistency-Aware Training Step**
>
> **输入 (Input)**: 
> *   高分辨率真值 Batch $U$
> *   统一观测算子 $H(\cdot; \Theta)$
>
> **输出 (Output)**: 
> *   更新后的模型参数 $\theta$
>
> **过程 (Process)**:
>
> 1.  **观测生成 (Observation Generation)**
>     *   $y_{\text{obs}}, M \leftarrow H(U; \Theta)$ (应用退化算子)
>     *   构建输入 $X_{\text{in}} \leftarrow \text{Concat}(y_{\text{obs}}, M, \text{Coords})$
>
> 2.  **前向重建 (Forward Reconstruction)**
>     *   $\hat{U} \leftarrow \text{Model}(X_{\text{in}}; \theta)$
>
> 3.  **一致性校验 (Consistency Check)**
>     *   $\hat{y}_{\text{proj}}, \_ \leftarrow H(\hat{U}; \Theta)$ (将预测结果重投影回观测空间)
>
> 4.  **损失计算 (Loss Calculation)**
>     *   **数据保真项**: $\mathcal{L}_{\text{rec}} \leftarrow || \hat{U} - U ||_2^2$
>     *   **谱一致性项**: $\mathcal{L}_{\text{spec}} \leftarrow || \text{FFT}(\hat{U}) \cdot W_{\text{low}} - \text{FFT}(U) \cdot W_{\text{low}} ||_2^2$
>     *   **观测一致性项**: $\mathcal{L}_{\text{dc}} \leftarrow || \hat{y}_{\text{proj}} - y_{\text{obs}} ||_2^2$ (核心约束)
>     *   **总损失**: $\mathcal{L}_{\text{total}} \leftarrow \mathcal{L}_{\text{rec}} + \lambda_s \mathcal{L}_{\text{spec}} + \lambda_{dc} \mathcal{L}_{\text{dc}}$
>
> 5.  **优化 (Optimization)**
>     *   计算梯度 $\nabla_{\theta} \mathcal{L}_{\text{total}}$
>     *   更新参数 $\theta \leftarrow \text{Optimizer}(\theta, \nabla_{\theta} \mathcal{L}_{\text{total}})$
