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
>     10. **稀疏采样**: $y \leftarrow U \odot \mathcal{W}$ (在全尺寸画布上提取局部窗口，其余置0)
>     11. **掩码更新**: $M \leftarrow \mathcal{W}$
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
>     10. **特征提取**: $z_{1:T} \leftarrow E_s(x_{1:T})$ (在线无梯度提取)
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
>     *   **反归一化**: $\hat{U}_{\text{orig}} \leftarrow \hat{U} \cdot \sigma_{\text{stat}} + \mu_{\text{stat}}$ (还原至物理值域)
>     *   **重投影**: $\hat{y}_{\text{proj}}, \_ \leftarrow H(\hat{U}_{\text{orig}}; \Theta)$ (将预测结果投影回观测空间)
>
> 4.  **损失计算 (Loss Calculation)**
>     *   **数据保真项**: $\mathcal{L}_{\text{rec}} \leftarrow || \hat{U} - U ||_2^2$ (z-score域)
>     *   **谱一致性项**: $\mathcal{L}_{\text{spec}} \leftarrow || \text{FFT}(\hat{U}_{\text{orig}}) \cdot W_{\text{low}} - \text{FFT}(U_{\text{orig}}) \cdot W_{\text{low}} ||_2^2$
>     *   **观测一致性项**: $\mathcal{L}_{\text{dc}} \leftarrow || \hat{y}_{\text{proj}} - y_{\text{obs}} ||_2^2$ (核心约束)
>     *   **总损失**: $\mathcal{L}_{\text{total}} \leftarrow \mathcal{L}_{\text{rec}} + \lambda_s \mathcal{L}_{\text{spec}} + \lambda_{dc} \mathcal{L}_{\text{dc}}$
>
> 5.  **优化 (Optimization)**
>     *   计算梯度 $\nabla_{\theta} \mathcal{L}_{\text{total}}$
>     *   更新参数 $\theta \leftarrow \text{Optimizer}(\theta, \nabla_{\theta} \mathcal{L}_{\text{total}})$

---

## 算法 4：带计划采样的多步自回归预测 (Autoregressive Rollout with Scheduled Sampling)

**对应章节**：3.3 时空融合与解码器 & 3.4 训练策略  
**核心逻辑**：在多步时间序列预测中，根据训练进程动态调整使用真实值（Teacher Forcing）还是模型自身预测值（Rollout）作为下一步的输入，以缓解误差累积。

> **Algorithm 4: Autoregressive Rollout with Scheduled Sampling**
>
> **输入 (Input)**: 
> *   初始输入帧 $x_{t_0} \in \mathbb{R}^{B \times C \times H \times W}$
> *   输出时间步数 $T_{out}$
> *   真实未来序列 $Y_{true} = \{y_1, y_2, ..., y_{T_{out}}\}$ (仅训练期可用)
> *   单帧预测模型 $\mathcal{M}_{\theta}$, 当前计划采样概率 $P_{sample} \in [0, 1]$
> *   状态标志 $\text{Train\_Mode}$
>
> **输出 (Output)**: 
> *   预测序列 $\hat{Y}_{seq} \in \mathbb{R}^{B \times T_{out} \times C \times H \times W}$
>
> **过程 (Process)**:
> 1.  **初始化**: $\hat{Y}_{seq} \leftarrow \text{Empty Tensor}$, 当前输入 $x_{curr} \leftarrow x_{t_0}$
> 2.  **For** $t = 1$ **to** $T_{out}$ **Do**
> 3.      **单步预测**: $\hat{y}_t \leftarrow \mathcal{M}_{\theta}(x_{curr})$
> 4.      **记录输出**: $\hat{Y}_{seq}[:, t, \dots] \leftarrow \hat{y}_t$
> 5.      **If** 不在最后一步 ($t < T_{out}$) **Then**
> 6.          **If** $\text{Train\_Mode}$ is **True** **Then**
> 7.              **采样**: 生成均匀分布随机数 $r \sim U(0, 1)$
> 8.              **If** $r < P_{sample}$ **Then**
> 9.                  $x_{curr} \leftarrow \text{Detach}(\hat{y}_t)$  *(使用模型预测)*
> 10.             **Else**
> 11.                 $x_{curr} \leftarrow Y_{true}[:, t, \dots]$  *(使用教师强制真值)*
> 12.             **End If**
> 13.         **Else**  *(推理模式/无真值)*
> 14.             $x_{curr} \leftarrow \hat{y}_t$  *(纯自回归)*
> 15.         **End If**
> 16.     **End If**
> 17. **End For**
> 18. **Return** $\hat{Y}_{seq}$

---

## 算法 5：物理启发的频域谱损失计算 (Physics-Informed Spectral Loss Computation)

**对应章节**：3.5 损失函数设计  
**核心逻辑**：针对非周期性物理场，先进行对称镜像延拓避免边界截断效应，随后通过 2D-FFT 提取核心低频动力学模态，并计算相对谱误差，引导模型学习全局物理守恒特性。

> **Algorithm 5: Physics-Informed Spectral Loss ($\mathcal{L}_{\text{spec}}$)**
>
> **输入 (Input)**: 
> *   预测场 $\hat{U} \in \mathbb{R}^{B \times C \times H \times W}$ (已反归一化至物理值域)
> *   真实场 $U \in \mathbb{R}^{B \times C \times H \times W}$ (物理值域)
> *   低频截断阈值 $k_{max}$ (默认为 16)
> *   边界模式 $\text{boundary\_mode}$ (如 'mirror')
>
> **输出 (Output)**: 
> *   标量损失值 $\mathcal{L}_{\text{spec}}$
>
> **过程 (Process)**:
> 1.  **边界处理 (Boundary Extension)**:
> 2.  **If** $\text{boundary\_mode}$ == 'mirror' **Then**
> 3.      $\hat{U}_{ext} \leftarrow \text{MirrorPad}(\hat{U})$, $U_{ext} \leftarrow \text{MirrorPad}(U)$
> 4.  **Else**
> 5.      $\hat{U}_{ext} \leftarrow \hat{U}$, $U_{ext} \leftarrow U$  *(适用于周期性边界)*
> 6.  **End If**
> 7.
> 8.  **傅里叶变换 (2D-FFT)**:
> 9.  $\mathcal{F}_{\hat{U}} \leftarrow \text{FFT2D}(\hat{U}_{ext})$, $\mathcal{F}_{U} \leftarrow \text{FFT2D}(U_{ext})$
> 10.
> 11. **低频模态截取 (Low-Frequency Mode Extraction)**:
> 12. $\mathcal{F}_{\hat{U}}^{\text{low}} \leftarrow \mathcal{F}_{\hat{U}}[..., :k_{max}, :k_{max}]$
> 13. $\mathcal{F}_{U}^{\text{low}} \leftarrow \mathcal{F}_{U}[..., :k_{max}, :k_{max}]$
> 14.
> 15. **计算复数域相对误差 (Compute Relative L2 in Complex Domain)**:
> 16. $\Delta_{real} \leftarrow \text{Re}(\mathcal{F}_{\hat{U}}^{\text{low}}) - \text{Re}(\mathcal{F}_{U}^{\text{low}})$
> 17. $\Delta_{imag} \leftarrow \text{Im}(\mathcal{F}_{\hat{U}}^{\text{low}}) - \text{Im}(\mathcal{F}_{U}^{\text{low}})$
> 18. $\text{Diff}_{sq} \leftarrow \sum (\Delta_{real}^2 + \Delta_{imag}^2)$
> 19. $\text{Target}_{sq} \leftarrow \sum (\text{Re}(\mathcal{F}_{U}^{\text{low}})^2 + \text{Im}(\mathcal{F}_{U}^{\text{low}})^2) + \epsilon$
> 20. $\mathcal{L}_{\text{spec}} \leftarrow \frac{\text{Diff}_{sq}}{\text{Target}_{sq}}$
> 21.
> 22. **Return** $\mathcal{L}_{\text{spec}}$
