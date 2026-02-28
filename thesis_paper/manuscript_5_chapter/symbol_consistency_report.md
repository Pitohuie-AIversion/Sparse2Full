# 符号一致性检查报告 (Symbol Consistency Report)

基准文件: `chapter0_notation.md`

检查时间: Fri Feb 27 23:43:19 CST 2026

## 1. 已定义符号的使用情况 (Defined Symbols Usage)

此表列出符号说明表中的符号在各章节中是否被检测到使用。

| 符号 (Symbol) | 核心 (Core) | 说明 (Description) | 出现章节 (Found In) | 状态 (Status) |
| :--- | :--- | :--- | :--- | :--- |
| $u(\mathbf{x},t)$ | u | 真实物理场，定义在时空域 $\Omega\times[0,T]$ | Ch1, Ch2, Ch3, Ch4, Ch5, App.md, App | ✅ Used |
| $\mathbf{U}$ | \mathbf{U} | 真实场的高分辨率离散表示（原值域），维度记为 $B\times T\times C\times H\times W$ | Ch4 | ✅ Used |
| $\mathbf{U}^{(z)}$ | \mathbf{U}^{(z)} | $\mathbf{U}$ 的 z-score 标准化表示，用于训练 | - | ⚠️ **Unused** |
| $\mathbf{y}$ | \mathbf{y} | 稀疏观测数据：$\mathbf{y}=H(\mathbf{U})+\boldsymbol{\varepsilon}$ | Ch1, Ch4 | ✅ Used |
| $\tilde{\mathbf{U}}$ | \tilde{\mathbf{U}} | 模型重建的预测场（原值域），用于最终评测，与 $\mathbf{U}$ 同分辨率与维度 | Ch4 | ✅ Used |
| $\hat{\mathbf{U}}^{(z)}$ | \hat{\mathbf{U}}^{(z)} | 网络直接输出的预测场（z-score 域），反归一化后得到 $\tilde{\mathbf{U}}$ | - | ⚠️ **Unused** |
| $\mathbf{x}$ | \mathbf{x} | 空间坐标向量，$\mathbf{x}\in\mathbb{R}^d$（本文取 $d=2$） | Ch3 | ✅ Used |
| $t$ | t | 时间变量，$t\in[0,T]$ | Ch1, Ch2, Ch3, Ch4, Ch5, App.md, App | ✅ Used |
| $\boldsymbol{\varepsilon}$ | \boldsymbol{\varepsilon} | 观测噪声，常设 $\boldsymbol{\varepsilon}\sim\mathcal{N}(0,\sigma_n^2)$ | - | ⚠️ **Unused** |
| $\sigma_{\mathrm{blur}}$ | \sigma_{\mathrm{blur}} | 观测算子 $H$ 中高斯抗混叠滤波器的标准差 | - | ⚠️ **Unused** |
| $\sigma_n$ | \sigma_n | 观测噪声标准差 | Ch4 | ✅ Used |
| $\boldsymbol{\mu}_z$ | \boldsymbol{\mu}_z | 逐通道 z-score 标准化均值 | - | ⚠️ **Unused** |
| $\boldsymbol{\sigma}_z$ | \boldsymbol{\sigma}_z | 逐通道 z-score 标准化标准差 | - | ⚠️ **Unused** |
| $\odot$ | \odot | 逐元素乘（Hadamard 乘积）；反归一化：$\tilde{\mathbf{U}}=\hat{\mathbf{U}}^{(z)}\odot\boldsymbol{\sigma}_z+\boldsymbol{\mu}_z$ | App.md | ✅ Used |
| $H(\cdot)$ | H | **观测算子 (Observation Operator)**。从高分辨率原值域离散场到稀疏观测的映射，包含抗混叠、降采样、裁剪、掩码/采样等过程 | Ch1, Ch2, Ch3, Ch4, Ch5, App.md, App | ✅ Used |
| $DC(\cdot)$ | DC | **训练退化算子 (Training Degradation Operator)**。训练阶段用于模拟观测生成，本文约束 $DC\equiv H$（同参数、同实现） | Ch1, Ch2, Ch3, Ch4, Ch5, App | ✅ Used |
| $G_{\sigma_{\mathrm{blur}}}(\cdot)$ | G_{\sigma_{\mathrm{blur}}} | 高斯低通（抗混叠）滤波算子，参数为 $\sigma_{\mathrm{blur}}$ | - | ⚠️ **Unused** |
| $D_s(\cdot)$ | D_s | 下采样算子，降采样倍率为 $s$ | Ch2 | ✅ Used |
| $C_{h_c,w_c}(\cdot)$ | C_{h_c,w_c} | 裁剪算子，输出窗口大小为 $(h_c,w_c)$（常用中心对齐） | - | ⚠️ **Unused** |
| $M(\cdot)$ | M | 掩码/采样算子：将全域场映射到稀疏观测位置或执行缺失掩码 | Ch2, Ch3, Ch4, App.md | ✅ Used |
| $f_{\boldsymbol{\theta}}(\cdot)$ | f_{\boldsymbol{\theta}} | 深度神经网络重建模型，参数为 $\boldsymbol{\theta}$，输入为 $\mathbf{y}$（及辅助信息），输出为 $\hat{\mathbf{U}}^{(z)}$ | - | ⚠️ **Unused** |
| $\mathcal{F}(\cdot)$ | \mathcal{F} | 傅里叶变换 (Fourier Transform)，用于频域分析与谱损失计算 | Ch2, Ch3, App.md, App | ✅ Used |
| $\mathcal{L}_{\mathrm{total}}$ | \mathcal{L}_{\mathrm{total}} | 总损失函数，用于反向传播优化 | - | ⚠️ **Unused** |
| $\mathcal{L}_{\mathrm{rec}}$ | \mathcal{L}_{\mathrm{rec}} | **重建损失**：衡量 $\hat{\mathbf{U}}^{(z)}$ 与 $\mathbf{U}^{(z)}$ 的逐点误差（常用 $L_1/L_2$） | - | ⚠️ **Unused** |
| $\mathcal{L}_{\mathrm{spec}}$ | \mathcal{L}_{\mathrm{spec}} | **谱一致性损失**：衡量频域一致性（本文强调低频段或指定频段） | - | ⚠️ **Unused** |
| $\mathcal{L}_{\mathrm{dc}}$ | \mathcal{L}_{\mathrm{dc}} | **观测一致性损失**：$\mathcal{L}_{\mathrm{dc}}=\ | - | ⚠️ **Unused** |
| $\lambda_{\mathrm{spec}},\lambda_{\mathrm{dc}}$ | \lambda_{\mathrm{spec}},\lambda_{\mathrm{dc}} | 损失加权超参数 | - | ⚠️ **Unused** |
| $\mathrm{Rel}\text{-}L_2$ | \mathrm{Rel}\text{-}L_2 | 相对误差：$\displaystyle \frac{\ | Ch4 | ✅ Used |
| $H_{\mathrm{err}}$ | H_{\mathrm{err}} | 观测口径误差：$H_{\mathrm{err}}=\ | Ch4, App.md | ✅ Used |
| $\mathrm{fRMSE}$ | \mathrm{fRMSE} | 频域 RMSE（Frequency RMSE），可按 Low/Mid/High 频段统计 | - | ⚠️ **Unused** |
| $\Omega$ | \Omega | 空间定义域 | Ch2 | ✅ Used |
| $\mathcal{D}_{\mathrm{train}}$ | \mathcal{D}_{\mathrm{train}} | 训练数据集 | - | ⚠️ **Unused** |
| $\mathcal{D}_{\mathrm{val}},\mathcal{D}_{\mathrm{test}}$ | \mathcal{D}_{\mathrm{val}},\mathcal{D}_{\mathrm{test}} | 验证集与测试集 | - | ⚠️ **Unused** |
| $\mathcal{K}_{\mathrm{low}}$ | \mathcal{K}_{\mathrm{low}} | 低频索引集合（示例）：$\{(k_x,k_y): \rho\le K_1\}$ | - | ⚠️ **Unused** |
| $\mathcal{K}_{\mathrm{mid}}$ | \mathcal{K}_{\mathrm{mid}} | 中频索引集合（示例）：$\{(k_x,k_y): K_1<\rho\le K_2\}$ | - | ⚠️ **Unused** |
| $\mathcal{K}_{\mathrm{high}}$ | \mathcal{K}_{\mathrm{high}} | 高频索引集合（示例）：$\{(k_x,k_y): \rho> K_2\}$ | - | ⚠️ **Unused** |
| $\rho$ | \rho | 径向频率：$\rho=\sqrt{k_x^2+k_y^2}$ | - | ⚠️ **Unused** |
| $K_1,K_2$ | K_1,K_2 | 频段阈值（由本文评测设置给定） | - | ⚠️ **Unused** |
| 缩略语 | 缩略语 | 中文含义 | - | ⚠️ **Unused** |
| PDE | PDE | 偏微分方程 | App.md | ✅ Used |
| CFD | CFD | 计算流体力学 | - | ⚠️ **Unused** |
| DNS | DNS | 直接数值模拟 | - | ⚠️ **Unused** |
| PINN | PINN | 物理信息神经网络 | - | ⚠️ **Unused** |
| FNO | FNO | 傅里叶神经算子 | - | ⚠️ **Unused** |
| DeepONet | DeepONet | 深度算子网络 | - | ⚠️ **Unused** |
| ViT | ViT | 视觉 Transformer | - | ⚠️ **Unused** |
| Swin | Swin | 移动窗口 Transformer | - | ⚠️ **Unused** |
| SR | SR | 超分辨率 | - | ⚠️ **Unused** |
| AR | AR | 自回归 | - | ⚠️ **Unused** |
| FFT | FFT | 快速傅里叶变换 | - | ⚠️ **Unused** |
| SWE | SWE | 浅水方程 | - | ⚠️ **Unused** |
| DRD | DRD | 扩散–反应（数据集/动力学） | - | ⚠️ **Unused** |
| E2E | E2E | 端到端 | - | ⚠️ **Unused** |
| PSNR | PSNR | 峰值信噪比 | - | ⚠️ **Unused** |
| FLOPs | FLOPs | 浮点运算量 | - | ⚠️ **Unused** |

**统计**: 共定义 55 个符号，已使用 17 个，未使用 38 个。

## 2. 章节数学公式统计 (Math Expression Stats)

| 文件名 (Chapter) | 公式数量 (Math Count) | 已定义符号覆盖率 (Defined Coverage) |
| :--- | :--- | :--- |
| chapter1_intro_related.md | 24 | 5/55 |
| chapter2_problem_framework.md | 165 | 8/55 |
| chapter3_implementation_setup.md | 50 | 7/55 |
| chapter4_results_verification.md | 132 | 11/55 |
| chapter5_discussion_conclusion.md | 11 | 4/55 |
| appendix.md | 38 | 8/55 |
| appendix_proofs.md | 55 | 5/55 |

## 3. 潜在未定义符号 (Potential Undefined Symbols)

*(此功能为实验性功能，列出频繁出现但不在表中的单字母变量)*

> 提示：请人工核对以下章节中频繁出现的变量是否遗漏在符号表中。
