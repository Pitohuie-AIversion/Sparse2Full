/# 符号说明表 (Notation Table)

为确保论文叙述的严谨性与一致性，本文主要数学符号及其含义约定如下。除特殊说明外，全文遵循此表定义。

| 符号 (Symbol) | 类型 | 含义与说明 (Description) |
| :--- | :--- | :--- |
| **基础变量** |  |  |
| $u(\mathbf{x},t)$ | 连续场 | 真实物理场，定义在时空域 $\Omega\times[0,T]$ |
| $\mathbf{U}$ | 张量 | 真实场的高分辨率离散表示（原值域），维度记为 $B\times T\times C\times H\times W$ |
| $\mathbf{U}^{(z)}$ | 张量 | $\mathbf{U}$ 的 z-score 标准化表示，用于训练 |
| $\mathbf{y}$ | 张量 | 稀疏观测数据：$\mathbf{y}=H(\mathbf{U})+\boldsymbol{\varepsilon}$ |
| $\tilde{\mathbf{U}}$ | 张量 | 模型重建的预测场（原值域），用于最终评测，与 $\mathbf{U}$ 同分辨率与维度 |
| $\hat{\mathbf{U}}^{(z)}$ | 张量 | 网络直接输出的预测场（z-score 域），反归一化后得到 $\tilde{\mathbf{U}}$ |
| $\mathbf{x}$ | 向量 | 空间坐标向量，$\mathbf{x}\in\mathbb{R}^d$（本文取 $d=2$） |
| $t$ | 标量 | 时间变量，$t\in[0,T]$ |
| $\boldsymbol{\varepsilon}$ | 张量 | 观测噪声，常设 $\boldsymbol{\varepsilon}\sim\mathcal{N}(0,\sigma_n^2)$ |
| $\sigma_{\mathrm{blur}}$ | 标量 | 观测算子 $H$ 中高斯抗混叠滤波器的标准差 |
| $\sigma_n$ | 标量 | 观测噪声标准差 |
| $\boldsymbol{\mu}_z$ | 向量 | 逐通道 z-score 标准化均值 |
| $\boldsymbol{\sigma}_z$ | 向量 | 逐通道 z-score 标准化标准差 |
| $\odot$ | 运算 | 逐元素乘（Hadamard 乘积）；反归一化：$\tilde{\mathbf{U}}=\hat{\mathbf{U}}^{(z)}\odot\boldsymbol{\sigma}_z+\boldsymbol{\mu}_z$ |
| **算子与映射** |  |  |
| $H(\cdot)$ | 算子 | **观测算子 (Observation Operator)**。从高分辨率原值域离散场到稀疏观测的映射，包含抗混叠、降采样、裁剪、掩码/采样等过程 |
| $DC(\cdot)$ | 算子 | **训练退化算子 (Training Degradation Operator)**。训练阶段用于模拟观测生成，本文约束 $DC\equiv H$（同参数、同实现） |
| $G_{\sigma_{\mathrm{blur}}}(\cdot)$ | 算子 | 高斯低通（抗混叠）滤波算子，参数为 $\sigma_{\mathrm{blur}}$ |
| $D_s(\cdot)$ | 算子 | 下采样算子，降采样倍率为 $s$ |
| $C_{h_c,w_c}(\cdot)$ | 算子 | 裁剪算子，输出窗口大小为 $(h_c,w_c)$（常用中心对齐） |
| $M(\cdot)$ | 算子 | 掩码/采样算子：将全域场映射到稀疏观测位置或执行缺失掩码 |
| $f_{\boldsymbol{\theta}}(\cdot)$ | 函数 | 深度神经网络重建模型，参数为 $\boldsymbol{\theta}$，输入为 $\mathbf{y}$（及辅助信息），输出为 $\hat{\mathbf{U}}^{(z)}$ |
| $\mathcal{F}(\cdot)$ | 变换 | 傅里叶变换 (Fourier Transform)，用于频域分析与谱损失计算 |
| **损失函数** |  |  |
| $\mathcal{L}_{\mathrm{total}}$ | 标量 | 总损失函数，用于反向传播优化 |
| $\mathcal{L}_{\mathrm{rec}}$ | 标量 | **重建损失**：衡量 $\hat{\mathbf{U}}^{(z)}$ 与 $\mathbf{U}^{(z)}$ 的逐点误差（常用 $L_1/L_2$） |
| $\mathcal{L}_{\mathrm{spec}}$ | 标量 | **谱一致性损失**：衡量频域一致性（本文强调低频段或指定频段） |
| $\mathcal{L}_{\mathrm{dc}}$ | 标量 | **观测一致性损失**：$\mathcal{L}_{\mathrm{dc}}=\|H(\tilde{\mathbf{U}})-\mathbf{y}\|_F^2$（或其均值形式） |
| $\lambda_{\mathrm{spec}},\lambda_{\mathrm{dc}}$ | 标量 | 损失加权超参数 |
| **评价指标** |  |  |
| $\mathrm{Rel}\text{-}L_2$ | 标量 | 相对误差：$\displaystyle \frac{\|\tilde{\mathbf{U}}-\mathbf{U}\|_F}{\|\mathbf{U}\|_F}$ |
| $H_{\mathrm{err}}$ | 标量 | 观测口径误差：$H_{\mathrm{err}}=\|H(\tilde{\mathbf{U}})-\mathbf{y}\|_F$ |
| $\mathrm{fRMSE}$ | 标量 | 频域 RMSE（Frequency RMSE），可按 Low/Mid/High 频段统计 |
| **集合与空间** |  |  |
| $\Omega$ | 集合 | 空间定义域 |
| $\mathcal{D}_{\mathrm{train}}$ | 集合 | 训练数据集 |
| $\mathcal{D}_{\mathrm{val}},\mathcal{D}_{\mathrm{test}}$ | 集合 | 验证集与测试集 |
| $\mathcal{K}_{\mathrm{low}}$ | 集合 | 低频索引集合（示例）：$\{(k_x,k_y): \rho\le K_1\}$ |
| $\mathcal{K}_{\mathrm{mid}}$ | 集合 | 中频索引集合（示例）：$\{(k_x,k_y): K_1<\rho\le K_2\}$ |
| $\mathcal{K}_{\mathrm{high}}$ | 集合 | 高频索引集合（示例）：$\{(k_x,k_y): \rho> K_2\}$ |
| $\rho$ | 标量 | 径向频率：$\rho=\sqrt{k_x^2+k_y^2}$ |
| $K_1,K_2$ | 标量 | 频段阈值（由本文评测设置给定） |

---

# 缩略语表 (Abbreviations)

| 缩略语 | 全称 | 中文含义 |
| :--- | :--- | :--- |
| PDE | Partial Differential Equation | 偏微分方程 |
| CFD | Computational Fluid Dynamics | 计算流体力学 |
| DNS | Direct Numerical Simulation | 直接数值模拟 |
| PINN | Physics-Informed Neural Network | 物理信息神经网络 |
| FNO | Fourier Neural Operator | 傅里叶神经算子 |
| DeepONet | Deep Operator Network | 深度算子网络 |
| ViT | Vision Transformer | 视觉 Transformer |
| Swin | Shifted Window Transformer | 移动窗口 Transformer |
| SR | Super-Resolution | 超分辨率 |
| AR | Auto-Regressive | 自回归 |
| FFT | Fast Fourier Transform | 快速傅里叶变换 |
| SWE | Shallow Water Equations | 浅水方程 |
| DRD | Diffusion–Reaction (Dataset/Dynamics) | 扩散–反应（数据集/动力学） |
| E2E | End-to-End | 端到端 |
| PSNR | Peak Signal-to-Noise Ratio | 峰值信噪比 |
| FLOPs | Floating Point Operations | 浮点运算量 |
