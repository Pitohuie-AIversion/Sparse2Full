# 符号说明表 (Notation Table)

为确保论文叙述的严谨性与一致性，本文所使用的主要数学符号及其物理含义约定如下。除特殊说明外，全文遵循此表定义。  
**维度约定**：除另有说明外，张量统一采用维度顺序 **(B, T, C, H, W)**，分别表示 batch、时间长度、通道数、空间高度与宽度。

| 符号 (Symbol) | 类型 | 含义与说明 (Description) |
| :--- | :--- | :--- |
| **基础变量** |  |  |
| $u(\mathbf{x}, t)$ | 连续场 | 真实的物理场函数，定义在时空域 $\Omega \times [0, T]$ 上 |
| $u_t$ | 离散场 | 第 $t$ 帧高分辨率物理场，$u_t:\Omega_h\rightarrow\mathbb{R}^{C}$ |
| $u$ | 张量 | 真实场的高分辨率离散表示（原值域），通常维度为 $B \times T \times C \times H \times W$ |
| $u^{(z)}$ | 张量 | 真实场的 z-score 标准化表示：$u^{(z)}=(u-\mu_z)/\sigma_z$ |
| $y_t$ | 张量 | 第 $t$ 帧观测数据：$y_t = H(u_t) + n_t$（注意：$H$ 为确定性观测过程，噪声 $n_t$ 单独加性注入） |
| $y$ | 张量 | 观测序列张量表示，$y=\{y_t\}_{t=1}^{T}$；其分辨率由观测口径决定（SR / Crop） |
| $\hat{u}^{(z)}$ | 张量 | 模型直接输出的预测场（z-score 域） |
| $\tilde{u}$ | 张量 | 模型重建的预测场（原值域），由反标准化得到：$\tilde{u}=\hat{u}^{(z)}\odot\sigma_z+\mu_z$ |
| $m_t$ | 张量 | 第 $t$ 帧观测掩码（mask），指示观测位置或缺失区域；与 $y_t$ 的几何口径一致（SR/Crop 时需同步更新） |
| $m$ | 张量 | 掩码序列张量表示，$m=\{m_t\}_{t=1}^{T}$ |
| $\mathbf{x}$ | 向量 | 空间坐标向量，$\mathbf{x} \in \mathbb{R}^d$（本文 $d=2$） |
| $t$ | 索引/标量 | 时间索引或时间变量；离散序列中 $t\in\{1,\dots,T\}$ |
| $\Omega$ | 集合 | 连续空间定义域 |
| $\Omega_h$ | 网格 | 离散网格空间，分辨率 $N_x\times N_y$（或 $H\times W$） |
| $n_t$ | 张量 | 观测噪声项，常用假设 $n_t\sim\mathcal{N}(0,\sigma_n^2)$（可按通道扩展） |
| $\sigma_n$ | 标量 | 观测噪声标准差 |
| $\sigma_{\mathrm{blur}}$ | 标量 | 观测算子中的高斯抗混叠预滤波标准差（SR 口径） |
| $\mu_z,\sigma_z$ | 向量 | 数据预处理中的逐通道均值与标准差（z-score），用于标准化与反标准化 |
| $p$ | 参数/向量 | 辅助输入（如显式坐标编码、Fourier 特征编码参数等） |
| **算子与映射** |  |  |
| $H(\cdot)$ | 算子 | **观测算子 (Observation Operator)**：确定性观测过程，从高分辨率原值域场映射到观测数据（如抗混叠预滤、下采样、裁剪、对齐、边界处理等）；噪声不属于 $H$ |
| $H_{\mathrm{SR}}(\cdot)$ | 算子 | SR 口径观测算子：$H_{\mathrm{SR}}(u_t)=D_s(G_{\sigma_{\mathrm{blur}}}\ast u_t)$ |
| $H_{\mathrm{Crop}}(\cdot)$ | 算子 | Crop 口径观测算子：$H_{\mathrm{Crop}}(u_t)=C_{h_c,w_c}(u_t)$ |
| $DC(\cdot)$ | 算子 | **训练退化算子 (Training Degradation Operator)**：训练阶段用于合成输入/一致性损失计算；本文硬约束 $DC \equiv H$（同参数、同实现、同边界与对齐策略） |
| $G_{\sigma_{\mathrm{blur}}}$ | 算子 | 高斯低通滤波核（抗混叠预滤波），参数为 $\sigma_{\mathrm{blur}}$ 与核大小 $k$ |
| $D_s(\cdot)$ | 算子 | 下采样算子，降采样倍率为 $s$（参考实现可采用面积型重采样；边界与对齐规则需固化） |
| $C_{h_c,w_c}(\cdot)$ | 算子 | 裁剪算子，输出窗口大小为 $h_c\times w_c$（通常中心对齐，并显式规定像素网格对应关系） |
| $\Phi_{\omega}(\cdot)$ | 映射 | 端到端重建映射（可学习参数 $\omega$），输入为 $(y_{1:T},m_{1:T},p)$，输出为 $\tilde{u}_{1:T}$ |
| $f_{\theta}(\cdot)$ | 函数 | 神经网络重建模型（参数 $\theta$）；常用于输出 $\hat{u}^{(z)}$，经反标准化得到 $\tilde{u}$（全文建议与 $\Phi_{\omega}$ 二选一统一记号） |
| $\mathcal{F}(\cdot)$ | 变换 | 傅里叶变换（Fourier Transform），用于频域分析与谱损失计算 |
| $\ast$ | 运算 | 卷积运算 |
| **损失函数** |  |  |
| $\mathcal{L}$ / $\mathcal{L}_{\mathrm{total}}$ | 标量 | 总损失函数，用于模型反向传播优化 |
| $\mathcal{L}_{\mathrm{rec}}$ | 标量 | **重建损失 (Reconstruction Loss)**：衡量 $\hat{u}^{(z)}$ 与 $u^{(z)}$ 的逐点逼近（常用 $L_2$ 或 $L_1$） |
| $\mathcal{L}_{\mathrm{spec}}$ | 标量 | **低频谱一致性损失 (Spectral Consistency Loss)**：衡量预测与真值在频域低频集合 $\mathcal{K}_{\mathrm{low}}$ 上的差异 |
| $\mathcal{L}_{\mathrm{dc}}$ | 标量 | **观测一致性损失 (Observation Consistency Loss)**：$\mathcal{L}_{\mathrm{dc}}=\|H(\tilde{u})-y\|_2^2$（原值域） |
| $\lambda_{\mathrm{spec}},\lambda_{\mathrm{dc}}$ | 标量 | 损失函数权重超参数 |
| $\mathcal{L}_{\mathrm{deriv}}$ | 标量 | 时序导数一致性正则项（用于长时预测，约束时序变化率一致性） |
| $\mathcal{L}_{\mathrm{energy}}$ | 标量 | 能量一致性正则项（用于长时预测，约束系统能量演化一致性） |
| **评价指标** |  |  |
| $\mathrm{Rel\text{-}L2}$ | 标量 | 相对 $L_2$ 误差：$\frac{\|\tilde{u}-u\|_2}{\|u\|_2}$ |
| $H_{\mathrm{err}}$ | 标量 | **观测口径误差 (Observation Consistency Error)**：$\|H(\tilde{u})-y\|_2$ |
| $\mathrm{fRMSE}$ | 标量 | 频域均方根误差（Frequency RMSE），可按频段（Low/Mid/High）统计 |
| $\mathrm{bRMSE}$ | 标量 | 边界均方根误差（Boundary RMSE），衡量非周期边界附近伪影程度 |
| **集合与频段** |  |  |
| $\mathcal{D}_{\mathrm{train}}$ | 集合 | 训练数据集 |
| $\mathcal{D}_{\mathrm{val}},\mathcal{D}_{\mathrm{test}}$ | 集合 | 验证集与测试集 |
| $(k_x,k_y)$ | 索引 | 2D 频域索引（离散频率坐标） |
| $\rho$ | 标量 | 径向频率：$\rho=\sqrt{k_x^2+k_y^2}$（用于频段划分） |
| $\mathcal{K}_{\mathrm{low}}$ | 集合 | 低频索引集合（示例）：$\{(k_x,k_y): \rho\le K_1\}$ |
| $\mathcal{K}_{\mathrm{mid}}$ | 集合 | 中频索引集合（示例）：$\{(k_x,k_y): K_1<\rho\le K_2\}$ |
| $\mathcal{K}_{\mathrm{high}}$ | 集合 | 高频索引集合（示例）：$\{(k_x,k_y): \rho> K_2\}$ |

---

## 缩略语表 (Abbreviations)

| 缩略语 | 全称 | 中文含义 |
| :--- | :--- | :--- |
| PDE | Partial Differential Equation | 偏微分方程 |
| CFD | Computational Fluid Dynamics | 计算流体力学 |
| DNS | Direct Numerical Simulation | 直接数值模拟 |
| SciML | Scientific Machine Learning | 科学机器学习 |
| PINN | Physics-Informed Neural Network | 物理信息神经网络 |
| NO | Neural Operator | 神经算子 |
| FNO | Fourier Neural Operator | 傅里叶神经算子 |
| DeepONet | Deep Operator Network | 深度算子网络 |
| ViT | Vision Transformer | 视觉 Transformer |
| Swin | Shifted Window Transformer | 移动窗口 Transformer |
| SR | Super-Resolution | 超分辨率 |
| AR | Auto-Regressive | 自回归 |
| FFT | Fast Fourier Transform | 快速傅里叶变换 |
| UOO | Unified Observation Operator | 统一观测算子 |
