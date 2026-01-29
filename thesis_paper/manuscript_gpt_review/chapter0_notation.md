# 符号说明表 (Notation Table)

为了确保论文叙述的严谨性与一致性，本文所使用的主要数学符号及其物理含义约定如下。除特殊说明外，全文遵循此表定义。

| 符号 (Symbol) | 类型 | 含义与说明 (Description) |
| :--- | :--- | :--- |
| **基础变量** | | |
| $u(\mathbf{x}, t)$ | 连续场 | 真实的物理场函数，定义在时空域 $\Omega \times [0, T]$ 上 |
| $u$ | 张量 | 真实场的高分辨率离散表示，通常维度为 $B \times T \times C \times H \times W$ |
| $y$ | 张量 | 稀疏观测数据，由真实场 $u$ 经观测算子作用并叠加噪声后得到 |
| $\hat{u}$ | 张量 | 模型重建的预测场，与 $u$ 具有相同的分辨率和维度 |
| $\mathbf{x}$ | 向量 | 空间坐标向量，$\mathbf{x} \in \mathbb{R}^d$ $d=2$ 或 $3$ |
| $t$ | 标量 | 时间变量，$t \in [0, T]$ |
| $n$ | 张量 | 观测噪声，通常假设服从高斯分布 $n \sim \mathcal{N}(0, \sigma_n^2)$ |
| $\sigma_{\mathrm{blur}}$ | 标量 | 观测算子 $H$ 中使用的高斯抗混叠滤波器标准差 |
| $\sigma_n$ | 标量 | 观测噪声标准差 |
| $\sigma_z$ | 向量 | 数据预处理中的逐通道 z-score 标准化标准差 |
| **算子与映射** | | |
| $H(\cdot)$ | 算子 | **观测算子 (Observation Operator)**。定义了从高分辨率真实场到稀疏观测的映射，包含降采样、裁剪、掩码及抗混叠滤波等物理过程。 |
| $DC(\cdot)$ | 算子 | **训练退化算子 (Training Degradation Operator)**。在训练阶段用于模拟观测生成的算子，本文约束 $DC \equiv H$ (同参数、同实现)。 |
| $G_{\sigma_{\mathrm{blur}}}$ | 算子 | 高斯低通滤波核，参数为 $\sigma_{\mathrm{blur}}$ 和核大小 $k$ |
| $D_s(\cdot)$ | 算子 | 下采样算子，降采样倍率为 $s$ (通常使用 `INTER_AREA` 插值) |
| $C_{h_c, w_c}(\cdot)$ | 算子 | 裁剪算子，输出窗口大小为 $h_c, w_c$ (通常使用中心对齐) |
| $f_\theta(\cdot)$ | 函数 | 深度神经网络重建模型，参数为 $\theta$，输入为观测 $y$ 及辅助信息，输出为 $\hat{u}$ |
| $\mathcal{F}(\cdot)$ | 变换 | 傅里叶变换 (Fourier Transform)，用于频域分析和谱损失计算 |
| **损失函数** | | |
| $\mathcal{L}_{total}$ | 标量 | 总损失函数，用于模型反向传播优化 |
| $\mathcal{L}_{rec}$ | 标量 | **重建损失 (Reconstruction Loss)**。通常为 $L_1$ 或 $L_2$ 范数，衡量 $\hat{u}$ 与 $u$ 的逐点逼近程度 |
| $\mathcal{L}_{spec}$ | 标量 | **谱一致性损失 (Spectral Consistency Loss)**。衡量 $\hat{u}$ 与 $u$ 在频域（特别是低频段）的幅度/相位一致性 |
| $\mathcal{L}_{dc}$ | 标量 | **观测一致性损失 (Observation Consistency Loss)**。衡量 $H(\hat{u})$ 与 $y$ 的差异，即 $H_{\mathrm{err}}$ 的平方形式 |
| $\lambda_{\mathrm{spec}}, \lambda_{\mathrm{dc}}$ | 标量 | 损失函数中的加权超参数，用于平衡各项贡献 |
| **评价指标** | | |
| $\text{Rel-}L_2$ | 标量 | 相对 $L_2$ 误差，$\frac{\| \hat{u} - u \|_2}{\| u \|_2}$，核心评价指标 |
| $H_{\mathrm{err}}$ | 标量 | **评测口径误差 (Evaluation Consistency Error)**，$\| H(\hat{u}) - y \|_2$，衡量预测结果是否符合观测约束 |
| $\text{fRMSE}$ | 标量 | 频域均方根误差 (Frequency RMSE)，可分频段 (Low/Mid/High) 统计 |
| $\text{bRMSE}$ | 标量 | 边界均方根误差 (Boundary RMSE)，衡量非周期边界附近的伪影程度 |
| **集合与空间** | | |
| $\Omega$ | 集合 | 物理场的空间定义域 |
| $\mathcal{D}_{train}$ | 集合 | 训练数据集 |
| $\mathcal{D}_{val}, \mathcal{D}_{test}$ | 集合 | 验证集与测试集 |
| $\mathcal{K}_{\mathrm{low}}$ | 集合 | 低频索引集合，通常定义为 $\{ (k_x, k_y) : k_x \le K, k_y \le K\}$ |
| $\mathcal{K}_{\mathrm{mid}}$ | 集合 | 中频索引集合，通常定义为 $\{ (k_x, k_y) : K < \rho \le K_2\}$ |
| $\mathcal{K}_{\mathrm{high}}$ | 集合 | 高频索引集合，通常定义为 $\{ (k_x, k_y) : \rho > K_2\}$ |

---
**缩略语表 (Abbreviations)**

| 缩略语 | 全称 | 中文含义 |
| :--- | :--- | :--- |
| **PDE** | Partial Differential Equation | 偏微分方程 |
| **CFD** | Computational Fluid Dynamics | 计算流体力学 |
| **DNS** | Direct Numerical Simulation | 直接数值模拟 |
| **PINN** | Physics-Informed Neural Network | 物理信息神经网络 |
| **FNO** | Fourier Neural Operator | 傅里叶神经算子 |
| **DeepONet** | Deep Operator Network | 深度算子网络 |
| **ViT** | Vision Transformer | 视觉 Transformer |
| **Swin** | Shifted Window Transformer | 移动窗口 Transformer |
| **SR** | Super-Resolution | 超分辨率 |
| **AR** | Auto-Regressive | 自回归 |
| **FFT** | Fast Fourier Transform | 快速傅里叶变换 |
