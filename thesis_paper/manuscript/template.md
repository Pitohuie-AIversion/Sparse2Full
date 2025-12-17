# 稀疏观测驱动的时空流场重建方法研究（论文模板）

> 按 `thesis_paper/drafts/outline.md` 的章节结构，并对齐项目规则（H/DC 一致性、可复现、指标与资源、paper_package）。

## 摘要
- 研究背景与意义（2-3 段）
- 主要贡献（列点，3-5 点，避免与相关工作混淆）
- 关键结果（含指标与显著性，简述资源成本）
- 结论与展望（1 段）

## 第1章 绪论
### 1.1 研究背景
- 流场重建的重要性；应用场景（CFD、天气、海洋等）
- 稀疏观测的挑战（采样约束、噪声、时空不齐性）

### 1.2 研究意义
- 理论意义与方法学价值
- 实际应用价值与潜在影响

### 1.3 研究内容与贡献
- 主要研究内容概述
- 创新点列表（与摘要一致但更详尽）
- 论文组织结构说明

## 第2章 相关工作
### 2.1 流场重建方法综述
- 物理/数据驱动/混合方法的对比

> 文献综述与引证（APA 示例）：
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Li, Z., Kovachki, N. B., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. In ICLR 2021. https://openreview.net/forum?id=c8P9NQVtmnO
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), 218–229. https://doi.org/10.1038/s42256-021-00302-5
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). PDEBench: An extensive benchmark for scientific machine learning. NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2210.07182
- OpenCV documentation and community guidance on anti-aliasing downsampling (INTER_AREA + Gaussian prefilter). See e.g., OpenCV Q&A. https://answers.opencv.org/question/222322/how-to-choose-sigmakernel-size-when-downsampling-image/

### 2.2 稀疏观测重建技术
- 压缩感知、深度学习、时空建模方法

> 方法比较要点：
- PINN 强化物理约束，适于数据匮乏但在高维与复杂边界时训练困难（收敛与优化挑战；Raissi et al., 2019）。
- FNO 通过谱域核高效学习解算算子，适合参数化 PDE 的快速外推，但对网格与边界条件的外部分布敏感（Li et al., 2021）。
- DeepONet 基于算子近似理论，分支/主干结构学习函数到函数的映射，具备强表达力但训练稳定性与数据设计重要（Lu et al., 2021）。
- PDEBench 提供多类 PDE、充足规模与公平对比基准，便于统一评测与复现实验（Takamoto et al., 2022）。

### 2.3 时空耦合建模
- 统一表示、动态系统、尺度机制

> 观点异同与矛盾：
- PINN 与算子学习（FNO/DeepONet）在物理约束显式程度与数据需求上存在差异：前者以残差损失显式嵌入 PDE；后者以数据驱动的算子近似为主。
- 在稀疏观测场景，显式一致性（观测算子 H 与训练 DC 复用）常被忽视，导致域外性能与可复现性差；我们的规划将其作为核心约束。

### 2.4 现有方法的不足
- 观测一致性缺失、耦合不足、可解释性差

> 我们的见解：
- 建议以统一的观测口径（GaussianBlur + INTER_AREA / Crop 对齐策略）定义 H，并在训练端复用同一 DC，实现“口径一致性→统计可比性→复现可靠性”。
- 采用统一指标集与资源成本报告（见第6章），并以 ≥3 种子显著性检验约束结论稳健性。

## 第3章 方法论
### 3.1 问题定义
- 数学建模（变量、域、边界）与优化目标、约束条件

#### 数学形式化
- 定义时空域：\(\Omega\subset\mathbb{R}^2\), \(t\in[0,T]\)。目标是从稀疏观测恢复场函数 \(u(t,\mathbf{x})\)。
- PDE 约束：存在算子 \(\mathcal{F}\) 与参数 \(\boldsymbol{\theta}\)，满足
  \[ \mathcal{F}(u;\boldsymbol{\theta}) = 0,\quad \mathbf{x}\in\Omega,\ t\in[0,T] \]
  并满足初始/边界条件 \(\mathcal{B}(u)=0\)。
- 观测模型：给定观测算子 \(H\)，对空间超分辨任务（SR）与裁剪任务（Crop）：
  \[ y_t^{\text{SR}} = D_s\big( (k_{\sigma} * u_t) \big) + n,\quad y_t^{\text{Crop}} = C_{h_c,w_c}(u_t) \]
  其中 \(k_{\sigma}\) 为高斯核（\(k=5\)），\(D_s\) 为 `INTER_AREA` 下采样因子 \(s\)，\(C_{h_c,w_c}\) 为中心对齐裁剪算子，\(n\) 为噪声。

#### 优化目标（统一损失）
令模型预测 \(\hat{u}\) 在 z-score 域；反归一化到原值域 \(\tilde{u}=\sigma\hat{u}+\mu\)。损失由三部分组成：
- 重建损失：\( L_{\text{rec}} = \|\hat{u} - u\|_2^2 \)
- 频谱低频损失（仅 \(k_x,k_y\le 16\)）：
  \[ L_{\text{spec}} = \sum_{k_x,k_y\in \mathcal{K}_{\text{low}}} \big\| \mathcal{F}_{2\text{D}}(\hat{u})_{k_x,k_y} - \mathcal{F}_{2\text{D}}(u)_{k_x,k_y} \big\|_2^2 \]
- 一致性损失（原值域）：\( L_{\text{dc}} = \| H(\tilde{u}) - y \|_2^2 \)
总损失：\( L = L_{\text{rec}} + \lambda_s L_{\text{spec}} + \lambda_{dc} L_{\text{dc}} \)（默认权重 1.0/0.5/1.0）。

### 3.2 观测算子设计
- 稀疏观测建模与退化算子构造
- 明确核/σ/插值/对齐/边界；与训练 DC 完全复用

#### 观测口径
- SR：`GaussianBlur(σ,k=5)+INTER_AREA downsample×s`，遵循抗混叠原则；\(\sigma\) 与 \(s\) 的关系按Nyquist近似与经验规则设定。
- Crop：窗口 \((h_c,w_c)\) 与中心对齐且为 patch_size 的倍数；边界策略明示（mirror/zero/wrap）。
- 训练时 DC 与数据观测 H 完全复用同一实现与配置，确保口径一致性与统计可比性。

### 3.3 时空耦合架构
- 网络结构与特征提取、信息融合策略
- 统一接口：`forward(x[B,C_in,H,W])→y[B,C_out,H,W]`；输入包含 `[baseline, coords, mask, (fourier_pe?)]`

#### 编码与解码
- 编码：多尺度卷积/注意力/频域块提取空间特征；时间维以时序模块或显式坐标编码融合。
- 解码：优先“双线性 + 3×3”减少棋盘格；输出在 z-score 域。
- 可选：FNO/DeepONet 算子层用于全局泛化；PINN 残差项用于显式条件化物理约束。

### 3.4 损失函数设计
- 重建损失 + 频谱损失 + 一致性损失（默认权重 1.0/0.5/1.0）
- 频域仅低频模（kx=ky=16）；DC 与频域损失在原值域计算

#### 训练确定性与复现
- 统一随机种子与精度策略；同一 YAML+种子验证指标方差 ≤ \(10^{-4}\)。
- 训练开始写入配置快照 `runs/<exp>/config_merged.yaml` 与环境指纹 `runs/<exp>/env_fingerprint.json`。

### 3.5 变量与符号说明表

| 符号 | 含义 |
|---|---|
| \(\Omega\), \(T\) | 空间域与时间跨度 |
| \(u, \hat{u}, \tilde{u}\) | 真值、z-score 预测、原值域预测 |
| \(\mu, \sigma\) | 逐通道均值与标准差（z-score） |
| \(\mathcal{F}, \boldsymbol{\theta}\) | PDE 算子与参数 |
| \(H, \text{DC}\) | 观测算子与训练退化一致性算子 |
| \(k_{\sigma}, D_s\) | 高斯核与 `INTER_AREA` 下采样 |
| \(C_{h_c,w_c}\) | 居中对齐裁剪算子 |
| \(\lambda_s, \lambda_{dc}\) | 频谱/一致性损失权重 |
| \(\mathcal{K}_{\text{low}}\) | 低频索引集合（\(k_x,k_y\le 16\)） |
| metrics | `Rel-L2, MAE, PSNR, SSIM, fRMSE-low/mid/high, bRMSE, cRMSE, ||H(ŷ)−y||` |

### 3.6 研究设计与实验流程

- 数据与切分：采用 PDEBench（NeurIPS 2022）；固定 `splits/{train,val,test}.txt`；逐通道 z-score 标准化并产出 `norm_stat.npz`。
- 观测生成：由单一入口实现的 \(H\)；训练 \(\text{DC}\) 复用同一实例；通过 `tools/check_dc_equivalence.py` 抽样≥100 验证 \(\text{MSE}(H(GT), y) < 10^{-8}\)。
- 训练策略：AdamW(lr=1e-3, wd=1e-4)、Cosine+1k warmup、AMP、梯度裁剪 1.0；SR 课程：×2→×4；Crop 课程：窗 40%→20%。
- 评测统计：≥3 种子报告 `均值±标准差`；对主基线进行 paired t-test（Rel-L2）与 Cohen’s d；资源成本四项记录。
- 敏感性分析：\(\sigma,s,(h_c,w_c)\) 与边界策略；低频阈值 \(k_x,k_y\)；随机种子与网格分辨率；对口径不一致进行对比。
- 可视化：GT/Pred/Err（统一色标）、功率谱（log）、边界带局部放大；失败案例分类与建议。

#### 具体步骤
- 步骤1 数据准备：下载并校验 PDEBench 与数据卡，生成 `splits` 与 `norm_stat.npz`。
- 步骤2 配置观测：设定 \(H\) 参数（\(\sigma,k=5,s,(h_c,w_c)\) 与边界策略），绑定到训练 \(\text{DC}\)。
- 步骤3 训练：固定 YAML 与种子，记录配置快照与环境指纹。
- 步骤4 评测：统一指标与资源统计；生成 `paper_package/metrics/` 与可视化图组。
- 步骤5 显著性与敏感性：运行统计脚本与参数扫描，形成报告。

### 3.7 假设与检验

- H1：复用一致的 \(H/\text{DC}\) 能显著提升域外泛化与复现稳定性（Rel-L2 显著下降）。
- H2：低频谱一致性损失可降低大尺度误差并改善 `||H(ŷ)−y||` 同步下降。
- 检验：paired t-test（Rel-L2）与 Cohen’s d；敏感性分析覆盖 \(\sigma,s,(h_c,w_c),k_x,k_y\)。

### 3.8 方法论引用（APA）

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Li, Z., Kovachki, N. B., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). ICLR 2021. https://openreview.net/forum?id=c8P9NQVtmnO
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Nature Machine Intelligence, 3(3), 218–229. https://doi.org/10.1038/s42256-021-00302-5
- Wang, S., Yu, X., & Perdikaris, P. (2022). Journal of Computational Physics, 449, 110768. https://doi.org/10.1016/j.jcp.2022.110768
- Takamoto, M., Praditia, T., Leiteritz, R., et al. (2022). NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2210.07182
- Bartolucci, F., de Bézenac, E., Raonić, B., Molinaro, R., Mishra, S., & Alaifari, R. (2023). NeurIPS 2023. https://arxiv.org/abs/2305.19913
- Kovachki, N. B., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A. M., & Anandkumar, A. (2023). Journal of Machine Learning Research, 24, 1–97. https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf
- Acta Numerica (2023). Numerical analysis of PINNs and related models. https://www.cambridge.org/…/numerical-analysis-of-physicsinformed-neural-networks-and-related-models-in-physicsinformed-machine-learning
- SIAM Journal on Scientific Computing (2024). Operator Learning Using Random Features. https://doi.org/10.1137/24M1648703
- Science Advances (2021). Physics-informed DeepONet. https://doi.org/10.1126/sciadv.abi8605
- CMAME (2022). PhyCRNet: Physics-informed convolutional-recurrent network. https://doi.org/10.1016/j.cma.2022.114399
- OpenCV Anti-aliasing Guidance. https://answers.opencv.org/question/222322/how-to-choose-sigmakernel-size-when-downsampling-image/

## 第4章 理论分析
### 4.1 收敛性分析
- 算法收敛性、复杂度、稳定性

### 4.2 泛化能力分析
- 误差界与适用性、边界条件

### 4.3 观测一致性理论
- 一致性条件、误差界与重建保证

## 第5章 算法设计
### 5.1 整体算法流程
- 训练与推理流程、优化策略（AdamW、Cosine+Warmup、AMP、梯度裁剪）

### 5.2 关键组件实现
- 观测算子、网络细节、损失计算的实现要点

### 5.3 超参数设置
- 网络/训练/正则化参数；Hydra YAML 分层配置

## 第6章 实验结果与分析
### 6.1 实验设置
- 数据集介绍；固定切分 `splits/{train,val,test}.txt`
- 标准化：逐通道 z-score，产出并引用 `norm_stat.npz`
- 观测生成：与训练 DC 完全一致的观测算子 H（核/σ/插值/对齐/边界）
- 确定性：统一随机种子与精度策略，方差 ≤ 1e-4
- 配置快照：`runs/<exp>/config_merged.yaml`；环境指纹：`runs/<exp>/env_fingerprint.json`

### 6.2 主实验结果
- 指标：Rel-L2、MAE、PSNR、SSIM、fRMSE-low/mid/high、bRMSE、cRMSE、`||H(ŷ)−y||`
- 统计：≥3 种子均值±标准差；显著性检验（paired t-test + Cohen’s d）
- 资源：Params(M)、FLOPs(G@256²)、显存峰值(GB)、推理延迟(ms)

### 6.3 消融实验
- 组件贡献与敏感性；DC/H 一致性影响（`tools/check_dc_equivalence.py`）

### 6.4 可视化分析
- GT/Pred/Err（统一色标）、功率谱（log）、边界带局部放大
- 失败案例分类与改进建议

## 第7章 理论验证
### 7.1 一致性验证
- 观测一致性、误差与收敛性验证

### 7.2 泛化能力验证
- 跨数据集/任务、鲁棒性分析

### 7.3 计算效率分析
- 时间/空间复杂度与实际效率

## 第8章 讨论
### 8.1 方法优势
### 8.2 局限性分析
### 8.3 未来工作

> 未来方向建议：
- 稀疏时空采样自适应设计与主动学习；
- 频域低/中/高频分层一致性损失的权衡与自适应；
- 面向复杂边界与非周期条件的算子外推鲁棒性；
- `paper_package/` 的盲审化与跨环境复现实验工具链完善。

## 第9章 结论
### 9.1 研究总结
### 9.2 创新点回顾
### 9.3 展望

## 参考文献

## 附录
### A. 数学推导
### B. 实验细节
### C. 代码说明
### D. 数据集详情
### E. 复现材料包（`paper_package/`）结构与生成命令
### F. 训练环境指纹与配置快照说明

---

## 写作提示（内嵌）
- 先写方法与实验，再补理论，最后绪论与摘要/结论
- 所有指标与资源按 ≥3 种子统计并进行显著性检验
- 图表统一色标与矢量格式；在 `paper_package/figs/` 产出代表案例
- 实验口径与训练脚本一致，避免与 H/DC 配置不一致

*最后更新：2025年*
