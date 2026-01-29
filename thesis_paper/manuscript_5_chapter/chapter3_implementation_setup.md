# 第5章 算法设计

## 5.0 引言

本章旨在详细阐述本文提出的稀疏观测时空场重建方法的工程化设计与实现细节。算法设计的核心目标是构建一个“可复现、可替换、可对照”的闭环系统。该系统以统一观测口径为核心约束（数据侧观测算子 $H$ 与训练侧退化算子 $DC$ 同源复用），以统一模型接口为工程规范，以确定性训练与标准化产出为复现保障。本章将围绕端到端流程、算子实现、网络架构拆解及训练策略展开论述，并给出关键工程约束与审计机制，确保方法贡献可被独立复核。

---

## 5.1 端到端流程与数据契约

### 5.1.1 数据张量与维度定义

设时空场真值序列为 $u_{1:T}$。单帧真值张量定义为 $u_t \in \mathbb{R}^{C\times H\times W}$，其中 $C$ 为通道数，$H,W$ 为空间分辨率。观测数据由任务特定的观测算子 $H$ 生成：
$$
y_t = H(u_t) + n_t,
$$
其中 $n_t$ 为噪声项。

为便于批处理与模块化替换，本系统统一采用 Batch-First（批在前）张量格式：
- 观测数据：$y \in \mathbb{R}^{B\times C_y\times H_y\times W_y}$
- 真值数据：$u \in \mathbb{R}^{B\times C\times H\times W}$

其中 $B$ 为 batch size。对不同口径任务，有：
- **超分辨（SR）**：$H_y = H/s,\; W_y = W/s$（$s$ 为降采样倍率）；
- **裁剪（Crop）**：$H_y \times W_y = h_c \times w_c$（$h_c,w_c$ 为裁剪窗口尺寸）。

---

### 5.1.2 统一模型接口设计

为确保不同骨干网络（Backbone）的可替换性与评测的一致性，本文定义标准化的模型签名。所有重建模型均继承统一基类并遵循如下接口规范：
- **初始化**：`__init__(in_ch, out_ch, img_size, **kwargs)`
- **前向传播**：`forward(x) -> u_hat_z`

输入张量 $x$ 由多源信息按通道拼接（Concatenation）构造：
$$
x = \mathrm{Concat}\big(\mathrm{baseline}(y),\, m,\, \mathrm{coords},\, \mathrm{PE}_{\mathrm{Fourier}}\big),
$$
其中：
1. `baseline`：基础插值/上采样结果（例如双线性上采样），为网络提供初始低频解；
2. $m$：观测掩码（mask），指示有效观测区域与缺失区域；
3. `coords`：归一化坐标网格（例如 $(x,y)\in[0,1]^2$），提供空间位置先验；
4. $\mathrm{PE}_{\mathrm{Fourier}}$：可选 Fourier 特征位置编码，用于增强高频表达能力。

输出 `u_hat_z` 严格定义在 z-score 标准化域，以提升数值稳定性与不同实验设置下的可比性。反标准化得到原值域预测 $\tilde u$ 用于计算观测一致性指标与一致性损失。

---

## 5.2 观测算子与退化算子的同源复用实现

### 5.2.1 单一入口原则

为消除隐性域偏差，工程实现中强制执行“单一入口（Single Entry Point）”原则：观测算子 $H$ 与训练退化算子 $DC$ 均由同一工厂函数（例如 `build_degradation(cfg)`）实例化，并共享完全相同的配置参数（Configuration）。在代码层面，禁止出现“训练端单独实现退化逻辑”的分叉路径。

---

### 5.2.2 SR 算子实现细节

超分辨观测算子 $H_{\mathrm{SR}}$ 包含高斯预滤与降采样两个步骤：
$$
y_t^{\mathrm{SR}} = D_s\!\left(G_{\sigma_{\mathrm{blur}}}\ast u_t\right) + n_t.
$$

工程实现要点如下：
- **高斯预滤**：固定核大小 $k$ 与标准差 $\sigma_{\mathrm{blur}}$，并显式指定边界填充模式（如 `reflect`）；
- **降采样**：强制使用 `INTER_AREA` 插值算法（基于 OpenCV 实现），以获得较好的缩小效果并降低混叠风险；
- **参数固化**：所有参数以 YAML 形式写入实验配置（包括 $s,k,\sigma_{\mathrm{blur}}$ 与边界模式），并随实验产出一起保存，确保可追溯。

---

### 5.2.3 Crop 算子实现细节

裁剪观测算子 $H_{\mathrm{Crop}}$ 执行中心对齐裁剪：
$$
y_t^{\mathrm{Crop}} = C_{h_c,w_c}(u_t) + n_t.
$$

工程实现要点如下：
- **对齐规则**：严格定义中心点坐标与窗口边界计算公式，确保奇/偶尺寸下行为一致；
- **尺寸约束**：裁剪窗口 $h_c,w_c$ 必须为网络 Patch Size 的整数倍，以降低 Padding/对齐误差引入的边缘伪影；
- **掩码同步**：裁剪操作必须同步更新观测掩码 $m$，保证输入观测、掩码与标签在几何口径上的一致性。

---

### 5.2.4 一致性阻断审计

在训练循环启动前，系统自动执行一致性审计脚本。随机抽取 $N$ 个样本（$N\ge 100$），验证数据管线生成的观测 $y$ 与算子直接作用生成的观测 $H(u)$ 之间的误差满足：
$$
\mathrm{MSE}\big(H(u), y\big) < 10^{-8}.
$$
若审计失败，系统将抛出异常并终止运行，同时生成差异诊断报告（记录样本索引、最大误差位置、误差统计与对应配置快照），从工程层面阻断口径不一致风险。

---

## 5.3 网络架构的模块化设计

本研究采用模块化设计，将网络解耦为编码器、算子层、时空融合模块与解码器四个部分，以支持可替换对比实验与组件级消融分析。

### 5.3.1 编码器与算子层

- **编码器（Encoder）**：从多源输入 $x$ 中提取多尺度空间特征，可选 CNN 或 Transformer 结构；
- **算子层（Operator Block）**：作为核心计算单元，在特征空间执行非局部映射。该模块设计为可插拔接口，支持 FNO、Attention/Transformer、以及其他神经算子变体的无缝切换，以便在统一输入/口径下开展横向对比。

### 5.3.2 时空融合与解码器

- **时空融合（Spatiotemporal Fusion）**：提供两条路径：
  1) 显式时序建模（如 ConvLSTM、ARWrapper 或 Transformer 时序模块），用于长时预测；  
  2) 隐式条件化（例如 Conditional Normalization），用于短时高精重建或弱时序依赖任务。
- **解码器（Decoder）**：将特征映射回物理空间。为抑制转置卷积常见的棋盘格伪影（checkerboard artifacts），本文优先采用“双线性上采样 + 卷积”的解码策略。

---

## 5.4 序列化训练策略的工程实现

为落实第 3 章提出的三阶段训练策略，`SequentialTrainer` 类以状态机方式实现如下流程：

1. **阶段一：空间预训练（Spatial Pretraining）**
   - 冻结时序模块参数（`requires_grad=False`）；
   - 数据加载器以单帧模式运行；
   - 优化目标聚焦空间重建损失：$L_{\mathrm{rec}}, L_{\mathrm{spec}}, L_{\mathrm{dc}}$。

2. **阶段二：时序预训练（Temporal Pretraining）**
   - 冻结空间编码器与解码器，解冻时序模块；
   - 启用 Teacher Forcing：输入真实历史特征序列；
   - 重点优化潜在空间/特征空间的演化轨迹，使模型学习动力学规律。

3. **阶段三：联合微调（Joint Fine-tuning）**
   - 解冻全模型参数；
   - 执行 $K$ 步自回归滚动预测（AR rollout）；
   - 引入 Teacher Forcing Decay：随 epoch 增加逐步降低真值注入比例；
   - 加入时序一致性正则化项（$L_{\mathrm{deriv}}, L_{\mathrm{energy}}$），抑制误差累积与非物理漂移。

---

## 5.5 训练配置与复现保障

### 5.5.1 优化器与学习率

- **优化器**：采用 AdamW，实现权重衰减与梯度更新的解耦；
- **学习率策略**：采用 Cosine Annealing 调度，并配合 Warmup 预热以稳定训练初期；
- **混合精度**：启用自动混合精度（AMP），在可控数值误差下提升吞吐并降低显存占用。

---

### 5.5.2 确定性与环境指纹

为满足学位论文对可复现性的要求，本文实施严格的确定性控制与环境记录：
- 固定全局随机种子（Python、NumPy、PyTorch）；
- 设置 `torch.use_deterministic_algorithms(True)`（在硬件与算子支持范围内）；
- 实验开始时自动抓取并保存环境指纹 `env_fingerprint.json`，记录 CUDA 版本、驱动信息、GPU 型号与关键依赖库版本；
- 保存完整配置快照（YAML）与训练日志（含 loss 曲线、指标曲线与关键超参），确保实验可追溯与可复核。

---

## 5.6 本章小结

本章从工程实现角度详细阐述了稀疏观测时空场重建算法的系统设计。通过 $H/DC$ 同源复用机制、模块化网络架构、序列化训练状态机以及严格的复现保障措施，本文构建了一个可审计、可扩展且可对照的算法框架。该设计为后续实验验证（第 6 章）与理论验证（第 7 章）提供了统一的工程基座。

# 第6章 实验结果与分析

> 本章在第3–5章提出的“**统一观测口径（H/DC 同源复用）+ 三件套损失（$L_{\mathrm{rec}}+\lambda_s L_{\mathrm{spec}}+\lambda_{dc}L_{\mathrm{dc}}$）+ 确定性训练闭环**”框架下，系统评估稀疏观测驱动的时空场重建性能，并从**主结果—消融—可视化—资源—统计显著性**五个层面给出可审计证据链。
> 为避免“训练口径与评测口径不一致”导致的指标断裂，本章所有实验均执行：
> $$
> \mathrm{DC}\equiv H\quad\text{（同一实现、同一参数、同一边界/插值/对齐策略）}.
> $$

承接第5章的算法设计与工程实现，本章将正式进入实验验证环节。首先，我们将详细介绍实验设置，包括数据集、基线模型与评测指标（6.1节）；随后展示主实验结果，验证所提方法在统一口径下的整体有效性（6.2节）；接着通过消融实验，逐一验证三件套损失与序列化训练策略的贡献（6.3节）；最后通过定性可视化与资源成本分析，提供更直观的性能画像（6.4-6.5节）。

---

## 6.1 实验设置

### 6.1.1 数据集与任务设定（PDEBench）

* **数据来源**：采用 PDEBench 基准数据集与其公开数据发布入口。本研究具体选用了以下两个具有代表性的 PDE 子集进行实验：
  1. **2D Diffusion-Reaction Equation (2D-Diff-React)**: 描述了扩散与化学反应的耦合过程，具有复杂的非线性动力学特征。
     - 分辨率：$128 \times 128$
     - 物理量：$u(x, y, t)$ (标量场)
     - 参数设置：扩散系数 $D \in [0.01, 0.2]$，反应速率 $k \in [0.01, 1.0]$。
  2. **2D Darcy Flow**: 描述多孔介质中的流体流动，常用于验证稳态问题的求解能力。
     - 分辨率：$128 \times 128$
     - 物理量：渗透率 $a(x, y)$ (输入) 与 压力 $u(x, y)$ (输出)
     - 边界条件：Dirichlet 边界条件。
* **数据发布与可复用性**：PDEBench 数据集以 DOI 形式发布，满足可追溯引用与可复现实验的基本条件（见参考文献）。

---

### 6.1.2 训练/验证/测试切分与标准化

* **固定切分**：使用固定文件
  `splits/{train,val,test}.txt`
  确保跨实验横向对照公平。
* **标准化（Normalization）**：采用逐通道 z-score 标准化。从训练集计算统计量，产出 `norm_stat.npz`（包含每通道 $\mu,\sigma_z$），并在训练与评测中严格复用。
    * 真值（z-score 域）：$u^{(z)}=\frac{u-\mu}{\sigma_z}$
    * 预测（原值域）：$\tilde u=\sigma_z \hat u^{(z)}+\mu$

---

### 6.1.3 观测生成与口径一致性门禁（H/DC 同源复用）

* **观测生成**：对每个样本按统一观测算子生成观测：
  $$
  y = H(u) + n.
  $$
* **训练退化**：训练侧严格使用同一算子：
  $$
  \mathrm{DC}\equiv H.
  $$
* **一致性审计（阻断式）**：训练开始前抽样 $N\ge 100$ 个样本执行：
  $$
  \mathrm{MSE}\big(H(u),,\mathrm{DC}(u)\big) < \varepsilon,\quad \varepsilon=10^{-8},
  $$
  失败则终止实验并将差异（核大小/σ/插值/边界/对齐偏移）归档至：
  `runs/<exp>/consistency_report.json`

---

### 6.1.4 观测类型与课程策略（SR / Crop）

为覆盖典型稀疏观测情形，本章采用两类任务：

* **SR (Super-Resolution)**：
  $$
  y^{\mathrm{SR}}=D_s\big(G_{\sigma_{\mathrm{blur}}}\ast u\big)+n.
  $$
* **Crop（裁剪观测）**：中心对齐裁剪并同步掩码
  $$
  y^{\mathrm{Crop}}=C_{h_c,w_c}(u)+n.
  $$

**课程学习（curriculum）**用于降低欠定程度的突变（与第4章动机一致）：

* SR：×2 → ×4（由弱欠定到强欠定）
* Crop：40% → 20% 可观测窗口（由大窗口到小窗口）

> 课程切换点必须在日志中标注，并在第6章结果表中注明“阶段 A / 阶段 B”对应区间，否则读者无法判断提升来自算法还是来自任务难度变化。

---

### 6.1.5 模型与对比方法（建议写成“方法族 + 统一接口”）

本章所有模型均遵循第5章统一接口：
$$
\texttt{forward}:\ \mathbb{R}^{B\times C_{\mathrm{in}}\times H\times W}\rightarrow
\mathbb{R}^{B\times C_{\mathrm{out}}\times H\times W}.
$$

建议将对比方法按“**口径一致**”与“**损失配置**”分组（便于讲清楚贡献来源）：

* **插值基线**：Bilinear / Bicubic（仅用于 sanity check 与可视化参照）
* **算子/网络基线**：FNO-family、DeepONet-family、Conv/UNet-family、Conv-Attn/Transformer-hybrid
* **物理基线（可选）**：PINN/残差正则（若采用，需声明方程、采样与权重）

---

### 6.1.6 评测指标（完整定义，避免“只列名词”）

本章同时报告两类误差：**重建域误差**与**观测口径误差**。

#### (1) 重建域误差

* **Rel-L2**：
  $$
  \mathrm{Rel\text{-}L2}=\frac{\|\tilde u-u\|_2}{\|u\|_2}.
  $$
* **MAE**：
  $$
  \mathrm{MAE}=\frac{1}{N}\sum_i |\tilde u_i-u_i|.
  $$
* **PSNR**（以峰值 $I_{\max}$ 定义）：
  $$
  \mathrm{PSNR}=20\log_{10}\frac{I_{\max}}{\sqrt{\mathrm{MSE}}}.
  $$
* **SSIM**：采用经典 SSIM 定义与实现（见参考文献）。

#### (2) 观测口径误差（H-一致性误差）

* **$H_{\mathrm{err}}$**（强制在原值域）：
  $$
  H_{\mathrm{err}} \triangleq \|H(\tilde u)-y\|_2.
  $$

> 说明：若在 z-score 域计算 $H_{\mathrm{err}}$，会引入尺度偏差，与第3章“口径一致性”目标冲突。

#### (3) 频域分段误差：fRMSE-low/mid/high（可复现口径）

定义二维 FFT：
$$
U=\mathcal{F}_{2\mathrm{D}}(u),\quad \tilde U=\mathcal{F}_{2\mathrm{D}}(\tilde u).
$$
定义三个互不重叠的频域掩码集合（以径向频率 $\rho=\sqrt{k_x^2+k_y^2}$ 分段）：

* $\mathcal{K}_{\mathrm{low}}:\ 0\le\rho<\rho_1$
* $\mathcal{K}_{\mathrm{mid}}:\ \rho_1\le\rho<\rho_2$
* $\mathcal{K}_{\mathrm{high}}:\ \rho_2\le\rho\le\rho_{\max}$

则分段频域 RMSE 定义为：
$$
\mathrm{fRMSE}(\mathcal{K})=
\sqrt{
\frac{1}{|\mathcal{K}|}\sum_{k\in\mathcal{K}}
\left| |\tilde U_k|-|U_k| \right|^2
}.
$$
其中幅值谱 $\|U_k\|$ 使该指标对相位误差更稳健；若你希望同时惩罚相位，可将 $\|\tilde U_k-U_k\|^2$ 作为替代口径，并在文中声明。

> **必须声明**：$\rho_1,\rho_2$ 的具体取值规则（固定索引 vs 随分辨率缩放）。建议用“按比例缩放”的径向阈值，避免跨分辨率时 low/mid/high 含义漂移。

#### (4) 区域误差：bRMSE 与 cRMSE（边界与中心）

设边界带宽为 $w_b$（像素），定义边界区域：
$$
\Omega_{\mathrm{b}}={(i,j)\mid i<w_b\ \vee\ i\ge H-w_b\ \vee\ j<w_b\ \vee\ j\ge W-w_b},
$$
中心区域 $\Omega_{\mathrm{c}}=\Omega\setminus\Omega_{\mathrm{b}}$。

则
$$
\mathrm{bRMSE}=
\sqrt{\frac{1}{|\Omega_{\mathrm{b}}|}\sum_{(i,j)\in\Omega_{\mathrm{b}}}(\tilde u_{ij}-u_{ij})^2},\quad
\mathrm{cRMSE}=
\sqrt{\frac{1}{|\Omega_{\mathrm{c}}|}\sum_{(i,j)\in\Omega_{\mathrm{c}}}(\tilde u_{ij}-u_{ij})^2}.
$$

---

### 6.1.7 统计检验与报告规范（≥3 seeds）

* **重复次数**：同一配置至少 3 个随机种子，报告均值±标准差。
* **显著性检验**：对同一测试样本集合上的 Rel-L2 序列做 **paired t-test**。
* **效应量**：报告 Cohen’s (d)（配对设计可用差值序列归一化）。

> 你需要在附录或脚本中固定：检验的样本数、显著性水平 $\alpha$、是否多重比较校正（若你同时比较很多方法，建议说明是否做 Holm–Bonferroni 等）。

---

### 6.1.8 资源四项统计（固定口径）

统一在 `img_size=256`、固定 batch、固定设备与固定预热策略下统计：

* Params（M）：可训练参数量
* FLOPs（G@256²）：固定输入尺度的 FLOPs
* 显存峰值（GB）：峰值显存占用
* 推理延迟（ms）：预热后重复计时的均值±标准差

---

