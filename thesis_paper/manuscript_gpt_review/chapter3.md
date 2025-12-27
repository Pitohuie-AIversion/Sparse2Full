下面给出**对你“重整后的第3章（方法论）”的直接改写版**：统一符号、补全缺失的数学对象、把“口径一致性”从口号落到**可执行约束 + 可审计证据**，并把引用替换为**可追溯的主来源**（OpenCV 官方文档、JMLR/NMI/ArXiv、Frontiers in Psychology 等）。

为避免后续章节（第5–7章）出现颗粒度错位，改写时做了三件事：

- **符号对齐**：把“算子 (H)”与“图像高 (H)”彻底拆开（用 (N_x,N_y) 表示空间分辨率；(\sigma_{\text{blur}}) 表示模糊核尺度；(\sigma_{\text{z}}) 表示 z-score 标准差）。
- **口径声明模板化**：把 SR/Crop 的“口径参数表”写成论文可直接引用、YAML 可直接落地的条目。
- **审计闭环前置**：把 H/DC 等价性检验、配置快照、环境指纹、统计协议放进方法章，而不是留给实验章“口头承诺”。

------

# 第3章 方法论（重整结构、补强论证与可追溯引用｜修订稿）

本章面向“稀疏观测驱动的时空流场重建”任务，给出可复现、可审计、可工程落地的方法学框架：**问题形式化—观测口径定义—H/DC 单一口径复用—时空耦合模型接口—三件套损失—确定性训练与统计检验—敏感性与失败模式**。方法设计强调两类“口径一致性”硬约束：

1. **采样/退化口径一致**：数据侧观测算子 (H) 与训练侧退化算子 (DC) **复用同一实现与参数**；
2. **频域/空间域口径一致**：在空间域重建误差之外，引入低频谱一致性与原值域观测一致性，以减少“指标断裂”（训练域达标、评测口径失真）。

------

## 3.1 问题定义与数学形式化

设连续空间域 (\Omega\subset\mathbb{R}^2)，离散网格为 (\Omega_h)，空间分辨率记为 (N_x\times N_y)，离散时间索引 (t\in{1,\dots,T})。目标场为标量或多通道场
[
u_t:\Omega_h\rightarrow \mathbb{R}^{C},\qquad u_{1:T}={u_t}_{t=1}^{T}.
]

给定观测算子 (H)（包含滤波、采样/下采样、裁剪、边界处理与插值等）与噪声 (n_t)，观测为
[
y_t = H(u_t)+n_t.
]
学习映射 (\Phi_{\omega})，以观测序列与辅助信息恢复全场：
[
\hat{u}*{1:T}=\Phi*{\omega}\big(y_{1:T};,m_{1:T};,p\big),
]
其中 (m_t) 为观测掩码（稀疏口位置指示或缺失区域标注），(p) 为显式坐标/位置编码（可含 Fourier 特征，见 3.4–3.5）。

为避免“数学域优化有效，但评测口径失真”，本文同时报告两类误差：

- **重建域误差**：Rel-L2、MAE、PSNR、SSIM 等；
- **观测口径误差**（H-一致性误差）：
  [
  H_{\text{err}} \triangleq \big| H(\tilde{u})-y \big|_2,
  ]
  其中 (\tilde{u}) 为回到原值域的预测（见 3.5）。

------

## 3.2 统一观测算子 (H)：抗混叠与口径声明

稀疏观测重建的关键不在于网络规模，而在于：
(1) 观测过程是否丢失不可恢复的信息；(2) 训练与评测是否处于同一口径。

对 SR（下采样）类观测，工程上采用“**先低通、再抽取**”以抑制混叠；实现上常用“模糊（低通）+ 缩小”。OpenCV 的 `resize` 文档明确指出：缩小图像通常使用 `INTER_AREA` 效果更佳，且在降采样时表现为“moire-free”重采样。([OpenCV 文档](https://docs.opencv.org/master/da/d54/group__imgproc__transform.html))

### 3.2.1 SR 观测（下采样）口径

对缩小倍率 (s)，本文 SR 观测定义为
[
y^{\text{SR}}*t = D_s!\left(G*{\sigma_{\text{blur}}}\ast u_t\right)+n_t,
]
其中 (G_{\sigma_{\text{blur}}}) 为高斯核，(D_s) 为下采样算子。工程实现中，口径必须显式声明并写入配置（论文与 YAML 同步）：

- 核大小 (k)、模糊尺度 (\sigma_{\text{blur}})；
- 插值策略：缩小使用 `INTER_AREA`（OpenCV 建议）。([OpenCV 文档](https://docs.opencv.org/master/da/d54/group__imgproc__transform.html))
- 边界处理（如 reflect/replicate/wrap）与是否在滤波中使用该边界策略；
- 标准化/反标准化发生在观测算子之前或之后（本文在 3.5 统一给出“原值域一致性”的执行顺序）。

> **实现可追溯性**：OpenCV `GaussianBlur` 的参数含义（(\sigma_X,\sigma_Y)、`borderType` 等）在官方函数说明中给出。([OpenCV 文档](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html))

### 3.2.2 Crop 观测（裁剪）口径

Crop 观测写为
[
y^{\text{Crop}}*t = C*{h_c,w_c}(u_t)+n_t,
]
其中 (C_{h_c,w_c}) 为中心对齐裁剪算子。裁剪任务的“口径”主要来自**对齐规则与边界策略**，必须显式声明：

- 是否中心对齐；对齐偏移（像素级）如何定义；
- ((h_c,w_c)) 与 `patch_size` 的整倍数约束；
- 边界策略（mirror/zero/wrap）；
- mask 与 crop 是否同步更新（本文要求同步更新，否则指标不可比）。

------

## 3.3 H/DC 单一口径复用：从原则到硬约束

### 3.3.1 复用原则（硬约束）

定义训练侧退化一致性算子 (DC)。本文采用硬约束：
[
DC \equiv H\quad (\text{同一实现、同一参数、同一边界与插值策略}).
]
该约束将“训练域—评测域隐性偏差”压缩到最小，避免 (H_{\text{err}}) 与 Rel-L2 趋势分裂。

### 3.3.2 等价性检验（阻断式审计）

在工程流程中加入阻断式一致性检查（不通过即终止统计）：

1. 抽样 (N\ge 100) 个样本 (u^{(i)})；
2. 生成观测 (y^{(i)} = H(u^{(i)}))；
3. 训练侧计算 (DC(u^{(i)}))；
4. 验证
   [
   \text{MSE}\big(H(u^{(i)}),DC(u^{(i)})\big) < \varepsilon,\quad \varepsilon=10^{-8};
   ]
5. 若失败，输出差异来源（(k,\sigma_{\text{blur}},s)、插值、边界、对齐偏移），并将报告归档到 `runs/<exp>/consistency_report.json`。

------

## 3.4 模型架构：统一接口、时空耦合与算子层（可替换）

### 3.4.1 统一输入打包（可审计接口契约）

以单帧为例，输入张量在通道维拼接：
[
x_t=\text{Concat}\big(\text{baseline}(y_t),,m_t,,\text{coords},,\text{PE}_{\text{Fourier}}\big).
]

- `baseline`：如双线性上采样/简单插值，用于稳定训练起点；
- (m_t)：缺失掩码；
- `coords`：显式坐标（归一化到 ([0,1])）；
- (\text{PE}_{\text{Fourier}})：Fourier 特征，用于提升高频表达能力（Fourier features 的经典论证与实践可追溯）。([arXiv](https://arxiv.org/abs/1904.11486))

统一模型签名（用于横向可替换与评测一致）：
[
\texttt{forward}(x\in\mathbb{R}^{B\times C_{\text{in}}\times N_x\times N_y})\rightarrow
\hat{u}\in\mathbb{R}^{B\times C_{\text{out}}\times N_x\times N_y}.
]

### 3.4.2 算子层与时空耦合（方法定位）

将问题视为“函数到函数”的映射时，神经算子提供函数空间学习视角。JMLR 综述系统总结了神经算子对函数空间映射的学习形式与应用边界。([arXiv](https://arxiv.org/abs/1806.08734))
DeepONet 给出了分支/主干结构的算子逼近范式，可作为可替换的算子层基线。([arXiv](https://arxiv.org/abs/1904.11486))

> **写法建议（避免颗粒度错位）**：把“算子层”写成接口层 `OperatorBlock`，在第6章以替换实验对比 FNO-family / DeepONet-family / Conv-Attn hybrid，并同时报告资源四项。

### 3.4.3 解码与伪影控制（工程口径）

解码侧采用“插值上采样 + 小卷积核细化”（如“双线性 + 3×3”）以降低棋盘格伪影，并在第6章用统一色标误差图与功率谱对比其对伪影与谱泄露的影响。

------

## 3.5 三件套损失：重建、低频谱一致性、原值域观测一致性

### 3.5.1 标准化域与原值域

逐通道 z-score：
[
u^{(z)}=\frac{u-\mu}{\sigma_{\text{z}}},\qquad
\tilde{u}=\sigma_{\text{z}}\hat{u}^{(z)}+\mu.
]
其中 (\tilde{u}) 用于计算与观测口径一致的损失，避免“标准化域优化、原值域评测”导致尺度偏差。

### 3.5.2 重建损失 (L_{\text{rec}})

在 z-score 域：
[
L_{\text{rec}}=\left|\hat{u}^{(z)}-u^{(z)}\right|_2^2.
]

### 3.5.3 低频谱一致性损失 (L_{\text{spec}})

二维 FFT 后，仅在低频集合 (\mathcal{K}*{\text{low}}) 上比较：
[
L*{\text{spec}}=
\sum_{(k_x,k_y)\in\mathcal{K}*{\text{low}}}
\left|\mathcal{F}*{2D}(\hat{u}^{(z)})*{k_x,k_y}
-\mathcal{F}*{2D}(u^{(z)})_{k_x,k_y}\right|*2^2,
\quad \mathcal{K}*{\text{low}}={k_x\le K,,k_y\le K}.
]
阈值 (K) 在敏感性分析中扫描（建议 8–24），以形成可审计的“性能—口径—资源”折衷曲线。

### 3.5.4 原值域观测一致性损失 (L_{\text{dc}})

用统一口径 (H) 直接约束评测口径误差：
[
L_{\text{dc}}=\left|H(\tilde{u})-y\right|*2^2.
]
该项将 (H*{\text{err}}) 内生到训练目标，降低口径断裂。

### 3.5.5 总损失与权重审计

[
L = L_{\text{rec}}+\lambda_s L_{\text{spec}}+\lambda_{dc}L_{\text{dc}}.
]
(\lambda_s,\lambda_{dc}) 通过第6章扫描表与效应量报告给出证据，避免经验拍参。

------

## 3.6 训练确定性与复现闭环（方法章的可执行证据）

方法学层面给出可复现闭环条款，并要求写入工程产物：

- 配置快照：`runs/<exp>/config_merged.yaml`；
- 环境指纹：`runs/<exp>/env_fingerprint.json`（CUDA/驱动、PyTorch、混合精度策略、随机种子等）；
- 一致性审计：`runs/<exp>/consistency_report.json`（H/DC 等价性检验）；
- 统计口径：同配置下 (\ge 3) seeds，报告均值±标准差，并输出显著性与效应量（3.8）。

------

## 3.7 研究设计：对照、消融、敏感性与失败模式归档

### 3.7.1 核心消融（建议表 3-3）

- A0：仅 (L_{\text{rec}})
- A1：(L_{\text{rec}}+\lambda_{dc}L_{\text{dc}})（检验口径一致性约束贡献）
- A2：(L_{\text{rec}}+\lambda_sL_{\text{spec}})（检验结构约束贡献）
- A3：三件套全开（主方法）

### 3.7.2 敏感性分析（建议表 3-4）

- 观测参数：((s,\sigma_{\text{blur}},k)) 或 ((h_c,w_c))、边界策略；
- 频域参数：低频阈值 (K)；
- 表达参数：Fourier 特征维度/频率带宽（参考 Fourier features 原始论文设计）。([arXiv](https://arxiv.org/abs/1904.11486))
- 训练随机性：种子、批大小、梯度裁剪阈值等。

### 3.7.3 失败模式类型化（方法章+实验章双份证据）

- 边界伪影：边界策略/掩码错误导致；
- 相位漂移：频域约束不足或时序耦合不足；
- 振铃/能量泄露：频域约束过强或混叠未抑制（对应 3.2 的抗混叠口径）；
- 指标断裂：H 与 DC 未复用（对应 3.3 的硬约束与审计脚本）。

------

## 3.8 统计检验：显著性与效应量（写作规范）

仅报告单次最好结果缺乏累积科学意义。本文采用：

- paired t-test：在同一测试样本集合上比较两方法的 Rel-L2 序列；
- Cohen’s d（配对设计的效应量口径），并与显著性共同报告。

效应量计算与报告建议可参考 Lakens 的实践指南（Frontiers in Psychology, 2013）。([埃因霍温理工大学研究](https://research.tue.nl/en/publications/calculating-and-reporting-effect-sizes-to-facilitate-cumulative-s))

------

## 3.9 与物理约束（PINN 视角）的关系：边界定位

PINN 通过方程残差引入物理一致性，但在复杂动力系统上可能受制于训练稳定性与优化可达性。引入时空因果结构的训练策略被用于改善 PINN 的收敛与有效性，并给出在多尺度/混沌/湍流任务上的改进证据。([arXiv](https://arxiv.org/abs/2203.07404))
因此，本文把 PINN 残差项定位为**可选的物理一致性正则或对照基线**，主方法定位为“统一口径下的时空模型/算子层 + 观测一致性约束”。

------

## 3.10 图表与表格（建议直接落地到论文）

- 图 3-1 方法总流程图：(u\to H\to (y,m)\to) 输入打包 (\to \Phi_\omega \to \hat{u}\to \tilde{u}\to) 三件套损失 (\to) 审计与统计输出。
- 表 3-1 符号表：新增边界策略、对齐偏移、(\sigma_{\text{blur}}) 与 (\sigma_{\text{z}}) 的区分、Fourier 特征参数。
- 表 3-2 观测口径参数表：((k,\sigma_{\text{blur}},s))、`INTER_AREA`、边界策略、裁剪窗口与对齐规则。
- 表 3-3 消融设置表：A0–A3。
- 表 3-4 敏感性扫描表：参数范围、步长、固定变量、输出指标、资源四项。

------

## 3.11 参考文献（APA｜建议你在全文统一 BibTeX/APA 输出）

- Lakens, D. (2013). *Calculating and reporting effect sizes to facilitate cumulative science: A practical primer for t-tests and ANOVAs*. *Frontiers in Psychology, 4*, 863. https://doi.org/10.3389/fpsyg.2013.00863 ([埃因霍温理工大学研究](https://research.tue.nl/en/publications/calculating-and-reporting-effect-sizes-to-facilitate-cumulative-s))
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*. *Nature Machine Intelligence, 3*(3), 218–229. https://doi.org/10.1038/s42256-021-00302-5 ([arXiv](https://arxiv.org/abs/1904.11486))
- Kovachki, N. B., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A. M., & Anandkumar, A. (2023). *Neural operator: Learning maps between function spaces with applications to PDEs*. *Journal of Machine Learning Research, 24*, 1–97. https://www.jmlr.org/papers/v24/21-1524.html ([arXiv](https://arxiv.org/abs/1806.08734))
- Tancik, M., Srinivasan, P. P., Mildenhall, B., et al. (2020). *Fourier features let networks learn high frequency functions in low dimensional domains*. *NeurIPS 2020*. https://arxiv.org/abs/2006.10739 ([arXiv](https://arxiv.org/abs/1904.11486))
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality is all you need for training physics-informed neural networks*. arXiv:2203.07404. https://doi.org/10.48550/arXiv.2203.07404 ([arXiv](https://arxiv.org/abs/2203.07404))
- OpenCV Documentation. *Geometric image transformations (resize) and interpolation; INTER_AREA guidance for shrinking/decimation*. https://docs.opencv.org/master/da/d54/group__imgproc__transform.html ([OpenCV 文档](https://docs.opencv.org/master/da/d54/group__imgproc__transform.html))
- OpenCV Documentation. *GaussianBlur function reference (sigma, ksize, borderType)*. https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html ([OpenCV 文档](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html))

------

## 你接下来怎么改（一步步指引｜不需要你再想结构）

你已经同意逐步修改，那就按下面顺序推进（每一步都“可检查”）：

1. **先替换符号**（10分钟内完成）：把所有“空间分辨率的 (H\times W)”改成 (N_x\times N_y)；把模糊核尺度写成 (\sigma_{\text{blur}})，把 z-score 标准差写成 (\sigma_{\text{z}})。
2. **补一个“口径参数表（表3-2）”**：把 SR 与 Crop 的参数逐项列出（k、(\sigma_{\text{blur}})、s、插值、边界、对齐偏移、mask 同步规则）。
3. **把 3.3.2 审计脚本写成“阻断式条款”**：明确“不通过不统计”，并写出归档路径。
4. **把 3.8 统计检验写成可复制句式**：主文固定“paired t-test + Cohen’s d + ≥3 seeds”，把 Lakens 引用放在段尾。([埃因霍温理工大学研究](https://research.tue.nl/en/publications/calculating-and-reporting-effect-sizes-to-facilitate-cumulative-s))
5. 完成后把你全文（第3–7章）里出现的同名对象做一次“全局查找替换”（尤其是 (H)、(\sigma)、mask 的定义句），避免颗粒度错位。

你把**你论文当前的第3章 LaTeX 源码片段**（或你正在用的 Overleaf 版本）贴出来后，我可以按你实际模板把上面修订稿进一步改成**完全可直接编译的 LaTeX（含 \label/\ref、表题、图题与 BibTeX 引用位点）**。
