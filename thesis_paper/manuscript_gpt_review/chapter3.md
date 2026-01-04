# 第3章 方法论（重整结构、补强论证与可追溯引用｜修订版）

本章面向“稀疏观测驱动的时空场重建（sparse-to-full spatiotemporal field reconstruction）”任务，给出一套可复现、可审计、可工程落地的方法学框架。与仅强调网络结构改进的写法不同，本章将“**观测口径一致性**”提升为方法论硬约束，并将其落实为**可执行的算子定义、等价性审计与材料产出链路**。

本章核心思想可概括为两条硬约束与一个统一接口：

- **硬约束 A（采样/退化口径一致）**：数据侧观测算子 \(H\) 与训练侧退化算子 \(DC\) 复用同一实现与同一参数（同插值、同边界、同对齐、同预滤）。
- **硬约束 B（评测口径回灌训练）**：在空间域重建误差之外，引入“原值域的观测一致性”项，使评测口径误差 \(H_{\mathrm{err}}\) 被直接优化，而非仅在测试阶段被动暴露。
- **统一接口（可替换模型栈）**：将不同模型（CNN/UNet/Transformer/Neural Operator/Hybrid）约束在统一输入打包与统一输出签名下，确保对比实验可审计、可复核。

---

## 3.1 问题定义与数学形式化

### 3.1.1 离散时空场与学习目标

设空间域 \(\Omega\subset\mathbb{R}^2\)，离散网格为 \(\Omega_h\)，网格分辨率为 \(N_x\times N_y\)。离散时间索引 \(t\in\{1,\dots,T\}\)。目标物理场为标量或多通道张量场：
\[
u_t:\Omega_h\rightarrow \mathbb{R}^{C},\qquad u_{1:T}=\{u_t\}_{t=1}^{T}.
\]

观测由观测算子 \(H\) 与噪声项 \(n_t\) 生成：
\[
y_t = H(u_t)+n_t,\qquad y_{1:T}=\{y_t\}_{t=1}^{T}.
\]

学习映射 \(\Phi_{\omega}\)（参数为 \(\omega\)），从观测序列与辅助信息恢复全场：
\[
\hat{u}_{1:T}=\Phi_{\omega}\big(y_{1:T},\, m_{1:T},\, p\big),
\]
其中 \(m_t\) 为观测掩码（观测位置/缺失区域指示），\(p\) 为显式坐标编码（可包含 Fourier 特征）。

### 3.1.2 两类误差口径：重建域误差与观测口径误差

为避免“训练指标改善但评测口径不降”的断裂，本研究同时报告两类误差：

- **重建域误差（逼近真值）**：如 Rel-L2、MAE 等。
\[
\mathrm{Rel\text{-}L2}=\frac{\lVert \hat{u}-u\rVert_2}{\lVert u\rVert_2}.
\]

- **观测口径误差（口径一致性）**：将预测回到原值域后再施加 \(H\)，与观测 \(y\) 比较：
\[
H_{\mathrm{err}} \triangleq \big\| H(\tilde{u})-y \big\|_2,
\]
其中 \(\tilde{u}\) 为反标准化后的预测（见 3.5.1）。

> 解释性要点：\(\mathrm{Rel\text{-}L2}\) 衡量“数学意义上接近真值”的程度；\(H_{\mathrm{err}}\) 衡量“观测意义上与真实测量口径一致”的程度。两者同步下降，才对应可部署的工程改进。

---

## 3.2 统一观测算子 \(H\)：抗混叠、插值、边界与对齐的口径声明

观测算子 \(H\) 不仅决定“观测长什么样”，还决定“评测如何裁判”。因此，本研究把 \(H\) 视为**唯一口径入口**：数据生成、训练退化、一致性损失与测试评测均从同一 \(H\) 实现出发。

### 3.2.1 SR（下采样）观测口径：先低通、再缩小

对超分辨/降采样观测，抗混叠的工程流程为“先低通、再缩小”。OpenCV 的官方教程明确给出经验性建议：缩小时优先使用 `INTER_AREA`，放大时常用 `INTER_CUBIC` 或 `INTER_LINEAR`。:contentReference[oaicite:0]{index=0}

本文将 SR 观测口径写为：
\[
y^{\mathrm{SR}}_t = D_s\!\left(G_{\sigma}\ast u_t\right)+n_t,
\]
其中 \(G_{\sigma}\) 为高斯低通核，\(\sigma\) 为模糊尺度，\(D_s\) 为倍率为 \(s\) 的缩小算子（实现时使用 `INTER_AREA` 作为默认缩小插值）。:contentReference[oaicite:1]{index=1}

**必须显式声明并写入配置（论文与 YAML 同步）**：

- 预滤核：核大小 \(k\)、\(\sigma\)；
- 插值策略：缩小使用 `INTER_AREA`；:contentReference[oaicite:2]{index=2}
- 边界策略：`reflect/replicate/wrap/constant`；
- 对齐策略：缩小前后坐标对齐（如是否 `align_corners` 等价语义）；
- 噪声模型：\(n_t\) 的分布、强度与是否逐通道。

> 工程可追溯性：OpenCV 的 `GaussianBlur` 在函数层面明确给出参数含义（如 \(\sigma_x,\sigma_y\)、核大小、边界类型等），可作为“预滤实现细节”的可核验依据。:contentReference[oaicite:3]{index=3}

### 3.2.2 Crop（裁剪）观测口径：对齐规则与掩码同步

裁剪观测可写为：
\[
y^{\mathrm{Crop}}_t = C_{h_c,w_c}(u_t)+n_t,
\]
其中 \(C_{h_c,w_c}\) 为裁剪算子（常用中心对齐裁剪）。

裁剪任务的口径风险集中在“对齐偏移”和“掩码是否同步更新”。因此本文要求：

- 裁剪窗口 \((h_c,w_c)\) 与网络 patch/stride 的整倍数约束显式记录；
- 采用中心对齐时，中心点与像素网格的定义方式（偶数尺寸时的偏移规则）需要固定；
- 裁剪同时更新掩码 \(m_t\)：否则输入与标签不在同一几何口径上，指标不可比。

### 3.2.3 点采样/稀疏传感器观测（可选扩展口径）

当观测来自稀疏点集（传感器阵列/探针），可将 \(H\) 写为“采样矩阵/插值到点集”的组合：
\[
y_t = S(\Pi(u_t)) + n_t,
\]
其中 \(\Pi\) 为从网格场到连续场的插值（或从连续场到点的投影），\(S\) 为点采样算子。该形式为后续将“真实传感器口径”纳入统一框架提供接口（本研究如暂不涉及真实点集部署，可在实验章作为扩展讨论）。

---

## 3.3 \(H/DC\) 单一口径复用：从原则到硬约束与阻断式审计

### 3.3.1 硬约束定义

训练侧退化算子 \(DC\) 用于（1）合成训练输入；（2）构建一致性损失。本文采用硬约束：
\[
DC \equiv H\quad \text{（同一实现、同一参数、同一边界与对齐策略）}.
\]

该约束的目标是消除“训练端自造口径”带来的隐性域偏差，使 \(H_{\mathrm{err}}\) 不再成为测试阶段才出现的系统性误差。

### 3.3.2 阻断式等价性审计（不通过即终止实验统计）

为保证“口径一致性”从口号变为可执行证据，本研究在工程流程中加入阻断式审计：

1. 抽样 \(N\ge 100\) 个样本 \(u^{(i)}\)；
2. 计算 \(y^{(i)}=H(u^{(i)})\)；
3. 计算 \(DC(u^{(i)})\)；
4. 验证
\[
\mathrm{MSE}\big(H(u^{(i)}),\,DC(u^{(i)})\big) < \varepsilon,
\]
其中 \(\varepsilon\) 默认取 \(10^{-8}\)（浮点实现下可根据 dtype 放宽到 \(10^{-6}\)）；
5. 失败则输出差异来源（\(\sigma,k,s\)、插值、边界、对齐）并归档到 `runs/<exp>/consistency_report.json`。

> 建议写法：在论文中给出“审计失败样例 + 差异热力图 + 配置差异 diff”，用于支撑“训练/评测断裂会造成指标断裂”的论证力度。

---

## 3.4 模型架构：统一接口、时空耦合与可替换算子层

本研究不把贡献限定为某个特定网络，而把网络视为“可替换模块”。为保证对比实验可审计，先定义统一输入/输出接口，再讨论可替换结构族。

### 3.4.1 统一输入打包（接口契约）

对单帧输入，按通道拼接：
\[
x_t=\mathrm{Concat}\big(\mathrm{baseline}(y_t),\,m_t,\,\mathrm{coords},\,\mathrm{PE}\_{\mathrm{Fourier}}\big).
\]

- `baseline`：如双线性上采样或简单插值，提供稳定的初始解；
- \(m_t\)：掩码（观测位置/缺失区域）；
- `coords`：归一化坐标 \((x,y)\in[0,1]^2\)；
- \(\mathrm{PE}_{\mathrm{Fourier}}\)：Fourier 特征，提升高频表达能力。Fourier 特征的经典结果表明：将输入映射到随机 Fourier 特征空间可显著改善网络对高频函数的学习能力。:contentReference[oaicite:4]{index=4}

统一模型签名：
\[
\mathrm{forward}:\mathbb{R}^{B\times C_{\mathrm{in}}\times N_x\times N_y}\rightarrow
\mathbb{R}^{B\times C_{\mathrm{out}}\times N_x\times N_y}.
\]

### 3.4.2 可替换结构族：时空耦合的三种实现路径

**路径 A：时序显式建模（autoregressive / seq2seq）**  
将 \((y_{1:T_{\mathrm{in}}})\) 映射到 \((\hat{u}_{1:T_{\mathrm{out}}})\)，适合长时滚动误差分析。

**路径 B：把时间当作额外维度（3D 卷积/时空注意力）**  
将 \((t,x,y)\) 共同建模，适合短窗高质量重建。

**路径 C：神经算子/算子层（operator block）**  
神经算子强调学习函数空间之间的映射，并讨论离散化变化下的建模与泛化问题，可作为“跨分辨率/跨网格”讨论的结构基础。:contentReference[oaicite:5]{index=5}  
在算子学习脉络中，DeepONet 通过 branch/trunk 的结构逼近算子映射，并在 Nature Machine Intelligence 论文中给出系统化论证。:contentReference[oaicite:6]{index=6}

> 论文落笔建议：将算子层抽象为 `OperatorBlock`，实验章中以“替换 block”方式对比（FNO-family / DeepONet-family / Conv-Attn hybrid），并同步报告资源四项（参数量、FLOPs、显存峰值、推理延迟）。

### 3.4.4 分阶段时空顺序训练策略（工程落地）

针对时空耦合模型直接端到端训练收敛难、梯度不稳定的问题，本研究提出并实施了**三阶段顺序训练策略（Sequential Training Strategy）**，将空间重建与时序演化解耦学习：

1. **阶段一：空间预训练（Spatial Pretraining）**
   - **目标**：建立高质量的空间重建算子。
   - **操作**：冻结时序模块或仅实例化空间网络（如 SwinUNet、FNO），将时序输入视为独立批次（Batch），仅优化单帧空间重建损失 \(L_{\mathrm{rec}} + L_{\mathrm{spec}} + L_{\mathrm{dc}}\)。
   - **收益**：确保模型首先具备从稀疏/低质观测恢复高频细节的能力，为时序模块提供可信特征。

2. **阶段二：时序预训练（Temporal Pretraining）**
   - **目标**：学习潜在空间的动力学演化。
   - **操作**：冻结已训练的空间编码器与解码器，仅训练时序演化模块（如 ARWrapper、LSTM、Transformer）。
   - **策略**：采用 Teacher Forcing 策略，输入真实历史特征，预测下一时刻特征。

3. **阶段三：联合微调（Joint Fine-tuning）**
   - **目标**：端到端协同优化，消除模块间适配误差。
   - **操作**：解冻所有参数，进行**20步自回归（AR）滚动预测**（20-step Autoregressive Rollout）。
   - **课程学习**：引入 **Teacher Forcing Decay**，训练初期高频使用真值引导，随 Epoch 增加逐步切换为完全自回归模式（使用前一步预测值），以缓解 Exposure Bias 并提升长时稳定性。

---

## 3.5 三件套损失：重建、低频谱一致性、原值域观测一致性

### 3.5.1 标准化域与原值域：避免尺度口径偏差

逐通道 z-score 标准化：
\[
u^{(z)}=\frac{u-\mu}{\sigma_{z}},\qquad
\hat{u}^{(z)}=\Phi_{\omega}(\cdot),\qquad
\tilde{u}=\sigma_{z}\hat{u}^{(z)}+\mu.
\]

其中 \(\tilde{u}\) 用于计算与观测口径一致的损失与指标（因为 \(H\) 的实现通常定义在原值域或工程量纲域上）。

### 3.5.2 重建损失 \(L_{\mathrm{rec}}\)

在 z-score 域计算：
\[
L_{\mathrm{rec}}=\left\|\hat{u}^{(z)}-u^{(z)}\right\|_2^2.
\]

### 3.5.3 低频谱一致性损失 \(L_{\mathrm{spec}}\)

对二维 FFT，仅在低频集合 \(\mathcal{K}_{\mathrm{low}}\) 上约束：
\[
L_{\mathrm{spec}}=
\sum_{(k_x,k_y)\in\mathcal{K}_{\mathrm{low}}}
\left|\mathcal{F}(\hat{u}^{(z)})_{k_x,k_y}
-\mathcal{F}(u^{(z)})_{k_x,k_y}\right|^2,
\quad \mathcal{K}_{\mathrm{low}}=\{k_x\le K,\,k_y\le K\}.
\]

设置该项的工程动机是：低频结构往往对应大尺度主导形态（平均流、主涡、锋面/团簇），对下游任务（识别、控制、诊断）具有更稳定的意义；同时低频约束对抗“仅靠像素误差导致的结构漂移”。

### 3.5.4 原值域观测一致性损失 \(L_{\mathrm{dc}}\)

以统一口径 \(H\) 直接约束评测口径一致性：
\[
L_{\mathrm{dc}}=\left\|H(\tilde{u})-y\right\|_2^2.
\]

该项的关键在于：把 \(H_{\mathrm{err}}\) 内生到训练目标，从机制上削弱“训练域好看、评测口径失真”的风险。

### 3.5.5 时序一致性正则化（AR特化）

针对自回归（AR）长时预测中可能出现的误差累积与物理量漂移，本研究引入两项辅助正则化损失（权重由 Teacher Forcing Decay 动态调制）：

1. **时序导数一致性（Derivative Consistency）**：约束预测序列的时间差分与真值一致，强化动力学趋势捕捉。
   \[
   L_{\mathrm{deriv}} = \left\| \partial_t \hat{u} - \partial_t u \right\|_2^2 \approx \sum_t \left\| (\hat{u}_{t+1}-\hat{u}_t) - (u_{t+1}-u_t) \right\|_2^2
   \]

2. **能量演化一致性（Energy Consistency）**：约束全场能量（\(L^2\) 范数）的演化轨迹，防止长时预测中的能量耗散或爆炸。
   \[
   L_{\mathrm{energy}} = \sum_t \left| \|\hat{u}_t\|_2^2 - \|u_t\|_2^2 \right|
   \]

> 工程注记：这两项正则化通常在联合微调阶段随 Teacher Forcing 退出而逐步生效，作为“三件套”的补充。

### 3.5.6 总损失与权重审计

\[
L = L_{\mathrm{rec}}+\lambda_{s}L_{\mathrm{spec}}+\lambda_{dc}L_{\mathrm{dc}}.
\]

权重 \((\lambda_s,\lambda_{dc})\) 不采用“经验拍参”写法，而在实验章通过扫描表、效应量与显著性检验给出证据链（见 3.8 与第6章设置）。

---

## 3.6 训练确定性与复现闭环（可执行证据要求）

为满足研究生论文对“可复核、可追溯”的要求，本研究将复现实验的关键要素固化为**必须产物**：

- 配置快照：`runs/<exp>/config_merged.yaml`；
- 环境指纹：`runs/<exp>/env_fingerprint.json`（CUDA/驱动、PyTorch、混合精度、确定性开关、随机种子等）；
- 口径一致性审计：`runs/<exp>/consistency_report.json`（H/DC 等价性检验）；
- 训练曲线与指标主表：`runs/<exp>/metrics.csv` + 可视化图；
- 失败样例归档：`runs/<exp>/failure_cases/`（边界伪影、振铃、相位漂移等类型化保存）。

> 论文写作建议：把“产物列表 + 文件树 + 每项产物用于回答的审稿问题”写成方法章的“可审计性小节”，通常比“口头承诺可复现”更有说服力。

---

## 3.7 研究设计：对照、消融、敏感性与失败模式归档

### 3.7.1 核心消融（建议表 3-3）

- A0：仅 \(L_{\mathrm{rec}}\)
- A1：\(L_{\mathrm{rec}}+\lambda_{dc}L_{\mathrm{dc}}\)（检验口径一致性约束贡献）
- A2：\(L_{\mathrm{rec}}+\lambda_sL_{\mathrm{spec}}\)（检验低频结构约束贡献）
- A3：三件套全开（主方法）

### 3.7.2 敏感性分析（建议表 3-4）

- 观测参数：\((s,\sigma,k)\) 或 \((h_c,w_c)\)、边界策略；
- 频域参数：低频阈值 \(K\)（建议 8–24 扫描）；
- 编码参数：Fourier 特征维度与频带（可据 Fourier 特征论文设置范围）。:contentReference[oaicite:7]{index=7}
- 训练随机性：seed、batch size、梯度裁剪阈值等。

### 3.7.3 失败模式类型化（方法章 + 实验章双份证据）

- **边界伪影**：边界策略不一致、掩码未同步更新；
- **相位漂移/时序漂移**：时空耦合不足或谱约束缺失；
- **振铃/能量泄露**：抗混叠不足或谱约束权重过强；
- **指标断裂**：\(H\) 与 \(DC\) 未严格复用（由审计报告定位）。

---

## 3.8 统计检验：显著性与效应量（写作规范）

仅报告“单次最好结果”难以支撑稳健结论。本文建议在同一测试样本集合上进行多种子重复，并报告：

- 配对设计的显著性检验（paired t-test）；
- 配对设计的效应量（Cohen’s d 或其配对变体口径）。

效应量计算与报告规范可参考 Lakens 的实践指南，该文针对 t-test/ANOVA 的效应量报告给出可操作建议。:contentReference[oaicite:8]{index=8}

---

## 3.9 与物理约束学习（PINN 视角）的关系与边界定位

PINN 通过 PDE 残差引入物理一致性，但在多尺度、混沌或湍流类动力系统上可能出现训练困难。相关研究从 NTK 视角解释了 PINN 的训练病理，并讨论了损失项收敛速率不一致等问题。:contentReference[oaicite:9]{index=9}  
另外，因果结构约束被用于改善 PINN 在动态系统上的训练稳定性与长期预测表现。:contentReference[oaicite:10]{index=10}

因此，本研究对 PINN 的定位为：

- PINN 残差项可作为可选正则或对照基线；
- 主方法以“统一口径下的数据一致性（\(L_{\mathrm{dc}}\)）+ 结构性谱约束（\(L_{\mathrm{spec}}\)）+ 可替换时空模型栈”为中心。

---

## 3.10 图表与表格建议（可直接落地到论文）

- **图 3-1（方法总流程图）**：\(u\to H\to (y,m)\to\) 输入打包 \(\to \Phi_\omega \to \hat{u}\to \tilde{u}\to\) 三件套损失 \(\to\) 审计与统计产物。
- **表 3-1（符号表）**：区分 \(\sigma\)（预滤）与 \(\sigma_z\)（标准化），列出边界策略与对齐偏移。
- **表 3-2（观测口径参数表）**：\((k,\sigma,s)\)、`INTER_AREA`、边界策略、裁剪窗口与对齐规则。
- **表 3-3（消融设置）**：A0–A3。
- **表 3-4（敏感性扫描表）**：参数范围、步长、固定变量、输出指标、资源四项。

---

## 3.11 本章小结

本章围绕“稀疏观测驱动的时空场重建”任务，给出以观测口径一致性为核心的可执行方法论：用统一观测算子 \(H\) 固化评测口径，并通过硬约束 \(DC\equiv H\) 与阻断式等价性审计消除训练/评测断裂；在模型层面采用统一输入打包与可替换架构接口；在目标函数层面采用重建损失、低频谱一致性与原值域观测一致性三件套，直接优化 \(H_{\mathrm{err}}\)；在实验规范层面固化配置快照、环境指纹与统计检验，为后续章节的严格评测与可复核讨论奠定基础。

---

## 本章参考文献（APA｜建议全文统一 BibTeX/APA）

- Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science: A practical primer for t-tests and ANOVAs. *Frontiers in Psychology, 4*, 863. doi:10.3389/fpsyg.2013.00863 :contentReference[oaicite:11]{index=11}
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence, 3*(3), 218–229. doi:10.1038/s42256-021-00302-5 :contentReference[oaicite:12]{index=12}
- Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A. M., & Anandkumar, A. (2023). Neural operator: Learning maps between function spaces with applications to PDEs. *Journal of Machine Learning Research, 24*(89), 1–97. :contentReference[oaicite:13]{index=13}
- Tancik, M., Srinivasan, P. P., Mildenhall, B., et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS 2020*. arXiv:2006.10739. :contentReference[oaicite:14]{index=14}
- Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics, 449*, 110768. doi:10.1016/j.jcp.2021.110768 :contentReference[oaicite:15]{index=15}
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). Respecting causality is all you need for training physics-informed neural networks. arXiv:2203.07404. :contentReference[oaicite:16]{index=16}
- OpenCV Documentation. Geometric Transformations of Images: Preferable interpolation methods are `INTER_AREA` for shrinking. :contentReference[oaicite:17]{index=17}
- OpenCV Documentation. `GaussianBlur` function reference (parameters include ksize, sigma, borderType). :contentReference[oaicite:18]{index=18}
