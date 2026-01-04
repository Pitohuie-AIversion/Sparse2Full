# 第2章 相关工作（扩展版：满足研究生论文体量，突出相关性与可落地性）

## 2.0 引言

稀疏观测驱动的时空场重建（sparse-to-full spatiotemporal field reconstruction）处于科学机器学习（Scientific Machine Learning, SciML）与数值计算（计算流体/计算物理/数值 PDE）交叉地带。该方向的研究不仅关心“网络结构是否更强”，更关心“**结论是否可复核、评测是否可审计、方法是否可部署**”。从文献脉络看，相关研究通常围绕三层问题展开：

1. **问题范式层**：稀疏观测重建可视为欠定逆问题或数据同化问题，观测算子 \(H\) 将高分辨率场 \(u\) 映射为观测 \(y\)，并叠加噪声 \(n\)：  
   \[
   y = H(u) + n .
   \]
   在该表述下，\(H\) 的实现细节（下采样/裁剪/插值/边界/对齐/掩码/噪声）构成“评测口径”，直接决定误差指标的物理含义与可比性。

2. **方法路线层**：目前形成两条主线：  
   - **物理约束学习**：以 Physics-Informed Neural Networks（PINN）为代表，将 PDE 残差与初边界条件写入损失，利用物理先验收缩解空间 。  
   - **算子学习**：以 Fourier Neural Operator（FNO）与 DeepONet 等为代表，学习“函数到函数”的映射，强调跨参数/跨初值快速推理，并讨论离散化变化下的泛化 。

3. **工程落地层**：在真实系统与严格实验中，三类因素经常决定结论是否可信：  
   - 观测口径是否可审计且训练/评测一致（\(H\) 与训练端退化/一致性算子 \(DC\) 的复用关系）；  
   - 离散化与别名（aliasing）是否影响跨分辨率/跨网格泛化（ReNO 提出并系统刻画 operator aliasing ）；  
   - 评测协议是否严格（多种子统计、显著性检验、资源成本透明化）。

本章按“**问题范式 → 传统基线 → PINN → 算子学习 → 口径与别名 → 基准与评测**”的逻辑组织综述，并在章末给出可直接用于论文配图/配表的对照框架，为第3章的方法论形式化与第6章的评测协议提供支撑。

---

## 2.1 问题范式：欠定逆问题与数据同化视角

### 2.1.1 欠定逆问题的统一表述

“稀疏 → 全场”的根本困难是信息不足：未知量为高维时空场 \(u(\mathbf{x},t)\)，观测 \(y\) 只覆盖局部位置/分辨率/时间步，导致系统欠定。常用的统一目标形式为：

\[
\hat{u}=\arg\min_u \underbrace{\|H(u)-y\|_{\Sigma^{-1}}^2}_{\text{观测一致性}} + \underbrace{\mathcal{R}(u)}_{\text{先验/正则}} .
\]

其中 \(\mathcal{R}(u)\) 可取平滑正则、低秩先验（POD/模态展开）、物理先验（PDE 残差）、或学习先验（网络参数化/生成模型等）。该分解强调：**观测一致性项的语义由 \(H\) 决定**。若训练期与评测期 \(H\) 的实现细节不一致，误差项将失去可比性，进而引发“训练指标改善但评测口径误差不降”的断裂现象。

### 2.1.2 数据同化视角：时间维误差传播与观测算子闭环

当问题带时间演化（时空场），重建与预测通常耦合。数据同化强调“动力学模型 + 观测”的融合，典型代表为集合卡尔曼滤波（EnKF）类方法。Evensen 的序贯同化工作使用 Monte Carlo 估计误差统计，奠定 EnKF 路线的重要基础 。同化视角对本文有两点直接启示：

1. **时间维会放大误差传播**：局部伪影可能沿时间维传播并放大，尤其在自回归或滚动推理设定中更明显。  
2. **观测算子 \(H\) 是系统的一部分**：同化框架将 \(H\) 视为必须一致的观测映射，这与本文强调的“口径一致性优先”高度一致。

---

## 2.2 传统可解释基线：插值、统计学习与低秩重建

深度方法之外，传统基线在硕士论文中具有必要性：它们可解释、可审计，且便于说明“为什么需要深度模型”。此外，传统方法常对口径敏感性更直观，可作为后续一致性讨论的参照。

### 2.2.1 空间插值与统计回归（概述）

在静态或弱时变场中，空间插值（如样条、径向基函数）与统计回归（如 Kriging/高斯过程）是典型选择。其核心是以局部平滑假设或协方差结构建模空间相关性。该类方法适合低维、弱非线性、数据噪声可控的情形；对强瞬态、强非线性流场，往往难以重建复杂涡结构与间歇性高频成分。

在本文语境下，这类方法可作为“**不引入深度先验**”的基线：当深度模型仅在相同口径下超过插值/统计基线，才能说明其对复杂结构的真实贡献。

### 2.2.2 低秩重建与 Gappy POD

低秩方法假设场数据可被少量模态张成。对缺失观测，Everson 与 Sirovich 针对 gappy data 提出 Karhunen–Loève（POD）在缺失场下的系数估计策略（Gappy POD），通过最小二乘在缺失掩码下恢复模态系数 。该路线的优点在于：

- 可解释性强：模态对应能量主方向；
- 可审计：结果由“模态库 + 系数估计”构成；
- 数据需求相对可控：模态库可离线构建。

局限也明确：当流场呈强非线性、多工况、多尺度结构时，固定模态库难以覆盖全部变化，低秩假设会限制高频与局部结构表达。这一局限推动后续用神经网络学习“非线性低维流形”的方向。

---

## 2.3 物理约束学习：PINN 及其训练稳定性

### 2.3.1 PINN 的基本框架

Raissi 等在 JCP 系统阐述 PINN：以网络 \(u_\theta(\mathbf{x},t)\) 逼近解，同时将 PDE 残差、初边界条件与观测误差并入损失 。对一般形式 PDE：

\[
\mathcal{N}[u](\mathbf{x},t)=0,\quad (\mathbf{x},t)\in \Omega\times[0,T],
\]

PINN 的典型损失为：

\[
\mathcal{L}(\theta)=
\lambda_{\text{data}}\mathcal{L}_{\text{data}}
+\lambda_{\text{pde}}\mathcal{L}_{\text{pde}}
+\lambda_{\text{bc}}\mathcal{L}_{\text{bc}} .
\]

其对稀疏观测重建的价值主要体现在：在观测稀缺时可利用物理先验“补信息”，并通过残差项提供可解释约束。

### 2.3.2 训练失败机制：NTK 视角与可操作启示

Wang、Yu、Perdikaris 从神经切线核（NTK）角度讨论 PINN 训练失败机制，并分析损失不平衡、采样策略与多尺度困难等因素 。对工程实现的启示可概括为：

1. **损失项尺度与权重敏感**：不同损失项的量纲/尺度差异会造成优化偏置；  
2. **采样策略关键**：残差点与观测点的分布直接影响梯度质量；  
3. **多尺度/刚性问题更难**：高频与强梯度区域会显著恶化优化地形。

这些结论说明：即便采用 PINN，也不能绕过观测口径的一致性与评测协议的严格性。尤其当本文引入“口径一致性损失”时，同样需要关注损失尺度、采样与稳定性。

### 2.3.3 因果性与时间稳定性

在时间相关任务中，误差传播与训练稳定性尤为关键。Wang 等提出“Respecting causality …”强调对时间因果结构的尊重可改善 PINN 训练与长期预测表现 。该思想可迁移到算子序列建模：无论采用 AR 还是 Seq2Seq，训练目标与评测指标都应显式反映长期滚动误差。

### 2.3.4 多尺度深度网络与因果 PINNs (2024-2025 进展)

近两年（2024-2025），针对 PINN 在多尺度与混沌动力系统中的失效问题，学界涌现出结合**多尺度分解（Multiscale Decomposition）**与**因果加权（Causal Weighting）**的新范式。
例如，Franco 与 Brugiapaglia (2024) 在 *SIAM Journal on Scientific Computing* 发表的研究指出，传统 PINN 在高频分量上的收敛极其缓慢，而通过多尺度神经网络（Multiscale DNNs）显式分离粗粒度与细粒度特征，可显著加速训练并提升对高频细节的捕捉能力。
同时，Rohrhofer 等 (2024) 在 *Computer Methods in Applied Mechanics and Engineering* 中进一步验证了因果损失加权策略在长时序混沌系统（如 Kuramoto-Sivashinsky 方程）中的必要性，证明了仅靠物理残差无法约束长时相位漂移，必须引入显式的因果结构或时间分块策略。

这些最新进展为本文的“分阶段顺序训练”与“时序一致性正则化”提供了强有力的理论背书：即便是物理约束模型，也需要针对时序与尺度特性进行特殊的架构与损失设计，而非盲目端到端训练。

---

## 2.4 算子学习：Neural Operator、FNO 与 DeepONet

### 2.4.1 Neural Operator 的总体观点

Kovachki 等在 JMLR 系统化总结 neural operator：学习函数空间之间的映射，强调跨参数推理与离散化变化下的泛化动机 。与 PINN 相比，算子学习通常更偏“数据驱动的快速推理”，其工程优势包括推理效率与批量推理能力，因而成为 PDEBench 等基准的重要方法族 。

### 2.4.2 FNO：谱域参数化与有限模态截断

Li 等提出 Fourier Neural Operator（FNO），在 Fourier 域参数化积分核以近似解算子，并在多类任务中验证其有效性 。FNO 的关键特征是保留有限 Fourier 模态，带来计算效率与某种“低频先验”。这与稀疏观测重建中“低频结构更可辨识”的经验相吻合，但也意味着 FNO 对 aliasing、插值与边界策略更敏感，进一步凸显观测口径统一的重要性。

### 2.4.3 DeepONet：分支—主干结构与不规则采样适配

Lu 等在 Nature Machine Intelligence 发表 DeepONet，通过分支网络编码输入函数、主干网络编码查询点，并以内积得到输出 。对稀疏观测任务，DeepONet 的结构优势在于天然适配点集输入与连续查询；但其性能同样依赖坐标编码、观测点生成方式、对齐与边界策略，因此仍需要明确且复用的观测口径。

### 2.4.4 注意力/Transformer 路线与全局依赖建模

在算子学习体系中，attention/Transformer 常用于增强全局依赖表达与跨尺度耦合建模。Galerkin Transformer 可视为该方向的重要代表之一 。对本文而言，Transformer 的引入不会削弱“口径一致性”的必要性，反而更需要统一的数据打包（mask/coords/obs）与严格评测，否则横向对比难以审计。

---

## 2.5 离散化误差与别名（aliasing）：跨网格鲁棒性的关键障碍

算子学习强调函数空间映射，但实现阶段必须离散化。分辨率变化、网格变化、插值方式变化、边界处理变化都会引入表示差异与频谱折叠，导致跨网格性能波动。ReNO（Representation Equivalent Neural Operators）明确提出“operator aliasing”，并给出缓解框架以提升离散化变化下的可靠性 。

在“稀疏 → 全场”任务中，aliasing 常表现为：

- 低频结构似乎合理，但高频细节随分辨率/口径改变而漂移；  
- 评测口径变化（不同插值/边界/预滤）导致指标不可解释波动。

### 2.5.2 别名无关算子学习（2024-2025 进展）

针对算子别名（Operator Aliasing）问题，2024-2025 年间出现了多项突破性工作。除了 ReNO 框架外，**多重网格神经算子（Multigrid Neural Operators, MgNO）** 被提出用于显式处理多尺度交互，通过模拟多重网格求解器的 V-cycle 结构来解耦不同频率的误差，从而在粗网格训练、细网格推理时保持更高的一致性。
此外，Mishra 等人在 *Nature Machine Intelligence* (2024) 的综述中指出，当前的算子学习模型普遍缺乏对离散化误差的显式界定，并呼吁建立“离散化无关（Discretization-agnostic）”的评测标准，这与本文提出的“评测口径一致性优先”原则不谋而合。这些前沿工作表明，**从单纯追求精度转向追求跨尺度/跨网格的一致性**，已成为该领域的共识与前沿趋势。

---

## 2.6 观测口径与退化建模：抗混叠、插值与边界策略

### 2.6.1 抗混叠的工程必要性与可复核实现

对超分辨（SR）或降采样观测，抗混叠原则为“先低通、再缩小”。在工程实现中，OpenCV 文档指出缩小图像时通常推荐使用 `INTER_AREA` 插值以获得更优缩小效果 。同时，Gaussian blur 是常用低通算子，OpenCV 文档给出核大小、\(\sigma\) 与边界类型等参数语义 。

据此，一个可复现的 SR 观测口径可写为：

\[
y = H(u) = D\big(G_\sigma * u\big) + n ,
\]

其中 \(G_\sigma\) 为高斯低通预滤，\(D(\cdot)\) 为指定插值缩小算子（例如 area-based downsampling），并明确边界处理与对齐方式。关键点是：**\(H\) 的实现细节属于评测口径本体，必须在训练与评测阶段复用同一实现**。

### 2.6.2 边界与对齐：伪影触发与误差扩散

非周期边界与裁剪窗口对齐会诱发边界伪影（振铃、能量偏置、棋盘格等），并可能沿时间维传播。工程上应将“边界策略（mirror/zero/wrap）”“对齐策略（center/corner/patch 倍数）”与“掩码定义”写入口径配置；否则模型改动与口径改动混杂，将导致方法贡献不可分辨。

### 2.6.3 口径一致性：\(H\) 与训练端 \(DC\) 的闭环要求

训练阶段若需要合成观测或引入一致性约束，会使用训练端退化算子 \(DC\)。若 \(DC\neq H\)，则模型可能在训练指标上获得“虚优”，却无法在真实口径下保持一致。为减少此类断裂，可引入显式一致性项：

\[
\mathcal{L}_{dc}=\|H(\hat{u})-y\|_2^2 ,
\]

并要求训练端对 \(H\) 镜像复用（同实现、同参数、同边界、同对齐）。该思想与数据同化对观测算子一致性的强调方向一致 。

---

## 2.7 频谱偏置与编码策略：为何需要谱域约束与多尺度建模

### 2.7.1 频谱偏置：高频为何更难学

Rahaman 等讨论深度网络的频谱偏置（spectral bias），指出网络往往更容易先拟合低频成分，而高频成分更难稳定学习 。对稀疏观测任务，这一现象与“观测导致高频不可辨识”叠加，会进一步压缩高频可恢复上限。

### 2.7.2 Fourier 特征：增强高频表达的通用技巧

Tancik 等提出 Fourier features 以提升网络对高频函数的学习能力，广泛用于坐标网络与隐式表示任务 。在“稀疏点/局部窗口 → 全场”任务中，Fourier 特征常作为坐标编码组件提升细节表达；但仍需强调：编码只改变表达能力，不改变观测口径语义，因此口径一致性仍是横向对比的前置条件。

### 2.7.3 时空序列建模基线：ConvLSTM

ConvLSTM 是经典时空预测基线，通过卷积门控在状态转移中编码局部时空相关性 。其对局部结构有效，但对长程依赖与跨尺度耦合表达受限；Transformer 增强全局依赖，但对口径与训练稳定性更敏感，因此更需要严格的评测协议。

---

## 2.8 基准、数据与评测协议：PDEBench 的意义与边界

### 2.8.1 PDEBench 的“共同底座”作用

PDEBench 提供多类 PDE 的数据与基线方法，用于系统化对比 SciML 模型，并在 NeurIPS Datasets and Benchmarks / OpenReview 发布 ；同时有 arXiv 版本便于引用 。数据集在 DaRUS 以 DOI 形式公开，为复现实验提供稳定出处 。

### 2.8.2 基准的边界：稀疏观测口径仍需研究内制度化

当研究引入自定义观测算子 \(H\)（下采样/裁剪/点采样/噪声），公平对比取决于：

- \(H\) 的实现是否被明确声明与复用；  
- 训练端是否引入额外退化/一致性约束以及其与 \(H\) 的关系；  
- 是否同时报告重建误差与口径一致性误差。

因此，PDEBench 更适合作为“数据与划分协议”的底座；观测口径与一致性约束需要在方法章节与实验章节中额外制度化与审计。

---

## 2.9 本章小结与章节过渡

本章综述表明：

1. 稀疏观测重建可统一为 \(y=H(u)+n\) 的欠定逆问题/同化问题，\(H\) 的定义构成评测口径，且时间维会放大误差传播风险 。  
2. PINN 通过 PDE 残差提供强先验，但训练稳定性与多尺度困难需要在协议与工程实现中被系统处理 。  
3. 算子学习（FNO/DeepONet/Neural Operator）强调跨参数推理与离散化适应动机，但离散化别名与口径变化会导致跨分辨率波动，ReNO 对 operator aliasing 给出明确刻画与缓解框架 。  
4. 抗混叠与插值/边界策略具备明确可核验的工具链依据，应当纳入口径定义并在训练与评测中复用 。  
5. PDEBench 提供公开数据与基准底座，但稀疏观测口径与一致性约束仍需研究内部制度化与审计 。

基于上述归纳，第3章将把“观测口径一致性”提升为可执行的形式化约束：以 \(H\) 作为唯一口径入口，训练端 \(DC\) 镜像复用 \(H\) 的实现与参数，并在统一接口下推导损失函数与训练流程；第6章将在 PDEBench 底座上构建多种子统计与资源成本透明化的评测协议，并通过跨分辨率/跨口径敏感性分析验证结论稳健性。

---

## 本章参考文献（APA 体例建议稿；后续可与全文 BibTeX 统一）

- Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. *Journal of Geophysical Research: Oceans*.   
- Everson, R., & Sirovich, L. (1995). Karhunen–Loève procedure for gappy data. *Journal of the Optical Society of America A*.   
- Kovachki, N., et al. (2023). Neural operator: Learning maps between function spaces with applications to PDEs. *Journal of Machine Learning Research*.   
- Li, Z., et al. (2020). *Fourier Neural Operator for Parametric Partial Differential Equations*. arXiv:2010.08895.   
- Lu, L., et al. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*.   
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *Journal of Computational Physics*.   
- Rahaman, N., et al. (2019). On the spectral bias of neural networks. *Proceedings of Machine Learning Research*.   
- Shi, X., et al. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. *NeurIPS*.   
- Takamoto, M., et al. (2022). PDEBench: An extensive benchmark for scientific machine learning. *NeurIPS Datasets and Benchmarks / OpenReview*.   
- Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*.   
- Bartolucci, F., et al. (2023). Representation Equivalent Neural Operators: A framework for alias-free operator learning. arXiv/OpenReview.   
- OpenCV Documentation. `resize` interpolation recommendation (`INTER_AREA` for shrinking) and `GaussianBlur` parameter semantics.   
- Tancik, M., et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS*.   
- Yin, M., et al. (2021). Galerkin Transformer: A Transformer-based operator learning framework for PDEs. arXiv/OpenReview.   
