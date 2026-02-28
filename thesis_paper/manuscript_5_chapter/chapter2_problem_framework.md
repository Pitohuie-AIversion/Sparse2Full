# 第2章 问题建模与理论分析 (Problem Formulation & Theory)

本章旨在建立稀疏观测下物理场重建问题的数学框架，并分析“观测一致性”在逆问题求解中的理论地位。首先给出逆问题与观测模型的统一表述；随后明确评测口径的“双域指标”；最后从不适定性、观测算子误差与误差界推导说明：若要在真实部署口径下获得可控误差与鲁棒泛化，需要将数据一致性约束建立在真实观测算子之上（$DC \equiv H$）。

## 2.1 物理场重建的数学框架

### 2.1.1 逆问题表述与观测模型

稀疏观测下的物理场重建可统一表述为“带噪声、带退化算子的逆问题”：从有限维（通常低维）的观测空间恢复高维状态空间中的物理场。逆问题文献中常用的抽象形式为
\[
g = A(f) + e,
\]
其中 $A$ 为前向（观测）算子，$e$ 为噪声或模型误差；研究重点在于“间接观测 + 噪声”导致的非唯一与不稳定，并通过正则化获得稳定近似解 [1]。

在本文语境中，设空间域 $\Omega \subset \mathbb{R}^d$（工程中常见 $d=2$ 或 $3$），时间域 $\mathcal{T}=[0,T]$。目标物理场记作
\[
u(x,t): \Omega \times \mathcal{T} \rightarrow \mathbb{R}^{C},
\]
其中 $C$ 为通道数（如速度分量 $(u,v)$、压力、温度等）。为与工程数据对齐，考虑离散化网格 $\Omega_h$（分辨率 $N_x\times N_y$ 或 $N_x\times N_y\times N_z$）以及离散时刻 $t_k$。在时刻 $t_k$ 的真实全场状态写作
\[
u_k \in \mathbb{R}^{N_x \times N_y \times C}.
\]

**观测模型与观测算子**：观测数据由观测算子（observation operator）产生。数据同化与统计反演文献通常用
\[
y = H[x] + \varepsilon
\]
刻画“模型空间到观测空间”的映射关系，并强调 $H$ 往往包含投影、插值、积分、变量变换等多种操作；当 $H$ 建模不完全时，会出现观测算子模型误差 [2]。对时序场，本文采用
\[
y_k = H(u_k) + \eta_k,
\]
其中 $y_k \in \mathcal{Y}$ 为稀疏或降质后的观测，$\eta_k$ 为噪声项。工程中常将噪声近似为加性高斯（或局部线性化后近似为高斯），并用观测误差协方差 $R$ 给出加权残差度量（Mahalanobis 范数）[2]。

**逆映射与学习目标**：重建任务旨在学习一个近似逆映射（或近似后验估计器）$\mathcal{F}_\theta$，利用观测序列及其辅助信息恢复全场：
\[
\hat{u}_{1:K}=\mathcal{F}_\theta\!\left(y_{1:K},\, m_{1:K},\, \text{coords/BC/params},\dots\right),\quad \hat{u}_k \approx u_k .
\]
其中 $m_k$ 可表示掩码、测点集合或采样几何（规则/非规则）。需要强调的是：当观测算子 $H$ 包含空间积分、抗混叠滤波或复杂几何裁剪时，$m_k$ 仅描述几何信息的一部分；真正决定“观测口径”的仍是 $H$ 的物理建模 [2]。

---

### 2.1.2 双域误差度量：重建域与观测域

仅用重建域逐点误差训练与汇报，容易出现“训练指标变好，但部署口径不一致导致上线效果变差”的断裂。更贴近逆问题与数据同化传统的做法是：同时在“状态空间（重建域）”与“观测空间（观测域）”度量误差与一致性；观测域指标直接对应前向模型（观测口径）是否被满足，属于可部署性的硬约束 [3]。

**重建域误差**：常用指标是相对 $L_2$（离散情形常等价于 Frobenius 范数）：
\[
\mathrm{Rel\text{-}L2}=\frac{\|\hat{u}-u\|_2}{\|u\|_2}.
\]
该指标反映“数学意义上逼近真值”的能力，但不直接保证输出与真实传感器观测口径一致。

**观测口径误差与加权数据一致性**：可先采用
\[
H_{\mathrm{err}}=\|H(\hat{u})-y\|_2,
\]
并进一步推广为带观测误差协方差的加权形式：
\[
H_{\mathrm{err},R}=\left\|R^{-1/2}\bigl(H(\hat{u})-y\bigr)\right\|_2.
\]
在数据同化与贝叶斯反演框架中，观测残差项通常以
\[
\bigl(y-H(x)\bigr)^\top R^{-1}\bigl(y-H(x)\bigr)
\]
进入代价函数，权重由 $R$ 决定；这不仅体现噪声强度，也体现观测相关性与代表性误差的建模 [2,3]。在变分同化（如 4D-Var）中，代价函数梯度计算依赖观测算子及其线性化/伴随，说明“观测口径”并非简单下采样 [3]。

---

### 2.1.3 不适定性、观测算子误差与观测一致性必要性

#### (1) 逆问题的不适定性与适定化需求

稀疏观测下的场重建在理论上通常不满足经典适定性条件。Hadamard 适定性准则要求问题满足“存在性、唯一性、稳定性（对数据连续依赖）”；违背任一条件即构成不适定 [4]。现代逆问题与正则化综述指出：反演困难正来自上述条件的破坏，噪声与建模误差会被逆映射放大，因此必须通过正则化构造稳定近似逆 [5]。

**非唯一性与零空间**：当 $H$ 含降维操作（稀疏采样、积分测量、下采样、投影等）时，通常不是一一映射：存在不同状态 $u$ 映射到相同或近似相同的观测 $y$。在线性或局部线性化情形下，若 $\mathcal{N}(H)\neq\{0\}$，则仅凭 $y$ 无法唯一恢复 $u$，必须引入先验或正则化缩小可行解集合 [6]。

![图 2-3: 稀疏观测逆问题的不适定性与零空间示意图。由于观测算子 $H$ 存在非平凡零空间 $\mathcal{N}(H)$，不同的物理状态 $u_1$ 与 $u_2$ 可能映射到相同的观测 $y$（即 $H(u_1)=H(u_2)=y$）。在缺乏先验约束的情况下，逆映射 $H^{-1}$ 无法从观测中唯一恢复真实解，且容易受到高频噪声分量（零空间扰动）的干扰。](images/fig2-3_null_space_illposedness.png)

**不稳定性与噪声放大**：除非 $H^{-1}$ 在相应函数空间上连续（稳定），否则即便观测噪声很小，也可能导致重建误差很大；这类对噪声/建模误差高度敏感的现象是工程应用中的核心困难之一，正则化与迭代稳定化方法用于获得稳定解 [5]。

#### (2) 观测算子的物理建模与误差来源

“观测一致性（data consistency）”是否成立，首先取决于 $H$ 是否真实刻画了传感器生成 $y$ 的物理过程。数据同化讲义常将观测误差分解为仪器误差与代表性/表征误差，并指出若对映射 $H$ 的知识不完备，还会出现观测算子建模误差 [2]。代表性误差可由“观测尺度/采样体积/变量定义”与模型离散状态之间的不匹配引起，并可从“真吸引子与预报吸引子之间缺乏可逆映射”解释其结构性来源 [7]。

工程中，观测算子往往不等同于“下采样”。例如遥感辐射同化中，$H$ 需要包含辐射传输计算并依赖仪器特性；观测误差不仅包含仪器噪声，也必须计入观测算子不确定性、数据筛选误差与代表性误差；错误设定观测误差可能导致分析场劣于背景场 [8]。在空气质量等应用中，观测误差同样需要计入尺度错配、子网格变率与缺失过程等代表性误差，且误差往往需要估计 [9]。

上述证据共同指向：若不显式建模 $H$ 的物理口径，重建与评测可能在“逐点误差下降”与“观测不一致不可部署”之间脱节。

#### (3) 观测一致性的理论地位：误差界与算子错配分析

**从贝叶斯/变分视角理解一致性约束**：在高斯观测误差假设下，似然项可写为
\[
p(y|x)\propto \exp\!\left(-\tfrac12\|y-H(x)\|_{R^{-1}}^2\right),
\]
对应 MAP/变分目标中的核心观测残差项；当 $H$ 非线性时需要线性化与伴随来计算梯度，说明 $H$ 是算法成立的基础部件 [2,3]。这一结构与深度重建中的“数据一致性层（DC layer）”同构：通过显式数据一致性步骤将输出拉回由前向模型定义的观测可行集，从而降低“看似合理但不满足测量方程”的风险 [10,11]。

![图 2-4: 贝叶斯反演视角下的观测一致性约束。重建解 $\hat{u}$ 是物理先验（Prior）与数据似然（Likelihood）的权衡结果。数据似然项 $p(y|u)$ 由观测算子 $H$ 定义，强制解落在观测流形附近。若训练中使用了错误的算子 $\tilde{H}$（Operator Mismatch），会导致似然流形发生偏移（Manifold Drift），从而产生系统性的后验偏差。](images/fig2-4_bayesian_consistency.png)

**基本误差界（条件化有界性）**：设真实观测由
\[
y=H(u)+\eta,\quad \|\eta\|\le \delta
\]
生成。对任意重建结果 $\hat{u}$，由三角不等式有
\[
\|H(\hat{u})-H(u)\|\le \|H(\hat{u})-y\|+\|y-H(u)\| = H_{\mathrm{err}}+\|\eta\|\le H_{\mathrm{err}}+\delta.
\]
若进一步存在先验集合 $S$（由物理约束、正则化或数据流形定义），并且在 $S$ 上满足一种“受限稳定性/可逆性”（例如受限 Lipschitz 逆）：
\[
\|u_1-u_2\|\le L_S\|H(u_1)-H(u_2)\|,\quad \forall u_1,u_2\in S,
\]
则在 $\hat{u},u\in S$ 时有
\[
\|\hat{u}-u\|\le L_S\|H(\hat{u})-H(u)\|\le L_S\,(H_{\mathrm{err}}+\delta).
\]
该形式可视为 Hadamard“稳定性”在先验集合上的版本：一旦受限稳定性成立，控制观测残差即可控制重建误差；反之，缺乏稳定性时，单纯最小化经验重建误差不足以保证可部署鲁棒性 [6,15]。

**观测算子错配为何破坏误差上界**：考虑训练/推理使用的算子为 $\tilde{H}$，但真实观测由 $H$ 生成。若某方法在结构上保证
\[
\|\tilde{H}(\hat{u})-y\|\le \varepsilon,
\]
则真实口径下残差满足
\[
\|H(\hat{u})-y\|\le \|H(\hat{u})-\tilde{H}(\hat{u})\|+\|\tilde{H}(\hat{u})-y\|.
\]
若进一步假设 $H,\tilde{H}$ 为有界线性算子，则
\[
\|H(\hat{u})-y\|\le \|H-\tilde{H}\|\,\|\hat{u}\|+\varepsilon.
\]
该不等式给出两个直接结论：

1. 若 $\|H-\tilde{H}\|$ 不可忽略，则即便 $\varepsilon$ 很小，真实口径残差也可能很大；
2. 若缺乏对 $\|\hat{u}\|$ 的显式约束（先验/正则化不足），则 $\|H-\tilde{H}\|\,\|\hat{u}\|$ 项在最坏情况下可无界，从而无法获得统一鲁棒的误差上界。

![图 2-1: 观测算子误差与重建误差的理论界分析。当训练阶段的退化算子 $\tilde{H}$ 与真实观测算子 $H$ 存在偏差时（Operator Mismatch），即使模型在训练域内实现了完美拟合（$\varepsilon \to 0$），在真实观测域的误差 $\|H(\hat{u})-y\|$ 仍受到算子偏差 $\|H-\tilde{H}\|$ 的支配，且可能随模型输出能量 $\|\hat{u}\|$ 无界增长。这从理论上证明了 $DC \equiv H$ 的必要性。](images/fig2-1_error_bound_theory.png)

因此，“观测一致性必须以真实 $H$ 为口径（$DC\equiv H$）”具有明确的理论动机：要在真实观测空间保证误差可控，DC 约束所使用的算子必须与数据生成口径一致，或至少需要显式估计/校正算子误差。相关理论与方法在多个方向均有支撑：不完美前向算子会改变可验证的收敛与误差界结论 [12]；观测模型误差需要在同化/过滤框架内迭代校正以提高一致性 [13]；学习型逆问题中对算子进行学习式校正或残差补偿可提升稳定性与跨分布鲁棒性 [14]；数据一致性网络可与经典正则化结合并给出随噪声收敛的分析 [10]。

---

### 参考文献

[1] Arridge S, Maass P, Oktem O, Schönlieb C-B. Solving inverse problems using data-driven models[J]. *Acta Numerica*, 2019, 28: 1–174. doi:10.1017/S0962492919000059.  
    [PDF](https://assets.cambridge.org/97811084/78687/excerpt/9781108478687_excerpt.pdf)

[2] Bocquet M. *Introduction to the principles and methods of data assimilation*[R]. Lecture notes, 2025.  
    [PDF](https://cerea.enpc.fr/HomePages/bocquet/teaching/assim-mb-en-0.52.pdf)

[3] Rabier F. *Variational data assimilation: theory and overview*[R]. ECMWF Seminar 2003.  
    [PDF](https://www.ecmwf.int/sites/default/files/elibrary/2003/76079-variational-data-assimiltion-theory-and-overview_0.pdf)

[4] Hadamard J. *Sur les problèmes aux dérivées partielles et leur signification physique*[R]. Princeton University Bulletin, 1902.  
    [PDF](https://illposed.net/hadamard.pdf)

[5] Clason C. *Regularization of Inverse Problems*[R]. arXiv:2001.00617, 2020.  
    [arXiv](https://arxiv.org/abs/2001.00617)

[6] Platte R B. *Condition Numbers and Inverse Problems*[R]. Arizona State University, 2017.  
    [PDF](https://math.asu.edu/sites/g/files/litvpz216/files/rtgintro_inv_problems.pdf)

[7] Hodyss D, Nichols N K. The error of representation: basic understanding[J]. *Tellus A*, 2015, 67: 24822. doi:10.3402/tellusa.v67.24822.  
    [Link](https://tellusjournal.org/articles/10.3402/tellusa.v67.24822)

[8] ECMWF Training Material. *Data assimilation algorithms and key elements*[R]. (Satellite radiances; observation errors and representativeness).  
    [PDF](https://www.ecmwf.int/sites/default/files/SAT_TC_Data_assimilation_key_elements_TM.pdf)

[9] Ménard R, Cossette J-F, Deshaies-Jacques M. On the Complementary Role of Data Assimilation and Machine Learning: An Example Derived from Air Quality Analysis[J]. 2020.  
    [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7304728/)

[10] Boink Y E, Haltmeier M, Holman S, Schwab J. Data-consistent neural networks for solving nonlinear inverse problems[J]. *Inverse Problems and Imaging*, 2023, 17(1): 203–229. doi:10.3934/ipi.2022037.  
     [Link](https://www.aimsciences.org/article/doi/10.3934/ipi.2022037)

[11] Liang D, Cheng J, Ke Z, Ying L. Deep Magnetic Resonance Image Reconstruction: Inverse Problems Meet Neural Networks[J]. *IEEE Signal Processing Magazine*, 2020, 37(1): 141–151. doi:10.1109/MSP.2019.2950557.  
     [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7977031/)

[12] Bungert L, Burger M, Korolev Y, Schönlieb C-B. Variational regularisation for inverse problems with imperfect forward operators and general noise models[J]. 2020.  
     [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8208616/)

[13] Hamilton F, Berry T, Sauer T. Correcting Observation Model Error in Data Assimilation[R]. arXiv:1803.06918, 2018.  
     [arXiv](https://arxiv.org/abs/1803.06918)

[14] Lunz S, Hauptmann A, Tarvainen T, Schönlieb C-B, Arridge S. On Learned Operator Correction in Inverse Problems[J]. 2021.  
     [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617273/)

[15] Hansen P C. *Inverse Problems*[R]. Meet DTU lecture slides, 2019.  
     [PDF](https://www.imm.dtu.dk/~pcha/MeetDTU.pdf)

## 2.2 观测算子 $H$ 的物理建模

### 2.2.1 观测算子建模的目标与原则

在稀疏观测驱动的物理场重建中，观测算子 $H$ 并非“数据预处理”的附属环节，而是决定训练口径、评测口径与部署口径能否统一的核心对象。现代数据同化框架将 $H$ 视为从“模型状态空间”到“观测空间”的映射：即便观测量与模型变量同名（例如温度），仍需要通过 $H$ 将有限分辨率模型插值到观测位置；当观测属于间接量（例如卫星辐射），$H$ 还需要包含辐射传输等复杂物理过程与仪器特性描述。[1]

工程上“合成数据—真实数据”鸿沟的一个常见根源来自训练阶段对 $H$ 的过度简化：若以“理想点采样/理想下采样”替代真实传感器的空间响应（spatial response）与测量支撑域（support/footprint），则模拟观测相较真实观测更“锐”、更接近点值，导致网络学习到的逆映射在真实口径下产生系统性失配。面向遥感高光谱重采样的研究指出：若忽略传感器点扩散函数（PSF）而直接采用最近邻/像元聚合/三次卷积等方式生成低分辨观测，会得到不真实的锐利影像；采用传感器 PSF 卷积生成模拟观测有助于获得更物理一致的退化口径。[2]

因此，本节将 $H$ 组织为“降质过程 + 几何过程”的分解，并强调两条建模原则：  
(1) **物理一致性优先**：$H$ 应尽可能贴合真实成像/测量链路，至少覆盖“支撑域积分（空间平均）→采样/重采样→噪声”。[1,2]  
(2) **可分析性与可实现性**：$H$ 需尽可能保持可微与可控的数学性质（例如有界线性，或在工作点附近可线性化），以便后续一致性理论用于误差传播与上界推导，并支持 DC（data consistency）结构化实现。[3]

---

### 2.2.2 标准化算子分解：降质与几何，以及数学性质与扩展

#### (1) 降质模型：空间响应、抗混叠与离散采样

**从空间响应到卷积分解**  
大量成像与遥感测量中，一个像元值通常不对应“某点真值”，而对应目标场在一定空间支撑域上的加权积分或平均。例如，早期遥感资料指出：扫描仪一个像元的辐亮度来自瞬时视场（IFOV）覆盖区域的积分。[4] 相关科普/工程解释也常用“地面足迹（footprint）接收并积分辐射”描述空间支撑域效应。[5]

在局部可近似为线性、空间不变（LSI）的条件下，空间积分/扩散可由点扩散函数 PSF 表达；传感器空间响应资料给出典型形式：观测量可写为地表能量分布与 PSF 的卷积积分，并满足
\[
\iint \mathrm{PSF}(x,y)\,dx\,dy = 1
\]
的归一化约束。[6] 在频域视角下，PSF 的二维傅里叶变换对应光学传递函数 OTF，OTF 幅值对应调制传递函数 MTF；并可写成“图像频谱 = 系统 OTF × 目标频谱”的乘积关系，从而将“空间响应导致高频衰减”形式化为系统低通特性。[7]

基于上述物理与算子抽象，可将“预滤波 + 降采样”作为对成像链路的标准化近似：
\[
y^{\mathrm{SR}} = D_s\!\left(G_\sigma * u\right) + \eta,
\]
其中 $G_\sigma$ 表示 PSF 的参数化近似（常用正态核等），$D_s$ 表示步长为 $s$ 的抽取/重采样算子，$\eta$ 为观测误差项。遥感空间统计建模中也常以二维正态或相近函数近似像元支撑域权重函数。[8] 该抽象与超分辨率成像的经典观测模型同构：综述工作将低分辨率传感器 PSF 作为“空间平均算子”，并将模糊/运动/降采样统一写为矩阵观测模型，从而为后续把 $H$ 视为（近似）线性算子并开展一致性与误差界分析提供支撑。[3]

**抗混叠：预滤波的必要性**  
当分辨率从高网格降至低网格时，若缺乏带限预滤波，高频能量会在采样后折叠到低频并产生混叠（aliasing）。采样定理的工程化阐释指出：实际信号往往包含宽频成分，采样前通常需要抗混叠滤波以保证进入采样器的频谱受控。[9] 在二维物理场/遥感影像中，倍率 $s$ 的降采样降低奈奎斯特上限；若未抑制超出新奈奎斯特上限的高频分量，会产生不可逆混叠失真，进而造成“训练阶段理想退化”与“真实传感器带限退化”在频谱层面的口径错配。[9,10] 因此，用 $G_\sigma * u$ 显式模拟“可解析尺度上的空间平均/低通衰减”更贴近真实链路，并使“高频信息缺失”具备物理可解释性。[7,9]

**$\sigma$ 与 $s$：从经验超参到可校准参数**  
倍率 $s$ 决定目标采样间隔与可达空间频率上限；$\sigma$ 更应优先由传感器空间响应指标确定（PSF/LSF/MTF、FWHM 与像元尺寸比值、MTF@Nyquist 等），或由标定数据估计。传感器空间响应资料给出以 PSF/LSF、Nyquist 频率与 MTF 指标刻画空间响应的参数化路径，并展示以正态型 LSF/PSF 作为评价基准的做法。[6] 遥感统计建模也指出：权重函数可来自理论推导或实验测量，常见形式包括均匀、指数、正态与 sinc 等。[8] 因此，论文表述可采用“$s$ 决定目标带宽上限，$\sigma$ 由 PSF 指标或数据拟合确定”的策略；缺少精确标定时，正态核可作为工程近似，并通过匹配 MTF@Nyquist 或 FWHM/像元比等方式选取 $\sigma$。[6,8]

**噪声项：从白噪声到观测误差预算**  
将 $\eta$ 简化为加性独立同分布噪声（例如 $\mathcal{N}(0,\sigma^2 I)$）便于推导，但面向部署更合适的表述应将 $\eta$ 视为“观测误差”统称：包含仪器误差与代表性误差（representativeness/representation error），必要时还需计入观测算子误差。变分同化综述明确指出：观测误差协方差 $R$ 通常包含仪器误差与代表性误差，观测项以 $\|y-Hx\|_{R^{-1}}^2$ 进入代价函数。[11] 代表性误差的资料也将“代表性误差 + 仪器误差”作为观测误差预算的重要组成，并给出统一处理视角。[12] 近期研究进一步系统讨论代表性误差与观测算子误差的关系，强调其在同化系统误差预算中的关键地位。[13]

---

#### (2) 几何模型：视窗裁剪、稀疏站点与非均匀采样

**有限视窗观测与局部裁剪**  
实验流体力学、局部窗口成像与部分区域监测中，观测常具有明确的可视域/可测域边界。以 PIV 平面测量为例，资料指出 PIV 可获取平面瞬时速度场，但空间分辨率受激光片厚度与分析中的询问窗（interrogation window）限制，询问窗尺度常在 8–16 像素量级。[14] 在算子层面，观测不仅体现“截取区域”，还隐含局部相关/局部平均的处理链路；因此，将视窗裁剪写为算子 $C_{h,w}$ 并配合掩码属于合理的标准化起点，高精度建模时可进一步将裁剪与局部平均核耦合。[14]

标准化表达可写为
\[
y^{\mathrm{Crop}} = C_{h,w}(u)\odot m_{\mathrm{crop}} + \eta,
\]
其中 $m_{\mathrm{crop}}$ 为几何掩码。若将 $C_{h,w}$ 理解为“选择子区域并嵌入观测空间”的线性映射，则其本质属于投影/选择算子，呈现典型的“信息不可逆丢弃”特征，与逆问题不适定性相一致。[11]

**非均匀掩码与站点型观测**  
站点、浮标、测井或稀疏传感网络的观测可近似为“若干位置点值（或小支撑域平均）”。数据同化实现中，即便观测量与模型变量一致，仍需要 $H$ 将模型状态插值到观测位置；当观测密度不均匀时，插值/重采样本身构成观测算子的核心组成部分。[1] 在规则网格表示下，将稀疏观测近似为逐元素掩码
\[
y^{\mathrm{Mask}} = u\odot m_{\mathrm{sparse}} + \eta
\]
等价于将 $H$ 简化为对网格节点的选择矩阵（对角投影），有利于将复杂缺测模式抽象为结构化输入，从而在统一框架下比较不同观测几何对重建的影响。[1,11]

更一般的站点观测更接近连续形式
\[
y_i = \int_{\Omega} h_i(x)\,u(x)\,dx + \eta_i,
\]
其中 $h_i(x)$ 表示站点测量支撑域/权重函数；仅当支撑域远小于网格尺度时，才可近似为狄拉克型点采样。遥感与空间统计建模将该类权重函数视为跨尺度变换的关键对象，并指出权重函数选择会影响不确定性传播与统计推断。[8]

---

#### (3) 标准化算子组合与数学性质

将“降质 + 几何”组合，一个面向训练/评测/部署口径统一的标准化观测算子可写为（单帧）
\[
H(u)= M\,D_s\,(K*u),
\]
其中 $K$ 表示 PSF（可用正态核等参数化近似），$D_s$ 表示重采样/降采样，$M$ 表示裁剪与掩码（几何投影到观测支撑集）。该分解与成像 SR 的观测模型以及遥感“支撑域卷积 + 采样”的解释一致。[2,3,8]

在离散张量的 $\ell_2$（Frobenius）度量下，上述子算子通常可视为有界算子，便于后续开展一致性误差传播与上界推导：  
- **卷积算子有界性**：对 $K\in L_1$，映射 $f\mapsto K*f$ 在 $L_p$ 上有界，满足 $\|K*f\|_p\le \|K\|_1\|f\|_p$（Young 不等式）。[15] 离散实现中若核 $K$ 归一化（求和为 1），卷积在能量意义上往往不扩张，符合“支撑域空间平均”物理直觉。  
- **掩码与裁剪的非扩张性**：二值掩码对应对角投影，裁剪对应选择投影，均倾向于丢弃信息而非放大信息，从而使整体 $H$ 的算子范数具备可控上界，便于推导 $\|H(\hat{u})-H(u)\|$ 与 $\|\hat{u}-u\|$ 的关系并建立误差传播界。[11]

变分同化代价函数中的观测项采用 $\|y-Hx\|_{R^{-1}}^2$ 形式；此处将 $H$ 明确为具有物理含义的有界算子，有助于把“观测一致性误差”与“统计意义下的似然/数据项”在同一算子框架内统一表述。[11]

![图 2-2: 统一观测算子 $H$ 的物理建模与分解示意图。我们将观测过程标准化为“降质 (Degradation)”与“几何 (Geometry)”两个子模块的串联。降质模块包含空间响应卷积（PSF）与抗混叠滤波；几何模块负责离散采样、视窗裁剪与非均匀掩码生成。通过参数化 $G_\sigma, D_s, M$，实现了对各类稀疏观测场景（SR, Crop, Inpainting）的统一描述。](images/fig2-2_operator_decomposition.png)

---

#### (4) 更真实的观测口径：空间变异、扫描效应与非线性观测

上述 $H(u)=D_s(K*u)+\eta$ 的优点在于结构清晰、可实现、可微且便于理论分析；工程传感器往往满足更复杂条件，论文可在本节末明确扩展方向及对后续理论适用范围的影响：

1. **空间变异 PSF 与扫描效应**：PSF 未必严格空间不变；扫描型传感器可能沿轨/横轨方向具有不同 LSF，且空间响应随视角与扫描位置变化。传感器空间响应资料展示不同传感器的 LSF/PSF 与以 Nyquist 频率归一的 MTF 指标化方式。[6] SR2 工作也指出：PSF 可由数据采集参数推导并用于更逼真的空间响应降质。[2]

2. **非线性观测算子**：当观测属于间接量（如卫星辐射），$H$ 可能成为显著非线性算子，通常需要辐射传输模型与多层廓线才能计算观测波段辐射。[1] 若后续理论依赖“$H$ 为有界线性算子”，需要在章节中标注适用范围，并给出“线性或可线性化有效观测算子”上的主结论，再讨论非线性情形的局部线性化扩展。

3. **观测算子错配与代表性误差的结构性影响**：一维研究指出：观测算子不完美会引入相关误差，使得即便观测密度提高，分析误差也未必趋于零；完美观测算子条件下误差可随密度提升渐近下降。[16] 该结论支撑“观测一致性（DC 口径等价于真实 $H$）”的必要性，并与代表性误差/观测算子误差的现代讨论形成呼应。[13]

---

#### 参考文献

[1] ECMWF. *Assimilation algorithms (Data Assimilation algorithms)*[R].  
[PDF](https://www.ecmwf.int/sites/default/files/elibrary/2008/16931-assimilation-algorithms.pdf)

[2] Spatial response resampling (SR2): Accounting for the spatial point spread function in hyperspectral image resampling[J]. *ScienceDirect*, 2023.  
[Link](https://www.sciencedirect.com/science/article/pii/S2215016123000031)

[3] Park S C, Park M K, Kang M G. Super-resolution image reconstruction: A technical overview[J]. *IEEE Signal Processing Magazine*, 2003.  
[PDF](https://cse.buffalo.edu/courses/cse725/peter/Park_2003.pdf)

[4] NASA. Remote sensing scanner / IFOV integration material[R]. 1976.  
[PDF](https://ntrs.nasa.gov/api/citations/19760025534/downloads/19760025534.pdf)

[5] GIM International. Understanding Spatial Resolution[EB/OL].  
[Link](https://www.gim-international.com/content/article/understanding-spatial-resolution)

[6] Lin G. *Sensor Spatial Response (PSF/LSF/MTF; Nyquist metrics)*[R]. NASA NTRS, 2024.  
[PDF](https://ntrs.nasa.gov/api/citations/20250000520/downloads/2024-VH-RODA_sensorSpatialResponse_GLin.pdf)

[7] NASA. *Optical Transfer Functions, Pointing and Requirements*[R]. 2021.  
[PDF](https://ntrs.nasa.gov/api/citations/20210020948/downloads/UPDATED%20Optical%20Transfer%20Functions%20Pointing%20and%20Requriements.pdf)

[8] Remote Sensing (MDPI). An Approach for Spatial Statistical Modelling Remote Sensing Data of Land Cover by Fusing Data of Different Types[J]. 2025.  
[Link](https://www.mdpi.com/2072-4292/17/1/123)

[9] Texas Instruments. *AN-236 An Introduction to the Sampling Theorem (Rev. C)*[R].  
[PDF](https://www.ti.com/lit/pdf/snaa079)

[10] MIT. Shannon sampling / noise notes[EB/OL].  
[PDF](https://fab.cba.mit.edu/classes/S62.12/docs/Shannon_noise.pdf)

[11] Rabier F. *Variational data assimilation: theory and overview*[R]. ECMWF Seminar, 2003.  
[PDF](https://www.ecmwf.int/sites/default/files/elibrary/2003/76079-variational-data-assimiltion-theory-and-overview_0.pdf)

[12] NOAA. Observation/representation error material (repository)[R].  
[PDF](https://repository.library.noaa.gov/view/noaa/11487/noaa_11487_DS1.pdf)

[13] Janjić T, et al. On the representation error in data assimilation[J]. *Quarterly Journal of the Royal Meteorological Society*, 2018.  
[Link](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3130)

[14] IITK reading material. PIV interrogation window / resolution notes[R].  
[PDF](https://www.iitk.ac.in/che/PG_research_lab/pdf/resources/MPIV-reading-material.pdf)

[15] University of Washington. Lecture 2: Convolution (Young inequality notes)[R].  
[PDF](https://sites.math.washington.edu/~hart/m526/Lecture2.pdf)

[16] Liu Z Q, Rabier F. The interaction between model resolution, observation resolution and observation density in data assimilation: A one-dimensional study[J]. 2002.  
[PDF](https://rainbow.ldeo.columbia.edu/~alexeyk/Papers/LiuRabier2002.pdf)

## 2.3 观测一致性理论 (Consistency Theory)

本节将阐述本研究的核心理论基础——**观测一致性（Observation Consistency）**。我们首先给出定义，然后通过两个命题证明其在误差控制中的必要性与鲁棒性。

### 2.3.1 定义：同源性约束

在深度学习训练中，通常需要构造“输入-标签”对 $(x, y)$。生成训练输入的过程被称为训练退化算子（Training Degradation Operator），记为 $DC$。

**定义 2.1（观测一致性）**：
当且仅当训练退化算子 $DC$ 与测试/部署阶段的真实观测算子 $H$ 在数学形式、参数配置及边界条件上完全一致时，称该系统满足观测一致性，即：
$$
DC \equiv H.
$$

这一约束要求在代码实现层面，训练数据合成与测试评测必须复用同一算子实例，以消除隐性的算子错配（Operator Mismatch）。

### 2.3.2 命题 2.3：一致性误差的控制定理

我们将观测一致性误差 $H_{\text{err}}$ 的控制问题形式化为如下定理。

**定理 2.3（一致性误差上界）**：
设 $H: \mathcal{U} \to \mathcal{Y}$ 为有界线性观测算子，$\mathcal{F}_\theta$ 为重建模型。令 $\hat{u} = \mathcal{F}_\theta(y)$ 为重建解，$u$ 为真值。若模型满足重建误差界 $\|\hat{u} - u\|_\mathcal{U} \le \epsilon$，则观测一致性误差满足：

$$
\|H(\hat{u}) - y\|_\mathcal{Y} \le \|H\|_{\text{op}} \cdot \epsilon + \|\eta\|_\mathcal{Y},
$$

其中 \|H\|_{\text{op}} = \sup_{v \neq 0} \frac{\|Hv\|}{\|v\|} 为算子范数。

**物理直觉分析**：
算子范数 \|H\|_{\text{op}} 在物理上代表了观测系统对输入扰动的“放大敏感度”。定理 2.3 表明，即使重建模型 \mathcal{F}_\theta 在全场意义上逼近了真值（即 \epsilon \to 0），最终的观测一致性误差仍受限于系统噪声 \|\eta\|。更关键的是推论部分：当存在训练-测试算子偏差 \delta > 0 时，这一偏差会作为乘性因子与解的能量 \|\hat{u}\| 耦合。这意味着，**对于高能量的物理场（如强湍流或高压场），微小的算子错配 \delta 会被显著放大，导致巨大的系统性观测误差**。这从理论上解释了为何在复杂流场重建中，简单的近似退化模型往往失效。

**推论**：
若训练阶段采用近似算子 $DC \approx H$，且存在算子偏差 $\|H - DC\|_{\text{op}} = \delta$，则测试阶段的一致性误差下界为：
$$
\|H(\hat{u}) - y\|_\mathcal{Y} \ge \left| \|DC(\hat{u}) - y\|_\mathcal{Y} - \delta \|\hat{u}\|_\mathcal{U} \right|.
$$
该推论从理论上证明了：若训练算子 $DC$ 与物理算子 $H$ 不一致（$\delta > 0$），即使训练损失降至 0，测试误差仍存在不可消除的系统性偏差（Systematic Bias）。

**物理意义**：该命题表明，只有当训练目标（最小化 $\|\hat{u}-u\|$）与评测口径 $H$ 保持一致时，提升重建精度才能理论上保证评测误差的下降。它是“护栏”理论的基础，保证了优化方向的正确性。

### 2.3.3 命题 2.2：口径错配的鲁棒性分析

**命题 2.2（错配误差下界）**：
若训练阶段使用退化算子 $DC$ 进行约束，且 $DC \neq H$（存在口径错配），则评测口径误差包含不可消去的系统性偏差。具体地：
$$
\|H(\hat{u}) - y\|_2 \ge \left| \|DC(\hat{u}) - y\|_2 - \|(H - DC)(\hat{u})\|_2 \right|.
$$
特别地，当模型在训练域完美过拟合（即 $\|DC(\hat{u}) - y\| \to 0$）时，评测误差下界为：
$$
\|H(\hat{u}) - y\|_2 \to \|(H - DC)(\hat{u})\|_2.
$$

**分析：误差的不可约性 (Irreducibility)**
该不等式揭示了一个残酷的事实——算子错配项 $\|(H - DC)(\hat{u})\|$ 构成了模型性能的“理论天花板”。无论我们拥有多少训练数据（Data Scale），无论神经网络有多深（Model Capacity），只要训练口径 $DC$ 与真实口径 $H$ 不对齐，这一误差项始终存在且无法通过优化消除。我们将这种误差定义为**“口径诱导的系统性偏差 (Consistency-induced Systematic Bias)”**。它与过拟合或欠拟合无关，而是属于方法论层面的**模型设定错误 (Model Misspecification)**。这解释了为何现有文献中常出现“训练 loss 很低，但实际部署效果差”的现象，因此，坚持 $DC \equiv H$ 是保证模型泛化鲁棒性的前提。

### 2.3.4 理论验证实验设计

为验证上述理论命题的有效性，我们设计如下验证性实验（将在第 4 章详细展开）：

1.  **一致性敏感度分析**：
    构建 $DC \neq H$ 的对照组（如 $H$ 使用高斯降采样，而 $DC$ 使用双线性插值），对比其在测试集上的 $H_{\text{err}}$ 与 $\text{Rel-L2}$ 指标，验证命题 2.2 中的系统性偏差。

2.  **算子等价性审计**：
    在工程实现中引入“阻断式审计”，随机抽取样本检查 $\text{MSE}(H(u), DC(u))$，确保数值误差低于 $10^{-8}$，从实验上保证定义 2.1 的严格执行。

## 2.4 观测一致性优先框架的总体框架与端到端流程

### 2.4.1 框架定位与研究动机

“观测一致性优先（Consistency-First）”框架将稀疏观测重建明确视为**前向观测模型已知（或可建模）的逆问题**：观测数据由真实物理场经观测算子映射并叠加噪声得到
\[
y = H(u) + \eta .
\]
在该设定下，模型不仅需要在状态空间输出“视觉上合理”的 \(\hat{u}\)，还需要满足更基本的测量事实：\(\hat{u}\) 经 \(H\) 回投影后能够解释观测 \(y\)，即 \(H(\hat{u}) \approx y\)。将该约束显式嵌入学习回路，与模型驱动深度学习（model-based deep learning）在医学成像与一般逆问题中的主流设计一致：网络结构或训练目标中包含数据一致性（data consistency）步骤，使方法具备更明确的可解释性与稳定性抓手。[1]

该选择直接回应“算子错配（operator mismatch）”的工程痛点。针对线性逆问题，已有研究系统分析深度网络在测试阶段可能出现的“测量不一致（measurement inconsistency）”：网络输出代入真实测量模型后无法复现输入测量，导致性能退化；并指出通过显式强制一致性或在结构/后处理中加入一致性步骤可显著改善结果。[2]

从方法谱系看，Consistency-First 也可视为“数据项—先验项分离”的现代重述：经典 Plug-and-Play（PnP）与一系列展开式方法在迭代中交替执行数据保真/一致性步与先验/去噪步，把前向模型 \(H\) 作为迭代护栏，再用学习模块表达复杂先验。[3]

---

### 2.4.2 端到端闭环流程与关键接口

Consistency-First 的端到端闭环包含四个阶段：**输入构造 → 编码与演化 → 解码与重建 → 一致性校验（训练闭环）**。该流程与数据一致性网络（data-consistent neural network）的“两步结构”（学习式重建 + 与前向算子一致性校正/约束）同构，并已在正则化理论框架下给出噪声收敛意义上的分析，强调一致性约束对稳定性的作用。[4]

#### (1) 决定性接口：\(DC \equiv H\) 的同源复用

闭环的决定性接口是：训练回路中的退化/观测算子 \(DC\) 需要与评测/部署口径的 \(H\) **同源复用**（数学形式、参数配置、边界处理、插值策略与数值实现细节同时一致）。该要求对应对测量不一致风险的直接治理：若训练约束的是 \(DC(\hat{u})\approx y\)，部署要求的是 \(H(\hat{u})\approx y\)，则残差的不可约项会被 \((H-DC)\hat{u}\) 主导，从而形成系统性偏差。[2,4]

#### (2) 输入构造：把“观测值 + 几何 + 坐标”显式化

输入构造的目标是将观测值与观测几何显式提供给网络，降低网络隐式学习采样机制的负担。

- **观测值与掩码通道**：将稀疏观测 \(y\) 与掩码 \(m\) 拼接输入，使“哪些像素可信/可用”成为网络可利用的显式信息。部分卷积（Partial Convolution）将掩码纳入前向传播并层间更新掩码，用于处理不规则孔洞缺失，体现“掩码作为观测几何显式输入”的可行性。[5]

- **坐标网格与坐标通道**：在几何敏感任务中，卷积的平移不变性并不充分。CoordConv 研究表明：对需要位置敏感输出或坐标变换的任务，显式提供 \((x,y)\) 坐标通道可显著改善学习能力；将 \(x_{\mathrm{grid}}\) 作为输入通道有助于网络处理裁剪边界、站点分布与非均匀掩码。[6]

- **傅里叶特征编码（可选）**：神经网络存在高频学习困难（spectral bias）。Fourier Features 给出可操作路径：对输入坐标施加傅里叶特征映射可提升高频函数拟合能力，并给出相应的理论解释；将 Fourier features 作为可选 PE 通道可增强稀疏观测下的细尺度恢复能力。[7]

- **插值初值 \(y'\)**：先对 \(y\) 做双线性/双三次插值获得粗场 \(y'\)，作为低频背景注入；插值提供连续性与大尺度轮廓，网络在一致性护栏下补全细节。该思路与超分辨任务中“插值基线（如 bicubic）作为参照/初值”的工程实践一致。[8]

#### (3) 编码—演化—解码：空间表征与时空依赖

中间网络模块承担两类任务：建立多尺度空间表征以应对欠定性；在时变场中建模时空依赖以抑制时间误差累积与噪声传播。模块化三段式结构为
\[
(y, m, x_{\mathrm{grid}}, \mathrm{PE}, y') \xrightarrow{\ \mathrm{Encoder}\ } z_{1:K}
\xrightarrow{\ \mathrm{Temporal\ Module}\ } \tilde{z}_{1:K}
\xrightarrow{\ \mathrm{Decoder}\ } \hat{u}_{1:K}.
\]

- **时序模块候选 1：ConvLSTM**。ConvLSTM 在 LSTM 转移中引入卷积，保持空间结构并学习时序动态，最初用于降水临近预报，适合建模局部对流、平移传播等动力学特征。[9]

- **时序模块候选 2：Video Swin Transformer**。Video Swin 在时空 Transformer 中引入窗口注意力与 shifted windows 机制，以局部性归纳偏置获得更佳速度—精度折中，并以分层结构支持多尺度表示，适合在潜在空间建模长程依赖。[10]

#### (4) 一致性校验：作为训练目标与结构模块的双重实现

Consistency-First 的决定性步骤是一致性校验：对重建场 \(\hat{u}\) 施加 \(DC\) 得到重投影观测 \(\hat{y}=DC(\hat{u})\)，并强制 \(\hat{y}\approx y\)。

- **作为训练目标的数据项**：将 \(\|DC(\hat{u})-y\|\) 纳入损失，等价于在学习式重建中显式加入数据保真项（data fidelity），使网络在强表达能力下仍需解释观测口径。[4]

- **作为结构中的数据一致性块**：模型驱动网络可将一致性实现为可解释的数值子步骤并嵌入网络。MoDL 在网络中交替执行 CNN 正则化与基于前向算子的 DC 子问题（可用共轭梯度等块求解），强调“在网络内部强制数据一致性”。[1]

---

### 2.4.3 工程可复现性与一致性审计要点

Consistency-First 的工程优势来自“把 \(H\) 做实”。实现层面需要把观测算子封装为**可复用、可审计、可微分**的模块，并避免数值细节破坏口径一致。

#### (1) 算子封装与参数固化

建议将退化算子封装为 PyTorch 的 `nn.Module`，并将静态核/掩码/坐标网格等以 buffer 形式注册，使其随 `state_dict` 保存并在训练/评测复用同一参数对象。[11,12]

#### (2) 数值确定性与插值算子风险控制

若 \(DC\) 依赖 `torch.nn.functional.interpolate` 并参与反向传播，需关注 CUDA 下可能出现的非确定性梯度与 float16 梯度不准确提示；建议优先采用 float32，并记录确定性设置（例如 `torch.use_deterministic_algorithms` 的使用策略）以避免实现噪声干扰模型对比结论。[13,14]

#### (3) 下采样口径与跨库一致性

若 SR 任务使用 OpenCV 的 `INTER_AREA` 作为缩小插值策略，OpenCV 文档与教程均将其推荐用于图像缩小；该选择可被解释为对“空间积分/平均效应”的离散近似。关键要求在于：训练与评测必须严格复用同一降采样口径与实现路径，避免出现跨库差异导致的隐性错配。[15]

#### (4) 边界条件审计

边界条件属于最容易被忽略且最容易造成口径错配的细节之一。若采用 reflection padding，需要把 `ReflectionPad2d` 的配置作为观测算子的一部分固化并纳入审计；否则边界处一致性残差可能呈现系统性偏差。[16]

---

#### 参考文献

[1] Aggarwal H K, Mani M P, Jacob M. **MoDL: Model Based Deep Learning Architecture for Inverse Problems**[R]. arXiv:1712.02862, 2017.  
[arXiv](https://arxiv.org/abs/1712.02862)

[2] Vella M, Mota J F C. **Overcoming Measurement Inconsistency in Deep Learning for Linear Inverse Problems: Applications in Medical Imaging**[R]. 2020.  
[PDF](https://jmota.eps.hw.ac.uk/documents/Vella20-OvercomingMeasurementInconsistencyInDeepLearningForLinearInverseProblems.pdf)

[3] Venkatakrishnan S V, Bouman C A, Wohlberg B. **Plug-and-Play Priors for Model Based Reconstruction**[J]. 2013.  
[PDF](https://brendt.wohlberg.net/publications/pdf/venkatakrishnan-2013-plugandplay2.pdf)

[4] Boink Y E, Haltmeier M, Holman S, Schwab J. **Data-consistent neural networks for solving nonlinear inverse problems**[J]. *Inverse Problems and Imaging*, 2023.  
[Link](https://www.aimsciences.org/article/doi/10.3934/ipi.2022037)

[5] Liu G, Reda F A, Shih K J, Wang T C, Tao A, Catanzaro B. **Image Inpainting for Irregular Holes Using Partial Convolutions**[C]. ECCV 2018.  
[PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf)

[6] Liu R, Lehman J, Molino P, Such F P, Frank E, Sergeev A, Yosinski J. **An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution**[R]. arXiv:1807.03247, 2018.  
[arXiv](https://arxiv.org/abs/1807.03247)

[7] Tancik M, Srinivasan P P, Mildenhall B, Fridovich-Keil S, Raghavan N, Singhal U, Ramamoorthi R, Barron J T, Ng R. **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains**[C]. NeurIPS 2020.  
[PDF](https://papers.neurips.cc/paper_files/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)

[8] Ledig C, Theis L, Huszár F, Caballero J, Cunningham A, Acosta A, Aitken A, Tejani A, Totz J, Wang Z, Shi W. **Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network**[C]. CVPR 2017.  
[PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)

[9] Shi X, Chen Z, Wang H, Yeung D Y, Wong W K, Woo W C. **Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting**[R]. arXiv:1506.04214, 2015.  
[arXiv](https://arxiv.org/abs/1506.04214)

[10] Liu Z, Lin Y, Cao Y, Hu H, Wei Y, Zhang Z, Lin S, Guo B. **Video Swin Transformer**[R]. arXiv:2106.13230, 2021.  
[arXiv](https://arxiv.org/abs/2106.13230)

[11] PyTorch. **Module — `torch.nn.Module` Documentation**[EB/OL].  
[Link](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)

[12] PyTorch. **Buffer — `torch.nn.parameter.Buffer` Documentation**[EB/OL].  
[Link](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Buffer.html)

[13] PyTorch. **`torch.nn.functional.interpolate` Documentation**[EB/OL].  
[Link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)

[14] PyTorch. **`torch.use_deterministic_algorithms` Documentation**[EB/OL].  
[Link](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)

[15] OpenCV. **Geometric Image Transformations — `resize` Interpolation Guidance**[EB/OL].  
[Link](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html)

[16] PyTorch. **ReflectionPad2d Documentation**[EB/OL].  
[Link](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html)

## 2.5 本章小结

本章从数学角度形式化了稀疏物理场重建问题，指出了其不适定性本质。通过对观测算子 $H$ 的物理建模（降质与几何模型），明确了数据生成的机理。核心在于，本章建立了观测一致性理论，通过两个关键命题证明了 $DC \equiv H$ 不仅是工程实现的规范，更是保证重建误差有界和消除系统性泛化偏差的理论必要条件。这一理论框架为第 3 章提出的“统一算子模块”与“一致性损失函数”提供了坚实的数学支撑。
