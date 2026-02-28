# 第1章 绪论

*本章目标：阐明稀疏观测下时空物理场重建的研究背景、科学问题与工程意义，综述国内外研究现状并指出当前方法的局限性，进而引出本文的研究动机、主要研究内容、创新点及论文组织结构。*

## 1.1 研究背景与意义

### 1.1.1 工程背景

伴随着工业4.0浪潮与数字孪生（Digital Twin, DT）技术的深化应用，复杂工程系统对关键物理过程的全时空状态感知、实时诊断及闭环决策能力提出了更为严苛的要求[1-2]。数字孪生技术强调物理实体与虚拟模型之间的高频交互与动态映射，其效能很大程度上取决于能否持续获取高质量、多源异构的观测数据[3-6]。在航空航天、智能制造、能源管理及海洋环境监测等关键领域，速度场、温度场、压力场等物理量的时空演化信息构成了故障预警、预测性维护与风险评估的决策基石。

尽管如此，在实际工程场景中，部署高密度、全覆盖且时间同步的观测网络仍面临巨大的工程挑战。一方面，受限于部署成本、设备功耗、长期可靠性及极端环境适应性等因素，高精度传感器难以实现长期且均匀的覆盖[7-8]。以海洋观测为例，联合国教科文组织发布的**《全球海洋观测系统 2025 现状报告》[9]** 显示，卫星遥感的大范围覆盖与原位浮标的稀疏分布之间存在显著的“观测鸿沟” (Observation Gap)，尤其在深海与极地区域，部分关键参数的观测密度处于“亚临界”状态[9]。即便是Argo全球剖面浮标阵列已具备约4000个浮标的观测能力，但在空间分辨率和特定海域覆盖率上仍显不足[10]。

另一方面，在物联网（IoT）与边缘计算架构下，海量传感节点产生的高频数据流对通信带宽与传输时延构成了巨大压力。现有研究表明，工程系统往往需要在边缘侧进行数据预处理、压缩或特征提取以减少传输量，但这不可避免地牺牲了部分原始时空分辨率[11-13]。当前的数字孪生系统正遭遇严峻的**“数据匮乏瓶颈” (Data-Scarcity Bottleneck)**。**Shahzad 等人 [4]** 在最新的综述中指出，从智能电网到生物医学监测，传感器分布稀疏与采样异步是限制系统性能的主要因素。**Hossain 等人 [6]** 亦强调，这种数据约束已阻碍了数字孪生从“被动监视”向“主动预测控制”的演进。

此外，长期运行的观测数据还面临噪声污染、零点漂移及数据缺失等质量问题。光学遥感图像常因云层遮挡或传感器故障出现大面积缺失，需依赖重建算法进行修复[14-16]；工业与环境传感器则可能因老化、温漂或硬件故障导致数据漂移或间歇性中断[17-18]。因此，实际工程数据普遍呈现出“空间稀疏、时间异步、质量退化”的复杂特征。

![图 1-1: 数字孪生系统中的“数据匮乏瓶颈 (Data-Scarcity Bottleneck)”示意图。尽管数字孪生强调物理实体与虚拟模型的高频交互，但在实际工程链路中，受限于传感器部署成本、通信带宽（IoT Edge）与环境噪声，流向模型的观测数据往往是稀疏、非均匀且带噪的。这种“感知端稀疏”与“应用端高保真需求”之间的矛盾，构成了制约数字孪生闭环能力的关键瓶颈。](images/fig1-1_digital_twin_bottleneck.png)

综上所述，感知端观测能力的稀疏性与应用端对高分辨率、物理一致全场信息的需求之间存在显著差距。这种“强需求—弱数据”的结构性矛盾，已成为制约数字孪生系统由可视化展示向可计算推断转型的关键瓶颈。因此，探索如何在稀疏观测条件下重建高分辨率且符合物理规律的时空场，具有重大的工程实用价值与理论研究意义。

![图 1-2: 稀疏观测下的时空场重建挑战示意图。左侧展示了高分辨率的真实物理场（如 PDEBench 中的浅水波 SWE 涡旋结构），具有丰富的多尺度纹理与清晰的波前；中间表示经过物理降质（抗混叠滤波、积分、稀疏采样）后的观测数据，其全域覆盖率通常低于 5%，且伴随非均匀分布与噪声干扰；右侧展示了重建目标，即从极度欠定的稀疏观测中恢复出与左图一致的高保真物理场。](images/fig1-2_reconstruction_challenge.png)

---

#### 参考文献（GB/T 7714—2015）

[1] 杨林瑶, 等. 数字孪生与平行系统: 发展现状、对比及展望[J]. 自动化学报, 2019, 45(11): 2001-2017.
[2] 肖淑南, 等. 国内外数字孪生研究热点主题与演进趋势[J]. 计量科学与技术, 2023, 67(4): 1-12.
[3] Wu H, Lee J, Guo Y, et al. A comprehensive review of digital twin from the perspective of data, model, network and application[J]. IEEE Access, 2023, 11: 85762-85785.
[4] Shahzad M, Shafiq M, et al. Technologies and techniques in digital twins for real-time monitoring: A systematic review[J]. Advanced Engineering Informatics, 2025, 61: 102066.
[5] Mchirgui N, et al. The applications and challenges of digital twin technology in smart grids[J]. Applied Sciences, 2024, 14(23): 10933.
[6] Hossain R, et al. Virtual sensing-enabled digital twin framework for real-time monitoring[J]. npj Digital Medicine, 2025, 8: 57.
[7] Lin M, et al. Ocean observation technologies: A review[J]. Journal of Ocean University of China, 2020, 19(5): 965-980.
[8] Penny S G, et al. Observational needs for improving ocean and coupled predictions[J]. Bulletin of the American Meteorological Society, 2019, 100(2): 247-264.
[9] UNESCO-IOC. The Global Ocean Observing System 2025 Status Report[R]. Paris: Intergovernmental Oceanographic Commission, 2025.
[10] Roemmich D, Alford M, Claustre H, et al. On the future of Argo: A global, full-depth, multi-disciplinary array[J]. Frontiers in Marine Science, 2019, 6: 439.
[11] Ray P P. Edge computing for Internet of Things: A survey[J]. Future Generation Computer Systems, 2019, 98: 680-694.
[12] Pioli L, et al. An overview of data reduction solutions at the edge of IoT[J]. Sensors, 2022, 22(6): 2231.
[13] Cao L, et al. Cost optimization in edge computing: A survey[J]. Artificial Intelligence Review, 2024, 57: 10947.
[14] Meraner A, Ebel P, Zhu X X, et al. Cloud removal in Sentinel-2 imagery using a deep residual neural network[J]. Remote Sensing of Environment, 2020, 236: 111484.
[15] Wu J, et al. Progressive gap-filling in optical remote sensing imagery[J]. Remote Sensing of Environment, 2024, 302: 113982.
[16] Zhang Q, Yuan Q, Li J, et al. Missing data reconstruction in remote sensing image with deep learning[J]. IEEE Geoscience and Remote Sensing Letters, 2018, 15(4): 515-519.
[17] Gaddam A, et al. Detecting sensor faults, anomalies and outliers in IoT: A survey[J]. Electronics, 2020, 9(3): 511.
[18] Rudnitskaya A, et al. Calibration update and drift correction methods for electronic sensors: A review[J]. Frontiers in Chemistry, 2018, 6: 433.

---


### 1.1.2 科学挑战：从病态反问题到物理一致性约束

从数学角度来看，稀疏观测下的时空场重建问题可统一表述为一个典型的**欠定逆问题（Underdetermined Inverse Problem）**。根据 Hadamard 对适定问题（Well-posed Problem）的经典定义，一个问题若满足“解存在、解唯一、解对数据连续依赖”三个条件，则称为适定[1]。然而在稀疏观测条件下，观测模型

$$
\mathbf{y}=H(u)+\eta
$$

通常无法满足唯一性与稳定性要求，其中 $H$ 为观测算子，$\eta$ 为噪声扰动。由于观测算子存在显著零空间（Null Space），即存在无穷多个候选解 $\hat{u}$ 满足 $H(\hat{u}) \approx \mathbf{y}$，该问题天然呈现不适定性（Ill-posedness）[2-3]。

为恢复稳定解，经典反问题理论引入正则化框架。Tikhonov 正则化通过构造

$$
\min_u \|H(u)-\mathbf{y}\|^2+\lambda \mathcal{R}(u)
$$

在数据一致性项之外加入先验约束，从而收缩解空间[4]。贝叶斯反问题理论则将未知场视为随机变量，通过后验分布刻画不确定性传播与稳定性结构，为稀疏条件下的不确定性量化提供了严格理论基础[5-6]。近年来，SIAM 与 *Inverse Problems* 等期刊进一步系统化了无限维函数空间中的反问题稳定性分析[7]。

然而，与传统图像重建不同，物理场重建面临更为复杂的多尺度频谱结构。根据奈奎斯特采样定理，若采样频率低于信号最高频率的两倍，则高频成分将在频域发生折叠（Aliasing）[8]。对于湍流等多尺度流动系统，其能谱遵循 Kolmogorov $k^{-5/3}$ 定律[9]，高频能量虽小但对梯度与耗散结构具有决定性影响。稀疏采样会导致频谱能量跨尺度混叠，从而生成非物理伪影。近年来，算子学习领域提出“operator aliasing”概念，指出离散化变化可能引发算子表示误差与跨网格泛化失效[10-11]。因此，在重建过程中恢复真实频谱分布成为核心科学挑战之一。

另一方面，物理系统本身受守恒律支配。若重建结果违反质量守恒或动量守恒，将导致下游导数运算与动力学预测失效。物理信息神经网络（PINN）通过在损失函数中引入 PDE 残差项实现物理约束嵌入[12]，神经算子方法则尝试直接学习解算子映射[13]。但已有研究指出，在多尺度与刚性问题下，PINN 训练存在优化失衡与频谱偏置问题[14]，物理一致性约束如何与数据驱动模型协同设计，仍是开放问题。

此外，在机器学习框架下还存在“训练—部署算子错配（Operator Mismatch）”问题。当训练阶段采用理想化退化算子，而部署阶段观测算子具有不同的积分核、边界处理或采样结构时，模型泛化误差将显著放大[15]。在神经算子理论中，这一现象可归因于离散化依赖性与表示等价性破坏[16]。因此，如何在理论层面确保观测算子的一致性与可复用性，是稀疏重建从实验验证走向工程部署的关键环节。

![图 1-4: 观测算子错配（Operator Mismatch）示意图。路径 (A) 为深度学习常用的理想化训练流程，采用双三次（Bicubic）插值或简单下采样生成低分数据；路径 (B) 为真实物理观测链路，包含传感器抗混叠滤波（Anti-aliasing）、空间积分与非均匀噪声。当模型仅在路径 (A) 上训练时，面对路径 (B) 的真实观测数据会产生系统性的频谱混叠误差与纹理伪影。](images/fig1-4_operator_mismatch.png)

综上所述，稀疏观测下的时空场重建问题不仅涉及不适定逆问题的稳定性恢复，还必须同时处理多尺度混叠、物理守恒嵌入以及算子一致性等多重挑战。其本质是一类融合反问题理论、频谱分析与科学机器学习方法的跨学科研究问题。

---

### 二、参考文献（GB/T 7714）

#### 经典与理论基础

[1] Hadamard J. Lectures on Cauchy’s problem in linear partial differential equations[M]. New Haven: Yale University Press, 1923.

[2] Engl H W, Hanke M, Neubauer A. Regularization of inverse problems[M]. Dordrecht: Kluwer Academic Publishers, 1996.

[3] Hansen P C. Discrete inverse problems: insight and algorithms[M]. Philadelphia: SIAM, 2010.

[4] Tikhonov A N, Arsenin V Y. Solutions of ill-posed problems[M]. Washington: Winston & Sons, 1977.

[5] Stuart A M. Inverse problems: a Bayesian perspective[J]. Acta Numerica, 2010, 19: 451-559.

[6] Dashti M, Stuart A M. The Bayesian approach to inverse problems[J]. Handbook of Uncertainty Quantification, 2017: 311-428.

[7] Kaipio J, Somersalo E. Statistical and computational inverse problems[M]. New York: Springer, 2005.

#### 采样与混叠

[8] Shannon C E. Communication in the presence of noise[J]. Proceedings of the IRE, 1949, 37(1): 10-21.

[9] Kolmogorov A N. The local structure of turbulence in incompressible viscous fluid[J]. Doklady Akademii Nauk SSSR, 1941, 30: 301-305.

[10] Li Z, Kovachki N, Azizzadenesheli K, et al. Fourier neural operator for parametric PDEs[J]. ICLR, 2021.

[11] Kovachki N, et al. Neural operator: learning maps between function spaces[J]. JMLR, 2023, 24(89): 1-97.

#### 物理约束与SciML

[12] Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks[J]. Journal of Computational Physics, 2019, 378: 686-707.

[13] Lu L, Jin P, Karniadakis G E. DeepONet[J]. Nature Machine Intelligence, 2021, 3: 218-229.

[14] Wang S, Teng Y, Perdikaris P. Understanding and mitigating gradient pathologies in PINNs[J]. SIAM Journal on Scientific Computing, 2021, 43(5): A3055-A3081.

#### 算子错配与离散化泛化

[15] Mishra S, Molinaro R. Estimates on the generalization error of physics-informed neural networks[J]. IMA Journal of Numerical Analysis, 2022.

[16] Kovachki N, et al. On universal approximation and discretization invariance of neural operators[J]. JMLR, 2023.

---

## 1.2 国内外研究现状

---

### 1.2.1研究背景与问题界定
稀疏观测驱动的时空场重建，通常指在传感器稀疏、观测噪声不可忽略、测量方式与理想采样不一致的条件下，从有限观测推断高分辨率、物理一致的时空连续场（例如速度/压力/温度/浓度场等），并在需要时进一步支持短期预测与在线更新。其本质接近“逆问题 + 动力学约束”的融合：一方面需要从欠定信息中恢复细节，另一方面又必须满足或近似满足控制方程、边界条件以及统计规律。[1]

在工程与复杂系统应用中，该问题往往与数字孪生的在线校准、实时分析和可操作时间尺度上的更新直接相关：传统高保真数值仿真在强非线性、多尺度场景下可极其昂贵，而数字孪生强调与现实系统的闭环交互与近实时更新，使“高精度 + 低时延”的矛盾更加突出。[2] 近年来，科学机器学习（SciML）将深度学习的表达能力与物理/算子/概率结构结合，为该类问题提供了从“纯数值求解/统计估计”向“可泛化的学习式算子与约束建模”转变的路径，也促使传统方法与学习方法在同一框架下重新被审视。[3]

---

### 1.2.2 传统方法：插值、统计学习与数据同化
在深度学习成为主流之前，物理场重建主要依赖三条技术路线：数值插值/函数逼近、空间统计建模与数据同化（DA）。

数值插值与函数拟合的代表方法包括多项式插值、样条插值与径向基函数（RBF）插值等，优点是实现成熟、计算开销低，且在场量足够光滑时具有良好的局部逼近性质。[4] 需要强调的是，许多经典插值核（尤其是样条类）可从频域角度理解为对信号进行某种形式的平滑重建，其频率响应往往近似低通，从而天然倾向于抑制高频分量；这也是“插值能填补空缺但容易抹平尖峰/细尺度结构”的重要原因之一。[5] 对于包含激波、剪切层、湍流涡等多尺度高频结构的场，单纯依赖平滑假设的插值往往难以同时兼顾整体轮廓与细节物理。[6]

空间统计与不确定性量化方面，克里金（Kriging）与高斯过程回归（Gaussian Process Regression, GPR）通过显式建模协方差/核函数来刻画空间相关性，可在预测未知点的同时给出不确定性估计，在地学与工程场景得到广泛应用。[7] 与之相近的另一支传统路线是降维与模态重建：以本征正交分解（POD）为代表的方法，用一组最优线性基在能量意义上压缩流场维度；在此基础上，针对缺测数据提出的 Gappy POD 通过最小二乘拟合稀疏观测来恢复模态系数，从而重建全场。[8] 这类方法在模态库“覆盖充分”时可以非常高效，但其性能高度依赖线性叠加的表示能力与先验快照库的代表性：当系统动力学呈现强非线性、多尺度耦合或工况变化超出库分布时，重建质量容易显著下降。[8]

数据同化是气象、海洋等领域的标准范式，通过显式引入动力学模型（例如数值模式）将观测与预测进行统计最优融合，典型方法包括集合滤波路线与变分路线。集合滤波的经典代表是 EnKF，其通过集合传播来近似误差协方差并完成更新。[9] 变分同化的代表是 4D-Var，核心是最小化代价函数并依赖线性化与伴随/梯度信息迭代求解。[10] DA 的突出优势是物理一致性与可解释性：观测进入模型并非“贴点拟合”，而是通过观测算子映射到观测空间、在误差统计约束下与动力学演化共同决定分析场。[11] 但其代价同样显著：4D-Var 需要维护线性化与伴随并进行多次迭代；即便在业务化推动下也长期面临“计算成本需进一步下降”的压力。[10] 当问题转向工业数字孪生的在线更新与实时性诉求时，传统 DA 的计算与工程维护成本常被视为重要瓶颈之一。[2]

---

### 1.2.3 深度学习方法：端到端超分辨与物理约束
深度学习在该领域最直接的切入点，是把物理场视作规则网格上的“多通道图像/视频”，将重建问题转写为超分辨率（super-resolution, SR）或缺失补全任务，并借鉴计算机视觉的成熟架构（CNN、GAN、Transformer 等）实现端到端映射。[12] 在流体与湍流重建中，早期代表性工作展示了 CNN 及多尺度结构可从极粗分辨率恢复较高分辨率的速度场，并激发了大量后续研究。[6] 随着 Transformer 在视觉任务中取得进展，基于 Transformer 主干的湍流场超分辨也被提出，用以增强长程依赖建模能力。[13]

然而，端到端方法的核心风险在于：若不显式建模物理与观测过程，网络可能在像素空间获得较低误差，却在物理结构（涡结构、能谱分布、守恒律等）上产生偏差。为缓解该问题，越来越多研究引入物理约束损失或结构化训练策略，例如将物理残差、守恒约束或时空一致性融入生成模型/重建网络，使生成结果更贴近物理可行域。[14]

在理论与经验层面，深度网络存在显著的“低频偏置/频谱偏置”：训练过程中往往优先拟合低频、全局平滑成分，而高频细节需要更长训练或更强的结构/损失引导才能可靠恢复。[15] 这与物理场重建的需求存在张力，因为湍流、剪切层等现象的关键物理信息常体现为小尺度与高频统计特征；因此，单纯依赖逐点误差（例如 MSE 或相对 L2）容易出现“数值好看但物理不可信”的情况。[16]

---

### 1.2.4 科学机器学习范式：PINN 与神经算子学习
相比“把场当图像”的端到端重建，SciML 更强调把物理结构写进模型或学习目标。两条最具代表性的路线是物理信息神经网络（PINN）与神经算子学习（Neural Operator Learning）。

PINN 通过在损失函数中加入偏微分方程残差、初边值条件等，使网络在拟合数据的同时遵循物理规律，可用于方程求解、反演与稀疏观测下的场重建，并具有无网格/连续表示的特点。[17] 对稀疏观测重建而言，PINN 的吸引力在于：当观测极少时，物理方程可起到强先验约束作用，从而减少对大规模配对数据的依赖。[17] 但大量研究表明，PINN 的训练存在系统性优化困难：由微分算子引入的病态性、损失项梯度不平衡、以及在对流主导或高频/多尺度问题上捕捉关键结构的困难，都可能导致收敛缓慢或失败。[18] 国内研究也围绕 PINN 的损失平衡与训练稳定性提出改进思路，并在工程/多物理场问题中持续拓展应用边界。[19]

神经算子学习则试图学习“输入函数到输出函数”的算子映射，而不是固定维度向量到向量的映射；这使其天然更贴近 PDE 求解器的角色，并具备跨分辨率/跨离散的潜在泛化能力。以 FNO 与 DeepONet 为代表的工作在多个 PDE 基准上展现了对不同网格分辨率的适配能力与较高推理效率，从而成为近年 SciML 的主流方向之一。[20] 与此同时，神经算子也暴露出“离散化一致性/混叠误差”等关键问题：当模型在某种离散表示上训练、在另一种表示上推理时，算子连续表示与离散实现之间可能出现不一致，从而影响跨尺度泛化。围绕这一痛点，已有工作从“表示等价/抗混叠”与“多重网格结构”等角度提出系统框架与改进结构。[21] 国内综述也开始将神经算子与类 PINN 方法放在统一视角下梳理其发展脉络与挑战，为工程应用提供路线图式总结。[22]

![图 1-3: 科学机器学习 (SciML) 两大主流范式的架构对比。(a) 物理信息神经网络 (PINN)：基于无网格配点法 (Collocation Points)，将 PDE 残差直接作为损失函数项优化网络参数，适合正/反问题但训练优化困难；(b) 神经算子 (Neural Operator, 如 FNO)：学习函数空间之间的映射算子，通常在频域或积分核空间参数化，具备离散化无关性与极快的推理速度，但依赖大量配对数据。本文提出的方法融合了算子学习的高效性与物理约束的一致性。](images/fig1-3_sciml_comparison.png)

---

### 1.2.5 研究空白与挑战：观测一致性、评测断裂与长期稳定性
尽管上述方法在公开基准与典型方程上取得显著进展，但面向真实工程与复杂观测时，仍存在若干尚未被充分解决的“落地型”空白，其中最关键者可概括为三点。

第一，观测一致性不足（观测算子错配）。大量学习式重建工作在训练阶段默认观测过程是“规则下采样/点值采样/随机丢弃”，即把低分辨率场视为高分辨率场的简单退化版本。[6] 但在真实系统中，观测往往由复杂的观测算子产生：在数据同化理论中，观测算子用于将模型状态映射到观测空间，可能包含插值、变量变换，甚至是辐射传输等复杂前向过程。[11] 从采样理论角度，当测量/采样与抗混叠滤波、空间积分（有限传感器覆盖范围、点扩散/光学传递效应等）耦合时，观测不再等价于理想点采样；此时若训练时使用过度简化的退化模型，部署到真实观测会出现系统性偏差。[23] 类似的问题在真实图像超分辨中已被反复验证：假设“理想双三次下采样”的模型在真实退化下会出现明显性能下滑，因此需要显式退化建模或盲退化适配。[24] 这提示物理场重建也需要更系统地把“传感器与观测口径”纳入训练与评测闭环，并与数据一致性约束或逆问题正则化框架结合。[1]

第二，评测指标与物理可信度之间存在断裂。当前许多论文仍以逐点误差（MSE、相对 L2 等）作为主指标，但在湍流等多尺度问题中，逐点误差不足以衡量惯性区间与高频统计的恢复质量；已有研究在重建评估中明确强调需要结合能谱等统计量进行检验。[16] 因而，面向工程可信应用的评测体系需要更关注结构与频谱一致性（例如能谱、涡识别指标、守恒误差、关键派生量统计等），并明确“低误差是否意味着物理可用”的判别原则。[16]

第三，时空演化中的误差累积与长期稳定性。当重建任务与时间推进耦合（例如把模型用于自回归预测、滚动同化或长时序补全）时，训练阶段常见的“单步监督/教师强制”与推理阶段的“多步滚动”存在分布差异，会触发误差累积与发散风险；序列学习领域已提出通过多步训练或调度采样缓解该问题。[25] 在神经算子长时序预测中，这一现象被进一步系统化讨论，并提出面向稳定性的结构与训练原则以抑制自回归误差增长。[26] 对 PINN 而言，相关研究也提出课程式正则化等策略来改善优化景观与训练过程。[27] 因此，面向真实在线监控/数字孪生的时空场重建，亟需将“观测一致性 + 结构化评测 + 长期稳定性”作为统一目标进行方法设计，而非把三者割裂为独立问题。[28]

---

#### 参考文献（去重后，唯一编号）
[1] Deep learning methods for inverse problems - PMC - NIH  
<https://pmc.ncbi.nlm.nih.gov/articles/PMC9137882/>

[2] Methods for enabling real-time analysis in digital twins  
<https://www.sciencedirect.com/science/article/pii/S0045794924000713>

[3] An Extensive Benchmark for Scientific Machine Learning (PDEBench)  
<https://arxiv.org/abs/2210.07182>

[4] A Practical Guide to Splines  
<https://www.stat.cmu.edu/~brian/valerie/617-2022/week07/spline%20references/pdfcookie.com_a-practical-guide-to-splines.pdf>

[5] B-Spline Signal Processing: Part II—Efficient Design  
<https://users.fmrib.ox.ac.uk/~jesper/papers/future_readgroups/unser9302.pdf>

[6] Super-resolution reconstruction of turbulent flows with machine learning (JFM)  
<https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/superresolution-reconstruction-of-turbulent-flows-with-machine-learning/0DEBFE07FD949054E7E5046AB5632F22>

[7] Statistics for Spatial Data  
<https://rongxie.files.wordpress.com/2011/01/statistics-for-spatial-data-revised-version-1993.pdf>

[8] Turbulence and the dynamics of coherent structures  
<https://www.jstor.org/stable/43637457>

[9] The Ensemble Kalman Filter: Theoretical Formulation and Practical Implementation (ECMWF)  
<https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf>

[10] A strategy for operational implementation of 4D-Var, using an incremental approach  
<https://www2.mmm.ucar.edu/people/duda/files/courtier_etal.pdf>

[11] Atmospheric modeling, data assimilation and predictability  
<https://catdir.loc.gov/catdir/samples/cam033/2001052687.pdf>

[12] Photo-Realistic Single Image Super-Resolution Using a GAN (SRGAN)  
<https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf>

[13] Super-resolution reconstruction of turbulent flows with a Transformer (PoF)  
<https://pubs.aip.org/aip/pof/article/35/5/055130/2890201/Super-resolution-reconstruction-of-turbulent-flows>

[14] Using physics-informed enhanced super-resolution  
<https://www.sciencedirect.com/science/article/pii/S1540748920300481>

[15] On the Spectral Bias of Neural Networks  
<https://proceedings.mlr.press/v97/rahaman19a/rahaman19a.pdf>

[16] Deep learning methods for super-resolution reconstruction ...（含能谱评估强调）  
<https://staff.ustc.edu.cn/~huanghb/LiuB_POF.pdf>

[17] Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs  
<https://www.sciencedirect.com/science/article/pii/S0021999118307125>

[18] Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks  
<https://epubs.siam.org/doi/10.1137/20M1318043>

[19] 一种求解偏微分方程的动态平衡物理信息神经网络  
<https://scis.scichina.com/cn/2024/SSI-2023-0195.pdf>

[20] Fourier Neural Operator (FNO)  
<https://leap.columbia.edu/wp-content/uploads/2023/01/Li-et-al.2021.pdf>

[21] Are Neural Operators Really ...  
<https://arxiv.org/abs/2305.19913>

[22] 基于神经算子与类物理信息神经网络智能求解新进展  
<https://pubs.cstam.org.cn/data/article/lxxb/preview/pdf/lxxb2023-407.pdf>

[23] Communication In The Presence Of Noise  
<https://webusers.imj-prg.fr/~antoine.chambert-loir/enseignement/2020-21/shannon/shannon1949.pdf>

[24] Real-World Video Super-Resolution with a Degradation ...  
<https://pmc.ncbi.nlm.nih.gov/articles/PMC11014003/>

[25] Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks  
<https://arxiv.org/abs/1506.03099>

[26] Towards Stability of Autoregressive Neural Operators  
<https://arxiv.org/abs/2306.10619>

[27] Characterizing possible failure modes in physics-informed neural networks (NeurIPS 2021)  
<https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf>

[28] Definition of a digital twin（National Academies 报告章节）  
<https://www.nationalacademies.org/read/26894/chapter/2>

[29] Radial Basis Functions: Theory and Implementations  
<https://catdir.loc.gov/catdir/samples/cam033/2002034983.pdf>

[30] Gaussian Processes for Machine Learning  
<https://gaussianprocess.org/gpml/chapters/RW.pdf>

[31] Karhunen-Loeve procedure for gappy data  
<https://www.sdss.jhu.edu/~szalay/class/2024/etc/everson-shirovich-josaa-12-8-1657.pdf>

[32] Unsteady Flow Sensing and Estimation via the Gappy Proper Orthogonal Decomposition  
<https://acdl.mit.edu/GappyWillcox.pdf>

[33] A review of operational methods of variational and ensemble ...  
<https://centaur.reading.ac.uk/68685/7/qj2982.pdf>

[34] Data assimilation in the geosciences - Overview  
<https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wcc.535>

[35] SwinIR: Image Restoration Using Swin Transformer  
<https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf>

[36] Video Swin Transformer  
<https://arxiv.org/pdf/2106.13230.pdf>

[37] A physics-constrained Transformer framework for spatio ...  
<https://www.sciencedirect.com/science/article/abs/pii/S1877750322002654>

[38] Challenges in Training PINNs: A Loss Landscape ...  
<https://arxiv.org/pdf/2402.01868.pdf>

[39] Physics-Informed Neural Networks for High-Frequency and Multi-Scale Problems using Transfer Learning  
<https://arxiv.org/abs/2401.02810>

[40] Learning nonlinear operators via DeepONet  
<https://www.nature.com/articles/s42256-021-00302-5>

[41] Efficient Parameterization of Linear Operators via Multigrid (MgNO)  
<https://proceedings.iclr.cc/paper_files/paper/2024/file/eb3c8135137c8a60425a0320869ad87e-Paper-Conference.pdf>

[42] An Assessment of Satellite Radiance Data Assimilation in ...  
<https://www.mdpi.com/2072-4292/11/1/54>

[43] Real-ESRGAN: Training Real-World Blind Super-Resolution With Pure Synthetic Data  
<https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf>

## 1.3 本文研究内容与主要创新

面向稀疏观测条件下的时空场重建难题与现有研究缺口，本文确立“**评测口径一致性优先（Consistency-First）**”的研究理念，目标是形成一套**可复用、可审计、物理可解释**的稀疏观测时空场重建框架。本文的研究内容与创新性贡献概括如下。

### 1.3.1 提出基于“统一观测算子”的物理场重建方法论

为系统性降低训练—评测—部署链路中的“口径错配”，本文将观测算子的显式建模与一致性复用上升至方法论层面。

- **统一算子建模**：将观测过程形式化为可微算子 $H$，对抗混叠预滤波（如 Gaussian pre-filtering）、插值核与采样策略、边界延拓（padding/reflect 等）以及空间掩码/传感器布局等关键细节进行统一、可追溯的参数化描述。
- **镜像复用机制**：训练阶段退化算子 $DC$ 与评测阶段观测算子 $H$ 采用同一实现路径与同一组参数配置，逐项对齐并强制满足 $DC \equiv H$。
- **创新价值**：通过显式消除训练—评测口径不一致导致的隐性分布偏移，使实验结论在真实观测口径下具备更强的可复现性与可审计性，为工程级 SciML 的方法评估与落地提供一致性范式。

### 1.3.2 设计序列化时空课程学习策略

针对时空联合建模易出现优化不稳定、局部最优与长时序误差累积的问题，本文提出“**序列化时空课程学习（Serialized Spatio-Temporal Curriculum Learning）**”训练策略。

- **分阶段优化路径**：采用“空间重构预训练（看清瞬态）$\rightarrow$ 动力学演化预训练（看懂规律）$\rightarrow$ 时空联合微调（端到端收敛）”的递进式训练流程，使空间表征学习与动力学学习的难度逐步释放。
- **Teacher Forcing Decay / Scheduled Sampling**：引入计划采样（Scheduled Sampling）式教学强制衰减机制，使模型输入由真值逐步过渡至模型预测，从而缓解自回归推理阶段的曝光偏差（exposure bias）与误差累积风险。
- **创新价值**：在不牺牲端到端建模能力的前提下提升长时序外推的稳定性，并增强模型对稀疏观测与噪声扰动的鲁棒性，从训练可行性角度改进复杂动力系统重建任务的可训练性。

### 1.3.3 构建兼顾精度与物理守恒的三元混合损失

针对仅采用 MSE 易诱发频谱偏置（spectral bias）并导致高频细节缺失的问题，本文构建面向精度与物理一致性协同优化的复合损失体系。

- **多维约束体系**：联合像素级重建损失 $\mathcal{L}_{\text{rec}}$、频域谱一致性损失 $\mathcal{L}_{\text{spec}}$ 与观测一致性损失 $\mathcal{L}_{\text{dc}}$，形成三元混合目标：
  
  $$
  \mathcal{L}
  =\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}
  +\lambda_{\text{spec}}\mathcal{L}_{\text{spec}}
  +\lambda_{\text{dc}}\mathcal{L}_{\text{dc}}.
  $$

- **物理一致性“护栏”**：$\mathcal{L}_{\text{dc}}$ 基于统一算子 $H$ 约束重建场在观测子空间内与观测数据一致；$\mathcal{L}_{\text{spec}}$ 约束能谱分布与尺度能量分配特征，抑制过度平滑与非物理振荡。
- **创新价值**：实现重建精度指标（如 Rel-L2）与物理一致性指标（如 Spectrum Error）的协同提升，降低“数值指标优但物理一致性不足”的评估风险。

### 1.3.4 建立标准化稀疏重建评测协议

针对稀疏重建领域评测口径分散、对比不公平的问题，本文建立可复现、可审计的标准化评测协议。

- **协议内容**：基于 PDEBench 构建固定随机种子（Seed=2025）的全流程复现管线，明确数据划分、观测生成（统一算子 $H$）、训练配置与评测脚本，并纳入资源成本统计（Params/FLOPs/Latency）。
- **创新价值**：补齐稀疏重建任务在“公平对比、可复现复核、成本—精度权衡”方面的评测基础设施，为后续方法迭代与工程部署提供统一标尺。

## 1.4 论文组织结构

本文共分为五章，章节结构与逻辑安排如下：

- **第1章 绪论**：介绍研究背景、工程需求与科学挑战；综述国内外相关研究进展；分析现有方法在评测口径、可复现性与物理一致性方面的不足；明确研究目标、主要研究内容、创新点与论文组织结构。
- **第2章 问题建模与理论分析**：构建稀疏观测重建的数学描述；严格定义统一观测算子 $H$（包含预滤、采样/插值、边界处理与掩码等关键环节）；从理论层面讨论逆问题的适定性条件与观测一致性约束的必要性，并推导一致性误差的上界，为后续方法设计提供可解释的理论依据。
- **第3章 算法设计与实现**：系统阐述本文提出的 **Consistency-First** 重建框架；给出网络结构（Encoder–Propagator–Decoder）的设计动机与实现细节；说明序列化训练流程（空间预训练 $\rightarrow$ 时序预训练 $\rightarrow$ 联合微调）的具体策略，并给出损失函数的数学形式与计算流程。
- **第4章 实验结果与分析**：基于 PDEBench 数据集（如浅水波 SWE、扩散反应 DRD 等）开展对比实验与消融实验；从重建精度、频谱一致性、长时序稳定性与计算效率等维度进行系统评估，全面验证所提方法的有效性、鲁棒性与工程可用性。
- **第5章 总结与展望**：归纳研究工作与核心结论；讨论方法在数据分布变化、复杂几何与真实观测噪声等场景下的局限性；展望后续研究方向，包括非规则网格上的图神经网络建模、多尺度/多物理耦合扩展，以及与大模型能力融合等潜在路径。

本文的整体组织结构与逻辑流程如图 1-3 所示。

![图 1-5: 本文“Consistency-First”重建框架与论文组织结构概览。本文针对观测算子错配与病态逆问题挑战（第1章），建立了一致性理论与误差界分析（第2章），设计了融合统一算子约束、三元混合损失与序列化课程学习的重建模型（第3章），并在 PDEBench 多物理场基准上验证了方法的精度、鲁棒性与工程可行性（第4章），最终形成了一套可复用、可解释的时空场重建方法论（第5章）。](images/fig1-5_thesis_overview.png)
