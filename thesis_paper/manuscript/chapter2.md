# 第2章 相关工作（经润色与扩写）

## 引言
稀疏观测驱动的时空流场重建涉及物理约束神经网络（PINN）与神经算子学习（例如 FNO 与 DeepONet）两条主线，同时牵涉到观测口径设计与抗混叠原则、频谱偏置缓解与输入编码策略、以及统一评测与复现实验的规范。本章围绕上述维度进行系统综述与批判性分析，强调“观测一致性（H/DC 复用）、可复现（YAML+seed 方差≤1e-4）、统一指标与显著性检验”的必要性，并明确本文方法在口径、损失与评测协议方面的差异化定位。

## 2.1 流场重建方法综述：物理约束与算子学习两条路径
物理约束神经网络（PINN）通过在损失中显式嵌入 PDE 残差与初边界条件，在数据稀缺场景下维持物理一致性（Raissi, Perdikaris, & Karniadakis, 2019）。近年的理论与数值分析揭示了 PINN 的误差界、训练稳定性与失败原因，并从神经切线核（NTK）与优化过程给出改进建议（Acta Numerica, 2023；Wang, Yu, & Perdikaris, 2022）。另一方面，神经算子学习（FNO/DeepONet）直接学习函数空间到函数空间的映射，作为参数化 PDE 的解算器替代与快速外推路径（Li et al., 2021；Lu, Jin, Pang, Zhang, & Karniadakis, 2021；Kovachki et al., 2023）。其中 FNO 通过谱域核实现高效泛化与批量推理，DeepONet 基于算子近似理论提供分支/主干结构的通用框架。综合而言，算子学习与物理约束形成互补：前者提供高效外推能力，后者提供显式物理一致性。本文在统一观测口径与评测协议前提下融合两者优势：以算子层提升全局泛化能力，以一致性与频谱损失维护真实口径下的稳健性。

## 2.2 稀疏观测重建：口径一致性与抗混叠原则
时空采样不齐与下采样会引入混叠与信息丢失，使得高频结构恢复困难，且训练口径与评测口径不一致（Kovachki et al., 2023）。经典抗混叠策略为低通预滤后再缩小分辨率，如 Gaussian 预滤与 `INTER_AREA` 面积插值的组合；裁剪任务要求居中对齐且与 patch_size 对齐，并明确边界策略（mirror/zero/wrap）。本文将数据侧观测算子 H 按上述协议统一定义，并在训练端复用为 DC，使核/σ/插值/对齐/边界口径完全一致，从而消除隐性域偏差（OpenCV Anti-aliasing Guidance；Takamoto et al., 2022）。统一口径显著降低 `H_err` 与 `Rel-L2` 之间的断裂，提升横向对比的公平性与结论稳健性。我们以原值域一致性损失约束预测，并通过低频谱一致性损失减少大尺度结构误差，使在真实评测口径下获得稳定改进（Wang, Yu, & Perdikaris, 2022）。

## 2.3 时空耦合建模：多尺度表示、谱域核与因果性约束
时空流场具有宽频谱与多尺度结构，高频训练偏置与学习困难常影响重建质量。近年来的工作从输入编码（Fourier 特征）、多尺度网络（MscaleDNN）、频谱域损失等方向进行缓解与加速（Neural Networks, 2024；Neural Networks, 2025）。FNO 以谱域核近似解算算子，适合快速外推与批量推理；DeepONet 基于算子近似理论提供分支/主干结构的通用框架（Li et al., 2021；Lu et al., 2021）。在时序维度，因果约束与记忆机制可提升稳定性与可解释性（Wang, Sankaran, & Perdikaris, 2021；Rao et al., 2022）。本文在多尺度空间编码与时序融合的基础上，以统一接口与解码策略（双线性+3×3）减少棋盘格伪影，保持与真实评测口径一致的结构表现。

## 2.4 算子离散化与别名无关学习：跨网格鲁棒性
神经算子从连续函数空间到离散实现不可避免引入离散化误差与表示别名，影响跨网格与跨分辨率外推（Bartolucci et al., 2023）。Representation Equivalent Neural Operators（ReNO）提出别名无关框架，缓解算子与离散表示不一致所致的误差。本文通过统一观测口径与频谱一致性损失，实证降低别名引起的评测断裂，并在多分辨率与网格设置下进行敏感性分析与显著性检验，以保证结论的鲁棒性与可复现性。

## 2.5 数据与基准：PDEBench 的统一评测协议
PDEBench（NeurIPS 2022）提供多类 PDE、大规模数据与统一评测协议，遵循 FAIR 原则并给出 DOI 数据发布（Takamoto et al., 2022）。在稀疏观测重建任务中，本文采用固定切分与逐通道 z-score 标准化，并产出 `norm_stat.npz` 与配置快照 `runs/<exp>/config_merged.yaml` 以保证可复现。指标集合包括 `Rel-L2、MAE、PSNR、SSIM、fRMSE-low/mid/high、bRMSE、cRMSE、H_err`，采用 ≥3 种子统计报告均值±标准差，并对主基线执行 paired t-test（Rel-L2）与 Cohen’s d；同时记录资源四项（Params、FLOPs@256²、显存峰值、推理延迟），以保证学术与工程的可比性。

## 2.6 差异与定位：本文方法的创新之处
本文方法的差异化体现在三方面：其一，统一观测口径，将数据侧 H 与训练侧 DC 强制复用同一实现与配置，形成从数据生成到训练评测的单一源口径；其二，三件套损失（重建 + 低频谱一致性 + 原值域观测一致性）替代仅重建的做法，使 `H_err` 与 Rel-L2 同步下降，减少在真实评测口径下的断裂；其三，统一接口与评测协议，标准化模型接口、指标与显著性检验、资源四项与可视化规范，搭配 PDEBench 的 FAIR 数据与工具链，提升横向对比与跨环境复现的可靠性。与此同时，通过口径一致性与频谱一致性配合算子层，提升对离散化别名的鲁棒性。

## 2.7 文献关系图与对照表（建议）
- 关系图：将 PINN、FNO、DeepONet、Neural Operator 综述、ReNO（别名无关）、PDEBench（基准）与频谱偏置研究组织为“方法-理论-基准-问题”四象限，标注本文方法在“口径一致性”和“评测严格性”上的差异化定位。
- 对照表：列出主要方法与本文的对照维度（物理约束显式程度、算子外推能力、采样抗混叠口径、跨网格鲁棒性、可复现协议与显著性检验、资源四项记录）。

## 2.8 写作与引用规范自检清单
- 篇章完整：涵盖两类核心路径、抗混叠与口径一致性、时空耦合与因果约束、别名无关与跨网格鲁棒、数据与基准、差异与定位。
- 引用规范：APA 格式且与论述紧密结合；近五年高质量文献覆盖（JMLR 2023、Acta Numerica 2023、NeurIPS 2023/2022、SIAM JSC 2024、Neural Networks 2024/2025）。
- 口径一致性：明确 SR/Crop 的 H 参数与复用原则；后续章节与实验脚本一致。
- 可复现性：固定切分与 z-score 标准化；同一 YAML+种子指标方差 ≤ 1e-4；配置快照与环境指纹。
- 评测与显著性：统一指标与 ≥3 种子；paired t-test（Rel-L2）与 Cohen’s d；资源四项记录完整。

## 总结
本章以叙述体方式综述了稀疏观测驱动流场重建的两条主线（物理约束与算子学习），并讨论了频谱偏置、离散化别名与口径不一致导致的评测断裂。我们强调统一观测口径与可复现协议的必要性，并以三件套损失与统一评测协议兼顾物理一致性与算子外推潜力。上述分析为第3章的方法论与数学形式化、第4章的理论分析、第5章的算法实现与第6章的实验验证提供理论与经验依据。

## 2.8 评价标准与审稿关注点（扩展）
为便于审稿与复现，本文相关工作部分建议按以下标准进行评价：
- 口径声明：是否明确数据侧 `H` 与训练侧 `DC` 的实现与参数；是否避免口径不一致对比。
- 统计稳健性：是否采用 ≥3 种子并报告均值±标准差；是否进行 paired t-test（Rel-L2）与 Cohen’s d。
- 资源透明：是否报告 `Params、FLOPs@256²、显存峰值、推理延迟`；是否在统一分辨率下统计。
- 可复现材料：是否提供配置快照与环境指纹；是否有图表与脚本的最小材料包。
- 失败案例与解释：是否类型化归档失败案例并提供可操作的改进建议。

## 2.9 近五年补充引用（APA 示例）
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Li, Z., Kovachki, N. B., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=c8P9NQVtmnO
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), 218–229. https://doi.org/10.1038/s42256-021-00302-5
- Kovachki, N. B., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A. M., & Anandkumar, A. (2023). Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24, 1–97. https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf
- Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449, 110768. https://doi.org/10.1016/j.jcp.2022.110768
- Bartolucci, F., de Bézenac, E., Raonić, B., Molinaro, R., Mishra, S., & Alaifari, R. (2023). Representation equivalent neural operators: A framework for alias-free operator learning. Advances in Neural Information Processing Systems (NeurIPS). https://arxiv.org/abs/2305.19913
- Franco, N. R., & Brugiapaglia, S. (2024). Operator learning using random features: A tool for scientific computing. SIAM Journal on Scientific Computing. https://doi.org/10.1137/24M1648703
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). PDEBench: An extensive benchmark for scientific machine learning. NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2210.07182
- Neural Networks (2024/2025). Spectral bias reduction papers. https://www.sciencedirect.com/science/journal/08936080

---

## 参考文献（APA）
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Li, Z., Kovachki, N. B., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=c8P9NQVtmnO
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), 218–229. https://doi.org/10.1038/s42256-021-00302-5
- Kovachki, N. B., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A. M., & Anandkumar, A. (2023). Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24, 1–97. https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf
- Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449, 110768. https://doi.org/10.1016/j.jcp.2022.110768
- Bartolucci, F., de Bézenac, E., Raonić, B., Molinaro, R., Mishra, S., & Alaifari, R. (2023). Representation equivalent neural operators: A framework for alias-free operator learning. Advances in Neural Information Processing Systems (NeurIPS). https://arxiv.org/abs/2305.19913
- Franco, N. R., & Brugiapaglia, S. (2024). Operator learning using random features: A tool for scientific computing. SIAM Journal on Scientific Computing. https://doi.org/10.1137/24M1648703
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). PDEBench: An extensive benchmark for scientific machine learning. NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2210.07182
- Rao, C., Ren, P., Wang, Q., Buyukozturk, O., Sun, H., & Liu, Y. (2022). Encoding physics to learn reaction–diffusion processes. Computer Methods in Applied Mechanics and Engineering, 389, 114399. https://doi.org/10.1016/j.cma.2022.114399
- Wang, S., Sankaran, S., & Perdikaris, P. (2021). Respecting causality is all you need for training physics-informed neural networks. arXiv:2203.07404
- Diminishing spectral bias in physics-informed neural networks using adaptive Fourier encoding policy（2024）. Neural Networks. https://www.sciencedirect.com/science/article/abs/pii/S0893608024008153
- On spectral bias reduction of multi-scale neural networks for highly oscillatory PDE solutions（2025）. Neural Networks. https://www.sciencedirect.com/science/article/abs/pii/S0893608025000589
- OpenCV Anti-aliasing Guidance: Gaussian prefilter and INTER_AREA downsampling. https://answers.opencv.org/question/222322/how-to-choose-sigmakernel-size-when-downsampling-image/

---

*最后更新：2025年*
## 2.7 文献关系与对照维度（扩展）
为便于后续章节引用与对比，本文建议以“方法—理论—基准—问题”四象限关系图呈现 PINN、FNO、DeepONet、Neural Operator 综述、ReNO（别名无关）与 PDEBench（基准）之间的关联；并以表格形式梳理以下对照维度：物理约束显式程度、算子外推能力、采样抗混叠口径、跨网格鲁棒性、可复现协议与显著性检验、资源四项记录与工程可行性。
