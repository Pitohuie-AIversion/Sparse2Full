- 下面给出**按研究生论文写作体例扩写**后的《第7章 理论验证》，并将文中关键论断对应到**可核验的信息来源**（带在线可访问出处的引用）。为避免“引用但不可追溯”的问题，本章只对我已核验到的公开来源进行标注；其余属于通用统计学常识或工程经验的内容不强行加引文。

  ------

  # 第7章 理论验证（扩写版，含可核验出处）

  ## 引言

  第4章给出了三条核心理论命题：
  （i）**评测一致性命题**：若训练端退化算子 DC 与数据观测算子 H 在实现与参数上严格一致，则当观测一致性损失收敛时，评测口径误差 (H_\mathrm{err}=|H(\tilde{u})-y|) 可被有效约束，并与重建误差同步变化；
  （ii）**低频约束稳健性命题**：在低频子空间上约束可优先稳定大尺度结构，配合观测一致性项可减少“Rel-L2 降而 H_\mathrm{err} 不降”的评测断裂；
  （iii）**跨网格稳定性命题**：在离散化一致性与别名控制得到保证时，多分辨率/跨网格评测的性能差异应更可控。

  本章以“**可运行脚本 + 明确验收阈值 + 统计检验**”为主线，对上述命题给出实证验证闭环，并将结果落到 `paper_package/metrics/` 与 `paper_package/figs/` 的材料产出规范上。

  ------

  ## 7.1 一致性验证（对应命题1）

  ### 7.1.1 H/DC 等价性测试（硬门槛）

  **目的**：证明训练端使用的 DC 与数据观测 H **同一入口、同一实现、同一参数镜像**。
  **方法**：运行 `tools/check_dc_equivalence.py` 随机抽样 (N\ge 100) 组样本 ({u^{(i)}, y^{(i)}})，对每个样本计算
  [
  e^{(i)}=\mathrm{MSE}\big(H(u^{(i)}), y^{(i)}\big),
  \quad
  e_{\max}=\max_i e^{(i)},\ \bar{e}=\frac{1}{N}\sum_i e^{(i)}.
  ]
  **验收阈值**（建议保持与你第5章一致）：

  - (\bar{e} < 10^{-8}) 且 (e_{\max} < 10^{-7}) 视为通过；否则判定为**口径不一致**，必须阻断统计汇总。

  > 工程层面，“确定性与可复现”需要显式控制随机性与确定性算法开关；PyTorch 官方对随机性来源与可复现设置给出了明确说明（随机数种子、cuDNN、非确定性算子等）。([PyTorch](https://pytorch.org/docs/stable/notes/randomness.html))

  **推荐输出表（写入 `runs/<exp>/consistency_report.json` 并在论文中汇总）**：

  | 任务 | 参数组                   | N    | mean MSE | max MSE | 结论      |
  | ---- | ------------------------ | ---- | -------- | ------- | --------- |
  | SR   | (σ,k,s,interp,boundary)  | 100  | …        | …       | Pass/Fail |
  | Crop | (h_c,w_c,boundary,align) | 100  | …        | …       | Pass/Fail |

  ------

  ### 7.1.2 评测口径一致性验证：相关性与“断裂”负例

  **核心检验**：在统一口径下，验证 (H_\mathrm{err}) 与 Rel-L2 的相关性显著增强；同时构造**负例**（人为打破 H/DC 镜像），观察两者不同步。

  - **统一口径条件**：`DC ≡ H`（同实例或同参数构造）
  - **负例条件**：例如 SR 下把 `INTER_AREA` 改成 `INTER_LINEAR`，或把 (\sigma) 改成 (\sigma+\Delta)，或 Crop 改边界策略（mirror→zero）

  **统计量**：对测试集每个样本得到 ((\mathrm{RelL2}_j, H_{\mathrm{err},j}))，计算

  - Pearson 相关系数 (r)（线性相关）
  - Spearman (\rho)（秩相关，抗异常值）

  并给出 95% 置信区间（Pearson 可用 Fisher z 变换）。

  **论文呈现建议**：

  - 图：散点图（Rel-L2 vs H_\mathrm{err}）+ 线性拟合；统一口径与负例口径并排对比。
  - 表：给出 (r,\rho) 及 p-value。

  ------

  ## 7.2 收敛性与稳定性验证（对应命题2与第4章稳定性讨论）

  ### 7.2.1 课程学习对收敛的影响（SR ×2→×4；Crop 40%→20%）

  **实验设计**（二因素对照，建议固定其他超参不变）：

  - A：无课程（直接 SR×4 / Crop 20%）
  - B：有课程（先易后难：SR×2→×4 或 Crop 40%→20%）

  **记录与指标**：

  - 训练/验证曲线：(L, L_\mathrm{rec}, L_\mathrm{spec}, L_\mathrm{dc})
  - 梯度稳定：每 step 的 (|\nabla\theta|) 分位数（p50/p90/p99）
  - 最终泛化：测试集 Rel-L2 与 H_\mathrm{err}

  **验收逻辑**：若课程学习显著降低 early-stage 的梯度爆炸/震荡，并在最终指标上取得稳定收益，则支持“课程改善优化路径”的结论。

  ------

  ### 7.2.2 因果/时序约束验证（时序打乱负例）

  若你的模型含时序模块（ConvLSTM/Transformer/AR 等），建议增加**时序一致性负例**：

  - 正例：保持时间顺序输入
  - 负例：在 batch 内随机打乱时间维（保持同一帧集合但破坏顺序）

  比较两者在 (T_{out}) 上的误差增长曲线（例如按步长报告 Rel-L2(t)）。

  关于“因果性约束有助于 PINN/物理学习稳定训练”的观点，可引用对应的公开论文条目作为背景支撑。([arXiv](https://arxiv.org/abs/2203.07404))

  ------

  ### 7.2.3 解码策略验证（双线性+3×3 vs 反卷积）

  **实验目的**：验证你在第3章提出的“抑制棋盘格伪影与高频噪声累积”的工程结论。
  **对照**：

  - Baseline：转置卷积（deconv）上采样
  - Ours：双线性插值 + 3×3 卷积（固定）

  **量化指标**：

  - 空域：PSNR/SSIM、边界带误差（bRMSE/cRMSE）
  - 频域：功率谱差异、fRMSE-mid/high

  **定性材料**：输出“棋盘格/振铃”典型失败案例图组，写入 `paper_package/figs/failure_modes/decoder/`。

  ------

  ## 7.3 泛化能力与跨网格鲁棒性（对应命题3）

  ### 7.3.1 多分辨率外推评测（img_size = 128/256/512）

  **统一原则**：保持观测口径 H 与指标定义不变，改变仅限于评测分辨率与重采样策略（写入 YAML 与图注）。
  **输出**：每个分辨率报告同一指标集 + 资源四项（Params、FLOPs@256²、显存峰值、延迟）。

  ### 7.3.2 网格/离散化变化与别名诊断

  当出现“256 上好、512 上崩”或相反的异常，需要把原因定位到：

  - 观测口径是否仍一致（优先排查 H/DC）
  - 离散化与表示别名（aliasing）是否被放大
  - 频谱约束阈值是否对高分辨率不再合适

  关于“别名无关（alias-free）算子学习/离散一致性影响跨网格表现”的论点，可引用 ReNO/别名无关框架条目作为背景支撑。([arXiv](https://arxiv.org/abs/2305.19913))

  ------

  ## 7.4 显著性检验与效应大小（统一统计协议）

  ### 7.4.1 paired t-test（主指标 Rel-L2）

  对每个种子 (s) 计算相同测试集上的差值
  [
  d_s=\mathrm{RelL2}^{(\mathrm{baseline})}_s-\mathrm{RelL2}^{(\mathrm{ours})}_s,
  ]
  对 ({d_s}) 做**配对 t 检验**，并报告 p-value 与均值±标准差。t 分布的来源可追溯到 Student（Gosset）提出的相关工作。([IME-USP](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf?utm_source=chatgpt.com))

  ### 7.4.2 Cohen’s d（配对版本）与置信区间

  建议报告配对效应量
  [
  d=\frac{\bar{d}}{s_d},
  ]
  其中 (s_d) 为差值样本标准差；并给出 bootstrap 置信区间（工程实现简单且更稳健）。

  > 若你同时比较很多模型/很多任务（多重比较），建议在论文中说明控制策略（例如控制 FDR 或采用更保守的校正），并把“主结论仅绑定主对照”写清楚，避免过度显著性解读。

  ------

  ## 7.5 敏感性分析（参数—口径—资源三维折衷）

  ### 7.5.1 扫描维度与最小覆盖集

  建议把敏感性分析拆为三层：

  1. **口径层（必须覆盖）**：(\sigma, s, (h_c,w_c),) boundary、interp
  2. **损失层（核心）**：(\lambda_s, \lambda_{dc}, k_x,k_y)（低频阈值）
  3. **训练层（稳健性）**：seed、AMP 开关、grad clip、warmup steps

  ### 7.5.2 结果呈现模板（建议写入论文主表+附录）

  - 主表：固定模型结构，仅变口径/损失关键参数，报告 Rel-L2 与 H_\mathrm{err} 同步性
  - 附录：给出更细粒度的扫描曲线（例如 (k_{\max}\in{8,12,16,20,24})）

  ------

  ## 7.6 可复现性验证（确定性、快照、指纹）

  ### 7.6.1 确定性配置与方差门槛

  **目标**：同一 YAML + 同一种子条件下，多次运行指标方差 ≤ (10^{-4})。
  **必要措施**：

  - 固定 Python/NumPy/PyTorch seeds
  - 明确 cuDNN/算子确定性策略
  - 记录环境指纹（CUDA、驱动、torch 版本、GPU 型号等）

  PyTorch 官方对“随机性来源”“确定性算法开关（例如 `torch.use_deterministic_algorithms`）”给出了明确说明，可作为论文中“可复现设置依据”的出处。([PyTorch](https://pytorch.org/docs/stable/notes/randomness.html))

  ### 7.6.2 复现闭环材料检查

  - `runs/<exp>/config_merged.yaml`：配置快照
  - `runs/<exp>/env_fingerprint.json`：环境指纹
  - `paper_package/scripts/`：一键复现与汇总
  - `paper_package/metrics/`：主表 + 显著性 + 资源表
  - `paper_package/figs/`：代表案例 + 失败案例 + 频谱图

  ------

  ## 7.7 小结（命题→证据→材料）

  本章以硬门槛一致性测试（H/DC 等价性）、相关性与负例断裂实验、课程/时序/解码消融、跨分辨率与跨网格评测、显著性与效应量报告、以及确定性复现闭环材料六类证据，构成对第4章理论命题的实证验证链条。其输出直接沉淀到 `consistency_report.json`、主结果表与显著性报告、以及标准化图组与失败案例归档中，从而满足审稿对“可核验、可复现、可解释”的要求。

  ------

  # 参考文献（APA；仅列本章使用且可核验来源）

  - Bartolucci, F., de Bézenac, E., Raonić, B., Molinaro, R., Mishra, S., & Alaifari, R. (2023). *Representation equivalent neural operators: A framework for alias-free operator learning*. arXiv:2305.19913. ([arXiv](https://arxiv.org/abs/2305.19913))
  - Gosset, W. S. (1908). The probable error of a mean. *Biometrika, 6*(1), 1–25. ([IME-USP](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf?utm_source=chatgpt.com))
  - Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). *PDEBench: An extensive benchmark for scientific machine learning*. arXiv:2210.07182. ([arXiv](https://arxiv.org/abs/2210.07182))
  - Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality is all you need for training physics-informed neural networks*. arXiv:2203.07404. ([arXiv](https://arxiv.org/abs/2203.07404))
  - PyTorch. (n.d.). *Reproducibility (randomness)*. PyTorch Documentation. ([PyTorch](https://pytorch.org/docs/stable/notes/randomness.html))
  - PyTorch. (n.d.). *torch.use_deterministic_algorithms*. PyTorch Documentation. ([PyTorch](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html))
  - PyTorch. (n.d.). *torch.set_deterministic_debug_mode*. PyTorch Documentation. ([PyTorch](https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html))

  ------

  如果你希望我“同样处理”的方式继续到**第8章讨论**或**第9章结论与展望**，我会沿用本章这种“命题—实验—阈值—材料—统计”的写法，并把每条关键论断都绑定到可核验出处或你自己的实验材料路径上。
