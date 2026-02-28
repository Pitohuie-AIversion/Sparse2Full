# 第7章 理论验证（扩写版：命题—脚本—阈值—统计—材料闭环）

## 7.0 引言

承接第6章的实验结果，尽管模型在标准测试集上表现优异，但科学计算模型的可靠性并不等价于“在单一数据集上取得较低误差”。面向工程部署与科研可核验要求，仍需进一步回答三个关键问题：  
(1) 评测口径是否严格一致，是否存在“训练口径与评测口径不一致”导致的指标断裂；  
(2) 模型对大尺度结构（低频模态）是否稳定，是否出现“像素域指标改善但结构漂移”的隐性失真；  
(3) 在非标准工况（跨分辨率/跨网格）下是否具备鲁棒性，异常是否可定位、可解释、可修复。

第4章从欠定逆问题角度提出三条理论命题，并在第5–6章给出工程化实现。本章将三条命题进一步制度化为**可运行脚本 + 明确验收阈值 + 统计检验 + 材料归档**的验证闭环，并将全部产出固化到 `runs/<exp>/` 与 `paper_package/`，形成研究生论文可审计证据链。

为避免符号漂移，沿用第3–6章口径。对任意测试样本真值场记为 $u$，网络输出（z-score 域）记为 $\hat u^{(z)}$，回到原值域后的预测为
$$
\tilde u = \sigma_z \hat u^{(z)} + \mu. \qquad (7-1)
$$
统一观测算子为 $H$，生成观测
$$
y = H(u) + n,\qquad n \text{ 为噪声（可为 0）}. \qquad (7-2)
$$
评测口径误差（与第3章一致）定义为
$$
H_{\mathrm{err}} \triangleq \|H(\tilde u)-y\|_2. \qquad (7-3)
$$

本章建立并执行三类验证协议：
1. **评测一致性验证（7.1）**：对应命题1，建立 $H/\mathrm{DC}$ 同源复用的阻断式审计机制，系统性消除“口径断裂”风险。  
2. **结构稳健性验证（7.2）**：对应命题2，确立低频约束（$L_{\mathrm{spec}}$）的判定标准与参数扫描区间，避免“指标好看但结构漂移”。  
3. **跨域鲁棒性验证（7.3）**：对应命题3，定义跨分辨率/跨网格评测与异常诊断流程，确保异常可定位、可解释、可修复。  

---

## 7.1 一致性验证：$H/\mathrm{DC}$ 同源复用（对应命题1）

### 7.1.1 $H/\mathrm{DC}$ 等价性测试（硬门槛，阻断式）

**目的**：在进入任何统计汇总与横向对比之前，证明训练端退化算子 $\mathrm{DC}$ 与评测端观测算子 $H$ 满足硬约束：
$$
\mathrm{DC} \equiv H 
\quad\text{（同一入口、同一实现、同一参数镜像、同一边界/插值/对齐策略）}. \qquad (7-4)
$$

**脚本**：`tools/check_dc_equivalence.py`

**关键口径修正（避免“含噪比较导致误判”）**：  
等价性测试必须比较**算子输出的一致性**，即比较 $\mathrm{DC}(u)$ 与 $H(u)$；不得直接用含噪观测 $y$ 去对齐 $10^{-8}$ 级阈值。若业务实现中观测缓存为 $y$，需同时缓存 noise-free 的 $y_{\mathrm{clean}}=H(u)$，或在审计脚本中显式将噪声置零。

**方法**：随机抽样 $N\ge 100$ 个样本 $u^{(i)}$，计算
$$
e^{(i)}=\mathrm{MSE}\!\left(H(u^{(i)}),\,\mathrm{DC}(u^{(i)})\right),\quad
\bar e=\frac{1}{N}\sum_{i=1}^N e^{(i)},\quad
e_{\max}=\max_i e^{(i)}. \qquad (7-5)
$$
并同时记录更敏感的最大绝对误差：
$$
a_{\max}=\max_{i}\left\|H(u^{(i)})-\mathrm{DC}(u^{(i)})\right\|_\infty. \qquad (7-6)
$$

**验收阈值（默认，需与数值精度匹配）**：
- Pass：$\bar e < 10^{-8}$ 且 $e_{\max} < 10^{-7}$，并且 $a_{\max}$ 不超过实现精度容许范围；  
- Fail：任一条件不满足，**直接阻断**该实验进入第6章统计汇总（避免不公平横向对比）。

> 工程备注：当 $H$ 内含 FFT、浮点插值、混合精度、或 GPU 非确定性算子时，阈值需与实际数值精度匹配。任何阈值调整必须写入归档文件并在论文中说明原因（例如 AMP 使最小可达误差上移）。

**归档**：`runs/<exp>/consistency_report.json`（必须包含：任务类型、参数签名、dtype/设备、$N$、$\bar e$、$e_{\max}$、$a_{\max}$、Pass/Fail、差异定位日志）

**论文汇总表模板（建议放第6章或附录）**

| 任务 | 参数签名（摘要） | dtype/device | $N$ | mean MSE $\bar e$ | max MSE $e_{\max}$ | max abs $a_{\max}$ | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| SR | $s,k,\sigma_{\mathrm{blur}},interp,boundary$ | … | 100 | … | … | … | Pass/Fail |
| Crop | $h_c,w_c,align,boundary,mask$ | … | 100 | … | … | … | Pass/Fail |

---

### 7.1.2 负例构造与“断裂率”指标（反证一致性的必要性）

为证明一致性门禁的必要性，本节构造若干“故意错配”的负例，使训练端 $\mathrm{DC}$ 与评测端 $H$ 在实现或参数上发生偏移：

- **SR 负例**：插值策略 `INTER_AREA → INTER_LINEAR` 或 $\sigma_{\mathrm{blur}}\to \sigma_{\mathrm{blur}}+\Delta\sigma$  
- **Crop 负例**：边界 `mirror → zero` 或对齐 `center → corner`（引入系统性偏移）

对测试集样本 $j=1,\dots,N_{\text{test}}$，计算相关性指标：
$$
r=\mathrm{corr}_{\text{Pearson}}(\mathrm{RelL2}_j,\,H_{\mathrm{err},j}),\qquad
\rho=\mathrm{corr}_{\text{Spearman}}(\mathrm{RelL2}_j,\,H_{\mathrm{err},j}). \qquad (7-7)
$$
并报告 Pearson 的 95% 置信区间（Fisher z 变换）及 p-value；Spearman 报告 p-value 作为抗异常值补充结论。

**断裂率（Breakage Rate）定义（建议作为硬指标写入表格）**：  
设某对照（如 consistent vs mismatch）下，若出现“重建域指标改善但口径误差恶化”的样本，计为断裂。定义
$$
\mathrm{BR}\triangleq \frac{1}{N_{\text{test}}}\sum_{j=1}^{N_{\text{test}}}
\mathbb{I}\!\left(\mathrm{RelL2}^{(A)}_j-\mathrm{RelL2}^{(B)}_j>\tau_{\mathrm{rel}}
\ \wedge\ 
H_{\mathrm{err},j}^{(B)}-H_{\mathrm{err},j}^{(A)}>\tau_{H}\right). \qquad (7-8)
$$
其中 $A$ 为正例（consistent），$B$ 为负例（mismatch），$\tau_{\mathrm{rel}},\tau_H$ 为容忍阈值（在附录或脚本中固定）。

**图表归档（写入 `paper_package/figs/theory_verif/`）**：
- 散点图：$H_{\mathrm{err}}$–Rel-L2（正例 vs 负例并排）
- 分箱曲线：按 Rel-L2 分箱后的 $H_{\mathrm{err}}$ 均值±置信带（更直观暴露“断裂”）
- BR 表：不同负例设置下 BR 的均值±标准差（≥3 seeds）

**判定准则（建议写成“可验收条款”）**：
- 正例（consistent）：Rel-L2 下降时 $H_{\mathrm{err}}$ 同步下降（或至少不系统性背离），BR 显著低；  
- 负例（mismatch）：BR 显著上升，且 $H_{\mathrm{err}}$ 与错配强度呈单调恶化趋势。

---

### 7.1.3 课程/顺序训练稳定性验证（与第6章结果闭环）

为验证“空间 → 时序 → 联合”三阶段策略的必要性与可控性，本节将训练过程本身纳入验证闭环（强调**稳定可复现**而非仅看最终指标）。

**脚本**：`tools/check_curriculum_stability.py`（或在训练日志汇总脚本中实现）

1) **阶段切换稳定性**  
记录每个切换点（Transition Epoch）前后若干 step 的损失均值，定义
$$
\Delta L_{\text{trans}} = \frac{\overline{L}_{\text{after}}-\overline{L}_{\text{before}}}{\overline{L}_{\text{before}}}. \qquad (7-9)
$$
验收目标：在绝大多数 seeds 上 $\Delta L_{\text{trans}}$ 不出现灾难性飙升（阈值写入脚本），并在若干 epoch 内恢复单调下降趋势。

2) **端到端 vs 顺序训练收敛对比**  
在同一组随机种子下对比验证集曲线，报告达到相同验证损失所需的训练量（epoch/step）与最终收敛值差异，并进行配对统计（见 7.4）。

3) **长时稳定性（可选）**  
若启用时序正则化/能量约束，报告长时预测（如 20 步）下的漂移率指标（定义与阈值固定写入脚本），并给出 paired t-test 与效应量。

> 该节强调“训练范式作为系统组件”的可核验性；定量结果与第6.6节形成互证闭环。

---

## 7.2 低频约束稳健性验证（对应命题2）

### 7.2.1 消融：$L_{\mathrm{spec}}$ 是否带来结构稳定收益

**对照组（与第3章 A0–A3 对齐）**：
- A0：仅 $L_{\mathrm{rec}}$
- A1：$L_{\mathrm{rec}}+\lambda_{dc}L_{\mathrm{dc}}$
- A2：$L_{\mathrm{rec}}+\lambda_{\mathrm{spec}}L_{\mathrm{spec}}$
- A3：$L_{\mathrm{rec}}+\lambda_{\mathrm{spec}}L_{\mathrm{spec}}+\lambda_{dc}L_{\mathrm{dc}}$（主方法）

**低频指标口径（与第6章一致：幅值谱 fRMSE）**：  
对 2D FFT 的幅值谱，定义低频集合 $\mathcal K_{\text{low}}$，则
$$
\mathrm{fRMSE}_{\text{low}} \triangleq
\sqrt{\frac{1}{|\mathcal K_{\text{low}}|}
\sum_{k\in\mathcal K_{\text{low}}}
\left(\,|\tilde U_k|-|U_k|\,\right)^2},\quad
U=\mathcal F_{2\mathrm{D}}(u),\ \tilde U=\mathcal F_{2\mathrm{D}}(\tilde u). \qquad (7-10)
$$
并与 Rel-L2、$H_{\mathrm{err}}$ 同表报告（≥3 seeds：均值±标准差 + 显著性 + 效应量）。

**判定逻辑（建议写成审计条款）**：
- 在固定 $\lambda_{dc}$ 的前提下，A3 相对 A1 若能**显著降低** $\mathrm{fRMSE}_{\text{low}}$，且 Rel-L2 改善在多 seed 下稳定，则支持“低频结构先稳”的命题；  
- 若 A2 仅改善低频但 $H_{\mathrm{err}}$ 不稳定，则进一步支持命题1：$L_{\mathrm{dc}}$ 在“评测口径绑定”上不可或缺。

---

### 7.2.2 频谱阈值与权重扫描：结构—口径—资源折衷

**扫描变量**：
$$
k_{\max} \in \{8,12,16,20,24\},\qquad 
\lambda_{\mathrm{spec}} \in \{10^{-4},10^{-3},10^{-2}\}. \qquad (7-11)
$$

**固定变量**：模型结构、训练步数、学习率计划、batch、数据切分、以及 $H/\mathrm{DC}$ 口径签名全部固定（并通过 7.1.1 的硬门槛）。

**输出**：
- 主表：Rel-L2、$H_{\mathrm{err}}$、$\mathrm{fRMSE}_{\text{low}}$、资源四项（Params/FLOPs/显存/Latency）
- 曲线/热力图：$(k_{\max},\lambda_{\mathrm{spec}})\rightarrow$ 指标响应与拐点

**验收结论写法（避免“只报最好点”）**：以“稳定区间 + 拐点 + 资源代价”叙述，例如：  
- $k_{\max}$ 过小：低频稳定但细节不足；  
- $k_{\max}$ 过大或 $\lambda_{\mathrm{spec}}$ 过强：出现训练不稳/高频噪声上升；  
- 存在一段稳定区间使结构与口径同步改善且资源可接受，将其作为默认配置并固化到 YAML。

---

## 7.3 跨分辨率/跨网格鲁棒性验证（对应命题3）

### 7.3.1 多分辨率外推评测（img\_size = 128 / 256 / 512）

**设计原则**：训练分辨率固定为 256；评测阶段仅改变分辨率与重采样路径。所有重采样策略必须写入 YAML 与图注，并归档至 `runs/<exp>/config_merged.yaml`，保证可解释与可复现。

**输出表（建议）**：
- 每个分辨率报告：Rel-L2、MAE、PSNR、SSIM、$\mathrm{fRMSE}_{\text{low/mid/high}}$、$H_{\mathrm{err}}$
- 同时报告资源四项（统一设备与 batch；Latency 预热后统计均值±标准差）

**关键规定：频段集合必须按 Nyquist 比例缩放（避免语义漂移）**  
当分辨率变化时，低/中/高频集合阈值需按 Nyquist 比例缩放，例如设径向频率上限 $\rho_{\max}$，则
$$
\rho_1=\alpha_1\rho_{\max},\qquad \rho_2=\alpha_2\rho_{\max},
\quad 0<\alpha_1<\alpha_2<1, \qquad (7-12)
$$
其中 $\alpha_1,\alpha_2$ 固定写入配置与脚本，确保跨分辨率的 low/mid/high 含义一致。

**判定逻辑**：
- 若主方法在 128/512 上相对基线仍保持“同步改善”（Rel-L2 与 $H_{\mathrm{err}}$ 同向改善），支持命题3；  
- 若出现单一分辨率异常退化，进入 7.3.2 的诊断流程并归档诊断日志。

---

### 7.3.2 异常诊断流程：口径 → 混叠 → 阈值（必须归档）

当出现“256 上好、512 上崩（或相反）”的异常，按以下顺序定位原因，并将诊断记录写入  
`paper_package/metrics/diagnosis_log.md`：

1) **口径复核**：重新运行 `check_dc_equivalence.py`，确认 $\mathrm{DC}\equiv H$ 仍通过（优先排除口径漂移）。  
2) **别名/混叠诊断**：对比不同分辨率的功率谱与误差谱，检查是否出现能量折叠或特定频带尖峰；必要时对重采样路径引入抗混叠滤波并显式记录。  
3) **阈值自适应**：若发现频段集合语义漂移，必须切换为按比例阈值（见 7.3.1）并在附录报告替代口径对结论的影响。  

> 写作定位：跨网格泛化不稳定往往与表示别名（representation aliasing）相关；本章将其作为诊断流程的理论背景支撑，而非作为额外方法主张。

---

## 7.4 统计显著性与效应量（统一协议，避免口径混用）

### 7.4.1 paired t-test：以“同一样本对”为统计单位

配对检验以**同一测试样本**为配对单位。对某一次训练—评测，记录测试样本级指标序列：
$$
a_j=\mathrm{RelL2}^{\text{baseline}}_j,\qquad
b_j=\mathrm{RelL2}^{\text{ours}}_j,\qquad
d_j=a_j-b_j,\quad j=1,\dots,N_{\text{test}}. \qquad (7-13)
$$
对 $\{d_j\}$ 做 paired t-test，报告 $t$、p-value、以及 $\bar d \pm s_d$。

**多 seed 呈现（二选一，写清楚即可）**：  
- 方案A：每个 seed 单独检验，报告 p-value 的 min/median/max；  
- 方案B：对每个样本先对 seed 求平均 $\bar a_j,\bar b_j$，再对 $\bar d_j$ 做 paired t-test（强调跨 seed 稳健平均）。

> 多重比较声明：当同时比较多个 PDE 场景/多个模型，主结论仅绑定“主对照组”；其余比较放入附录并说明校正策略（如 Holm–Bonferroni 或 FDR）。

---

### 7.4.2 配对 Cohen’s $d$ 与置信区间

配对效应量定义为
$$
d=\frac{\bar d}{s_d}. \qquad (7-14)
$$
其中 $\bar d, s_d$ 来自 (7-13)。为降低正态性假设带来的偏差，置信区间建议采用 bootstrap（对样本索引 $j$ 重采样），并在脚本中固定重采样次数与随机种子。

---

## 7.5 可复现性验证（确定性、快照、指纹）

### 7.5.1 确定性设置与方差门槛

**目标门槛**：在同一 YAML + 同一种子条件下，多次运行关键指标方差 $\le 10^{-4}$（门槛与计算方式写入脚本与归档）。

**必须记录（写入 `runs/<exp>/env_fingerprint.json`）**：
- Python / NumPy / PyTorch seeds  
- cuDNN deterministic / benchmark 开关  
- `torch.use_deterministic_algorithms` 与 deterministic debug mode（启用与否、告警级别）  
- AMP 开关与 scaler 配置  
- GPU 型号/驱动/CUDA/torch 版本、关键算子后端信息

---

### 7.5.2 材料闭环检查（强制）

本章全部验证必须落地以下材料，缺一项则该实验不进入论文主结论汇总：

- `runs/<exp>/config_merged.yaml`
- `runs/<exp>/env_fingerprint.json`
- `runs/<exp>/consistency_report.json`
- `paper_package/scripts/`（一键复现 + 汇总 + 显著性 + 作图）
- `paper_package/metrics/`（主表、显著性、资源表、诊断日志）
- `paper_package/figs/`（代表案例、失败案例、功率谱、边界带放大、断裂率图）

---

## 7.6 章节小结（命题 → 证据 → 文件）

本章将第4章三条理论命题落实为可核验证据链：

- **命题1（口径一致性）**：以 `check_dc_equivalence.py` 的硬门槛 + 负例错配 + 断裂率（BR）与相关性对比，证明一致性门禁可显著抑制评测断裂；  
- **命题2（低频结构稳健）**：以 $L_{\mathrm{spec}}$ 消融与 $(k_{\max},\lambda_{\mathrm{spec}})$ 扫描，证明低频约束对大尺度结构稳定与口径同步改善具有可重复收益；  
- **命题3（跨分辨率鲁棒）**：以多分辨率评测与“口径→混叠→阈值”的诊断流程，证明跨网格异常可定位、可解释、可修复。

上述验证的全部中间产物均落地到 `runs/<exp>/` 与 `paper_package/`，从而满足研究生论文“可复现、可审计、可复核”的要求。验证方法的理论一致性与鲁棒性后，第8章将进一步从物理意义、局限性与未来扩展（如三维场、大模型结合）展开讨论。

---

## 参考文献（APA 7；本章引用且可核验）

- Bartolucci, F., de Bézenac, E., Raonić, B., Molinaro, R., Mishra, S., & Alaifari, R. (2023). *Representation equivalent neural operators: A framework for alias-free operator learning* (arXiv:2305.19913). arXiv.
- Gosset, W. S. (1908). The probable error of a mean. *Biometrika, 6*(1), 1–25. https://doi.org/10.1093/biomet/6.1.1
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). *PDEBench: An extensive benchmark for scientific machine learning* (arXiv:2210.07182). arXiv.
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). *PDEBench* (Version 1.0) [Data set]. DaRUS. https://doi.org/10.18419/darus-2986
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality is all you need for training physics-informed neural networks* (arXiv:2203.07404). arXiv.
- PyTorch Contributors. (n.d.). *Reproducibility*. In *PyTorch documentation*. （建议在定稿时补充访问日期与对应 torch 版本）
- PyTorch Contributors. (n.d.). *torch.use_deterministic_algorithms*. In *PyTorch documentation*. （建议在定稿时补充访问日期与对应 torch 版本）
- PyTorch Contributors. (n.d.). *torch.set_deterministic_debug_mode*. In *PyTorch documentation*. （建议在定稿时补充访问日期与对应 torch 版本）
