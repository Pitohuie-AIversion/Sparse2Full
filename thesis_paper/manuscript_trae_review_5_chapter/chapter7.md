# 第7章 理论验证（扩写版：命题—脚本—阈值—统计—材料闭环）

## 7.0 引言

承接第6章的实验结果，虽然模型在标准测试集上表现优异，但科学计算模型的可靠性不仅取决于单一数据集上的精度，更取决于其是否符合理论预期、是否具备物理一致性以及在非标准工况下的鲁棒性。

第4章从“欠定逆问题”的角度提出了三条理论命题，并在第5–6章给出了工程化实现。本章面向研究生论文的可核验要求，将这三条命题进一步**制度化为可运行脚本 + 明确验收阈值 + 统计检验 + 材料归档**的验证闭环，并将全部产出固化到 `runs/<exp>/` 与 `paper_package/`。

为避免符号漂移，沿用第3–6章口径。对任意时刻（或任意测试样本）真值场记为 $u$，网络输出（z-score 域）记为 $\hat u^{(z)}$，回到原值域后的预测记为
$$
\tilde u = \sigma_z \hat u^{(z)} + \mu. \qquad (7-1)
$$
数据观测由统一观测算子 $H$ 给出
$$
y = H(u) + n, \qquad n \text{ 为噪声（可为 0）}. \qquad (7-2)
$$
评测口径误差（与第3章一致）定义为
$$
H_{\mathrm{err}} \triangleq \|H(\tilde u)-y\|_2. \qquad (7-3)
$$

本章主要阐述以下三类验证协议的建立与执行：

1. **评测一致性验证（Section 7.1）**：针对命题1，建立 $H/\mathrm{DC}$ 同源复用的阻断式审计机制，确保“口径断裂”风险被系统性消除。
2. **结构稳健性验证（Section 7.2）**：针对命题2，确立低频约束（$L_{\mathrm{spec}}$）的有效性判定标准与参数扫描区间。
3. **跨域鲁棒性验证（Section 7.3）**：针对命题3，定义跨分辨率/跨网格评测的诊断流程与异常定位策略。

---

## 7.1 一致性验证：$H/\mathrm{DC}$ 同源复用（对应命题1）

### 7.1.1 $H/\mathrm{DC}$ 等价性测试（硬门槛）

**目的**：在统计汇总之前，证明训练端退化算子 $\mathrm{DC}$ 与数据观测算子 $H$ 满足硬约束  
$$
\mathrm{DC} \equiv H \quad \text{（同一入口、同一实现、同一参数镜像、同一边界/插值/对齐策略）}. \qquad (7-4)
$$

**脚本**：`tools/check_dc_equivalence.py`

**方法**：随机抽样 $N\ge 100$ 个样本 $u^{(i)}$，计算数据侧观测 $y^{(i)}$ 与复用算子输出 $H(u^{(i)})，并记录
$$
e^{(i)}=\mathrm{MSE}\!\left(H(u^{(i)}),\,y^{(i)}\right),\quad
\bar e=\frac{1}{N}\sum_{i=1}^N e^{(i)},\quad
e_{\max}=\max_i e^{(i)}. \qquad (7-5)
$$

**验收阈值（与第5章保持一致）**：
- $\bar e < 10^{-8}$ 且 $e_{\max} < 10^{-7}$ 判定为 **Pass**；
- 否则判定为 **Fail**，直接阻断该实验进入第6章统计汇总（避免不公平横向对比）。

> **工程备注（避免“误判”）**：当 $H$ 内含浮点插值、FFT、混合精度或 GPU 非确定性算子时，阈值需要与实际数值精度匹配；阈值调整必须写入 `consistency_report.json`，并在论文中说明原因（例如从 FP32 改为 AMP 导致最小可达误差上移）。

**归档**：`runs/<exp>/consistency_report.json`（必须包含：任务类型、参数签名、$N$、$\bar e$、$e_{\max}$、Pass/Fail、差异定位日志）

**论文汇总表模板**（建议写入第6章或附录）：

| 任务 | 参数签名（摘要） | $N$ | mean MSE $\bar e$ | max MSE $e_{\max}$ | 结论 |
|---|---|---:|---:|---:|---|
| SR | $s,k,\sigma_{\mathrm{blur}},\text{interp},\text{boundary}$ | 100 | … | … | Pass/Fail |
| Crop | $h_c,w_c$、`align`、`boundary`、`mask_update` | 100 | … | … | Pass/Fail |

> **注**：$\dots$ 表示具体数值需在实验中填入。Pass 判定标准为 $\text{MSE}(H(u), DC(u)) < 10^{-8}$。

#### 7.1.3 负例构造与反证法
为了证明一致性的必要性，设计若干“故意错配”的负例条件：
- **操作层**：
  - SR：`INTER_AREA → INTER_LINEAR` 或 $\sigma_{\mathrm{blur}} \to \sigma_{\mathrm{blur}}+\Delta\sigma_{\mathrm{blur}}$
  - Crop：`mirror → zero` 或 `center → corner` (对齐偏移)

**统计量与可视化**：对测试集样本 $j=1,\dots,N_{\text{test}}$，计算
$$
r=\mathrm{corr}_{\text{Pearson}}(\mathrm{RelL2}_j,\,H_{\mathrm{err},j}),\qquad
\rho=\mathrm{corr}_{\text{Spearman}}(\mathrm{RelL2}_j,\,H_{\mathrm{err},j}). \qquad (7-6)
$$
并报告 Pearson 的 95% 置信区间（Fisher z 变换）及对应 p-value；Spearman 给出 p-value 与稳健结论（抗异常值）。

**图表呈现**（写入 `paper_package/figs/theory_verif/`）：
- 散点图：$H_{\mathrm{err}}$–Rel-L2（正例 vs 负例并排）
- 分箱曲线：按 Rel-L2 分箱后的 $H_{\mathrm{err}}$ 均值±置信带（更直观暴露“断裂”）

**判定准则（建议）**：
- 正例：$\|r\|$ 与 $\|\rho\|$ 同时显著高于负例，并且 Rel-L2 下降时 $H_{\mathrm{err}}$ 同步下降；
- 负例：出现“Rel-L2 改善但 $H_{\mathrm{err}}$ 无改善/变差”的样本比例显著升高（将该比例写入表格，作为“断裂率”指标）。

### 7.1.3 顺序训练课程有效性验证

为验证“空间 $\to$ 时序 $\to$ 联合”三阶段策略的必要性，本研究设计了如下消融验证实验：

1. **课程阶段切换稳定性**
   记录每个阶段切换点（Transition Epoch）前后的 Loss 变化率。
   **验证目标**：验证 $\Delta \text{Loss}_{\text{transition}} < 0$，即阶段切换未导致模型崩溃，且新阶段的训练任务（如从单帧到多步）能够平滑承接上一阶段的特征空间。
   
2. **端到端 vs 顺序训练收敛对比**
   在同一组随机种子下，对比两种策略的验证集 Loss 收敛曲线。
   **验证目标**：顺序训练策略在达到相同 Loss 水平时所需的总 Epoch 数显著少于端到端训练，或最终收敛值更优。

3. **时序正则化贡献**
   对比开启与关闭时序导数/能量损失时的长时预测（20步）稳定性。
   **验证目标**：开启正则化后，长时预测的能量漂移率（Energy Drift Rate）显著降低。

相关实验结果详见第 6.6 节。

---

## 7.2 低频约束稳健性验证（对应命题2）

### 7.2.1 消融：是否引入 $L_{\mathrm{spec}}$ 的结构稳定收益

**对照组**（与第3章 A0–A3 对齐）：
- A0：仅 $L_{\mathrm{rec}}$
- A1：$L_{\mathrm{rec}}+\lambda_{dc}L_{\mathrm{dc}}$
- A2：$L_{\mathrm{rec}}+\lambda_s L_{\mathrm{spec}}$
- A3：$L_{\mathrm{rec}}+\lambda_s L_{\mathrm{spec}}+\lambda_{dc}L_{\mathrm{dc}}$（主方法）

**低频指标（与第6章一致）**：将频域误差分段为 low/mid/high；以 low 段为主验证对象（大尺度结构）。例如对 2D FFT 频率索引集合 $\mathcal K_{\text{low}}$ 定义
$$
\mathrm{fRMSE}_{\text{low}} \triangleq
\sqrt{\frac{1}{|\mathcal K_{\text{low}}|}
\sum_{k\in\mathcal K_{\text{low}}}
\left|\mathcal F(\tilde u)_k-\mathcal F(u)_k\right|^2}. \qquad (7-7)
$$
并与 Rel-L2、$H_{\mathrm{err}}$ 同表报告。

**判定逻辑**：
- 若 A3 相对 A1（固定 $\lambda_{dc}$）显著降低 $\mathrm{fRMSE}_{\text{low}}$ 且带来 Rel-L2 的稳健改善，则支持“低频结构先稳”的命题；
- 若 A2 在部分任务中改善低频但 $H_{\mathrm{err}}$ 不稳定，则提示 $L_{\mathrm{dc}}$ 在“评测口径绑定”上的必要性（与命题1衔接）。

---

### 7.2.2 频谱阈值 $k_{\max}$ 扫描：结构—口径—资源折衷

**扫描变量**：
$$
k_{\max} \in \{8,12,16,20,24\},\qquad \lambda_s \in \{10^{-4},10^{-3},10^{-2}\}. \qquad (7-8)
$$

**固定变量**：模型结构、训练步数、学习率计划、batch、数据切分、$H/\mathrm{DC}$ 口径签名全部固定。

**输出**：
- 主表：Rel-L2、$H_{\mathrm{err}}$、$\mathrm{fRMSE}_{\text{low}}$、资源四项
- 曲线：$k_{\max},\lambda_s$ → 指标热力图（便于呈现拐点）

**验收结论写法建议**：不以“最好点”叙述，而以“稳定区间 + 拐点 + 资源代价”叙述，例如：
- $k_{\max}\le 12$ 低频稳定但细节不足；
- $k_{\max}\ge 24$ 训练不稳或高频噪声上升；
- $k_{\max}=16$ 出现结构与口径同步改善且资源可接受（作为默认设置）。

---

## 7.3 跨分辨率/跨网格鲁棒性验证（对应命题3）

### 7.3.1 多分辨率外推评测（img\_size = 128 / 256 / 512）

**设计原则**：训练分辨率固定为 256；评测阶段仅改变输出分辨率与重采样路径，并将重采样策略写入 YAML 与图注，确保可解释。

**输出表（建议）**：
- 每个分辨率报告：Rel-L2、MAE、PSNR、SSIM、$\mathrm{fRMSE}_{\text{low/mid/high}}$、$H_{\mathrm{err}}$
- 同时报告资源四项：Params、FLOPs@256²、显存峰值、推理延迟（统一设备与 batch）

**判定逻辑**：
- 若主方法在 128/512 上相对基线保持“同步下降”（Rel-L2 与 $H_{\mathrm{err}}$ 同向改善），支持命题3；
- 若出现单一分辨率异常退化，进入 7.3.2 的诊断流程。

---

### 7.3.2 异常诊断流程：口径 → 别名 → 阈值

当出现“256 上好、512 上崩（或相反）”的异常，需要按以下顺序定位原因，并将诊断记录写入 `paper_package/metrics/diagnosis_log.md`：

1. **口径复核**：重新运行 `check_dc_equivalence.py`，确认 $\mathrm{DC}\equiv H$ 仍通过（优先排除口径漂移）。
2. **别名/混叠诊断**：对比不同分辨率的功率谱与误差谱，检查是否出现“能量折叠”或特定频带异常尖峰。
3. **阈值自适应**：当分辨率改变导致“低频集合语义漂移”，需要将 $k_{\max}$ 改为“按比例阈值”（例如按 Nyquist 比例），并在附录报告替代口径的影响。

> **背景引用（写作定位）**：别名无关（alias-free）的算子学习框架将“表示别名”作为跨网格不稳定的重要来源之一，可用作第4章理论背景与本章诊断流程的文献支撑。

---

## 7.4 统计显著性与效应量（统一协议，避免口径混用）

### 7.4.1 paired t-test：以“同一样本对”为统计单位

配对检验必须以**同一测试样本**为配对单位。对每个 seed 的一次完整训练—评测，记录测试集样本级指标序列：
$$
a_j=\mathrm{RelL2}^{\text{baseline}}_j,\qquad
b_j=\mathrm{RelL2}^{\text{ours}}_j,\qquad
d_j=a_j-b_j,\quad j=1,\dots,N_{\text{test}}. \qquad (7-9)
$$
对 $\{d_j\}$ 做 paired t-test，报告 $t$、p-value、以及 $\bar d \pm s_d$。

**多 seed 呈现**（建议二选一，写清楚即可）：
- 方案A：每个 seed 单独检验，报告 p-value 的分布（min/median/max）；
- 方案B：对每个样本先对 seed 求平均 $\bar a_j,\bar b_j$，再对 $\bar d_j$ 做 paired t-test（强调“跨 seed 稳健平均”）。

> **多重比较声明**：当同时比较多个 PDE 场景/多个模型，主结论仅绑定“主对照组”，其余比较放入附录并说明控制策略（FDR 或保守校正）。

---

### 7.4.2 配对 Cohen’s $d$ 与置信区间

配对效应量定义为
$$
d=\frac{\bar d}{s_d}. \qquad (7-10)
$$
其中 $\bar d$ 与 $s_d$ 来自 (7-9)。为避免正态性假设过强，置信区间建议采用 bootstrap（对样本索引 $j$ 重采样）。

---

## 7.5 可复现性验证（确定性、快照、指纹）

### 7.5.1 确定性设置与方差门槛

**目标门槛**：同一 YAML + 同一种子条件下，多次运行关键指标方差 $\le 10^{-4}$（写入第6章自检清单）。

**必要记录**（必须写入 `env_fingerprint.json`）：
- Python / NumPy / PyTorch seed
- cuDNN deterministic / benchmark 开关
- `torch.use_deterministic_algorithms` 与 debug mode（是否启用、告警级别）
- AMP 开关与 scaler 配置
- GPU/驱动/CUDA/torch 版本、算子后端信息

### 7.5.2 可复现材料闭环检查（强制）

- `runs/<exp>/config_merged.yaml`
- `runs/<exp>/env_fingerprint.json`
- `runs/<exp>/consistency_report.json`
- `paper_package/scripts/`（一键复现 + 汇总 + 显著性 + 画图）
- `paper_package/metrics/`（主表、显著性、资源表、诊断日志）
- `paper_package/figs/`（代表案例、失败案例、功率谱、边界带放大）

---

## 7.6 章节小结（命题 → 证据 → 文件）

本章将第4章三条理论命题落实为可核验证据链：

- 命题1：以 `check_dc_equivalence.py` 的硬门槛 + 相关性增强 + 口径错配负例，证明“口径一致性”可显著抑制评测断裂；
- 命题2：以 $L_{\mathrm{spec}}$ 消融与 $k_{\max},\lambda_s$ 扫描，证明低频约束对大尺度结构稳定与口径同步改善具有可重复收益；
- 命题3：以跨分辨率评测与“口径→别名→阈值”的诊断流程，证明跨网格异常可定位、可解释、可修复。

上述验证的全部中间产物均落地到 `runs/<exp>/` 与 `paper_package/`，从而满足“可复现、可审计、可复核”的研究生论文要求。

验证了方法的理论一致性与鲁棒性后，第8章将进一步跳出具体指标，从更宏观的视角讨论本研究在物理意义（如能量谱）、局限性（如极端工况失效）以及未来扩展（如三维场、大模型结合）方面的思考。

---

## 参考文献（APA 7；本章引用且可核验）

- Bartolucci, F., de Bézenac, E., Raonić, B., Molinaro, R., Mishra, S., & Alaifari, R. (2023). *Representation equivalent neural operators: A framework for alias-free operator learning* (arXiv:2305.19913). arXiv.
- Gosset, W. S. (1908). The probable error of a mean. *Biometrika, 6*(1), 1–25. https://doi.org/10.1093/biomet/6.1.1
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). *PDEBench: An extensive benchmark for scientific machine learning* (arXiv:2210.07182). arXiv.
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). *PDEBench* (Version 1.0) [Data set]. DaRUS. https://doi.org/10.18419/darus-2986
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality is all you need for training physics-informed neural networks* (arXiv:2203.07404). arXiv.
- PyTorch Contributors. (n.d.). *Reproducibility*. In *PyTorch documentation*. https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch Contributors. (n.d.). *torch.use_deterministic_algorithms*. In *PyTorch documentation*. https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- PyTorch Contributors. (n.d.). *torch.set_deterministic_debug_mode*. In *PyTorch documentation*. https://docs.pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html

---

*最后更新：2026-01-01*
