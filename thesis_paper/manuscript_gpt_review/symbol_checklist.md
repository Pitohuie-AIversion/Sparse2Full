# 附录A（或第3章末）统一口径卡：符号、域与评测口径（Observational Protocol Card）

> 目标：**全书统一符号、统一变量所在域、统一观测算子 \(H\) 与一致性指标 \(H_{\mathrm{err}}\) 的定义，并彻底消除 “\(\sigma\)” 歧义。**  
> 本口径卡一经采用，正文与附录中出现与其冲突的表述应以本口径卡为准。

---

## A.1 变量域与符号约定（全书统一）

设空间域 \(\Omega\subset\mathbb{R}^2\)，离散网格分辨率为 \(H\times W\)，时间索引为 \(t=1,\dots,T\)。本文考虑的物理量场为
\[
u_t:\Omega\to\mathbb{R}^{C},
\]
其中 \(C\) 为通道数（例如压力、速度分量等）。

为避免“归一化域”与“原值域”混用导致的口径漂移，本文对所有训练与评测变量采用如下**严格域约定**：

- **原值域（physical/raw domain）**：
  \[
  u_t \in \mathbb{R}^{C\times H\times W},\qquad \tilde{u}_t \in \mathbb{R}^{C\times H\times W}.
  \]
  其中 \(u_t\) 为真值（GT），\(\tilde{u}_t\) 为模型预测在原值域下的重建结果。

- **z-score 域（normalized domain）**：  
  逐通道统计量 \(\mu\in\mathbb{R}^{C}\)、\(\sigma_z\in\mathbb{R}^{C}\) 来自训练集并固定保存为 `norm_stat.npz`。定义
  \[
  u^{(z)}_t = \frac{u_t-\mu}{\sigma_z},\qquad \hat{u}^{(z)}_t = \Phi(\cdot),
  \]
  其中 \(\Phi\) 为网络映射，输出 \(\hat{u}^{(z)}_t\) 处于 z-score 域。

- **反归一化（从 z-score 域回到原值域）**：
  \[
  \tilde{u}_t = \sigma_z \odot \hat{u}^{(z)}_t + \mu,
  \]
  其中 \(\odot\) 表示逐通道广播乘法。

> **强制约定 1（全书不变）**：观测算子 \(H(\cdot)\) **只作用于原值域变量**（如 \(u_t\)、\(\tilde{u}_t\)），不得直接对 \(u^{(z)}_t\)、\(\hat{u}^{(z)}_t\) 施加 \(H\)。

---

## A.2 观测生成口径：统一观测算子 \(H\)（数据侧）与退化算子 DC（训练侧）

给定观测算子
\[
H:\mathbb{R}^{C\times H\times W}\to\mathbb{R}^{C\times h\times w},
\]
观测由原值域真值产生：
\[
y_t = H(u_t) + n_t,
\]
其中 \(y_t\) 为观测输入，\(n_t\) 为噪声（可为零或测量噪声）。

本文包含两类任务口径如下。

### A.2.1 超分辨（SR）口径（Anti-aliasing + Downsample）

SR 观测算子定义为：
\[
H_{\mathrm{SR}}(u)\ \equiv\ D_s\!\left(G_{\sigma_{\mathrm{blur}}} * u\right),
\]
其中：

- \(G_{\sigma_{\mathrm{blur}}}\) 为二维高斯核（核大小记为 \(k\)，例如固定 \(k=5\)，并在配置中显式记录）；
- \(\sigma_{\mathrm{blur}}\) 为高斯模糊标准差（**仅用于观测算子**，不得与 z-score 标准差混淆）；
- \(D_s(\cdot)\) 表示下采样因子 \(s\) 的降采样算子；
- 下采样插值方式固定为 `INTER_AREA`（面积重采样），并在配置中显式记录。

### A.2.2 裁剪（Crop）口径（Centered Crop + Boundary Mode）

Crop 观测算子定义为：
\[
H_{\mathrm{Crop}}(u)\ \equiv\ C_{h_c,w_c}^{(\mathrm{mode})}(u),
\]
其中 \(C_{h_c,w_c}^{(\mathrm{mode})}\) 为中心对齐裁剪窗口算子，输出大小为 \((h_c,w_c)\)。边界处理策略 \(\mathrm{mode}\in\{\mathrm{mirror},\mathrm{zero},\mathrm{wrap}\}\) 必须在配置中显式声明。窗口 \((h_c,w_c)\) 需与 `patch_size` 对齐（为其整数倍），以避免额外 padding 或对齐偏差导致的隐性口径漂移。

### A.2.3 H/DC 复用原则（Single Source of Truth）

训练侧退化算子 DC **必须**与数据侧观测算子 \(H\) 完全一致：
\[
\mathrm{DC}\ \equiv\ H,
\]
即核大小、\(\sigma_{\mathrm{blur}}\)、插值方式、对齐方式、边界策略等参数均来自同一入口、同一实例或同一配置派生结果。

> **强制约定 2（全书不变）**：任何实验若出现 \(H\neq \mathrm{DC}\)（实现或参数不一致），应被标注为“**口径不一致负例**”，不得与主结果并列比较（仅可用于断裂示例或反例论证）。

---

## A.3 评测口径指标：\(H_{\mathrm{err}}\) 的严格定义

为将“数学重建误差”与“评测口径误差”区分并统一，本文定义：

- **重建误差（示例：Rel-L2）**  
  若在原值域上报告：
  \[
  \mathrm{Rel\text{-}L2}(\tilde{u},u) = \frac{\|\tilde{u}-u\|_2}{\|u\|_2}.
  \]
  （若在 z-score 域上报告必须显式写为 \(\mathrm{Rel\text{-}L2}(\hat{u}^{(z)},u^{(z)})\)。本文默认主表报告原值域版本。）

- **评测口径误差（观测一致性误差）**  
  \[
  H_{\mathrm{err}}\ \equiv\ \|H(\tilde{u})-y\|_2,
  \]
  或使用归一化形式：
  \[
  H_{\mathrm{err}}^{(\mathrm{rel})}\ \equiv\ \frac{\|H(\tilde{u})-y\|_2}{\|y\|_2}.
  \]

> **强制约定 3（全书不变）**：\(H_{\mathrm{err}}\) 中的 \(H(\cdot)\) 输入必须是 **\(\tilde{u}\)**（反归一化后的原值域预测），不得直接对 \(\hat{u}^{(z)}\) 计算。

---

## A.4 三件套损失的域一致性（训练目标口径）

训练阶段，网络输出为 \(\hat{u}^{(z)}\)，反归一化得到 \(\tilde{u}\)。本文损失统一写为：
\[
L = L_{\mathrm{rec}} + \lambda_s L_{\mathrm{spec}} + \lambda_{dc}L_{dc}.
\]

- **重建损失（z-score 域）**：
  \[
  L_{\mathrm{rec}} = \|\hat{u}^{(z)}-u^{(z)}\|_2^2.
  \]

- **低频谱一致性损失（z-score 域）**：设二维傅里叶变换为 \(\mathcal{F}_{2D}(\cdot)\)，低频索引集合为 \(\mathcal{K}_{\mathrm{low}}\)。定义
  \[
  L_{\mathrm{spec}} = \sum_{(k_x,k_y)\in\mathcal{K}_{\mathrm{low}}}
  \left\|\mathcal{F}_{2D}(\hat{u}^{(z)})_{k_x,k_y}-\mathcal{F}_{2D}(u^{(z)})_{k_x,k_y}\right\|_2^2.
  \]
  默认可采用
  \[
  \mathcal{K}_{\mathrm{low}}=\{(k_x,k_y)\,:\,k_x\le 16,\ k_y\le 16\},
  \]
  但若跨分辨率评测，应在配置中声明是否按比例缩放 \(\mathcal{K}_{\mathrm{low}}\)（避免“低频”含义随分辨率漂移）。

- **原值域观测一致性损失（原值域）**：
  \[
  L_{dc} = \|H(\tilde{u})-y\|_2^2.
  \]

---

## A.5 “\(\sigma\)” 符号消歧义（全书统一写法）

为避免歧义，本文规定：

- \(\sigma_{\mathrm{blur}}\)：观测算子 \(H_{\mathrm{SR}}\) 中的高斯模糊标准差；
- \(\sigma_z\)：z-score 标准化的逐通道标准差（来自 `norm_stat.npz`）。

> **强制约定 4（全书不变）**：正文中单独写 “\(\sigma\)” 视为不合格表述，必须写为 \(\sigma_{\mathrm{blur}}\) 或 \(\sigma_z\)。

---

## A.6 一致性脚本与验收阈值（工程与统计口径）

运行一致性脚本 `tools/check_dc_equivalence.py`，随机抽样不少于 \(N\ge 100\) 个样本，计算
\[
e^{(i)}=\mathrm{MSE}\!\left(H(u^{(i)}),\,y^{(i)}\right),\qquad
\bar{e}=\frac{1}{N}\sum_{i=1}^{N} e^{(i)},\quad e_{\max}=\max_i e^{(i)}.
\]

- **无噪声情形（\(n_t=0\)）建议阈值**：  
  \[
  \bar{e}<10^{-8},\qquad e_{\max}<10^{-7}.
  \]
- **有噪声情形**：阈值需与噪声尺度绑定（例如按 \(\mathbb{E}\|n\|^2\) 或观测域方差设定），并在报告中记录。

> **验收原则**：若未通过一致性阈值，则该实验不进入主结果汇总（仅可作为口径不一致负例）。

---

## A.7 口径卡使用规则（写作与汇总的硬约束）

1. **口径先行**：任何消融、对比、SOTA 复现必须先通过 A.6 一致性脚本。  
2. **域清晰**：凡出现 \(\hat{u}^{(z)}\)、\(u^{(z)}\)、\(\tilde{u}\)、\(u\) 的指标或损失，必须明确其所在域。  
3. **指标一致**：主表默认报告原值域 Rel-L2 与 \(H_{\mathrm{err}}\)；如需报告 z-score 域版本，必须额外标注 “(z-score)”。

---
