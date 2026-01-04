# 第5章 算法设计（工程化闭环：统一口径、可替换模块、可复现实验）

## 5.0 引言

本章给出本文方法的工程化算法设计与实现细节，目标是形成“可复现、可替换、可对照”的最小闭环：以统一观测口径为核心约束（数据侧观测算子 (H) 与训练侧退化算子 (\mathrm{DC}) **同源复用**），以统一模型接口为工程约束（模块可插拔、指标可横向对照），以确定性训练与材料化产出为复现约束（配置快照、环境指纹、统计检验与资源四项齐备）。

算法设计围绕两条硬约束组织全文结构：

* **口径硬约束**：( \mathrm{DC}\equiv H )（同一实现、同一参数、同一边界与插值策略），阻断“训练口径—评测口径”错配导致的评测断裂；
* **接口硬约束**：统一输入打包与 `forward()` 签名，保证不同 backbone / 算子层 / 时序模块仅在内部替换，不改变观测口径与评测协议。

---

## 5.1 端到端流程与输入输出契约

### 5.1.1 数据对象、张量形状与监督信号

设真值时空场为 (u_{1:T})，单帧张量表示为

$$
u_t \in \mathbb{R}^{C\times H\times W},\qquad t\in{1,\dots,T}.
$$

观测由任务指定观测算子 (H) 生成：

$$
y_t = H(u_t) + n_t,
$$

其中 (n_t) 为噪声项（可取 0 或合成噪声）。训练样本至少包含 ((y_t, u_t))，并随样本或随数据集提供标准化统计量 ((\mu,\sigma))（逐通道 z-score；第6章报告生成策略与复用方式）。

为便于模块替换与批处理，统一采用 batch-first 表示：

$$
y \in \mathbb{R}^{B\times C_y\times H_y\times W_y},\quad
u \in \mathbb{R}^{B\times C\times H\times W}.
$$

* SR 任务通常满足 (H_y=W_y=\frac{H}{s})；
* Crop 任务通常满足 (H_y\times W_y = h_c\times w_c)。

### 5.1.2 统一接口：可替换模型的最小协议

为保证模块可替换与评测一致性，模型采用统一签名（输出位于 z-score 域）：

* 初始化：`__init__(in_ch, out_ch, img_size, **kwargs)`
* 前向：`forward(x[B,C_in,H,W]) -> u_hat_z[B,C_out,H,W]`

输入张量 (x) 由如下分量按通道拼接构成：

1. `baseline`：基础重建（SR：上采样到 (H\times W)；Crop：零填充+mask 约束或简单插值），用于稳定训练起点；
2. `coords`：显式坐标编码（归一化到 ([0,1])，可叠加 Fourier 特征）；
3. `mask`：观测缺失区域指示（Crop 场景关键，用于区分“观测/未观测”区域）；
4. `fourier_pe`（可选）：频域位置编码，用于缓解频谱偏置并提升高频表达。

对应的统一输入打包形式为：

$$
x_t=\mathrm{Concat}\big(\mathrm{baseline}(y_t),,m_t,,\mathrm{coords},,\mathrm{PE}_{\mathrm{Fourier}}\big).
$$

---

## 5.2 统一观测算子 (H) 与训练退化算子 (\mathrm{DC}) 的同源复用

### 5.2.1 单一入口原则与参数镜像

工程实现将 (H) 的构造收敛到**唯一入口函数**（例如 `build_degradation(cfg.degradation)`），并强制训练端 (\mathrm{DC}) 直接复用同一实现（同一实例或同一配置签名生成的等价实例）。设计目标是消除隐性域偏差：一旦 (H\neq \mathrm{DC})，即使像素域误差下降，也可能出现评测口径误差 (|H(\tilde u)-y|) 不降反升的“评测断裂”。

口径复用属于“可审计工程约束”，需要通过一致性门禁脚本把约束落到可执行证据上（见 5.2.4）。

### 5.2.2 SR 观测算子：抗混叠与插值选择

SR 任务采用“先低通、再缩小”的口径：

$$
y_t^{\mathrm{SR}} = D_s!\left(G_{\sigma}\ast u_t\right) + n_t,
$$

其中 (G_{\sigma}) 为高斯预滤波核，(D_s) 为下采样算子（缩小倍率 (s)）。缩小插值策略采用 OpenCV 的建议：缩小场景优先使用 `INTER_AREA`，其 decimation 行为更符合抗混叠需求。高斯滤波的参数语义（核大小、(\sigma)、边界处理等）以 `GaussianBlur` 的官方说明为准。

工程口径声明写入配置快照（材料包可核验）：

* `scale s`、`blur_sigma`、`blur_ksize`；
* 缩小插值：`INTER_AREA`；
* 边界策略（reflect/replicate/wrap）与滤波阶段的边界策略一致性；
* 标准化/反标准化发生顺序（与 5.4.3 的原值域一致性损失保持一致）。

### 5.2.3 Crop 观测算子：中心对齐、块对齐与边界策略

Crop 观测写为：

$$
y_t^{\mathrm{Crop}} = C_{h_c,w_c}(u_t)+n_t,
$$

其中 (C_{h_c,w_c}) 为裁剪算子。为与网络 patch/窗口结构兼容，裁剪窗口建议满足 `patch_size` 的整数倍，避免额外 padding 引入对齐偏差。口径声明包含：

* 对齐规则（中心对齐/左上对齐/偏移量定义）；
* 窗口尺寸 ((h_c,w_c)) 与边界策略；
* mask 同步更新规则（裁剪与 mask 更新绑定，否则跨实验指标不可比）。

### 5.2.4 一致性门禁（阻断式审计）

训练开始阶段执行一致性检验，不通过则终止实验并输出差异诊断。对随机抽样的真值 (u^{(i)})：

1. 数据管线生成 (y^{(i)})；
2. 算子入口生成 (\hat y^{(i)}=H(u^{(i)}))；
3. 验证

$$
\mathrm{MSE}!\left(\hat y^{(i)},,y^{(i)}\right)<\varepsilon,\quad \varepsilon=10^{-8}.
$$

4. 失败时记录并归档差异来源：((s,\sigma,k))、插值方式、边界策略、对齐偏移、dtype/rounding 等，输出到 `runs/<exp>/consistency_report.json`。

---

## 5.3 网络结构的工程化拆分：编码—算子层—时空融合—解码

### 5.3.1 编码器：多尺度空间表征

编码器负责从 `baseline/coords/mask/(fourier_pe)` 提取多尺度空间特征。工程上常用稳定组合：

* 多尺度卷积块（成本可控、复现友好）；
* 注意力/Transformer 块（增强长程依赖，需配套资源统计）；
* 频域分支（与谱约束损失互补，需与第6章消融对齐）。

编码器输出记为 (z_t\in\mathbb{R}^{C_z\times H\times W})。

### 5.3.2 算子层（OperatorBlock）：横向可替换的统一接口

算子层定位为“可替换模块”，统一形式：

$$
\mathrm{OperatorBlock}: z_t \mapsto \bar z_t.
$$

在不改变输入输出张量形状的前提下替换 FNO-family / DeepONet-family / Conv-Attn hybrid。替换实验只允许修改 `OperatorBlock` 内部实现与其超参数，不允许改变 (H/\mathrm{DC}) 或数据口径，从而保证对照公平。
### 5.3.3 时空融合：显式时序模块与隐式条件化两条路线

时间维融合采用两条工程路线（第6章以“误差传播—推理成本”对照）：

* 显式时序：ConvLSTM / Temporal Transformer / **自回归封装 (ARWrapper)**（直接建模 \(t\) 维耦合，适合长时预测）；
* 隐式条件：把时间索引/物理参数作为条件输入，通过条件归一化或条件注意力调制特征。

两条路线保持统一 `forward()` 签名，避免接口变化引入不可控实验差异。

---

## 5.4 分阶段时空顺序训练实现细节

为解决时空耦合模型端到端训练收敛困难的问题，本研究在 `SequentialSpatiotemporalTrainer` 类中实现了三阶段顺序训练策略。该实现不仅涉及损失函数的切换，还涉及模型参数的冻结与解冻逻辑，具体如下：

### 5.4.1 阶段一：空间预训练 (Spatial Pretraining)

在此阶段，时序模块被显式冻结（`requires_grad=False`），仅训练空间编码器与解码器。数据加载器以单帧模式（或将时序维度视为 batch 维度）提供输入。
训练目标仅包含单帧重建损失、谱损失与单帧观测一致性损失：
\[
L_{\text{spatial}} = L_{\mathrm{rec}} + \lambda_s L_{\mathrm{spec}} + \lambda_{dc} L_{\mathrm{dc}}
\]
该阶段通常持续 50-100 Epochs，确保空间重建质量达到稳定基线。

### 5.4.2 阶段二：时序预训练 (Temporal Pretraining)

在此阶段，已训练的空间模块被冻结，仅训练时序演化模块（如 ARWrapper）。采用 Teacher Forcing 策略，即输入为真实历史特征 \(z_{t-1}^{\text{GT}}\)，预测下一时刻特征 \(\hat{z}_t\)。
损失函数聚焦于特征空间的演化误差，避免解码器误差的干扰。

### 5.4.3 阶段三：联合微调与自回归滚动 (Joint Fine-tuning & AR Rollout)

全模型参数解冻。训练逻辑切换为 **20 步自回归滚动预测 (20-step AR Rollout)**：
1. **初始输入**：前 \(k\) 步使用真实观测；
2. **滚动预测**：后续步骤使用模型上一时刻的预测值 \(\hat{u}_{t-1}\) 作为输入（Teacher Forcing Decay 策略控制真值注入比例）；
3. **时序正则化**：引入时序导数损失 \(L_{\text{deriv}}\) 与能量演化损失 \(L_{\text{energy}}\)（见 3.5.5），约束长时动力学漂移。

这种分阶段策略在代码中通过 `freeze_spatial()`、`freeze_temporal()` 等方法与 `stage` 配置项严格控制，确保了从静态重建到动态演化的平滑过渡。

---

## 5.5 三件套损失的实现细则与数值注意事项5.3.4 解码器：双线性上采样 + (3\times 3) 卷积

上采样阶段采用“插值上采样 + 小卷积核细化”（典型为双线性 + (3\times 3)），用于抑制转置卷积常见的棋盘格伪影。经典分析指出：转置卷积因 overlap 模式易产生 checkerboard artifacts，“resize + conv”属于可行替代路径。

---

## 5.4 三件套损失的实现细则与数值注意事项

模型输出位于 z-score 域，逐通道反标准化得到原值域预测：

$$
\tilde u = \sigma \hat u^{(z)} + \mu.
$$

### 5.4.1 重建损失 (L_{\mathrm{rec}})

在 z-score 域：

$$
L_{\mathrm{rec}}=\left|\hat u^{(z)}-u^{(z)}\right|_2^2.
$$

该项提供主要收敛驱动力，适合作为所有实验的公共基线项。

### 5.4.2 低频谱一致性损失 (L_{\mathrm{spec}})

对二维 FFT 后的低频子空间 (\mathcal K_{\mathrm{low}}) 施加约束：

$$
L_{\mathrm{spec}}=\sum_{(k_x,k_y)\in\mathcal K_{\mathrm{low}}}
\left|\mathcal F_{2\mathrm{D}}(\hat u^{(z)})*{k_x,k_y}-\mathcal F*{2\mathrm{D}}(u^{(z)})_{k_x,k_y}\right|_2^2.
$$

工程实现注意点：

* 建议使用 `rfft2`（实输入）并固定归一化方式（例如 `norm="ortho"`），避免尺度漂移导致权重不可比；
* 多通道（(C>1)）时明确通道聚合方式（逐通道求和/加权），并写入配置；
* 为保证 SR 与 Crop 横向可比，(\mathcal K_{\mathrm{low}}) 的定义与分辨率绑定：可采用固定频率索引阈值 (k_{\max})，或采用按比例截断并在全实验保持一致。

### 5.4.3 原值域观测一致性损失 (L_{\mathrm{dc}})

$$
L_{\mathrm{dc}}=\left|H(\tilde u)-y\right|_2^2.
$$

该项把训练目标锚定到评测口径，减少“像素域更好但观测口径更差”的断裂风险。由于 (H) 可能包含模糊与下采样等操作，强制同源复用 (H\equiv \mathrm{DC}) 是该损失成立的先决条件。

### 5.4.4 总损失与权重配置

$$
L = L_{\mathrm{rec}}+\lambda_s L_{\mathrm{spec}}+\lambda_{dc}L_{\mathrm{dc}}.
$$

其中 (\lambda_s,\lambda_{dc}) 作为一级配置项写入 YAML，并与低频阈值联合扫描（例如 (k_{\max}\in[8,24])），为第6章敏感性分析提供可审计的“性能—口径—资源”折衷证据。

---

## 5.5 训练策略：优化器、学习率、混合精度与确定性

### 5.5.1 AdamW：权重衰减解耦

优化器采用 AdamW，以实现权重衰减与梯度更新解耦，使正则项行为更接近“真正的 L2 正则”。该做法由 AdamW 原始论文系统论证，属于成熟默认选项之一。

### 5.5.2 Cosine 学习率与 warmup

学习率日程采用 cosine 退火（可选 warm restarts）。warmup（如前若干 step 线性升温）作为工程稳定措施保留，并与 batch size、梯度裁剪阈值、混合精度策略共同写入配置快照，保证复现实验可严格对齐。

### 5.5.3 自动混合精度（AMP）

GPU 训练启用自动混合精度以提升吞吐、降低显存占用，典型组合为 autocast + GradScaler。AMP 开关与 scaler 配置写入环境指纹，确保跨实验对照一致。

### 5.5.4 确定性与可复现设置

确定性训练不仅依赖随机种子，还涉及算子选择与后端实现差异。建议将以下内容写入实验快照并在材料包可核验：

* 全局随机种子；
* 是否启用确定性算法（如 `torch.use_deterministic_algorithms(True)`）；
* cuDNN 的 deterministic/benchmark 配置；
* AMP 开关与 scaler 配置；
* 数据加载顺序与 worker 初始化策略；
* 分布式训练环境变量快照（如 DDP 的 rank/world size）。

---

## 5.6 资源四项统计：可复核的测量流程

资源统计采用固定口径：固定输入尺寸（例如 (256\times256)）、固定 batch、固定设备与固定预热策略。

### 5.6.1 显存峰值（GB）

建议测量流程：

1. `torch.cuda.reset_peak_memory_stats()` 清零峰值计数；
2. 预热若干次前向；
3. 记录 `torch.cuda.max_memory_allocated()` 作为峰值显存占用；
4. 记录 dtype、AMP 开关、batch size 与输入尺寸，避免口径漂移。

### 5.6.2 推理延迟（ms）

建议使用固定次数预热后进行多次计时（例如 (N=100)），报告均值±标准差；并记录 CUDA 同步策略（计时前后 `torch.cuda.synchronize()` 或使用 CUDA events），保证不同机器之间的对照可解释。

### 5.6.3 Params 与 FLOPs@(256^2)

* Params：统计可训练参数总量；
* FLOPs：采用成熟 FLOPs 工具（fvcore/ptflops/torch profiler 等）在固定输入尺寸上统计，并把工具版本与统计口径写入资源表，避免不同统计器导致的口径漂移。

---

## 5.7 伪代码与“可实现的算法盒”

### 5.7.1 训练主循环（伪代码）

```python
seed_all(SEED)

# --- build H / DC (single entrypoint) ---
H  = build_degradation(cfg.degradation)
DC = H  # hard constraint: same implementation & params

# --- model ---
model = Model(
    in_ch=cfg.model.in_ch,
    out_ch=cfg.model.out_ch,
    img_size=cfg.data.img_size,
    operator=cfg.model.operator,     # FNO/DeepONet/Hybrid/None
    decoder="bilinear+conv3x3",
    use_fourier_pe=cfg.model.use_fourier_pe
)

opt   = AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd)
sched = CosineScheduleWithWarmup(opt, warmup_steps=cfg.sched.warmup)

# --- gate: H/DC equivalence audit ---
assert check_equivalence(H, data_pipe, eps=1e-8)

for epoch in range(cfg.train.epochs):
    model.train()
    for batch in train_loader:
        y, u_z, mu, std = batch  # u_z: z-score GT

        x = pack_input(
            y=y,
            baseline=make_baseline(y),
            coords=make_coords(),
            mask=make_mask(y),
            fourier_pe=make_fourier_pe(cfg)
        )

        u_hat_z = model(x)           # z-score pred
        u_hat   = std * u_hat_z + mu # physical domain

        L_rec  = mse(u_hat_z, u_z)
        L_spec = lowfreq_fft_mse(u_hat_z, u_z, kmax=cfg.loss.kmax)
        L_dc   = mse(DC(u_hat), y)

        L = L_rec + cfg.loss.lam_spec * L_spec + cfg.loss.lam_dc * L_dc

        opt.zero_grad(set_to_none=True)
        L.backward()
        clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step(); sched.step()

    validate_and_log(...)
    save_snapshots_fingerprint(...)
```

上述伪代码把 “(H/\mathrm{DC}) 同源复用、原值域一致性损失、低频谱约束、门禁审计与快照归档” 放入同一最小闭环，便于第6章直接据此复现实验。

---

## 5.8 配置映射与材料包结构（建议落地标准）

为支持盲审与跨环境复现，建议将材料包结构固定为：

* `runs/<exp>/config_merged.yaml`：合并后的唯一配置源；
* `runs/<exp>/env_fingerprint.json`：PyTorch/CUDA/驱动/设备信息与 determinism 开关；
* `runs/<exp>/consistency_report.json`：(H/\mathrm{DC}) 等价性门禁报告；
* `paper_package/metrics/`：主指标表、显著性检验、效应量、资源表；
* `paper_package/figs/`：GT/Pred/Err、功率谱与边界放大、失败案例；
* `paper_package/scripts/`：一键复现脚本与一致性检查脚本；
* `README.md`：数据版本、口径参数、统计口径与运行命令。

---

## 5.9 小结

本章从工程可实现与可审计出发，给出本文算法的端到端设计：以 (H/\mathrm{DC}) 同源复用保证观测口径一致性，以编码—算子层—时空融合—解码的统一接口保证模块可替换，以三件套损失把优化目标锚定到评测口径与可解释频谱结构，并通过 AdamW、cosine 学习率、AMP 与确定性设置形成可复现训练闭环。资源四项统计流程固定测量口径，为第6章“性能—资源—口径”三维对照提供直接支撑。

---

## 参考文献（APA｜建议全文统一 BibTeX/APA 管线导出）

* Loshchilov, I., & Hutter, F. (2017). *Decoupled Weight Decay Regularization*. arXiv:1711.05101.
* Loshchilov, I., & Hutter, F. (2016). *SGDR: Stochastic Gradient Descent with Warm Restarts*. arXiv:1608.03983.
* Odena, A., Dumoulin, V., & Olah, C. (2016). *Deconvolution and Checkerboard Artifacts*. Distill, 1(10). doi:10.23915/distill.00003.
* OpenCV Documentation. *Geometric Image Transformations — resize() interpolation guidance (INTER_AREA preferable for shrinking)*.
* OpenCV Documentation. *GaussianBlur function reference (sigma, ksize, borderType)*.
* PyTorch Documentation. *Reproducibility / Randomness notes*.
* PyTorch Documentation. *Automatic Mixed Precision (AMP) and GradScaler*.
* PyTorch Documentation. *CUDA memory statistics: reset_peak_memory_stats / max_memory_allocated*.
