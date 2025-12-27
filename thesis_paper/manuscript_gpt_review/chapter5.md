- # 第5章 算法设计（重写：按“相关性—可行性”梳理，并核验关键出处）

  ## 引言

  本章给出本文方法的工程化算法设计与实现细节，目标是形成“可复现、可替换、可对照”的最小闭环：以统一观测口径为核心约束（数据侧观测算子 (H) 与训练侧退化算子 (\mathrm{DC}) **同源复用**），以统一模型接口为工程约束（模块可插拔、指标可横向对照），以确定性训练与材料化产出为复现约束（配置快照、环境指纹、统计检验与资源四项齐备）。本章内容面向研究生论文写作标准，强调每个工程选择背后的可验证性与审稿可复核性。

  ------

  ## 5.1 端到端流程与输入输出契约

  ### 5.1.1 任务输入与监督信号

  设真值流场为 (u_t\in\mathbb{R}^{C\times H\times W})，观测为
  [
  y_t = H(u_t) + n_t,
  ]
  其中 (H) 为任务指定的观测算子（SR 或 Crop），(n_t) 为噪声项（可为 0 或合成噪声）。训练样本包含 ((y_t,u_t)) 及标准化统计量 ((\mu,\sigma))（逐通道 z-score）。

  ### 5.1.2 统一接口：可替换模型的最小协议

  为保证模块可替换与评测一致性，模型采用统一签名：

  - 初始化：`__init__(in_ch, out_ch, img_size, **kwargs)`
  - 前向：`forward(x[B,C_in,H,W]) -> u_hat[B,C_out,H,W]`（输出位于 z-score 域）

  输入张量 (x) 由如下分量按通道拼接构成：

  1. `baseline`：基础重建（例如双线性上采样），提供稳定起点；
  2. `coords`：显式坐标编码（可叠加 Fourier 特征）；
  3. `mask`：观测缺失区域指示（Crop 场景尤其关键）；
  4. `fourier_pe`（可选）：频域位置编码，用于缓解频谱偏置与提升高频表达。

  ------

  ## 5.2 统一观测算子 (H) 与训练退化算子 (\mathrm{DC}) 的同源复用

  ### 5.2.1 单一入口原则与参数镜像

  工程上将 (H) 与 (\mathrm{DC}) 的实现收敛到**唯一入口函数**（例如 `ops/degradation.py::build_degradation(**cfg)`），并强制训练端直接复用同一实例（或同一配置签名构造出的完全一致实例）。该设计的核心目标是消除“训练口径—评测口径”不一致造成的隐性域偏差：一旦 (H\neq\mathrm{DC})，即使像素域误差下降，也可能出现评测口径误差 (||H(\hat u)-y||) 不降反升的“评测断裂”。

  为使该约束可审计、可阻断，建议在训练开始阶段执行一致性检验：对随机抽样的真值 (u) 计算 (H(u)) 并与数据管线给出的 (y) 对齐，要求 (\mathrm{MSE}(H(u),y)) 低于阈值（如 (10^{-8})）。不通过则直接终止实验并输出差异参数（核大小、(\sigma)、插值方式、对齐方式、边界策略等）。

  ### 5.2.2 SR 观测算子：抗混叠与插值选择

  SR 任务典型观测为
  [
  y_t^{\mathrm{SR}} = D_s!\left(k_\sigma * u_t\right),
  ]
  其中 (k_\sigma) 为高斯预滤波核，(D_s) 为缩小分辨率的下采样算子。

  工程实现中，下采样插值方式应遵循 OpenCV 的官方建议：对“缩小图像”的场景，`INTER_AREA` 通常优于其他插值方式，适合作为默认选项。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html?utm_source=chatgpt.com))
  在此基础上叠加高斯预滤波可进一步降低混叠风险，使训练端 (\mathrm{DC}) 与评测端 (H) 的频谱行为一致，便于后续通过频谱一致性损失进行可控优化。

  ### 5.2.3 Crop 观测算子：中心对齐、块对齐与边界策略

  Crop 观测可写为
  [
  y_t^{\mathrm{Crop}} = C_{h_c,w_c}(u_t),
  ]
  其中 (C_{h_c,w_c}) 为中心对齐裁剪算子。为与网络 patch/窗口机制兼容，裁剪窗口建议设置为 `patch_size` 的整数倍，避免训练与推理阶段发生额外 padding 或对齐偏差；边界策略在实现层面必须显式声明（mirror/zero/wrap），并写入配置快照，保证跨实验对照公平。

  ------

  ## 5.3 网络结构的工程化拆分：编码—算子层—解码

  ### 5.3.1 编码器：多尺度表征与时空融合

  编码器负责从 `baseline/coords/mask/(fourier_pe)` 中提取多尺度空间特征，并与时间维信息融合。可行实现包含：

  - 多尺度卷积块（稳定、成本可控）；
  - 注意力/Transformer 块（增强长程依赖）；
  - 频域模块（与算子层的谱域表达互补）。

  时间维融合可采用两类工程路线：
  （1）显式时序模块（ConvLSTM/Temporal Transformer）；（2）将时间作为坐标/条件输入，使用条件归一化或条件注意力进行隐式融合。二者应通过统一接口封装，确保替换仅影响模块内部而不改变训练与评测口径。

  ### 5.3.2 解码器：双线性上采样 + 卷积的选择依据

  上采样阶段采用“**双线性上采样 + 3×3 卷积**”的组合，目的在于抑制转置卷积常见的棋盘格伪影，提升空间重建的稳定性与可解释性。该选择与经典分析一致：对上采样过程进行显式插值再卷积，能够在结构上降低棋盘格伪影出现的概率。([Distill](https://distill.pub/2016/deconv-checkerboard/))

  ------

  ## 5.4 三件套损失的实现细则与数值注意事项

  模型输出 (\hat u) 位于 z-score 域，原值域预测为
  [
  \tilde u = \sigma \hat u + \mu.
  ]

  ### 5.4.1 重建损失 (L_{\mathrm{rec}})

  [
  L_{\mathrm{rec}} = |\hat u - u|_2^2,
  ]
  对应像素/网格点层面的直接监督，作为主要收敛驱动力。

  ### 5.4.2 低频谱一致性损失 (L_{\mathrm{spec}})

  对二维 FFT 后的低频子空间 (\mathcal K_{\mathrm{low}})（例如 (k_x,k_y\le 16)）约束：
  [
  L_{\mathrm{spec}} = \sum_{(k_x,k_y)\in\mathcal K_{\mathrm{low}}}
  \left| \mathcal F_{2\mathrm{D}}(\hat u)*{k_x,k_y} - \mathcal F*{2\mathrm{D}}(u)_{k_x,k_y}\right|_2^2.
  ]

  工程实现注意点：

  - 建议使用 `rfft2`（实输入）并固定归一化方式（如 `norm="ortho"`），避免尺度漂移导致权重不可比；
  - 若多通道（(C>1)），应明确通道聚合方式（逐通道求和/加权）并写入配置；
  - 为保证 SR 与 Crop 可横向对比，(\mathcal K_{\mathrm{low}}) 的定义需要与分辨率绑定（例如按比例阈值或固定频率索引），避免不同分辨率下“低频”含义漂移。

  ### 5.4.3 原值域观测一致性损失 (L_{\mathrm{dc}})

  [
  L_{\mathrm{dc}} = |H(\tilde u) - y|_2^2.
  ]

  该项直接把训练目标锚定到评测口径，减少“像素域更好但观测口径更差”的断裂风险。由于 (H) 可能包含模糊与下采样等操作，强制同源复用 (H\equiv \mathrm{DC}) 是该损失成立的先决条件。

  ### 5.4.4 总损失

  [
  L = L_{\mathrm{rec}} + \lambda_s L_{\mathrm{spec}} + \lambda_{dc} L_{\mathrm{dc}},
  ]
  其中 (\lambda_s,\lambda_{dc}) 为可扫描超参数。为便于第6章开展敏感性分析，本章建议将三项权重作为一级配置项纳入 YAML，并对 (\lambda_s) 与低频阈值联合扫描（例如 (k_{\max}\in[8,24])）。

  ------

  ## 5.5 训练策略：优化器、学习率、混合精度与确定性

  ### 5.5.1 AdamW：权重衰减解耦的选择依据

  优化器采用 AdamW，其关键在于将权重衰减与梯度更新解耦，使正则项行为更接近“真正的 L2 正则”。该做法由 AdamW 原始论文系统论证，属于深度学习训练的成熟默认选项之一。([arXiv](https://arxiv.org/abs/1711.05101))

  ### 5.5.2 Cosine 学习率与 warmup

  学习率日程采用 cosine 退火（可选 warm restarts），其动机与形式在 SGDR 工作中得到系统讨论与验证。([arXiv](https://arxiv.org/abs/1608.03983))
  warmup（如前 1k step 线性升温）建议作为工程稳定措施保留，并与 batch size、混合精度策略共同写入配置快照，保证复现实验可严格对齐。

  ### 5.5.3 自动混合精度（AMP）

  在 GPU 训练中启用自动混合精度以提升吞吐、降低显存占用，典型组合为 autocast + GradScaler。PyTorch 对 GradScaler 的用法与行为在官方文档中给出明确接口与注意事项。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/amp.html))

  ### 5.5.4 确定性与可复现设置

  确定性训练不仅依赖随机种子，还涉及算子选择与后端实现差异。PyTorch 的官方“Reproducibility”说明明确指出：完全可复现往往需要设置确定性算法、控制 cuDNN 行为并接受一定性能损失。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html))
  因此建议将以下内容写入实验快照（并在论文材料包中可核验）：

  - 全局随机种子；
  - 是否启用确定性算法（如 `torch.use_deterministic_algorithms(True)`）；
  - cuDNN 的 deterministic/benchmark 配置；
  - AMP 开关与 scaler 配置；
  - 数据加载顺序与 worker 初始化策略。

  ------

  ## 5.6 资源四项统计：可复核的测量流程

  为保证“性能—资源”对照可复核，资源统计应固定输入尺寸（例如 256²）、固定 batch、固定设备与固定预热策略。

  ### 5.6.1 显存峰值（GB）

  建议测量流程：

  1. `torch.cuda.reset_peak_memory_stats()` 清零峰值计数；
  2. 预热若干次前向；
  3. 记录 `torch.cuda.max_memory_allocated()` 作为峰值显存占用。PyTorch 对峰值显存查询接口给出明确语义说明。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html))

  ### 5.6.2 推理延迟（ms）

  建议使用固定次数预热后进行多次计时（例如 N=100），报告均值±标准差；并记录 CUDA 同步策略（计时前后 `torch.cuda.synchronize()`），保证不同机器之间的对照可解释。

  ### 5.6.3 Params 与 FLOPs@256²

  - Params：直接统计可训练参数总量；
  - FLOPs：建议采用成熟 FLOPs 工具（例如 fvcore/ptflops/torch profiler）在固定输入尺寸上统计，并将工具版本与统计口径写入资源表，避免不同统计器导致的口径漂移。

  ------

  ## 5.7 伪代码与“可实现的算法盒”

  ### 5.7.1 训练主循环（伪代码）

  ```python
  seed_all(SEED)
  
  # --- build H / DC (single entrypoint) ---
  H = build_degradation(cfg.degradation)  # includes SR/Crop params
  DC = H  # hard constraint: same instance or same signature
  
  # --- model ---
  model = Model(in_ch=cfg.model.in_ch,
                out_ch=cfg.model.out_ch,
                img_size=cfg.data.img_size,
                operator=cfg.model.operator,          # FNO/DeepONet/None
                decoder="bilinear+conv3x3",
                use_fourier_pe=cfg.model.use_fourier_pe)
  
  opt = AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd)
  sched = CosineScheduleWithWarmup(opt, warmup_steps=cfg.sched.warmup)
  
  for epoch in range(cfg.train.epochs):
      model.train()
      for batch in train_loader:
          y, u_z, mu, std = batch            # u_z: z-score GT
          x = pack_input(y=y,
                         baseline=make_baseline(y),
                         coords=make_coords(),
                         mask=make_mask(y),
                         fourier_pe=make_fourier_pe())
  
          u_hat_z = model(x)                 # z-score pred
          u_hat = std * u_hat_z + mu         # back to physical domain
  
          L_rec  = mse(u_hat_z, u_z)
          L_spec = lowfreq_fft_mse(u_hat_z, u_z, kmax=cfg.loss.kmax)
          L_dc   = mse(DC(u_hat), y)
  
          L = L_rec + cfg.loss.lam_spec * L_spec + cfg.loss.lam_dc * L_dc
  
          opt.zero_grad(set_to_none=True)
          L.backward()
          clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
          opt.step(); sched.step()
  
      validate_and_log(...)
      save_snapshot_and_fingerprint(...)
  ```

  > 上述伪代码将 “H/DC 同源复用、原值域一致性损失、低频谱约束、资源统计与快照” 放入同一最小闭环，便于第6章直接据此复现实验。

  ------

  ## 5.8 配置映射与材料包结构（建议落地标准）

  为支持盲审与跨环境复现，建议将材料包结构固定为：

  - `runs/<exp>/config_merged.yaml`：合并后的唯一配置源；
  - `runs/<exp>/env_fingerprint.json`：PyTorch/CUDA/驱动/设备信息与 determinism 开关；
  - `paper_package/metrics/`：主表、显著性检验结果、资源表；
  - `paper_package/figs/`：GT/Pred/Err、功率谱与边界带放大、失败案例；
  - `paper_package/scripts/`：一键复现脚本与一致性检查脚本；
  - `README.md`：命令行、数据版本、口径参数与统计口径说明。

  ------

  ## 5.9 小结

  本章从工程可实现与可审计出发，给出了本文算法的端到端设计：以 (H/\mathrm{DC}) 同源复用保证观测口径一致性，以编码—算子层—解码的统一接口保证模块可替换，以三件套损失将优化目标锚定到评测口径与可解释频谱结构，并通过 AdamW、cosine 学习率、AMP 与确定性设置形成可复现训练闭环。资源四项统计流程给出可复核口径，为第6章“性能—资源—口径”三维对照提供直接支撑。

  ------

  ## 参考文献（APA，去除裸链接；关键出处已核验）

  - Loshchilov, I., & Hutter, F. (2017). *Decoupled Weight Decay Regularization*. arXiv:1711.05101. ([arXiv](https://arxiv.org/abs/1711.05101))
  - Loshchilov, I., & Hutter, F. (2016). *SGDR: Stochastic Gradient Descent with Warm Restarts*. arXiv:1608.03983. ([arXiv](https://arxiv.org/abs/1608.03983))
  - Odena, A., Dumoulin, V., & Olah, C. (2016). *Deconvolution and Checkerboard Artifacts*. Distill. doi:10.23915/distill.00003. ([Distill](https://distill.pub/2016/deconv-checkerboard/))
  - OpenCV Documentation. *Geometric Image Transformations — resize() interpolation guidance (INTER_AREA preferable for shrinking)*. ([PyTorch Documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html?utm_source=chatgpt.com))
  - PyTorch Documentation. *Reproducibility (randomness and determinism notes)*. ([PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html))
  - PyTorch Documentation. *torch.cuda.amp GradScaler*. ([PyTorch Documentation](https://docs.pytorch.org/docs/stable/amp.html))
  - PyTorch Documentation. *torch.cuda.memory.max_memory_allocated*. ([PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html))

  如果你希望继续保持与前几章一致的写作颗粒度，我可以按同样标准把第6章“实验结果与分析”重写为：实验设置→指标与显著性→主结果→消融与敏感性→跨网格→失败案例→资源对照→可复现清单，并把每个结果表格对应到本章的“算法盒/配置字段/口径参数”。
