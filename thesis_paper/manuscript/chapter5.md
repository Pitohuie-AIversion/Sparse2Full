# 第5章 算法设计

## 引言
本章详细阐述本文方法的算法实现：整体流程、关键组件（观测算子、网络架构、损失计算）、超参数设置与训练策略、伪代码与工程要点。所有实现遵循统一口径（H/DC 复用）、统一接口与确定性训练，配合评测与显著性检验协议，确保可复现与横向可比。

## 5.1 整体算法流程
- 输入：固定切分数据 `splits/{train,val,test}.txt`，逐通道 z-score 标准化（生成 `norm_stat.npz`）。
- 观测生成：统一入口构造 \(H\)，SR 为 `GaussianBlur(σ,k=5)+INTER_AREA downsample×s`；Crop 为居中对齐窗口 \((h_c,w_c)\) 与明确边界策略。
- 训练退化：\(\text{DC}\) 复用 \(H\) 的同一实现与配置；一致性脚本 `tools/check_dc_equivalence.py` 抽样≥100 验证 \(\text{MSE}(H(GT), y) < 10^{-8}\)。
- 模型：时空耦合编码 + 算子层（可选 FNO/DeepONet）+ 双线性+3×3 解码；输出在 z-score 域。
- 损失：`L = L_rec + λ_s L_spec + λ_dc L_dc`（默认 1.0/0.5/1.0），其中 \(L_{spec}\) 仅比较低频模（\(k_x,k_y\le 16\)），\(L_{dc}\) 在原值域计算。
- 训练策略：AdamW、Cosine+1k warmup、AMP、梯度裁剪 1.0；课程：SR ×2→×4；Crop 窗 40%→20%。
- 评测：指标集与资源四项；≥3 种子显著性检验；标准图与失败案例可视化；`paper_package/` 产出材料包。

## 5.2 关键组件实现
### 5.2.1 观测算子与一致性入口
- 统一模块：`ops/degradation.py` 提供 H/DC 的唯一实现，参数包括 \(\sigma,k=5,s,(h_c,w_c)\)、边界策略与插值方式；训练脚本与数据管线强制从此入口获取实例。
- SR：先 `GaussianBlur(σ,k=5)` 再 `INTER_AREA` 下采样×`s`；\(\sigma\) 可依据 Nyquist 近似与经验规则设定。
- Crop：窗口居中与 patch_size 倍数对齐，边界策略镜像/零填充/环绕明确。

### 5.2.2 时空耦合网络架构
- 输入打包：`[baseline, coords, mask, (fourier_pe?)]`。`baseline` 可为双线性上采样；`coords` 为显式坐标；`mask` 表示缺失。
- 编码：多尺度卷积/注意力/频域块（可含 Fourier 特征编码）抽取空间特征；时间维通过序列模块或坐标编码融合。
- 算子层：可选 FNO（谱域核）或 DeepONet（分支/主干）实现函数空间映射；可选 PINN 残差项引入物理约束。
- 解码：双线性 + 3×3 卷积减少棋盘格；输出置于 z-score 域，便于与标准化流程衔接。

### 5.2.3 损失计算与训练循环
- 反归一化：\(\tilde{u}=\sigma\hat{u}+\mu\)。
- \(L_{rec}\)：预测与真值在 z-score 域的点误差。
- \(L_{spec}\)：二维 FFT 后低频模（\(k_x,k_y\le 16\)）的差异。
- \(L_{dc}\)：\(\| H(\tilde{u}) - y \|_2^2\) 在原值域计算。
- 总损失加权与反传；记录训练/验证日志与资源四项。

## 5.3 超参数设置
- 优化器：AdamW(lr=1e-3, wd=1e-4)。
- 学习率策略：Cosine annealing + 1000 warmup steps。
- 精度与稳定：AMP 开启，梯度裁剪 1.0。
- 课程：SR 由 ×2 过渡至 ×4；Crop 由 40% 窗过渡至 20%。
- 频谱阈值：\(k_x,k_y\le 16\)。
- 观测参数：\(\sigma,k=5,s,(h_c,w_c)\) 与边界策略（mirror/zero/wrap）。
- 随机性：固定随机种子；DDP 时统一精度策略与确定性开关。

## 5.4 训练与评测伪代码

```
# 伪代码（Python 风格）
seed_all(SEED)
H = build_degradation(sigma, k=5, s, crop=(h_c,w_c), boundary=mode, interp="INTER_AREA")
DC = H  # 强制复用
model = Model(in_ch, out_ch, img_size, use_fno=True, decoder="bilinear+conv3x3", fourier_pe=True)
opt = AdamW(model.params(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineLR(opt, warmup_steps=1000)

for epoch in range(E):
    for batch in train_loader:
        y, u, mu, std = batch  # y=H(u)+noise; std 为 z-score 标准差，避免与模糊 σ 混淆
        u_hat = model.forward(pack_input(y, baseline, coords, mask, fourier_pe))
        u_tilde = std * u_hat + mu
        L_rec = mse(u_hat, u)
        L_spec = lowfreq_fft_mse(u_hat, u, kx_ky_max=16)
        L_dc = mse(H(u_tilde), y)
        L = L_rec + 0.5 * L_spec + 1.0 * L_dc
        L.backward()
        clip_grad_norm_(model.params(), max_norm=1.0)
        opt.step(); opt.zero_grad(); scheduler.step()
    validate_and_log(metrics=[RelL2, MAE, PSNR, SSIM, fRMSE_low_mid_high, bRMSE, cRMSE, ||H(u_tilde)-y||],
                     resources=[Params, FLOPs_2562, MaxMemGB, Latency_ms])
```

## 5.5 工程要点与规范
- 唯一入口与配置快照：`ops/degradation.py` 为 H/DC 单一源；训练开始写入 `runs/<exp>/config_merged.yaml` 与 `runs/<exp>/env_fingerprint.json`。
- CI 与测试：`ruff+black+isort`、`mypy --strict`、`pytest -q`；一致性脚本与确定性验证通过后才合并。
- 结果材料：`paper_package/metrics/` 产出主表与显著性报告；`paper_package/figs/` 产出代表图与失败案例；脚本与 README 形成一键复现。

## 5.6 小结
本章完成了算法的端到端设计与实现细节，包括统一观测口径与训练复用、时空耦合架构的组件选择、三件套损失与训练确定性、超参数与伪代码，以及工程落地的规范与材料产出。为第6章的实验结果与分析提供明确的执行与评测依据。

---

## 5.7 配置映射与 YAML 字段
- 数据与切分：`dataset.name`、`splits.{train,val,test}`、`norm_stat_path`；确保固定切分与逐通道 z-score。
- 观测算子：`degradation.sr.{sigma,k=5,s,interp=INTER_AREA}`、`degradation.crop.{h_c,w_c,boundary}`；训练端 `dc:` 字段复用同一入口与参数。
- 训练超参：`optim.{type=AdamW,lr=1e-3,wd=1e-4}`、`sched.{type=Cosine,warmup_steps=1000}`、`amp=true`、`grad_clip=1.0`、`seed`、`ddp.precision_policy`。
- 课程策略：`curriculum.sr={stages:[2,4]}`、`curriculum.crop={windows:[0.4,0.2]}`。
- 评测与日志：`metrics.enabled=[RelL2,MAE,PSNR,SSIM,fRMSE,bRMSE,cRMSE,H_err]`、`resources.enabled=[Params,FLOPs_2562,MaxMemGB,Latency_ms]`、`logging.save_config_merged=true`、`logging.save_env_fingerprint=true`。
- 说明：`H_err ≡ ||H(ŷ)−y||`，与第3章变量表中定义一致。
- 输出材料：`paper_package.{metrics_dir,figs_dir,scripts_dir}`；生成主表、显著性与代表图。

## 5.8 资源统计方法与测量流程
- Params(M)：统计模型参数总量并换算到百万单位。
- FLOPs(G@256²)：在 `img_size=256` 的标准输入上进行前向 FLOPs 估算并以 G 次为单位报告。
- 显存峰值(GB)：训练与推理阶段使用 `torch.cuda.max_memory_allocated()` 捕获峰值并换算为 GB。
- 推理延迟(ms)：预热若干次后测量 N 次前向时间，报告均值与标准差；固定设备、批量与输入尺寸。
- 稳定性：所有测量在同一 YAML+种子、同一设备指纹下进行；记录到 `runs/<exp>/resources.json`。

## 5.9 一致性与可视化脚本
- 一致性脚本：运行 `tools/check_dc_equivalence.py`，抽样≥100，验证 `MSE(H(GT), y) < 1e-8`；失败时输出差异参数并阻断合并。
- 可视化生成：标准化色标的 GT/Pred/Err、log 功率谱与边界带放大图，输出到 `paper_package/figs/`；图注包含数据来源、口径参数与尺寸。
- 失败案例归档：按类型（边界层溢出/相位漂移/振铃/能量偏差）归档并附改进建议。

## 5.10 写作与实现自检清单
- 接口与输入：`forward(x[B,C_in,H,W])→y[B,C_out,H,W]`；`pack_input` 字段齐全且一致。
- H/DC 复用：训练与数据从同一入口获取实例；一致性脚本通过。
- 训练确定性：固定种子、AMP 与梯度裁剪、Cosine+warmup；验证方差 ≤ `1e-4`。
- 评测与统计：统一指标、≥3 种子显著性（paired t-test 与 Cohen’s d）、资源四项完整。
- 材料与快照：`config_merged.yaml` 与 `env_fingerprint.json` 存在；`paper_package/metrics/` 与 `paper_package/figs/` 完整；脚本与 README 可运行。
*最后更新：2025年*
