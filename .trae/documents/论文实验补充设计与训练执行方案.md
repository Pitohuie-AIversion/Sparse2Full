## 实验目标与核心假设

* 强化“稀疏观测驱动的时空流场重建”主张：在统一观测算子 `H` 与训练数据一致性（DC 一致）下，模型在不同任务/数据/稀疏度/噪声下稳定优于基线。

* 以“可复现、可比性、统一接口”为约束（黄金法则 0），覆盖空间与时序，包含损失消融、课程训练、鲁棒性与跨数据集泛化。

## 数据与任务覆盖

* 任务：`SR×2`、`SR×4`、`Crop-40%`、`Crop-20%`；可选：时序 `Temporal`（AR/NAR）。

* 数据集（PDEBench 选段）：`DarcyFlow-2D`、`Diffusion-Reaction-2D`、`Navier–Stokes-2D`（与现有 YAML 对齐）。

* 切分：固定 `splits/{train,val,test}.txt`；标准化：逐通道 z-score，训练产出 `norm_stat.npz`。

* 观测生成：`SR` 用 `GaussianBlur(σ,k=5)+INTER_AREA×s`；`Crop` 对齐中心与 `patch_size` 倍数，边界策略显式（mirror/zero/wrap）。

## 观测算子与一致性验证

* 统一使用 `ops/degradation.py` 作为 `H`；训练中的 DC 复用同一实现与配置（核/σ/插值/对齐/边界）。

* 运行一致性脚本：`tools/check_dc_equivalence.py`，随机 100 case 验证 `MSE(H(GT), y) < 1e-8`。

* 配置键位在 Hydra YAML 显式声明，禁止硬编码：`configs/datasets/*` 与 `configs/train/*`、`configs/ops/*` 参数一一对应。

## 模型与基线集

* 最小主模型：`Swin-UNet`（可选 FNO 瓶颈）。

* 扩展比较：`U-Net/U-Net++`、`FNO/U-FNO`、`Hybrid(Attn∥FNO∥UNet)`、`MLP/MLP-Mixer`、`LIIF-Head`、`SegFormer/UNetFormer`。

* 统一接口：`forward(x[B,C_in,H,W])→y[B,C_out,H,W]`；`__init__(in_ch,out_ch,img_size,**kwargs)`。

* 参考配置：`configs/model/swin_unet.yaml`、`unet.yaml`、`fno2d.yaml`、`hybrid.yaml`、`liif.yaml`、`segformer.yaml`。

## 训练协议与资源记录

* 优化：`AdamW(lr=1e-3, wd=1e-4)`、`Cosine + 1k warmup`、AMP、梯度裁剪 `1.0`、DDP。

* 种子：`[0,1,2]`，固定随机与确定性开关；同一 YAML + 种子，验证指标方差 ≤ `1e-4`。

* 课程：`SR` 先 ×2 再 ×4；`Crop` 先大窗后小窗（如 40%→20%）。

* 采样（Crop）：均匀 40% + 边界 30% + 高梯度 30%。

* 资源统计：记录 Params(M)、FLOPs(G\@256²)、显存峰值(GB)、推理延迟(ms)。

* 实验命名：`<task>-<data>-<res>-<model>-<keyhyper>-<seed>-<date>`；训练开始写入快照 `runs/<exp>/config_merged.yaml`。

## 损失与值域处理

* 总损失：`L = L_rec + λ_s L_spec(low-freq) + λ_dc`，默认权重 `1.0/0.5/1.0`。

* 频域：比较 `kx=ky=16` 低频模；非周期数据可镜像延拓。

* 值域：模型输出在 z-score 域；`DC` 与频域损失在原值域计算（反归一化 `μ/σ`）。

* 单测验证“z-score→原值域→H”管线无偏；训练曲线中 `||H(ŷ)−y||` 与 Rel-L2 同步下降。

## 评测与统计

* 指标：`Rel-L2`、`MAE`、`PSNR`、`SSIM`、`fRMSE-low/mid/high`、`bRMSE(边界带16px)`、`cRMSE`、`||H(ŷ)−y||`。

* 聚合：先通道后等权平均；≥3 种子，报告均值±标准差；对主基线做 `paired t-test(Rel-L2)` 与 `Cohen’s d`。

* 工具：训练后用 `tools/eval.py` 产出 `metrics.jsonl`；`tools/summarize_runs.py` 自动生成 `paper_package/metrics/` 的主表与显著性报告与资源表。

## 消融与鲁棒性补充设计

* 损失消融：移除 `L_dc`、移除 `L_spec`、权重扫描（如 `λ_s∈{0,0.25,0.5,1.0}`、`λ_dc∈{0,0.5,1.0,2.0}`）。

* H/DC 一致性消融：刻意制造轻微不一致（例如 `σ` 或插值方式差异），观察 `||H(ŷ)−y||` 与重建质量变化。

* 解码器对比：`双线性+3×3` vs 反卷积，检验棋盘格伪影。

* 分辨率与稀疏度：`SR×2/×4/×8`、`Crop-20/40/60%`；掩膜密度与分布（均匀/边界/高梯度）。

* 噪声鲁棒性：在观测 `y` 注入 `Gaussian/Poisson` 噪声不同 SNR；报告鲁棒曲线。

* 边界策略：`mirror/zero/wrap` 对比，关注 `bRMSE` 与边界层误差。

## 可视化与失败诊断

* 标准图：GT/Pred/Err 热图（统一色标）、功率谱（log）、边界带局部放大。

* 失败案例标注：边界层溢出/相位漂移/振铃/能量偏差；给出改进建议（如增广、频域权重）。

* 输出到 `paper_package/figs/`，至少每主实验 3 个代表 case。

## 时序扩展实验（可选）

* 使用 `models/temporal/*` 与 `wrappers/ar_nar_wrapper.py` 比较 `AR` 与 `NAR`。

* 验证 `temporal_consistency` 与 `sequential_dc_consistency`；评估随时间步误差增长与 `||H(ŷ_t)−y_t||`。

* 课程：短序列→长序列；资源与稳定性报告。

## 复现与材料打包

* 完整材料包目录：`paper_package/`（数据卡、最终 YAML 快照、关键 ckpt(LFS)、metrics、figs、scripts、README）。

* 支持盲审模式导出：隐藏作者/路径；`make paper_package` 一键产出。

## 训练与评估命令模板（由你执行）

* 训练：`python tools/train.py task=sr_x4 datasets=<dataset_yaml> model=swin_unet seed=<0|1|2> +logging.resource=true`

* 评估：`python tools/eval.py runs=<exp_dir> eval=comprehensive`

* 汇总：`python tools/summarize_runs.py runs=<exp_parent_dir> out=paper_package/metrics`

* 一致性：`python tools/check_dc_equivalence.py configs/check_dc.yaml`

## 交付物清单（论文口径）

* 主表（各任务/数据上的指标，均值±标准差，显著性）与资源表。

* 标准可视化与失败诊断图集；代表 case 的谱图与边界放大图。

* 配套 YAML/脚本与一致性/复现日志；`runs/<exp>/config_merged.yaml` 快照齐备。

## 风险与检查清单

* `H` 与训练 DC 完全一致；一致性脚本通过。

* 统一接口与类型；`ruff+black+isort`、`mypy --strict`、`pytest -q` 通过。

* 种子控制与指标方差 ≤ `1e-4`；资源四项完整记录。

* 汇总脚本产出 `paper_package/metrics/` 主表、显著性与资源表；可匿名化导出。

