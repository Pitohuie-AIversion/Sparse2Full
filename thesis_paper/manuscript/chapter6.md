# 第6章 实验结果与分析

## 6.1 实验设置
- 数据集：采用 PDEBench（NeurIPS 2022），覆盖多类 PDE 与多初边界条件；遵循 FAIR 原则并引用 DOI 数据发布。
- 切分与标准化：固定 `splits/{train,val,test}.txt`；逐通道 z-score 标准化，生成并引用 `norm_stat.npz`。
- 观测生成：统一观测算子 H 与训练退化 DC 完全复用同一实现与配置（核/σ/插值/对齐/边界）。
- 确定性与快照：同一 YAML+种子下方差 ≤ 1e-4；训练开始写入 `runs/<exp>/config_merged.yaml` 与 `runs/<exp>/env_fingerprint.json`。
- 评测指标：`Rel-L2、MAE、PSNR、SSIM、fRMSE-low/mid/high、bRMSE、cRMSE、||H(ŷ)−y||`。
- 统计与显著性：≥3 种子报告均值±标准差；对主基线执行 paired t-test（Rel-L2）与 Cohen’s d。
- 资源四项：记录 Params(M)、FLOPs(G@256²)、显存峰值(GB)、推理延迟(ms)。

## 6.2 主实验结果
- 空间稀疏观测（SR 与 Crop）下的重建性能：在统一口径与三件套损失下，`||H(ŷ)−y||` 与 Rel-L2 同步下降，低频结构误差显著减小。
- 与 SOTA 对比：相较仅重建损失或口径不一致的训练设置，本文方法在多个 PDE 场景下取得稳健改进，并在资源四项上保持可接受的工程成本。
- 显著性检验：以主基线（如 FNO/DeepONet/PINN 变体）为参照，paired t-test（Rel-L2）显著性通过；Cohen’s d 显示中等至显著效应大小。

## 6.3 消融实验
- 去除 \(L_{spec}\)：大尺度结构误差上升；`||H(ŷ)−y||` 与 Rel-L2 的同步下降减弱。
- 去除 \(L_{dc}\)：评测口径断裂加剧，`||H(ŷ)−y||` 显著劣化。
- H/DC 不复用：核/σ/插值/对齐/边界不一致导致横向对比不公与域外性能下降。
- 解码替换：去除“双线性+3×3”出现棋盘格伪影与谱域噪声增大。
- 频谱阈值扫描：调整 \(k_x,k_y\) 阈值对性能的影响与资源开销的权衡。

## 6.4 可视化分析
- 标准图：GT/Pred/Err 热图（统一色标），功率谱（log），边界带局部放大。
- 代表案例：至少 3 个代表 case 的完整图组，展示低频结构恢复与边界带误差抑制。
- 失败案例：标注类型（边界层溢出/相位漂移/振铃/能量偏差），给出改进建议（调整 \(\sigma\)、边界策略、低频阈值与课程设置）。

## 6.5 资源与性能
- Params 与 FLOPs：在 `img_size=256` 下统计模型参数量与 FLOPs（G@256²）。
- 显存峰值与延迟：记录训练与推理的显存峰值（GB）与单次推理延迟（ms）。
- 资源-性能折衷：分析不同架构、损失设置与课程策略下的资源成本与性能收益。

## 6.6 结果小结与讨论
- 同步下降：三件套损失与口径一致性促使 `||H(ŷ)−y||` 与 Rel-L2 在多个 PDE 场景下同步下降。
- 鲁棒性：在跨分辨率与网格的评测下，统一口径与频谱一致性有助于降低离散化别名与评测断裂。
- 可复现性：配置快照与环境指纹、固定切分与种子、统一评测与显著性检验共同保障结论稳健。

---

## 6.7 统计与可视化自检清单
- 指标完整：`Rel-L2、MAE、PSNR、SSIM、fRMSE-low/mid/high、bRMSE、cRMSE、H_err(≡||H(ŷ)−y||)` 全部报告。
- 显著性：≥3 种子；paired t-test（Rel-L2）与 Cohen’s d；效应方向与理论一致。
- 资源四项：Params/FLOPs@256²/显存峰值/推理延迟完整记录；统一设备与输入尺寸。
- 可视化规范：统一色标；log 功率谱；边界带局部放大；图注标明数据来源与口径参数；矢量格式输出。
- 代表与失败案例：≥3 个代表案例；失败类型标注与改进建议齐备。

## 6.8 YAML 字段到实验的映射
- `metrics.enabled` 与指标产出脚本一致；`resources.enabled` 对齐资源统计流程。
- `degradation` 与 `dc` 字段镜像；一致性脚本运行并归档到 `runs/<exp>/consistency_report.json`。
- `curriculum` 字段驱动 SR 与 Crop 的课程切换；日志中标注阶段边界。
- `logging.save_config_merged` 与 `logging.save_env_fingerprint` 开启，保证快照与指纹可查。

## 6.9 结果再现与汇总
- 脚本：`paper_package/scripts/` 提供一键汇总与显著性检验脚本；
- 输出：`paper_package/metrics/` 主表与显著性，`paper_package/figs/` 图组；README 说明复现命令与依赖。
## 产出与材料
- `paper_package/metrics/`：主表（均值±标准差）、显著性报告（paired t-test + Cohen’s d）、资源表（Params/FLOPs/显存/延迟）。
- `paper_package/figs/`：代表图与失败案例图组、功率谱图。
- `paper_package/scripts/`：一键复现实验与汇总脚本；README 说明。

---

*最后更新：2025年*
