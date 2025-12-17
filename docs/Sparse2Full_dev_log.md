# Sparse2Full开发文档记录

## 项目总览
- 目标：构建统一的稀疏观测到稠密时空重建框架（Sparse2Full），兼顾空间细节与时间稳定性，并满足黄金法则（H/DC一致、可复现、统一接口、可比性、文档先行、TDD）。
- 里程碑：
  - 空间主干（Swin-UNet/UNet/FNO/Hybrid）联通与统一接口
  - 时间Transformer与NAR预测头集成，支持AR/NAR/Hybrid
  - 观测算子H与训练DC一致性验证通过
  - 评测指标与资源成本标准化输出
- 模块清单：`models/`（spatial/temporal）、`datasets/`、`ops/`、`tools/`、`configs/`、`paper_package/`。

## 架构与接口
- 统一接口：所有模型遵循`forward(x[B,C_in,H,W])→y[B,C_out,H,W]`，输入打包`[baseline, coords, mask, (fourier_pe?)]`。
- 空间主干（Swin-UNet）：层次化窗口注意力，编码器–解码器对称结构与跳跃连接；可选FNO瓶颈增强频域全局耦合。
- 时间建模（Temporal Transformer）：因果掩码的自注意力，跨时间步全局依赖；与NAR/AR头解耦。
- 预测头：
  - AR：逐步递推，适配短时依赖但易误差累积与时延增加
  - NAR：并行多步预测，稳定性与延迟优势
- 关键代码锚点：
  - 稀疏注意力编码器：`models/spatial/sparse_attention_encoder.py:14`, `:159`
  - 稀疏Swin-UNet集成：`models/spatial/sparse_attention_encoder.py:313`, `:367`
  - 时序包装统一接口：`models/temporal/wrappers/swin_temporal_wrapper.py:360`
  - NAR预测头：`models/temporal/components/nar_prediction_head.py`

## 数据与观测算子H一致性
- 观测管线：
  - 超分辨率（SR）：`GaussianBlur(σ,k=5)+INTER_AREA`下采样×s
  - 裁剪（Crop）：中心对齐，patch_size倍数约束；边界策略（mirror/zero/wrap）明确
- 训练DC与数据观测一致：训练中的数据一致性损失与生成观测的H完全复用同一实现与配置。
- 代码锚点：
  - 统一退化入口：`ops/degradation.py:197`
  - SR观测算子：`ops/degradation.py:241`
  - 裁剪观测算子：`ops/degradation.py:280`
- DC损失与值域：模型输出在z-score域；DC与频域损失在原值域计算（反归一化`μ/σ`）；参考`ops/loss.py`。

## 模型模块
- 空间：Swin-UNet（层次窗口注意力）、UNet（卷积基线）、FNO（频域神经算子）、U-FNO/Hybrid（瓶颈融合）。
- FNO瓶颈：在编码器输出与解码器输入之间插入，增强低频能量与全局耦合，适用于扩散/对流–扩散/高雷诺数场景。
- 时间：Temporal Transformer（层数/头数可配、因果掩码）；与AR/NAR头解耦。
- 预测：NAR并行生成`T_out`未来时刻，抑制误差累积与延迟线性增长；AR用于短时或教师强制/调度采样场景。

## 训练与配置
- Hydra分层：数据/模型/训练/损失分层；关键超参不应硬编码。
- 示例配置：`configs/train/ar_training_config_debug_temporal.yaml`（时序调试），`configs/train/sr_curriculum.yaml`（SR课程），`configs/train/crop_curriculum.yaml`（裁剪课程）。
- 优化与调度：AdamW（lr=1e-3, wd=1e-4）、Cosine+1k warmup、AMP、梯度裁剪1.0；DDP优先；固定随机种子与确定性开关。
- 采样策略（Crop）：均匀40% + 边界30% + 高梯度30%。

## 评测与资源
- 指标：Rel-L2、MAE、PSNR、SSIM、fRMSE-low/mid/high、bRMSE（边界带16px比例）、cRMSE、`||H(ŷ)−y||`。
- 聚合：每通道先算后等权平均；≥3种子报告均值±标准差；对主基线做paired t-test（Rel-L2）+ Cohen’s d。
- 资源：记录Params(M)、FLOPs(G@256²)、显存峰值(GB)、推理延迟(ms)。
- 工具：`eval.py`产出`metrics.jsonl`；`tools/summarize_runs.py`生成主表与显著性报告。

## 复现与集成
- YAML快照：训练开始时合并后的YAML写入`runs/<exp>/config_merged.yaml`，保留环境与版本信息。
- 数据：PDEBench数据卡（来源/许可/切分）置于`paper_package/data_cards/`；固定`splits/{train,val,test}.txt`；逐通道z-score标准化，`norm_stat.npz`随实验产出。

## 问题与迭代记录
- 训练卡顿与AMP精度不一致：参考故障分析与修复计划文档（`.trae/documents/修复训练卡顿与AMP精度不一致的代码与配置改动方案.md`等）。
- 已知失败类型：边界层溢出/相位漂移/振铃/能量偏差；为代表案例提供可视化与改进建议。

## 相关工作与引用
- Senseiver（稀疏注意力集合到场）：跨注意力编码将稀疏传感器集映射到统一潜在空间，输入规模无关；适合作为稀疏到稠密重建的代表方法引用，与Sparse2Full的层次Swin+FNO瓶颈+时间Transformer+NAR路线互补。参考：
  - Santos et al., “Development of the Senseiver for efficient field reconstruction from sparse observations,” Nat. Mach. Intell., 2023. URL: https://www.nature.com/articles/s42256-023-00746-x
  - Senseiver GitHub: https://github.com/OrchardLANL/Senseiver
- PDEBench基准：NeurIPS 2022主文与补充、OpenReview与GitHub，作为数据与评测基线。参考：
  - arXiv: https://arxiv.org/abs/2210.07182
  - NeurIPS PDF: https://papers.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf
  - OpenReview: https://openreview.net/forum?id=dh_MkX0QfrK
  - GitHub: https://github.com/pdebench/PDEBench

## 维护策略
- 每次主要改动更新本记录：新增模块需补最小单测与接口说明；训练脚本演进补充变更日志；发布前确保CI全绿与`paper_package`材料一致。

