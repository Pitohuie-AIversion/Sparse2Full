## 文档目标
- 形成一份系统的开发文档，覆盖架构、数据与观测算子H一致性、模型模块、训练配置、评测与资源、复现与集成、问题与迭代记录。
- 与仓库现有规范保持一致（黄金法则、统一接口与Hydra配置、可复现与统计报告）。

## 放置位置与命名
- 目录：`.trae/documents/`
- 文件名：`Sparse2Full开发文档记录.md`
- 语言：中文（与现有内部文档保持一致）。

## 文档结构
- **项目总览**：目标与里程碑；与论文草稿的对应关系；关键模块清单与依赖。
- **架构与接口**：
  - 模块图：空间主干（Swin-UNet/FNO瓶颈/稀疏注意力前端）、时间建模（Temporal Transformer）、预测头（AR/NAR）。
  - 统一接口：`forward(x[B,C_in,H,W])→y[B,C_out,H,W]`；输入打包约定。
  - 代码锚点：
    - 稀疏注意力集成：`models/spatial/sparse_attention_encoder.py:14`, `:159`, `:313`, `:367`
    - 时序包装与接口：`models/temporal/wrappers/swin_temporal_wrapper.py:360`
    - NAR预测头：`models/temporal/components/nar_prediction_head.py`
- **数据与观测算子H一致性**：
  - 观测管线：SR（`GaussianBlur(σ,k=5)+INTER_AREA×s`）、Crop（中心对齐、边界策略）；与训练DC完全一致。
  - 代码锚点：`ops/degradation.py:197`（统一退化入口）、`:241`（SR观测算子）、`:280`（Crop观测算子）。
  - DC损失与值域：在原值域计算`||H(ŷ)−y||`；与z-score域输出的转换约定；引用`ops/loss.py`。
- **模型模块**：
  - 空间主干：Swin-UNet（层次窗口注意力）、FNO瓶颈（频域耦合，何时启用）、UNet/FNO/Hybrid对比要点。
  - 时间建模：Temporal Transformer（因果掩码、层数/头数配置）。
  - 预测头：AR与NAR（并行多步，误差累积抑制与延迟优势）。
- **训练与配置**：
  - Hydra配置分层说明（数据/模型/训练/损失）；示例配置清单（如`configs/ar_training_config_debug_temporal.yaml`）。
  - 优化与调度：AdamW、Cosine+warmup、AMP/DDP、梯度裁剪与采样策略（SR/裁剪课程）。
- **评测与资源**：
  - 指标：Rel-L2、MAE、PSNR、SSIM、频域低/中/高fRMSE、边界带bRMSE、cRMSE、`||H(ŷ)−y||`。
  - 统计报告：≥3种子均值±标准差；paired t-test与Cohen’s d。
  - 资源表：Params(M)、FLOPs(G@256²)、显存峰值(GB)、推理延迟(ms)。
- **复现与集成**：
  - YAML快照与环境指纹；`runs/<exp>/config_merged.yaml`约定。
  - PDEBench数据卡与切分；z-score标准化与固定splits。
- **问题与迭代记录**：
  - 训练卡顿与AMP精度问题的修复要点（指向现有故障分析文档）。
  - 已知边界案例与失败类型（边界层溢出/相位漂移/振铃/能量偏差）。
- **相关工作与引用**：
  - Senseiver（稀疏注意力集合到场）：引入与差异对比；参考：Nature Machine Intelligence 2023，GitHub。
  - PDEBench基准：NeurIPS 2022主文与补充、GitHub/DaRUS。

## 引用与格式
- 采用IEEE数字引用；在文末给出完整条目（作者、标题、 venue、年份、DOI/URL）。
- 外部主引用：
  - Santos et al., Senseiver（Nat Mach Intell 2023，URL: https://www.nature.com/articles/s42256-023-00746-x）
  - Senseiver GitHub: https://github.com/OrchardLANL/Senseiver
  - PDEBench：arXiv:2210.07182，NeurIPS论文PDF与OpenReview，GitHub: https://github.com/pdebench/PDEBench

## 交付方式与校验
- 新增文档文件并提交到`.trae/documents/`；不修改现有代码。
- 自检清单：
  - 与黄金法则一致（H/DC复用、统一接口、可复现）。
  - 关键路径均有代码锚点与配置示例。
  - 引用格式正确、可访问。

## 后续维护
- 随训练/评测脚本演进补充“变更日志”与“已知问题”；按Release节点更新资源与指标表。

## 下一步
- 获得确认后，我将生成该文档并加入仓库相应目录，同时补充Senseiver与PDEBench引用、代码锚点与配置示例。