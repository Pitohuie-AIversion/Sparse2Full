## 写作目标
- 基于最新文献与现有代码实现，系统完善论文草稿，补齐方法、评测协议、实验与显著性、资源成本与可复现材料。
- 保证“黄金法则”：观测算子 H 与训练 DC 复用同一实现与配置；同一 YAML+种子可复现；统一接口与可比性。

## 资料来源
- 互联网：
  - ST-SwinMAE/Spatio-Temporal SwinUNet（时空Swin发展）[1]
  - FNO 应用于海洋环流与时变系统（验证频域建模）[2][3]
  - 非自回归/一阶段多步训练（SPF 框架）[4]
  - PDEBench 基准与评测协议[5]
- 代码空间：
  - 指标定义与实现：`ops/metrics.py:24-181,184-216,219-357,407-443`
  - 损失三件套与值域：`ops/losses.py:23-121,123-311,314-448`
  - 观测算子与一致性：`ops/degradation.py:197-367,219-239`
  - 模型架构：
    - Swin-UNet：`models/spatial/swin_unet.py:820-830,833-1018,951-956`
    - FNO2D：`models/spatial/fno2d.py:18-160,239-306`
    - Hybrid：`models/spatial/hybrid.py:16-121`
  - 训练与评测：`tools/eval.py:1120-1211`，`tools/enhanced_summarize.py:560-679,592-635`
  - 资源与性能：`runs/*/resource_summary.json`，`tools/benchmark_models.py:58-66`
  - 论文材料打包：`training_system/paper_package/paper_package.py:617-667`

## 章节改写与新增内容
- 引言（1）：
  - 强化动机与挑战：空间稀疏+时间演化；多尺度/长程依赖；频域-时域互补。
  - 引入近期进展：ST-SwinMAE 的层次化时空/3D扩展[1]；FNO 在真实物理系统中的成功[2][3]；NAR/一阶段多步的稳定性-内存优势[4]。
- 相关工作（2）：
  - 2.4 时空 Transformer 最新进展：Video Swin/CLSTM、ST-SwinUNet，定位本方法的差异与优势（并行多步+频域瓶颈）。
  - 2.5 FNO 在 PDE/物理中的应用：强调分辨率不变与高效特性，呼应我们 FNO2D 设计。
  - 2.6 基准与评测：采用 PDEBench 协议与指标集合[5]。
- 问题定义（3）：
  - 明确输入输出与掩码 M；统一接口 `forward(x[B,C,H,W])→y[B,C,H,W]`；时序任务 `Tin/Tout` 定义。
- 方法（4）：
  - 4.1 训练框架与配置：Hydra 管理；课程学习阶段；AMP/DDP；四层回退模型创建（`train_real_data_ar.py:1570-1616`）。
  - 4.2 空间编码与稀疏注意力集成：Swin-UNet+稀疏编码器引用；双线性+3×3 解码；避免棋盘格。
  - 4.3 FNO2D 瓶颈：modes1/2、width、n_layers；频谱卷积与坐标拼接；接入点（`swin_unet.py:951-956`）。
  - 4.4 时间模块与 NAR 并行头：Transformer（num_heads/num_layers）；并行生成 Tout 帧；与 SPF 思想呼应。
  - 4.5 损失与值域（三件套）：`L = L_rec + λ_s L_spec + λ_dc L_dc`；输出在 z-score 域，DC/频域在原值域；低频 kx=ky≤16。
  - 4.6 观测算子 H 与一致性：SR/Crop 统一入口；`MSE(H(GT), y) < 1e-8` 验收。
- 评测协议（5）：
  - 指标集：Rel-L2/MAE/PSNR/SSIM/fRMSE-low/mid/high/bRMSE/DC Error；边界带 16px；频段划分与 rFFT 掩码。
  - 统计显著性：paired t-test、Cohen’s d、改进百分比；≥3 种子报告均值±标准差；资源四项 Params/FLOPs/显存/延迟。
- 实验（6）：
  - 主对比：UNet、Swin-UNet、FNO、Hybrid、Sparse2Full（相同数据与损失）。
  - 消融：
    - 去掉 FNO 瓶颈；改为 AR 逐步；替换时间模块（Transformer↔LSTM）。
    - 频域损失/数据一致性权重扫描（λ_s/λ_dc）。
  - 课程学习：`Tout=1→3→5` 分阶段训练；报告每阶段资源与指标变化。
  - 种子与稳定性：≥3（建议5）种子，方差≤1e-4。
- 结果与材料（7）：
  - 表格：主表/显著性/频域/资源（LaTeX/Markdown，`tools/enhanced_summarize.py`）。
  - 图表：GT/Pred/Err 热图（统一色标）、功率谱（log）、边界带放大、训练曲线（Rel-L2 与 ||H(ŷ)−y|| 同步下降）。
  - 失败案例：边界层溢出/相位漂移/振铃/能量偏差类型与改进建议。
  - PaperPackage：配置快照、权重（LFS）、指标、图表与脚本。

## 实验与评测补充计划（可执行清单）
- 数据与 H/DC：固定 `splits/{train,val,test}.txt`；逐通道 z-score；SR/Crop 参数与 `ops/degradation.py` 对齐并跑一致性检查。
- 训练：
  - 课程学习 3 阶段（Tout=1/3/5），统一优化器 AdamW(lr=1e-3, wd=1e-4)、Cosine+1k warmup、AMP、梯度裁剪 1.0。
  - 资源统计：记录 Params/FLOPs/显存峰值/延迟（`utils/resource_monitor.py:45`）。
- 评测：`tools/eval.py` 输出逐样本与聚合；`tools/enhanced_summarize.py` 生成主表与显著性报告。
- 可视化：统一色标与频谱图；失败案例归类。
- 复现：训练开始写入 `runs/<exp>/config_merged.yaml`；生成 `paper_package/` 完整材料。

## 交付物与更新点
- 文本更新：在草稿的“3 问题定义”、“4 方法”、“5 评测协议”、“6 结果与显著性”插入上述技术内容与代码锚点（`file_path:line_number`）。
- 图表与表格：主表（均值±标准差）、资源表、显著性（t-test+d）、频域、训练曲线与失败案例图。
- 材料包：`paper_package/` 完整可复现包。

## 执行步骤
1) 收集并校验近期 runs/* 指标与资源日志，合并多种子为主表与显著性报告。
2) 按章节将方法/评测/结果的技术细节与代码锚点补齐到草稿 md。
3) 生成图表与 LaTeX 表格，统一数值格式与单位。
4) 运行 H 一致性与频域损失单测，记录无偏差链路。
5) 打包 `paper_package/`，核对盲审模式导出。

## 参考链接
[1] ST-SwinMAE / ST-SwinUNet: https://arxiv.org/html/2405.02512v1  
[2] FNO in Ocean Prediction: https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1383997/full  
[3] Sparsified Time-dependent FNO: https://pubs.aip.org/aip/pop/article/31/12/123902/3323878  
[4] SPF (一阶段多步)：https://ui.adsabs.harvard.edu/abs/2025CMAME.44718332Z/abstract  
[5] PDEBench: https://arxiv.org/abs/2210.07182