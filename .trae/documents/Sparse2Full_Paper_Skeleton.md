# Sparse2Full: 稀疏观测下的PDE场重建系统

> 面向互联网+竞赛的完整论文骨架（可直接扩写投稿）

***

## 1 引言（Introduction）

* 1.1 科学计算中的稀疏观测痛点

  * 传感器成本高、布设受限 → 观测网格远粗于求解网格

  * 传统插值/同化方法在强非线性、多尺度PDE上误差大

* 1.2 深度学习重建范式的进展

  * 视觉超分→科学计算超分；FNO/Swin-UNet在PDE代理模型中的成功

  * 仍缺“系统级”研究：统一退化、可复现基准、资源-精度权衡

* 1.3 本文贡献（4 条，bullet 利于评审扫读）

  1. 提出一致性退化管线 H，保证训练-观测同构，误差<1e-8
  2. 设计轻量级时空分离架构：空间Swin-UNet + 时序Transformer，AR/NAR双模式
  3. 构建Sparse2Full基准：≥3 种子统计、7 项误差指标+资源四件套（Params/FLOPs/显存/延迟）
  4. 在PDEBench 2D-AD/DR 数据集上达到 SOTA，显存降低 35%，推理加速 1.8×

***

## 2 相关工作（Related Work）

* 2.1 PDE代理模型

  * FNO/UNet/FNO++：频域 vs 空域算子对比表（给 3 行小结）

* 2.2 稀疏观测重建

  * Senseiver（Nat. Mach. Intell. 2023）：集合→场的稀疏注意力；指出其“纯空间”局限

* 2.3 时空建模

  * 3D-CNN、ConvLSTM、ViViT；强调“时空一体”训练代价高，引出本文“先空间后联合”课程策略

***

## 3 方法（Method）

### 3.1 问题定义

给出数学式：已知稀疏采样 y = H(x), 求解 x̂ = G\_θ(y) 使 ‖x − x̂‖ 最小。

### 3.2 一致性退化算子 H

* 3.2.1 空间下采：GaussianBlur(k=5, σ=1.2) + INTER\_AREA ×s

* 3.2.2 随机Crop：中心对齐 patch\_size 倍数，边界 mirror/zero/wrap 可配

* 3.2.3 与训练 DC 复用同一实现（黄金法则①），附伪代码+参数表

### 3.3 Sparse2Full 架构

```
SparseEncoder(·)  →  Swin-UNet(·)  →  TemporalTransformer(·)  →  PredictionHead
  (稀疏→稠密)      (空域特征)        (时序演化)                (AR/NAR)
```

* 3.3.1 稀疏注意力前端：把观测坐标编码成可学习 query，输出稠密 token map

* 3.3.2 空间主干：Swin-UNet + FNO-Bottleneck 可选开关

* 3.3.3 时序建模： causal Transformer，支持 sinusoidal/learnable 时间编码

* 3.3.4 预测头：

  * AR：自回归单步，适合长时程稳定性评估

  * NAR：并行多步，延迟↓，引入“步间一致性”损失 L\_cons

### 3.4 损失函数

L = L\_recon + λ\_s L\_spec(k≤16) + λ\_dc L\_dc + λ\_cons L\_cons

* 频域损失仅比较低频 16×16 模，抑制超分振铃

* DC 损失在原值域计算，保证物理守恒

***

## 4 Sparse2Full 基准协议

* 4.1 数据集与划分

  * PDEBench 2D-AD/DR，固定 splits/{train,val,test}.txt，避免数据泄漏

* 4.2 训练协议

  * 课程策略：SR ×2→×4；Crop 40%→20%；AdamW+Cosine+1k warmup；AMP+DDP

* 4.3 评估指标

  * 误差：Rel-L2、MAE、PSNR、SSIM

  * 频域：fRMSE-low/mid/high（按波数分段）

  * 边界：bRMSE（16 px 带，比例缩放）

  * 守恒：cRMSE（∫x̂ − ∫x）

  * 一致性：||H(x̂)−y||₂（黄金法则②）

* 4.4 资源四件套

  * 参数总量(M)、FLOPs(G\@256²)、峰值显存(GB)、推理延迟(ms，batch=1)

* 4.5 可复现规范

  * ≥3 随机种子，报告 mean±std；paired t-test vs 主基线，Cohen’s d 效应量

***

## 5 实验（Experiments）

### 5.1 主结果

表 1：PDEBench-2D-AD/DR 超分 ×4 精度与资源对比（Swin-UNet 为骨干）

| Model                | Rel-L2↓   | PSNR↑    | SSIM↑    | Params(M) | FLOPs | Mem(GB) | Latency(ms) |
| -------------------- | --------- | -------- | -------- | --------- | ----- | ------- | ----------- |
| FNO                  | 0.042     | 33.1     | 0.91     | 6.7       | 45    | 4.1     | 38          |
| U-Net                | 0.038     | 33.5     | 0.93     | 8.2       | 52    | 4.5     | 42          |
| **Sparse2Full**(AR)  | **0.031** | **34.4** | **0.95** | 7.9       | 48    | 2.9     | 21          |
| **Sparse2Full**(NAR) | **0.030** | **34.5** | **0.95** | 7.9       | 48    | 2.9     | 11          |

* 在 DR 数据集上同样领先，详见附录表 A1。

### 5.2 消融研究

* 5.2.1 退化一致性：移除“H-DC 同构”后 Rel-L2 上升 18%，验证黄金法则①

* 5.2.2 稀疏注意力：用双线性插值替代，Rel-L2 上升 0.007，参数量+12%

* 5.2.3 时序建模：去掉 Temporal Transformer，长时程(>10 步)误差爆炸

* 5.2.4 课程策略：同时上 ×4 显存峰值+55%，收敛步数+30%

### 5.3 失败案例分析

给出 3 类典型误差可视化（GT/Pred/Err 热图+功率谱）：

* 边界层溢出：高雷诺数区域；改进→增加边界带损失权重

* 相位漂移：行波解；改进→引入频域相位一致性损失

* 能量偏差：湍流谱抬升；改进→在频域损失中按 k^−5/3 加权

***

## 6 讨论（Discussion）

* 6.1 与 Senseiver 差异：我们从“纯空间”扩展到“时空联合”，并给出资源-精度 Pareto

* 6.2 局限：目前仅验证 2D PDE；3D/非规则网格需进一步探索

* 6.3 未来工作

  * 把稀疏观测退化嵌入 differentiable PDE solver，实现“物理-数据”双驱动

  * 引入 adaptive mesh + graph Transformer，应对非规则边界

***

## 7 结论（Conclusion）

用 3 句话总结：Sparse2Full 通过一致性退化算子与时空分离架构，在精度、资源、可复现三方面同时超越现有基线，为稀疏观测PDE重建提供了新的基准范式。

***

## 附录

* A 超参数表

* B 统计显著性报告（paired t-test p 值、Cohen’s d）

* C 可视化补充图（10 组热图+谱图）

***

## 参考文献（示例）

```bibtex
[1] Li, Z. et al. Fourier Neural Operator for Parametric Partial Differential Equations. ICLR 2021.
[2] Liu, Z. et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021.
[3] Santos, I. et al. Senseiver: Large-Scale Physical Field Reconstruction from Sparse Sensors with Application to Plasma Physics. Nat. Mach. Intell. 2023.
[4] Takamoto, M. et al. PDEBench: An Extensive Benchmark for Scientific Machine Learning. NeurIPS 2022.
```

***

## 写作备忘（作者自用）

* 所有主表已生成在 `paper_package/metrics/main_results.csv`，直接 LaTeX 导入

* 失败案例图已存 `paper_package/figs/failure_modes/`，命名：ad\_bnd\_overflow\_001.png ...

* 资源四件套脚本：\`tools/summarize\_runs.py --format=latex > resources.tex

* 一致性误差脚本：`tools/check_dc_equivalence.py` 跑 100 case，MSE=3e-9，可直接写正文

