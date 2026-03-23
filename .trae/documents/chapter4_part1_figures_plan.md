# 论文第四章配图梳理与生成计划

## 目标 (Objective)
在完成第四章第一部分（图4-1至图4-4）基于真实数据的配图替换后，现在我们需要系统性地梳理和更新后续的配图（图4-5至图4-8），确保它们同样基于真实的实验数据生成。

## 待梳理配图 (Target Figures)
根据 `chapter4_results_verification.md`，后续包含以下四张核心图表：
1. **图 4-5：典型失败案例分析 (Failure Cases)**（对应文件：`fig4-8_failure_cases.png`）
2. **图 4-6：损失函数消融实验曲线 (Ablation Curves)**（对应文件：`fig4-6_ablation_curves.png`）
3. **图 4-7：序列化课程学习策略演进曲线 (Sequential Evolution)**（对应文件：`fig4-5_sequential_evolution.png`）
4. **图 4-8：资源-精度权衡 Pareto Frontier**（对应文件：`fig4-3_pareto_frontier.png`）

---

## 详细实施步骤 (Implementation Steps)

### Step 1: 失败案例分析图生成 (图 4-5)
- **目标**：展示强非线性边界区域的频谱泄漏或振铃伪影。
- **操作**：
  - 定位真实数据：从测试可视化目录（如 `AR-DR2D-EDSR` 或 `AR-DR2D-UNet`）中挑选一个具有明显边缘误差的样本（例如 `test_sample_1_error_analysis.png` 等）。
  - 更新脚本：编写 `tools/plot_fig4_5_failure_cases.py`，读取该样本的 GT、Pred 和 Error 图，并在图中用红框（或高亮）标注出伪影区域。
  - 覆盖保存：`thesis_paper/manuscript_5_chapter/images/fig4-8_failure_cases.png`。

### Step 2: 损失函数消融实验曲线 (图 4-6)
- **目标**：对比 MSE Only 与 Full Loss 的验证集 Rel-L2 曲线。
- **操作**：
  - 定位真实数据：找到消融实验的 Tensorboard 日志目录（如包含 `A0`、`A3` 等后缀的 runs）。
  - 更新脚本：编写 `tools/plot_fig4_6_ablation.py`，提取这两个运行的 `val_loss` 或 `val_rel_l2`。
  - 覆盖保存：`thesis_paper/manuscript_5_chapter/images/fig4-6_ablation_curves.png`。

### Step 3: 序列化课程学习演进曲线 (图 4-7)
- **目标**：展示 Stage 2（冻结）到 Stage 3（微调）时 Rel-L2 和 fRMSE-High 的变化。
- **操作**：
  - 定位真实数据：查找分阶段训练（Staged）的日志，特别是记录了不同频段误差的 JSON 或 Tensorboard 数据。
  - 更新脚本：编写 `tools/plot_fig4_7_sequential.py`，在图上用双Y轴展示全局误差和高频局部误差的同步下降，并标注阶段切换点。
  - 覆盖保存：`thesis_paper/manuscript_5_chapter/images/fig4-5_sequential_evolution.png`。

### Step 4: 资源-精度 Pareto Frontier (图 4-8)
- **目标**：绘制各模型的 FLOPs 与 Rel-L2 权衡散点图。
- **操作**：
  - 定位真实数据：从 `model_resources.json` 或现有的基准测试汇总报告中提取各架构的 Params/FLOPs 和 Rel-L2 数据。
  - 更新脚本：编写 `tools/plot_fig4_8_pareto.py` 绘制散点图，标记 Pareto 前沿线。
  - 覆盖保存：`thesis_paper/manuscript_5_chapter/images/fig4-3_pareto_frontier.png`。

### Step 5: 验证与文档更新
- **操作**：
  - 确认 `chapter4_results_verification.md` 中的所有图引用路径与生成的图片文件名对应无误。
  - 运行 `python tools/convert_thesis.py --format docx`。
