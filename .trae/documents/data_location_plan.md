# 论文第四章实验数据定位与梳理计划

## 目标 (Objective)
在进行第四章的配图和表格更新之前，首先需要系统性地盘点和定位本项目中庞大的实验结果目录（`runs*` 系列），以确保后续所有的图表绘制和数据提取都能对应到正确、最新的真实实验结果。

## 核心任务 (Core Tasks)
本次计划的核心是“**找准数据、理清映射**”，即将论文第四章提到的各个实验（如空间重建、时空演化、消融实验、视野受限等）与磁盘上具体的 `runs` 目录进行一一对应。

---

## 详细实施步骤 (Implementation Steps)

### Step 1: 扫描并汇总主要运行目录 (Scan Run Directories)
- **目标**：列出所有相关的 `runs*` 顶级目录及其包含的任务类型。
- **操作**：
  - 遍历 `` 下的所有 `runs*` 目录。
  - 已知存在的目录包括：`runs/`（主要包含 SWE 数据）、`runs_drd/`（主要包含 DR2D 空间重建数据）、`runs_drd_paper/`（主要包含 DR2D 时空序列数据）、`runs_3loss_ablation_unet/`（消融实验数据）等。

### Step 2: 定位“空间重建性能”实验数据 (Locate Spatial Reconstruction Data)
- **目标**：找到表 4-2、表 4-3 及图 4-1、图 4-8 所需的数据（EDSR, UNet, FNO, NAFNet 等）。
- **操作**：
  - 查找 `runs/AR-SW-10M-*` 和 `runs_drd/AR-DR2D-*` 目录。
  - 提取关键文件路径：`test_results.json` (用于 PSNR, Rel-L2 等指标)、`model_resources.json` (用于 Params, FLOPs, Latency) 和 `tensorboard/` 目录 (用于收敛曲线)。

### Step 3: 定位“视野受限下的空间重建”数据 (Locate Crop/Inpainting Data)
- **目标**：找到表 4-4 不同 Crop Size 下（112到1）的重建数据。
- **操作**：
  - 查找名称包含 `Crop` 或 `Inpainting` 的目录（例如 `runs/AR-DR2D-Crop-Inpainting-*` 或 `UNet_Crop_Scan`）。
  - 记录各个 Crop Size 下对应的 `test_results.json` 路径。

### Step 4: 定位“时空演化性能”实验数据 (Locate Spatiotemporal Rollout Data)
- **目标**：找到表 4-5 及图 4-2 所需的长时预测（Rollout）数据。
- **操作**：
  - 查找名称包含 `Sequential`、`Temporal`、`Seq2Seq` 等关键词的目录（如 `runs_drd_paper/AR-DR2D-Sequential-EDSR-VideoSwin-*`）。
  - 验证这些目录中是否存在多步预测的评估日志或 `test_results.json`。

### Step 5: 定位“消融实验与可视化”数据 (Locate Ablation & Visualization Data)
- **目标**：找到图 4-3（视觉对比）、图 4-5（失败案例）、图 4-6（消融曲线）和图 4-7（分阶段训练曲线）的数据。
- **操作**：
  - 消融实验：定位 `runs_3loss_ablation_unet/A0_Baseline` 和 `A3_Full`。
  - 分阶段训练：定位 `runs_drd_paper/AR-DR2D-Stage2-*` 和 `Stage3-*`。
  - 视觉图与失败案例：扫描 `test_visualizations/visualizations/` 下的 `*obs_gt_pred_error*.png` 及 `*error_analysis.png`。

### Step 6: 输出数据映射报告 (Generate Mapping Report)
- **目标**：将上述找到的所有路径整理成一份 Markdown 映射清单。
- **操作**：在终端打印或生成一个中间文档，明确标出“论文章节/图表 -> 对应的磁盘绝对路径”。

---
请您确认此数据定位计划。确认后，我将立即执行扫描命令并为您输出完整的数据路径映射报告。