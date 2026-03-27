# 论文第四章配图梳理与生成计划

## 目标 (Objective)
为论文第四章基于表 4-3 绘制一张“效率–精度权衡图”（Latency vs Rel-L2 散点图），取代原本的四联图设计，以直观展示在同一参数预算（~1M）下，各模型的算力、推理时延与精度的 Trade-off。

## 待梳理数据来源与模型 (Data Sources)
数据来源于 `drd 1m` 实验扫描，目录为 `./drd_paper_1m/`。包含以下核心模型：
- **EDSR** (`AR-DR2D-edsr-SRx4-1M-100ep`)
- **ConvUNetLite** (`AR-DR2D-ConvUNetLite-SRx4-1M-100ep`)
- **UNet** (`AR-DR2D-UNet-SRx4-1M-100ep`)
- **StableFNO2d** (`AR-DR2D-stablefno2d-SRx4-1M-100ep`)
- **NAFNet** (`AR-DR2D-nafnet-SRx4-1M-100ep`)

---

## 详细实施步骤 (Implementation Steps)

### Step 1: 提取各模型的定量指标
- **目标**：从 `./drd_paper_1m/` 各模型目录下的日志文件（如 `metrics.json`、`test_results.json` 或 `model_info.txt`）中提取以下指标：
  - 推理时延 `Latency (ms)`
  - 相对误差 `Rel-L2`
  - 参数量 `Params (M)`
- **操作**：编写一个简单的数据抓取脚本，或手动提取这些关键指标作为绘图数据。

### Step 2: 绘制效率–精度权衡散点图 (Trade-off Scatter Plot)
- **目标**：使用 Python (`matplotlib`/`seaborn`) 绘制满足 IEEE Transactions 标准的散点图。
- **具体要求**：
  - 横轴：推理时延（Latency, ms）
  - 纵轴：误差（Rel-L2）
  - 散点大小：表示参数量（Params, M），或者使用固定大小并标注具体数值。
  - 散点颜色/形状：区分不同的模型。
  - 标注：为每个点添加文本标签（如 EDSR, ConvUNetLite 等）。
  - 格式：分辨率 ≥ 600 dpi，字号 ≥ 8 pt，保证打印清晰。
- **输出路径**：`thesis_paper/figures/edsr/efficiency_accuracy_tradeoff.png` （及 `.pdf` 格式以备高质量插入）。

### Step 3: 更新正文引用与图例 (Update Manuscript)
- **操作**：
  - 在文件 `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md` 的 L149-155 段落附近，插入新绘制的权衡图。
  - 编写规范的 caption（包含数据集、缩放因子、评价指标等信息）。
  - 确保交叉引用编号与正文一致。
