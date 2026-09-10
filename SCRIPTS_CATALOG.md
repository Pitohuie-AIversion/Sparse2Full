# Sparse2Full 根目录脚本架构索引与职责清单 (Scripts Catalog)

> **导读与维护规范**：
> 本项目经历多轮实验迭代，根目录下沉淀了一批历史研究、指标分析与阶段性图表脚本。
> 为明确**工程主干 (Production Backbone)** 与 **研究脚本 (Research/Ad-hoc Scripts)** 的界限，特建立本清单。

---

## 一、核心工程主干 (Core Production Backbone)

| 脚本文件 | 模块职责 | 依赖环境 / 调用方式 |
| :--- | :--- | :--- |
| `train.py` | **主训练主控入口**：Hydra 配置加载、数据流编排、双卡/DDP 训练、模型保存 | `bash run_train_*.sh` 或 `python3 train.py [overrides]` |
| `eval.py` | **主评估入口**：物理场重构评估、误差分布统计、指标输出 | `python3 eval.py config=... checkpoint=...` |
| `train_temporal.py` | **时序自回归/NAR 主干训练入口**：多步预测与时空解耦模型训练 | `python3 train_temporal.py` |
| `setup.py` | **工程安装与依赖打包**：`pip install -e .` 本地可编辑包配置 | `pip install -e .` |
| `run_train_*.sh` | **标准 Launcher 启动脚本**：环境线程设置、GPU 分配、可移植路径 | `./run_train_swin_fluid_sr.sh` 等 |

---

## 二、历史研究与阶段性分析脚本分类 (Research & Ad-hoc Scripts)

以下脚本为论文各章节与消融实验期间编写的单次验证脚本，目前已被测试体系覆盖。

### 1. 自动回归与展开评测 (Rollout & AR Evaluation)
- `test_eval_ar*.py`（共 12 个版本，涵盖 `fno1~12`, `unet1~5`, `bicubic`）：针对特定模型在自回归序列上的误差展开步长扫描。
- `test_gen_rollout*.py`（1~4）：用于生成特定 PDE 时间步展开预测轨迹的临时验证脚本。
- `get_real_edsr_rollout.py` / `plot_rollout_mock.py`：针对 EDSR 模型的实测与 Mock 数据绘图脚本。

### 2. 消融实验与指标抽取 (Ablation & Metrics Extraction)
- `collect_ablation_results.py` / `get_ablation_metrics*.py`：从日志中正则匹配提取消融实验表格数据。
- `check_ablation_metrics.py` / `check_edsr_ablation.py`：校验不同采样策略（A0~A3）下的 MSE/PSNR 指标。
- `extract_metrics.py` / `collect_metrics.py`：聚合多个实验目录中的 `metrics.jsonl`。

### 3. 图表与论文修订工具 (Manuscript & Figure Utilities)
- `update_chapter4*.py`（共 6 个变体）：更新学位论文第四章各实验结果表格。
- `fix_table_4_7.py` / `update_table47.py`：专项修复论文 Table 4.7 的排版与数据格式。
- `combine_pngs*.py` / `combine_svgs*.py`：将多张物理场误差图横向拼接用于论文排版。
- `convert_pdf_to_svg.py` / `renumber_figures.py`：图表格式转换与重新编号辅助工具。

### 4. 数据一致性与语法快速检测 (Quick Lint & Diagnostics)
- `check_syntax.py` / `debug_config.py`：单次排查 Hydra 配置语法或模型初始化错误的快速探针。

---

## 三、后续结构演进建议
在后续代码库发版或归档阶段，可按上述分类建立子目录（如 `research_scripts/rollout/`、`research_scripts/paper_tables/`），并在根目录下保持最纯净的工程主干。
