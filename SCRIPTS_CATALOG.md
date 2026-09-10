# Sparse2Full 根目录脚本架构索引与职责清单 (Scripts Catalog)

> **导读与维护规范**：
> 本项目经历多轮实验迭代，为明确**工程主干 (Production Backbone)** 与 **研究脚本 (Research/Ad-hoc Scripts)** 的界限，所有历史单次研究、消融验证与论文排版脚本均已归入 `tools/research_scripts/`，历史切片代码已压缩归档于 `archives/`。

---

## 一、核心工程主干 (Core Production Backbone - 保留在根目录)

| 脚本文件 | 模块职责 | 依赖环境 / 调用方式 |
| :--- | :--- | :--- |
| `train.py` | **主训练主控入口 (Facade)**：Hydra 配置加载、数据流编排、双卡/DDP 训练、模型保存 | `bash run_train_*.sh` 或 `python3 train.py [overrides]` |
| `eval.py` | **主评估入口**：物理场重构评估、误差分布统计、指标输出 | `python3 eval.py config=... checkpoint=...` |
| `evaluate_test_set.py` | **测试集专项评测入口**：自动化扫描验证全集并输出标准指标表格 | `python3 evaluate_test_set.py` |
| `train_temporal.py` | **时序自回归/NAR 主干训练入口**：多步预测与时空解耦模型训练 | `python3 train_temporal.py` |
| `setup.py` | **工程安装与依赖打包**：`pip install -e .` 本地可编辑包配置 | `pip install -e .` |
| `run_train_*.sh` | **标准 Launcher 启动脚本**：环境线程设置、GPU 分配、可移植路径 | `./run_train_swin_fluid_sr.sh` 等 |

---

## 二、历史研究与论文辅助脚本集 (`tools/research_scripts/`)

全套 150+ 项历史实验与消融排版工具均按生命周期归档于以下目录：

### 1. 自动回归与展开评测 (`tools/research_scripts/rollout_and_ar/`)
- `test_eval_ar*.py`（共 12 个版本，涵盖 `fno1~12`, `unet1~5`, `bicubic`）：针对特定模型在自回归序列上的误差展开步长扫描。
- `test_gen_rollout*.py`（1~4）：用于生成特定 PDE 时间步展开预测轨迹的临时验证脚本。
- `get_real_edsr_rollout.py` / `plot_rollout_mock.py` / `generate_rollout_mock.py`：针对 EDSR 模型的实测与 Mock 数据绘图脚本。

### 2. 消融实验与指标抽取 (`tools/research_scripts/ablation_and_metrics/`)
- `collect_ablation_results.py` / `get_ablation_metrics*.py`：从日志中正则匹配提取消融实验表格数据。
- `check_ablation_metrics.py` / `check_edsr_ablation.py`：校验不同采样策略（A0~A3）下的 MSE/PSNR 指标。
- `extract_metrics.py` / `collect_metrics.py`：聚合多个实验目录中的 `metrics.jsonl`。
- `find_*.py`（共 16 个）：用于快速检索特定实验指标、超参组合及曲线数据。

### 3. 图表与论文草稿工具 (`tools/research_scripts/paper_and_manuscript/`)
- `update_chapter4*.py`（共 6 个变体）：更新学位论文第四章各实验结果表格。
- `fix_table_4_7.py` / `update_table47.py`：专项修复论文 Table 4.7 的排版与数据格式。
- `combine_pngs*.py` / `combine_svgs*.py`：将多张物理场误差图横向拼接用于论文排版。
- `convert_pdf_to_svg.py` / `renumber_figures.py`：图表格式转换与重新编号辅助工具。
- 历史排版中间稿与附录（`test_*.docx`, `test_*.md`, `test_*.html`, `test_*.tex` 等）。

### 4. 调试排查与快速探针 (`tools/research_scripts/diagnostics/`)
- `check_syntax.py` / `debug_config.py`：单次排查 Hydra 配置语法或模型初始化错误的快速探针。
- `test_omegaconf*.py` / `test_ddp_simple.py`：配置合并与多卡环境快速连通性测试。
- `revert_unet*.py` / `final_revert_unet.py`：历史模型权重结构逆向提取探针。

### 5. 历史运行与单次消融启动脚本 (`tools/research_scripts/legacy_launchers/`)
- `run_missing_RecDC*.sh`（共 7 个变体）：过往补跑 RecDC 物理约束消融实验的启动脚本。
- `repro_all_edsr.sh` / `repro_all_unet.sh` / `run_repro_edsr.sh`：过往复现 UNet 与 EDSR 对比基线的批量脚本。
- `run_missing_A0.sh` / `run_A2_final.sh` / `run_edsr_a2.sh` / `run_edsr_a3.sh`：针对特定稀疏掩码策略的实验启动脚本。
- `run_crop_unet_quick.sh` / `wait_bicubic_drd.sh` / `wait_rbf_swe.sh`：历史快速干跑与等待队列脚本。

---

## 三、静态大文件归档 (`archives/`)
- `archives/clean_export_backup.tar.gz`：包含 610 个历史副本文件（142k 行代码）的完整 gzip 备份。从 Git 跟踪树中移出，消除了 45% 的全局冗余符号与死代码索引。
