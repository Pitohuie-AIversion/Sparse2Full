# 稀疏观测驱动的时空流场重建方法研究

## 论文组织结构

本文件夹用于存储和组织论文相关的所有材料，按照学术写作的标准结构进行分类。

## 目录结构

```
thesis_paper/
├── manuscript/          # 论文正文手稿
│   ├── introduction.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── conclusion.tex
│   └── main.tex
├── figures/           # 论文图表
│   ├── architecture/
│   ├── results/
│   ├── comparison/
│   └── visualization/
├── tables/            # 数据表格
│   ├── performance/
│   ├── ablation/
│   └── statistics/
├── references/        # 参考文献
│   ├── bibliography.bib
│   └── related_work/
├── supplementary/     # 补充材料
│   ├── appendix/
│   ├── code/
│   └── data/
├── notes/            # 研究笔记
│   ├── ideas.md
│   ├── todo.md
│   └── meeting_notes/
└── drafts/           # 草稿文件
    ├── outline.md
    └── fragments/
```

## 使用说明

### 写作规范
- 使用LaTeX进行论文写作
- 图表需要高分辨率，建议矢量图格式
- 表格使用专业的表格格式
- 参考文献使用BibTeX管理

### 文件命名规范
- 使用小写字母和下划线
- 描述性文件名，如 `sparse_observation_method.tex`
- 版本控制使用日期后缀，如 `method_2024_01_15.tex`

### 版本管理
- 重要的里程碑版本需要备份
- 使用Git进行版本控制
- 定期提交有意义的更新

## 当前状态

- [ ] 完成方法论部分
- [ ] 实验结果整理
- [ ] 图表制作
- [ ] 参考文献整理
- [ ] 补充材料准备

## 相关资源

- 实验数据：`../paper_package/`
- 代码实现：`../models/`, `../ops/`
- 配置文件：`../configs/`
- 实验结果：`../runs/`

## 训练入口与配置管理（空间模型，空间-only）

- 训练入口：`tools/training/train_real_data_ar.py`
- 默认配置：`configs/train/ar_training_config_debug.yaml`（请设置 `ar.enabled: false`）
- 使用约定：每次复制该 YAML 并重命名后修改，新文件放入 `thesis_paper/configs/`，再用于本次实验；

示例：
```bash
cp configs/train/ar_training_config_debug.yaml thesis_paper/configs/spatial_training_config_srx2_20251127.yaml
# 编辑新配置以调整数据/模型/训练与评测参数
python tools/training/train_real_data_ar.py --config thesis_paper/configs/spatial_training_config_srx2_20251127.yaml --seed 123
```

命名建议：`spatial_training_config_<task>_<date>_<note>.yaml`

口径提醒：保持 `datasets/` 与 `ops/degradation.py` 的 H/DC 一致性；评测指标与表格生成对齐论文 6.2.x 章节。

## 横向对比设置与模型清单

**模型清单**：UNet、UNet++、SegFormer/UNetFormer、FNO2D、Hybrid(SwinUNet+FNO)、Sparse2Full

**统一训练设置**：
- 优化器：AdamW（`lr=1e-3`，`wd=1e-4`）、Cosine+1000 warmup、AMP、`grad_clip=1.0`
- Epoch：DR2D 主任务 `E=100`，RDB 主任务 `E=80`；`early_stopping.patience=20`
- Batch：`batch_size=16@128×128`；不足时 `gradient_accumulation=2`
- 随机性：≥5 种子；确定性模式；Hydra YAML 管理
- 观测一致性：训练 `DC` 与数据观测 `H` 完全一致（核/σ/插值/对齐/边界）

**评测与资源口径**：
- 指标：Rel-L2、MAE、PSNR、SSIM、fRMSE-low/mid/high、`||H(ŷ)−y||`；`均值±标准差（n=5）` + paired t-test + Cohen’s d
- 资源：Params(M)、FLOPs(G@256²)、显存峰值(GB)、推理延迟(ms)；硬件/AMP/Batch注明
- 统计与制表：`tools/summarize_runs.py`、`tools/generate_latex_tables.py`、`paper_package/figs/`


## 联系信息

如有问题或建议，请联系研究团队。

---
*最后更新：2024年*
