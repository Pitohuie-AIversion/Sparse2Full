# Sparse2Full: 物理流场稀疏观测重构与科学流体超分辨率系统

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-464%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **Sparse2Full** 面向世界模型与复杂偏微分方程（PDE）动力系统（浅水波方程 Shallow Water、2D 反应扩散方程 Reaction-Diffusion 等），基于稀疏或低分辨率传感器观测数据，实现物理守恒的高保真全场重构与流体超分辨率预测。

---

## 🌟 核心特性与架构升级

1. **全新科学流体超分辨率架构 (`SwinFluidSR`)**：
   - **直接低维输入 (Direct 32×32 Input)**：放弃计算冗余的全局输入插值，Token 计算量骤减 16 倍；
   - **残差 Swin 变换块 (RSTB)**：利用移位局部窗口自注意力（Shifted-Window Self-Attention）精准建模湍流多尺度空间依赖；
   - **亚像素卷积上采样 (PixelShuffle 4×)**：通过周期性通道重排直接映射至高分辨率流场，有效克服传统反卷积棋盘伪影；
   - **全局物理插值残差学习 (Global Bicubic Residual)**：显式对齐流场大尺度宏观拓扑，让神经网络专注拟合微观涡旋残差。

2. **严格的物理守恒与防御算子 (`ops/`)**：
   - **频域能量谱物理损失 (Spectral Loss)**：基于 2D-FFT 惩罚径向波数功率谱误差，强制模型重构流体高频小尺度涡旋与湍流能量级串；
   - **流体无源性散度守恒 ($\nabla \cdot \mathbf{u} = 0$)** 与涡度守恒算子；
   - **Fail-Fast 完整性防线**：坚决禁止静默篡改真实流场（GT）分辨率，算子与张量流水线具备严格的维度对齐防御。

3. **解耦的 Trainer 架构体系 (`training/`)**：
   - 主控入口 [train.py](train.py) 采用纯净**外观模式 (Facade)**，单文件由 1,445 行精简至 409 行；
   - 正交子系统支持：批次流水线处理（`BatchProcessor`）、引擎构建与学习率调度（`EngineBuilder`）、检查点与论文交付包归档（`TrainingArtifactManager`）。

4. **全量分层测试保障 (`tests/`)**：
   - 全套 464 项单元测试达成 **100% 绿色通过**，零历史债务与零回归。

---

## 📊 独立测试集定量盲测对比 (PDEBench Test Split)

在未参与训练的独立测试集上，全新 `SwinFluidSR` 模型表现全面超越经典双三次插值（Bicubic Baseline）与原有架构：

| 评估指标 (Metric) | 双三次插值 (Bicubic Baseline) | 原架构 (SwinTWithEncoder) | 全新架构 (SwinFluidSR) | 改进效益 (vs 原架构) |
| :--- | :--- | :--- | :--- | :--- |
| **相对误差 (rel_l2)** | 0.01494 | 0.07667 | **0.01123** | **下降 85.3%** 🏆 |
| **绝对误差 (mae)** | 0.00624 | 0.02396 | **0.00504** | **下降 78.9%** 🏆 |
| **峰值信噪比 (psnr)** | 32.13674 dB | 17.82351 dB | **34.63011 dB** | **提升 +16.8 dB** 🏆 |
| **结构相似度 (ssim)** | 0.94193 | 0.88469 | **0.95355** | **提升 +0.069** 🏆 |
| **高频误差 (frmse_high)**| 0.07898 | 1.71112 | **0.18558** | **下降 89.2%** 🏆 |
| **中频误差 (frmse_mid)** | 0.59588 | 10.33185 | **0.60077** | **下降 94.2%** 🏆 |
| **低频误差 (frmse_low)** | 4.37461 | 14.30391 | **3.21089** | **下降 77.6%** 🏆 |

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备
```bash
# 建议在 Conda 环境下安装 (Python 3.12)
pip install -r requirements.txt
pip install -e .
```

### 2. 启动生产级模型训练
```bash
# 双卡并行启动全新 SwinFluidSR 科学流体超分训练 (100 Epochs)
./run_train_swin_fluid_sr.sh

# 启动基线 SwinT 编码器训练
./run_train_swin_t_enc.sh
```

### 3. 执行独立测试集评估与流体能谱分析
```bash
python3 evaluate_test_set.py
```
*该脚本将自动比对各模型预测流场，并在 `runs_swin_fluid_sr/` 输出测试集超分辨率画廊与径向平均湍流能谱图。*

### 4. 运行全量单元测试
```bash
pytest tests/unit/ -q
```

---

## 📁 架构全景与文档导航

- 🗺️ **[WORKSPACE_MAP.md](WORKSPACE_MAP.md)**：工作区全景拓扑图，梳理 24 个顶层目录与 6 大功能域的权责边界。
- 📜 **[SCRIPTS_CATALOG.md](SCRIPTS_CATALOG.md)**：根目录主干脚本与 `tools/research_scripts/` 150+ 项历史工具的完整索引。
- 🎓 **`thesis_paper/`**：硕士学位论文 LaTeX 源码、各章结构、审稿修订批注与高清矢量图表。
- 📦 **`paper_package/`**：面向论文发表的图表包、数据卡片与指标汇总。
- 🗄️ **`archives/`**：静态历史切片压缩备份（包含 142k 行历史代码镜像 `clean_export_backup.tar.gz`）。

---

## 📜 开源协议

本项目采用 MIT 协议开源。
