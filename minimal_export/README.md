# Real Data AR Training Project

这是一个专注于真实扩散-反应（Diffusion-Reaction）系统数据的自回归（Autoregressive, AR）训练项目。本项目旨在通过深度学习模型（如 Swin-UNet）捕捉复杂的时空动态，实现高精度的未来状态预测。

## 📦 项目结构

本项目经过精简优化，仅包含核心训练所需的组件：

```
minimal_export/
├── tools/training/           # 训练入口脚本
│   ├── train_real_data_ar.py            # 原始训练脚本
│   └── train_real_data_ar_refactored.py # 重构版训练脚本（推荐）
├── configs/                  # Hydra 配置文件系统
├── datasets/                 # 数据加载与预处理
├── models/                   # 模型定义 (SwinUNet, ARWrapper, Temporal Models)
├── ops/                      # 核心算子 (损失函数, 退化算子)
├── utils/                    # 辅助工具 (日志, 可视化, 检查点)
├── requirements.txt          # Python 依赖
└── setup.py                  # 项目安装配置
```

## 🚀 快速开始

### 1. 环境安装

首先，确保您的环境满足 `requirements.txt` 中的依赖。建议在虚拟环境中操作：

```bash
cd minimal_export
pip install -r requirements.txt
pip install -e .
```

### 2. 准备数据

项目默认使用 HDF5 格式的数据集。请确保数据文件路径正确，并在配置文件中指定。
默认配置指向：`data/real_diffusion_reaction.h5`

### 3. 运行训练

您可以使用原始脚本或重构后的脚本启动训练。

**使用重构版脚本（推荐）：**
该版本采用了模块化的三阶段训练策略（空间 -> 时间 -> 联合），结构更清晰。

```bash
# 使用默认配置运行
python tools/training/train_real_data_ar_refactored.py

# 指定配置文件
python tools/training/train_real_data_ar_refactored.py --config configs/train/train_real_data_ar.yaml

# 指定输出目录和设备
python tools/training/train_real_data_ar_refactored.py --output-dir runs/my_experiment --device cuda:0
```

**使用原始脚本：**

```bash
python tools/training/train_real_data_ar.py
```

## 🧠 核心功能

### 模型架构
*   **Spatial Model**: 基于 Swin Transformer 的 U-Net (`models/swin_unet.py`)，负责单帧的空间特征提取与重建。
*   **Temporal Model**: 负责序列间的时序演化 (`models/temporal/`)。
*   **AR Wrapper**: 自回归包装器 (`models/ar/wrapper.py`)，支持 Teacher Forcing 训练和 Rollout 推理。

### 训练策略
*   **三阶段训练**: 
    1.  **Spatial Pretraining**: 仅训练空间重建能力。
    2.  **Temporal Pretraining**: 冻结空间参数，训练时序预测能力。
    3.  **Joint Finetuning**: 端到端联合微调。
*   **物理一致性约束**: 集成了退化算子 (`ops/degradation.py`) 和数据一致性损失，确保预测结果符合物理规律。

### 配置管理
项目使用 **Hydra** 进行配置管理。您可以灵活地组合和覆盖配置：
*   `configs/data/`: 数据集参数
*   `configs/model/`: 模型架构参数
*   `configs/train/`: 训练超参数（学习率、Epochs、Loss权重等）

## 📊 输出与监控

训练过程中会生成以下产物：
*   **日志**: `training.log` 记录训练进度和指标。
*   **TensorBoard**: 包含 Loss 曲线和可视化图像。
*   **Checkpoints**: `.pth` 模型权重文件，支持断点续训 (`--resume`)。
*   **Config Snapshot**: `config_merged.yaml` 记录当前运行的完整配置快照。
