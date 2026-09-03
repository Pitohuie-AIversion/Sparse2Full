# 实验 A：跨方程泛化性测试 (Generalization Test) 实施计划

## 1. 目标与背景
回应“模型是否只记住了 SWE/DRD 的特征？”的质疑。通过在 2D Navier-Stokes (Turbulence) 数据集上测试在 Shallow Water Equation (SWE) 数据集上预训练的 EDSR 模型，验证“统一算子”在零样本 (Zero-shot) 和少样本 (Few-shot) 下的泛化能力，证明模型学到的是物理观测规律而非单纯的数据分布。

## 2. 准备工作：数据集与数据加载模块 (DataModule)
- **数据集确认**：项目中已存在 2D Navier-Stokes 数据集，路径为 `data/2D/NS_incom/ns_incom_inhom_2d_512-0.h5`。该数据形状为 `(4, 1000, 512, 512, 2)`，包含 4 个样本，每个样本 1000 个时间步。
- **数据加载模块适配**：
  - 新建 `datasets/navier_stokes_dataset.py`，实现 `NavierStokesDataset` 和 `NavierStokesDataModule`。
  - 在 `NavierStokesDataset` 中，正确读取 `velocity` 数据集。针对 EDSR (SWE 预训练权重只有 1 个通道输入)，提取单通道（例如 u 速度场，即 channel 0），并可选择将其中心裁剪或降采样至 `128x128` (以对齐 SWE 训练时的分辨率设置，保持测试效率)。
  - 修改 `tools/training/train_real_data_ar.py` 第 1108 行附近，引入 `dataset_name == 'navier_stokes'` 的判断分支，实例化 `NavierStokesDataModule`。

## 3. 实验配置设计
创建两个配置文件，统一使用 `runs/AR-SW-10M-edsr/best.ckpt` 作为预训练权重。
### 3.1 Zero-shot 实验配置 (`thesis_paper/configs/expA_zeroshot_ns.yaml`)
- **数据**：`dataset_name: navier_stokes`，加载 Navier-Stokes 测试集。
- **任务**：SR 任务（例如 Scale=4，与 SWE 预训练保持一致），设置 `observation.mode: SR`。
- **模型**：EDSR 架构，`in_channels: 1, out_channels: 1`。
- **加载权重**：`pretrained_path: runs/AR-SW-10M-edsr/best.ckpt`。
- **评估设置**：仅进行测试，不更新梯度。

### 3.2 Few-shot 实验配置 (`thesis_paper/configs/expA_fewshot_ns.yaml`)
- **数据**：同上，但取极少量的 Navier-Stokes 样本作为训练集（例如取 10% 的时间步数据进行 few-shot 训练）。
- **优化器**：设置较小的学习率（如 `lr: 1e-4` 或 `5e-5`）。
- **训练设置**：快速微调（例如 50-100 个 Epoch），启用梯度更新。
- **加载权重**：`pretrained_path: runs/AR-SW-10M-edsr/best.ckpt`。

## 4. 执行实验
1. **运行 Zero-shot 测试**：
   使用 `train_real_data_ar.py` 的仅测试模式（如果有）或直接加载 `expA_zeroshot_ns.yaml` 在 Navier-Stokes 上进行前向推理，记录 `Rel-L2` 和 $H_{err}$ (`||H(ŷ)-y||`)。
2. **运行 Few-shot 微调**：
   使用 `train_real_data_ar.py` 加载 `expA_fewshot_ns.yaml` 进行微调训练，并在测试集上评估最终性能。

## 5. 结果分析与可视化
- **对比指标**：汇总 Zero-shot 与 Few-shot 的 `Rel-L2`、`PSNR`、`SSIM` 及 $H_{err}$ 指标。
- **论点支撑**：若 Zero-shot 性能下 $H_{err}$ 保持在极低水平（说明重建结果严格符合观测矩阵 H 的约束），且 Few-shot 能够快速收敛到较好的 Rel-L2 性能，则证明统一算子具有跨物理场的通用约束力。
- **可视化诊断**：生成代表性样本的 GT、Pred、Error 热图（包含 Zero-shot 和 Few-shot），统一色标并保存至 `paper_package/figs/expA_generalization/`。