# 智能数据集选择与配置报告

**生成时间**: 2025-10-17 02:44:54  
**随机种子**: 42  
**数据根目录**: E:/2D

## 📋 执行摘要

本报告展示了智能数据集选择器从 `E:/2D` 目录中自动选择的 3 种不同类型的PDE数据集，并为每个数据集生成了 6 个优化的训练配置。

## 🎯 选中的数据集

| 序号 | 数据集名称 | PDE类型 | 大小(MB) | 分辨率 | 时间步 | 通道数 | 复杂度 |
|------|------------|---------|----------|--------|--------|--------|--------|
| 1 | ns_incom_inhom_2d_512-0 | navier_stokes | 9447.5 | 512×512 | 1000 | 2 | very_high |
| 2 | 2D_diff-react_NA_NA | diffusion_reaction | 12630.0 | 未知 | 0 | 0 | high |
| 3 | 2D_rdb_NA_NA | shallow_water | 6319.1 | 未知 | 0 | 0 | 未知 |

## ⚙️ 生成的训练配置

总计生成了 **6** 个优化配置文件：

### 📊 ns_incom_inhom_2d_512-0

**sr_x2** (超分辨率重建 2x):
- 学习率: 1.00e-04
- 批处理大小: 1
- 训练轮数: 80
- 图像尺寸: 512×512
- 配置文件: `configs\auto_generated\ns_incom_inhom_2d_512_0_sr_x2_optimized.yaml`

**crop_20** (稀疏观测重建 20%):
- 学习率: 8.00e-05
- 批处理大小: 1
- 训练轮数: 60
- 图像尺寸: 512×512
- 配置文件: `configs\auto_generated\ns_incom_inhom_2d_512_0_crop_20_optimized.yaml`

### 📊 2D_diff-react_NA_NA

**sr_x2** (超分辨率重建 2x):
- 学习率: 5.00e-04
- 批处理大小: 2
- 训练轮数: 80
- 图像尺寸: 128×128
- 配置文件: `configs\auto_generated\2d_diff_react_na_na_sr_x2_optimized.yaml`

**crop_40** (稀疏观测重建 40%):
- 学习率: 4.00e-04
- 批处理大小: 2
- 训练轮数: 50
- 图像尺寸: 128×128
- 配置文件: `configs\auto_generated\2d_diff_react_na_na_crop_40_optimized.yaml`

### 📊 2D_rdb_NA_NA

**sr_x4** (超分辨率重建 4x):
- 学习率: 1.00e-03
- 批处理大小: 4
- 训练轮数: 100
- 图像尺寸: 128×128
- 配置文件: `configs\auto_generated\2d_rdb_na_na_sr_x4_optimized.yaml`

**crop_20** (稀疏观测重建 20%):
- 学习率: 8.00e-04
- 批处理大小: 4
- 训练轮数: 60
- 图像尺寸: 128×128
- 配置文件: `configs\auto_generated\2d_rdb_na_na_crop_20_optimized.yaml`

## 🔬 PDE类型分析

### SHALLOW_WATER

**描述**: 未知
**复杂度**: 未知
**推荐任务**: 
**数据集数量**: 1

### DIFFUSION_REACTION

**描述**: 扩散反应方程 - 化学反应扩散
**复杂度**: high
**推荐任务**: sr_x2, crop_40
**数据集数量**: 1

### NAVIER_STOKES

**描述**: Navier-Stokes方程 - 不可压缩流体
**复杂度**: very_high
**推荐任务**: sr_x2, crop_20
**数据集数量**: 1

## 🧠 优化策略

### 数据集选择策略
1. **多样性优先**: 确保选择不同类型的PDE方程
2. **质量评分**: 基于数据大小、格式、分辨率等因素评分
3. **平衡选择**: 兼顾计算复杂度和训练效果

### 配置优化策略
1. **PDE特性适配**: 根据不同PDE类型的特性调整参数
2. **任务类型优化**: SR和Crop任务使用不同的优化策略
3. **资源平衡**: 根据图像尺寸动态调整批处理大小
4. **稳定配置**: 使用经过验证的稳定损失函数权重

## 🚀 使用方法

### 1. 执行批量训练
```bash
python batch_train_selected_datasets.py
```

### 2. 单独训练某个配置
```bash
python train.py --config-path configs/auto_generated --config-name [配置文件名]
```

### 3. 监控训练进度
训练日志将保存在各自的 `runs/` 目录下。

## 📈 预期结果

基于优化策略，预期各数据集的训练效果：
- **Darcy流**: 适合SR任务，预期Rel-L2 < 0.1
- **扩散反应**: 复杂度高，需要更多训练轮数
- **Navier-Stokes**: 最具挑战性，需要小学习率和长时间训练

## 📝 注意事项

1. 训练时间可能较长，建议在GPU环境下运行
2. 监控显存使用，必要时调整批处理大小
3. 可根据实际效果进一步调整超参数
4. 建议保存训练日志用于后续分析

---
*本报告由智能数据集选择器自动生成*
