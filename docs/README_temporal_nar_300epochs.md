# 时序NAR模型300轮训练指南

## 概述

本指南介绍如何使用增强版时序NAR模型进行300轮长期训练，包含完整的监控、可视化和管理功能。

## 文件结构

```
Sparse2Full/
├── configs/experiment/temporal_nar_300epochs.yaml    # 300轮训练配置
├── train_temporal_nar_300epochs.py                   # 增强版训练脚本
├── launch_temporal_nar_300epochs.py                  # Python启动器
├── launch_temporal_nar_300epochs.bat                 # Windows批处理启动器
└── README_temporal_nar_300epochs.md                  # 本文档
```

## 快速开始

### 方法1: 使用批处理脚本 (推荐)

```bash
# Windows用户直接双击或运行
launch_temporal_nar_300epochs.bat
```

### 方法2: 使用Python启动器

```bash
# 基本训练
python launch_temporal_nar_300epochs.py --action train

# 恢复训练
python launch_temporal_nar_300epochs.py --action train --resume

# 调试模式
python launch_temporal_nar_300epochs.py --action train --debug

# 自定义参数
python launch_temporal_nar_300epochs.py --action train --batch-size 8 --lr 5e-4
```

### 方法3: 直接运行训练脚本

```bash
python train_temporal_nar_300epochs.py
```

## 配置说明

### 主要改进

相比100轮版本，300轮配置包含以下优化：

1. **学习率调度器优化**
   - 使用 `CosineAnnealingWarmRestarts`
   - T_0=50, T_mult=2 (周期性重启)
   - 更长的warmup期 (2000步)

2. **早停策略调整**
   - patience增加到30轮
   - 更小的min_delta (1e-6)

3. **增强监控**
   - TensorBoard实时日志
   - 每20轮保存可视化
   - 性能和资源监控
   - 里程碑检查点保存

4. **数据增强**
   - 翻转、旋转、噪声增强
   - 提高模型泛化能力

### 关键参数

```yaml
train:
  max_epochs: 300                    # 训练轮数
  
scheduler:
  name: "CosineAnnealingWarmRestarts"
  T_0: 50                           # 初始周期
  T_mult: 2                         # 周期倍增
  eta_min: 1e-7                     # 最小学习率
  warmup_steps: 2000                # 预热步数

early_stopping:
  patience: 30                      # 早停耐心值
  
checkpoint:
  every_n_epochs: 10                # 检查点保存频率
  save_top_k: 5                     # 保存最佳模型数量
```

## 训练监控

### 实时监控

```bash
# 监控训练进度
python launch_temporal_nar_300epochs.py --action monitor --output-dir runs/temporal_nar_300epochs
```

### TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir runs/temporal_nar_300epochs/tensorboard
```

访问 http://localhost:6006 查看训练曲线和指标。

### 输出目录结构

```
runs/temporal_nar_300epochs/
├── TemporalNAR-DR2D-128-300epochs-s2025/
│   ├── checkpoints/                # 模型检查点
│   │   ├── best.pth               # 最佳模型
│   │   ├── latest.pth             # 最新模型
│   │   └── epoch_*.pth            # 定期保存
│   ├── tensorboard/               # TensorBoard日志
│   ├── visualizations/            # 可视化图表
│   │   ├── training_curves.png    # 训练曲线
│   │   ├── loss_breakdown.png     # 损失分解
│   │   └── learning_rate.png      # 学习率曲线
│   ├── config_snapshot.yaml       # 配置快照
│   ├── environment.json           # 环境信息
│   ├── training_history.json      # 训练历史
│   └── train.log                  # 训练日志
```

## 训练管理

### 停止训练

```bash
# 方法1: 使用启动器停止
python launch_temporal_nar_300epochs.py --action stop

# 方法2: Ctrl+C 优雅停止
# 方法3: 任务管理器强制终止
```

### 恢复训练

训练支持自动恢复，会从最新检查点继续：

```bash
python launch_temporal_nar_300epochs.py --action train --resume
```

### 检查点管理

- **自动保存**: 每10轮保存一次
- **最佳模型**: 验证损失最低时保存
- **里程碑**: 在50, 100, 150, 200, 250, 300轮保存
- **最新模型**: 每轮都更新

## 性能优化

### 系统要求

- **GPU**: 建议RTX 3080或更高 (≥10GB显存)
- **内存**: 建议32GB或更高
- **存储**: 至少50GB可用空间
- **CUDA**: 11.8或更高版本

### 优化建议

1. **批次大小调整**
   ```bash
   # 根据GPU显存调整
   python launch_temporal_nar_300epochs.py --action train --batch-size 8
   ```

2. **学习率调整**
   ```bash
   # 更保守的学习率
   python launch_temporal_nar_300epochs.py --action train --lr 5e-4
   ```

3. **数据加载优化**
   - 使用SSD存储数据
   - 增加num_workers (配置文件中)
   - 启用pin_memory

## 实验跟踪

### 关键指标

训练过程中监控以下指标：

- **损失指标**: AR Loss, NAR Loss, Total Loss
- **评估指标**: Rel-L2, MAE, PSNR, SSIM
- **系统指标**: GPU内存, 训练速度, 学习率

### 结果分析

训练完成后，查看以下文件进行结果分析：

1. `training_history.json` - 完整训练历史
2. `visualizations/` - 可视化图表
3. `tensorboard/` - TensorBoard日志
4. `train.log` - 详细训练日志

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python launch_temporal_nar_300epochs.py --action train --batch-size 2
   ```

2. **数据文件不存在**
   - 检查配置文件中的data_path
   - 确保数据文件路径正确

3. **训练中断**
   ```bash
   # 使用恢复模式
   python launch_temporal_nar_300epochs.py --action train --resume
   ```

4. **学习率过高导致发散**
   ```bash
   # 使用更小的学习率
   python launch_temporal_nar_300epochs.py --action train --lr 1e-4
   ```

### 调试模式

```bash
# 启用调试模式，获取更详细的信息
python launch_temporal_nar_300epochs.py --action train --debug
```

## 高级用法

### 自定义配置

可以通过Hydra覆盖任何配置参数：

```bash
python train_temporal_nar_300epochs.py \
    train.max_epochs=500 \
    train.optimizer.lr=5e-4 \
    data.batch_size=8 \
    experiment.name="CustomRun"
```

### 多GPU训练

配置文件支持分布式训练，需要修改：

```yaml
experiment:
  device: "cuda"
  distributed: true
  world_size: 2
```

### 实验对比

可以同时运行多个实验进行对比：

```bash
# 实验1: 标准配置
python train_temporal_nar_300epochs.py experiment.name="Standard"

# 实验2: 高学习率
python train_temporal_nar_300epochs.py experiment.name="HighLR" train.optimizer.lr=2e-3

# 实验3: 大批次
python train_temporal_nar_300epochs.py experiment.name="LargeBatch" data.batch_size=16
```

## 结果验证

### 性能基准

预期的训练结果基准：

- **收敛性**: 训练损失应在50轮内开始收敛
- **验证指标**: Rel-L2 < 0.1, PSNR > 25dB
- **训练时间**: 约12-24小时 (RTX 3080)
- **最终损失**: < 1e-3

### 质量检查

1. 检查训练曲线是否平滑下降
2. 验证损失是否与训练损失同步
3. 可视化结果是否合理
4. 模型是否过拟合

## 总结

300轮时序NAR训练提供了完整的长期训练解决方案，包含：

- ✅ 优化的学习率调度
- ✅ 增强的监控和可视化
- ✅ 自动检查点管理
- ✅ 便捷的启动和管理工具
- ✅ 完整的故障排除支持

通过本指南，您可以轻松启动和管理长期训练任务，获得更好的模型性能。