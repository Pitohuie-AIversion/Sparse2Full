# 高级使用指南与最佳实践

本指南提供了针对 `minimal_export` 库的进阶使用说明，包括特定任务的配置、调试技巧以及自定义扩展建议。

## 1. 任务特定配置

### 1.1 纯空间恢复 (Spatial Super-Resolution Only)
如果您仅需对单帧图像进行超分辨率重建，而不涉及时间序列预测，请参考 `configs/spatial_sr_only.yaml`。

**使用场景**:
- 图像增强前处理
- 静态流场重建
- 空间模型预训练

**运行命令**:
```bash
python tools/training/train_real_data_ar_refactored.py \
    --config configs/spatial_sr_only.yaml \
    --stage spatial
```

### 1.2 纯时间预测 (Temporal Prediction Only)
如果您已有高质量的空间数据，仅需训练时序演化模型（假设空间特征已冻结或不需要），请使用 `configs/train/ar_training_config_temporal_only.yaml`。

**运行命令**:
```bash
python tools/training/train_real_data_ar_refactored.py \
    --config configs/train/ar_training_config_temporal_only.yaml \
    --stage temporal
```

## 2. 调试与性能调优

### 2.1 快速调试模式
为了快速验证代码逻辑，建议使用极小的数据子集和简化的模型配置。

**调试配置 (`configs/minimal_debug.yaml`)**:
- `batch_size`: 2
- `num_workers`: 0 (单进程，便于断点调试)
- `epochs`: 2
- `model`: 轻量级配置

**运行命令**:
```bash
python tools/training/train_real_data_ar_refactored.py \
    --config configs/minimal_debug.yaml \
    --device cpu  # 使用 CPU 进行逻辑验证
```

### 2.2 内存优化
对于显存受限的环境（如 16GB VRAM），请参考 `configs/train/ar_training_config_memory_intensive.yaml`。

**关键优化点**:
- 启用 `gradient_checkpointing`
- 减小 `batch_size` 并增加 `accumulate_grad_batches`
- 使用 `fp16` 混合精度

## 3. 自定义扩展

### 3.1 添加新模型
1. 在 `models/spatial/` 或 `models/temporal/` 下新建模型文件。
2. 确保模型类继承自 `nn.Module` 并实现标准 `forward` 接口。
3. 在 `configs/model/` 下创建对应的 YAML 配置文件。
4. 在 `models/factory.py` 中注册新模型（如果使用工厂模式）。

### 3.2 自定义数据集
如果您的数据格式不符合标准 HDF5 规范（参见 `datasets/real_diffusion_reaction_dataset.py`），建议：
1. 继承 `RealDiffusionReactionDataset` 类。
2. 重写 `_load_data` 和 `__getitem__` 方法。
3. 在 `configs/data/` 中创建新配置指向您的数据集类。

## 4. 常见问题 (FAQ)

**Q: 为什么 `num_workers > 0` 时会报错？**
A: HDF5 文件句柄不能跨进程共享。确保您的数据集类实现了 `__getstate__` 和 `__setstate__` 来正确处理文件句柄，或者在调试时设置 `num_workers=0`。

**Q: 如何查看训练曲线？**
A: 训练日志默认保存在 `runs/<experiment_name>/`。
运行 TensorBoard:
```bash
tensorboard --logdir runs/
```

**Q: 遇到 `CUDA out of memory` 怎么办？**
A: 尝试减小 `batch_size`，或者在配置中启用 `amp.enabled: true`。
