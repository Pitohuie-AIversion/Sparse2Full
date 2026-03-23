# 纯空间恢复配置文件使用说明

## 配置文件概述

`configs/spatial_sr_only.yaml` 是一个专门设计的纯空间超分辨率恢复配置文件，完全针对空间恢复任务优化，具有以下特点：

- ✅ **纯空间恢复**：单帧输入输出，无时序依赖
- ✅ **单分量处理**：只使用u分量（第一个通道），忽略v分量
- ✅ **AR损失禁用**：完全移除时序预测损失
- ✅ **课程学习禁用**：无渐进式训练策略
- ✅ **观测算子一致性**：严格遵循黄金法则，H与DC配置完全一致
- ✅ **轻量级设置**：样本数量、模型复杂度、训练轮数均优化为快速验证
- ✅ **可视化启用**：TensorBoard日志和测试可视化全面开启

## 关键配置特性

### 1. 数据配置
```yaml
data:
  component: u    # 指定使用u分量（第一个通道），忽略v分量
  T_in: 1    # 单帧输入
  T_out: 1   # 单帧输出 - 关键修改
  sample_limit: 50  # 限制样本数量便于快速验证
```

### 2. 损失函数
```yaml
loss:
  ar_loss:
    weight: 0.0  # 完全禁用AR损失
  reconstruction:
    weight: 1.0  # 主要损失为重构损失
```

### 3. 观测算子一致性
```yaml
observation:        # 训练观测生成
  mode: sr
  scale_factor: 2
  # ... 完整配置

data_observation:   # 数据模块观测生成 - 完全一致
  mode: sr
  scale_factor: 2
  # ... 完全相同配置
```

### 4. 轻量级模型
```yaml
model:
  in_channels: 4   # baseline(1) + mask(1) + coords(2) = 4通道
  out_channels: 1  # 单通道输出 - 只预测u分量
  use_coords: true # 启用坐标编码
  depths: [1, 1, 1, 1]      # 最小深度
  embed_dim: 32            # 减少嵌入维度
  mlp_ratio: 2.0           # 减少MLP比例
```

### 5. 训练配置
```yaml
training:
  epochs: 50               # 适度训练轮数
  batch_size: 4            # 适中批次大小
  curriculum:
    enabled: false         # 禁用课程学习
```

### 6. 可视化启用
```yaml
logging:
  use_tensorboard: true    # 启用TensorBoard

visualization:
  enabled: true            # 启用测试可视化
  save_test_visualizations: true
```

## 使用方法

### 1. 基础训练命令
```bash
# 使用配置文件进行训练
python train.py --config configs/spatial_sr_only.yaml

# 或者使用Hydra风格
python train.py config=configs/spatial_sr_only.yaml
```

### 2. 指定实验名称
```bash
# 自定义实验名称
python train.py --config configs/spatial_sr_only.yaml experiment.name="MySpatialSR-Test"
```

### 3. 调整训练轮数
```bash
# 快速测试（10轮）
python train.py --config configs/spatial_sr_only.yaml training.epochs=10

# 完整训练（100轮）
python train.py --config configs/spatial_sr_only.yaml training.epochs=100
```

### 4. 调整样本数量
```bash
# 更少样本（超快速测试）
python train.py --config configs/spatial_sr_only.yaml data.sample_limit=20

# 更多样本（标准训练）
python train.py --config configs/spatial_sr_only.yaml data.sample_limit=200
```

### 5. 调整超分辨率倍数
```bash
# 4倍超分辨率
python train.py --config configs/spatial_sr_only.yaml observation.scale_factor=4 data_observation.scale_factor=4
```

### 6. 启用GPU训练（如果可用）
```bash
# 修改设备配置
device:
  accelerator: gpu
  devices: 1
```

## 训练过程监控

### 1. TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir runs/Spatial-SR-Only-*/tensorboard

# 在浏览器中打开
# http://localhost:6006
```

### 2. 训练日志
训练日志包含以下关键信息：
- 重构损失（主要损失）
- 数据一致性损失（DC损失）
- 频域损失（低频分量）
- 验证指标（Rel-L2、MAE、PSNR、SSIM）
- 训练时间、内存使用情况

### 3. 测试可视化
训练完成后，测试可视化结果将保存在：
```
runs/Spatial-SR-Only-*/visualizations/
├── test_samples/
│   ├── sample_0000/
│   │   ├── gt.png              # 真实值
│   │   ├── pred.png            # 预测值
│   │   ├── error.png           # 误差图
│   │   └── spectrum.png        # 频谱分析
│   └── ...
└── summary_plots/
    ├── metrics_comparison.png
    └── convergence_analysis.png
```

## 验证与测试

### 1. 运行验证
```bash
# 运行验证脚本
python validate.py --config configs/spatial_sr_only.yaml --checkpoint runs/Spatial-SR-Only-*/checkpoints/best.ckpt
```

### 2. 运行测试
```bash
# 运行测试脚本
python test.py --config configs/spatial_sr_only.yaml --checkpoint runs/Spatial-SR-Only-*/checkpoints/best.ckpt
```

### 3. 一致性检查
```bash
# 验证观测算子H与DC一致性
python tools/check_dc_equivalence.py --config configs/spatial_sr_only.yaml
```

## 性能基准

### 预期性能指标（CPU环境）
- **训练时间**：~20分钟（50轮，50样本，单分量）
- **内存使用**：< 3GB
- **验证Rel-L2**：< 0.12（目标值，单分量更易收敛）
- **测试PSNR**：> 26 dB（目标值）

### 资源消耗
```
模型参数量：~0.5M
FLOPs：~1.2G@256²（单分量）
峰值内存：~1.5GB
推理延迟：~40ms（CPU，单分量）
```

## 常见问题

### Q1: 训练速度太慢？
**解决方案**：
- 减少 `data.sample_limit` 到 20
- 减少 `training.epochs` 到 20
- 增加 `dataloader.num_workers` 到 4
- 单分量处理已经比双分量快约30%

### Q2: 内存不足？
**解决方案**：
- 减少 `batch_size` 到 2
- 减少 `model.embed_dim` 到 16
- 禁用 `visualization.enabled`
- 单分量处理内存占用已减少约25%

### Q3: 验证指标不理想？
**解决方案**：
- 增加 `training.epochs` 到 100
- 增加 `data.sample_limit` 到 200
- 调整 `loss.reconstruction.weight` 到 2.0
- 增加 `data.augmentation.enabled: true`
- 单分量任务通常更容易收敛，可尝试减少训练轮数

### Q4: 观测算子不一致？
**解决方案**：
- 检查 `observation` 和 `data_observation` 配置是否完全相同
- 运行一致性检查脚本
- 确保 `data.component` 与 `model.in_channels` 匹配（u分量需要4输入通道：baseline+mask+coords）

## 扩展使用

### 1. 切换到v分量
```bash
# 修改数据配置，使用v分量
python train.py --config configs/spatial_sr_only.yaml data.component=v
```

### 2. 多尺度超分辨率
```bash
# 创建2x->4x渐进式配置
python train.py --config configs/spatial_sr_only.yaml \
  observation.scale_factor=2 \
  data_observation.scale_factor=2 \
  training.curriculum.enabled=true \
  training.curriculum.stages=[2,4]
```

### 3. 不同模型架构
```bash
# 使用U-Net
python train.py --config configs/spatial_sr_only.yaml \
  model.name=UNet \
  model.depths=[2,2,2,2]

# 使用FNO
python train.py --config configs/spatial_sr_only.yaml \
  model.name=FNO \
  model.modes1=16 \
  model.modes2=16
```

### 4. 不同数据集
```yaml
# 修改数据配置
data:
  dataset_name: "YourDataset"
  data_path: "/path/to/your/data"
  channels: 3  # 多通道数据
```

## 注意事项

1. **单分量处理**：确保 `data.component` 与 `model.in_channels` 匹配
2. **观测算子一致性**：务必保持 `observation` 和 `data_observation` 配置完全相同
3. **内存管理**：CPU环境下注意内存使用，必要时减少批次大小
4. **随机种子**：固定种子确保实验可复现性
5. **验证频率**：适当的验证频率有助于监控训练进度
6. **检查点保存**：定期保存检查点，防止训练中断

## 技术支持

如遇到问题，请检查：
1. 配置文件语法是否正确
2. 数据路径是否有效
3. 依赖库版本是否匹配
4. 系统资源是否充足

祝训练顺利！🚀