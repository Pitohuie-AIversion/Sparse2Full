# 纯空间预测训练技术指南

## 1. 纯空间预测的概念和目标

纯空间预测（Pure Spatial Prediction）是指模型仅基于当前时刻的空间信息进行预测，不涉及时间序列建模。与时空预测不同，纯空间预测专注于从空间观测数据（如稀疏观测、降采样观测）重建完整的空间场。

### 主要应用场景：
- **超分辨率重建**：从低分辨率观测重建高分辨率场
- **空间插值**：从稀疏观测点重建完整场
- **去噪/去模糊**：从噪声或模糊观测重建清晰场
- **数据同化**：结合观测和背景场生成最优估计

### 核心目标：
- 最大化空间重建精度（Rel-L2、MAE、PSNR等指标）
- 保持物理一致性（边界条件、能量守恒等）
- 实现高效训练和推理

## 2. 配置文件结构和关键参数

### 基础配置结构
```yaml
# 纯空间预测配置模板
task:
  type: "spatial"  # 纯空间任务标识
  mode: "super_resolution"  # 或 "crop", "denoise"
  
data:
  name: "PDEBenchDataset"
  observation:
    mode: "SR"  # SR: 超分辨率, Crop: 裁剪, Full: 全观测
    scale: 4    # 超分辨率倍数
    sigma: 1.0  # 高斯模糊标准差（如适用）
  
model:
  type: "SwinUNet"  # 或 "UNet", "FNO", "Hybrid" 等
  in_channels: 1
  out_channels: 1
  
training:
  temporal_mode: "spatial_only"  # 关键：禁用时间建模
  T_in: 1    # 输入时间步数（必须为1）
  T_out: 1   # 输出时间步数（必须为1）
  
ar_wrapper:
  enabled: false  # 禁用AR包装器
```

### 关键参数说明

| 参数类别 | 参数名 | 说明 | 纯空间推荐值 |
|---------|--------|------|-------------|
| 任务类型 | task.type | 任务类型标识 | "spatial" |
| 观测模式 | data.observation.mode | 观测生成方式 | "SR"/"Crop"/"Full" |
| 时间配置 | training.T_in/T_out | 时间步数 | 1 |
| AR包装器 | ar_wrapper.enabled | 是否启用AR | false |
| 时间建模 | training.temporal_mode | 时间建模模式 | "spatial_only" |

## 3. 训练流程和步骤

### 步骤1：准备配置文件
```bash
# 复制基础配置并修改
cp configs/basic/train_real_dr_data_ar.yaml configs/spatial/spatial_sr_config.yaml
```

### 步骤2：修改关键参数
在配置文件中设置纯空间预测参数：
```yaml
# 纯空间预测专用配置
training:
  temporal_mode: "spatial_only"
  T_in: 1
  T_out: 1
  
ar_wrapper:
  enabled: false
  
data:
  observation:
    mode: "SR"
    scale: 4
```

### 步骤3：启动训练
```bash
# 单GPU训练
python training_system/scripts/train.py --config configs/spatial/spatial_sr_config.yaml

# 多GPU训练（推荐）
cd training_system
python launch_real_dr_ar_training.py --config ../configs/spatial/spatial_sr_config.yaml --gpus 0,1
```

### 步骤4：监控训练
训练过程中会自动生成：
- 训练曲线（损失、学习率等）
- 预测可视化（GT vs Pred）
- 验证指标（Rel-L2、MAE、PSNR等）

## 4. 与时空预测的区别

| 特征 | 纯空间预测 | 时空预测 |
|------|-----------|----------|
| 输入数据 | 单时间步空间场 | 多时间步序列 |
| 模型架构 | 纯空间网络（UNet、Swin等） | 时空网络（带时间建模） |
| 训练目标 | 空间重建精度 | 时空演化精度 |
| AR包装器 | 禁用 | 启用 |
| 时间步数 | T_in=T_out=1 | T_in≥1, T_out≥1 |
| 应用场景 | 超分辨率、插值、去噪 | 时间序列预测 |

## 5. 最佳实践和注意事项

### 5.1 数据准备
- **标准化**：使用z-score标准化，确保数据分布一致
- **观测一致性**：训练和验证使用相同的观测算子H
- **边界处理**：明确边界条件（mirror/zero/wrap）

### 5.2 模型选择
- **UNet**：经典选择，适合各种空间任务
- **SwinUNet**：Transformer架构，长距离依赖建模能力强
- **Hybrid**：结合CNN和Transformer优势
- **FNO**：频域建模，适合周期性边界

### 5.3 训练策略
- **学习率**：1e-3起始，余弦退火
- **损失函数**：L_rec + λ_s L_spec + λ_dc L_dc
- **数据增强**：适度旋转、翻转增强
- **验证频率**：每epoch验证，早停防止过拟合

### 5.4 性能优化
- **混合精度**：启用AMP加速训练
- **梯度累积**：大批量训练时启用
- **多GPU**：DDP分布式训练提升效率
- **内存管理**：启用gradient_checkpointing

## 6. 具体的YAML配置示例

### 超分辨率配置（×4）
```yaml
# configs/spatial/spatial_sr4_config.yaml
experiment:
  name: "SRx4-DR2D-256-SwinUNet-spatial"
  seed: 42

task:
  type: "spatial"
  mode: "super_resolution"

data:
  name: "PDEBenchDataset"
  path: "/path/to/pdebench/data"
  resolution: 256
  observation:
    mode: "SR"
    scale: 4
    sigma: 1.0
    kernel_size: 5
  normalization: "z_score"
  train_split: "splits/train.txt"
  val_split: "splits/val.txt"
  test_split: "splits/test.txt"

model:
  type: "SwinUNet"
  in_channels: 1
  out_channels: 1
  img_size: 256
  patch_size: 4
  embed_dim: 96
  depths: [2, 2, 6, 2]
  num_heads: [3, 6, 12, 24]

training:
  temporal_mode: "spatial_only"
  T_in: 1
  T_out: 1
  epochs: 100
  batch_size: 16
  learning_rate: 1e-3
  weight_decay: 1e-4
  warmup_steps: 1000
  
ar_wrapper:
  enabled: false
  
loss:
  reconstruction:
    type: "L2"
    weight: 1.0
  spectral:
    type: "L2"
    weight: 0.5
    max_freq: 16
  data_consistency:
    type: "L2"
    weight: 1.0

validation:
  metrics: ["Rel_L2", "MAE", "PSNR", "SSIM"]
  save_best: "Rel_L2"
  
visualization:
  enabled: true
  save_plots: true
  plot_interval: 5
  max_samples: 10
  
device:
  gpu: 0
  mixed_precision: true
  compile_model: false
  
logging:
  level: "INFO"
  save_interval: 10
```

### 空间裁剪配置（40%观测）
```yaml
# configs/spatial/spatial_crop40_config.yaml
task:
  type: "spatial"
  mode: "crop"

data:
  observation:
    mode: "Crop"
    crop_ratio: 0.4
    crop_strategy: "mixed"  # uniform: 40%, boundary: 30%, high_gradient: 30%
    boundary_width: 16
```

### 去噪配置
```yaml
# configs/spatial/spatial_denoise_config.yaml
task:
  type: "spatial"
  mode: "denoise"

data:
  observation:
    mode: "Full"  # 全观测但添加噪声
    noise_level: 0.1
    noise_type: "gaussian"
```

## 7. 训练启动命令

### 基础训练
```bash
cd training_system

# 超分辨率训练
python scripts/train.py --config ../configs/spatial/spatial_sr4_config.yaml

# 空间裁剪训练  
python scripts/train.py --config ../configs/spatial/spatial_crop40_config.yaml

# 去噪训练
python scripts/train.py --config ../configs/spatial/spatial_denoise_config.yaml
```

### 高级训练（推荐）
```bash
# 双GPU分布式训练
python launch_real_dr_ar_training.py \
  --config ../configs/spatial/spatial_sr4_config.yaml \
  --gpus 0,1 \
  --distributed \
  --benchmark

# 带硬件优化的训练
python scripts/train.py \
  --config ../configs/spatial/spatial_sr4_config.yaml \
  --hardware_optimize \
  --auto_batch_size
```

## 8. 结果验证和可视化

训练完成后，结果将保存在：`runs/<experiment_name>/`

### 主要输出文件：
- `checkpoints/`：模型权重文件
- `visualizations/`：训练过程可视化
- `metrics/`：验证指标记录
- `logs/`：训练日志
- `config_merged.yaml`：完整配置快照

### 验证指标：
- **Rel-L2**：相对L2误差（主要指标）
- **MAE**：平均绝对误差
- **PSNR**：峰值信噪比
- **SSIM**：结构相似性

### 可视化内容：
- 训练损失曲线
- 验证指标变化
- 预测结果对比（GT vs Pred）
- 误差分布图
- 频谱分析图

## 9. 故障排除

### 常见问题：
1. **内存不足**：减小batch_size，启用gradient_checkpointing
2. **训练不稳定**：降低学习率，增加warmup步骤
3. **过拟合**：增加数据增强，启用早停，减小模型规模
4. **收敛慢**：启用混合精度，使用更大的学习率

### 调试建议：
- 先使用小数据集快速验证配置
- 启用详细日志记录训练过程
- 使用tensorboard监控训练曲线
- 定期检查验证可视化结果

---

通过遵循本指南，您可以在training_system中成功进行纯空间预测训练，获得高质量的空间重建结果。