# 真实扩散-反应数据AR训练配置

本配置基于training_system框架，专门用于真实扩散-反应数据的20步自回归(AR)预测训练。

## 配置特点

### 1. 遵循项目开发规范
- **一致性优先**：观测算子与训练DC复用同一实现
- **可复现性**：同一YAML+种子，验证指标方差≤1e-4
- **统一接口**：模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
- **完整监控**：包含验证、可视化、论文包生成

### 2. AR训练特定功能
- **20步自回归预测**：支持长序列时序建模
- **课程学习**：分阶段训练（5→15→20步预测）
- **教师强制**：动态调整教师强制比例
- **时序一致性**：专门的AR损失函数

### 3. 数据配置
- **数据集**：PDEBench 2D Diffusion-Reaction
- **数据路径**：`/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codx/pdebench_extended/data/PDEBench/pdebench/data_download/....data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5`
- **样本数**：前10个样本（keys 0000-0009）
- **时间步**：80个时间步，步长1
- **图像尺寸**：128×128
- **通道数**：2（u和v变量）

### 4. 模型配置
- **基础模型**：SwinUNet
- **AR包装器**：ARWrapper，支持自回归预测
- **输入通道**：2（T_in=1）
- **输出通道**：2（单步预测）
- **AR步数**：20步

### 5. 训练策略
- **课程学习**：
  - 阶段1：20轮，预测5步，教师强制0.8
  - 阶段2：40轮，预测15步，教师强制0.6
  - 阶段3：140轮，预测20步，教师强制0.3
- **优化器**：AdamW(lr=1e-3, wd=1e-4)
- **调度器**：CosineAnnealing + Warmup
- **梯度裁剪**：1.0
- **早停**：patience=30，监控val_rel_l2

### 6. 损失函数（三件套）
- **重建损失**：L2，权重1.0
- **频谱损失**：低频16模式，权重0.5
- **数据一致性**：权重1.0
- **AR特定损失**：
  - 时序一致性：权重1.0
  - 累积误差：权重0.5

### 7. 观测算子
- **模式**：超分辨率（SR×2）
- **模糊**：高斯模糊，σ=1.0，k=5
- **边界**：mirror
- **插值**：area

### 8. 验证指标
- **基础指标**：rel_l2, mae, psnr, ssim, dc_error
- **AR特定**：ar_temporal_error, long_term_stability
- **验证频率**：每10轮

### 9. 可视化
- **预测图**：GT vs Pred
- **误差图**：绝对误差分布
- **频谱图**：功率谱对比
- **AR轨迹**：20步预测轨迹
- **时序演化**：时序发展过程

### 10. 性能监控
- **资源监控**：GPU、CPU、IO使用率
- **报告间隔**：60秒
- **自动调优**：根据资源使用情况调整参数

## 使用说明

### 基本训练
```bash
# 进入training_system目录
cd /share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/training_system

# 执行训练
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar
```

### 高级用法
```bash
# 自定义实验名称
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar experiment.name=MyARExperiment

# 修改训练轮数
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar training.epochs=300

# 调整学习率
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar training.optimizer.params.lr=5e-4

# 修改AR步数
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar ar_wrapper.T_out=15

# 禁用课程学习
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar curriculum.enabled=false

# 多GPU训练
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar --multirun

# 多种子训练
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar --seeds 42 123 456
```

### 调试模式
```bash
# 详细日志
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar --debug

# 干运行（只打印命令）
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar --dry-run

# 检查配置
python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar --check-config
```

## 输出结构

训练完成后，输出目录结构如下：

```
runs/Real-DR2D-AR-T20-128-SwinUNet-AR-s2025/
├── config_merged.yaml          # 合并后的完整配置
├── checkpoints/                # 模型检查点
│   ├── epoch_020.ckpt         # 每20轮保存
│   ├── epoch_040.ckpt
│   ├── epoch_060.ckpt
│   ├── epoch_080.ckpt
│   ├── epoch_100.ckpt
│   ├── epoch_120.ckpt
│   ├── epoch_140.ckpt
│   ├── epoch_160.ckpt
│   ├── epoch_180.ckpt
│   └── epoch_200.ckpt
├── logs/                      # 训练日志
│   ├── train.log
│   └── validation.log
├── tensorboard/               # TensorBoard日志
│   └── events.out.tfevents.*
├── metrics/                   # 评估指标
│   ├── train_metrics.jsonl
│   ├── val_metrics.jsonl
│   └── test_metrics.jsonl
├── visualizations/            # 可视化结果
│   ├── epoch_020/
│   ├── epoch_040/
│   └── ...
└── paper_package/             # 论文材料包
    ├── data_cards/
    ├── configs/
    ├── checkpoints/
    ├── metrics/
    ├── figs/
    └── scripts/
```

## 关键特性

### 1. 课程学习（Curriculum Learning）
分阶段逐步增加预测步数，帮助模型更好地学习时序依赖关系：
- 阶段1：预测5步，建立基础空间特征
- 阶段2：预测15步，增强时序建模能力
- 阶段3：预测20步，完整AR能力

### 2. 教师强制（Teacher Forcing）
动态调整教师强制比例，平衡训练稳定性和预测准确性：
- 初期：高比例（0.8）确保稳定训练
- 中期：中等比例（0.6）逐步过渡
- 后期：低比例（0.3）增强自主预测能力

### 3. AR特定损失函数
- **时序一致性损失**：确保相邻时间步的预测一致性
- **累积误差损失**：控制长期预测的误差累积

### 4. 完整监控体系
- **训练监控**：损失、指标、学习率
- **验证监控**：每10轮验证，早停机制
- **性能监控**：资源使用情况
- **可视化监控**：预测质量、频谱分析

## 注意事项

1. **CPU模式**：由于NVIDIA驱动问题，强制使用CPU模式
2. **确定性**：启用确定性计算确保可复现性
3. **内存管理**：小批次训练避免OOM
4. **数据路径**：确保数据文件路径正确
5. **权限**：确保有数据文件读取权限

## 故障排除

### 常见问题

1. **CUDA错误**：配置已强制使用CPU模式
2. **内存不足**：减小batch_size或增加num_workers_step
3. **数据加载失败**：检查数据文件路径和权限
4. **训练不收敛**：调整学习率或检查数据预处理

### 调试建议

1. 使用`--debug`启用详细日志
2. 使用`--dry-run`检查命令正确性
3. 检查配置文件语法（YAML格式）
4. 验证数据文件完整性
5. 监控资源使用情况

## 扩展性

本配置支持以下扩展：
- **模型替换**：可替换为其他模型（如U-Net、FNO等）
- **数据扩展**：支持其他PDE数据集
- **损失函数**：可添加自定义损失
- **观测算子**：支持多种降采样模式
- **分布式训练**：支持多GPU训练