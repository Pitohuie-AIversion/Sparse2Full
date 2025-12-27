# 时序PDE训练系统使用指南

本文档介绍如何使用时序PDE数据集进行模型训练和可视化展示。

## 🚀 快速开始

### 1. 环境准备

确保您的系统满足以下要求：
- Python 3.8+
- PyTorch 2.0+
- CUDA (推荐，用于GPU加速)

安装依赖包：
```bash
pip install torch torchvision torchaudio
pip install hydra-core omegaconf matplotlib seaborn tqdm tensorboard h5py opencv-python
```

### 2. 数据准备

将您的时序PDE数据集放在指定目录中，支持的格式：
- HDF5文件 (`.h5` 或 `.hdf5`)
- 数据格式: `[batch, time, height, width, channels]` 或 `[time, channels, height, width]`

数据目录结构示例：
```
your_data/
├── train_data.h5
├── val_data.h5
├── test_data.h5
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 3. 一键运行

使用一键运行脚本开始训练：

```bash
# 使用默认配置
python run_temporal_training.py

# 指定数据路径
python run_temporal_training.py --data_path /path/to/your/data

# 指定配置文件
python run_temporal_training.py --config configs/experiment/temporal_training.yaml

# 从检查点恢复训练
python run_temporal_training.py --resume

# 只检查环境
python run_temporal_training.py --check_only
```

## 📁 项目结构

```
Sparse2Full/
├── configs/
│   ├── experiment/
│   │   └── temporal_training.yaml      # 时序训练配置
│   └── data/
│       └── temporal_pdebench.yaml      # 时序数据配置
├── datasets/
│   └── temporal_pdebench.py            # 时序数据集加载器
├── models/
│   └── ar/
│       └── wrapper.py                  # AR模型包装器
├── utils/
│   └── visualization.py               # 可视化工具
├── train_temporal.py                  # 时序训练脚本
├── run_temporal_training.py           # 一键运行脚本
└── README_temporal.md                 # 本文档
```

## ⚙️ 配置说明

### 时序参数配置

在 `configs/experiment/temporal_training.yaml` 中配置时序相关参数：

```yaml
temporal:
  T_in: 10          # 输入时间步长
  T_out: 5          # 输出时间步长
  dt: 0.01          # 时间步长
  
  # AR模型设置
  ar:
    teacher_forcing_ratio: 0.5    # Teacher forcing比例
    scheduled_sampling:
      enabled: true
      start_epoch: 10
      decay_rate: 0.95
```

### 数据配置

在 `configs/data/temporal_pdebench.yaml` 中配置数据相关参数：

```yaml
data:
  data_path: "/path/to/your/data"
  key_names: ["u", "v"]           # 数据键名
  task_type: "base"               # 任务类型: base/SR/Crop
  img_size: [256, 256]           # 图像尺寸
  
  temporal:
    temporal_mode: "sequence"     # 时序模式
    sequence_length: 15          # 序列长度
    overlap_ratio: 0.5           # 重叠比例
```

### 可视化配置

```yaml
visualization:
  enabled: true
  save_dir: "visualizations"
  
  training:
    plot_curves: true            # 绘制训练曲线
    save_predictions: true       # 保存预测结果
    plot_interval: 100          # 绘图间隔
  
  results:
    plot_error_maps: true        # 绘制误差图
    plot_temporal_profiles: true # 绘制时序剖面
    plot_spectral_analysis: true # 绘制频谱分析
    create_animations: true      # 创建动画
```

## 📊 训练监控

### TensorBoard

训练过程中会自动记录TensorBoard日志：

```bash
tensorboard --logdir runs/your_experiment/tensorboard
```

可以监控：
- 训练/验证损失
- 各种评估指标 (Rel-L2, MAE, PSNR, SSIM)
- 学习率变化
- 预测结果可视化

### 实时可视化

训练过程中会自动生成：
- 训练曲线图 (`visualizations/training/curves_*.png`)
- 预测结果对比 (`visualizations/training/pred_*.png`)

## 🎯 结果分析

训练完成后，系统会自动生成丰富的可视化结果：

### 1. 误差分析
- 绝对误差热图
- 相对误差热图
- 时序误差变化

### 2. 时序分析
- 时序剖面图
- 时序动画
- 对比动画

### 3. 频谱分析
- 功率谱对比
- 频域误差分析

### 4. 综合报告
- 指标对比表格
- 统计分析结果

## 🔧 高级用法

### 自定义模型

在配置文件中指定不同的模型：

```yaml
model:
  name: "SwinUNetAR"
  backbone: "swin_unet"
  ar_config:
    hidden_dim: 256
    num_layers: 4
```

### 课程学习

启用课程学习以提高训练效果：

```yaml
curriculum:
  enabled: true
  stages:
    - epoch: 0
      T_out: 2
      teacher_forcing_ratio: 0.8
    - epoch: 20
      T_out: 5
      teacher_forcing_ratio: 0.5
```

### 多GPU训练

使用多GPU加速训练：

```bash
python -m torch.distributed.launch --nproc_per_node=2 train_temporal.py
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 减小img_size
   - 减小sequence_length

2. **训练不收敛**
   - 调整学习率
   - 增加teacher_forcing_ratio
   - 启用课程学习

3. **数据加载错误**
   - 检查数据路径
   - 确认HDF5文件格式
   - 检查key_names配置

### 调试模式

启用调试模式获取更多信息：

```bash
python run_temporal_training.py --config configs/experiment/temporal_training.yaml debug=true
```

## 📈 性能优化

### 训练加速
- 使用AMP混合精度训练
- 启用数据并行
- 优化数据加载器

### 内存优化
- 使用梯度累积
- 启用梯度检查点
- 优化批处理大小

## 🤝 贡献指南

欢迎提交问题和改进建议！

### 开发环境设置

```bash
git clone <repository>
cd Sparse2Full
pip install -e .
```

### 代码规范

- 遵循PEP 8代码风格
- 添加类型注解
- 编写单元测试
- 更新文档

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**祝您训练愉快！** 🎉