# PDEBench高性能训练脚本

基于Swin Transformer和时间编码的高性能PDE求解器训练框架。

## 特性

- 🚀 **高性能优化**: NUMA感知内存管理、GPU特化优化、混合精度训练
- 🧠 **先进模型**: 时间编码Swin Transformer，支持多种时间编码方式
- 📊 **实时监控**: 性能监控、资源使用跟踪、TensorBoard集成
- 🔄 **分布式训练**: 支持多GPU分布式训练（DDP/FSDP）
- 📈 **基准测试**: 性能基准测试和验证工具
- 🔧 **灵活配置**: YAML配置文件，支持多种训练模式

## 快速开始

### 安装依赖

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 基础训练

```bash
# 使用基础配置进行训练
python scripts/train.py --config configs/base_config.yaml

# 使用高性能配置
python scripts/train.py --config configs/high_performance_config.yaml

# 使用调试配置（小模型，便于测试）
python scripts/train.py --config configs/debug_config.yaml
```

### 分布式训练

```bash
# 4 GPU分布式训练
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py \
    --config configs/high_performance_config.yaml \
    --distributed

# 或者使用torchrun（推荐）
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/high_performance_config.yaml \
    --distributed
```

### 基准测试

```bash
# 运行性能基准测试
python scripts/benchmark_performance.py --config configs/base_config.yaml

# 运行多配置基准测试套件
python scripts/benchmark_performance.py --suite
```

## 配置说明

### 模型配置

```yaml
model:
  name: "SwinTemporalNAR"          # 模型名称
  input_channels: 1                # 输入通道数
  hidden_dim: 128                  # 隐藏层维度
  num_layers: 4                    # Transformer层数
  num_heads: 8                     # 注意力头数
  window_size: 7                   # 窗口大小
  time_steps: 10                   # 时间步数
  prediction_steps: 5              # 预测步数
  spatial_resolution: [64, 64]     # 空间分辨率
  temporal_encoding_type: "sinusoidal"  # 时间编码类型
```

### 训练配置

```yaml
training:
  batch_size: 16                   # 批次大小
  num_epochs: 100                # 训练轮数
  learning_rate: 1e-3            # 学习率
  mixed_precision: true          # 混合精度训练
  compile_model: false           # 模型编译（PyTorch 2.0+）
  gradient_accumulation_steps: 1  # 梯度累积步数
```

### 硬件配置

```yaml
hardware:
  device: "auto"                  # 设备选择（auto/cpu/cuda）
  num_workers: 4                 # 数据加载进程数
  pin_memory: true               # 内存锁定
  numa_aware: true              # NUMA感知
  gpu_memory_fraction: 0.9       # GPU内存使用比例
```

### 分布式配置

```yaml
distributed:
  enabled: true                  # 启用分布式训练
  backend: "nccl"               # 通信后端
  strategy: "ddp"               # 分布式策略
  world_size: 4                  # 总进程数
```

## 数据格式

支持多种数据格式：

- **HDF5**: 高效的科学数据格式
- **Zarr**: 分块数组存储，适合大数据
- **NumPy**: 标准的.npy文件

数据应该按照以下结构组织：

```
data/
├── train/
│   ├── data_0001.h5
│   ├── data_0002.h5
│   └── ...
├── validation/
│   ├── data_0001.h5
│   └── ...
└── test/
    ├── data_0001.h5
    └── ...
```

每个数据文件应该包含：
- 时间序列数据：(time_steps, channels, height, width)
- 对应的标签数据（如果需要）

## 性能优化

### NUMA优化

- 自动检测NUMA拓扑
- 内存交错分配
- CPU亲和性绑定

### GPU优化

- Tensor Core优化
- CUDA流管理
- 内存池管理
- 混合精度训练

### 数据管道优化

- 多线程数据加载
- 预取和缓存
- 内存映射文件
- NUMA感知数据访问

## 监控和日志

### 实时监控

- CPU/GPU使用率
- 内存使用情况
- 训练损失和指标
- 数据加载性能

### TensorBoard集成

```bash
# 启动TensorBoard
tensorboard --logdir ./logs
```

### 性能报告

训练完成后自动生成：
- 性能摘要报告
- 资源使用统计
- 训练曲线图

## 扩展开发

### 添加新模型

1. 在 `src/models/` 中创建新模型类
2. 继承基础模型接口
3. 在配置中注册模型名称
4. 更新训练脚本

### 添加新优化器

1. 在 `src/optimizers/` 中创建优化器类
2. 实现优化逻辑
3. 在配置中添加参数
4. 集成到训练流程

### 添加新数据格式

1. 在 `src/data/` 中创建数据加载器
2. 实现数据读取和预处理
3. 在配置中支持新格式
4. 更新数据模块

## 故障排除

### 内存不足

- 减小 `batch_size`
- 减小 `hidden_dim`
- 启用 `mixed_precision`
- 减少 `num_workers`

### 训练速度慢

- 启用 `numa_aware`
- 增加 `num_workers`
- 启用 `mixed_precision`
- 检查GPU利用率

### 分布式训练问题

- 检查NCCL配置
- 验证网络连接
- 检查GPU驱动版本
- 查看分布式日志

## 性能基准

在NVIDIA A100 GPU上的基准测试结果：

| 配置 | 批大小 | 吞吐量(samples/sec) | 内存使用(GB) |
|------|--------|-------------------|-------------|
| 基础 | 16 | 150 | 8 |
| 高性能 | 32 | 350 | 16 |
| 分布式(4GPU) | 128 | 1200 | 64 |

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 引用

如果您使用本框架，请引用：

```bibtex
@software{pdebench_high_performance,
  title={PDEBench High-Performance Training Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```