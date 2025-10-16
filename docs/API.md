# API 文档

本文档详细说明了PDEBench稀疏观测重建系统各模块的API接口和使用方法。

## 目录

- [数据集模块 (datasets)](#数据集模块-datasets)
- [模型模块 (models)](#模型模块-models)
- [核心算子 (ops)](#核心算子-ops)
- [工具模块 (utils)](#工具模块-utils)
- [训练工具 (tools)](#训练工具-tools)
- [配置管理](#配置管理)
- [扩展开发](#扩展开发)
- [最佳实践](#最佳实践)

## 数据集模块 (datasets)

### PDEBenchDataset

PDEBench数据集的主要接口类。

```python
from datasets.pdebench import PDEBenchDataset

dataset = PDEBenchDataset(
    root="data/pdebench",           # 数据根目录
    task="sr",                      # 任务类型: "sr" 或 "crop"
    split="train",                  # 数据切分: "train", "val", "test"
    scale_factor=4,                 # 超分辨率倍数 (仅SR任务)
    crop_ratio=0.2,                 # 裁剪比例 (仅Crop任务)
    normalize=True,                 # 是否标准化
    cache_data=False,               # 是否缓存数据到内存
    transform=None                  # 数据变换
)
```

#### 参数说明

- `root` (str): 数据集根目录路径
- `task` (str): 任务类型，支持 "sr" (超分辨率) 和 "crop" (裁剪重建)
- `split` (str): 数据切分，支持 "train", "val", "test"
- `scale_factor` (int): 超分辨率倍数，仅在SR任务中使用
- `crop_ratio` (float): 裁剪比例，仅在Crop任务中使用
- `normalize` (bool): 是否进行z-score标准化
- `cache_data` (bool): 是否将数据缓存到内存中
- `transform` (callable): 数据变换函数

#### 方法

```python
# 获取数据项
data_item = dataset[index]
# 返回: {"input": tensor, "target": tensor, "metadata": dict}

# 获取数据集大小
length = len(dataset)

# 获取标准化统计信息
norm_stats = dataset.get_norm_stats()
# 返回: {"mean": tensor, "std": tensor}

# 反标准化
denormalized = dataset.denormalize(normalized_tensor)
```

### 数据变换 (transforms)

```python
from datasets.transforms import (
    RandomCrop, RandomFlip, GaussianNoise, 
    ToTensor, Normalize, Compose
)

# 组合多个变换
transform = Compose([
    RandomCrop(size=(128, 128)),
    RandomFlip(p=0.5),
    GaussianNoise(std=0.01),
    ToTensor(),
    Normalize(mean=0.0, std=1.0)
])
```

## 模型模块 (models)

### 基础模型接口

所有模型都继承自 `torch.nn.Module` 并遵循统一接口：

```python
class BaseModel(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            **kwargs: 模型特定参数
        """
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量 [B, C_in, H, W]
        Returns:
            torch.Tensor: 输出张量 [B, C_out, H, W]
        """
```

### Swin-UNet

```python
from models.swin_unet import SwinUNet

model = SwinUNet(
    in_channels=3,              # 输入通道数
    out_channels=3,             # 输出通道数
    embed_dim=96,               # 嵌入维度
    depths=[2, 2, 6, 2],        # 各层深度
    num_heads=[3, 6, 12, 24],   # 注意力头数
    window_size=7,              # 窗口大小
    mlp_ratio=4.0,              # MLP扩展比例
    drop_rate=0.0,              # Dropout率
    drop_path_rate=0.1,         # DropPath率
    patch_size=4,               # 补丁大小
    norm_layer=nn.LayerNorm,    # 标准化层
    use_checkpoint=False        # 是否使用梯度检查点
)
```

### Hybrid模型

```python
from models.hybrid import HybridModel

model = HybridModel(
    in_channels=3,              # 输入通道数
    out_channels=3,             # 输出通道数
    embed_dim=64,               # 嵌入维度
    num_layers=4,               # 层数
    attention_heads=8,          # 注意力头数
    fno_modes=16,               # FNO模态数
    mlp_hidden_dim=256,         # MLP隐藏维度
    fusion_type="concat"        # 融合方式: "concat", "add", "attention"
)
```

### FNO (傅里叶神经算子)

```python
from models.fno import FNO2D

model = FNO2D(
    in_channels=3,              # 输入通道数
    out_channels=3,             # 输出通道数
    modes1=12,                  # x方向模态数
    modes2=12,                  # y方向模态数
    width=64,                   # 通道宽度
    num_layers=4,               # 层数
    activation="gelu"           # 激活函数
)
```

### 模型注册与创建

```python
from models import create_model

# 通过配置创建模型
model = create_model(
    model_type="swin_unet",
    in_channels=3,
    out_channels=3,
    **model_config
)

# 获取可用模型列表
available_models = get_available_models()
```

## 核心算子 (ops)

### 退化算子 (观测算子H)

```python
from ops.degradation import (
    SuperResolutionOperator, 
    CropOperator, 
    create_degradation_operator
)

# 超分辨率退化算子
sr_op = SuperResolutionOperator(
    scale_factor=4,             # 下采样倍数
    blur_kernel_size=5,         # 模糊核大小
    blur_sigma=1.0,             # 高斯模糊标准差
    noise_level=0.0,            # 噪声水平
    interpolation="bilinear"    # 插值方法
)

# 裁剪退化算子
crop_op = CropOperator(
    crop_ratio=0.2,             # 裁剪比例
    crop_strategy="random",     # 裁剪策略: "random", "center", "boundary"
    boundary_width=16,          # 边界宽度
    fill_value=0.0             # 填充值
)

# 应用退化算子
degraded = sr_op(high_res_image)
degraded = crop_op(full_image)
```

### 损失函数

```python
from ops.loss import (
    ReconstructionLoss, SpectralLoss, 
    ConsistencyLoss, CombinedLoss
)

# 重建损失
recon_loss = ReconstructionLoss(
    loss_type="mse",            # 损失类型: "mse", "l1", "huber"
    reduction="mean"            # 归约方式: "mean", "sum", "none"
)

# 频域损失
spectral_loss = SpectralLoss(
    low_freq_modes=16,          # 低频模态数
    loss_type="mse",            # 损失类型
    weight=0.5                  # 权重
)

# 一致性损失
consistency_loss = ConsistencyLoss(
    degradation_op=sr_op,       # 退化算子
    weight=1.0                  # 权重
)

# 组合损失
combined_loss = CombinedLoss(
    losses=[recon_loss, spectral_loss, consistency_loss],
    weights=[1.0, 0.5, 1.0]
)

# 计算损失
loss_value = combined_loss(prediction, target, degraded_input)
```

### 评估指标

```python
from ops.metrics import (
    RelativeL2Error, MAE, PSNR, SSIM,
    FrequencyRMSE, BoundaryRMSE, ConsistencyError,
    MetricsCalculator
)

# 创建指标计算器
metrics_calc = MetricsCalculator(
    metrics=["rel_l2", "mae", "psnr", "ssim", "frmse", "brmse"],
    degradation_op=sr_op,
    boundary_width=16
)

# 计算指标
metrics = metrics_calc.calculate(prediction, target, degraded_input)
# 返回: {"rel_l2": float, "mae": float, "psnr": float, ...}

# 批量计算
batch_metrics = metrics_calc.calculate_batch(pred_batch, target_batch, degraded_batch)
```

## 工具模块 (utils)

### 分布式训练

```python
from utils.distributed import DistributedManager, launch_distributed_training

# 初始化分布式环境
dist_manager = DistributedManager()
dist_manager.init_process_group(backend="nccl")

# 包装模型和优化器
model = dist_manager.wrap_model(model)
optimizer = dist_manager.wrap_optimizer(optimizer)

# 创建分布式数据加载器
dataloader = dist_manager.create_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 启动分布式训练
launch_distributed_training(
    train_func=train_function,
    num_gpus=4,
    config=config
)
```

### 可视化工具

```python
from utils.visualization import PDEBenchVisualizer

# 创建可视化器
visualizer = PDEBenchVisualizer(
    save_dir="visualizations",
    dpi=300,
    figsize=(12, 8),
    colormap="viridis"
)

# 场对比图
visualizer.plot_field_comparison(
    gt=ground_truth,            # 真实值
    pred=prediction,            # 预测值
    degraded=degraded_input,    # 退化输入
    title="重建结果对比",
    save_name="comparison.png",
    channel_names=["u", "v", "p"]  # 通道名称
)

# 功率谱分析
visualizer.plot_power_spectrum(
    data=prediction,
    title="功率谱分析",
    save_name="power_spectrum.png",
    log_scale=True
)

# 误差分析
visualizer.plot_error_analysis(
    gt=ground_truth,
    pred=prediction,
    save_name="error_analysis.png",
    error_types=["absolute", "relative", "frequency"]
)

# 边界效应分析
visualizer.plot_boundary_effects(
    gt=ground_truth,
    pred=prediction,
    boundary_width=16,
    save_name="boundary_effects.png"
)

# 训练曲线
visualizer.plot_training_curves(
    metrics_history=metrics_dict,
    save_name="training_curves.png"
)
```

### 配置管理

```python
from utils.config import ConfigManager, merge_configs, save_config

# 加载配置
config_manager = ConfigManager()
config = config_manager.load_config("configs/sr_swin_unet.yaml")

# 合并配置
base_config = config_manager.load_config("configs/base.yaml")
model_config = config_manager.load_config("configs/model/swin_unet.yaml")
merged_config = merge_configs([base_config, model_config])

# 保存配置快照
save_config(config, "runs/experiment/config_snapshot.yaml")

# 命令行覆盖
config = config_manager.override_config(
    config, 
    overrides=["train.lr=1e-4", "model.embed_dim=128"]
)
```

### 性能分析

```python
from utils.performance import PerformanceProfiler, benchmark_models

# 性能分析器
profiler = PerformanceProfiler()

# 开始分析
profiler.start()

# 运行代码
output = model(input_tensor)

# 结束分析并获取报告
report = profiler.stop()
print(f"Memory: {report['memory_mb']:.2f} MB")
print(f"Time: {report['time_ms']:.2f} ms")
print(f"FLOPs: {report['flops']:.2e}")

# 批量基准测试
benchmark_results = benchmark_models(
    models={"swin_unet": model1, "fno": model2},
    input_shape=(1, 3, 256, 256),
    num_runs=100
)
```

## 训练工具 (tools)

### 训练脚本

```python
from tools.train import Trainer, CompleteTrainer

# 基础训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=loss_function,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    config=config
)

# 完整训练器 (支持分布式、可视化等)
complete_trainer = CompleteTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=loss_function,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    config=config,
    distributed=True,
    visualizer=visualizer,
    logger=logger
)

# 开始训练
trainer.train(num_epochs=100)

# 恢复训练
trainer.resume_from_checkpoint("checkpoints/latest.pth")
```

### 评估脚本

```python
from tools.eval import Evaluator

# 创建评估器
evaluator = Evaluator(
    model=model,
    test_loader=test_loader,
    metrics_calculator=metrics_calc,
    device=device,
    config=config
)

# 运行评估
results = evaluator.evaluate(
    checkpoint_path="checkpoints/best.pth",
    save_predictions=True,
    visualize=True,
    output_dir="results"
)

# 结果包含:
# - metrics: 评估指标字典
# - predictions: 预测结果 (可选)
# - visualizations: 可视化图片路径 (可选)
```

### 模型基准测试

```python
from tools.benchmark_models import ModelBenchmark

# 创建基准测试器
benchmark = ModelBenchmark(
    device="cuda",
    input_shape=(1, 3, 256, 256),
    num_warmup=10,
    num_benchmark=100
)

# 单模型基准测试
results = benchmark.benchmark_single_model(
    model=model,
    model_name="swin_unet"
)

# 批量基准测试
model_configs = {
    "swin_unet": "configs/model/swin_unet.yaml",
    "fno": "configs/model/fno.yaml",
    "hybrid": "configs/model/hybrid.yaml"
}

benchmark_results = benchmark.run_benchmark_suite(
    model_configs=model_configs,
    output_file="benchmark_results.json"
)
```

### 论文材料生成

```python
from tools.generate_paper_package import PaperPackageGenerator

# 创建生成器
generator = PaperPackageGenerator(
    runs_dir="runs",
    output_dir="paper_package",
    anonymous=False
)

# 生成完整材料包
generator.generate_complete_package(
    include_checkpoints=True,
    include_visualizations=True,
    include_raw_data=False
)

# 生成特定组件
generator.generate_data_cards()
generator.generate_config_snapshots()
generator.generate_metrics_tables()
generator.generate_visualization_figures()
generator.generate_reproduction_scripts()
```

## 配置管理

### 配置文件结构

```yaml
# configs/experiment.yaml
defaults:
  - data: pdebench
  - model: swin_unet
  - train: default
  - loss: combined
  - _self_

# 实验特定配置
experiment:
  name: "SRx4-PDEBench-256-SwinUNet"
  seed: 2025
  output_dir: "runs"

# 数据配置
data:
  root: "data/pdebench"
  task: "sr"
  scale_factor: 4
  batch_size: 16
  num_workers: 4

# 模型配置
model:
  type: "swin_unet"
  in_channels: 3
  out_channels: 3
  embed_dim: 96

# 训练配置
train:
  epochs: 100
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: "cosine"

# 损失配置
loss:
  type: "combined"
  weights: [1.0, 0.5, 1.0]
```

### 配置验证

```python
from utils.config import validate_config, ConfigSchema

# 定义配置模式
schema = ConfigSchema()

# 验证配置
is_valid, errors = validate_config(config, schema)
if not is_valid:
    print("配置错误:", errors)
```

## 错误处理和调试

### 常见异常

```python
from utils.exceptions import (
    ConfigurationError,
    DataLoadingError,
    ModelError,
    TrainingError
)

try:
    model = create_model(config.model)
except ModelError as e:
    print(f"模型创建失败: {e}")
    
try:
    dataset = PDEBenchDataset(**config.data)
except DataLoadingError as e:
    print(f"数据加载失败: {e}")
```

### 调试工具

```python
from utils.debug import (
    check_model_gradients,
    visualize_data_flow,
    profile_memory_usage
)

# 检查梯度
gradient_info = check_model_gradients(model, loss)

# 可视化数据流
visualize_data_flow(model, input_tensor, save_path="debug/data_flow.png")

# 内存使用分析
memory_report = profile_memory_usage(model, input_tensor)
```

## 扩展开发

### 添加新模型

1. **实现模型类**:
```python
# models/my_model.py
from .base import BaseModel

class MyModel(BaseModel):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels)
        # 实现模型架构
        
    def forward(self, x):
        # 实现前向传播
        return output
```

2. **注册模型**:
```python
# models/__init__.py
from .my_model import MyModel

MODEL_REGISTRY["my_model"] = MyModel
```

3. **添加配置**:
```yaml
# configs/model/my_model.yaml
type: my_model
param1: value1
param2: value2
```

### 添加新损失函数

```python
# ops/loss.py
class MyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化参数
        
    def forward(self, pred, target, **kwargs):
        # 计算损失
        return loss_value

# 注册损失函数
LOSS_REGISTRY["my_loss"] = MyLoss
```

### 添加新指标

```python
# ops/metrics.py
class MyMetric:
    def __init__(self, **kwargs):
        # 初始化参数
        pass
        
    def __call__(self, pred, target, **kwargs):
        # 计算指标
        return metric_value

# 注册指标
METRICS_REGISTRY["my_metric"] = MyMetric
```

## 最佳实践

### 性能优化

1. **使用混合精度训练**:
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

2. **梯度累积**:
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with autocast():
        output = model(batch["input"])
        loss = criterion(output, batch["target"]) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

3. **数据预加载**:
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2
)
```

### 内存管理

```python
# 清理GPU内存
torch.cuda.empty_cache()

# 使用梯度检查点
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=4)

# 删除不需要的变量
del intermediate_tensor
```

### 可复现性

```python
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2025)
```

---

更多详细信息请参考各模块的源代码和单元测试。如有问题，请提交Issue或查看故障排除指南。