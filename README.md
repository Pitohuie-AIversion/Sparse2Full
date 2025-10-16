# PDEBench稀疏观测重建系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](./tests/)

基于深度学习的偏微分方程(PDE)稀疏观测重建系统，支持超分辨率(SR)和裁剪重建(Crop)任务。

## 🎯 项目特性

- **多任务支持**: 超分辨率重建和稀疏观测裁剪重建
- **多模型架构**: Swin-UNet、Hybrid模型、FNO等先进架构
- **分布式训练**: 支持多GPU训练和分布式数据并行
- **科学可视化**: 专业的热图、功率谱、误差分析工具
- **严格评测**: 多维度指标评估和统计显著性分析
- **可复现性**: 固定种子、确定性训练、配置快照
- **端到端测试**: 完整的训练-评估流程验证
- **论文材料**: 自动生成论文所需的表格、图表和材料包

## 📁 项目结构

```
Sparse2Full/
├── configs/                 # 配置文件
│   ├── data/               # 数据配置
│   ├── model/              # 模型配置
│   ├── train/              # 训练配置
│   └── loss/               # 损失函数配置
├── datasets/               # 数据集模块
│   ├── pdebench.py        # PDEBench数据集
│   └── transforms.py      # 数据变换
├── models/                 # 模型架构
│   ├── swin_unet.py       # Swin-UNet模型
│   ├── hybrid.py          # 混合模型
│   └── fno.py             # 傅里叶神经算子
├── ops/                    # 核心算子
│   ├── degradation.py     # 退化算子(观测算子H)
│   ├── loss.py            # 损失函数
│   └── metrics.py         # 评估指标
├── utils/                  # 工具模块
│   ├── distributed.py     # 分布式训练
│   ├── visualization.py   # 可视化工具
│   └── config.py          # 配置管理
├── tools/                  # 工具脚本
│   ├── train.py           # 训练脚本
│   ├── eval.py            # 评估脚本
│   ├── benchmark_models.py # 模型基准测试
│   └── generate_paper_package.py # 论文材料生成
├── tests/                  # 测试脚本
│   ├── unit/              # 单元测试
│   ├── integration/       # 集成测试
│   └── e2e/               # 端到端测试
├── runs/                   # 实验结果
└── paper_package/          # 论文材料包
    ├── data_cards/        # 数据卡片
    ├── configs/           # 配置快照
    ├── metrics/           # 评估指标
    ├── figs/              # 图表
    └── scripts/           # 复现脚本
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.10+ (推荐3.12)
- **PyTorch**: 2.1+ 
- **CUDA**: 11.8+ (GPU训练)
- **内存**: 16GB+ (推荐32GB)
- **显存**: 8GB+ (单GPU训练)

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd Sparse2Full

# 创建conda环境
conda create -n sparse2full python=3.12
conda activate sparse2full

# 安装PyTorch (根据CUDA版本选择)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 数据准备

1. **下载PDEBench数据集**：
```bash
# 创建数据目录
mkdir -p data/pdebench

# 下载示例数据（实际使用时需要下载完整数据集）
# 请参考 https://github.com/pdebench/PDEBench 获取完整数据集
```

2. **数据格式**：
   - 支持HDF5格式的PDEBench数据
   - 数据维度：`[N, C, H, W]` (批次, 通道, 高度, 宽度)
   - 支持多物理场变量（速度、压力、温度等）

### 基础使用

#### 1. 训练模型

```bash
# 超分辨率任务 (4x)
python tools/train.py \
    --config configs/sr_swin_unet.yaml \
    --data.root data/pdebench \
    --data.task sr \
    --data.scale_factor 4 \
    --model.type swin_unet \
    --train.epochs 100 \
    --train.batch_size 16

# 裁剪重建任务
python tools/train.py \
    --config configs/crop_hybrid.yaml \
    --data.root data/pdebench \
    --data.task crop \
    --data.crop_ratio 0.2 \
    --model.type hybrid \
    --train.epochs 100 \
    --train.batch_size 32
```

#### 2. 评估模型

```bash
# 评估训练好的模型
python tools/eval.py \
    --config runs/SR-PDEBench-256-SwinUNet-s2025/config.yaml \
    --checkpoint runs/SR-PDEBench-256-SwinUNet-s2025/best.pth \
    --data.split test \
    --output results/eval_results.json

# 生成可视化结果
python tools/eval.py \
    --config runs/SR-PDEBench-256-SwinUNet-s2025/config.yaml \
    --checkpoint runs/SR-PDEBench-256-SwinUNet-s2025/best.pth \
    --visualize \
    --output_dir results/visualizations/
```

#### 3. 模型基准测试

```bash
# 创建示例配置
python tools/benchmark_models.py --create_configs

# 运行基准测试
python tools/benchmark_models.py \
    --config_dir configs \
    --output benchmark_results.json \
    --num_benchmark 100

# 查看基准测试报告
cat benchmark_report.md
```

#### 4. 生成论文材料

```bash
# 生成完整的论文材料包
python tools/generate_paper_package.py \
    --runs_dir runs \
    --output_dir paper_package \
    --anonymous false

# 生成匿名版本（用于盲审）
python tools/generate_paper_package.py \
    --runs_dir runs \
    --output_dir paper_package_anonymous \
    --anonymous true
```

## 🔧 高级使用

### 分布式训练

```bash
# 单机多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py \
    --config configs/sr_swin_unet.yaml \
    --distributed

# 多机多GPU训练
python -m torch.distributed.launch \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=12345 \
    --nproc_per_node=4 \
    tools/train.py \
    --config configs/sr_swin_unet.yaml \
    --distributed
```

### 自定义配置

1. **创建新的模型配置**：
```yaml
# configs/my_model.yaml
model:
  type: swin_unet
  embed_dim: 128
  depths: [2, 2, 6, 2]
  num_heads: [4, 8, 16, 32]
  window_size: 8

data:
  name: pdebench
  task: sr
  scale_factor: 4
  batch_size: 16

train:
  epochs: 200
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: cosine
```

2. **自定义损失函数**：
```python
# ops/loss.py
class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target, degraded):
        # 重建损失
        recon_loss = F.mse_loss(pred, target)
        
        # 一致性损失
        consistency_loss = F.mse_loss(
            apply_degradation_operator(pred), 
            degraded
        )
        
        return self.alpha * recon_loss + self.beta * consistency_loss
```

### 可视化工具

```python
from utils.visualization import PDEBenchVisualizer

# 创建可视化器
visualizer = PDEBenchVisualizer(
    save_dir="visualizations",
    dpi=300,
    figsize=(12, 8)
)

# 生成对比图
visualizer.plot_field_comparison(
    gt=ground_truth,
    pred=prediction,
    degraded=degraded_input,
    title="超分辨率重建结果",
    save_name="sr_comparison.png"
)

# 生成功率谱分析
visualizer.plot_power_spectrum(
    data=prediction,
    title="功率谱分析",
    save_name="power_spectrum.png"
)

# 生成误差分析
visualizer.plot_error_analysis(
    gt=ground_truth,
    pred=prediction,
    save_name="error_analysis.png"
)
```

## 📊 评估指标

系统支持多种评估指标：

### 主要指标
- **Rel-L2**: 相对L2误差 `||pred - gt||_2 / ||gt||_2`
- **MAE**: 平均绝对误差 `mean(|pred - gt|)`
- **PSNR**: 峰值信噪比 `20 * log10(max_val / sqrt(MSE))`
- **SSIM**: 结构相似性指数

### 频域指标
- **fRMSE-low**: 低频均方根误差
- **fRMSE-mid**: 中频均方根误差  
- **fRMSE-high**: 高频均方根误差

### 边界指标
- **bRMSE**: 边界区域均方根误差
- **cRMSE**: 中心区域均方根误差

### 一致性指标
- **||H(ŷ) - y||**: 观测一致性误差

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行单元测试
python -m pytest tests/unit/ -v

# 运行集成测试
python -m pytest tests/integration/ -v

# 运行端到端测试
python tests/test_simple_e2e.py

# 运行综合端到端测试
python tests/test_e2e_comprehensive.py

# 运行系统集成测试
python tests/test_system_integration.py

# 测试覆盖率
python -m pytest tests/ --cov=. --cov-report=html
```

## 📈 实验管理

### 实验命名规范

实验名格式：`<task>-<data>-<res>-<model>-<keyhyper>-<seed>-<date>`

示例：
- `SRx4-PDEBench-256-SwinUNet_w8d2262_m16-s2025-20251011`
- `Crop20-PDEBench-128-Hybrid_e64-s2025-20251011`

### 实验目录结构

```
runs/
└── SRx4-PDEBench-256-SwinUNet-s2025-20251011/
    ├── config_merged.yaml      # 完整配置快照
    ├── train.log              # 训练日志
    ├── metrics.jsonl          # 逐步指标记录
    ├── checkpoints/           # 模型检查点
    │   ├── best.pth          # 最佳模型
    │   ├── latest.pth        # 最新模型
    │   └── epoch_*.pth       # 定期保存
    ├── visualizations/        # 可视化结果
    └── tensorboard/          # TensorBoard日志
```

## 🔬 开发指南

### 代码规范

- **语言**: Python 3.10+
- **格式化**: `black` + `isort`
- **类型检查**: `mypy --strict`
- **测试**: `pytest`
- **文档**: Google风格docstring

### 提交规范

```bash
# 格式: <scope>: <summary>
git commit -m "model: add Swin-UNet implementation"
git commit -m "data: support PDEBench crop task"
git commit -m "fix: resolve CUDA memory leak in training"
```

### 添加新模型

1. **实现模型类**：
```python
# models/my_model.py
class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # 模型实现
        
    def forward(self, x):
        # 前向传播
        return output
```

2. **注册模型**：
```python
# models/__init__.py
from .my_model import MyModel

MODEL_REGISTRY = {
    'my_model': MyModel,
    # 其他模型...
}
```

3. **添加配置**：
```yaml
# configs/my_model.yaml
model:
  type: my_model
  # 模型参数...
```

4. **添加测试**：
```python
# tests/unit/test_my_model.py
def test_my_model():
    model = MyModel(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 3, 64, 64)
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**：
   - 减小batch_size
   - 使用梯度累积
   - 启用混合精度训练

2. **训练不收敛**：
   - 检查学习率设置
   - 验证数据预处理
   - 确认损失函数权重

3. **分布式训练失败**：
   - 检查网络连接
   - 验证端口可用性
   - 确认NCCL版本兼容

4. **可视化异常**：
   - 检查matplotlib后端
   - 验证数据范围和类型
   - 确认保存路径权限

### 调试技巧

```python
# 启用调试模式
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 检查模型梯度
python tools/debug_gradients.py --config configs/debug.yaml

# 可视化数据流
python tools/visualize_data_flow.py --config configs/debug.yaml
```

## 🏗️ 系统架构

### 核心设计原则

1. **一致性优先**: 观测算子H与训练DC必须复用同一实现与配置
2. **可复现**: 同一YAML+种子，验证指标方差≤1e-4
3. **统一接口**: 所有模型`forward(x[B,C_in,H,W])→y[B,C_out,H,W]`
4. **可比性**: 横向对比必须报告均值±标准差（≥3种子）+资源成本
5. **文档先行**: 新增任务/算子/模型前，先提交PRD/技术文档补丁

### 模块依赖关系

```
tools/train.py
    ├── models/          # 模型架构
    ├── datasets/        # 数据加载
    ├── ops/            # 核心算子
    │   ├── degradation.py  # 观测算子H
    │   ├── loss.py        # 损失函数
    │   └── metrics.py     # 评估指标
    ├── utils/          # 工具模块
    │   ├── distributed.py # 分布式训练
    │   └── config.py     # 配置管理
    └── configs/        # 配置文件
```

### 数据流程

```
原始数据 → 数据加载器 → 观测算子H → 模型推理 → 损失计算 → 反向传播
    ↓           ↓           ↓          ↓         ↓
  PDEBench   DataLoader  degraded   prediction  loss
```

## 📚 参考文献

如果您使用本项目，请引用：

```bibtex
@article{sparse2full2025,
  title={PDEBench稀疏观测重建系统: 基于深度学习的偏微分方程重建方法},
  author={PDEBench Team},
  journal={arXiv preprint},
  year={2025}
}
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 贡献类型

- 🐛 Bug修复
- ✨ 新功能
- 📚 文档改进
- 🎨 代码优化
- 🧪 测试增强
- 🔧 工具改进

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [PDEBench](https://github.com/pdebench/PDEBench) - 数据集和基准
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Hydra](https://hydra.cc/) - 配置管理
- [Weights & Biases](https://wandb.ai/) - 实验跟踪

## 📞 联系方式

- **项目主页**: [GitHub Repository](https://github.com/your-org/Sparse2Full)
- **问题反馈**: [GitHub Issues](https://github.com/your-org/Sparse2Full/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/your-org/Sparse2Full/discussions)

---

<div align="center">
  <strong>🚀 让稀疏观测重建更简单、更高效！</strong>
</div>