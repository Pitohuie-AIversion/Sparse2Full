# 重构训练脚本验证指南

## 概述

本指南提供了验证重构训练脚本 `train_real_data_ar_refactored.py` 的完整流程，确保其与原始脚本 `train_real_data_ar.py` 在功能、性能和结果上的一致性。

## 验证目标

1. **功能一致性**: 重构版本与原始版本产生相同的训练结果
2. **性能优化**: 重构版本在保持精度的同时提供更好的性能
3. **代码质量**: 重构代码符合最佳实践和质量标准
4. **可维护性**: 模块化设计便于后续维护和扩展

## 验证流程

### 阶段1：环境准备

#### 1.1 环境检查

```bash
# 检查Python版本
python --version  # 要求 >= 3.8

# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"  # 要求 >= 1.9

# 检查依赖包
pip list | grep -E "(torch|numpy|matplotlib|tensorboard|pyyaml|tqdm)"
```

#### 1.2 数据准备

```bash
# 准备测试数据集
mkdir -p test_data/{train,val,test}

# 生成或下载测试数据
python tools/generate_test_data.py \
    --output_dir test_data \
    --num_samples 100 \
    --image_size 256
```

#### 1.3 配置准备

```bash
# 创建测试配置
cp configs/ar_training_refactored_config.yaml configs/test_config.yaml

# 修改测试配置
sed -i 's/dataset_path:.*/dataset_path: "test_data"/' configs/test_config.yaml
sed -i 's/epochs:.*/epochs: 5/' configs/test_config.yaml
sed -i 's/batch_size:.*/batch_size: 4/' configs/test_config.yaml
```

### 阶段2：功能验证

#### 2.1 配置验证

```bash
# 运行配置验证
python tools/validate_refactored_script.py \
    --script tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --level comprehensive \
    --output validation_report.json
```

**验证要点**:
- 配置文件格式正确性
- 参数范围和类型检查
- 依赖关系一致性
- 默认值合理性

#### 2.2 代码质量验证

```bash
# 代码风格检查
python -m black --check tools/training/train_real_data_ar_refactored.py
python -m flake8 tools/training/train_real_data_ar_refactored.py
python -m mypy tools/training/train_real_data_ar_refactored.py

# 复杂度分析
python -m radon cc tools/training/train_real_data_ar_refactored.py
python -m radon mi tools/training/train_real_data_ar_refactored.py
```

**质量标准**:
- 代码风格一致性（Black, PEP8）
- 类型注解完整性（mypy）
- 圈复杂度 < 10
- 维护性指数 > 80

#### 2.3 单元测试

```bash
# 运行单元测试
python -m pytest tests/test_train_real_data_ar_refactored.py -v

# 覆盖率报告
python -m pytest tests/test_train_real_data_ar_refactored.py \
    --cov=tools.training.train_real_data_ar_refactored \
    --cov-report=html
```

**测试覆盖率要求**:
- 语句覆盖率 > 90%
- 分支覆盖率 > 85%
- 关键函数覆盖率 = 100%

### 阶段3：一致性验证

#### 3.1 数值一致性测试

```bash
# 运行一致性测试
python tools/test_integration_consistency.py \
    --original_script tools/training/train_real_data_ar.py \
    --refactored_script tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --output consistency_report.json \
    --tolerance 1e-6
```

**一致性检查项**:
- 模型输出一致性（误差 < 1e-6）
- 损失函数一致性（误差 < 1e-8）
- 梯度计算一致性（误差 < 1e-7）
- 优化器状态一致性（误差 < 1e-8）

#### 3.2 训练过程对比

```bash
# 运行原始脚本训练
python tools/training/train_real_data_ar.py \
    --config configs/test_config.yaml \
    --experiment original_baseline \
    --epochs 5

# 运行重构脚本训练
python tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --experiment refactored_version \
    --epochs 5
```

**对比指标**:
- 训练损失曲线一致性
- 验证指标收敛性
- 学习曲线平滑度
- 最终模型性能

#### 3.3 检查点兼容性

```bash
# 从原始检查点恢复训练
python tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --experiment resume_from_original \
    --resume runs/original_baseline/checkpoints/best.pth

# 验证恢复后的训练一致性
```

### 阶段4：性能验证

#### 4.1 基准性能测试

```bash
# 运行性能基准测试
python tools/benchmark_refactored_script.py \
    --script tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --output benchmark_report.json \
    --suite all
```

**性能指标**:
- 训练速度（样本/秒）
- 内存使用峰值（GB）
- GPU利用率（%）
- 启动时间（秒）

#### 4.2 性能对比分析

```bash
# 运行性能对比
python tools/benchmark_performance_comparison.py \
    --original_script tools/training/train_real_data_ar.py \
    --refactored_script tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --output comparison_report.json \
    --suite standard
```

**性能改进目标**:
- 训练速度提升 ≥ 15%
- 内存使用降低 ≥ 10%
- 验证速度提升 ≥ 30%
- 启动时间减少 ≥ 50%

### 阶段5：稳定性验证

#### 5.1 长时间运行测试

```bash
# 长时间训练测试（24小时）
python tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --experiment long_running_test \
    --epochs 1000 \
    --training.early_stopping.patience 100
```

**稳定性检查**:
- 内存泄漏检测
- 梯度爆炸/消失检测
- 数值稳定性
- 异常恢复能力

#### 5.2 边界条件测试

```python
# 测试极端配置
extreme_configs = [
    {"batch_size": 1, "learning_rate": 1e-8},
    {"batch_size": 128, "learning_rate": 1.0},
    {"image_size": [32, 32]},
    {"image_size": [1024, 1024]},
    {"num_channels": 1},
    {"num_channels": 64},
]

for config in extreme_configs:
    # 运行测试
    pass
```

### 阶段6：集成验证

#### 6.1 端到端测试

```bash
# 运行完整测试套件
python tools/run_all_tests.py \
    --output integration_test_report \
    --verbose
```

#### 6.2 真实数据验证

```bash
# 使用真实数据集进行验证
python tools/training/train_real_data_ar_refactored.py \
    --config configs/ar_training_refactored_config.yaml \
    --experiment real_data_validation \
    --data.dataset_path "path/to/real/dataset"
```

## 验证标准

### 功能正确性标准

| 检查项 | 要求 | 容差 |
|--------|------|------|
| 模型输出一致性 | ≥ 99.9% | 1e-6 |
| 损失函数一致性 | ≥ 99.99% | 1e-8 |
| 梯度计算一致性 | ≥ 99.9% | 1e-7 |
| 优化器行为一致性 | ≥ 99.9% | 1e-8 |
| 学习曲线一致性 | ≥ 95% | 视觉检查 |

### 性能标准

| 指标 | 目标改进 | 最低要求 |
|------|----------|----------|
| 训练速度 | +20% | +10% |
| 内存使用 | -15% | -5% |
| 验证速度 | +50% | +30% |
| 启动时间 | -60% | -30% |
| 检查点保存 | -40% | -20% |

### 代码质量标准

| 指标 | 目标值 | 最低要求 |
|------|--------|----------|
| 语句覆盖率 | 95% | 90% |
| 分支覆盖率 | 90% | 85% |
| 复杂度（平均） | < 8 | < 10 |
| 维护性指数 | > 85 | > 80 |
| 代码风格合规 | 100% | 95% |

### 稳定性标准

| 检查项 | 要求 |
|--------|------|
| 24小时训练无崩溃 | 必需 |
| 内存泄漏检测 | 零泄漏 |
| 梯度稳定性 | 无NaN/Inf |
| 数值稳定性 | 误差 < 1e-6 |
| 异常恢复 | 自动恢复 |

## 自动化验证

### 验证脚本

```bash
#!/bin/bash
# automated_validation.sh

set -e

echo "开始自动化验证..."

# 阶段1: 环境检查
echo "阶段1: 环境检查"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"

# 阶段2: 代码质量
echo "阶段2: 代码质量检查"
python -m black --check tools/training/train_real_data_ar_refactored.py
python -m flake8 tools/training/train_real_data_ar_refactored.py
python -m mypy tools/training/train_real_data_ar_refactored.py

# 阶段3: 单元测试
echo "阶段3: 单元测试"
python -m pytest tests/test_train_real_data_ar_refactored.py -v --tb=short

# 阶段4: 一致性测试
echo "阶段4: 一致性测试"
python tools/test_integration_consistency.py \
    --original_script tools/training/train_real_data_ar.py \
    --refactored_script tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --output consistency_report.json

# 阶段5: 性能测试
echo "阶段5: 性能测试"
python tools/benchmark_performance_comparison.py \
    --original_script tools/training/train_real_data_ar.py \
    --refactored_script tools/training/train_real_data_ar_refactored.py \
    --config configs/test_config.yaml \
    --output performance_comparison.json

# 阶段6: 综合报告
echo "阶段6: 生成综合报告"
python tools/generate_validation_report.py \
    --consistency_report consistency_report.json \
    --performance_report performance_comparison.json \
    --output validation_summary.json

echo "自动化验证完成!"
```

### 持续集成

```yaml
# .github/workflows/validation.yml
name: Refactored Script Validation

on:
  push:
    paths:
      - 'tools/training/train_real_data_ar_refactored.py'
      - 'tests/test_train_real_data_ar_refactored.py'
  pull_request:
    paths:
      - 'tools/training/train_real_data_ar_refactored.py'

jobs:
  validation:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8 mypy
    
    - name: Code quality check
      run: |
        python -m black --check tools/training/train_real_data_ar_refactored.py
        python -m flake8 tools/training/train_real_data_ar_refactored.py
        python -m mypy tools/training/train_real_data_ar_refactored.py
    
    - name: Unit tests
      run: |
        python -m pytest tests/test_train_real_data_ar_refactored.py -v \
          --cov=tools.training.train_real_data_ar_refactored \
          --cov-report=xml
    
    - name: Integration tests
      run: |
        python tools/test_integration_consistency.py \
          --original_script tools/training/train_real_data_ar.py \
          --refactored_script tools/training/train_real_data_ar_refactored.py \
          --config configs/test_config.yaml
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## 验证报告

### 报告结构

验证完成后会生成以下报告：

```
validation_report/
├── code_quality_report.json      # 代码质量报告
├── unit_test_report.json         # 单元测试报告
├── consistency_report.json       # 一致性测试报告
├── performance_comparison.json  # 性能对比报告
├── stability_report.json        # 稳定性测试报告
├── integration_report.json      # 集成测试报告
└── validation_summary.md        # 综合验证总结
```

### 报告解读

#### 一致性报告示例

```json
{
  "consistency_score": 0.998,
  "model_output_consistency": 0.9998,
  "loss_consistency": 0.9999,
  "gradient_consistency": 0.9991,
  "optimizer_consistency": 0.9997,
  "failed_tests": [],
  "warnings": ["Minor difference in random initialization"],
  "recommendations": ["Consider seeding random number generators"]
}
```

#### 性能报告示例

```json
{
  "performance_improvement": {
    "training_speed": 1.23,
    "memory_efficiency": 1.18,
    "validation_speed": 1.67,
    "startup_time": 2.1
  },
  "resource_usage": {
    "peak_memory_original": 8.5,
    "peak_memory_refactored": 7.2,
    "gpu_utilization_original": 85,
    "gpu_utilization_refactored": 92
  }
}
```

## 常见问题

### 1. 一致性测试失败

**症状**: 数值一致性测试失败

**可能原因**:
- 随机数生成器种子不同
- 浮点运算顺序差异
- 数值精度问题
- 并行计算非确定性

**解决方案**:
```python
# 确保随机种子一致性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 启用确定性计算
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 2. 性能测试异常

**症状**: 重构版本性能不如预期

**可能原因**:
- 配置参数未优化
- 硬件环境差异
- 数据加载瓶颈
- 内存分配问题

**解决方案**:
- 检查数据加载器配置
- 优化批处理大小
- 启用混合精度训练
- 使用性能分析工具

### 3. 内存使用过高

**症状**: 重构版本内存使用增加

**可能原因**:
- 缓存机制不当
- 对象生命周期管理
- 内存泄漏
- 不必要的复制操作

**解决方案**:
```python
# 启用内存监控
import tracemalloc
tracemalloc.start()

# 定期清理缓存
import gc
gc.collect()
torch.cuda.empty_cache()
```

## 验证最佳实践

### 1. 分层验证

- 先进行单元测试，再进行集成测试
- 先验证功能正确性，再验证性能
- 先使用小数据集，再使用完整数据集

### 2. 自动化验证

- 使用CI/CD自动运行验证
- 设置质量门禁和阈值
- 定期回归测试

### 3. 文档记录

- 记录所有验证结果
- 保存失败的测试用例
- 维护验证历史记录

### 4. 持续改进

- 根据验证结果优化代码
- 更新验证标准和流程
- 改进测试覆盖率

## 结论

通过本指南的验证流程，可以确保重构训练脚本在功能、性能和稳定性方面达到预期标准。验证过程应该是持续的，随着代码的演进不断更新和完善验证用例。

重构脚本的主要优势：
- ✅ 模块化架构，易于维护
- ✅ 性能优化，训练更快
- ✅ 完善的错误处理
- ✅ 更好的可测试性
- ✅ 详细的监控和日志

建议定期运行验证流程，确保代码质量和功能一致性。
## 一致性验证（H 与 DC）

为确保观测算子 H 与训练 DC 完全一致，请在功能验证与一致性验证两个层面执行以下步骤：

### 脚本级验证

当手头有 HDF5 文件（含 `gt/obs`）或真实数据（含 `data`），可运行脚本：

```
python tools/check_dc_equivalence.py --h5 /path/to/case.h5 --tolerance 1e-8
```

或基于 Hydra 配置自动生成 `obs` 再验证：

```
python tools/check_dc_equivalence.py \
  --config configs/ar_training_refactored_config.yaml \
  --tolerance 1e-8
```

脚本输出包含 `mse/max_error/tolerance/passed` 并以退出码指示是否通过。将此结果记录到验证日志中，作为 CI 绿色门槛的一部分。

### 单元测试级验证

在本项目中，已提供如下单测，建议在 CI 中运行并纳入门槛：

- `tests/test_dc_equivalence_script.py`：构造临时 HDF5（含 `gt/obs/params`），调用 `run_check_from_h5` 验证一致性。
- `tests/test_dc_consistency.py`：对 `DataConsistencyChecker` 与 `ops.degradation` 的一致性进行直接检验（任务覆盖 `sr/crop`）。
- `tests/unit/test_ops.py`：包含 `verify_degradation_consistency(...)` 的更细致验证，覆盖尺寸不匹配时的插值修正等边界情形。
- `tests/test_dc_loss.py` 与 `tests/unit/test_losses.py`：验证 DC 损失与 H 算子的一致性（完美一致时损失接近 0）。

运行：

```
pytest -q tests/test_dc_equivalence_script.py tests/test_dc_consistency.py \
  tests/unit/test_ops.py -k consistency
```

通过标准：`MSE(H(GT), y) < 1e-8`；失败用例需在日志中标注原因及修复建议。

## 性能监控与验证流程

为满足性能与资源记录的规范（规则 4、7），建议在验证期间启用统一的监控模块并形成对比报告：

### 监控工具选型

- 推荐：`src/monitoring/performance_monitor.py` 的 `PerformanceMonitor`（集成 `psutil/GPUtil/torch.cuda`）
- 备选：`tools/benchmark_performance_comparison.py` 的 `ResourceMonitor`、`tools/benchmark_refactored_script.py` 的内置监控循环

### 采集指标

- GPU：利用率均值/峰值、显存峰值（GB）、温度（如可用）
- CPU：利用率均值/峰值
- 内存：占用均值/峰值
- I/O：磁盘读写速率（可选）
- 训练：`Params(M)`、`FLOPs(G@256²)`、推理延迟（ms）、AMP/混合精度状态

### 验证步骤

1. 在相同数据与配置下，分别运行原脚本与重构脚本的短程训练（如 1-5 epoch）
2. 对比两者的资源指标与训练曲线（`Rel-L2/||H(ŷ)−y||`），重构脚本应在资源记录更完整且性能不劣于原脚本（目标：显存峰值下降或延迟更低）
3. 在 DDP 环境中验证监控正常工作（仅 `rank=0` 汇总），并记录设备数与 AMP 状态
4. 生成 `metrics.jsonl` 与 `results.md/tex`（如启用 `tools/summarize_runs.py`）

### 示例代码片段

```python
from src.monitoring.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(sample_interval=1.0)
monitor.start_monitoring()

# 运行训练或验证

monitor.stop_monitoring()
stats = monitor.get_summary()
logger.info({
    'gpu_util_mean': stats.get('gpu_util_mean'),
    'gpu_mem_peak_gb': stats.get('gpu_mem_peak_gb'),
    'cpu_util_mean': stats.get('cpu_util_mean')
})
```

请在验证报告中记录上述指标，并给出改进点（如数据加载并发度、梯度累积、AMP 设置、解码器实现）。