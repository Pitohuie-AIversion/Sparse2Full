# 🚀 横向对比系统使用指南

## 系统概述

您的训练系统现在具备了完整的**横向对比能力**，支持多模型、多种子的自动化对比实验，生成论文级别的统计分析报告。

## 📋 主要功能

✅ **44个可用模型** - 从SwinUNet到传统UNet的全面对比  
✅ **多种子支持** - 确保统计显著性（推荐≥3个种子）  
✅ **自动化训练** - 批量运行所有对比实验  
✅ **智能汇总** - 自动生成论文级对比报告  
✅ **显著性检验** - Paired t-test + Cohen's d  
✅ **资源对比** - 参数量、FLOPs、显存、延迟统计  

## 🎯 快速开始

### 1. 查看可用模型
```bash
python -c "from tools.training.model_loader import list_models; print('可用模型:', len(list_models()), '个')"
```

### 2. 快速对比（推荐新手）
对比3个主流模型，每个模型2个种子：
```bash
python tools/batch_comparison.py \
    --config configs/train/ar_training_config\ debug.yaml \
    --models swin_unet unet fno2d \
    --seeds 42,123 \
    --output paper_package/quick_comparison/
```

### 3. 完整对比（研究级）
对比所有模型，每个模型5个种子：
```bash
python tools/batch_comparison.py \
    --config configs/train/ar_comparison_config.yaml \
    --all-models \
    --seeds 5 \
    --output paper_package/full_comparison/
```

### 4. 自定义对比
选择特定模型进行深度对比：
```bash
python tools/batch_comparison.py \
    --config configs/train/ar_comparison_config.yaml \
    --models swin_unet unet segformer unet_plus_plus hybrid \
    --seeds 42,123,456,789 \
    --output paper_package/custom_comparison/
```

## 📊 输出结果

### 主要输出文件

运行完成后，您将获得以下文件：

```
paper_package/comparison/
├── batch_results.json          # 详细实验结果
├── batch_comparison.log        # 实验日志
├── comparison_results/         # 对比分析报告
│   ├── comparison_report.md   # 综合对比报告
│   ├── comparison_results.csv # 原始数据表格
│   ├── main_comparison_table.tex  # 主要结果对比表
│   ├── resource_comparison_table.tex # 资源消耗对比表
│   ├── model_ranking_table.tex     # 模型排名表
│   └── significance_summary_table.tex # 显著性检验汇总
└── summary/                    # 汇总结果
    ├── aggregated_results.json
    ├── significance_results.json
    └── *.tex                   # LaTeX表格文件
```

### 关键指标

**主要性能指标：**
- **Rel-L2**: 相对L2误差 (越小越好)
- **MAE**: 平均绝对误差 (越小越好)  
- **PSNR**: 峰值信噪比 (越大越好)
- **SSIM**: 结构相似性 (越大越好)

**资源消耗指标：**
- **参数量(M)**: 模型参数数量
- **FLOPs(G)**: 浮点运算次数
- **显存(GB)**: GPU内存占用
- **延迟(ms)**: 推理延迟

## 🔧 高级用法

### 使用专用对比配置

我们为您准备了专门的对比实验配置：

```bash
# 使用对比专用配置（推荐）
python tools/batch_comparison.py \
    --config configs/train/ar_comparison_config.yaml \
    --models swin_unet unet fno2d segformer \
    --seeds 42,123,456 \
    --output paper_package/study_comparison/
```

对比配置特点：
- ✅ **统一数据分割** - 确保公平对比
- ✅ **相同训练设置** - 统一优化器、学习率等
- ✅ **禁用数据增强** - 减少随机性影响
- ✅ **启用完整测试** - 获得全面评估
- ✅ **资源统计** - 自动记录资源消耗

### 分阶段对比策略

对于大规模对比，建议分阶段进行：

```bash
# 阶段1: 基线模型对比
python tools/batch_comparison.py \
    --config configs/train/ar_comparison_config.yaml \
    --models unet swin_unet \
    --seeds 42,123,456,789,1012 \
    --output paper_package/phase1_baseline/

# 阶段2: 先进模型对比
python tools/batch_comparison.py \
    --config configs/train/ar_comparison_config.yaml \
    --models fno2d segformer unet_plus_plus hybrid \
    --seeds 42,123,456,789,1012 \
    --output paper_package/phase2_advanced/

# 阶段3: 汇总所有结果
python tools/enhanced_summarize.py \
    --runs_dir runs/ \
    --baseline_method unet \
    --output paper_package/final_comparison/
```

### 自定义种子策略

```bash
# 少量种子快速验证
--seeds 42,123

# 标准统计要求 (推荐)
--seeds 42,123,456

# 高置信度要求
--seeds 42,123,456,789,1012

# 指定种子个数 (自动生成连续种子)
--seeds 5  # 生成 42,43,44,45,46
```

## 📈 结果解读

### 如何阅读对比报告

1. **主要结果表**: 查看Rel-L2指标，值越小性能越好
2. **模型排名**: 综合多个指标的整体排名
3. **显著性检验**: 
   - `+` 表示相比基线有显著改进
   - `-` 表示相比基线性能显著下降
   - 空白表示无显著差异
4. **效应量(Cohen's d)**: 
   - 0.2-0.5: 小效应
   - 0.5-0.8: 中等效应  
   - >0.8: 大效应

### 选择建议

**性能优先**: 选择Rel-L2最低的模型  
**效率优先**: 选择资源消耗最小的模型  
**平衡选择**: 综合考虑性能和资源消耗  

## 🛠️ 故障排除

### 常见问题

**1. 模型加载失败**
```bash
# 检查模型是否可用
python -c "from tools.training.model_loader import list_models; print(list_models())"
```

**2. 内存不足**
- 减少batch size: 修改配置文件中的 `training.batch_size`
- 减少同时运行的模型数量
- 使用更小的图像尺寸

**3. 训练时间过长**
- 使用更少的种子数: `--seeds 42,123`
- 减少训练轮数: 修改配置文件中的 `training.epochs`
- 使用快速测试模式

**4. 结果汇总失败**
- 确保所有实验都成功完成
- 检查实验目录中是否有 `metrics_summary.json` 文件
- 查看日志文件了解详细错误

### 性能优化建议

1. **GPU使用**: 确保所有GPU都被充分利用
2. **数据加载**: 调整 `num_workers` 参数优化数据加载
3. **混合精度**: 确保AMP已启用以加速训练
4. **检查点**: 合理设置检查点保存频率

## 📝 最佳实践

### 实验设计建议

1. **种子数量**: 至少3个种子，推荐5个用于发表
2. **基线选择**: 选择领域标准模型作为基线(如UNet)
3. **公平对比**: 使用相同的数据分割和训练设置
4. **资源记录**: 记录所有模型的资源消耗
5. **统计检验**: 进行显著性检验验证改进

### 论文写作提示

1. **方法描述**: 详细描述对比实验设置
2. **结果呈现**: 使用表格展示主要结果
3. **统计分析**: 报告均值±标准差
4. **显著性标注**: 标注统计显著性
5. **资源对比**: 包含计算复杂度分析

## 🔗 相关脚本

- **批量训练**: `tools/batch_comparison.py`
- **增强汇总**: `tools/enhanced_summarize.py`  
- **标准汇总**: `tools/summarize_runs.py`
- **模型加载**: `tools/training/model_loader.py`
- **对比配置**: `configs/train/ar_comparison_config.yaml`

## 📞 支持

如遇到问题，请检查：
1. 日志文件中的详细错误信息
2. 确保所有依赖项已正确安装
3. 验证配置文件格式正确
4. 检查GPU和内存资源充足

---

**祝您实验顺利！🎉**