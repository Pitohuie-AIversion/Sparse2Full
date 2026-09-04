# AR训练脚本可视化集成指南

## 🎯 概述

AR训练脚本 (`train_real_data_ar.py`) 现已成功集成完整的可视化功能。在训练完成后，脚本会自动生成comprehensive的测试可视化报告。

## ✨ 新增功能

### 1. 自动测试可视化生成
- 训练完成后自动调用 `create_test_visualizations()` 方法
- 无需手动干预，完全自动化

### 2. 完整的可视化内容
- **AR预测序列可视化**: 展示模型的自回归预测结果
- **误差分析图表**: 详细的预测误差分布和统计
- **时间演化分析**: 时序预测的动态变化过程
- **综合HTML报告**: 整合所有可视化结果的交互式报告

### 3. 智能文件管理
- 可视化文件保存到 `runs/<experiment_name>/test_visualizations/`
- 自动复制到 `paper_package/figs/` 目录用于论文
- 支持多种格式输出 (PNG, HTML, JSON)

## 🚀 使用方法

### 基本使用
```bash
cd /path/to/Sparse2Full
python tools/training/train_real_data_ar.py
```

### 使用自定义配置
```bash
python tools/training/train_real_data_ar.py --config configs/ar_training_config.yaml
```

### 从检查点恢复训练
```bash
python tools/training/train_real_data_ar.py --resume runs/experiment_name/checkpoints/best_model.pth
```

## 📊 生成的可视化文件

训练完成后，你会在以下位置找到可视化文件：

```
runs/<experiment_name>/
├── test_visualizations/
│   ├── ar_predictions/
│   │   ├── ar_prediction_sample_0.png
│   │   ├── ar_prediction_sample_1.png
│   │   └── ...
│   ├── error_analysis/
│   │   ├── error_analysis_sample_0.png
│   │   ├── error_analysis_sample_1.png
│   │   └── ...
│   ├── temporal_analysis/
│   │   ├── temporal_analysis_sample_0.png
│   │   ├── temporal_analysis_sample_1.png
│   │   └── ...
│   └── test_visualization_report.html
└── test_results.json

paper_package/figs/<experiment_name>_test/
├── (所有可视化文件的副本)
└── test_visualization_report.html
```

## 🔧 技术细节

### 集成的可视化组件
1. **ARTrainingVisualizer**: 专门用于AR模型的可视化
2. **PDEBenchVisualizer**: 统一的PDE基准可视化接口
3. **误差分析修复**: 解决了之前的图像形状问题

### 关键方法
- `create_test_visualizations(test_metrics)`: 生成测试阶段可视化
- `create_final_report()`: 生成最终训练报告
- `create_ar_prediction_visualization()`: AR预测可视化
- `create_error_analysis()`: 误差分析可视化
- `create_temporal_analysis()`: 时间分析可视化

### 配置选项
可视化功能通过以下配置控制：
```yaml
visualization:
  enabled: true
  save_test_visualizations: true
  num_test_samples: 5
  formats: ['png', 'html']
```

## 🐛 故障排除

### 常见问题

1. **可视化模块导入失败**
   ```
   Warning: Visualization modules not available
   ```
   **解决方案**: 确保所有依赖包已安装：
   ```bash
   pip install matplotlib seaborn plotly
   ```

2. **内存不足错误**
   ```
   CUDA out of memory
   ```
   **解决方案**: 减少批次大小或可视化样本数量

3. **文件权限错误**
   ```
   Permission denied: paper_package/figs
   ```
   **解决方案**: 确保有写入权限：
   ```bash
   chmod -R 755 paper_package/
   ```

### 调试模式
启用详细日志记录：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 性能优化

### 可视化性能建议
1. **限制可视化样本数**: 默认5个样本，可根据需要调整
2. **选择合适的图像分辨率**: 平衡质量和文件大小
3. **使用GPU加速**: 确保CUDA可用以加速计算

### 存储优化
1. **定期清理**: 删除旧的可视化文件
2. **压缩输出**: 使用PNG格式减少文件大小
3. **选择性保存**: 只保存关键的可视化结果

## 🎨 自定义可视化

### 添加新的可视化类型
```python
def create_custom_visualization(self, data, save_name="custom"):
    # 自定义可视化逻辑
    fig, ax = plt.subplots(figsize=(10, 8))
    # ... 绘图代码 ...
    
    save_path = self.vis_dir / "custom" / f"{save_name}.png"
    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
```

### 修改可视化样式
```python
# 在可视化方法中自定义样式
plt.style.use('seaborn-v0_8')  # 使用seaborn样式
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
```

## 📚 相关文档

- [AR模型训练指南](docs/ar_training.md)
- [可视化API参考](docs/visualization_api.md)
- [配置文件说明](docs/configuration.md)

## 🤝 贡献

如果你发现问题或有改进建议，请：
1. 创建Issue描述问题
2. 提交Pull Request
3. 更新相关文档

---

**最后更新**: 2025-11-01  
**版本**: v1.0.0  
**状态**: ✅ 已完成集成并测试通过
