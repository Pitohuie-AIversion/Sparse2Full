# PDEBench模型性能指标报告

## 📊 报告概述

本目录包含PDEBench稀疏观测重建系统的完整性能评估报告，基于2025年10月15日的批量训练实验结果。

## 📁 文件结构

```
paper_package/metrics/
├── README.md                                    # 本说明文件
├── original_model_ranking.csv                  # 原始模型排名数据
├── enhanced_model_comparison.csv               # 增强的模型对比数据
├── comprehensive_model_comparison.xlsx         # Excel格式完整对比表
├── model_comparison_table.md                   # Markdown格式对比表
├── comparison_summary.json                     # 对比总结JSON数据
├── comprehensive_performance_report.html       # 综合性能HTML报告
├── performance_radar_chart.html                # 性能雷达图
├── performance_scatter_matrix.html             # 性能散点矩阵图
└── composite_score_chart.html                  # 综合评分图表
```

## 🎯 主要发现

### 模型性能排名（按Rel-L2误差）

1. **🥇 FNO2D** - Rel-L2: 0.01215 (最佳)
   - PSNR: 43.88 dB
   - 训练时间: 0.76 分钟
   - 推荐用于高精度科研项目

2. **🥈 SwinUNet** - Rel-L2: 0.0325
   - PSNR: 35.31 dB  
   - 训练时间: 3.05 分钟
   - 推荐用于平衡精度与通用性的应用

3. **🥉 UNet** - Rel-L2: 0.03775
   - PSNR: 34.00 dB
   - 训练时间: 1.1 分钟
   - 推荐用于快速原型开发

4. **MLP** - Rel-L2: 0.0472
   - PSNR: 32.06 dB
   - 训练时间: 4.25 分钟
   - 适合资源受限环境

### 失败模型

- **SegFormer**: 配置问题导致训练失败 (dim 64不能被num_heads 3整除)
- **SegFormer_UNetFormer**: 同样的配置问题

## 📈 关键指标说明

- **Rel-L2**: 相对L2误差，越小越好
- **MAE**: 平均绝对误差，越小越好  
- **PSNR**: 峰值信噪比，越大越好
- **SSIM**: 结构相似性指数，越接近1越好
- **BRMSE**: 边界区域均方根误差
- **CRMSE**: 中心区域均方根误差

## 🔍 使用建议

### 按应用场景选择模型

- **🔬 科研项目**: FNO2D (最高精度)
- **🏭 工业应用**: SwinUNet (精度与效率平衡)
- **🚀 快速原型**: UNet (简单快速)
- **📱 边缘计算**: MLP (参数少，推理快)

### 性能权衡

- **精度优先**: FNO2D > SwinUNet > UNet > MLP
- **速度优先**: FNO2D > UNet > SwinUNet > MLP
- **资源优先**: MLP > UNet > FNO2D > SwinUNet

## 📊 报告查看方式

1. **快速查看**: 打开 `model_comparison_table.md`
2. **详细分析**: 打开 `comprehensive_model_comparison.xlsx`
3. **交互式报告**: 在浏览器中打开 `comprehensive_performance_report.html`
4. **图表分析**: 查看各种HTML图表文件

## 🔧 技术细节

- **训练配置**: 稳定配置 (AdamW, lr=1e-3, Cosine调度)
- **损失函数**: L_rec + 0.5*L_spec + 1.0*L_dc
- **数据集**: PDEBench 2D扩散反应方程
- **任务类型**: 超分辨率重建 (4x)
- **评估指标**: 8项综合指标

## 📅 更新记录

- **2025-10-15**: 初始批量训练实验完成
- **2025-10-15**: 生成完整性能报告和可视化
- **2025-10-15**: 整理到paper_package/metrics目录

## 📞 联系信息

如有问题或需要更多信息，请参考项目文档或联系开发团队。

---

*本报告由PDEBench稀疏观测重建系统自动生成*