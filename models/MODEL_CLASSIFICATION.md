# 模型分类文档

本项目中的深度学习模型已按照功能重新组织为两个主要类别：**空间预测模型**和**时间预测模型**。

## 📁 目录结构

```
models/
├── spatial/                    # 空间预测模型
│   ├── __init__.py
│   ├── factory.py               # 模型工厂函数
│   ├── unet.py                 # U-Net
│   ├── unet_plus_plus.py      # U-Net++
│   ├── fno2d.py               # 2D傅里叶神经算子
│   ├── ufno_unet_bottleneck.py # U-FNO瓶颈
│   ├── segformer.py           # SegFormer
│   ├── unetformer.py          # UNetFormer
│   ├── segformer_unetformer.py # SegFormer-UNetFormer
│   ├── mlp.py                 # MLP模型
│   ├── mlp_mixer.py           # MLP-Mixer
│   ├── liif.py                # LIIF模型
│   ├── liif_head.py           # LIIF头部
│   ├── hybrid.py               # 混合模型
│   ├── swin_unet.py          # SwinUNet
│   ├── vit.py                # Vision Transformer
│   ├── swin_t.py             # Swin Transformer Tiny
│   ├── transformer.py          # 基础Transformer
│   └── sparse_attention_encoder.py # 稀疏注意力编码器
│
├── temporal/                   # 时间预测模型
│   ├── __init__.py
│   ├── factory.py              # 模型工厂函数
│   ├── wrappers/               # 高级包装器
│   │   ├── __init__.py
│   │   ├── swin_temporal.py   # Swin时序包装器
│   │   └── ar_nar_wrapper.py  # AR+NAR包装器
│   └── components/             # 基础组件
│       ├── __init__.py
│       ├── temporal_encoder.py     # 时序编码器
│       ├── temporal_block.py       # 时序块
│       ├── nar_prediction_head.py  # NAR预测头
│       ├── sequential_spatiotemporal.py # 序贯时空
│       ├── sequential_trainer.py     # 序贯训练器
│       └── sequential_dc_consistency.py # DC一致性
│
├── ar/                        # AR模型（保持原有结构）
│   ├── __init__.py
│   └── wrapper.py             # AR包装器
│
├── base.py                    # 基础模型类
├── mlp_model.py              # MLP模型（通用）
├── hybrid_model.py           # 混合模型（通用）
└── baseline_models.py        # 基线模型
```

## 🎯 模型分类

### 空间预测模型 (Spatial Models)

**功能**：处理单帧空间数据，输入输出形状为 `[B,C,H,W]`

| 模型类别 | 模型名称 | 主要特点 | 适用场景 |
|---------|---------|---------|---------|
| **CNN模型** | UNet | 经典编码器-解码器结构 | 基础图像分割/重建 |
| | UNetPlusPlus | 密集跳跃连接 | 改进的U-Net变体 |
| | FNO2d | 傅里叶神经算子 | 频域建模 |
| | UFNOUNet | U-Net + FNO瓶颈 | 混合频域-空域 |
| **Transformer模型** | SegFormer | 高效分割Transformer | 语义分割 |
| | UNetFormer | CNN + Transformer混合 | 医学图像分割 |
| | SegFormerUNetFormer | 双重Transformer融合 | 复杂分割任务 |
| **MLP模型** | MLPModel | 多层感知机 | 简单快速预测 |
| | MLPMixer | MLP-Mixer架构 | 纯MLP解决方案 |
| | LIIFModel | 隐式神经表示 | 连续空间建模 |
| **混合模型** | HybridModel | Attention∥FNO∥UNet组合 | 多模态融合 |
| | SwinUNet | Swin Transformer + U-Net | 先进分割架构 |
| **基础模型** | VisionTransformer | 标准ViT | 图像分类/特征提取 |
| | SwinTransformerTiny | 轻量级Swin | 资源受限环境 |
| | Transformer | 基础Transformer | 通用序列建模 |
| | SparseAttentionEncoder | 稀疏注意力编码器 | 高效注意力计算 |

### 时间预测模型 (Temporal Models)

**功能**：处理时间序列数据，支持多种时间维度接口

| 模型类别 | 模型名称 | 输入输出接口 | 主要特点 |
|---------|---------|-------------|---------|
| **AR包装器** | ARWrapper | `[B,T_in,C,H,W]` → `[B,T_out,C,H,W]` | 将空间模型包装为时间预测 |
| **时序Swin** | SwinTemporal | `[B,T,C,H,W]` → `[B,C,H,W]` | 时间聚合+空间预测 |
| | SwinTemporalNAR | `[B,T_in,C,H,W]` → `[B,T_out,C,H,W]` | 非自回归多步预测 |
| **混合包装器** | ARNARWrapper | 支持AR+NAR混合 | 组合预测策略 |

### 时序组件 (Temporal Components)

**功能**：构建复杂时间预测模型的基础模块

| 组件名称 | 功能描述 |
|---------|---------|
| TemporalEncoder | 时间序列编码器 |
| TemporalBlock | 基础时间处理块 |
| NARPredictionHead | 非自回归预测头 |
| SequentialSpatiotemporal | 序贯时空处理 |
| SequentialTrainer | 序贯训练逻辑 |
| SequentialDCConsistency | DC一致性检查 |

## 🚀 使用方式

### 基础使用

```python
# 导入空间模型
from models.spatial import UNet, SwinUNet

# 创建空间预测模型
spatial_model = UNet(in_ch=3, out_ch=3, features=[32, 64, 128])

# 导入时间模型  
from models.temporal import ARWrapper, SwinTemporal

# 创建时间预测模型（包装空间模型）
temporal_model = ARWrapper(backbone="SwinUNet", T_out=10)
```

### 工厂函数使用

```python
# 使用工厂函数创建模型
from models.spatial.factory import create_model as create_spatial_model
from models.temporal.factory import create_model as create_temporal_model

# 创建空间模型
spatial_model = create_spatial_model("UNet", in_ch=3, out_ch=3, features=[32, 64, 128])

# 创建时间模型
temporal_model = create_temporal_model("ARWrapper", backbone="SwinUNet", T_out=10)
```

### 向后兼容使用

```python
# 保持原有使用方式
from models import UNet, ARWrapper, create_model

# 原有工厂函数仍然可用
model1 = create_model("UNet", in_ch=3, out_ch=3, features=[32, 64, 128])
model2 = create_model("ARWrapper", backbone="SwinUNet", T_out=10)
```

## 📊 选择指南

### 空间预测任务选择

| 任务类型 | 推荐模型 | 理由 |
|---------|---------|------|
| **基础重建** | UNet | 经典架构，稳定可靠 |
| **高精度分割** | SwinUNet | 先进Transformer架构 |
| **频域建模** | FNO2d | 专门处理频域特征 |
| **快速推理** | MLPMixer | 并行计算，速度快 |
| **资源受限** | SwinTransformerTiny | 轻量级设计 |
| **多模态融合** | HybridModel | 组合多种架构优势 |

### 时间预测任务选择

| 预测模式 | 推荐模型 | 特点 |
|---------|---------|------|
| **自回归预测** | ARWrapper | 逐步生成，精度高 |
| **非自回归预测** | SwinTemporalNAR | 并行生成，速度快 |
| **时间聚合** | SwinTemporal | 利用历史信息增强 |
| **混合策略** | ARNARWrapper | 结合AR和NAR优势 |

## 🔧 迁移指南

### 从旧结构迁移

1. **导入路径不变**：原有导入方式仍然有效
2. **模型接口不变**：所有模型保持原有接口
3. **配置文件兼容**：现有配置无需修改
4. **新增分类导入**：可以使用更清晰的分类导入

### 新结构优势

1. **清晰的模型分类**：按功能组织，便于选择
2. **模块化设计**：独立的空间和时间模块
3. **工厂函数支持**：统一的模型创建接口
4. **更好的扩展性**：易于添加新模型类别
5. **维护性提升**：结构清晰，便于维护

## 📈 未来发展

1. **更多模型类别**：如图神经网络、物理信息神经网络
2. **自动模型选择**：基于任务特征推荐最优模型
3. **模型组合策略**：更复杂的混合架构支持
4. **性能优化**：针对特定硬件的优化版本
5. **标准化接口**：更严格的接口标准化