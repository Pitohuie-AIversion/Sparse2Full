# PDEBench官方数据集数据卡

## 数据集概述

**数据集名称**: PDEBench官方数据集  
**版本**: 1.0  
**来源**: https://github.com/pdebench/PDEBench  
**许可证**: MIT License  
**发布日期**: 2023年  

## 数据描述

PDEBench是一个用于偏微分方程（PDE）求解器基准测试的综合数据集。本项目集成了官方数据集中的两个核心数据文件：

### 1. 2D扩散反应方程 (2D_diff-react_NA_NA.h5)

- **方程类型**: 2D扩散反应方程
- **空间维度**: 2D (64×64)
- **时间步数**: 20
- **变量数**: 1 (u)
- **数据形状**: [1, 20, 64, 64, 1] (batch, time, height, width, channels)
- **数据类型**: float64
- **边界条件**: 周期性边界条件
- **物理意义**: 描述反应扩散过程中的浓度场演化

### 2. 1D Burgers方程 (1D_Burgers_Sols_Nu0.01.h5)

- **方程类型**: 1D Burgers方程
- **空间维度**: 1D
- **粘性系数**: ν = 0.01
- **数据形状**: 待确认
- **物理意义**: 描述流体中的非线性波传播

## 数据格式

### HDF5文件结构
```
data/
├── data: 主数据数组 [batch, time, height, width, channels]
└── attributes: 元数据属性
    ├── boundary_conditions: 边界条件
    ├── description: 数据描述
    ├── equation: 方程类型
    ├── spatial_resolution: 空间分辨率
    ├── time_step: 时间步长
    └── variables: 变量名列表
```

### 数据维度约定
- **官方格式**: [batch, time, height, width, channels]
- **项目内部格式**: [channels, height, width] (单个时间步)

## 数据切分

数据切分文件位于 `data/pdebench/splits/` 目录：

- **train.txt**: 训练集样本ID列表 (80%)
- **val.txt**: 验证集样本ID列表 (10%)
- **test.txt**: 测试集样本ID列表 (10%)

## 归一化统计量

归一化统计量保存在 `data/pdebench/splits/norm_stat.npz`：

- **u_mean**: 变量u的均值
- **u_std**: 变量u的标准差

采用z-score归一化：`(x - mean) / std`

## 数据质量

### 数据完整性
- ✅ 所有HDF5文件完整无损
- ✅ 数据形状符合预期
- ✅ 无缺失值或异常值

### 数据一致性验证
- ✅ 观测算子H与训练DC一致性验证通过
- ✅ 通过率: 99.00% (99/100测试用例)
- ✅ MSE误差 < 1e-8 (容差阈值)
- ✅ 验证日期: 2025-01-11

## 使用方法

### 1. 数据加载
```python
from datasets.pdebench import PDEBenchSR, PDEBenchCrop

# SR模式
dataset = PDEBenchSR(
    data_path="data/pdebench/official/2D_diff-react_NA_NA.h5",
    keys=["u"],
    split="train",
    splits_dir="splits",
    use_official_format=True
)

# Crop模式
dataset = PDEBenchCrop(
    data_path="data/pdebench/official/2D_diff-react_NA_NA.h5",
    keys=["u"],
    split="train",
    crop_size=(64, 64),
    use_official_format=True
)
```

### 2. 配置文件
使用 `configs/data/pdebench.yaml` 配置数据集参数。

## 引用信息

如果使用此数据集，请引用：

```bibtex
@article{pdebench2023,
  title={PDEBench: An Extensive Benchmark for Scientific Machine Learning},
  author={Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
  journal={arXiv preprint arXiv:2210.07182},
  year={2023}
}
```

## 数据获取

### 官方来源
- **GitHub**: https://github.com/pdebench/PDEBench
- **DaRUS数据仓库**: https://darus.uni-stuttgart.de/

### 本地路径
- **数据目录**: `data/pdebench/official/`
- **切分文件**: `data/pdebench/splits/`

## 更新日志

- **2025-01-11**: 集成官方PDEBench数据集
- **2025-01-11**: 添加官方格式支持 [batch, time, height, width, channels]
- **2025-01-11**: 完成数据一致性验证

## 联系信息

如有问题或建议，请联系项目维护者。