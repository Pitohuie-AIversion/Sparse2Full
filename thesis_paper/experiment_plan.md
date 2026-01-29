# 硕士论文实验规划 (Experiment Plan)

本文档旨在规划支撑硕士论文核心创新点的实验体系。实验设计围绕“统一观测算子”、“三重一致性损失”和“序列化时空训练”三个创新点展开，旨在提供充分的数据支撑以应对答辩。

## 1. 实验总体架构

实验分为三个层次：
1.  **主性能对比 (Main Results)**：与现有SOTA方法对比，证明整体框架的优越性。
2.  **消融实验 (Ablation Studies)**：逐一拆解创新点，证明每个模块的必要性。
3.  **分析实验 (Analysis)**：验证方法的鲁棒性、物理一致性和可视化效果。

---

## 2. 详细实验设置

### 实验一：主性能对比 (SOTA Comparison)
**目的**：证明 Sparse2Full 框架在稀疏观测下的时空重构能力优于现有方法。

*   **数据集**：PDEBench (流体力学数据, e.g., Navier-Stokes, Shallow Water)
*   **观测设置**：
    *   稀疏度：1% (极稀疏), 5%, 10%
    *   噪声：$\sigma=0.0$ (无噪), $\sigma=0.1$ (含噪)
*   **对比基线 (Baselines)**：
    1.  **传统方法**：
        *   `Bicubic/Bilinear`: 双三次/双线性插值（下界基准）。
    2.  **数据驱动基线**：
        *   `U-Net`: 标准图像修复模型（无物理约束）。
        *   `FNO (Fourier Neural Operator)`: 频域算子学习模型。
    3.  **先进模型**：
        *   `Swin-UNet`: 强注意力的纯视觉模型（无序列训练策略）。
        *   `EDSR`: 经典的超分辨率模型。
*   **评价指标**：
    *   **Rel-L2**: 相对L2误差 (主要指标)。
    *   **RMSE**: 均方根误差。
    *   **$L_{spec}$**: 频谱误差 (高频恢复能力)。
    *   **$L_{dc}$**: 观测点一致性误差。

### 实验二：消融实验 (Ablation Studies) —— 核心部分

#### 2.1 损失函数消融 (针对创新点：三重损失)
验证 $L_{rec}, L_{spec}, L_{dc}$ 的组合效果。

| 实验ID | 配置名称 | 重建损失 ($L_{rec}$) | 频域损失 ($L_{spec}$) | DC损失 ($L_{dc}$) | 预期结论 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **A** | Baseline Loss | ✅ | ❌ | ❌ | 只有基本的模糊恢复，高频丢失，观测点不准。 |
| **B** | + Spectral | ✅ | ✅ | ❌ | 纹理细节恢复变好，但观测点可能有偏差。 |
| **C** | + DC | ✅ | ❌ | ✅ | 观测点强制对齐，但可能缺乏高频细节。 |
| **D** | **Ours (Full)** | ✅ | ✅ | ✅ | **最佳平衡**：既有细节，又符合物理观测。 |

*   **运行命令参考**：
    ```bash
    # 实验 A
    python tools/training/train_real_data_ar.py loss.spectral.weight=0.0 loss.data_consistency.weight=0.0
    # 实验 B
    python tools/training/train_real_data_ar.py loss.spectral.weight=0.1 loss.data_consistency.weight=0.0
    # 实验 C
    python tools/training/train_real_data_ar.py loss.spectral.weight=0.0 loss.data_consistency.weight=1.0
    # 实验 D (Ours)
    python tools/training/train_real_data_ar.py loss.spectral.weight=0.1 loss.data_consistency.weight=1.0
    ```

#### 2.2 训练策略消融 (针对创新点：序列化训练)
验证课程学习（Curriculum Learning）对长时序预测稳定性的影响。

| 实验ID | 策略 | 描述 | 预期结论 |
| :--- | :--- | :--- | :--- |
| **S1** | Direct Joint | 直接训练 $T_{out}=10$ | 训练初期不稳定，容易陷入局部最优，收敛慢。 |
| **S2** | **Sequential (Ours)** | 阶段1($T=1$) $\to$ 阶段2($T=5$) $\to$ 阶段3($T=10$) | 训练平稳，最终误差更低，长时序累积误差小。 |

*   **运行命令参考**：
    ```bash
    # 实验 S1 (Direct)
    python tools/training/train_real_data_ar.py training.curriculum.enabled=false
    # 实验 S2 (Sequential - Default)
    python tools/training/train_real_data_ar.py training.curriculum.enabled=true
    ```

### 实验三：观测算子一致性分析 (针对创新点：统一算子)
**目的**：证明“训练与测试使用同一 $H$”的重要性。

*   **设置**：
    *   **Mismatch**: 训练时用 `Bicubic` 下采样，测试时用 `Gaussian Blur + Subsample` (模拟真实传感器)。
    *   **Unified (Ours)**: 训练和测试都用 `Gaussian Blur + Subsample`。
*   **指标**：重点看测试集上的 **DC Loss** ($||H(\hat{y}) - y||$)。
*   **结论**：Unified 组的 DC Loss 应显著低于 Mismatch 组（数量级差异）。

---

## 3. 图表规划 (Figures & Tables)

### 表 1：主性能对比表 (Quantitative Results)
| Model | Params (M) | FLOPs (G) | Rel-L2 ($\downarrow$) | RMSE ($\downarrow$) | PSNR ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bilinear | - | - | 0.xxx | ... | ... |
| U-Net | ... | ... | ... | ... | ... |
| FNO | ... | ... | ... | ... | ... |
| Swin-UNet | ... | ... | ... | ... | ... |
| **Ours** | ... | ... | **0.xxx** | **...** | **...** |

### 表 2：消融实验结果 (Ablation Study)
展示不同损失组合下的指标变化，证明每个模块都有正向增益。

### 图 1：可视化对比 (Qualitative Comparison)
*   **行**：GT, Ours, Baseline 1, Baseline 2
*   **列**：不同时间步 ($T=1, T=5, T=10$)
*   **内容**：流场热图 + **误差图 (Error Map)** (误差图能更直观地显示出你的方法在边界和细节上的优势)。

### 图 2：频谱分析图 (Spectral Analysis)
*   绘制 GT 和 Pred 的径向平均功率谱 (Radially Averaged Power Spectrum)。
*   展示你的方法在高频部分（曲线右侧）更贴近 GT，而 Baseline 往往会衰减过快（Blurry）。

### 图 3：稀疏度鲁棒性曲线 (Robustness)
*   X轴：观测比例 (0.1% $\to$ 10%)
*   Y轴：Rel-L2 Error
*   曲线：Ours vs Baseline。展示在极稀疏区域（左侧），你的曲线更平缓，Baseline 误差急剧上升。

---

## 4. 执行建议

1.  **优先级**：先跑通 **实验二 (消融)**。因为这是验证你创新点最直接的证据。主对比实验（实验一）如果算力不够，可以只选 1-2 个最具代表性的 Baseline (如 Swin-UNet 和 FNO)。
2.  **资源管理**：
    *   每个实验建议跑 3 个 Seed 取平均（如果时间不够，先跑 1 个 Seed 定性）。
    *   利用 `tools/training/train_real_data_ar.py` 的 `experiment_name` 参数区分不同实验的日志。
