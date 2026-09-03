# 实验基线补充计划 (Bicubic-CNN & RBF-CNN)

## 1. 摘要 (Summary)
为了回应外审专家关于“缺少与传统数值方法或统计插值方法的直接对比”的意见，本计划将在原有的深度学习基线（EDSR, UNet等）基础上，新增两组“传统插值 + 轻量CNN后处理”的对照基线：`Bicubic-CNN` 和 `RBF-CNN`。该基线严格控制变量，不使用复杂的时序模块、观测一致性损失及谱一致性损失，仅作为纯粹的空间重建对比方案。

## 2. 当前状态分析 (Current State Analysis)
- 当前论文的第4章（`chapter4_results_verification.md`）已有丰富的深度模型对比（表4-1至表4-5），但缺乏插值方法与浅层CNN结合的“混合流派”基线。
- 训练框架 `train_real_data_ar.py` 已支持多模型的灵活调用，观测算子 $H$ 定义在 `ops/degradation.py` 中。
- 之前的基线测试（`eval_baseline.py`）已经补齐了纯传统插值（Bicubic/Bilinear）的结果。

## 3. 具体修改方案 (Proposed Changes)

### 3.1 代码实现
1. **新增模型文件** `models/spatial/interp_cnn.py`：
   - 实现 `BicubicCNN` 类：对输入执行 Bicubic 插值，然后经过 5 层 $3\times3$ 残差卷积（通道数 64，GELU 激活）。
   - 实现 `RBFCNN` 类：通过高斯核计算 RBF 权重并进行全场插值，再经由相同的 5 层卷积后处理。
2. **注册模型**：在 `models/spatial/__init__.py` 或 `models/baseline_models.py` 暴露这两个类，确保可通过配置文件调用。
3. **配置适配**：
   - 复制并修改现有的 yaml 配置文件（例如 `ar_paper_aligned_sr4_shallow_water.yaml` 改为 `ar_paper_bicubic_cnn_sr4_shallow_water.yaml`）。
   - 调整损失函数权重（禁用谱一致性和DC损失），仅保留 MSE 重建损失。

### 3.2 论文修改
1. **4.1.3节 基线模型与选型依据**：新增描述段落，介绍插值加后处理基线的数学定义与选取理由，引入符号 $\tilde{u}$。
2. **4.2.1节 空间重建性能**：在原有表格中新增 `Bicubic-CNN` 和 `RBF-CNN` 行，补充对比分析。强调纯数据驱动方法、传统插值方法以及混合方法在不同场景下的表现差异。
3. **符号说明表**：在 `chapter0_notation.md` 中增加 $\tilde{u}$ 的定义（传统插值得到的粗重建场）。
4. **外审意见回复**：撰写《学生对外审意见修改说明》的相关回复文字。

## 4. 假设与决策 (Assumptions & Decisions)
- RBF 插值采用高斯核计算（`epsilon` 作为超参数）。针对 $16\times16$ 的 Crop 任务，稀疏点数量 $N=256$，在 GPU 上直接求解线性方程组 $K W = Y$ 计算量极小，完全可在训练中动态进行。
- 保持网络参数极小化（5层卷积），以体现“轻量后处理”的特性。

## 5. 验证步骤 (Verification Steps)
- 使用 `tools/training/train_real_data_ar.py` 分别运行 `BicubicCNN` 和 `RBFCNN`。
- 确保测试过程中指标（Rel-L2, PSNR, SSIM, H_err）正确生成。
- 检查生成的可视化文件，确认其重建表现符合预期。
