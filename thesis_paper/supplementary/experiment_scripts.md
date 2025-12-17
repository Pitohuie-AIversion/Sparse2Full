# 实验执行脚本

## 快速开始

### 1. 环境检查
```bash
# 检查Python环境和依赖
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"

# 检查配置文件
ls -la configs/ | head -10
```

### 2. 核心实验（推荐优先执行）
```bash
# 空间稀疏观测实验 - 最重要的基础实验
echo "开始执行空间稀疏观测实验..."
python tools/train.py --config configs/experiment/spatial_sparse_sr2.yaml --seed 42
python tools/train.py --config configs/experiment/spatial_sparse_sr4.yaml --seed 42
python tools/train.py --config configs/experiment/spatial_sparse_crop20.yaml --seed 42
python tools/train.py --config configs/experiment/spatial_sparse_crop40.yaml --seed 42

### 2.1 空间模型训练入口与配置管理（非 AR 命名）

**训练入口（空间模型，空间-only 模式）**：
```bash
python tools/training/train_real_data_ar.py --config configs/train/ar_training_config_debug.yaml --seed 42
```

**YAML 配置管理约定**：
- 每次实验从 `configs/train/ar_training_config_debug.yaml` 复制一份，重命名后修改（命名不含 `ar`）；
- 推荐命名：`spatial_training_config_<task>_<date>_<note>.yaml`
```bash
cp configs/train/ar_training_config_debug.yaml thesis_paper/configs/spatial_training_config_srx2_20251127.yaml
# 然后编辑 thesis_paper/configs/spatial_training_config_srx2_20251127.yaml 以修改数据/模型/训练参数
python tools/training/train_real_data_ar.py --config thesis_paper/configs/spatial_training_config_srx2_20251127.yaml --seed 42
```

**典型修改项**：
- 数据/观测：`datasets.*` 与 `ops.degradation.*` 保持与 H/DC 一致；
- 模型：`models.spatial`/`models.temporal`；
- 训练：`training.batch_size`、`training.max_epochs`、`optimizer.*`；
- 评测：`eval.metrics` 与 `paper_package` 输出路径。

备注：若使用 `train_real_data_ar.py`，请在 YAML 中确保 `ar.enabled: false`（空间-only 模式）。

# 损失函数消融实验 - 理论验证关键
echo "开始执行损失函数消融实验..."
python tools/train.py --config configs/ablation/loss_ablation_no_spectral.yaml --seed 42
python tools/train.py --config configs/ablation/loss_ablation_no_dc.yaml --seed 42
python tools/train.py --config configs/ablation/loss_ablation_baseline.yaml --seed 42
```

### 3. 对比实验
```bash
# 与SOTA方法对比
echo "开始执行对比实验..."
python tools/train.py --config configs/comparison/senseiver_comparison.yaml --seed 42
python tools/train.py --config configs/comparison/pinto_comparison.yaml --seed 42
python tools/train.py --config configs/comparison/sino_comparison.yaml --seed 42

### 3.1 横向对比统一设置（Params / Epoch / 训练口径）

**模型清单（空间-only主任务）**：
- Baselines：UNet、UNet++、SegFormer/UNetFormer、FNO2D、Hybrid(SwinUNet+FNO瓶颈)
- Ours：Sparse2Full（Swin-UNet 空间编码 + NAR 预测头，可关闭时序用于空间-only）

**统一超参与训练口径**：
- 优化器：AdamW（`lr=1e-3`，`wd=1e-4`），Cosine 调度 + 1000 warmup；AMP 开启；梯度裁剪 1.0
- Epoch：空间-only主任务统一 `E=100`（DR2D）、`E=80`（RDB）；早停 `patience=20`
- 批次：`batch_size=16`（128×128）；显存不足时 `gradient_accumulation=2`
- 随机性：固定 5 种子；PyTorch 确定性模式开启；Hydra YAML 全量管理
- 观测与一致性：`H` 与训练 `DC` 完全复用同一实现与配置（核/σ/插值/对齐/边界）
- 评测：测试集全量帧；指标为 `Rel-L2/MAE/PSNR/SSIM/fRMSE-low/mid/high/||H(ŷ)−y||`，报告 `均值±标准差（n=5）` + paired t-test + Cohen’s d

**资源统计统一口径**：
- Params（M）、FLOPs（G@256²）、显存峰值（GB）、推理延迟（ms）；硬件标注（GPU/AMP/Batch）
- 统计脚本：`tools/summarize_runs.py`、`paper_package/scripts/`；三线表统一单位与小数位

**示例命令（按统一设置运行）**：
```bash
# Baseline: FNO2D（DR2D，SR×2，E=100）
python tools/train.py --config configs/comparison/fno2d_srx2_dr2d.yaml --seed 42
# Ours: Sparse2Full（DR2D，SR×2，E=100）
python tools/train.py --config thesis_paper/configs/spatial_training_config_srx2_final.yaml --seed 42
```

### 3.2 参数量统计与 Epoch 确定口径

**参数量（Params）统计**：
- 统计范围：仅计入可训练参数（`requires_grad=True`），单位 M；不计入缓冲项与优化器状态。
- 统计方法：运行结束由 `utils/resource_monitor.py` 统一记录；或在模型构建后计算：
```python
params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
```
- 公平性：横向对比优先将各基线宽度/深度调至与 Sparse2Full 参数量相近（±5%）；无法严格对齐时在资源表列出差异。

**Epoch 策略**：
- 上限：DR2D 主任务 `E=100`；RDB 主任务 `E=80`；启用早停 `patience=20`（验证无提升即止）。
- 收敛准则：
  - 验证 `Rel-L2` 连续 20 轮无改善；或达到目标阈值（DR2D `<0.05`）。
  - 梯度范数稳定在 `0.1–1.0`；训练/验证曲线无异常震荡。
- 步数估算（用于资源规划）：
```text
steps_per_epoch ≈ ceil(N_train_samples / batch_size)
total_updates ≈ steps_per_epoch × E
```
其中 `N_train_samples` 为训练样本总数（按固定 splits；空间-only可按帧采样策略计算）。

**统一记录**：
- 运行后用 `tools/summarize_runs.py` 汇总 Params / Epoch / FLOPs / 显存峰值 / 延迟，并生成 LaTeX 表；与 6.2.10 资源主表口径一致。
```

### 4. 理论验证实验
```bash
# Kolmogorov宽度验证
echo "开始网络宽度实验..."
for width in 32 64 128 256 512; do
    python tools/train.py --config configs/theory/width_${width}.yaml --seed 42
done

# 信息恢复下界验证
echo "开始观测密度实验..."
for density in 64 128 256 512 1024; do
    python tools/train.py --config configs/theory/density_${density}.yaml --seed 42
done
```

### 5. 时序实验
```bash
# AR vs NAR对比
echo "开始AR vs NAR对比实验..."
python tools/train.py --config configs/temporal/ar_temporal.yaml --seed 42
python tools/train.py --config configs/temporal/nar_temporal.yaml --seed 42

# 时间稀疏观测
echo "开始时间稀疏观测实验..."
python tools/train.py --config configs/temporal/ts25_temporal.yaml --seed 42
python tools/train.py --config configs/temporal/ts50_temporal.yaml --seed 42
python tools/train.py --config configs/temporal/ts75_temporal.yaml --seed 42
```

### 6. 鲁棒性实验
```bash
# 噪声鲁棒性
echo "开始噪声鲁棒性实验..."
for noise in 0.01 0.02 0.05 0.10; do
    python tools/train.py --config configs/robustness/gaussian_noise_${noise}.yaml --seed 42
done

# 边界条件影响
echo "开始边界条件实验..."
for boundary in mirror zero wrap extend; do
    python tools/train.py --config configs/robustness/boundary_${boundary}.yaml --seed 42
done
```

## 批量实验执行

### 多种子实验
```bash
# 使用5重种子确保统计可靠性
seeds=(42 123 456 789 999)
for seed in "${seeds[@]}"; do
    echo "使用种子 $seed 执行实验..."
    python tools/train.py --config configs/experiment/spatial_sparse_baseline.yaml --seed $seed
done
```

### 并行实验
```bash
# 使用GNU parallel并行执行（如果可用）
parallel python tools/train.py --config {} --seed 42 ::: configs/experiment/*.yaml
```

## 实验结果收集

### 自动结果汇总
```bash
# 收集实验结果
echo "收集实验结果..."
python tools/summarize_runs.py --input_dir runs/ --output_dir paper_package/

# 生成LaTeX表格
python tools/generate_latex_tables.py --results_dir paper_package/

# 生成可视化图表
python tools/visualization/generate_paper_figures.py --results_dir paper_package/
```

### 一致性检查
```bash
# H/DC一致性验证
echo "执行H/DC一致性检查..."
python tools/check_dc_equivalence.py --config configs/check_consistency.yaml

# 统计显著性检验
echo "执行统计显著性检验..."
python tools/statistical_tests.py --results_dir paper_package/
```

## 资源监控

### GPU使用监控
```bash
# 监控GPU使用情况
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > gpu_monitor.log &

# 内存使用监控
python -c "
import psutil
import time
while True:
    mem = psutil.virtual_memory()
    print(f'Memory: {mem.percent}% used, {mem.free/1024**3:.1f}GB free')
    time.sleep(60)
" > memory_monitor.log &
```

### 实验时间估算
```bash
# 单组实验时间估算
echo "实验时间估算："
echo "- 基础实验 (SR×2): ~2小时"
echo "- 基础实验 (SR×4): ~2.5小时"
echo "- 消融实验: ~1.5小时/组"
echo "- 对比实验: ~3小时/组"
echo "- 理论验证: ~1小时/组"
echo "总计需要: 24-48小时"
```

## 故障排除

### 常见问题
```bash
# CUDA内存不足
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 数据加载问题
python tools/validate_data_paths.py --dataset pdebench

# 配置验证
python tools/validate_configs.py --config configs/experiment/
```

### 实验恢复
```bash
# 从检查点恢复实验
python tools/train.py --config configs/experiment/spatial_sparse_sr2.yaml --seed 42 --resume runs/latest_checkpoint.pth
```

## 质量保证

### 实验前检查
```bash
# 代码质量检查
ruff check .
black --check .
mypy --strict

# 单元测试
pytest tests/unit/ -v

# 集成测试
pytest tests/integration/ -v
```

### 实验后验证
```bash
# 结果完整性检查
python tools/validate_results.py --results_dir paper_package/

# 理论一致性验证
python tools/validate_theory.py --results_dir paper_package/
```

## 论文材料生成

### 自动生成
```bash
# 生成完整论文材料包
echo "生成论文材料包..."
python tools/generate_paper_package.py \
    --results_dir paper_package/ \
    --output_dir paper_package/final/ \
    --latex_template templates/paper_template.tex

# 生成盲审版本
python tools/generate_blind_review.py --input_dir paper_package/final/ --output_dir paper_package/blind/
```

### 手动检查项目
- [ ] 所有实验结果是否完整
- [ ] 图表是否清晰可读
- [ ] 表格数据是否准确
- [ ] 统计检验是否通过
- [ ] 理论验证是否一致
- [ ] 资源配置是否合理

---
*最后更新：2024年*
