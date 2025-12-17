## 目标
- 先固定 T_out=1 的训练阶段，使验证口径与训练一致，观察 `Val Loss` 是否同步下降；随后再逐步扩展到 T_out=3/5。
- 在训练与验证中引入轻量的时序一致性项（导数/能量），提升验证敏感度与泛化。

## 配置更新（CPU YAML）
- 修改 `thesis_paper/configs/temporal/ar_training_config_debug_temporal_cpu copy.yaml`：
  - `training.curriculum.stages` 调整：`[{T_out:1, epochs:40}, {T_out:3, epochs:30}, {T_out:5, epochs:30}]`（总计 100）
  - 在 `training.loss_weights` 下新增：
    - `derivative_consistency: 0.1`
    - `energy_consistency: 0.05`

## 训练执行
- 使用当前 CPU 配置启动训练（或断点继续）：
  - `CUDA_VISIBLE_DEVICES="" python tools/training/train_real_data_ar.py --config "thesis_paper/configs/temporal/ar_training_config_debug_temporal_cpu copy.yaml"`
  - 若存在检查点则 `--resume runs/<exp>/last.ckpt`，断点继续。

## 监控与验收
- 监控 `Train/Loss` 与 `Val/Loss` 的同步下降；记录 `Val/RelL2`、`Val/MAE`。
- 在 T_out=1 阶段稳定后，再切换到 T_out=3/5 继续训练。
- 产出 `metrics.jsonl` 与 `results.md` 汇总曲线与最后一步指标。

## 风险与回退
- 若 `Val Loss` 仍不降：进一步增大一致性权重或延长 T_out=1 阶段；必要时开启轻量数据增强。