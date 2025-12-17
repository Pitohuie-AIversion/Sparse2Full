## 目标
- 将样本上限从 10 提升到 100，在 CPU 上重新训练并观察验证指标是否收敛（特别是 `rel_l2` 随 epoch 下降）。

## 配置调整
- 修改文件：`thesis_paper/configs/temporal/ar_training_config_debug_temporal_cpu copy.yaml`
  - `data.sample_limit: 100`（原为 10）
  - 其他项保持不变：`sequential.enabled: true`、`two_stage_training: false`、批量大小仍为 `8`、`num_workers: 0`、AMP 关闭。
  - 现有划分按比例使用：train≈80、val≈15、test≈5。

## 运行与监控
- 训练命令（CPU）：
  - `CUDA_VISIBLE_DEVICES="" python tools/training/train_real_data_ar.py --config "thesis_paper/configs/temporal/ar_training_config_debug_temporal_cpu copy.yaml"`
- 训练期间监控：
  - 关注 `Train/Loss` 曲线与验证周期（每 5 epoch）输出的 `rel_l2` 与 `mae`。
  - 保持课程学习：`T_out: 1→3→5` 分阶段，预期 `rel_l2` 随阶段与 epoch 下降。

## 收敛验收
- 验证收敛标准：参考 YAML 的 `validation.convergence_criteria`（目标 `target_rel_l2: 0.3`、耐心 `5`、最小改进 `5e-4`）。
- 生成并提供：
  - `runs/<exp>/` 下的训练曲线与指标文件
  - 训练结束后的资源摘要与（可选）metrics 汇总文件

## 风险与调整
- CPU 较慢：若训练耗时长，可先跑到 `T_out=3` 阶段中途观察趋势；必要时降低批量到 `4`。
- 数值不稳定：当前模型含稳健检查与裁剪逻辑，无需额外更改。

## 交付
- 更新后的 YAML 与本次训练的日志、曲线与收敛说明。