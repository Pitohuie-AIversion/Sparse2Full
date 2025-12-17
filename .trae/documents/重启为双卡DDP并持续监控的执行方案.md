## 执行目标
- 以双卡 DDP 重启空间-only训练，应用 BF16 混精与低验证开销。
- 持续监控吞吐、显存峰值、GPU 利用率与验证时长，消除卡顿。

## 重启步骤
- 环境线程：设置 `NUMEXPR_MAX_THREADS=96`、`OMP_NUM_THREADS=96`、`MKL_NUM_THREADS=96`、`OPENBLAS_NUM_THREADS=96`。
- 启动命令：
  - `torchrun --standalone --nproc_per_node=2 tools/training/train_real_data_ar.py --config "configs/train/ar_training_config debug.yaml"`
- 空间-only确保：`ar.enabled: false`、`sequential.enabled: false`、`data.T_in=1/T_out=1`、`model.name: SwinUNet`。
- 验证降负：`validation.check_val_every_n_epoch: 5`、关闭训练期 rollout 可视化。

## 监控与采集
- 终端日志抓取关键行：`Epoch`、`[Perf]`、`[VRAM]`、`Validating`。
- 文件采集：
  - `runs/<exp>/resource_metrics.jsonl`（GPU 利用率、显存、CPU）
  - `runs/<exp>/resources_epoch.jsonl`（吞吐、峰值显存、fetch/data/compute时间）
- 核验 BF16：日志中 AMP dtype 报告为 `bfloat16`；若报 `float16`，记录并执行回退策略。

## 成功标准
- 两卡 GPU 利用率均衡（≥70%），吞吐较单卡提升明显。
- 验证阶段不再停留在 `Validating: 0%`，用时短、显存峰值稳定（<50%）。
- `T_out=1` 持续记录，空间-only路径稳态运行。

## 回退与微调
- 若 BF16 算子不支持：回退 `training.precision: fp16-mixed` 与 `amp.autocast_dtype: float16`。
- 若 IO 受限：降低 `num_workers`，提高 `prefetch_factor`，保持 `pin_memory: true`。
- 若显存接近阈值：适度降低 `batch_size` 或设置 `gradient_accumulation_steps`。

## 交付
- 提供重启后首轮训练的吞吐、显存峰值、GPU 利用率与验证时长对比。
- 列出关键日志片段与资源采样快照，确认空间-only与 BF16 生效。