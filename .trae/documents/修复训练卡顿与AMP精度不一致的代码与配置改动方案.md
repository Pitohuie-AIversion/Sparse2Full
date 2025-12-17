## 目标
- 解决验证阶段卡顿与日志“Validating: 0%”停留问题，确保双卡 DDP 下两卡均衡加载。
- 统一 AMP 为 BF16，在所有 autocast 使用点显式传入 dtype，避免回退到 FP16 或默认路径。
- 降低验证阶段的观测/可视化开销，稳定吞吐与显存峰值。

## 代码修复（tools/training/train_real_data_ar.py）
1) 统一 AMP autocast dtype 传递（防止默认/FP16回退）
- 位置：`validate` 的旧版本分支（tools/training/train_real_data_ar.py:2961）
- 现状：`with autocast(device_type='cuda', enabled=(self.device.type == 'cuda'))`
- 修改：`with autocast(device_type='cuda', dtype=getattr(self, 'autocast_dtype', torch.bfloat16), enabled=(self.device.type == 'cuda'))`
- 目的：显式使用 BF16（或配置中指定的 dtype），避免默认 dtype 带来的不一致

2) 保持权重为 FP32，避免错误的权重类型转换
- 位置：`setup_traditional_model`（tools/training/train_real_data_ar.py:1910）
- 现状：`self.sequential_model = self.sequential_model.to(self.autocast_dtype)`（将模型权重转为 BF16/FP16）
- 修改：移除此转换，保留权重为 FP32；仅通过 autocast 控制计算精度
- 目的：提升数值稳定性，符合 AMP 设计（权重 FP32 + 前向/反向自动混精）

3) 验证阶段观测构造开关（空间-only场景降负）
- 位置：验证路径观测生成（tools/training/train_real_data_ar.py:3189-3231）
- 修改：在 `ar.enabled == false` 且 `data.T_out == 1` 时跳过观测序列构造（`observation_seq=None`），仅传 h_params 参与 DC；或通过配置 `validation.use_observation=false` 控制
- 目的：减少验证显存占用与 CPU/GPU 开销，避免“Validating: 0%”卡住

4) DDP 日志与进度显示
- 位置：训练循环与验证日志（tools/training/train_real_data_ar.py:4139-4146；3174）
- 修改建议：仅在 `rank==0` 输出 tqdm 与关键指标，确保并发日志不互相阻塞；保持 `leave=False`
- 目的：双卡环境下日志干扰最小化，进度显示稳定

## 配置修复（configs/train/ar_training_config debug.yaml）
1) 移除不兼容的 `pin_memory_device` 字段
- 现状：`data.dataloader.pin_memory_device: "cuda"`
- 修改：删除该键；由训练脚本在兼容分支中条件性设置（tools/training/train_real_data_ar.py:1101-1138）
- 目的：避免旧版 DataLoader 因不支持该参数抛错或隐性回退

2) AMP/精度保持 BF16（已设置）
- `training.precision: bf16-mixed`
- `training.amp.autocast_dtype: bfloat16`
- `training.amp.cast_model_type: bfloat16`（如不需要可移除，避免权重类型误改；配合代码修复将权重保持 FP32）

3) 验证与可视化降负（已设置部分）
- `validation.check_val_every_n_epoch: 5`
- `logging.visualization.save_rollout_visualization: false`
- 可新增：`validation.use_observation: false`（用于代码中的观测开关）

## 验证与监控
- 双卡 DDP 启动：`torchrun --standalone --nproc_per_node=2 tools/training/train_real_data_ar.py --config "configs/train/ar_training_config debug.yaml"`
- 持续监控：
  - 终端关键行：`[Perf]`（吞吐/GPU利用）、`[VRAM]`（峰值显存）、`Validating`（进度）
  - 文件：`runs/<exp>/resource_metrics.jsonl` 与 `resources_epoch.jsonl`
- 成功标准：两卡 GPU 利用率均衡（≥70%）、吞吐提升、验证阶段不再停留在 0%、显存峰值稳定（<50%）

## 风险与回退
- 若 BF16 算子不支持：回退 `training.precision: fp16-mixed` 与 `training.amp.autocast_dtype: float16`
- 若 IO 为瓶颈：降低 `num_workers`、提高 `prefetch_factor`，保持 `pin_memory: true`
- 若显存接近阈值：下调 `batch_size` 或开启 `gradient_accumulation_steps`

若你确认，我将按以上改动实施修复并重启训练，随后提供首轮训练的吞吐、显存峰值、GPU 利用率与验证用时对比。