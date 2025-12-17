## 框架与版本
- Python: 3.12.2（docs/DEVELOPMENT.md:235-240）
- PyTorch: 2.7.0+cu118；Torch CUDA: 11.8（docs/DEVELOPMENT.md:235-240）
- 操作系统与内核: CentOS 8；Linux 4.18.0-193（docs/DEVELOPMENT.md:210-214）
- 驱动/CUDA驱动版本: NVIDIA Driver 545.23.06；CUDA 驱动 12.3（docs/DEVELOPMENT.md:228-234）

## 训练关键配置
- 设备与并行：`device.accelerator: cuda`、`devices: 2`、`strategy: ddp`（configs/train/ar_training_config debug.yaml:13-17）
- 精度与AMP：`training.precision: bf16-mixed`；`training.amp.autocast_dtype: bfloat16`（configs/train/ar_training_config debug.yaml:220；268-279）
- 空间-only：`ar.enabled: false`、`sequential.enabled: false`、`data.T_in: 1`、`data.T_out: 1`（configs/train/ar_training_config debug.yaml:19-22；34-36；新增 sequential.enabled）
- 模型：`model.name: SwinUNet`；`img_size: 128`；`in/out_channels: 2`（configs/train/ar_training_config debug.yaml:157-175；163）
- DataLoader：`num_workers: 32`、`prefetch_factor: 16`、`pin_memory: true`、`persistent_workers: true`（configs/train/ar_training_config debug.yaml:68-76）
- 验证与可视化：`validation.check_val_every_n_epoch: 5`，`logging.visualization.save_rollout_visualization: false`（configs/train/ar_training_config debug.yaml:210-217；361-366）

## 硬件规格
- GPU：NVIDIA L40 ×2，显存 46,068 MiB/卡（docs/DEVELOPMENT.md:228-234）
- CPU：AMD EPYC 9654，192 逻辑 CPU（docs/DEVELOPMENT.md:215-221）
- 内存：1.0 TiB（docs/DEVELOPMENT.md:222-226）

## 初步瓶颈信号（基于训练日志/资源采样）
- 单进程时验证阶段停留在 `Validating: 0%`；多卡 DDP后应均衡两卡负载（tools/training/train_real_data_ar.py:3148-3174；711-740）
- DataLoader回退提示：`⚠️ 无法提取底层dataset，使用数据模块的默认DataLoader`，需要确认并行参数实际生效（tools/training/train_real_data_ar.py:1487）
- 空间-only路径已生效：单帧前向与空间损失（tools/training/train_real_data_ar.py:2624-2679）

## 优化计划
1) 内存使用分析
- 持续读取 `runs/<exp>/resource_metrics.jsonl` 与 `resources_epoch.jsonl`：检查 GPU `mem_used_mib`、`cpu_percent`、`system_memory_percent` 与显存峰值（已存在文件）。
- 验证 `pin_memory: true` 与 `persistent_workers: true` 是否在当前 DataLoader 生效（tools/training/train_real_data_ar.py:1082-1138 的兼容分支）。
- 评估 `batch_size` 与显存峰值关系：通过 `resources_epoch.jsonl` 的 `gpu_peak_allocated_gb` 与 `throughput_samples_per_sec` 对比。

2) 性能优化措施
- 保持混合精度 BF16；如遇不支持算子，回退到 FP16（configs/train/ar_training_config debug.yaml:220；268-279）。
- 确认 `torch.backends.cudnn.benchmark = True` 已启用（tools/training/train_real_data_ar.py:681-706；793-805；854-886）。
- 数据管道：维持 `num_workers: 32`、`prefetch_factor: 16`，必要时按文档步进调整（docs/DEVELOPMENT.md:285-289）。

3) 硬件配置检查
- 驱动/CUDA兼容：记录自 `docs/DEVELOPMENT.md`，若运行环境变化需重新采样。
- 观察 PCIe 带宽是否在 `resource_metrics.jsonl` 间接反映为 GPU util 异常低与 iowait 高；如有 IO 迹象，调低 `num_workers` 并提高 `prefetch_factor`。
- 确保不存在竞争进程：若 GPU util 波动异常，检查 `nvidia-smi` 进程列表（仅建议，保持只读操作方案）。

4) 代码级优化
- 模型结构：已使用轻量 SwinUNet 深度与头数（configs/train/ar_training_config debug.yaml:166-173）；必要时进一步下调 `depths/embed_dim` 验证吞吐收益。
- 损失/优化器：`AdamW` with `fused: true` 已配置；如不兼容自动回退（configs/train/ar_training_config debug.yaml:226-236）。
- 梯度累积：根据显存与吞吐，适度设置 `training.gradient_accumulation_steps`（configs/train/ar_training_config debug.yaml:182-187）。

5) 交付与验证
- 输出一次双卡 DDP + BF16 的首轮训练指标：吞吐、GPU 利用率、显存峰值、验证时长。
- 若发现瓶颈，按计划逐项微调并给出对比报告（调整不改变空间-only目标）。

请确认以上信息与计划，我将继续按此执行并汇报监控结果与优化效果。