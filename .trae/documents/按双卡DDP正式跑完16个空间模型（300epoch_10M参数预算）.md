## 范围
- 按你列出的 16 个空间模型：unet、unetplusplus、fno2d、swint、swinirlite、restormerlite、uformerlite、segformer、unetformer、sparseswinunet、liif、mlpmixer、vit、transformer、ufnounet、hybridmodel。
- 每个模型均使用 ar_paper_aligned.yaml 的全量数据与训练配置，epochs=300，参数预算约 10M（容差±0.5M）。
- 双卡并行：CUDA_VISIBLE_DEVICES=0,1 + torchrun --nproc_per_node=2，全程DDP。

## 当前状况
- 顺序调度清单显示前一轮批量任务均标记为 failed，主要原因为：
  - DDP未生效的单进程启动（WORLD_SIZE/RANK缺失）导致只占用cuda:0。
  - 自动调参在UNet等模型上过度放大特征宽度（如 features=[248,504,1016,2040]）引发GPU OOM。

## 技术调整（不改变论文口径）
- DDP启动：统一改用 torchrun（nproc=2），训练器检测WORLD_SIZE/RANK/LOCAL_RANK后在两卡上运行。
- 设备选择健壮化：当存在CUDA_VISIBLE_DEVICES时，绑定cuda:0（可见列表索引）由DDP设置rank→cuda:0/1。
- 自动调参安全边界（仍目标10M±0.5M）：
  - UNet/UNet++：features逐层上限≤512；若超预算则按(scale=√(target/params))收缩，保证不OOM。
  - Swin系列/ViT/Transformer：embed_dim上限≤192；window_size与grid整除。
  - FNO：width上限≤192，n_layers固定4；必要时小步搜索。
  - MLP/MLPMixer：embed_dim上限≤1024，分块收敛逼近。
- 资源策略：
  - 重模型 batch_size 从128→64；必要时 gradient_accumulation_steps=2 保持有效批大小。
  - 继续AMP(bfloat16)、channels_last，保持数据/损失一致性。
- Loss与数据：
  - 仅重建项（Rel-L2 + 0.1×MAE），spectral=0、data_consistency=0；全量DR2D数据，T_in=1/T_out=1，SR观测一致。

## 执行步骤
1. 为16个模型生成临时配置（每模型写入 runs/tmp_configs/...yaml），包含：DDP=2、10M预算与安全上限、epochs=300。
2. 逐模型用 torchrun 启动；每个作业写入日志 runs/logs/AR-DR2D-10M-300ep-ddp2-model_<model>-...log 与 config_merged.yaml 快照、checkpoints/。
3. 监控：保留 GPU 监控（runs/monitor/gpu_util.jsonl），确认两卡持续满载。
4. 完成后收集 test_results.json（最终指标），汇总生成 runs/spatial_models_300ep_10M_summary.json（含均值±std/资源统计）。

## 失败兜底
- 若某模型仍OOM：降低 batch_size（至32）或提高accumulation至4；若仍OOM，暂时将目标参数预算下调至9M（容差+0.5M）并记录资源变更。
- 若DDP初始化失败：自动回退到单卡两路并行（各自CUDA_VISIBLE_DEVICES=0/1）保证两卡利用率，同时记录回退事件。

## 交付物
- 每模型：检查点、config_merged.yaml、模型信息（model_info.json）、最终测试指标（test_results.json）。
- 全局：summary.json（16模型指标与资源总表）、监控记录、训练日志路径索引。

## 确认
- 以上方案将依次跑完你列出的这16个模型，双卡DDP、300epoch、10M预算、仅重建损失。确认后我立即按该方案重启并持续监控。