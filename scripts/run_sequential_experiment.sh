#!/bin/bash
# 运行时空解耦序列化训练 (Sequential Spatiotemporal Training)
# 用法: bash scripts/run_sequential_experiment.sh [GPU_ID]

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================================"
echo "开始运行序列化训练 (Sequential Pipeline) - GPU $GPU_ID"
echo "策略: Spatial(Freeze T) -> Temporal(Freeze S) -> Joint"
echo "========================================================"

# 使用专门的配置文件
python tools/training/train_sequential_pipeline.py \
    --config-name sequential_pipeline \
    training.epochs=20 \
    data.dataloader.batch_size=4 \
    data.T_out=5 \
    experiment.name="Seq-SwinTrans-Demo"

echo "========================================================"
echo "序列化训练全部完成！"
echo "请查看 runs/Seq-SwinTrans-Demo_... 目录下的日志和模型"
echo "========================================================"
