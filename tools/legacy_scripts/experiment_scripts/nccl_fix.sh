#!/bin/bash
# NCCL优化配置 - 解决DDP通信问题

# 基本NCCL设置
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# 网络优化
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

# 缓冲区优化
export NCCL_BUFFSIZE=2097152

# 线程优化
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMEXPR_MAX_THREADS=64

# PyTorch分布式优化
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DETAIL=DEBUG

echo "NCCL环境变量已设置完成"
echo "开始优化训练..."

# 启动优化训练
torchrun --standalone --nproc_per_node=2 tools/training/train_real_data_ar.py --config "configs/train/ar_training_config debug.yaml"