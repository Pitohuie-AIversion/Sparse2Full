#!/usr/bin/env bash
# CPU threading optimization for high-core systems (balanced)
# Usage: source tools/cpu_optimize.sh && python tools/training/train_real_data_ar.py --config "configs/ar_training_config debug.yaml" --seeds 42

set -euo pipefail

# Balanced thread settings to reduce contention
export PYTORCH_NUM_THREADS=144
export OMP_NUM_THREADS=96
export MKL_NUM_THREADS=96
export OPENBLAS_NUM_THREADS=96
export NUMEXPR_NUM_THREADS=64

# Interop / affinity
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact,1,0

# Summary
echo "[cpu_optimize] PYTORCH=$PYTORCH_NUM_THREADS OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"
echo "[cpu_optimize] Bind policy: OMP_PROC_BIND=$OMP_PROC_BIND, OMP_PLACES=$OMP_PLACES"

echo "[cpu_optimize] To start training:"
echo "python tools/training/train_real_data_ar.py --config \"configs/ar_training_config debug.yaml\" --seeds 42"