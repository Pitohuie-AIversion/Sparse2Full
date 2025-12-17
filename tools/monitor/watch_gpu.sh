#!/usr/bin/env bash
set -euo pipefail
INTERVAL=${1:-2}
# 输出卡号/显存使用/总显存/利用率/温度/功耗
CMD="nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,nounits,noheader"
watch -n "$INTERVAL" "$CMD"

