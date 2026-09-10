#!/bin/bash
# 运行 H/DC 一致性检查
# 用法: bash scripts/run_consistency_check.sh

echo "========================================================"
echo "开始运行 H/DC 一致性检查 (Consistency Check)"
echo "========================================================"

# 默认使用主配置文件
CONFIG_PATH="configs/ar_training_config.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Warning: Default config $CONFIG_PATH not found, using tools/check_dc_equivalence.py defaults."
  python tools/check_dc_equivalence.py
else
  echo "Using config: $CONFIG_PATH"
  python tools/check_dc_equivalence.py --config "$CONFIG_PATH"
fi

echo "========================================================"
echo "检查完成！如果 MSE > 1e-8，请检查 ops/degradation.py 实现。"
echo "========================================================"
