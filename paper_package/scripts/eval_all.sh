#!/bin/bash
# 一键评估脚本
# 使用方法: bash eval_all.sh

set -e

echo "开始评估所有模型..."

# 评估所有实验
python tools/eval.py --runs_dir runs --output_dir paper_package/metrics

# 生成论文材料包
python tools/generate_paper_package.py --runs_dir runs --output_dir paper_package

echo "评估完成！"
