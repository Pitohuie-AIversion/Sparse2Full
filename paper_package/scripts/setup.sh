#!/bin/bash
# 环境安装脚本
# 使用方法: bash setup.sh

set -e

echo "安装PDEBench稀疏观测重建系统环境..."

# 创建conda环境
conda create -n pdebench python=3.10 -y
conda activate pdebench

# 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他依赖
pip install -r requirements.txt

echo "环境安装完成！"
echo "使用 'conda activate pdebench' 激活环境"
