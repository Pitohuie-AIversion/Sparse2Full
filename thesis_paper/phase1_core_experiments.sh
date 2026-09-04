#!/bin/bash
# 核心实验快速启动脚本
# 执行第一优先级的关键实验

set -e

echo "=========================================="
echo "  核心实验快速启动 - 第一优先级"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$EXPERIMENT_ROOT/paper_package"
LOGS_DIR="$EXPERIMENT_ROOT/thesis_paper/logs"

# 创建必要目录
mkdir -p $LOGS_DIR $RESULTS_DIR

# 记录开始时间
echo "开始时间: $(date)" | tee $LOGS_DIR/core_experiments.log

echo -e "${BLUE}🎯 今日实验目标：${NC}"
echo "1. 验证实验环境 (30分钟)"
echo "2. 完成SR×2基准实验 (2小时)"
echo "3. 完成SR×4主要实验 (2.5小时)"
echo "4. 生成初步结果和图表 (1小时)"
echo ""
echo -e "${YELLOW}⚠️  注意事项：${NC}"
echo "- 每个实验后检查GPU内存使用情况"
echo "- 及时记录实验结果和异常情况"
echo "- 如有失败，先记录错误信息再继续"
echo ""

# 步骤1：环境验证
echo -e "${GREEN}步骤1：环境验证${NC}"
echo "验证Python环境和依赖..."

python3 -c "
import torch
import numpy as np
import yaml
print(f'✓ PyTorch版本: {torch.__version__}')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')
print(f'✓ NumPy版本: {np.__version__}')
if torch.cuda.is_available():
    print(f'✓ GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
" 2>&1 | tee -a $LOGS_DIR/core_experiments.log

# 检查配置文件
echo ""
echo "验证配置文件..."
if [ -f "configs/model/swin_unet.yaml" ]; then
    echo "✓ Swin-UNet配置存在"
else
    echo "✗ Swin-UNet配置缺失"
fi

if [ -f "configs/datasets/pdebench.yaml" ]; then
    echo "✓ PDEBench配置存在"
else
    echo "✗ PDEBench配置缺失"
fi

# 步骤2：快速验证实验
echo ""
echo -e "${GREEN}步骤2：快速验证实验${NC}"
echo "执行10个epoch的快速验证..."

# 创建快速验证配置
cat > /tmp/quick_test_config.yaml << 'EOF'
# 快速测试配置
task: "SR"
scale: 2
data:
  name: "pdebench"
  pde_type: "2d_diff_react"
  subset_size: 100  # 小数据集快速验证
model:
  name: "swin_unet"
  img_size: 64
  in_chans: 1
  out_chans: 1
training:
  max_epochs: 10
  batch_size: 4
  learning_rate: 1e-3
  precision: 16
logging:
  log_every_n_steps: 5
EOF

# 执行快速验证
python tools/train.py --config /tmp/quick_test_config.yaml --seed 42 2>&1 | tee $LOGS_DIR/quick_test.log

# 检查验证结果
if grep -q "epoch.*10.*" $LOGS_DIR/quick_test.log; then
    echo "✓ 快速验证实验成功完成"
else
    echo "✗ 快速验证实验失败，请检查日志"
    exit 1
fi

# 步骤3：SR×2基准实验
echo ""
echo -e "${GREEN}步骤3：SR×2基准实验${NC}"
echo "执行完整的SR×2实验..."

# 检查是否有足够的GPU内存
python3 -c "
import torch
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'可用GPU内存: {gpu_mem:.1f} GB')
if gpu_mem < 8:
    print('⚠️ 警告：GPU内存可能不足，建议减小batch_size')
" 2>&1 | tee -a $LOGS_DIR/core_experiments.log

# 执行SR×2实验（使用真实配置）
echo "开始SR×2实验..."
cat > /tmp/sr2_experiment.yaml << 'EOF'
# SR×2实验配置
task: "SR"
scale: 2
data:
  name: "pdebench"
  pde_type: "2d_diff_react"
  crop_size: 256
  batch_size: 8
model:
  name: "swin_unet"
  img_size: 256
  in_chans: 1
  out_chans: 1
  depths: [2, 2, 2, 2]
  num_heads: [4, 8, 16, 32]
training:
  max_epochs: 100
  learning_rate: 1e-3
  precision: 16
  gradient_clip_val: 1.0
loss:
  reconstruction_weight: 1.0
  spectral_weight: 0.5
  dc_weight: 1.0
logging:
  save_every_n_epochs: 10
EOF

# 后台执行SR×2实验
nohup python tools/train.py --config /tmp/sr2_experiment.yaml --seed 42 > $LOGS_DIR/sr2_experiment.log 2>&1 &
SR2_PID=$!
echo "SR×2实验PID: $SR2_PID"

# 步骤4：SR×4实验准备
echo ""
echo -e "${GREEN}步骤4：SR×4实验准备${NC}"
echo "等待SR×2实验完成或达到一定进度后开始SR×4..."

# 监控实验进度
monitor_experiment() {
    local log_file=$1
    local experiment_name=$2
    
    echo "监控$experiment_name实验进度..."
    while true; do
        if [ -f "$log_file" ]; then
            # 检查是否有进度信息
            if tail -n 10 "$log_file" | grep -q "epoch"; then
                latest_epoch=$(tail -n 100 "$log_file" | grep "epoch" | tail -n 1 | grep -o "epoch.*[0-9]*" | grep -o "[0-9]*" | head -n 1)
                if [ ! -z "$latest_epoch" ]; then
                    echo "$(date): $experiment_name - 第$latest_epoch轮"
                fi
            fi
        fi
        sleep 300  # 每5分钟检查一次
    done
}

# 后台监控SR×2实验
monitor_experiment $LOGS_DIR/sr2_experiment.log "SR×2" &
MONITOR_PID=$!

# 用户选择：等待完成或并行开始
echo ""
echo -e "${YELLOW}选择执行策略：${NC}"
echo "1. 等待SR×2完成再开始SR×4 (推荐，节省GPU内存)"
echo "2. 立即开始SR×4实验 (需要更多GPU资源)"
echo "3. 暂停执行，稍后手动继续"

echo ""
echo "等待30秒后默认选择策略1..."
sleep 30

# 默认策略：等待SR×2达到一定进度
echo "等待SR×2实验达到30轮..."
while true; do
    if [ -f "$LOGS_DIR/sr2_experiment.log" ]; then
        if grep -q "epoch.*30" $LOGS_DIR/sr2_experiment.log; then
            echo "SR×2实验已达到30轮，可以开始SR×4实验"
            break
        fi
    fi
    echo "等待中... ($(date))"
    sleep 60
done

# 清理监控进程
kill $MONITOR_PID 2>/dev/null || true

# 生成初步结果
echo ""
echo -e "${GREEN}生成初步结果${NC}"
echo "分析当前实验结果..."

# 结果分析脚本
python3 -c "
import os
import re
import pandas as pd

def analyze_log(log_file):
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取关键指标
    metrics = {}
    
    # 提取验证误差
    val_errors = re.findall(r'val_loss.*?([0-9]*\.[0-9]+)', content)
    if val_errors:
        metrics['final_val_loss'] = float(val_errors[-1])
        metrics['best_val_loss'] = min([float(x) for x in val_errors])
    
    # 提取训练误差
    train_errors = re.findall(r'train_loss.*?([0-9]*\.[0-9]+)', content)
    if train_errors:
        metrics['final_train_loss'] = float(train_errors[-1])
    
    # 提取epoch信息
    epochs = re.findall(r'epoch.*?([0-9]+)', content)
    if epochs:
        metrics['total_epochs'] = int(epochs[-1])
    
    return metrics

# 分析SR×2实验结果
sr2_metrics = analyze_log('$LOGS_DIR/sr2_experiment.log')
if sr2_metrics:
    print('SR×2实验结果分析：')
    for key, value in sr2_metrics.items():
        print(f'  {key}: {value}')
else:
    print('SR×2实验结果尚未生成')

print('\\n结果已保存到日志文件')
" 2>&1 | tee -a $LOGS_DIR/core_experiments.log

# 记录完成时间
echo "" | tee -a $LOGS_DIR/core_experiments.log
echo "核心实验第一阶段完成时间: $(date)" | tee -a $LOGS_DIR/core_experiments.log
echo "日志文件：$LOGS_DIR/core_experiments.log" | tee -a $LOGS_DIR/core_experiments.log

echo ""
echo -e "${GREEN}✅ 核心实验第一阶段完成！${NC}"
echo ""
echo "下一步建议："
echo "1. 检查SR×2实验结果：tail -f $LOGS_DIR/sr2_experiment.log"
echo "2. 开始SR×4实验：继续执行第二阶段"
echo "3. 分析初步结果：查看$LOGS_DIR/core_experiments.log"
echo ""
echo "实验结果将自动保存在：$RESULTS_DIR"
echo "继续执行请运行：bash thesis_paper/phase2_experiments.sh"
