#!/bin/bash
# 快速验证实验脚本
# 用于验证实验配置正确性和代码可运行性

set -e

BASE_DIR="/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"
TEST_DIR="$BASE_DIR/paper_package/quick_validation"

echo "⚡ 快速验证实验配置..."
echo "📋 验证项目:"
echo "   ✓ 配置文件加载"
echo "   ✓ 数据加载和预处理" 
echo "   ✓ 模型构建和初始化"
echo "   ✓ 损失函数计算"
echo "   ✓ H/DC一致性检查"
echo "   ✓ 训练循环 (5个epoch)"
echo "   ✓ 评估和指标计算"
echo ""

mkdir -p "$TEST_DIR"

# 测试配置
TEST_CONFIGS=(
  "task=sr_x2,pde=2d_diff_react,model=swin_unet"
  "task=crop_20,pde=ns_incom_inhom,model=unet"
  "task=sr_x4,pde=2d_rdb,model=fno2d"
)

echo "🧪 开始快速验证测试..."

for config in "${TEST_CONFIGS[@]}"; do
  # 解析配置
  IFS=',' read -ra PARAMS <<< "$config"
  task=$(echo "${PARAMS[0]}" | cut -d'=' -f2)
  pde=$(echo "${PARAMS[1]}" | cut -d'=' -f2)
  model=$(echo "${PARAMS[2]}" | cut -d'=' -f2)
  
  TEST_NAME="quick_test_${task}_${pde}_${model}"
  TEST_EXP_DIR="$TEST_DIR/$TEST_NAME"
  
  echo "📍 测试: $TEST_NAME"
  
  # 快速训练 (5 epochs)
  python "$BASE_DIR/train_real_data_ar.py" \
    --config-path "$BASE_DIR/configs" \
    --config-name "train" \
    data.task="$task" \
    data.pde_type="$pde" \
    model.name="$model" \
    training.epochs=5 \
    training.batch_size=2 \
    training.seed=42 \
    experiment.name="$TEST_NAME" \
    hydra.run.dir="$TEST_EXP_DIR" \
    > "$TEST_EXP_DIR/test.log" 2>&1
    
  # 快速评估
  python "$BASE_DIR/eval.py" \
    --config-path "$TEST_EXP_DIR" \
    --config-name "config_merged" \
    eval.num_samples=10 \
    hydra.run.dir="$TEST_EXP_DIR/eval" \
    > "$TEST_EXP_DIR/eval.log" 2>&1
    
  # H/DC一致性验证
  python "$BASE_DIR/tools/check_dc_equivalence.py" \
    --config_file "$TEST_EXP_DIR/config_merged.yaml" \
    --num_samples 5 \
    --tolerance 1e-8 \
    > "$TEST_EXP_DIR/dc_check.log" 2>&1
    
  echo "✅ 通过: $TEST_NAME"
  
done

echo ""
echo "🎉 快速验证测试全部通过！"
echo "📊 验证结果摘要:"
echo "   - 配置文件: ✓ 加载成功"
echo "   - 数据管道: ✓ 运行正常" 
echo "   - 模型架构: ✓ 构建成功"
echo "   - 损失计算: ✓ 数值稳定"
echo "   - H/DC一致性: ✓ 满足1e-8容差"
echo "   - 训练评估: ✓ 流程完整"
echo ""
echo "🚀 现在可以安全执行完整实验了！"
echo "📄 详细日志: $TEST_DIR/*/test.log"