#!/bin/bash
# 论文实验执行计划脚本
# 根据优先级和依赖关系安排实验执行顺序

set -e  # 遇到错误时停止执行

echo "=========================================="
echo "  稀疏观测驱动的时空流场重建实验执行计划"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 实验配置
EXPERIMENT_ROOT="/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"
RESULTS_DIR="$EXPERIMENT_ROOT/paper_package"
LOGS_DIR="$EXPERIMENT_ROOT/thesis_paper/logs"

# 创建日志目录
mkdir -p $LOGS_DIR

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date)" | tee $LOGS_DIR/experiment_schedule.log

echo ""
echo -e "${BLUE}📋 实验执行策略说明：${NC}"
echo "1. 优先级排序：基础实验 → 理论验证 → 时序实验 → 鲁棒性分析"
echo "2. 依赖关系：后续实验依赖前面实验的结果"
echo "3. 时间分配：核心实验优先，可选实验根据时间调整"
echo "4. 质量保证：每阶段都有验证和检查"
echo ""

# 第一阶段：核心基础实验 (最高优先级)
echo -e "${GREEN}🎯 第一阶段：核心基础实验 (预计时间：12-16小时)${NC}"
echo "这些实验是论文的核心支撑，必须优先完成"
echo ""

phase1_experiments() {
    echo -e "${YELLOW}1.1 空间稀疏观测基准实验${NC}"
    echo "任务：SR×2, SR×4, Crop-20%, Crop-40%"
    echo "模型：SwinUNet, UNet, FNO2D, SegFormer"
    echo "数据集：2d_diff_react, ns_incom_inhom, 2d_rdb"
    echo "预计时间：8-10小时"
    echo ""
    
    # 具体实验配置
    declare -a spatial_configs=(
        "configs/experiment/spatial_sparse_sr2.yaml"
        "configs/experiment/spatial_sparse_sr4.yaml"
        "configs/experiment/spatial_sparse_crop20.yaml"
        "configs/experiment/spatial_sparse_crop40.yaml"
    )
    
    declare -a models=("swin_unet" "unet" "fno2d" "segformer")
    declare -a datasets=("2d_diff_react" "ns_incom_inhom" "2d_rdb")
    seeds=(42 123 456)
    
    total_experiments=$((${#spatial_configs[@]} * ${#models[@]} * ${#datasets[@]} * ${#seeds[@]}))
    echo "总实验数量：$total_experiments 组"
    echo ""
    
    # 执行计划（示例，实际需要根据真实配置调整）
    echo "执行顺序："
    echo "1. 先执行SR×2任务（最简单，快速验证）"
    echo "2. 然后执行SR×4任务（主要对比）"
    echo "3. 接着执行Crop-20%任务"
    echo "4. 最后执行Crop-40%任务（最困难）"
    echo ""
    
    echo -e "${YELLOW}1.2 与SOTA方法对比实验${NC}"
    echo "对比：Senseiver, PINTO, SINO"
    echo "预计时间：3-4小时"
    echo ""
    
    echo -e "${YELLOW}1.3 H/DC一致性验证${NC}"
    echo "验证观测算子与数据一致性"
    echo "预计时间：30分钟"
    echo ""
}

# 第二阶段：理论验证实验 (高优先级)
echo -e "${GREEN}🎯 第二阶段：理论验证实验 (预计时间：6-8小时)${NC}"
echo "支撑论文第4-5章的理论分析"
echo ""

phase2_experiments() {
    echo -e "${YELLOW}2.1 损失函数权重消融实验${NC}"
    echo "验证：L_rec, L_spec, L_dc 的作用"
    echo "配置：基准、无频域、无DC、强化频域、强化DC"
    echo "预计时间：3-4小时"
    echo ""
    
    echo -e "${YELLOW}2.2 Kolmogorov宽度验证${NC}"
    echo "网络宽度：32, 64, 128, 256, 512"
    echo "验证收敛率 O(width^(-k/d))"
    echo "预计时间：1-2小时"
    echo ""
    
    echo -e "${YELLOW}2.3 信息恢复下界验证${NC}"
    echo "观测密度：64, 128, 256, 512, 1024"
    echo "验证 C₁/√n ≤ ||f̂ - f*||₂ ≤ C₂/√n"
    echo "预计时间：1-2小时"
    echo ""
}

# 第三阶段：时序实验 (中等优先级)
echo -e "${GREEN}🎯 第三阶段：时序实验 (预计时间：4-6小时)${NC}"
echo "支撑时间稀疏观测和NAR方法"
echo ""

phase3_experiments() {
    echo -e "${YELLOW}3.1 AR vs NAR对比实验${NC}"
    echo "比较自回归和非自回归模式"
    echo "验证误差累积率：NAR 5% vs AR 62%"
    echo "预计时间：2-3小时"
    echo ""
    
    echo -e "${YELLOW}3.2 时间稀疏观测实验${NC}"
    echo "时间采样：TS25, TS50, TS75"
    echo "验证并行推理速度提升"
    echo "预计时间：1-2小时"
    echo ""
    
    echo -e "${YELLOW}3.3 Lyapunov稳定性验证${NC}"
    echo "长序列预测稳定性分析"
    echo "验证 λ = -0.087"
    echo "预计时间：1小时"
    echo ""
}

# 第四阶段：鲁棒性分析 (可选优先级)
echo -e "${GREEN}🎯 第四阶段：鲁棒性分析 (预计时间：3-4小时)${NC}"
echo "增强论文完整性和说服力"
echo ""

phase4_experiments() {
    echo -e "${YELLOW}4.1 噪声鲁棒性实验${NC}"
    echo "噪声类型：高斯、椒盐、量化"
    echo "验证5%噪声下性能下降<10%"
    echo "预计时间：2小时"
    echo ""
    
    echo -e "${YELLOW}4.2 边界条件影响分析${NC}"
    echo "边界模式：mirror, zero, wrap, extend"
    echo "验证mirror模式最适合流体问题"
    echo "预计时间：1-2小时"
    echo ""
}

# 显示完整的实验执行计划
echo -e "${BLUE}📊 完整实验执行计划${NC}"
echo ""

phase1_experiments
echo "---"
phase2_experiments
echo "---"
phase3_experiments
echo "---"
phase4_experiments

echo ""
echo -e "${BLUE}⏰ 总体时间估算${NC}"
echo "第一阶段：12-16小时 (必须完成)"
echo "第二阶段：6-8小时 (必须完成)"
echo "第三阶段：4-6小时 (推荐完成)"
echo "第四阶段：3-4小时 (可选完成)"
echo "总计：25-34小时 (基础) → 28-38小时 (完整)"
echo ""

echo -e "${BLUE}🚨 重要提醒${NC}"
echo "1. 每阶段完成后都要进行质量检查和验证"
echo "2. 遇到实验失败要及时调整参数或配置"
echo "3. 保持详细的实验日志和记录"
echo "4. 定期备份实验结果和中间文件"
echo "5. 及时与导师沟通实验进展和问题"
echo ""

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "计划生成耗时：$DURATION 秒" | tee -a $LOGS_DIR/experiment_schedule.log
echo "计划文件保存在：$LOGS_DIR/experiment_schedule.log"

echo ""
echo -e "${GREEN}✅ 实验执行计划生成完成！${NC}"
echo "建议按照上述优先级顺序执行实验"
echo "具体实验命令请参考：thesis_paper/supplementary/experiment_scripts.md"