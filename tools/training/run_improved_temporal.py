#!/usr/bin/env python3
"""
改进的时序增强训练脚本
基于路径A改进：更合理的时序一致性损失 + 动态权重调度
"""

import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import logging
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.training.train_temporal_enhanced import TemporalEnhancedTrainer
from utils.logging_utils import setup_logger

def main():
    """主函数"""
    
    # 设置日志
    setup_logger(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 启动改进的时序增强训练")
    logger.info("📋 改进内容：")
    logger.info("   ✓ 导数一致性损失 - 对齐变化模式而非简单平滑")
    logger.info("   ✓ 能量变化一致性 - 匹配物理能量演化")
    logger.info("   ✓ 曲率一致性损失 - 对齐加速度变化")
    logger.info("   ✓ 动态权重调度 - 课程学习权重自适应")
    logger.info("   ✓ 改进的稳定性损失 - 多维度稳定性检测")
    logger.info("   ✓ AR roll-out一致性 - 分段约束策略")
    
    # 加载配置
    config_path = project_root / "configs" / "improved_temporal_enhanced.yaml"
    config = OmegaConf.load(config_path)
    
    # 添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.experiment.name = f"{config.experiment.name}_{timestamp}"
    
    logger.info(f"📊 实验名称: {config.experiment.name}")
    logger.info(f"📈 训练轮数: {config.training.epochs}")
    logger.info(f"🎯 预测模式: {config.model.prediction_mode}")
    logger.info(f"⏱️  输入时序: {config.data.T_in} → 输出时序: {config.data.T_out}")
    
    # 创建训练器
    try:
        trainer = TemporalEnhancedTrainer(config)
        logger.info("✅ 训练器创建成功")
        
        # 开始训练
        logger.info("🎯 开始训练...")
        trainer.train()
        
        logger.info("✅ 训练完成！")
        
        # 保存最终模型
        final_model_path = trainer.save_checkpoint("final_model.pth")
        logger.info(f"💾 最终模型已保存: {final_model_path}")
        
        # 运行最终验证
        logger.info("🔍 运行最终验证...")
        final_metrics = trainer.final_test()
        
        logger.info("📊 最终验证结果:")
        for metric_name, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"   {metric_name}: {value:.6f}")
            else:
                logger.info(f"   {metric_name}: {value}")
        
        # 生成训练报告
        report_path = trainer.generate_training_report()
        logger.info(f"📄 训练报告已生成: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        raise e
    
    logger.info("🎉 改进的时序增强训练完成！")

if __name__ == "__main__":
    main()