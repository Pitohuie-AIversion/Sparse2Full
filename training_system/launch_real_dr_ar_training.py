#!/usr/bin/env python3
"""
真实扩散-反应数据AR训练启动脚本
基于training_system框架，支持20步自回归预测
遵循项目开发规范：一致性、可复现、统一接口、完整监控
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="真实扩散-反应数据AR训练启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 基本训练
    python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar
    
    # 多GPU训练
    python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar --multirun
    
    # 指定输出目录
    python launch_real_dr_ar_training.py --config-name train_real_dr_data_ar --output-dir runs/ar_experiment
    
    # 覆盖特定参数
    python launch_real_dr_data_ar_training.py --config-name train_real_dr_data_ar experiment.name=MyARExperiment training.epochs=300
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--config-name", "-c",
        type=str,
        default="train_real_dr_data_ar",
        help="配置文件名称（位于configs/basic/目录下）"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="runs",
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--multirun", "-m",
        action="store_true",
        help="启用多运行模式（用于超参数搜索或多种子训练）"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2025],
        help="随机种子列表"
    )
    
    # 实验参数覆盖（通过未知参数处理）
    # Hydra参数通过命令行直接传递，不需要单独定义
    
    # 调试参数
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用调试模式（详细日志）"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="干运行模式（只打印命令，不执行）"
    )
    
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="检查配置文件有效性"
    )
    
    # 解析参数
    args, unknown_args = parser.parse_known_args()
    
    # 构建Hydra命令
    cmd = [
        sys.executable,
        "scripts/train.py"
    ]
    
    # 基本配置
    cmd.extend(["--config-name", args.config_name])
    
    # 输出目录
    cmd.extend(["experiment.output_dir", args.output_dir])
    
    # 多运行模式
    if args.multirun:
        cmd.insert(1, "-m")  # 在python之后插入
    
    # 种子配置
    if len(args.seeds) > 1 or args.seeds[0] != 2025:
        cmd.extend([f"experiment.seeds", f"{args.seeds}"])
        cmd.extend(["experiment.use_multi_seeds", "true"])
    
    # Hydra参数通过未知参数直接传递，无需特殊处理
    
    # 处理未知参数（Hydra格式）
    for unknown_arg in unknown_args:
        if "=" in unknown_arg:
            cmd.extend(unknown_arg.split("="))
        else:
            logger.warning(f"忽略未知参数: {unknown_arg}")
    
    # 调试模式
    if args.debug:
        cmd.extend(["logging.log_level", "DEBUG"])
    
    # 检查配置模式
    if args.check_config:
        cmd.extend(["--help"])
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    # 干运行模式
    if args.dry_run:
        logger.info("干运行模式 - 不执行实际训练")
        print("命令:", " ".join(cmd))
        return 0
    
    # 设置环境变量
    env = os.environ.copy()
    
    # 确保使用正确的Python路径
    training_system_dir = Path(__file__).parent
    if str(training_system_dir) not in sys.path:
        sys.path.insert(0, str(training_system_dir))
    
    # 添加项目根目录到PYTHONPATH
    project_root = training_system_dir
    env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")
    
    # 执行训练
    try:
        logger.info("开始训练...")
        result = subprocess.run(cmd, env=env, check=True, cwd=str(project_root))
        logger.info("训练完成！")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        logger.error(f"训练失败，返回码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        return 130
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)