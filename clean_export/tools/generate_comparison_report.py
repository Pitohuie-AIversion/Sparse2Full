#!/usr/bin/env python3
"""
生成模型性能对比报告
从训练结果中提取指标并生成横向对比分析表格
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """设置日志"""
    log_dir = Path("runs/comparison_reports")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"comparison_report_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def find_latest_batch_results() -> Path:
    """查找最新的批量训练结果文件"""
    results_dir = Path("runs/batch_training_results")
    
    if not results_dir.exists():
        raise FileNotFoundError("批量训练结果目录不存在")
    
    # 查找所有结果文件
    result_files = list(results_dir.glob("simple_batch_results_*.json"))
    
    if not result_files:
        raise FileNotFoundError("未找到批量训练结果文件")
    
    # 返回最新的文件
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    return latest_file


def load_batch_results(results_file: Path) -> Dict:
    """加载批量训练结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_model_metrics(exp_dir: Path) -> Dict[str, Any]:
    """从实验目录中提取模型指标"""
    metrics = {}
    
    # 查找最佳模型检查点
    checkpoint_dir = exp_dir / "checkpoints"
    if checkpoint_dir.exists():
        best_ckpt = checkpoint_dir / "best_model.pth"
        if best_ckpt.exists():
            metrics["has_checkpoint"] = True
        else:
            metrics["has_checkpoint"] = False
    
    # 查找训练日志
    log_dir = exp_dir / "logs"
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            # 从日志中提取最佳验证指标
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            metrics.update(parse_training_log(latest_log))
    
    return metrics


def parse_training_log(log_file: Path) -> Dict[str, Any]:
    """解析训练日志文件，提取关键指标"""
    metrics = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找最佳验证损失
        if "Best validation loss:" in content:
            lines = content.split('\n')
            for line in lines:
                if "Best validation loss:" in line:
                    try:
                        loss_str = line.split("Best validation loss:")[1].strip()
                        metrics["best_val_loss"] = float(loss_str)
                    except:
                        pass
        
        # 查找训练时间
        if "Total training time:" in content:
            lines = content.split('\n')
            for line in lines:
                if "Total training time:" in line:
                    try:
                        time_str = line.split("Total training time:")[1].strip().replace('s', '')
                        metrics["training_time"] = float(time_str)
                    except:
                        pass
        
        # 查找最佳验证指标
        if "Best validation metrics:" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "Best validation metrics:" in line:
                    # 尝试解析指标字典
                    try:
                        # 查找包含指标的行
                        metrics_lines = []
                        for j in range(i, min(i+20, len(lines))):
                            if "rel_l2" in lines[j] or "mae" in lines[j] or "psnr" in lines[j]:
                                metrics_lines.append(lines[j])
                        
                        # 简单解析一些关键指标
                        for metrics_line in metrics_lines:
                            if "rel_l2" in metrics_line:
                                # 提取rel_l2值
                                try:
                                    import re
                                    rel_l2_match = re.search(r'rel_l2.*?(\d+\.\d+)', metrics_line)
                                    if rel_l2_match:
                                        metrics["rel_l2"] = float(rel_l2_match.group(1))
                                except:
                                    pass
                            
                            if "psnr" in metrics_line:
                                # 提取PSNR值
                                try:
                                    import re
                                    psnr_match = re.search(r'psnr.*?(\d+\.\d+)', metrics_line)
                                    if psnr_match:
                                        metrics["psnr"] = float(psnr_match.group(1))
                                except:
                                    pass
                    except:
                        pass
                    break
    
    except Exception as e:
        logging.warning(f"解析日志文件失败 {log_file}: {e}")
    
    return metrics


def count_model_parameters(model_name: str) -> int:
    """估算模型参数量（基于模型类型的经验值）"""
    # 这里使用经验值，实际应该从模型定义中计算
    param_estimates = {
        "unet": 7.8e6,
        "unet_plus_plus": 9.2e6,
        "fno2d": 2.3e6,
        "ufno_unet": 5.1e6,
        "segformer_unetformer": 12.5e6,
        "unetformer": 8.7e6,
        "mlp": 1.2e6,
        "mlp_mixer": 3.4e6,
        "liif": 0.8e6,
        "hybrid": 15.2e6,
        "segformer": 14.1e6,
        "swin_unet": 27.2e6,
    }
    
    return param_estimates.get(model_name, 0)


def generate_comparison_table(batch_results: Dict) -> pd.DataFrame:
    """生成模型对比表格"""
    logger = logging.getLogger(__name__)
    
    results = batch_results["results"]
    successful_results = [r for r in results if r["status"] == "success"]
    
    logger.info(f"成功训练的模型数量: {len(successful_results)}")
    
    # 按模型分组
    model_groups = {}
    for result in successful_results:
        model_name = result["model"]
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(result)
    
    # 构建对比表格数据
    table_data = []
    
    for model_name, model_results in model_groups.items():
        # 计算统计信息
        train_times = [r["train_time"] for r in model_results]
        
        # 尝试从实验目录提取指标
        metrics_list = []
        for result in model_results:
            if "exp_name" in result:
                exp_dir = Path("runs") / result["exp_name"]
                if exp_dir.exists():
                    metrics = extract_model_metrics(exp_dir)
                    metrics_list.append(metrics)
        
        # 计算平均指标
        avg_metrics = {}
        if metrics_list:
            for key in ["best_val_loss", "rel_l2", "psnr", "training_time"]:
                values = [m.get(key) for m in metrics_list if m.get(key) is not None]
                if values:
                    avg_metrics[f"avg_{key}"] = np.mean(values)
                    avg_metrics[f"std_{key}"] = np.std(values)
        
        # 构建表格行
        row = {
            "Model": model_name,
            "Seeds": len(model_results),
            "Success_Rate": f"{len(model_results)}/3",
            "Avg_Train_Time(s)": f"{np.mean(train_times):.1f}±{np.std(train_times):.1f}",
            "Est_Params(M)": f"{count_model_parameters(model_name)/1e6:.1f}",
        }
        
        # 添加性能指标
        if "avg_best_val_loss" in avg_metrics:
            row["Val_Loss"] = f"{avg_metrics['avg_best_val_loss']:.4f}±{avg_metrics['std_best_val_loss']:.4f}"
        
        if "avg_rel_l2" in avg_metrics:
            row["Rel_L2"] = f"{avg_metrics['avg_rel_l2']:.4f}±{avg_metrics['std_rel_l2']:.4f}"
        
        if "avg_psnr" in avg_metrics:
            row["PSNR"] = f"{avg_metrics['avg_psnr']:.1f}±{avg_metrics['std_psnr']:.1f}"
        
        table_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 按参数量排序
    df = df.sort_values("Est_Params(M)", key=lambda x: x.str.extract(r'(\d+\.?\d*)').astype(float))
    
    return df


def generate_failure_analysis(batch_results: Dict) -> pd.DataFrame:
    """生成失败分析表格"""
    results = batch_results["results"]
    failed_results = [r for r in results if r["status"] != "success"]
    
    if not failed_results:
        return pd.DataFrame({"Message": ["所有模型训练成功！"]})
    
    failure_data = []
    for result in failed_results:
        row = {
            "Model": result["model"],
            "Seed": result["seed"],
            "Status": result["status"],
            "Error": result.get("error", "Unknown"),
            "Train_Time(s)": f"{result.get('train_time', 0):.1f}",
        }
        failure_data.append(row)
    
    return pd.DataFrame(failure_data)


def save_reports(comparison_df: pd.DataFrame, failure_df: pd.DataFrame, batch_results: Dict):
    """保存报告文件"""
    output_dir = Path("runs/comparison_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存CSV文件
    comparison_csv = output_dir / f"model_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8-sig')
    
    failure_csv = output_dir / f"failure_analysis_{timestamp}.csv"
    failure_df.to_csv(failure_csv, index=False, encoding='utf-8-sig')
    
    # 生成Markdown报告
    markdown_file = output_dir / f"comparison_report_{timestamp}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(f"# 模型性能对比报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体统计
        total_tasks = batch_results["summary"]["total"]
        successful_tasks = batch_results["summary"]["successful"]
        failed_tasks = batch_results["summary"]["failed"]
        
        f.write(f"## 总体统计\n\n")
        f.write(f"- 总训练任务: {total_tasks}\n")
        f.write(f"- 成功任务: {successful_tasks}\n")
        f.write(f"- 失败任务: {failed_tasks}\n")
        f.write(f"- 成功率: {successful_tasks/total_tasks*100:.1f}%\n\n")
        
        # 模型对比表格
        f.write(f"## 模型性能对比\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")
        
        # 失败分析
        if not failure_df.empty and "Message" not in failure_df.columns:
            f.write(f"## 失败分析\n\n")
            f.write(failure_df.to_markdown(index=False))
            f.write("\n\n")
        
        # 说明
        f.write(f"## 说明\n\n")
        f.write(f"- **Seeds**: 成功训练的种子数量\n")
        f.write(f"- **Success_Rate**: 成功率 (成功数/总数)\n")
        f.write(f"- **Avg_Train_Time**: 平均训练时间±标准差 (秒)\n")
        f.write(f"- **Est_Params**: 估算参数量 (百万)\n")
        f.write(f"- **Val_Loss**: 验证损失 (均值±标准差)\n")
        f.write(f"- **Rel_L2**: 相对L2误差 (均值±标准差)\n")
        f.write(f"- **PSNR**: 峰值信噪比 (均值±标准差)\n")
    
    return comparison_csv, failure_csv, markdown_file


def main():
    """主函数"""
    logger = setup_logging()
    
    try:
        # 查找最新的批量训练结果
        results_file = find_latest_batch_results()
        logger.info(f"找到批量训练结果文件: {results_file}")
        
        # 加载结果
        batch_results = load_batch_results(results_file)
        logger.info(f"加载了 {batch_results['summary']['total']} 个训练任务的结果")
        
        # 生成对比表格
        comparison_df = generate_comparison_table(batch_results)
        logger.info(f"生成了 {len(comparison_df)} 个模型的对比表格")
        
        # 生成失败分析
        failure_df = generate_failure_analysis(batch_results)
        
        # 保存报告
        csv_file, failure_file, md_file = save_reports(comparison_df, failure_df, batch_results)
        
        logger.info(f"报告已保存:")
        logger.info(f"  对比表格: {csv_file}")
        logger.info(f"  失败分析: {failure_file}")
        logger.info(f"  Markdown报告: {md_file}")
        
        # 打印对比表格
        print("\n" + "="*80)
        print("模型性能对比表格")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        if not failure_df.empty and "Message" not in failure_df.columns:
            print("\n" + "="*80)
            print("失败分析")
            print("="*80)
            print(failure_df.to_string(index=False))
        
    except Exception as e:
        logger.error(f"生成对比报告失败: {e}")
        raise


if __name__ == "__main__":
    main()