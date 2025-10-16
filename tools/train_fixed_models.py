#!/usr/bin/env python3
"""
修复后的批量训练脚本
只训练已修复的模型，跳过有问题的模型
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class ModelTrainer:
    def __init__(self, python_path: str = "F:\\ProgramData\\anaconda3\\python.exe"):
        self.python_path = python_path
        self.base_dir = Path("F:\\Zhaoyang\\Sparse2Full")
        self.results_dir = self.base_dir / "runs" / "batch_training_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 只训练已修复的模型
        self.models = [
            'unet',
            'unet_plus_plus', 
            'fno2d',  # 已修复复数类型问题
            'segformer',  # 新增模型
            'unetformer',  # 新增模型
            'segformer_unetformer',
            'mlp',
            'mlp_mixer',
            'liif',
            'hybrid'
        ]
        
        # 跳过有问题的模型
        self.skip_models = [
            'ufno_unet'  # 可能需要更多修复
        ]
        
        self.epochs = 15
        self.seeds = [2025]
        self.batch_size = 2
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.models),
            'successful_models': 0,
            'failed_models': 0,
            'failed_model_names': [],
            'model_results': {},
            'training_summary': {}
        }
    
    def run_training(self, model_name: str, seed: int, timeout: int = 1800) -> Dict[str, Any]:
        """运行单个模型的训练"""
        exp_name = f"SRx4-DarcyFlow-128-{model_name.upper()}-fixed-s{seed}-{datetime.now().strftime('%Y%m%d')}"
        
        cmd = [
            self.python_path,
            "train.py",
            f"+model={model_name}",
            f"+train.epochs={self.epochs}",
            f"+dataloader.batch_size={self.batch_size}",
            f"experiment.seed={seed}",
            f"experiment.name={exp_name}"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # 设置环境变量避免编码问题
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ 模型 {model_name} 训练成功！")
                print(f"训练时间: {training_time:.2f}秒")
                
                # 解析训练结果
                metrics = self.parse_training_output(result.stdout)
                
                return {
                    'status': 'success',
                    'training_time': training_time,
                    'experiment_name': exp_name,
                    'metrics': metrics,
                    'stdout': result.stdout[-2000:],  # 保留最后2000字符
                    'stderr': result.stderr[-1000:] if result.stderr else ""
                }
            else:
                print(f"✗ 模型 {model_name} 训练失败")
                print(f"错误代码: {result.returncode}")
                print(f"错误信息: {result.stderr}")
                
                return {
                    'status': 'failed',
                    'training_time': training_time,
                    'error_code': result.returncode,
                    'error_message': result.stderr,
                    'stdout': result.stdout[-2000:] if result.stdout else "",
                    'stderr': result.stderr[-2000:] if result.stderr else ""
                }
                
        except subprocess.TimeoutExpired:
            print(f"✗ 模型 {model_name} 训练超时 ({timeout}秒)")
            return {
                'status': 'timeout',
                'training_time': timeout,
                'error_message': f"Training timeout after {timeout} seconds"
            }
        except Exception as e:
            print(f"✗ 模型 {model_name} 训练出现异常: {str(e)}")
            return {
                'status': 'error',
                'training_time': time.time() - start_time,
                'error_message': str(e)
            }
    
    def parse_training_output(self, output: str) -> Dict[str, Any]:
        """解析训练输出，提取关键指标"""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            # 提取最终训练损失
            if 'Final train loss:' in line:
                try:
                    metrics['final_train_loss'] = float(line.split(':')[-1].strip())
                except:
                    pass
            
            # 提取最终验证损失
            if 'Final val loss:' in line:
                try:
                    metrics['final_val_loss'] = float(line.split(':')[-1].strip())
                except:
                    pass
            
            # 提取Rel-L2指标
            if 'rel_l2' in line and 'tensor' in line:
                try:
                    # 简单解析tensor值
                    import re
                    numbers = re.findall(r'[\d.]+', line)
                    if numbers:
                        metrics['final_rel_l2'] = float(numbers[0])
                except:
                    pass
        
        return metrics
    
    def train_all_models(self):
        """批量训练所有模型"""
        print("开始批量训练修复后的模型...")
        print(f"模型列表: {self.models}")
        print(f"跳过模型: {self.skip_models}")
        print(f"训练轮数: {self.epochs}")
        print(f"随机种子: {self.seeds}")
        print()
        
        total_start_time = time.time()
        
        for seed in self.seeds:
            print(f"使用种子 {seed} 训练所有模型...")
            print()
            
            for model_name in self.models:
                print("=" * 60)
                print(f"开始训练模型: {model_name}")
                print("=" * 60)
                
                result = self.run_training(model_name, seed)
                
                # 记录结果
                self.results['model_results'][f"{model_name}_seed_{seed}"] = result
                
                if result['status'] == 'success':
                    self.results['successful_models'] += 1
                else:
                    self.results['failed_models'] += 1
                    self.results['failed_model_names'].append(model_name)
                
                print()
        
        total_time = time.time() - total_start_time
        
        print("=" * 60)
        print("批量训练完成！")
        print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"成功训练: {self.results['successful_models']} 个模型")
        print(f"失败训练: {self.results['failed_models']} 个模型")
        if self.results['failed_model_names']:
            print(f"失败模型: {', '.join(set(self.results['failed_model_names']))}")
        print("=" * 60)
        
        # 保存结果
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """保存训练结果到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"training_results_fixed_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: {results_file}")
    
    def generate_report(self):
        """生成训练报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"training_summary_fixed_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 修复后批量训练结果报告\\n\\n")
            f.write(f"**生成时间**: {self.results['timestamp']}\\n\\n")
            
            f.write("## 训练概况\\n\\n")
            f.write(f"- 总模型数: {self.results['total_models']}\\n")
            f.write(f"- 成功训练: {self.results['successful_models']}\\n")
            f.write(f"- 失败训练: {self.results['failed_models']}\\n")
            f.write(f"- 成功率: {self.results['successful_models']/self.results['total_models']*100:.1f}%\\n\\n")
            
            # 成功的模型
            f.write("## 成功训练的模型\\n\\n")
            f.write("| 模型 | 训练时间(秒) | 最终训练损失 | 最终验证损失 | 最终Rel-L2 |\\n")
            f.write("|------|-------------|-------------|-------------|------------|\\n")
            
            for key, result in self.results['model_results'].items():
                if result['status'] == 'success':
                    model_name = key.split('_seed_')[0]
                    training_time = result.get('training_time', 0)
                    metrics = result.get('metrics', {})
                    train_loss = metrics.get('final_train_loss', 'N/A')
                    val_loss = metrics.get('final_val_loss', 'N/A')
                    rel_l2 = metrics.get('final_rel_l2', 'N/A')
                    
                    f.write(f"| {model_name} | {training_time:.1f} | {train_loss} | {val_loss} | {rel_l2} |\\n")
            
            # 失败的模型
            if self.results['failed_model_names']:
                f.write("\\n## 失败的模型\\n\\n")
                for key, result in self.results['model_results'].items():
                    if result['status'] != 'success':
                        model_name = key.split('_seed_')[0]
                        f.write(f"- **{model_name}**: {result.get('error_message', 'Unknown error')}\\n\\n")
        
        print(f"训练报告已保存到: {report_file}")


def main():
    trainer = ModelTrainer()
    trainer.train_all_models()


if __name__ == "__main__":
    main()