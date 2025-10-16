#!/usr/bin/env python3
"""
批量训练所有模型的脚本
依次训练所有12个已验证的模型，生成对比报告
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class ModelTrainer:
    """模型训练管理器"""
    
    def __init__(self):
        self.python_exe = "F:\\ProgramData\\anaconda3\\python.exe"
        self.models = [
            "unet",
            "unet_plus_plus", 
            "fno2d",
            "ufno_unet",
            "segformer_unetformer",
            "unetformer",
            "mlp",
            "mlp_mixer",
            "liif",
            "hybrid",
            "segformer"
        ]
        self.results = []
        self.failed_models = []
        
    def train_single_model(self, model_name: str, epochs: int = 15, seed: int = 2025) -> Dict[str, Any]:
        """训练单个模型"""
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_name}")
        print(f"{'='*60}")
        
        timestamp = datetime.now().strftime("%Y%m%d")
        exp_name = f"SRx4-DarcyFlow-128-{model_name.upper()}-batch-s{seed}-{timestamp}"
        
        # 构建训练命令
        cmd = [
            self.python_exe, "train.py",
            f"+model={model_name}",
            f"+train.epochs={epochs}",
            f"+dataloader.batch_size=2",
            f"experiment.seed={seed}",
            f"experiment.name={exp_name}"
        ]
        
        start_time = time.time()
        
        try:
            # 执行训练命令
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"✓ 模型 {model_name} 训练成功！")
                print(f"训练时间: {training_time:.2f}秒")
                
                # 记录成功结果
                model_result = {
                    "model": model_name,
                    "status": "success",
                    "training_time": training_time,
                    "epochs": epochs,
                    "seed": seed,
                    "experiment_name": exp_name,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 尝试解析训练日志获取最终指标
                try:
                    output_lines = result.stdout.split('\n')
                    for line in reversed(output_lines):
                        if "Val Rel-L2:" in line:
                            # 提取最终验证指标
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "Loss:" and i+1 < len(parts):
                                    model_result["final_train_loss"] = float(parts[i+1])
                                elif part == "Loss:" and i+1 < len(parts):
                                    model_result["final_val_loss"] = float(parts[i+1])
                                elif part == "Rel-L2:" and i+1 < len(parts):
                                    model_result["final_rel_l2"] = float(parts[i+1])
                            break
                except:
                    pass
                
                return model_result
                
            else:
                print(f"✗ 模型 {model_name} 训练失败！")
                print(f"错误代码: {result.returncode}")
                print(f"错误信息: {result.stderr}")
                
                # 记录失败结果
                model_result = {
                    "model": model_name,
                    "status": "failed",
                    "training_time": training_time,
                    "error_code": result.returncode,
                    "error_message": result.stderr,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.failed_models.append(model_name)
                return model_result
                
        except subprocess.TimeoutExpired:
            print(f"✗ 模型 {model_name} 训练超时！")
            model_result = {
                "model": model_name,
                "status": "timeout",
                "training_time": 3600,
                "error_message": "Training timeout after 1 hour",
                "timestamp": datetime.now().isoformat()
            }
            self.failed_models.append(model_name)
            return model_result
            
        except Exception as e:
            print(f"✗ 模型 {model_name} 训练出现异常: {e}")
            model_result = {
                "model": model_name,
                "status": "error",
                "training_time": time.time() - start_time,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.failed_models.append(model_name)
            return model_result
    
    def train_all_models(self, epochs: int = 15, seeds: List[int] = [2025]) -> None:
        """训练所有模型"""
        print("开始批量训练所有模型...")
        print(f"模型列表: {self.models}")
        print(f"训练轮数: {epochs}")
        print(f"随机种子: {seeds}")
        
        total_start_time = time.time()
        
        for seed in seeds:
            print(f"\n使用种子 {seed} 训练所有模型...")
            
            for model_name in self.models:
                result = self.train_single_model(model_name, epochs, seed)
                self.results.append(result)
                
                # 每个模型训练完后稍作休息
                time.sleep(2)
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*60}")
        print("批量训练完成！")
        print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"成功训练: {len([r for r in self.results if r['status'] == 'success'])} 个模型")
        print(f"失败训练: {len(self.failed_models)} 个模型")
        if self.failed_models:
            print(f"失败模型: {', '.join(self.failed_models)}")
        print(f"{'='*60}")
        
        # 保存结果
        self.save_results()
    
    def save_results(self) -> None:
        """保存训练结果"""
        results_dir = Path("runs/batch_training_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = results_dir / f"training_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.models),
                "successful_models": len([r for r in self.results if r['status'] == 'success']),
                "failed_models": len(self.failed_models),
                "failed_model_names": self.failed_models,
                "results": self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: {results_file}")
        
        # 生成简要报告
        self.generate_summary_report(results_dir / f"training_summary_{timestamp}.md")
    
    def generate_summary_report(self, output_file: Path) -> None:
        """生成简要报告"""
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 批量训练结果报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 训练概况\n\n")
            f.write(f"- 总模型数: {len(self.models)}\n")
            f.write(f"- 成功训练: {len(successful_results)}\n")
            f.write(f"- 失败训练: {len(self.failed_models)}\n")
            f.write(f"- 成功率: {len(successful_results)/len(self.models)*100:.1f}%\n\n")
            
            if successful_results:
                f.write("## 成功训练的模型\n\n")
                f.write("| 模型 | 训练时间(秒) | 最终训练损失 | 最终验证损失 | 最终Rel-L2 |\n")
                f.write("|------|-------------|-------------|-------------|------------|\n")
                
                for result in successful_results:
                    train_loss = result.get('final_train_loss', 'N/A')
                    val_loss = result.get('final_val_loss', 'N/A')
                    rel_l2 = result.get('final_rel_l2', 'N/A')
                    
                    f.write(f"| {result['model']} | {result['training_time']:.1f} | {train_loss} | {val_loss} | {rel_l2} |\n")
            
            if self.failed_models:
                f.write("\n## 失败的模型\n\n")
                failed_results = [r for r in self.results if r['status'] != 'success']
                for result in failed_results:
                    f.write(f"- **{result['model']}**: {result.get('error_message', 'Unknown error')}\n")
        
        print(f"训练报告已保存到: {output_file}")


def main():
    """主函数"""
    trainer = ModelTrainer()
    
    # 训练所有模型，使用15个epoch和单个种子
    trainer.train_all_models(epochs=15, seeds=[2025])


if __name__ == "__main__":
    main()