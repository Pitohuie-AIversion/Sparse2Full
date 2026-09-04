#!/usr/bin/env python3
"""
批量重新训练所有模型脚本
使用稳定的训练配置参数重新训练所有模型
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import shutil

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 稳定的训练配置参数
STABLE_CONFIG = {
    "loss": {
        "rec_weight": 1.0,
        "spec_weight": 0.0,
        "dc_weight": 0.0,
        "low_freq_modes": 4,
        "mirror_padding": False
    },
    "training": {
        "batch_size": 4,
        "lr": 1e-4,
        "use_amp": True,
        "epochs": 200,
        "grad_clip_norm": 1.0
    },
    "optimizer": {
        "name": "AdamW",
        "weight_decay": 1e-4
    },
    "scheduler": {
        "name": "cosine_warmup",
        "warmup_steps": 1000
    }
}

# 所有需要训练的模型列表
MODELS_TO_TRAIN = [
    "SwinUNet",
    "Hybrid", 
    "MLP",
    "UNet",
    "FNO2D",
    "UNetPlusPlus",
    "UFNO_UNet",
    "MLP_Mixer",
    "SegFormer",
    "SegFormer_UNetFormer"
]

# 模型配置映射
MODEL_CONFIGS = {
    "SwinUNet": {
        "patch_size": 4,
        "window_size": 8,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "embed_dim": 96,
        "mlp_ratio": 4.0,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1
    },
    "Hybrid": {
        "patch_size": 4,
        "window_size": 8,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "embed_dim": 96,
        "mlp_ratio": 4.0,
        "fno_modes": 16,
        "fno_layers": 4
    },
    "MLP": {
        "patch_size": 8,
        "hidden_dim": 512,
        "num_layers": 6,
        "dropout": 0.1
    },
    "UNet": {
        "base_channels": 64,
        "num_levels": 4,
        "dropout": 0.1
    },
    "FNO2D": {
        "modes": 16,
        "width": 64,
        "num_layers": 4
    },
    "UNetPlusPlus": {
        "base_channels": 64,
        "num_levels": 4,
        "deep_supervision": True
    },
    "UFNO_UNet": {
        "base_channels": 64,
        "fno_modes": 16,
        "fno_layers": 2
    },
    "MLP_Mixer": {
        "patch_size": 8,
        "hidden_dim": 512,
        "num_layers": 8,
        "tokens_mlp_dim": 256,
        "channels_mlp_dim": 2048
    },
    "SegFormer": {
        "embed_dims": [64, 128, 256, 512],
        "num_heads": [1, 2, 4, 8],
        "mlp_ratios": [4, 4, 4, 4],
        "depths": [3, 4, 6, 3]
    },
    "SegFormer_UNetFormer": {
        "embed_dims": [64, 128, 256, 512],
        "num_heads": [1, 2, 4, 8],
        "mlp_ratios": [4, 4, 4, 4],
        "depths": [3, 4, 6, 3],
        "use_unet_decoder": True
    }
}

class BatchTrainer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "configs"
        self.runs_dir = self.project_root / "runs"
        self.batch_runs_dir = self.runs_dir / f"batch_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建批量训练目录
        self.batch_runs_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练日志
        self.training_log = []
        self.failed_models = []
        self.successful_models = []
        
    def create_model_config(self, model_name: str) -> Path:
        """为指定模型创建配置文件"""
        config_path = self.batch_runs_dir / f"train_{model_name.lower()}.yaml"
        
        # 读取基础配置
        base_config_path = self.config_dir / "train.yaml"
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 修改模型名称和参数
        config_lines = config_content.split('\n')
        new_config_lines = []
        
        in_model_section = False
        in_kwargs_section = False
        indent_level = 0
        
        for line in config_lines:
            if line.strip().startswith('model:'):
                in_model_section = True
                new_config_lines.append(line)
            elif in_model_section and line.strip().startswith('name:'):
                new_config_lines.append(f'  name: "{model_name}"')
            elif in_model_section and line.strip().startswith('kwargs:'):
                in_kwargs_section = True
                new_config_lines.append(line)
                # 添加模型特定配置
                if model_name in MODEL_CONFIGS:
                    for key, value in MODEL_CONFIGS[model_name].items():
                        if isinstance(value, list):
                            new_config_lines.append(f'      {key}: {value}')
                        elif isinstance(value, str):
                            new_config_lines.append(f'      {key}: "{value}"')
                        else:
                            new_config_lines.append(f'      {key}: {value}')
            elif in_kwargs_section and line.strip() and not line.startswith('      '):
                # 退出kwargs部分
                in_kwargs_section = False
                new_config_lines.append(line)
            elif in_model_section and line.strip() and not line.startswith('  ') and not line.startswith('model:'):
                # 退出model部分
                in_model_section = False
                new_config_lines.append(line)
            elif not in_kwargs_section:
                new_config_lines.append(line)
        
        # 写入新配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_config_lines))
        
        return config_path
    
    def train_model(self, model_name: str) -> bool:
        """训练单个模型"""
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_name}")
        print(f"{'='*60}")
        
        try:
            # 创建模型配置文件
            config_path = self.create_model_config(model_name)
            print(f"配置文件已创建: {config_path}")
            
            # 创建模型专用输出目录
            model_output_dir = self.batch_runs_dir / model_name.lower()
            model_output_dir.mkdir(exist_ok=True)
            
            # 构建训练命令 - 使用Hydra配置覆盖
            python_exe = r"python"
            train_cmd = [
                python_exe, "train.py",
                f"model.name={model_name}",
                f"experiment.output_dir={model_output_dir}",
                f"experiment.name={model_name}_stable_retrain"
            ]
            
            print(f"执行训练命令: {' '.join(train_cmd)}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行训练
            with open(model_output_dir / "training.log", "w") as log_file:
                process = subprocess.Popen(
                    train_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=self.project_root
                )
                
                # 实时输出训练日志
                for line in process.stdout:
                    print(line.rstrip())
                    log_file.write(line)
                    log_file.flush()
                
                process.wait()
            
            # 记录结束时间
            end_time = time.time()
            training_time = end_time - start_time
            
            if process.returncode == 0:
                print(f"✅ 模型 {model_name} 训练成功! 耗时: {training_time:.2f}秒")
                self.successful_models.append(model_name)
                
                # 记录训练信息
                self.training_log.append({
                    "model": model_name,
                    "status": "success",
                    "training_time": training_time,
                    "config_path": str(config_path),
                    "output_dir": str(model_output_dir),
                    "timestamp": datetime.now().isoformat()
                })
                return True
            else:
                print(f"❌ 模型 {model_name} 训练失败! 返回码: {process.returncode}")
                self.failed_models.append(model_name)
                
                # 记录失败信息
                self.training_log.append({
                    "model": model_name,
                    "status": "failed",
                    "training_time": training_time,
                    "return_code": process.returncode,
                    "config_path": str(config_path),
                    "output_dir": str(model_output_dir),
                    "timestamp": datetime.now().isoformat()
                })
                return False
                
        except Exception as e:
            print(f"❌ 模型 {model_name} 训练过程中发生异常: {str(e)}")
            self.failed_models.append(model_name)
            
            # 记录异常信息
            self.training_log.append({
                "model": model_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return False
    
    def save_training_summary(self):
        """保存训练总结"""
        summary = {
            "batch_training_info": {
                "start_time": datetime.now().isoformat(),
                "stable_config": STABLE_CONFIG,
                "total_models": len(MODELS_TO_TRAIN),
                "successful_models": len(self.successful_models),
                "failed_models": len(self.failed_models)
            },
            "successful_models": self.successful_models,
            "failed_models": self.failed_models,
            "detailed_log": self.training_log
        }
        
        summary_path = self.batch_runs_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n训练总结已保存到: {summary_path}")
        return summary_path
    
    def run_batch_training(self):
        """执行批量训练"""
        print("🚀 开始批量重新训练所有模型")
        print(f"使用稳定配置: {STABLE_CONFIG}")
        print(f"训练模型列表: {MODELS_TO_TRAIN}")
        print(f"输出目录: {self.batch_runs_dir}")
        
        total_start_time = time.time()
        
        # 逐个训练模型
        for i, model_name in enumerate(MODELS_TO_TRAIN, 1):
            print(f"\n进度: {i}/{len(MODELS_TO_TRAIN)}")
            self.train_model(model_name)
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # 打印最终结果
        print(f"\n{'='*80}")
        print("🎉 批量训练完成!")
        print(f"{'='*80}")
        print(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
        print(f"成功训练: {len(self.successful_models)}/{len(MODELS_TO_TRAIN)} 个模型")
        print(f"成功模型: {', '.join(self.successful_models)}")
        if self.failed_models:
            print(f"失败模型: {', '.join(self.failed_models)}")
        
        # 保存训练总结
        summary_path = self.save_training_summary()
        
        return {
            "success": len(self.failed_models) == 0,
            "successful_models": self.successful_models,
            "failed_models": self.failed_models,
            "summary_path": summary_path,
            "total_time": total_time
        }

def main():
    """主函数"""
    trainer = BatchTrainer()
    result = trainer.run_batch_training()
    
    # 返回适当的退出码
    if result["success"]:
        print("\n✅ 所有模型训练成功!")
        sys.exit(0)
    else:
        print(f"\n❌ 有 {len(result['failed_models'])} 个模型训练失败")
        sys.exit(1)

if __name__ == "__main__":
    main()