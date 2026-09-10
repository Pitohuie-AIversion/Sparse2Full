#!/usr/bin/env python3
"""
模型切换训练演示 - 针对train_real_data_ar.py
展示如何使用不同的空间模型进行训练
"""

import subprocess
import sys
from pathlib import Path

def run_model_training(model_name: str, epochs: int = 2, config: str = None):
    """运行指定模型的训练"""
    print(f"\n{'='*60}")
    print(f"🚀 训练模型: {model_name}")
    print(f"{'='*60}")
    
    # 基础命令
    cmd = [
        sys.executable, 
        "tools/training/train_real_data_ar.py",
        "--model", model_name,
        "--config", config or "configs/train_real_ar.yaml"
    ]
    
    # 如果配置文件不存在，使用默认配置
    config_path = Path(config or "configs/train_real_ar.yaml")
    if not config_path.exists():
        print(f"⚠️  配置文件不存在，使用默认训练配置")
        # 创建一个简单的配置文件
        default_config = """
experiment:
  name: "model_switching_demo"
  device: "cuda:0"
  seed: 2025
  output_dir: "runs"

data:
  data_path: "test_data.h5"
  dataset_name: "PDEBench"
  keys: ["tensor"]
  splits_dir: "splits"
  image_size: 128
  observation:
    mode: "SR"
    sr:
      scale_factor: 4
      blur_sigma: 1.0
      blur_kernel_size: 5
      boundary_mode: "mirror"
  preprocessing:
    normalize: true
  dataloader:
    batch_size: 2
    num_workers: 2

model:
  name: "{model_name}"
  in_channels: 1
  out_channels: 1
  img_size: 128

training:
  epochs: {epochs}
  batch_size: 4
  optimizer:
    name: "AdamW"
    lr: 1e-4
  use_amp: true
  log_interval: 10

loss:
  rec_weight: 1.0
  spec_weight: 0.0
  dc_weight: 0.0
""".format(model_name=model_name, epochs=epochs)
        
        config_path = Path("temp_model_switching_config.yaml")
        with open(config_path, 'w') as f:
            f.write(default_config)
        cmd = [sys.executable, "tools/training/train_real_data_ar.py", "--model", model_name, "--config", str(config_path)]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        if result.returncode == 0:
            print(f"✅ {model_name} 训练成功！")
            # 显示关键输出
            lines = result.stdout.split('\n')
            for line in lines[-20:]:  # 显示最后20行
                if any(keyword in line.lower() for keyword in ['loss', 'accuracy', 'metric', 'epoch']):
                    print(f"  {line}")
        else:
            print(f"❌ {model_name} 训练失败")
            print(f"错误信息: {result.stderr[:500]}...")  # 显示前500字符
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} 训练超时 (5分钟)")
    except Exception as e:
        print(f"❌ {model_name} 训练异常: {e}")

def main():
    """主函数 - 演示模型切换"""
    print("🎯 模型切换训练演示")
    print("使用 train_real_data_ar.py 脚本测试不同的空间模型")
    print("\n基于我们之前的测试，以下模型可以正常工作:")
    
    # 基于我们测试框架验证过的模型
    tested_models = [
        "unet",           # ✅ 已验证 - 轻量级，快速
        "swinunet",       # ✅ 已验证 - 高精度，稍慢
    ]
    
    # 其他值得测试的空间模型
    additional_models = [
        "fno2d",          # 频域神经网络
        "segformer",      # 分割变换器
        "ufnounet",       # U-FNO混合模型
        "unetplusplus",   # UNet++
        "hybrid",         # 混合模型
    ]
    
    print("\n✅ 已验证模型 (推荐优先使用):")
    for i, model in enumerate(tested_models, 1):
        print(f"  {i}. {model}")
    
    print("\n🔍 其他可选模型:")
    for i, model in enumerate(additional_models, 1):
        print(f"  {i}. {model}")
    
    print("\n" + "="*60)
    print("开始演示已验证的模型训练...")
    
    # 演示已验证的模型
    for model_name in tested_models:
        run_model_training(model_name, epochs=2)
    
    print(f"\n{'='*60}")
    print("✅ 模型切换演示完成！")
    print("\n您现在可以使用以下命令切换任何支持的模型:")
    print("  python tools/training/train_real_data_ar.py --model <模型名称>")
    print("\n例如:")
    print("  python tools/training/train_real_data_ar.py --model unet")
    print("  python tools/training/train_real_data_ar.py --model swinunet")
    print("  python tools/training/train_real_data_ar.py --model fno2d")
    
    # 清理临时配置文件
    temp_config = Path("temp_model_switching_config.yaml")
    if temp_config.exists():
        temp_config.unlink()

if __name__ == "__main__":
    main()