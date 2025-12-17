import yaml
import sys

# 测试配置文件加载
try:
    with open('/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/train/test_sequential_temporal_stable.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("✅ 配置文件加载成功")
    print(f"模型名称: {config['model']['name']}")
    print(f"时序模型启用: {config['model']['sequential']['enabled']}")
    print(f"空间骨干: {config['model']['sequential']['spatial_backbone']}")
    print(f"学习率: {config['training']['learning_rate']}")
    
except Exception as e:
    print(f"❌ 配置文件加载失败: {e}")
    sys.exit(1)