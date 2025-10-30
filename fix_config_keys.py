#!/usr/bin/env python3
"""
修复配置文件中的键名格式问题
"""
import yaml
import os
from pathlib import Path

def fix_config_keys(config_path):
    """修复配置文件中的键名格式"""
    print(f"修复配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查并修复keys字段
    if 'data' in config and 'keys' in config['data']:
        original_keys = config['data']['keys']
        print(f"原始键名数量: {len(original_keys)}")
        print(f"原始键名示例: {original_keys[:5]}")
        
        # 将所有键名转换为字符串格式，并确保4位数字格式
        fixed_keys = []
        for key in original_keys:
            if isinstance(key, int):
                # 整数转换为4位字符串
                fixed_keys.append(f"{key:04d}")
            elif isinstance(key, str):
                # 字符串确保4位格式
                if key.isdigit():
                    fixed_keys.append(f"{int(key):04d}")
                else:
                    fixed_keys.append(key)
            else:
                fixed_keys.append(str(key))
        
        config['data']['keys'] = fixed_keys
        print(f"修复后键名数量: {len(fixed_keys)}")
        print(f"修复后键名示例: {fixed_keys[:5]}")
        
        # 保存修复后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 配置文件已修复: {config_path}")
        return True
    else:
        print(f"❌ 配置文件中未找到data.keys字段")
        return False

def main():
    """修复所有自动生成的配置文件"""
    config_dir = Path("configs/auto_generated")
    
    if not config_dir.exists():
        print(f"❌ 配置目录不存在: {config_dir}")
        return
    
    yaml_files = list(config_dir.glob("*.yaml"))
    print(f"找到 {len(yaml_files)} 个配置文件")
    
    success_count = 0
    for yaml_file in yaml_files:
        if yaml_file.name != "workflow_results.json":
            try:
                if fix_config_keys(yaml_file):
                    success_count += 1
            except Exception as e:
                print(f"❌ 修复失败 {yaml_file}: {e}")
    
    print(f"\n📊 修复完成: {success_count}/{len(yaml_files)} 个文件成功修复")

if __name__ == "__main__":
    main()