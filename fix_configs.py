#!/usr/bin/env python3
"""修复批量训练配置文件中的问题"""

import yaml
from pathlib import Path

def fix_configs():
    """修复配置文件中的问题"""
    results_dir = Path('runs/batch_training_results')
    
    # 修复FNO相关模型的AMP配置
    fno_models = ['fno2d', 'hybrid', 'ufno_unet']
    for model in fno_models:
        config_file = results_dir / f'config_{model}.yaml'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 禁用AMP
            config['training']['use_amp'] = False
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            print(f'Fixed AMP config for {model}')
    
    # 修复UNetFormer的num_heads配置
    unetformer_config = results_dir / 'config_unetformer.yaml'
    if unetformer_config.exists():
        with open(unetformer_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'model' in config and 'params' in config['model']:
            if 'num_heads' in config['model']['params']:
                num_heads = config['model']['params']['num_heads']
                if isinstance(num_heads, (list, tuple)):
                    config['model']['params']['num_heads'] = num_heads[0] if num_heads else 8
                elif not isinstance(num_heads, int):
                    config['model']['params']['num_heads'] = 8
                
                with open(unetformer_config, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                print(f'Fixed num_heads config for unetformer: {config["model"]["params"]["num_heads"]}')

if __name__ == "__main__":
    fix_configs()