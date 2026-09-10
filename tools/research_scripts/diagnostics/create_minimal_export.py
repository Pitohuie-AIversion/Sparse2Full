import json
import os
import shutil
from pathlib import Path

def parse_pydeps_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    files_to_keep = set()
    
    # 遍历所有节点
    for module_name, info in data.items():
        # 我们只关心项目内的文件，忽略 site-packages
        path = info.get('path')
        if path and 'site-packages' not in path and 'anaconda3' not in path and 'lib/python' not in path:
            # 获取相对路径
            try:
                rel_path = os.path.relpath(path, os.getcwd())
                if not rel_path.startswith('..'):
                    files_to_keep.add(rel_path)
            except ValueError:
                pass
                
    return files_to_keep

# 1. 解析依赖
files_original = parse_pydeps_json('deps_original.json')
files_refactored = parse_pydeps_json('deps_refactored.json')

# 2. 合并依赖
all_files = files_original.union(files_refactored)

# 3. 添加必须的入口脚本和配置文件
all_files.add('tools/training/train_real_data_ar.py')
all_files.add('tools/training/train_real_data_ar_refactored.py')
all_files.add('requirements.txt')
all_files.add('setup.py')

# 4. 创建极简导出目录
export_dir = Path('minimal_export')
if export_dir.exists():
    shutil.rmtree(export_dir)
export_dir.mkdir()

print(f"Found {len(all_files)} essential files.")

# 5. 复制文件
for file_path in all_files:
    src = Path(file_path)
    if src.exists() and src.is_file():
        dst = export_dir / file_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {file_path}")
    else:
        print(f"Warning: File not found or not a file: {file_path}")

# 6. 特殊处理 configs 目录 (Hydra 需要完整的 config 目录结构)
# 虽然 pydeps 无法检测 yaml 依赖，但为了运行，我们需要复制用到的 config
# 既然要求“只要能运行”，我们只复制 config.yaml 和引用的子配置
# 为安全起见，复制整个 configs 目录是最小代价的“保证能运行”方案
# 因为 Hydra 的组合非常灵活，静态分析很难确定到底用了哪些 yaml
if Path('configs').exists():
    shutil.copytree('configs', export_dir / 'configs')
    print("Copied: configs directory")

print("Minimal export completed.")
