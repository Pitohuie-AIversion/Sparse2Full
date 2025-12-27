#!/usr/bin/env python3
"""
整合所有可视化文件脚本
将分散在各个文件夹中的可视化文件整合到paper_package/figs/目录
"""

import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict

class VisualizationIntegrator:
    """可视化文件整合器"""
    
    def __init__(self):
        self.root_dir = Path(".")
        self.target_dir = Path("paper_package/figs")
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建分类子目录
        self.subdirs = {
            'training_results': self.target_dir / 'training_results',
            'model_predictions': self.target_dir / 'model_predictions', 
            'flow_fields': self.target_dir / 'flow_fields',
            'temporal_analysis': self.target_dir / 'temporal_analysis',
            'comprehensive_reports': self.target_dir / 'comprehensive_reports',
            'input_output_analysis': self.target_dir / 'input_output_analysis'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def find_visualization_files(self) -> Dict[str, List[Path]]:
        """查找所有可视化文件"""
        self.logger.info("Scanning for visualization files...")
        
        files_by_category = {
            'html_reports': [],
            'png_images': [],
            'json_data': [],
            'other_files': []
        }
        
        # 定义要扫描的目录
        scan_dirs = [
            'comprehensive_visualizations',
            'flow_field_visualizations', 
            'input_output_viz',
            'model_predictions_visualization',
            'temporal_predictions_viz',
            'temporal_nar_analysis'
        ]
        
        for dir_name in scan_dirs:
            dir_path = self.root_dir / dir_name
            if dir_path.exists():
                self.logger.info(f"Scanning directory: {dir_path}")
                
                # 查找HTML文件
                html_files = list(dir_path.rglob("*.html"))
                files_by_category['html_reports'].extend(html_files)
                
                # 查找PNG图片
                png_files = list(dir_path.rglob("*.png"))
                files_by_category['png_images'].extend(png_files)
                
                # 查找JSON数据文件
                json_files = list(dir_path.rglob("*.json"))
                files_by_category['json_data'].extend(json_files)
                
                # 查找其他文件
                other_files = []
                for ext in ['*.md', '*.txt', '*.log']:
                    other_files.extend(dir_path.rglob(ext))
                files_by_category['other_files'].extend(other_files)
        
        # 统计文件数量
        total_files = sum(len(files) for files in files_by_category.values())
        self.logger.info(f"Found {total_files} visualization files:")
        for category, files in files_by_category.items():
            self.logger.info(f"  {category}: {len(files)} files")
        
        return files_by_category
    
    def copy_files_with_organization(self, files_by_category: Dict[str, List[Path]]):
        """按类别复制文件"""
        self.logger.info("Copying and organizing files...")
        
        copied_files = []
        
        # 复制HTML报告
        for html_file in files_by_category['html_reports']:
            # 根据文件名确定目标目录
            if 'comprehensive' in html_file.name.lower():
                target_subdir = self.subdirs['comprehensive_reports']
            elif 'flow' in html_file.name.lower():
                target_subdir = self.subdirs['flow_fields']
            elif 'temporal' in html_file.name.lower():
                target_subdir = self.subdirs['temporal_analysis']
            elif 'prediction' in html_file.name.lower():
                target_subdir = self.subdirs['model_predictions']
            elif 'input_output' in html_file.name.lower():
                target_subdir = self.subdirs['input_output_analysis']
            else:
                target_subdir = self.subdirs['comprehensive_reports']
            
            target_path = target_subdir / html_file.name
            try:
                shutil.copy2(html_file, target_path)
                copied_files.append(target_path)
                self.logger.info(f"Copied: {html_file} -> {target_path}")
            except Exception as e:
                self.logger.error(f"Failed to copy {html_file}: {e}")
        
        # 复制PNG图片
        for png_file in files_by_category['png_images']:
            # 根据路径和文件名确定目标目录
            if 'flow' in str(png_file).lower():
                target_subdir = self.subdirs['flow_fields']
            elif 'temporal' in str(png_file).lower():
                target_subdir = self.subdirs['temporal_analysis']
            elif 'prediction' in str(png_file).lower():
                target_subdir = self.subdirs['model_predictions']
            elif any(keyword in str(png_file).lower() for keyword in ['input', 'output', 'error', 'heatmap']):
                target_subdir = self.subdirs['input_output_analysis']
            else:
                target_subdir = self.subdirs['training_results']
            
            target_path = target_subdir / png_file.name
            try:
                shutil.copy2(png_file, target_path)
                copied_files.append(target_path)
                self.logger.info(f"Copied: {png_file} -> {target_path}")
            except Exception as e:
                self.logger.error(f"Failed to copy {png_file}: {e}")
        
        # 复制JSON数据文件
        json_target_dir = self.target_dir / 'data'
        json_target_dir.mkdir(exist_ok=True)
        
        for json_file in files_by_category['json_data']:
            target_path = json_target_dir / json_file.name
            try:
                shutil.copy2(json_file, target_path)
                copied_files.append(target_path)
                self.logger.info(f"Copied: {json_file} -> {target_path}")
            except Exception as e:
                self.logger.error(f"Failed to copy {json_file}: {e}")
        
        self.logger.info(f"Successfully copied {len(copied_files)} files")
        return copied_files
    
    def create_file_index(self, copied_files: List[Path]):
        """创建文件索引"""
        self.logger.info("Creating file index...")
        
        index_content = f"""# 可视化文件索引
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 文件统计
总文件数: {len(copied_files)}

## 按类别分组

"""
        
        # 按目录分组文件
        files_by_dir = {}
        for file_path in copied_files:
            relative_path = file_path.relative_to(self.target_dir)
            dir_name = str(relative_path.parent)
            
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(relative_path.name)
        
        # 生成索引内容
        for dir_name, files in sorted(files_by_dir.items()):
            index_content += f"### {dir_name}\n"
            for file_name in sorted(files):
                index_content += f"- {file_name}\n"
            index_content += "\n"
        
        # 保存索引文件
        index_path = self.target_dir / "file_index.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        self.logger.info(f"File index saved to: {index_path}")
        return index_path
    
    def integrate_all(self):
        """执行完整的整合流程"""
        self.logger.info("Starting visualization integration process...")
        
        # 查找所有可视化文件
        files_by_category = self.find_visualization_files()
        
        # 复制和组织文件
        copied_files = self.copy_files_with_organization(files_by_category)
        
        # 创建文件索引
        index_path = self.create_file_index(copied_files)
        
        self.logger.info("Integration process completed successfully!")
        
        return {
            'copied_files': copied_files,
            'index_path': index_path,
            'target_directory': self.target_dir
        }

def main():
    """主函数"""
    print("Starting Visualization Integration...")
    print("="*60)
    
    # 创建整合器
    integrator = VisualizationIntegrator()
    
    # 执行整合
    result = integrator.integrate_all()
    
    print("\n" + "="*60)
    print("Visualization Integration Complete!")
    print("="*60)
    print(f"Target directory: {result['target_directory']}")
    print(f"Copied files: {len(result['copied_files'])}")
    print(f"File index: {result['index_path']}")
    print("="*60)

if __name__ == "__main__":
    main()