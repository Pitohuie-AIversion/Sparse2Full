#!/usr/bin/env python3
"""
数据集诊断脚本 - 检查时序NAR模型的数据集配置问题
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DatasetDiagnostic:
    """数据集诊断工具"""
    
    def __init__(self, config_path="configs/experiment/temporal_nar_300epochs.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # 从配置中提取数据路径，支持多种配置结构
        self.data_path = self._extract_data_path()
        
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'config_path': config_path,
            'data_path': self.data_path,
            'issues': [],
            'recommendations': []
        }
        
    def _extract_data_path(self):
        """从配置中提取数据路径"""
        # 尝试多种可能的配置结构
        possible_paths = [
            self.config.get('data', {}).get('config', {}).get('data_path', ''),  # 嵌套结构
            self.config.get('data', {}).get('data_path', ''),
            self.config.get('data_path', ''),
        ]
        
        # 查找第一个非空路径
        for path in possible_paths:
            if path:
                print(f"🔍 找到数据路径: {path}")
                return path
        
        # 如果都没找到，尝试从defaults中查找数据配置
        defaults = self.config.get('defaults', [])
        if defaults:
            for default in defaults:
                if isinstance(default, dict) and 'data' in default:
                    data_config_name = default['data']
                    print(f"🔍 尝试从数据配置文件加载: {data_config_name}")
                    # 尝试加载对应的数据配置文件
                    data_config_path = f"configs/datasets/{data_config_name}.yaml"
                    try:
                        with open(data_config_path, 'r', encoding='utf-8') as f:
                            data_config = yaml.safe_load(f)
                            data_path = data_config.get('data_path', '')
                            if data_path:
                                print(f"✅ 从数据配置文件找到路径: {data_path}")
                                return data_path
                    except Exception as e:
                        print(f"⚠️ 无法加载数据配置文件 {data_config_path}: {e}")
                elif isinstance(default, str) and default.startswith('data='):
                    # 处理 'data=temporal_2d_diff_react_na_na_crop_20' 格式
                    data_config_name = default.split('=')[1]
                    print(f"🔍 尝试从数据配置文件加载: {data_config_name}")
                    data_config_path = f"configs/datasets/{data_config_name}.yaml"
                    try:
                        with open(data_config_path, 'r', encoding='utf-8') as f:
                            data_config = yaml.safe_load(f)
                            data_path = data_config.get('data_path', '')
                            if data_path:
                                print(f"✅ 从数据配置文件找到路径: {data_path}")
                                return data_path
                    except Exception as e:
                        print(f"⚠️ 无法加载数据配置文件 {data_config_path}: {e}")
                elif isinstance(default, str) and '/' in default:
                    # 处理 '../datasets/temporal_2d_diff_react_na_na_crop_20' 格式
                    data_config_name = default.split('/')[-1]
                    print(f"🔍 尝试从数据配置文件加载: {data_config_name}")
                    data_config_path = f"configs/datasets/{data_config_name}.yaml"
                    try:
                        with open(data_config_path, 'r', encoding='utf-8') as f:
                            data_config = yaml.safe_load(f)
                            data_path = data_config.get('data_path', '')
                            if data_path:
                                print(f"✅ 从数据配置文件找到路径: {data_path}")
                                return data_path
                    except Exception as e:
                        print(f"⚠️ 无法加载数据配置文件 {data_config_path}: {e}")
        
        print("❌ 未找到数据路径")
        return ''
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            return {}
    
    def check_hdf5_structure(self):
        """检查HDF5文件结构"""
        print("🔍 检查HDF5文件结构...")
        
        if not os.path.exists(self.data_path):
            issue = f"数据文件不存在: {self.data_path}"
            self.report['issues'].append(issue)
            print(f"❌ {issue}")
            return False
            
        try:
            with h5py.File(self.data_path, 'r') as f:
                print(f"✅ 成功打开数据文件: {self.data_path}")
                
                # 检查文件结构
                print("\n📁 文件结构:")
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  📄 {name}: {obj.shape} {obj.dtype}")
                    else:
                        print(f"  📁 {name}/")
                
                f.visititems(print_structure)
                
                # 检查关键数据集
                required_keys = ['u', 't', 'x', 'y']  # 常见的PDE数据集键
                available_keys = list(f.keys())
                print(f"\n🔑 可用键: {available_keys}")
                
                # 分析数据维度
                for key in available_keys:
                    if isinstance(f[key], h5py.Dataset):
                        data = f[key]
                        print(f"\n📊 {key} 数据分析:")
                        print(f"  - 形状: {data.shape}")
                        print(f"  - 数据类型: {data.dtype}")
                        print(f"  - 数值范围: [{np.min(data[...]):.6f}, {np.max(data[...]):.6f}]")
                        
                        # 检查是否有时序维度
                        if len(data.shape) >= 3:
                            print(f"  - 可能的时序维度: {data.shape}")
                            if len(data.shape) == 4:  # [N, T, H, W] 或 [N, H, W, T]
                                print(f"  - 4D数据，可能格式: [样本, 时间, 高, 宽] 或 [样本, 高, 宽, 时间]")
                            elif len(data.shape) == 5:  # [N, T, C, H, W]
                                print(f"  - 5D数据，可能格式: [样本, 时间, 通道, 高, 宽]")
                
                return True
                
        except Exception as e:
            issue = f"HDF5文件读取错误: {e}"
            self.report['issues'].append(issue)
            print(f"❌ {issue}")
            return False
    
    def analyze_temporal_config(self):
        """分析时序配置"""
        print("\n⏰ 分析时序配置...")
        
        # 尝试从多个位置获取时序配置
        T_in = None
        T_out = None
        dt = None
        
        # 方法1: 从data配置中获取
        data_config = self.config.get('data', {})
        T_in = data_config.get('T_in', None)
        T_out = data_config.get('T_out', None)
        dt = data_config.get('dt', None)
        
        # 方法2: 从temporal配置中获取
        if T_in is None or T_out is None:
            temporal_config = self.config.get('temporal', {})
            T_in = T_in or temporal_config.get('T_in', None)
            T_out = T_out or temporal_config.get('T_out', None)
            dt = dt or temporal_config.get('dt', None)
        
        # 方法3: 从data.config.temporal中获取
        if T_in is None or T_out is None:
            temporal_config = self.config.get('data', {}).get('config', {}).get('temporal', {})
            T_in = T_in or temporal_config.get('T_in', None)
            T_out = T_out or temporal_config.get('T_out', None)
            dt = dt or temporal_config.get('dt', None)
            if T_in and T_out:
                print(f"✅ 从data.config.temporal获取时序参数")
        
        # 方法4: 从defaults中的数据配置文件获取
        if T_in is None or T_out is None:
            defaults = self.config.get('defaults', [])
            for default in defaults:
                if isinstance(default, dict) and 'data' in default:
                    data_config_name = default['data']
                    data_config_path = f"configs/datasets/{data_config_name}.yaml"
                    try:
                        with open(data_config_path, 'r', encoding='utf-8') as f:
                            data_config = yaml.safe_load(f)
                            T_in = T_in or data_config.get('T_in', None)
                            T_out = T_out or data_config.get('T_out', None)
                            dt = dt or data_config.get('dt', None)
                            if T_in and T_out:
                                print(f"✅ 从数据配置文件获取时序参数: {data_config_name}")
                                break
                    except Exception as e:
                        continue
                elif isinstance(default, str) and '/' in default:
                    # 处理 '../datasets/temporal_2d_diff_react_na_na_crop_20' 格式
                    data_config_name = default.split('/')[-1]
                    data_config_path = f"configs/datasets/{data_config_name}.yaml"
                    try:
                        with open(data_config_path, 'r', encoding='utf-8') as f:
                            data_config = yaml.safe_load(f)
                            T_in = T_in or data_config.get('T_in', None)
                            T_out = T_out or data_config.get('T_out', None)
                            dt = dt or data_config.get('dt', None)
                            if T_in and T_out:
                                print(f"✅ 从数据配置文件获取时序参数: {data_config_name}")
                                break
                    except Exception as e:
                        continue
        
        print(f"📋 配置参数:")
        print(f"  - T_in (输入时间步): {T_in}")
        print(f"  - T_out (输出时间步): {T_out}")
        print(f"  - dt (时间步长): {dt}")
        
        # 检查配置合理性
        if T_in is None or T_out is None:
            issue = "时序配置缺失 T_in 或 T_out"
            self.report['issues'].append(issue)
            print(f"❌ {issue}")
        
        if T_in and T_out and T_in + T_out > 50:  # 假设总时间步不应超过50
            issue = f"时序配置可能过长: T_in({T_in}) + T_out({T_out}) = {T_in + T_out}"
            self.report['issues'].append(issue)
            print(f"⚠️ {issue}")
        
        return T_in, T_out, dt
    
    def check_data_module_compatibility(self):
        """检查数据模块兼容性"""
        print("\n🔧 检查数据模块兼容性...")
        
        # 检查数据模块配置
        data_config = self.config.get('data', {})
        module_class = data_config.get('_target_', '')
        
        print(f"📦 数据模块: {module_class}")
        
        if 'TemporalPDEBenchDataModule' not in module_class:
            issue = f"数据模块类型可能不匹配: {module_class}"
            self.report['issues'].append(issue)
            print(f"⚠️ {issue}")
        
        # 检查关键参数
        batch_size = data_config.get('batch_size', None)
        image_size = data_config.get('image_size', None)
        crop_ratio = data_config.get('crop_ratio', None)
        
        print(f"⚙️ 数据模块参数:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - image_size: {image_size}")
        print(f"  - crop_ratio: {crop_ratio}")
        
        return data_config
    
    def sample_data_analysis(self):
        """采样数据分析"""
        print("\n🎯 采样数据分析...")
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                # 尝试找到主要数据键
                main_key = None
                for key in ['u', 'data', 'solution']:
                    if key in f:
                        main_key = key
                        break
                
                if main_key is None:
                    # 使用第一个数据集
                    datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
                    if datasets:
                        main_key = datasets[0]
                
                if main_key:
                    data = f[main_key]
                    print(f"📊 分析主数据集: {main_key}")
                    print(f"  - 完整形状: {data.shape}")
                    
                    # 采样少量数据进行分析
                    if len(data.shape) >= 3:
                        sample_size = min(5, data.shape[0])
                        sample_data = data[:sample_size]
                        
                        print(f"  - 采样数据形状: {sample_data.shape}")
                        print(f"  - 数值统计:")
                        print(f"    * 均值: {np.mean(sample_data):.6f}")
                        print(f"    * 标准差: {np.std(sample_data):.6f}")
                        print(f"    * 最小值: {np.min(sample_data):.6f}")
                        print(f"    * 最大值: {np.max(sample_data):.6f}")
                        
                        # 检查是否有异常值
                        if np.any(np.isnan(sample_data)):
                            issue = "数据中包含NaN值"
                            self.report['issues'].append(issue)
                            print(f"❌ {issue}")
                        
                        if np.any(np.isinf(sample_data)):
                            issue = "数据中包含无穷大值"
                            self.report['issues'].append(issue)
                            print(f"❌ {issue}")
                        
                        return sample_data
                
        except Exception as e:
            issue = f"数据采样分析失败: {e}"
            self.report['issues'].append(issue)
            print(f"❌ {issue}")
            
        return None
    
    def visualize_sample_data(self, sample_data=None):
        """可视化采样数据"""
        print("\n📈 生成数据可视化...")
        
        if sample_data is None:
            print("⚠️ 无采样数据，跳过可视化")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('数据集诊断可视化', fontsize=16)
            
            # 数据形状分析
            axes[0, 0].text(0.1, 0.5, f'数据形状: {sample_data.shape}\n'
                                     f'数据类型: {sample_data.dtype}\n'
                                     f'数值范围: [{np.min(sample_data):.3f}, {np.max(sample_data):.3f}]',
                           transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 0].set_title('数据基本信息')
            axes[0, 0].axis('off')
            
            # 数值分布直方图
            axes[0, 1].hist(sample_data.flatten(), bins=50, alpha=0.7)
            axes[0, 1].set_title('数值分布')
            axes[0, 1].set_xlabel('数值')
            axes[0, 1].set_ylabel('频次')
            
            # 如果是4D或5D数据，显示第一个样本的第一个时间步
            if len(sample_data.shape) >= 3:
                if len(sample_data.shape) == 4:  # [N, T, H, W] 或 [N, H, W, T]
                    # 尝试两种可能的格式
                    try:
                        img = sample_data[0, 0]  # 假设是 [N, T, H, W]
                        if img.shape[0] == img.shape[1]:  # 正方形，可能是空间维度
                            axes[0, 2].imshow(img, cmap='viridis')
                            axes[0, 2].set_title('样本数据 (t=0)')
                        else:
                            img = sample_data[0, :, :, 0]  # 尝试 [N, H, W, T]
                            axes[0, 2].imshow(img, cmap='viridis')
                            axes[0, 2].set_title('样本数据 (t=0)')
                    except:
                        axes[0, 2].text(0.5, 0.5, '无法显示图像', ha='center', va='center')
                        axes[0, 2].set_title('图像显示失败')
                
                elif len(sample_data.shape) == 5:  # [N, T, C, H, W]
                    try:
                        img = sample_data[0, 0, 0]  # 第一个样本，第一个时间步，第一个通道
                        axes[0, 2].imshow(img, cmap='viridis')
                        axes[0, 2].set_title('样本数据 (t=0, c=0)')
                    except:
                        axes[0, 2].text(0.5, 0.5, '无法显示图像', ha='center', va='center')
                        axes[0, 2].set_title('图像显示失败')
            
            # 时序分析（如果适用）
            if len(sample_data.shape) >= 3:
                try:
                    # 计算时间维度上的均值变化
                    if len(sample_data.shape) == 4:
                        time_means = np.mean(sample_data[0], axis=(1, 2))  # [T]
                    elif len(sample_data.shape) == 5:
                        time_means = np.mean(sample_data[0], axis=(1, 2, 3))  # [T]
                    
                    axes[1, 0].plot(time_means)
                    axes[1, 0].set_title('时序均值变化')
                    axes[1, 0].set_xlabel('时间步')
                    axes[1, 0].set_ylabel('均值')
                except:
                    axes[1, 0].text(0.5, 0.5, '时序分析失败', ha='center', va='center')
                    axes[1, 0].set_title('时序分析')
            
            # 配置信息
            config_text = f"配置文件: {self.config_path}\n"
            config_text += f"数据路径: {self.data_path}\n"
            data_config = self.config.get('data', {})
            config_text += f"T_in: {data_config.get('T_in', 'N/A')}\n"
            config_text += f"T_out: {data_config.get('T_out', 'N/A')}\n"
            config_text += f"batch_size: {data_config.get('batch_size', 'N/A')}\n"
            config_text += f"image_size: {data_config.get('image_size', 'N/A')}"
            
            axes[1, 1].text(0.1, 0.5, config_text, transform=axes[1, 1].transAxes, fontsize=10)
            axes[1, 1].set_title('配置信息')
            axes[1, 1].axis('off')
            
            # 问题总结
            issues_text = "发现的问题:\n"
            if self.report['issues']:
                for i, issue in enumerate(self.report['issues'][:5], 1):
                    issues_text += f"{i}. {issue}\n"
            else:
                issues_text += "暂未发现明显问题"
            
            axes[1, 2].text(0.1, 0.5, issues_text, transform=axes[1, 2].transAxes, fontsize=10)
            axes[1, 2].set_title('问题总结')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            output_dir = Path("dataset_diagnosis_output")
            output_dir.mkdir(exist_ok=True)
            
            plt.savefig(output_dir / "dataset_diagnosis.png", dpi=300, bbox_inches='tight')
            print(f"✅ 可视化图像已保存: {output_dir / 'dataset_diagnosis.png'}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
    
    def generate_recommendations(self):
        """生成修正建议"""
        print("\n💡 生成修正建议...")
        
        recommendations = []
        
        # 基于发现的问题生成建议
        for issue in self.report['issues']:
            if "数据文件不存在" in issue:
                recommendations.append("检查数据文件路径是否正确，确保数据文件已下载到指定位置")
            elif "时序配置缺失" in issue:
                recommendations.append("在配置文件中添加完整的时序参数 T_in 和 T_out")
            elif "时序配置可能过长" in issue:
                recommendations.append("考虑减少 T_in 或 T_out 的值，避免内存溢出和训练不稳定")
            elif "数据模块类型可能不匹配" in issue:
                recommendations.append("确认使用正确的数据模块类型，如 TemporalPDEBenchDataModule")
            elif "NaN值" in issue:
                recommendations.append("数据预处理时需要处理NaN值，可以使用插值或删除含NaN的样本")
            elif "无穷大值" in issue:
                recommendations.append("数据预处理时需要处理无穷大值，检查数据生成过程")
        
        # 通用建议
        if not recommendations:
            recommendations.extend([
                "数据集基本结构看起来正常，建议检查数据预处理流程",
                "验证时序数据的采样逻辑是否与模型期望一致",
                "检查可视化代码中的数据维度处理是否正确"
            ])
        
        # 添加具体的技术建议
        recommendations.extend([
            "建议在训练前添加数据验证步骤，确保数据格式正确",
            "可以添加数据统计信息的日志记录，便于调试",
            "考虑使用更小的数据子集进行快速验证"
        ])
        
        self.report['recommendations'] = recommendations
        
        print("📋 修正建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    def save_report(self):
        """保存诊断报告"""
        output_dir = Path("dataset_diagnosis_output")
        output_dir.mkdir(exist_ok=True)
        
        # 保存JSON报告
        report_path = output_dir / "diagnosis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown报告
        md_path = output_dir / "diagnosis_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 数据集诊断报告\n\n")
            f.write(f"**生成时间**: {self.report['timestamp']}\n\n")
            f.write(f"**配置文件**: {self.report['config_path']}\n\n")
            f.write(f"**数据路径**: {self.report['data_path']}\n\n")
            
            f.write("## 发现的问题\n\n")
            if self.report['issues']:
                for i, issue in enumerate(self.report['issues'], 1):
                    f.write(f"{i}. {issue}\n")
            else:
                f.write("暂未发现明显问题\n")
            
            f.write("\n## 修正建议\n\n")
            for i, rec in enumerate(self.report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"\n📄 诊断报告已保存:")
        print(f"  - JSON: {report_path}")
        print(f"  - Markdown: {md_path}")
        
        return report_path, md_path
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("🚀 开始数据集诊断...")
        print("=" * 60)
        
        # 1. 检查HDF5结构
        hdf5_ok = self.check_hdf5_structure()
        
        # 2. 分析时序配置
        T_in, T_out, dt = self.analyze_temporal_config()
        
        # 3. 检查数据模块兼容性
        data_config = self.check_data_module_compatibility()
        
        # 4. 采样数据分析
        sample_data = self.sample_data_analysis()
        
        # 5. 可视化
        self.visualize_sample_data(sample_data)
        
        # 6. 生成建议
        self.generate_recommendations()
        
        # 7. 保存报告
        json_path, md_path = self.save_report()
        
        print("\n" + "=" * 60)
        print("✅ 数据集诊断完成!")
        
        return self.report

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集诊断工具")
    parser.add_argument("--config", default="configs/experiment/temporal_nar_300epochs.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 运行诊断
    diagnostic = DatasetDiagnostic(args.config)
    report = diagnostic.run_full_diagnosis()
    
    # 输出总结
    print(f"\n📊 诊断总结:")
    print(f"  - 发现问题: {len(report['issues'])} 个")
    print(f"  - 修正建议: {len(report['recommendations'])} 条")
    
    if report['issues']:
        print(f"\n⚠️ 主要问题:")
        for issue in report['issues'][:3]:
            print(f"  • {issue}")
    
    print(f"\n💡 关键建议:")
    for rec in report['recommendations'][:3]:
        print(f"  • {rec}")

if __name__ == "__main__":
    main()