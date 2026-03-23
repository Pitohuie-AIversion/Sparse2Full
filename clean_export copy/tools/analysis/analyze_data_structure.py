#!/usr/bin/env python3
"""
PDEBench时序数据集结构分析工具

详细分析PDEBench数据集的结构，包括：
1. HDF5文件结构分析
2. 数据维度和形状信息
3. 时序配置参数
4. 数据加载流程示例
5. 保存结果到JSON文件
"""

import os
import sys
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append('.')

class DataStructureAnalyzer:
    """数据结构分析器"""
    
    def __init__(self, data_path: str = "e:/2d/diffusion-reaction/2D_diff-react_NA_NA.h5"):
        """初始化分析器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.analysis_results = {}
        
        print(f"数据结构分析器初始化完成")
        print(f"数据路径: {self.data_path}")
    
    def analyze_hdf5_structure(self) -> Dict[str, Any]:
        """分析HDF5文件结构"""
        print("\n" + "="*60)
        print("📁 HDF5文件结构分析")
        print("="*60)
        
        structure_info = {
            "file_path": str(self.data_path),
            "file_size_mb": 0,
            "root_keys": [],
            "detailed_structure": {},
            "data_types": {},
            "shapes": {},
            "attributes": {},
            "sample_case_analysis": {}
        }
        
        try:
            # 检查文件是否存在
            if not os.path.exists(self.data_path):
                print(f"❌ 数据文件不存在: {self.data_path}")
                structure_info["error"] = f"文件不存在: {self.data_path}"
                return structure_info
            
            # 获取文件大小
            file_size = os.path.getsize(self.data_path) / (1024 * 1024)  # MB
            structure_info["file_size_mb"] = round(file_size, 2)
            print(f"文件大小: {file_size:.2f} MB")
            
            with h5py.File(self.data_path, 'r') as f:
                # 获取根级键名
                root_keys = list(f.keys())
                structure_info["root_keys"] = root_keys
                print(f"根级键名数量: {len(root_keys)}")
                print(f"前10个键名: {root_keys[:10]}")
                
                # 分析第一个案例的详细结构
                if root_keys:
                    first_case = root_keys[0]
                    print(f"\n分析第一个案例: {first_case}")
                    
                    case_group = f[first_case]
                    case_info = {
                        "case_id": first_case,
                        "type": "Group",
                        "keys": list(case_group.keys()),
                        "attributes": dict(case_group.attrs)
                    }
                    
                    # 分析案例内的数据集
                    for key in case_group.keys():
                        item = case_group[key]
                        if isinstance(item, h5py.Dataset):
                            shape = item.shape
                            dtype = str(item.dtype)
                            print(f"  📄 {key}: {shape} {dtype}")
                            
                            case_info[key] = {
                                "type": "Dataset",
                                "shape": shape,
                                "dtype": dtype,
                                "size": item.size,
                                "attributes": dict(item.attrs)
                            }
                            
                            # 获取数据统计信息（仅对较小的数据集）
                            if item.size < 10000000:  # 小于10M个元素
                                try:
                                    # 只读取一小部分数据进行统计
                                    if len(shape) >= 3:  # 时序数据
                                        sample_data = item[0:min(5, shape[0])]  # 只读前5个时间步
                                    else:
                                        sample_data = item[...]
                                    
                                    if np.issubdtype(item.dtype, np.number):
                                        case_info[key]["statistics"] = {
                                            "min": float(np.min(sample_data)),
                                            "max": float(np.max(sample_data)),
                                            "mean": float(np.mean(sample_data)),
                                            "std": float(np.std(sample_data))
                                        }
                                        print(f"    统计: min={np.min(sample_data):.4f}, max={np.max(sample_data):.4f}")
                                except Exception as e:
                                    print(f"    统计信息获取失败: {e}")
                    
                    structure_info["sample_case_analysis"] = case_info
                
                # 获取文件级属性
                structure_info["attributes"] = dict(f.attrs)
                
                # 统计所有案例的基本信息
                structure_info["dataset_summary"] = {
                    "total_cases": len(root_keys),
                    "case_id_range": f"{root_keys[0]} - {root_keys[-1]}" if root_keys else "无数据"
                }
                
        except Exception as e:
            print(f"❌ HDF5文件分析失败: {e}")
            structure_info["error"] = str(e)
        
        return structure_info
    
    def analyze_temporal_config(self) -> Dict[str, Any]:
        """分析时序配置参数"""
        print("\n" + "="*60)
        print("⏰ 时序配置参数分析")
        print("="*60)
        
        config_info = {
            "default_config": {
                "data_path": self.data_path,
                "keys": ["u"],
                "batch_size": 16,
                "num_workers": 12,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 6,
                "image_size": 128,
                "normalize": True,
                "use_official_format": True
            },
            "temporal_config": {
                "T_in": 4,
                "T_out": 3,
                "dt": 0.1,
                "temporal_mode": "sequential",
                "sequence_length": None,
                "overlap_ratio": 0.0
            },
            "model_config": {
                "model_type": "SwinUNet with Temporal NAR",
                "in_channels": 1,
                "out_channels": 1,
                "img_size": 128
            },
            "training_config": {
                "max_epochs": 300,
                "optimizer": "AdamW",
                "lr": 0.002
            }
        }
        
        # 打印配置信息
        print(f"数据路径: {config_info['default_config']['data_path']}")
        print(f"数据键名: {config_info['default_config']['keys']}")
        print(f"批处理大小: {config_info['default_config']['batch_size']}")
        print(f"输入时间步: {config_info['temporal_config']['T_in']}")
        print(f"输出时间步: {config_info['temporal_config']['T_out']}")
        print(f"时间步长: {config_info['temporal_config']['dt']}")
        print(f"图像尺寸: {config_info['default_config']['image_size']}")
        print(f"训练轮数: {config_info['training_config']['max_epochs']}")
        
        return config_info
    
    def analyze_data_loading_flow(self) -> Dict[str, Any]:
        """分析数据加载流程"""
        print("\n" + "="*60)
        print("🔄 数据加载流程分析")
        print("="*60)
        
        flow_info = {
            "dataset_class": "TemporalPDEBenchBase",
            "loading_steps": [
                "1. 从HDF5文件读取原始数据",
                "2. 按时序配置切分时间序列",
                "3. 应用归一化处理",
                "4. 调整空间分辨率",
                "5. 分割输入和目标序列",
                "6. 返回字典格式数据"
            ],
            "data_transformations": [
                "时序切分: 根据T_in和T_out参数",
                "空间调整: 调整到指定image_size",
                "归一化: z-score标准化",
                "格式转换: 转换为PyTorch张量"
            ],
            "output_format": {
                "input_sequence": "[T_in, C, H, W]",
                "target_sequence": "[T_out, C, H, W]", 
                "full_sequence": "[T_in+T_out, C, H, W]",
                "case_id": "字符串",
                "time_info": "字典"
            }
        }
        
        # 尝试导入并分析数据集
        try:
            from datasets.temporal_pdebench import TemporalPDEBenchBase
            
            print("创建数据集实例...")
            
            # 使用默认参数创建数据集
            dataset = TemporalPDEBenchBase(
                data_path=self.data_path,
                keys=["u"],
                T_in=4,
                T_out=3,
                dt=0.1,
                temporal_mode="sequential",
                image_size=128,
                normalize=True,
                use_official_format=True,
                split="train"
            )
            
            # 记录数据集信息
            flow_info["dataset_info"] = {
                "total_samples": len(dataset),
                "n_timesteps": dataset.n_timesteps,
                "case_ids_count": len(dataset.case_ids),
                "temporal_indices_count": len(dataset.temporal_indices)
            }
            
            print(f"数据集样本总数: {len(dataset)}")
            print(f"时间步总数: {dataset.n_timesteps}")
            print(f"案例数量: {len(dataset.case_ids)}")
            
            # 获取样本数据
            if len(dataset) > 0:
                print("获取样本数据...")
                sample = dataset[0]
                
                if sample is not None:
                    flow_info["sample_data"] = {
                        "input_sequence_shape": list(sample['input_sequence'].shape),
                        "target_sequence_shape": list(sample['target_sequence'].shape),
                        "full_sequence_shape": list(sample['full_sequence'].shape),
                        "case_id": sample['case_id'],
                        "time_info": sample['time_info']
                    }
                    
                    print(f"输入序列形状: {sample['input_sequence'].shape}")
                    print(f"目标序列形状: {sample['target_sequence'].shape}")
                    print(f"完整序列形状: {sample['full_sequence'].shape}")
                    print(f"案例ID: {sample['case_id']}")
                    
                    # 数据统计
                    import torch
                    input_data = sample['input_sequence']
                    target_data = sample['target_sequence']
                    
                    flow_info["data_statistics"] = {
                        "input_min": float(torch.min(input_data)),
                        "input_max": float(torch.max(input_data)),
                        "input_mean": float(torch.mean(input_data)),
                        "input_std": float(torch.std(input_data)),
                        "target_min": float(torch.min(target_data)),
                        "target_max": float(torch.max(target_data)),
                        "target_mean": float(torch.mean(target_data)),
                        "target_std": float(torch.std(target_data))
                    }
                    
                    print(f"输入数据统计: min={flow_info['data_statistics']['input_min']:.4f}, "
                          f"max={flow_info['data_statistics']['input_max']:.4f}, "
                          f"mean={flow_info['data_statistics']['input_mean']:.4f}")
                else:
                    print("⚠️ 无法获取样本数据")
            
        except Exception as e:
            print(f"❌ 数据加载流程分析失败: {e}")
            flow_info["error"] = str(e)
        
        return flow_info
    
    def generate_data_format_documentation(self) -> Dict[str, Any]:
        """生成数据格式文档"""
        print("\n" + "="*60)
        print("📚 数据格式文档生成")
        print("="*60)
        
        doc_info = {
            "dataset_overview": {
                "name": "PDEBench 2D扩散反应方程时序数据集",
                "description": "用于时序预测的2D偏微分方程数据集",
                "equation_type": "2D扩散反应方程",
                "spatial_resolution": "128×128",
                "temporal_resolution": "可变时间步",
                "data_format": "HDF5"
            },
            "data_dimensions": {
                "input_format": "[T_in, C, H, W]",
                "output_format": "[T_out, C, H, W]",
                "full_sequence_format": "[T_in+T_out, C, H, W]",
                "batch_format": "[B, T, C, H, W]"
            },
            "coordinate_system": {
                "T": "时间维度",
                "C": "通道维度（变量数）",
                "H": "空间高度维度",
                "W": "空间宽度维度",
                "B": "批处理维度"
            },
            "data_flow": {
                "1": "从HDF5文件读取原始数据",
                "2": "按时序配置切分时间序列",
                "3": "应用归一化处理",
                "4": "调整空间分辨率",
                "5": "分割输入和目标序列",
                "6": "返回字典格式数据"
            },
            "usage_examples": {
                "training": "用于AR+NAR双头模型训练",
                "evaluation": "用于时序预测性能评估",
                "visualization": "用于结果可视化分析"
            }
        }
        
        print("数据集概览:")
        for key, value in doc_info["dataset_overview"].items():
            print(f"  {key}: {value}")
        
        print("\n数据维度说明:")
        for key, value in doc_info["data_dimensions"].items():
            print(f"  {key}: {value}")
        
        return doc_info
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """运行完整分析"""
        print("🚀 开始PDEBench时序数据集完整结构分析")
        print("="*80)
        
        # 记录分析开始时间
        start_time = datetime.now()
        
        # 执行各项分析
        self.analysis_results = {
            "analysis_metadata": {
                "timestamp": start_time.isoformat(),
                "data_file": str(self.data_path),
                "analyzer_version": "1.0.0"
            },
            "hdf5_structure": self.analyze_hdf5_structure(),
            "temporal_config": self.analyze_temporal_config(),
            "data_loading_flow": self.analyze_data_loading_flow(),
            "data_format_documentation": self.generate_data_format_documentation()
        }
        
        # 记录分析结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.analysis_results["analysis_metadata"]["duration_seconds"] = duration
        self.analysis_results["analysis_metadata"]["end_timestamp"] = end_time.isoformat()
        
        print(f"\n✅ 分析完成，耗时: {duration:.2f} 秒")
        
        return self.analysis_results
    
    def save_results(self, output_path: str = "data_structure_analysis.json") -> str:
        """保存分析结果到JSON文件"""
        print(f"\n💾 保存分析结果到: {output_path}")
        
        try:
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 分析结果已保存到: {os.path.abspath(output_path)}")
            return os.path.abspath(output_path)
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return ""


def main():
    """主函数"""
    print("🔍 PDEBench时序数据集结构分析工具")
    print("="*80)
    
    # 创建分析器
    analyzer = DataStructureAnalyzer()
    
    # 运行完整分析
    results = analyzer.run_complete_analysis()
    
    # 保存结果
    output_file = analyzer.save_results("paper_package/data_structure_analysis.json")
    
    # 打印总结
    print("\n" + "="*80)
    print("📊 分析总结")
    print("="*80)
    
    if "hdf5_structure" in results:
        hdf5_info = results["hdf5_structure"]
        print(f"HDF5文件大小: {hdf5_info.get('file_size_mb', 0)} MB")
        print(f"根级键名数量: {len(hdf5_info.get('root_keys', []))}")
    
    if "data_loading_flow" in results:
        flow_info = results["data_loading_flow"]
        if "dataset_info" in flow_info:
            dataset_info = flow_info["dataset_info"]
            print(f"数据集样本总数: {dataset_info.get('total_samples', 0)}")
            print(f"时间步总数: {dataset_info.get('n_timesteps', 0)}")
    
    if "temporal_config" in results:
        config_info = results["temporal_config"]
        temporal_config = config_info.get("temporal_config", {})
        print(f"时序配置: T_in={temporal_config.get('T_in')}, T_out={temporal_config.get('T_out')}")
    
    print(f"\n📁 详细结果已保存到: {output_file}")
    print("🎉 分析完成！")


if __name__ == "__main__":
    main()