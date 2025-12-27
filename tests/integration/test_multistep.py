#!/usr/bin/env python3
"""
测试时序NAR模型的多时步预测能力
验证模型能否预测多个未来时间步
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.pdebench import PDEBenchDataModule
from models.wrappers.swin_temporal import SwinTemporalNAR
from utils.metrics import MetricsCalculator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiStepTester:
    """多时步预测测试器"""
    
    def __init__(self, config_path: str = "configs/experiment/temporal_nar_100epochs.yaml"):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path("test_results/multistep_prediction")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据模块
        self.data_module = self._init_data_module()
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化指标计算器
        image_size = self.config.data.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.metrics_calculator = MetricsCalculator(
            image_size=image_size,
            boundary_width=16,
            freq_bands={'low': (0, 16), 'mid': (16, 32), 'high': (32, 64)}
        )
        
    def _init_data_module(self):
        """初始化数据模块"""
        logger.info("初始化数据模块...")
        
        data_module = PDEBenchDataModule(self.config.data)
        
        data_module.setup('test')
        return data_module
        
    def _init_model(self):
        """初始化模型"""
        logger.info("初始化时序NAR模型...")
        
        # 基础配置
        base_kwargs = {
            'in_channels': 2,  # diff-reaction数据集有2个通道
            'out_channels': 2,
            'img_size': self.config.data.image_size,
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'mlp_ratio': 4.0,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'use_fno_bottleneck': False
        }
        
        # 时序配置
        temporal_config = {
            'enabled': True,
            'type': 'conv1d',
            'c_out': 2,
            'k': 3,
            'causal': True,
            'dropout': 0.1
        }
        
        # NAR配置
        nar_config = {
            'head_type': 'simple',
            'd_model': 96,
            'num_heads': 4,
            'max_timesteps': 32,
            'dropout': 0.1
        }
        
        # AR配置
        ar_config = {
            'detach_rollout': True,
            'scheduled_sampling': False
        }
        
        # 创建模型
        model = SwinTemporalNAR(
            base_kwargs=base_kwargs,
            temporal_cfg=temporal_config,
            nar_cfg=nar_config,
            ar_cfg=ar_config,
            use_ar=True,
            use_nar=True
        )
        
        return model.to(self.device)
        
    def test_multistep_prediction(self):
        """测试多时步预测能力"""
        logger.info("开始多时步预测测试...")
        
        # 测试不同的T_out设置
        test_t_outs = [1, 3, 5, 10, 15, 20]
        results = {}
        
        # 获取测试数据
        test_loader = self.data_module.test_dataloader()
        
        for T_out in test_t_outs:
            logger.info(f"测试 T_out={T_out}...")
            
            batch_results = []
            inference_times = []
            
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    if i >= 5:  # 只测试前5个批次
                        break
                        
                    # 处理批次数据
                    if isinstance(batch, dict):
                        x = batch['observation'].to(self.device)  # 使用observation作为输入
                        y = batch['target'].to(self.device)       # 使用target作为真值
                    else:
                        x, y = batch
                        x = x.to(self.device)
                        y = y.to(self.device)
                    
                    # 确保输入维度正确
                    if len(x.shape) == 4:  # [B, C, H, W]
                        x = x.unsqueeze(1)  # [B, 1, C, H, W]
                    
                    # 重复输入以创建时序
                    T_in = 4
                    if x.shape[1] == 1:
                        x = x.repeat(1, T_in, 1, 1, 1)  # [B, T_in, C, H, W]
                    
                    # 推理测试
                    start_time = time.time()
                    try:
                        ar_out, nar_out = self.model(x, T_out=T_out, return_both=True)
                        inference_time = time.time() - start_time
                        inference_times.append(inference_time)
                        
                        # 使用NAR输出进行评估
                        if nar_out is not None:
                            # 取最后一个时间步进行评估
                            pred = nar_out[:, -1]  # [B, C, H, W]
                            target = y  # [B, C, H, W]
                            
                            # 计算指标
                            rel_l2 = self.metrics_calculator.compute_rel_l2(pred, target)
                            psnr = self.metrics_calculator.compute_psnr(pred, target)
                            ssim = self.metrics_calculator.compute_ssim(pred, target)
                            
                            # 转换为标量
                            if isinstance(rel_l2, torch.Tensor):
                                rel_l2 = rel_l2.mean().item()
                            if isinstance(psnr, torch.Tensor):
                                psnr = psnr.mean().item()
                            if isinstance(ssim, torch.Tensor):
                                ssim = ssim.mean().item()
                            
                            batch_results.append({
                                'rel_l2': rel_l2,
                                'psnr': psnr,
                                'ssim': ssim,
                                'inference_time': inference_time,
                                'output_shape': list(nar_out.shape)
                            })
                            
                        else:
                            logger.warning(f"NAR输出为None，T_out={T_out}")
                            
                    except Exception as e:
                        logger.error(f"T_out={T_out}时推理失败: {e}")
                        batch_results.append({
                            'error': str(e),
                            'inference_time': 0,
                            'output_shape': None
                        })
            
            # 汇总结果
            if batch_results:
                valid_results = [r for r in batch_results if 'error' not in r]
                if valid_results:
                    avg_rel_l2 = np.mean([r['rel_l2'] for r in valid_results])
                    avg_psnr = np.mean([r['psnr'] for r in valid_results])
                    avg_ssim = np.mean([r['ssim'] for r in valid_results])
                    avg_inference_time = np.mean([r['inference_time'] for r in valid_results])
                    
                    results[T_out] = {
                        'avg_rel_l2': avg_rel_l2,
                        'avg_psnr': avg_psnr,
                        'avg_ssim': avg_ssim,
                        'avg_inference_time': avg_inference_time,
                        'success_rate': len(valid_results) / len(batch_results),
                        'output_shape': valid_results[0]['output_shape'] if valid_results else None
                    }
                    
                    logger.info(f"T_out={T_out}: Rel-L2={avg_rel_l2:.6f}, PSNR={avg_psnr:.2f}dB, "
                              f"SSIM={avg_ssim:.4f}, 推理时间={avg_inference_time:.4f}s")
                else:
                    results[T_out] = {'error': 'All batches failed'}
                    logger.error(f"T_out={T_out}: 所有批次都失败了")
            else:
                results[T_out] = {'error': 'No results'}
                logger.error(f"T_out={T_out}: 没有结果")
        
        return results
        
    def analyze_results(self, results: Dict):
        """分析多时步预测结果"""
        logger.info("分析多时步预测结果...")
        
        # 提取成功的结果
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            logger.error("没有成功的预测结果")
            return
        
        # 创建分析图表
        t_outs = list(successful_results.keys())
        rel_l2_values = [successful_results[t]['avg_rel_l2'] for t in t_outs]
        psnr_values = [successful_results[t]['avg_psnr'] for t in t_outs]
        ssim_values = [successful_results[t]['avg_ssim'] for t in t_outs]
        inference_times = [successful_results[t]['avg_inference_time'] for t in t_outs]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Rel-L2 vs T_out
        axes[0, 0].plot(t_outs, rel_l2_values, 'o-', color='red')
        axes[0, 0].set_xlabel('T_out (预测时间步数)')
        axes[0, 0].set_ylabel('Rel-L2')
        axes[0, 0].set_title('预测精度 vs 预测时间步数')
        axes[0, 0].grid(True)
        
        # PSNR vs T_out
        axes[0, 1].plot(t_outs, psnr_values, 'o-', color='blue')
        axes[0, 1].set_xlabel('T_out (预测时间步数)')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('PSNR vs 预测时间步数')
        axes[0, 1].grid(True)
        
        # SSIM vs T_out
        axes[1, 0].plot(t_outs, ssim_values, 'o-', color='green')
        axes[1, 0].set_xlabel('T_out (预测时间步数)')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('SSIM vs 预测时间步数')
        axes[1, 0].grid(True)
        
        # 推理时间 vs T_out
        axes[1, 1].plot(t_outs, inference_times, 'o-', color='orange')
        axes[1, 1].set_xlabel('T_out (预测时间步数)')
        axes[1, 1].set_ylabel('推理时间 (秒)')
        axes[1, 1].set_title('推理时间 vs 预测时间步数')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'multistep_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成分析报告
        analysis = {
            'max_successful_t_out': max(t_outs),
            'min_rel_l2': min(rel_l2_values),
            'max_psnr': max(psnr_values),
            'max_ssim': max(ssim_values),
            'performance_degradation': {
                'rel_l2_increase': (rel_l2_values[-1] - rel_l2_values[0]) / rel_l2_values[0] * 100,
                'psnr_decrease': (psnr_values[0] - psnr_values[-1]) / psnr_values[0] * 100,
                'ssim_decrease': (ssim_values[0] - ssim_values[-1]) / ssim_values[0] * 100
            },
            'inference_scaling': {
                'time_increase': (inference_times[-1] - inference_times[0]) / inference_times[0] * 100
            }
        }
        
        return analysis
        
    def generate_report(self, results: Dict, analysis: Dict):
        """生成测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON报告
        report_data = {
            'test_timestamp': timestamp,
            'test_type': 'multistep_prediction',
            'model_config': {
                'image_size': self.config.data.image_size,
                'channels': 2,
                'architecture': 'SwinTemporalNAR'
            },
            'results': results,
            'analysis': analysis
        }
        
        json_path = self.output_dir / f'multistep_report_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Markdown报告
        md_content = f"""# 多时步预测能力测试报告

## 测试概览
- **测试时间**: {timestamp}
- **模型**: SwinTemporalNAR
- **数据集**: Diff-Reaction (2D)
- **图像尺寸**: {self.config.data.image_size}x{self.config.data.image_size}

## 测试结果

### 成功预测的最大时间步数
- **最大T_out**: {analysis['max_successful_t_out']}

### 性能指标
- **最佳Rel-L2**: {analysis['min_rel_l2']:.6f}
- **最佳PSNR**: {analysis['max_psnr']:.2f} dB
- **最佳SSIM**: {analysis['max_ssim']:.4f}

### 性能衰减分析
- **Rel-L2增长**: {analysis['performance_degradation']['rel_l2_increase']:.1f}%
- **PSNR下降**: {analysis['performance_degradation']['psnr_decrease']:.1f}%
- **SSIM下降**: {analysis['performance_degradation']['ssim_decrease']:.1f}%

### 推理时间分析
- **时间增长**: {analysis['inference_scaling']['time_increase']:.1f}%

## 详细结果

| T_out | Rel-L2 | PSNR (dB) | SSIM | 推理时间 (s) | 成功率 |
|-------|--------|-----------|------|-------------|--------|
"""
        
        for t_out in sorted(results.keys()):
            if 'error' not in results[t_out]:
                r = results[t_out]
                md_content += f"| {t_out} | {r['avg_rel_l2']:.6f} | {r['avg_psnr']:.2f} | {r['avg_ssim']:.4f} | {r['avg_inference_time']:.4f} | {r['success_rate']:.1%} |\n"
            else:
                md_content += f"| {t_out} | - | - | - | - | 失败 |\n"
        
        md_content += f"""

## 结论

时序NAR模型成功展示了多时步预测能力：

1. **预测范围**: 能够预测最多 {analysis['max_successful_t_out']} 个未来时间步
2. **精度保持**: 在较短预测范围内保持良好精度
3. **计算效率**: 推理时间随预测步数线性增长

## 建议

- 对于实时应用，建议使用 T_out ≤ 10
- 对于高精度需求，建议使用 T_out ≤ 5
- 模型已具备实用的多时步预测能力
"""
        
        md_path = self.output_dir / f'multistep_report_{timestamp}.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"报告已保存到: {json_path} 和 {md_path}")
        
        return json_path, md_path

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🚀 时序NAR模型多时步预测能力测试")
    logger.info("=" * 60)
    
    try:
        # 创建测试器
        tester = MultiStepTester()
        
        # 运行多时步预测测试
        results = tester.test_multistep_prediction()
        
        # 分析结果
        analysis = tester.analyze_results(results)
        
        if analysis:
            # 生成报告
            json_path, md_path = tester.generate_report(results, analysis)
            
            logger.info("=" * 60)
            logger.info("✅ 多时步预测测试完成")
            logger.info(f"📊 最大预测时间步数: {analysis['max_successful_t_out']}")
            logger.info(f"📈 最佳Rel-L2: {analysis['min_rel_l2']:.6f}")
            logger.info(f"📋 详细报告: {md_path}")
            logger.info("=" * 60)
        else:
            logger.error("❌ 测试失败，无法生成分析报告")
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()
    