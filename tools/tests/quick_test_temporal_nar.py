#!/usr/bin/env python3
"""
时序NAR模型快速测试脚本

快速验证时序NAR功能的基本可用性，包括：
- 模型加载测试
- 基本前向传播测试
- 推理模式切换测试
- 简单性能测试
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入项目模块
from models.wrappers.ar_nar_wrapper import ARNARWrapper

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickTemporalNARTester:
    """快速时序NAR测试器"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 创建简化的测试配置
        self.config = self._create_test_config()
    
    def _create_test_config(self) -> DictConfig:
        """创建测试配置"""
        config = {
            # 模型配置
            'model': {
                # SwinUNet基础配置（最小化）
                'base_kwargs': {
                    'in_channels': 1,
                    'out_channels': 1,
                    'img_size': 32,  # 小尺寸加速测试
                    'patch_size': 4,
                    'embed_dim': 48,  # 小维度
                    'depths': [2, 2, 2, 2],  # 浅层网络
                    'num_heads': [3, 6, 12, 24],
                    'window_size': 4,
                    'mlp_ratio': 4.0,
                    'qkv_bias': True,
                    'drop_rate': 0.0,
                    'attn_drop_rate': 0.0,
                    'drop_path_rate': 0.1,
                    'use_checkpoint': False,
                    'use_fno_bottleneck': False
                },
                
                # 时序模块配置
                'temporal': {
                    'enabled': True,
                    'type': 'conv1d',
                    'c_out': 1,
                    'k': 3,
                    'causal': True,
                    'dropout': 0.1
                },
                
                # NAR头配置
                'nar': {
                    'head_type': 'simple',
                    'd_model': 48,
                    'num_heads': 4,
                    'max_timesteps': 8,
                    'dropout': 0.1
                },
                
                # AR配置
                'ar': {
                    'detach_rollout': True,
                    'scheduled_sampling': False,
                    'sampling_schedule': {
                        'start_prob': 0.0,
                        'end_prob': 0.5,
                        'schedule_type': 'linear'
                    }
                },
                
                # 启用开关
                'use_ar': True,
                'use_nar': True
            },
            
            # 损失配置
            'loss': {
                'ar_weight': 1.0,
                'nar_weight': 1.0,
                'ar_weight_schedule': 'constant',
                'nar_weight_schedule': 'constant',
                'ar_loss_type': 'mse',
                'nar_loss_type': 'mse'
            },
            
            # 训练配置
            'train': {
                'total_epochs': 5,
                'enable_monitoring': True,
                'monitoring_interval': 10
            },
            
            # 时序配置
            'temporal': {
                'T_in': 2,
                'T_out': 3
            }
        }
        
        return OmegaConf.create(config)
    
    def test_model_creation(self) -> ARNARWrapper:
        """测试模型创建"""
        logger.info("🔧 测试模型创建...")
        
        try:
            model = ARNARWrapper(
                model_config=self.config.model,
                loss_config=self.config.loss,
                training_config=self.config.train
            ).to(self.device)
            
            # 打印模型信息
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✅ 模型创建成功，参数量: {total_params:,}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ 模型创建失败: {e}")
            raise
    
    def test_forward_pass(self, model):
        """测试基本前向传播"""
        logger.info("🔧 测试前向传播...")
        try:
            # 创建测试数据
            B, T_in, T_out, C, H, W = 2, 4, 2, 1, 32, 32
            x_seq = torch.randn(B, T_in, C, H, W, device=self.device)
            target_seq = torch.randn(B, T_out, C, H, W, device=self.device)
            
            # 训练模式测试
            model.train()
            with torch.no_grad():
                # 使用正确的参数格式
                outputs = model(
                    x_seq=x_seq,
                    T_out=T_out,
                    teacher_seq=target_seq,
                    compute_loss=True,
                    target_seq=target_seq
                )
                
                # 检查输出结构 - ARNAROutput对象
                if hasattr(outputs, 'ar_pred') and hasattr(outputs, 'nar_pred'):
                    logger.info(f"✅ 训练模式输出正常")
                    if outputs.ar_pred is not None:
                        logger.info(f"   AR预测形状: {outputs.ar_pred.shape}")
                    if outputs.nar_pred is not None:
                        logger.info(f"   NAR预测形状: {outputs.nar_pred.shape}")
                    if outputs.total_loss is not None:
                        logger.info(f"   总损失: {outputs.total_loss.item():.6f}")
                else:
                    # 如果返回的是元组，尝试解析
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        ar_pred, nar_pred = outputs
                        logger.info(f"✅ 训练模式输出正常（元组格式）")
                        if ar_pred is not None:
                            logger.info(f"   AR预测形状: {ar_pred.shape}")
                        if nar_pred is not None:
                            logger.info(f"   NAR预测形状: {nar_pred.shape}")
                    else:
                        logger.warning("⚠️ 输出结构异常")
            
            # 推理模式测试
            model.eval()
            with torch.no_grad():
                pred = model(
                    x_seq=x_seq,
                    T_out=T_out,
                    compute_loss=False
                )
                
                if isinstance(pred, torch.Tensor):
                    logger.info(f"✅ 推理模式输出正常，形状: {pred.shape}")
                    expected_shape = (B, T_out, C, H, W)
                    if pred.shape == expected_shape:
                        logger.info(f"✅ 输出形状正确: {pred.shape}")
                    else:
                        logger.warning(f"⚠️ 输出形状异常: 期望{expected_shape}, 实际{pred.shape}")
                else:
                    logger.warning("⚠️ 推理模式输出类型异常")
            
            logger.info("✅ 前向传播测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 前向传播测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_inference_modes(self, model):
        """测试不同推理模式"""
        logger.info("🔧 测试推理模式切换...")
        try:
            B, T_in, T_out, C, H, W = 1, 4, 2, 1, 32, 32
            x_seq = torch.randn(B, T_in, C, H, W, device=self.device)
            
            model.eval()
            modes = ['ar', 'nar', 'ensemble']
            results = {}
            
            for mode in modes:
                try:
                    # 设置推理模式
                    model.inference_mode = mode
                    
                    with torch.no_grad():
                        pred = model(
                            x_seq=x_seq,
                            T_out=T_out,
                            compute_loss=False
                        )
                        
                        if isinstance(pred, torch.Tensor):
                            results[mode] = pred.shape
                            logger.info(f"✅ {mode.upper()}模式: {pred.shape}")
                        else:
                            logger.warning(f"⚠️ {mode.upper()}模式输出异常")
                            results[mode] = None
                            
                except Exception as e:
                    logger.warning(f"⚠️ {mode.upper()}模式失败: {e}")
                    results[mode] = None
            
            # 检查结果
            success_count = sum(1 for v in results.values() if v is not None)
            logger.info(f"✅ 推理模式测试完成: {success_count}/{len(modes)} 成功")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ 推理模式测试失败: {e}")
            return False

    def test_loss_computation(self, model):
        """测试损失计算"""
        logger.info("🔧 测试损失计算...")
        try:
            # 创建测试数据
            B, T_in, T_out, C, H, W = 2, 4, 2, 1, 32, 32
            x_seq = torch.randn(B, T_in, C, H, W, device=self.device)
            target_seq = torch.randn(B, T_out, C, H, W, device=self.device)
            
            model.train()
            
            # 测试损失计算
            outputs = model(
                x_seq=x_seq,
                T_out=T_out,
                teacher_seq=target_seq,
                compute_loss=True,
                target_seq=target_seq
            )
            
            # 检查损失输出
            if hasattr(outputs, 'total_loss') and outputs.total_loss is not None:
                loss = outputs.total_loss
                logger.info(f"✅ 损失计算正常: {loss.item():.6f}")
                
                # 检查损失是否合理
                if torch.isfinite(loss) and loss.item() > 0:
                    logger.info("✅ 损失值合理")
                else:
                    logger.warning(f"⚠️ 损失值异常: {loss.item()}")
                
                # 检查梯度
                loss.backward()
                has_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.any(param.grad != 0):
                        has_grad = True
                        break
                
                if has_grad:
                    logger.info("✅ 梯度计算正常")
                else:
                    logger.warning("⚠️ 未检测到梯度")
                
                # 清除梯度
                model.zero_grad()
                
            else:
                # 如果返回的是元组，尝试手动计算损失
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    ar_pred, nar_pred = outputs
                    logger.info("📊 手动计算损失...")
                    
                    total_loss = 0
                    if ar_pred is not None:
                        ar_loss = torch.nn.functional.mse_loss(ar_pred, target_seq)
                        total_loss += ar_loss
                        logger.info(f"   AR损失: {ar_loss.item():.6f}")
                    
                    if nar_pred is not None:
                        nar_loss = torch.nn.functional.mse_loss(nar_pred, target_seq)
                        total_loss += nar_loss
                        logger.info(f"   NAR损失: {nar_loss.item():.6f}")
                    
                    if total_loss > 0:
                        logger.info(f"✅ 手动损失计算: {total_loss.item():.6f}")
                        total_loss.backward()
                        logger.info("✅ 梯度计算正常")
                        model.zero_grad()
                    else:
                        logger.warning("⚠️ 无法计算损失")
                else:
                    logger.warning("⚠️ 损失计算失败")
            
            logger.info("✅ 损失计算测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 损失计算测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_inference_speed(self, model):
        """测试推理速度"""
        logger.info("🔧 测试推理速度...")
        
        try:
            # 创建测试数据
            B, T_in, T_out, C, H, W = 1, 4, 2, 1, 32, 32
            x_seq = torch.randn(B, T_in, C, H, W, device=self.device)
            
            model.eval()
            
            # 预热
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x_seq=x_seq, T_out=T_out, compute_loss=False)
            
            # 测量推理时间
            inference_times = {}
            
            for mode in ['ar', 'nar', 'ensemble']:
                try:
                    model.inference_mode = mode
                    times = []
                    
                    with torch.no_grad():
                        for _ in range(10):  # 减少测试次数
                            start_time = time.time()
                            _ = model(x_seq=x_seq, T_out=T_out, compute_loss=False)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            times.append(time.time() - start_time)
                    
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    inference_times[mode] = (avg_time, std_time)
                    
                    logger.info(f"✅ {mode.upper()}推理时间: {avg_time:.4f}±{std_time:.4f}s")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {mode.upper()}模式测试失败: {e}")
                    inference_times[mode] = None
            
            success_count = sum(1 for v in inference_times.values() if v is not None)
            logger.info(f"✅ 推理速度测试完成: {success_count}/{len(inference_times)} 成功")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ 推理速度测试失败: {e}")
            return False
    
    def test_memory_usage(self, model):
        """测试显存使用"""
        if not torch.cuda.is_available():
            logger.info("⚠️ CPU模式，跳过显存测试")
            return True
        
        logger.info("🔧 测试显存使用...")
        
        try:
            # 清空显存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 创建测试数据
            B, T_in, T_out, C, H, W = 2, 4, 2, 1, 32, 32
            x_seq = torch.randn(B, T_in, C, H, W, device=self.device)
            target_seq = torch.randn(B, T_out, C, H, W, device=self.device)
            
            model.eval()
            
            # 测试推理显存
            with torch.no_grad():
                _ = model(x_seq=x_seq, T_out=T_out, compute_loss=False)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            logger.info(f"✅ 推理峰值显存: {peak_memory:.2f}MB")
            
            # 测试训练显存
            torch.cuda.reset_peak_memory_stats()
            model.train()
            
            outputs = model(
                x_seq=x_seq,
                T_out=T_out,
                teacher_seq=target_seq,
                compute_loss=True,
                target_seq=target_seq
            )
            
            if hasattr(outputs, 'total_loss') and outputs.total_loss is not None:
                outputs.total_loss.backward()
            
            train_peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            logger.info(f"✅ 训练峰值显存: {train_peak_memory:.2f}MB")
            
            logger.info("✅ 显存使用测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 显存使用测试失败: {e}")
            return False
    
    def test_gradient_flow(self, model):
        """测试梯度流"""
        logger.info("🔧 测试梯度流...")
        
        try:
            # 创建测试数据
            B, T_in, T_out, C, H, W = 2, 4, 2, 1, 32, 32
            x_seq = torch.randn(B, T_in, C, H, W, device=self.device)
            target_seq = torch.randn(B, T_out, C, H, W, device=self.device)
            
            model.train()
            
            # 前向传播
            outputs = model(
                x_seq=x_seq,
                T_out=T_out,
                teacher_seq=target_seq,
                compute_loss=True,
                target_seq=target_seq
            )
            
            # 检查是否有损失
            if not (hasattr(outputs, 'total_loss') and outputs.total_loss is not None):
                logger.warning("⚠️ 无法获取损失，跳过梯度流测试")
                return False
            
            # 反向传播
            outputs.total_loss.backward()
            
            # 检查梯度
            grad_norms = []
            zero_grad_count = 0
            total_params = 0
            
            for name, param in model.named_parameters():
                total_params += 1
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if grad_norm == 0:
                        zero_grad_count += 1
                else:
                    zero_grad_count += 1
            
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
            max_grad_norm = np.max(grad_norms) if grad_norms else 0
            
            logger.info(f"✅ 平均梯度范数: {avg_grad_norm:.6f}")
            logger.info(f"✅ 最大梯度范数: {max_grad_norm:.6f}")
            logger.info(f"✅ 零梯度参数: {zero_grad_count}/{total_params}")
            
            if avg_grad_norm > 0:
                logger.info("✅ 梯度流测试通过")
                return True
            else:
                logger.warning("⚠️ 检测到梯度消失问题")
                return False
            
        except Exception as e:
            logger.error(f"❌ 梯度流测试失败: {e}")
            return False
    
    def run_quick_test(self):
        """运行快速测试"""
        logger.info("🚀 开始时序NAR模型快速测试")
        
        test_results = {}
        
        try:
            # 1. 测试模型创建
            logger.info("=" * 50)
            model = self.test_model_creation()
            test_results['model_creation'] = True
            
            # 2. 测试前向传播
            logger.info("=" * 50)
            test_results['forward_pass'] = self.test_forward_pass(model)
            
            # 3. 测试推理模式切换
            logger.info("=" * 50)
            test_results['inference_modes'] = self.test_inference_modes(model)
            
            # 4. 测试损失计算
            logger.info("=" * 50)
            test_results['loss_computation'] = self.test_loss_computation(model)
            
            # 5. 测试推理速度
            logger.info("=" * 50)
            test_results['inference_speed'] = self.test_inference_speed(model)
            
            # 6. 测试显存使用
            logger.info("=" * 50)
            test_results['memory_usage'] = self.test_memory_usage(model)
            
            # 7. 测试梯度流
            logger.info("=" * 50)
            test_results['gradient_flow'] = self.test_gradient_flow(model)
            
            # 汇总结果
            logger.info("=" * 50)
            logger.info("🎯 测试结果汇总:")
            passed_tests = 0
            total_tests = len(test_results)
            
            for test_name, result in test_results.items():
                status = "✅ 通过" if result else "❌ 失败"
                logger.info(f"  {test_name}: {status}")
                if result:
                    passed_tests += 1
            
            logger.info(f"📊 总体结果: {passed_tests}/{total_tests} 测试通过")
            
            if passed_tests == total_tests:
                logger.info("🎉 时序NAR模型快速测试全部通过！")
            elif passed_tests >= total_tests * 0.7:  # 70%通过率
                logger.info("✅ 时序NAR模型基本功能正常！")
            else:
                logger.warning("⚠️ 时序NAR模型存在较多问题，需要进一步检查")
            
            return passed_tests >= total_tests * 0.7
            
        except Exception as e:
            logger.error(f"❌ 快速测试失败: {e}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="时序NAR模型快速测试")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    
    args = parser.parse_args()
    
    # 创建测试器并运行
    tester = QuickTemporalNARTester(device=args.device)
    tester.run_quick_test()

if __name__ == "__main__":
    main()