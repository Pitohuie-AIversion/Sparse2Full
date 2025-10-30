#!/usr/bin/env python3
"""
Diff-Reaction数据集时序NAR模型测试脚本

基于现有的temporal_nar_300epochs.yaml配置，测试新开发的时序NAR模型功能。
包括TimeQueryHead性能测试、时序编码器验证、AR/NAR模型对比等。
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.pdebench import PDEBenchDataModule
from models.wrappers.swin_temporal import SwinTemporalNAR
from utils.metrics import MetricsCalculator


class TemporalNARTester:
    """时序NAR模型测试器"""
    
    def __init__(self, config_path: str = "configs/experiment/temporal_nar_300epochs.yaml"):
        """初始化测试器"""
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置输出目录
        self.output_dir = Path("test_results") / f"temporal_nar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 初始化数据模块
        self.data_module = None
        self._setup_data_module()
        
        # 测试结果存储
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'config_path': str(config_path),
            'device': str(self.device),
            'tests': {}
        }
        
    def _load_config(self) -> DictConfig:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        config = OmegaConf.load(self.config_path)
        
        # 如果没有数据配置，使用diff-reaction默认配置
        if 'data' not in config:
            data_config_path = "configs/data/2d_diff_react_na_na_crop_20.yaml"
            if os.path.exists(data_config_path):
                data_config = OmegaConf.load(data_config_path)
                config.data = data_config
            else:
                # 创建默认数据配置
                config.data = OmegaConf.create({
                    'name': 'diff-reaction',
                    'data_path': 'data/2D/diff-react',
                    'image_size': [128, 128],
                    'batch_size': 4,
                    'num_workers': 2,
                    'observation': {
                        'mode': 'crop',
                        'crop_size': [20, 20]
                    }
                })
        
        # 如果没有temporal配置，创建默认配置
        if 'temporal' not in config:
            config.temporal = OmegaConf.create({
                'T_in': 4,
                'T_out': 3,
                'dt': 0.1
            })
        
        return config
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('TemporalNARTester')
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.output_dir / 'test.log')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_data_module(self):
        """设置数据模块"""
        try:
            self.data_module = PDEBenchDataModule(self.config.data)
            self.data_module.setup('test')
            # 使用更安全的方式获取数据集名称
            dataset_name = getattr(self.config.data, 'name', 'diff-reaction')
            self.logger.info(f"数据模块初始化成功: {dataset_name}")
        except Exception as e:
            self.logger.error(f"数据模块初始化失败: {e}")
            raise
    
    def create_model(self, use_ar=True, use_nar=True):
        """创建时序NAR模型"""
        # 从配置中获取数据信息，diff-reaction数据集有2个通道
        actual_channels = 2  # diff-reaction数据集固定为2通道 (u, v)
        
        self.logger.info(f"使用数据通道数: {actual_channels}")
        
        # 更新模型配置以匹配实际通道数
        model_config = self.config.model.copy()
        model_config.base_kwargs.in_channels = actual_channels
        model_config.base_kwargs.out_channels = actual_channels
        
        # 更新时序配置中的通道数
        if hasattr(model_config.temporal, 'c_out'):
            model_config.temporal.c_out = actual_channels
        
        # 确保NAR配置正确，移除不存在的参数
        if hasattr(model_config, 'nar'):
            # 移除不存在的num_queries参数，只保留实际需要的参数
            nar_config = {
                'head_type': model_config.nar.get('head_type', 'simple'),
                'd_model': model_config.nar.get('d_model', 96),
                'max_timesteps': model_config.nar.get('max_timesteps', 32),
                'dropout': model_config.nar.get('dropout', 0.1)
            }
            # 只有cross_attention类型才需要num_heads参数
            if nar_config['head_type'] == 'cross_attention':
                nar_config['num_heads'] = model_config.nar.get('num_heads', 8)
            
            model_config.nar = DictConfig(nar_config)
        
        model = SwinTemporalNAR(
            base_kwargs=model_config.base_kwargs,
            temporal_cfg=model_config.temporal,
            nar_cfg=model_config.nar,
            use_ar=use_ar,
            use_nar=use_nar
        )
        
        return model.to(self.device)
    
    def test_timequery_head_performance(self):
        """测试TimeQueryHead在不同T_out设置下的性能"""
        self.logger.info("开始测试TimeQueryHead性能...")
        
        # 使用配置文件中的T_out值，而不是硬编码
        base_t_out = self.config.temporal.T_out
        t_out_configs = [base_t_out, base_t_out + 2]  # 测试基础值和稍大的值
        results = {}
        
        for t_out in t_out_configs:
            self.logger.info(f"测试T_out={t_out}...")
            
            try:
                # 创建模型
                model = self.create_model()
                model.eval()
                
                # 性能测试
                inference_times = []
                memory_usage = []
                predictions = []
                targets = []
                
                test_loader = self.data_module.test_dataloader()
                
                with torch.no_grad():
                    for i, batch in enumerate(test_loader):
                        if i >= 5:  # 测试5个batch
                            break
                        
                        # 准备数据 - 处理字典格式的batch
                        if isinstance(batch, dict):
                            # 从字典中提取数据
                            if 'input_sequence' in batch and 'target_sequence' in batch:
                                x = batch['input_sequence'].to(self.device)
                                y = batch['target_sequence'].to(self.device)
                            elif 'observation' in batch and 'target' in batch:
                                x = batch['observation'].to(self.device)
                                y = batch['target'].to(self.device)
                            else:
                                # 使用第一个可用的张量作为输入
                                keys = list(batch.keys())
                                x = batch[keys[0]].to(self.device)
                                y = batch[keys[1]].to(self.device) if len(keys) > 1 else x
                        elif isinstance(batch, (list, tuple)):
                            x, y = batch[0].to(self.device), batch[1].to(self.device)
                        else:
                            x = batch.to(self.device)
                            y = x
                        
                        # 确保输入维度正确 [B, T, C, H, W]
                        if x.dim() == 4:  # [B, C, H, W] -> [B, T, C, H, W]
                            x = x.unsqueeze(1).repeat(1, 4, 1, 1, 1)
                        if y.dim() == 4:  # [B, C, H, W] -> [B, T, C, H, W]
                            y = y.unsqueeze(1).repeat(1, t_out, 1, 1, 1)
                        
                        # 记录显存使用
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        
                        # 推理
                        start_time = time.time()
                        outputs = model(x, T_out=t_out)
                        end_time = time.time()
                        
                        inference_times.append(end_time - start_time)
                        
                        if torch.cuda.is_available():
                            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
                        
                        # 收集预测结果
                        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                            # 返回的是(ar_output, nar_output)
                            ar_out, nar_out = outputs
                            pred = nar_out if nar_out is not None else ar_out
                        elif isinstance(outputs, dict):
                            pred = outputs.get('nar_output', outputs.get('ar_output', list(outputs.values())[0]))
                        else:
                            pred = outputs
                        
                        predictions.append(pred.cpu())
                        targets.append(y.cpu())
                
                # 计算指标
                if predictions and targets:
                    all_preds = torch.cat(predictions, dim=0)
                    all_targets = torch.cat(targets, dim=0)
                    
                    # 创建metrics计算器 - 修复频域带设置以避免与T_out冲突
                    img_size = self.config.data.image_size
                    max_freq = min(img_size, img_size) // 2
                    metrics_calc = MetricsCalculator(
                        image_size=(img_size, img_size),
                        boundary_width=16,
                        freq_bands={'low': (0, max_freq // 4), 'mid': (max_freq // 4, max_freq // 2), 'high': (max_freq // 2, max_freq)}
                    )
                    
                    rel_l2 = metrics_calc.compute_rel_l2(all_preds, all_targets)
                    psnr = metrics_calc.compute_psnr(all_preds, all_targets)
                    ssim = metrics_calc.compute_ssim(all_preds, all_targets)
                    
                    # 确保指标是标量值 - 修复scalar conversion错误
                    # 对于形状为[B, C]的张量，取平均值后转换为标量
                    if torch.is_tensor(rel_l2):
                        rel_l2_val = float(rel_l2.mean().item())
                    else:
                        rel_l2_val = float(rel_l2)
                    
                    if torch.is_tensor(psnr):
                        psnr_val = float(psnr.mean().item())
                    else:
                        psnr_val = float(psnr)
                    
                    if torch.is_tensor(ssim):
                        ssim_val = float(ssim.mean().item())
                    else:
                        ssim_val = float(ssim)
                    
                    results[f'T_out_{t_out}'] = {
                        'T_out': t_out,
                        'avg_inference_time': np.mean(inference_times),
                        'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
                        'throughput': len(predictions) * predictions[0].shape[0] / np.sum(inference_times),
                        'rel_l2': rel_l2_val,
                        'psnr': psnr_val,
                        'ssim': ssim_val
                    }
                    
                    self.logger.info(f"T_out={t_out} - Rel-L2: {rel_l2_val:.6f}, 推理时间: {np.mean(inference_times):.4f}s")
                
                # 清理显存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                self.logger.error(f"测试T_out={t_out}失败: {e}")
                results[f'T_out_{t_out}'] = {'error': str(e)}
        
        self.test_results['tests']['timequery_head_performance'] = results
        return results
    
    def test_temporal_encoders(self):
        """测试不同时序编码器的效果"""
        self.logger.info("开始测试时序编码器...")
        
        encoder_configs = [
            {'type': 'TemporalTransformer', 'name': 'Transformer_Encoder'},
            {'type': 'TemporalConv1D', 'name': 'Conv1D_Encoder'}
        ]
        
        results = {}
        
        for encoder_config in encoder_configs:
            encoder_type = encoder_config['type']
            name = encoder_config['name']
            
            self.logger.info(f"测试编码器: {name}")
            
            try:
                # 创建模型配置
                model_config = self.config.model.copy()
                model_config.temporal.encoder_type = encoder_type
                
                model = SwinTemporalNAR(
                    base_kwargs=model_config.base_kwargs,
                    temporal_cfg=model_config.temporal,
                    nar_cfg=model_config.nar,
                    use_ar=model_config.use_ar,
                    use_nar=model_config.use_nar
                )
                
                model = model.to(self.device)
                model.eval()
                
                # 测试性能
                inference_times = []
                predictions = []
                targets = []
                
                test_loader = self.data_module.test_dataloader()
                
                with torch.no_grad():
                    for i, batch in enumerate(test_loader):
                        if i >= 3:  # 测试3个batch
                            break
                        
                        # 准备数据 - 处理字典格式的batch
                        if isinstance(batch, dict):
                            # 从字典中提取数据
                            if 'input_sequence' in batch and 'target_sequence' in batch:
                                x = batch['input_sequence'].to(self.device)
                                y = batch['target_sequence'].to(self.device)
                            elif 'observation' in batch and 'target' in batch:
                                x = batch['observation'].to(self.device)
                                y = batch['target'].to(self.device)
                            else:
                                # 使用第一个可用的张量作为输入
                                keys = list(batch.keys())
                                x = batch[keys[0]].to(self.device)
                                y = batch[keys[1]].to(self.device) if len(keys) > 1 else x
                        elif isinstance(batch, (list, tuple)):
                            x, y = batch[0].to(self.device), batch[1].to(self.device)
                        else:
                            x = batch.to(self.device)
                            y = x
                        
                        # 确保输入维度正确
                        if x.dim() == 4:
                            x = x.unsqueeze(1).repeat(1, 4, 1, 1, 1)
                        if y.dim() == 4:
                            y = y.unsqueeze(1).repeat(1, 5, 1, 1, 1)
                        
                        # 推理
                        start_time = time.time()
                        outputs = model(x, T_out=5)
                        end_time = time.time()
                        
                        inference_times.append(end_time - start_time)
                        
                        # 收集预测结果
                        try:
                            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                                ar_out, nar_out = outputs
                                pred = nar_out if nar_out is not None else ar_out
                            elif isinstance(outputs, dict):
                                pred = outputs.get('nar_output', outputs.get('ar_output', list(outputs.values())[0]))
                            else:
                                pred = outputs
                        except (ValueError, TypeError) as e:
                            # 如果解包失败，直接使用outputs作为pred
                            pred = outputs
                        
                        predictions.append(pred.cpu())
                        targets.append(y.cpu())
                
                # 计算指标
                if predictions and targets:
                    all_preds = torch.cat(predictions, dim=0)
                    all_targets = torch.cat(targets, dim=0)
                    
                    # 创建metrics计算器
                    # 修复频域带设置以避免与T_out冲突
                    image_size = self.config.data.image_size
                    max_freq = min(image_size, image_size) // 2
                    metrics_calc = MetricsCalculator(
                        image_size=(image_size, image_size),
                        boundary_width=16,
                        freq_bands={'low': (0, max_freq // 4), 'mid': (max_freq // 4, max_freq // 2), 'high': (max_freq // 2, max_freq)}
                    )
                    
                    rel_l2 = metrics_calc.compute_rel_l2(all_preds, all_targets)
                    psnr = metrics_calc.compute_psnr(all_preds, all_targets)
                    ssim = metrics_calc.compute_ssim(all_preds, all_targets)
                    
                    # 确保指标是标量值 - 修复scalar conversion错误
                    # 对于形状为[B, C]的张量，取平均值后转换为标量
                    if torch.is_tensor(rel_l2):
                        rel_l2_val = float(rel_l2.mean().item())
                    else:
                        rel_l2_val = float(rel_l2)
                    
                    if torch.is_tensor(psnr):
                        psnr_val = float(psnr.mean().item())
                    else:
                        psnr_val = float(psnr)
                    
                    if torch.is_tensor(ssim):
                        ssim_val = float(ssim.mean().item())
                    else:
                        ssim_val = float(ssim)
                    
                    results[name] = {
                        'encoder_type': encoder_type,
                        'avg_inference_time': np.mean(inference_times),
                        'rel_l2': rel_l2_val,
                        'psnr': psnr_val,
                        'ssim': ssim_val
                    }
                
                self.logger.info(f"{name} - Rel-L2: {rel_l2_val:.6f}")
                self.logger.info(f"{name} - PSNR: {psnr_val:.2f}dB")
                self.logger.info(f"{name} - SSIM: {ssim_val:.4f}")
                
                # 清理显存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                self.logger.error(f"测试编码器 {name} 失败: {e}")
                results[name] = {'error': str(e)}
        
        self.test_results['tests']['temporal_encoders'] = results
        return results
    
    def compare_ar_nar_performance(self):
        """对比AR和NAR模型性能"""
        self.logger.info("开始对比AR和NAR模型性能...")
        
        model_configs = [
            {'use_ar': True, 'use_nar': False, 'name': 'AR_only'},
            {'use_ar': False, 'use_nar': True, 'name': 'NAR_only'},
            {'use_ar': True, 'use_nar': True, 'name': 'AR_NAR_hybrid'}
        ]
        
        results = {}
        
        for config in model_configs:
            name = config['name']
            self.logger.info(f"测试模型配置: {name}")
            
            try:
                # 创建模型
                model_config = self.config.model.copy()
                model_config.use_ar = config['use_ar']
                model_config.use_nar = config['use_nar']
                
                model = SwinTemporalNAR(
                    base_kwargs=model_config.base_kwargs,
                    temporal_cfg=model_config.temporal,
                    nar_cfg=model_config.nar,
                    use_ar=model_config.use_ar,
                    use_nar=model_config.use_nar
                )
                
                model = model.to(self.device)
                model.eval()
                
                # 测试性能
                inference_times = []
                predictions = []
                targets = []
                
                test_loader = self.data_module.test_dataloader()
                
                with torch.no_grad():
                    for i, batch in enumerate(test_loader):
                        if i >= 3:  # 测试3个batch
                            break
                        
                        # 准备数据 - 处理字典格式的batch
                        if isinstance(batch, dict):
                            # 从字典中提取数据
                            if 'input_sequence' in batch and 'target_sequence' in batch:
                                x = batch['input_sequence'].to(self.device)
                                y = batch['target_sequence'].to(self.device)
                            elif 'observation' in batch and 'target' in batch:
                                x = batch['observation'].to(self.device)
                                y = batch['target'].to(self.device)
                            else:
                                # 使用第一个可用的张量作为输入
                                keys = list(batch.keys())
                                x = batch[keys[0]].to(self.device)
                                y = batch[keys[1]].to(self.device) if len(keys) > 1 else x
                        elif isinstance(batch, (list, tuple)):
                            x, y = batch[0].to(self.device), batch[1].to(self.device)
                        else:
                            x = batch.to(self.device)
                            y = x
                        
                        # 确保输入维度正确
                        if x.dim() == 4:
                            x = x.unsqueeze(1).repeat(1, 4, 1, 1, 1)
                        if y.dim() == 4:
                            y = y.unsqueeze(1).repeat(1, 5, 1, 1, 1)
                        
                        # 推理
                        start_time = time.time()
                        outputs = model(x, T_out=5)
                        end_time = time.time()
                        
                        inference_times.append(end_time - start_time)
                        
                        # 收集预测结果
                        if isinstance(outputs, dict):
                            if config['use_nar'] and 'nar_output' in outputs:
                                pred = outputs['nar_output']
                            elif config['use_ar'] and 'ar_output' in outputs:
                                pred = outputs['ar_output']
                            else:
                                pred = list(outputs.values())[0]
                        elif isinstance(outputs, (tuple, list)):
                            # 处理tuple/list输出 (ar_output, nar_output)
                            try:
                                if len(outputs) == 2:
                                    ar_output, nar_output = outputs
                                elif len(outputs) == 1:
                                    # 如果只有一个输出，判断是AR还是NAR
                                    if config['use_nar'] and not config['use_ar']:
                                        ar_output, nar_output = None, outputs[0]
                                    elif config['use_ar'] and not config['use_nar']:
                                        ar_output, nar_output = outputs[0], None
                                    else:
                                        ar_output, nar_output = outputs[0], None
                                else:
                                    ar_output, nar_output = outputs[0], None
                                
                                # 根据配置选择合适的输出
                                if config['use_nar'] and not config['use_ar']:
                                    # NAR_only模式，使用nar_output
                                    pred = nar_output
                                elif config['use_ar'] and not config['use_nar']:
                                    # AR_only模式，使用ar_output
                                    pred = ar_output
                                else:
                                    # 混合模式，优先使用非None的输出
                                    pred = nar_output if nar_output is not None else ar_output
                                
                                if pred is None:
                                    # 如果选择的输出是None，跳过这个batch
                                    continue
                            except (ValueError, TypeError) as e:
                                # 如果解包失败，直接使用outputs作为pred
                                pred = outputs
                        else:
                            pred = outputs
                        
                        # 确保pred是tensor且不是None
                        if pred is not None and torch.is_tensor(pred):
                            predictions.append(pred.cpu())
                            targets.append(y.cpu())
                
                # 计算指标
                if predictions and targets:
                    all_preds = torch.cat(predictions, dim=0)
                    all_targets = torch.cat(targets, dim=0)
                    
                    # 创建metrics计算器
                    # 修复频域带设置以避免与T_out冲突
                    image_size = self.config.data.image_size
                    max_freq = min(image_size, image_size) // 2
                    metrics_calc = MetricsCalculator(
                        image_size=(image_size, image_size),
                        boundary_width=16,
                        freq_bands={'low': (0, max_freq // 4), 'mid': (max_freq // 4, max_freq // 2), 'high': (max_freq // 2, max_freq)}
                    )
                    
                    rel_l2 = metrics_calc.compute_rel_l2(all_preds, all_targets)
                    psnr = metrics_calc.compute_psnr(all_preds, all_targets)
                    ssim = metrics_calc.compute_ssim(all_preds, all_targets)
                    
                    # 确保指标是标量值 - 修复unpack错误
                    rel_l2_mean = torch.mean(rel_l2) if torch.is_tensor(rel_l2) and rel_l2.numel() > 1 else rel_l2
                    psnr_mean = torch.mean(psnr) if torch.is_tensor(psnr) and psnr.numel() > 1 else psnr
                    ssim_mean = torch.mean(ssim) if torch.is_tensor(ssim) and ssim.numel() > 1 else ssim
                    
                    results[name] = {
                        'use_ar': config['use_ar'],
                        'use_nar': config['use_nar'],
                        'avg_inference_time': np.mean(inference_times),
                        'rel_l2': float(rel_l2_mean.item()) if torch.is_tensor(rel_l2_mean) else float(rel_l2_mean),
                        'psnr': float(psnr_mean.item()) if torch.is_tensor(psnr_mean) else float(psnr_mean),
                        'ssim': float(ssim_mean.item()) if torch.is_tensor(ssim_mean) else float(ssim_mean)
                    }
                    
                    rel_l2_val = float(rel_l2_mean.item()) if torch.is_tensor(rel_l2_mean) else float(rel_l2_mean)
                    self.logger.info(f"{name} - Rel-L2: {rel_l2_val:.6f}")
                    self.logger.info(f"{name} - 推理时间: {np.mean(inference_times):.4f}s")
                
                # 清理显存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                self.logger.error(f"测试模型配置 {name} 失败: {e}")
                results[name] = {'error': str(e)}
        
        self.test_results['tests']['ar_nar_comparison'] = results
        return results
    
    def generate_performance_report(self):
        """生成性能报告"""
        self.logger.info("生成性能报告...")
        
        report_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 保存完整测试结果
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        md_report = self._generate_markdown_report()
        md_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        self.logger.info(f"性能报告已保存: {report_file}")
        self.logger.info(f"Markdown报告已保存: {md_file}")
        
        return report_file, md_file
    
    def _generate_markdown_report(self) -> str:
        """生成Markdown格式的报告"""
        report = f"""# Diff-Reaction数据集时序NAR模型测试报告

## 测试概览

- **测试时间**: {self.test_results['timestamp']}
- **配置文件**: {self.test_results['config_path']}
- **测试设备**: {self.test_results['device']}

## 测试结果

"""
        
        # TimeQueryHead性能测试
        if 'timequery_head_performance' in self.test_results['tests']:
            report += "### TimeQueryHead性能测试\n\n"
            report += "| 配置 | T_out | 推理时间(s) | 显存使用(MB) | 吞吐量(samples/s) |\n"
            report += "|------|-------|-------------|--------------|------------------|\n"
            
            for config_name, results in self.test_results['tests']['timequery_head_performance'].items():
                if 'error' not in results:
                    report += f"| {config_name} | {results['T_out']} | {results['avg_inference_time']:.4f} | {results['avg_memory_usage']:.1f} | {results['throughput']:.2f} |\n"
                else:
                    report += f"| {config_name} | - | 错误 | - | - |\n"
            
            report += "\n"
        
        # 时序编码器测试
        if 'temporal_encoders' in self.test_results['tests']:
            report += "### 时序编码器性能对比\n\n"
            report += "| 编码器类型 | 推理时间(s) | Rel-L2 | PSNR(dB) | SSIM |\n"
            report += "|------------|-------------|--------|----------|------|\n"
            
            for encoder_name, results in self.test_results['tests']['temporal_encoders'].items():
                if 'error' not in results:
                    report += f"| {results['encoder_type']} | {results['avg_inference_time']:.4f} | {results['rel_l2']:.6f} | {results['psnr']:.2f} | {results['ssim']:.4f} |\n"
                else:
                    report += f"| {encoder_name} | 错误 | - | - | - |\n"
            
            report += "\n"
        
        # AR/NAR模型对比
        if 'ar_nar_comparison' in self.test_results['tests']:
            report += "### AR/NAR模型性能对比\n\n"
            report += "| 模型配置 | 推理时间(s) | Rel-L2 | PSNR(dB) | SSIM |\n"
            report += "|----------|-------------|--------|----------|------|\n"
            
            for model_name, results in self.test_results['tests']['ar_nar_comparison'].items():
                if 'error' not in results:
                    report += f"| {model_name} | {results['avg_inference_time']:.4f} | {results['rel_l2']:.6f} | {results['psnr']:.2f} | {results['ssim']:.4f} |\n"
                else:
                    report += f"| {model_name} | 错误 | - | - | - |\n"
            
            report += "\n"
        
        report += """
## 结论

基于以上测试结果，可以得出以下结论：

1. **TimeQueryHead性能**: 不同T_out设置对模型性能和资源消耗的影响
2. **时序编码器对比**: 不同编码器类型的优劣势分析
3. **AR/NAR模型对比**: 自回归和非自回归方法的性能权衡

## 建议

根据测试结果，建议在实际应用中根据具体需求选择合适的配置：
- 对于实时性要求高的场景，推荐使用NAR模型
- 对于精度要求高的场景，可以考虑AR模型或混合模型
- 根据计算资源限制选择合适的T_out设置
"""
        
        return report
    
    def run_all_tests(self):
        """运行所有测试"""
        self.logger.info("开始运行所有测试...")
        
        try:
            # 1. TimeQueryHead性能测试
            self.test_timequery_head_performance()
            
            # 2. 时序编码器测试
            self.test_temporal_encoders()
            
            # 3. AR/NAR模型对比
            self.compare_ar_nar_performance()
            
            # 4. 生成报告
            report_files = self.generate_performance_report()
            
            self.logger.info("所有测试完成!")
            self.logger.info(f"报告文件: {report_files}")
            
            return self.test_results
            
        except Exception as e:
            self.logger.error(f"测试过程中发生错误: {e}")
            raise


def main():
    """主函数"""
    # 创建测试器
    tester = TemporalNARTester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    print("\n" + "="*50)
    print("测试完成!")
    print(f"结果保存在: {tester.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()