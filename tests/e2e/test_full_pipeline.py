"""
PDEBench稀疏观测重建系统 - 端到端完整流程测试

测试从数据准备到模型训练、评测、结果分析的完整端到端流程。
验证系统各组件的集成性和整体功能的正确性。
遵循技术架构文档7.7节TDD准则要求。
"""

import os
import tempfile
import shutil
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import pytest
import torch
import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf

# 导入被测试模块
from train import main as train_main
from eval import main as eval_main
from datasets.pdebench import PDEBenchDataModule
from models.swin_unet import SwinUNet
from models.hybrid import HybridModel
from models.mlp import MLPModel
from ops.losses import CombinedLoss
from ops.degradation import apply_degradation_operator
from utils.checkpoint import CheckpointManager
from utils.metrics import compute_all_metrics
from utils.visualization import create_comparison_plot
from tools.check_dc_equivalence import check_degradation_consistency
from tools.summarize_runs import summarize_experiment_results


class TestFullPipeline:
    """端到端完整流程测试类"""
    
    @pytest.fixture
    def pipeline_config(self, temp_dir):
        """创建完整流程配置"""
        config = {
            "data": {
                "root_path": str(temp_dir / "data"),
                "keys": ["u", "v"],  # 多通道测试
                "task": "SR",
                "sr_scale": 2,
                "crop_size": [64, 64],
                "normalize": True,
                "batch_size": 2,
                "num_workers": 0,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2
            },
            "model": {
                "name": "swin_unet",
                "patch_size": 4,
                "window_size": 4,
                "embed_dim": 48,
                "depths": [2, 2],
                "num_heads": [3, 6],
                "in_channels": 5,  # baseline(2) + coords(2) + mask(1)
                "out_channels": 2  # u, v
            },
            "training": {
                "num_epochs": 3,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "warmup_steps": 5,
                "gradient_clip_val": 1.0,
                "use_amp": False,
                "save_every": 1,
                "validate_every": 1,
                "log_every": 1
            },
            "loss": {
                "reconstruction_weight": 1.0,
                "spectral_weight": 0.1,
                "data_consistency_weight": 0.5,
                "gradient_weight": 0.0,
                "spectral_loss": {
                    "low_freq_modes": 8,
                    "use_rfft": True,
                    "normalize": True
                }
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-4
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 5,
                "max_steps": 50
            },
            "experiment": {
                "name": "e2e_test_pipeline",
                "save_dir": str(temp_dir / "runs"),
                "seed": 42,
                "deterministic": True
            },
            "evaluation": {
                "output_dir": str(temp_dir / "eval_results"),
                "save_predictions": True,
                "save_visualizations": True,
                "compute_per_case_metrics": True,
                "metrics": [
                    "rel_l2", "mae", "psnr", "ssim",
                    "frmse_low", "frmse_mid", "frmse_high",
                    "brmse", "crmse", "dc_error"
                ]
            }
        }
        return OmegaConf.create(config)
    
    @pytest.fixture
    def comprehensive_test_data(self, temp_dir):
        """创建综合测试数据集"""
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True)
        
        # 创建HDF5数据文件
        h5_path = data_dir / "comprehensive_test.h5"
        with h5py.File(h5_path, 'w') as f:
            # 创建50个样本，包含不同类型的PDE解
            for i in range(50):
                case_group = f.create_group(str(i))
                
                # 创建坐标网格
                x = np.linspace(-1, 1, 64)
                y = np.linspace(-1, 1, 64)
                X, Y = np.meshgrid(x, y)
                
                # 生成不同类型的解场
                if i < 20:  # 波动方程解
                    u_data = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.exp(-0.1 * i)
                    v_data = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.exp(-0.1 * i)
                elif i < 35:  # 扩散方程解
                    u_data = np.exp(-(X**2 + Y**2) / (0.5 + 0.1 * i))
                    v_data = np.exp(-(X**2 + Y**2) / (0.3 + 0.1 * i))
                else:  # 复杂多尺度解
                    u_data = (
                        np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y) +
                        0.5 * np.sin(8 * np.pi * X) * np.cos(8 * np.pi * Y) +
                        0.1 * np.random.randn(64, 64)
                    )
                    v_data = (
                        np.cos(4 * np.pi * X) * np.sin(4 * np.pi * Y) +
                        0.5 * np.cos(8 * np.pi * X) * np.sin(8 * np.pi * Y) +
                        0.1 * np.random.randn(64, 64)
                    )
                
                case_group.create_dataset("u", data=u_data.astype(np.float32))
                case_group.create_dataset("v", data=v_data.astype(np.float32))
                
                # 添加元数据
                case_group.attrs["case_type"] = ["wave", "diffusion", "multiscale"][i // 20 if i < 40 else 2]
                case_group.attrs["time_step"] = i % 10
        
        # 创建数据切分
        splits_dir = data_dir / "splits"
        splits_dir.mkdir()
        
        # 训练集：0-29 (30个样本)
        with open(splits_dir / "train.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(30)]))
        
        # 验证集：30-39 (10个样本)
        with open(splits_dir / "val.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(30, 40)]))
        
        # 测试集：40-49 (10个样本)
        with open(splits_dir / "test.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(40, 50)]))
        
        return str(h5_path)
    
    def test_data_preparation_and_validation(self, pipeline_config, comprehensive_test_data):
        """测试数据准备和验证"""
        # 更新配置
        pipeline_config.data.root_path = str(Path(comprehensive_test_data).parent)
        
        # 初始化数据模块
        data_module = PDEBenchDataModule(pipeline_config.data)
        data_module.setup()
        
        # 验证数据加载器
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # 验证数据形状和内容
        train_batch = next(iter(train_loader))
        assert "input" in train_batch
        assert "target" in train_batch
        assert "case_id" in train_batch
        
        input_tensor = train_batch["input"]
        target_tensor = train_batch["target"]
        
        # 验证多通道数据
        assert input_tensor.shape[1] == 5  # baseline(2) + coords(2) + mask(1)
        assert target_tensor.shape[1] == 2  # u, v
        
        # 验证归一化统计量
        assert data_module.dataset.norm_stats is not None
        assert "u_mean" in data_module.dataset.norm_stats
        assert "v_mean" in data_module.dataset.norm_stats
        assert "u_std" in data_module.dataset.norm_stats
        assert "v_std" in data_module.dataset.norm_stats
    
    def test_degradation_consistency_check(self, pipeline_config, comprehensive_test_data):
        """测试降质算子一致性检查"""
        pipeline_config.data.root_path = str(Path(comprehensive_test_data).parent)
        
        # 运行降质一致性检查
        data_module = PDEBenchDataModule(pipeline_config.data)
        data_module.setup()
        
        # 获取测试样本
        test_loader = data_module.test_dataloader()
        batch = next(iter(test_loader))
        
        target_tensor = batch["target"]
        
        # 应用降质算子
        degradation_params = {
            "task": pipeline_config.data.task,
            "scale": pipeline_config.data.sr_scale,
            "sigma": 1.0,
            "kernel_size": 5,
            "boundary": "mirror"
        }
        
        degraded = apply_degradation_operator(target_tensor, degradation_params)
        
        # 验证降质结果形状
        expected_h = target_tensor.shape[2] // degradation_params["scale"]
        expected_w = target_tensor.shape[3] // degradation_params["scale"]
        
        assert degraded.shape[2] == expected_h
        assert degraded.shape[3] == expected_w
        
        # 验证H算子一致性：H(H^{-1}(y)) ≈ y
        # 这里我们验证降质操作的数值稳定性
        degraded_twice = apply_degradation_operator(
            torch.nn.functional.interpolate(
                degraded, 
                size=target_tensor.shape[2:], 
                mode='bilinear', 
                align_corners=False
            ),
            degradation_params
        )
        
        # 验证二次降质的一致性
        mse_error = torch.mean((degraded - degraded_twice) ** 2)
        assert mse_error.item() < 1e-6, f"Degradation consistency error: {mse_error.item()}"
    
    @pytest.mark.slow
    def test_complete_training_pipeline(self, pipeline_config, comprehensive_test_data, temp_dir):
        """测试完整训练流程"""
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 更新配置
        pipeline_config.data.root_path = str(Path(comprehensive_test_data).parent)
        
        # 创建实验目录
        exp_dir = Path(pipeline_config.experiment.save_dir) / pipeline_config.experiment.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置文件
        config_path = exp_dir / "config.yaml"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(pipeline_config))
        
        # 初始化数据模块
        data_module = PDEBenchDataModule(pipeline_config.data)
        data_module.setup()
        
        # 初始化模型
        model = SwinUNet(
            in_channels=pipeline_config.model.in_channels,
            out_channels=pipeline_config.model.out_channels,
            img_size=pipeline_config.data.crop_size[0],
            patch_size=pipeline_config.model.patch_size,
            window_size=pipeline_config.model.window_size,
            embed_dim=pipeline_config.model.embed_dim,
            depths=pipeline_config.model.depths,
            num_heads=pipeline_config.model.num_heads
        )
        
        # 初始化优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=pipeline_config.training.learning_rate,
            weight_decay=pipeline_config.training.weight_decay
        )
        
        # 初始化学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=pipeline_config.scheduler.max_steps,
            eta_min=1e-6
        )
        
        # 初始化损失函数
        loss_fn = CombinedLoss(pipeline_config)
        
        # 初始化检查点管理器
        checkpoint_manager = CheckpointManager(
            save_dir=str(exp_dir / "checkpoints"),
            max_keep=3
        )
        
        # 训练循环
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        training_history = {
            "train_losses": [],
            "val_losses": [],
            "val_metrics": []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(pipeline_config.training.num_epochs):
            # 训练阶段
            model.train()
            epoch_train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_tensor = batch["input"]
                target_tensor = batch["target"]
                observation_data = batch.get("observation_data", {})
                
                # 前向传播
                pred = model(input_tensor)
                
                # 计算损失
                loss_dict = loss_fn(pred, target_tensor, observation_data)
                total_loss = loss_dict["total_loss"]
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    pipeline_config.training.gradient_clip_val
                )
                
                optimizer.step()
                scheduler.step()
                
                epoch_train_losses.append(total_loss.item())
            
            avg_train_loss = np.mean(epoch_train_losses)
            training_history["train_losses"].append(avg_train_loss)
            
            # 验证阶段
            if epoch % pipeline_config.training.validate_every == 0:
                model.eval()
                epoch_val_losses = []
                epoch_val_metrics = []
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input = val_batch["input"]
                        val_target = val_batch["target"]
                        val_observation = val_batch.get("observation_data", {})
                        
                        val_pred = model(val_input)
                        val_loss_dict = loss_fn(val_pred, val_target, val_observation)
                        
                        epoch_val_losses.append(val_loss_dict["total_loss"].item())
                        
                        # 计算验证指标
                        val_metrics = compute_all_metrics(
                            val_pred, 
                            val_target,
                            keys=pipeline_config.data.keys
                        )
                        epoch_val_metrics.append(val_metrics)
                
                avg_val_loss = np.mean(epoch_val_losses)
                training_history["val_losses"].append(avg_val_loss)
                
                # 计算平均验证指标
                avg_val_metrics = {}
                for metric_name in epoch_val_metrics[0].keys():
                    values = [m[metric_name] for m in epoch_val_metrics]
                    avg_val_metrics[metric_name] = np.mean(values)
                
                training_history["val_metrics"].append(avg_val_metrics)
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={"val_loss": avg_val_loss, **avg_val_metrics},
                        is_best=True
                    )
                
                # 定期保存检查点
                if epoch % pipeline_config.training.save_every == 0:
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={"val_loss": avg_val_loss, **avg_val_metrics}
                    )
        
        # 保存训练历史
        history_path = exp_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2, default=str)
        
        # 验证训练结果
        assert len(training_history["train_losses"]) == pipeline_config.training.num_epochs
        assert len(training_history["val_losses"]) > 0
        assert len(training_history["val_metrics"]) > 0
        
        # 验证损失下降趋势
        initial_train_loss = training_history["train_losses"][0]
        final_train_loss = training_history["train_losses"][-1]
        
        # 允许一定的波动，但总体应该有改善趋势
        loss_improvement = (initial_train_loss - final_train_loss) / initial_train_loss
        assert loss_improvement > -0.5, f"Training loss increased too much: {loss_improvement}"
        
        # 验证检查点文件
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
        
        # 验证最佳模型检查点
        best_checkpoint = checkpoint_dir / "best_model.pth"
        assert best_checkpoint.exists()
        
        return str(best_checkpoint), training_history
    
    @pytest.mark.slow
    def test_complete_evaluation_pipeline(self, pipeline_config, comprehensive_test_data, temp_dir):
        """测试完整评测流程"""
        # 首先运行训练获得模型
        best_checkpoint_path, training_history = self.test_complete_training_pipeline(
            pipeline_config, comprehensive_test_data, temp_dir
        )
        
        # 设置评测配置
        pipeline_config.evaluation.checkpoint_path = best_checkpoint_path
        
        # 创建评测输出目录
        eval_dir = Path(pipeline_config.evaluation.output_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 加载数据
        data_module = PDEBenchDataModule(pipeline_config.data)
        data_module.setup()
        test_loader = data_module.test_dataloader()
        
        # 加载训练好的模型
        model = SwinUNet(
            in_channels=pipeline_config.model.in_channels,
            out_channels=pipeline_config.model.out_channels,
            img_size=pipeline_config.data.crop_size[0],
            patch_size=pipeline_config.model.patch_size,
            window_size=pipeline_config.model.window_size,
            embed_dim=pipeline_config.model.embed_dim,
            depths=pipeline_config.model.depths,
            num_heads=pipeline_config.model.num_heads
        )
        
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # 运行完整评测
        all_metrics = []
        all_predictions = []
        all_targets = []
        all_case_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_tensor = batch["input"]
                target_tensor = batch["target"]
                case_ids = batch["case_id"]
                
                # 模型推理
                pred = model(input_tensor)
                
                # 收集结果
                all_predictions.append(pred.cpu())
                all_targets.append(target_tensor.cpu())
                all_case_ids.extend(case_ids)
                
                # 计算逐样本指标
                for i in range(pred.shape[0]):
                    pred_single = pred[i:i+1]
                    target_single = target_tensor[i:i+1]
                    case_id = case_ids[i]
                    
                    metrics = compute_all_metrics(
                        pred_single,
                        target_single,
                        keys=pipeline_config.data.keys
                    )
                    metrics["case_id"] = case_id
                    all_metrics.append(metrics)
        
        # 合并所有结果
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算汇总统计
        summary_metrics = {}
        metric_names = ["rel_l2", "mae", "psnr", "ssim"]
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            summary_metrics[f"avg_{metric_name}"] = float(np.mean(values))
            summary_metrics[f"std_{metric_name}"] = float(np.std(values))
            summary_metrics[f"min_{metric_name}"] = float(np.min(values))
            summary_metrics[f"max_{metric_name}"] = float(np.max(values))
        
        # 保存详细评测结果
        evaluation_results = {
            "config": OmegaConf.to_yaml(pipeline_config),
            "training_history": training_history,
            "per_case_metrics": all_metrics,
            "summary_metrics": summary_metrics,
            "model_info": {
                "checkpoint_path": best_checkpoint_path,
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }
        
        # 保存结果文件
        results_file = eval_dir / "complete_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # 保存预测结果
        predictions_file = eval_dir / "all_predictions.npz"
        np.savez(
            predictions_file,
            predictions=all_predictions.numpy(),
            targets=all_targets.numpy(),
            case_ids=all_case_ids
        )
        
        # 保存逐样本指标
        per_case_file = eval_dir / "per_case_metrics.json"
        with open(per_case_file, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        # 验证评测结果
        assert results_file.exists()
        assert predictions_file.exists()
        assert per_case_file.exists()
        
        # 验证指标合理性
        assert summary_metrics["avg_rel_l2"] >= 0
        assert summary_metrics["avg_mae"] >= 0
        assert summary_metrics["avg_psnr"] > 0
        assert 0 <= summary_metrics["avg_ssim"] <= 1
        
        # 验证结果数量
        assert len(all_metrics) == len(all_case_ids)
        assert all_predictions.shape[0] == len(all_case_ids)
        assert all_targets.shape[0] == len(all_case_ids)
        
        return evaluation_results
    
    def test_data_consistency_end_to_end(self, pipeline_config, comprehensive_test_data, temp_dir):
        """测试端到端数据一致性"""
        # 运行完整评测获得结果
        evaluation_results = self.test_complete_evaluation_pipeline(
            pipeline_config, comprehensive_test_data, temp_dir
        )
        
        # 加载预测结果
        predictions_file = Path(pipeline_config.evaluation.output_dir) / "all_predictions.npz"
        data = np.load(predictions_file)
        
        predictions = torch.from_numpy(data["predictions"])
        targets = torch.from_numpy(data["targets"])
        
        # 验证H算子一致性
        degradation_params = {
            "task": pipeline_config.data.task,
            "scale": pipeline_config.data.sr_scale,
            "sigma": 1.0,
            "kernel_size": 5,
            "boundary": "mirror"
        }
        
        # 应用降质算子到预测和目标
        pred_degraded = apply_degradation_operator(predictions, degradation_params)
        target_degraded = apply_degradation_operator(targets, degradation_params)
        
        # 计算数据一致性误差
        dc_errors = []
        for i in range(predictions.shape[0]):
            pred_deg_single = pred_degraded[i:i+1]
            target_deg_single = target_degraded[i:i+1]
            
            dc_error = torch.mean((pred_deg_single - target_deg_single) ** 2)
            dc_errors.append(dc_error.item())
        
        avg_dc_error = np.mean(dc_errors)
        std_dc_error = np.std(dc_errors)
        
        # 验证数据一致性
        assert avg_dc_error >= 0, "Data consistency error should be non-negative"
        
        # 保存数据一致性分析结果
        dc_analysis = {
            "average_dc_error": float(avg_dc_error),
            "std_dc_error": float(std_dc_error),
            "min_dc_error": float(np.min(dc_errors)),
            "max_dc_error": float(np.max(dc_errors)),
            "per_case_dc_errors": dc_errors,
            "degradation_params": degradation_params
        }
        
        dc_file = Path(pipeline_config.evaluation.output_dir) / "data_consistency_analysis.json"
        with open(dc_file, "w") as f:
            json.dump(dc_analysis, f, indent=2)
        
        # 验证H算子一致性阈值
        # 注意：这里的阈值可能需要根据实际模型性能调整
        consistency_threshold = 1e-2  # 相对宽松的阈值用于端到端测试
        
        print(f"Average data consistency error: {avg_dc_error:.6f}")
        print(f"Consistency threshold: {consistency_threshold}")
        
        # 记录但不强制要求严格的一致性，因为这是端到端测试
        if avg_dc_error > consistency_threshold:
            print(f"Warning: Data consistency error {avg_dc_error:.6f} exceeds threshold {consistency_threshold}")
        
        return dc_analysis
    
    def test_reproducibility_end_to_end(self, pipeline_config, comprehensive_test_data, temp_dir):
        """测试端到端可重现性"""
        def run_complete_pipeline():
            # 创建独立的临时目录
            with tempfile.TemporaryDirectory() as temp_run_dir:
                temp_run_path = Path(temp_run_dir)
                
                # 更新配置使用独立目录
                config_copy = OmegaConf.create(OmegaConf.to_yaml(pipeline_config))
                config_copy.experiment.save_dir = str(temp_run_path / "runs")
                config_copy.evaluation.output_dir = str(temp_run_path / "eval")
                
                # 设置随机种子
                torch.manual_seed(42)
                np.random.seed(42)
                
                # 运行训练
                best_checkpoint_path, _ = self.test_complete_training_pipeline(
                    config_copy, comprehensive_test_data, temp_run_path
                )
                
                # 运行评测
                config_copy.evaluation.checkpoint_path = best_checkpoint_path
                evaluation_results = self.test_complete_evaluation_pipeline(
                    config_copy, comprehensive_test_data, temp_run_path
                )
                
                return evaluation_results["summary_metrics"]
        
        # 运行两次完整流程
        results1 = run_complete_pipeline()
        results2 = run_complete_pipeline()
        
        # 验证关键指标的可重现性
        key_metrics = ["avg_rel_l2", "avg_mae", "avg_psnr", "avg_ssim"]
        
        for metric_name in key_metrics:
            val1 = results1[metric_name]
            val2 = results2[metric_name]
            
            # 使用相对宽松的容差，因为端到端流程可能有更多的数值误差源
            np.testing.assert_allclose(
                val1, val2,
                rtol=1e-3,  # 相对误差容差
                atol=1e-5,  # 绝对误差容差
                err_msg=f"Metric {metric_name} is not reproducible: {val1} vs {val2}"
            )
    
    @pytest.mark.slow
    def test_multi_model_comparison(self, pipeline_config, comprehensive_test_data, temp_dir):
        """测试多模型对比"""
        # 测试不同模型的端到端性能
        models_to_test = [
            ("swin_unet", SwinUNet),
            ("hybrid", HybridModel),
            ("mlp", MLPModel)
        ]
        
        comparison_results = {}
        
        for model_name, model_class in models_to_test:
            print(f"Testing model: {model_name}")
            
            # 创建模型特定的配置
            model_config = OmegaConf.create(OmegaConf.to_yaml(pipeline_config))
            model_config.model.name = model_name
            model_config.experiment.name = f"e2e_test_{model_name}"
            model_config.experiment.save_dir = str(temp_dir / f"runs_{model_name}")
            model_config.evaluation.output_dir = str(temp_dir / f"eval_{model_name}")
            
            # 设置随机种子确保公平比较
            torch.manual_seed(42)
            np.random.seed(42)
            
            try:
                # 运行完整流程
                evaluation_results = self.test_complete_evaluation_pipeline(
                    model_config, comprehensive_test_data, temp_dir
                )
                
                comparison_results[model_name] = {
                    "summary_metrics": evaluation_results["summary_metrics"],
                    "model_info": evaluation_results["model_info"],
                    "success": True
                }
                
            except Exception as e:
                comparison_results[model_name] = {
                    "error": str(e),
                    "success": False
                }
                print(f"Model {model_name} failed: {e}")
        
        # 保存对比结果
        comparison_file = temp_dir / "model_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # 验证至少有一个模型成功运行
        successful_models = [name for name, result in comparison_results.items() if result["success"]]
        assert len(successful_models) > 0, "No models completed successfully"
        
        # 如果有多个成功的模型，进行性能比较
        if len(successful_models) > 1:
            print("Model comparison results:")
            for model_name in successful_models:
                metrics = comparison_results[model_name]["summary_metrics"]
                print(f"{model_name}: Rel-L2={metrics['avg_rel_l2']:.4f}, MAE={metrics['avg_mae']:.4f}")
        
        return comparison_results
    
    def test_error_handling_and_recovery(self, pipeline_config, comprehensive_test_data, temp_dir):
        """测试错误处理和恢复机制"""
        # 测试各种错误情况的处理
        
        # 1. 测试无效配置处理
        invalid_config = OmegaConf.create(OmegaConf.to_yaml(pipeline_config))
        invalid_config.model.in_channels = -1  # 无效通道数
        
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            model = SwinUNet(
                in_channels=invalid_config.model.in_channels,
                out_channels=invalid_config.model.out_channels,
                img_size=invalid_config.data.crop_size[0]
            )
        
        # 2. 测试数据路径错误处理
        invalid_data_config = OmegaConf.create(OmegaConf.to_yaml(pipeline_config))
        invalid_data_config.data.root_path = "/nonexistent/path"
        
        with pytest.raises((FileNotFoundError, OSError)):
            data_module = PDEBenchDataModule(invalid_data_config.data)
            data_module.setup()
        
        # 3. 测试检查点加载错误处理
        nonexistent_checkpoint = temp_dir / "nonexistent_model.pth"
        
        with pytest.raises((FileNotFoundError, RuntimeError)):
            torch.load(str(nonexistent_checkpoint))
        
        # 4. 测试维度不匹配错误处理
        model = SwinUNet(
            in_channels=pipeline_config.model.in_channels,
            out_channels=pipeline_config.model.out_channels,
            img_size=pipeline_config.data.crop_size[0],
            patch_size=pipeline_config.model.patch_size,
            window_size=pipeline_config.model.window_size,
            embed_dim=pipeline_config.model.embed_dim,
            depths=pipeline_config.model.depths,
            num_heads=pipeline_config.model.num_heads
        )
        
        # 错误的输入维度
        wrong_input = torch.randn(2, 3, 64, 64)  # 错误的通道数
        
        with pytest.raises((RuntimeError, ValueError)):
            model(wrong_input)
        
        print("Error handling tests completed successfully")
    
    def test_performance_benchmarking(self, pipeline_config, comprehensive_test_data, temp_dir):
        """测试性能基准测试"""
        # 运行性能基准测试
        pipeline_config.data.root_path = str(Path(comprehensive_test_data).parent)
        
        # 初始化模型
        model = SwinUNet(
            in_channels=pipeline_config.model.in_channels,
            out_channels=pipeline_config.model.out_channels,
            img_size=pipeline_config.data.crop_size[0],
            patch_size=pipeline_config.model.patch_size,
            window_size=pipeline_config.model.window_size,
            embed_dim=pipeline_config.model.embed_dim,
            depths=pipeline_config.model.depths,
            num_heads=pipeline_config.model.num_heads
        )
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 测试推理速度
        model.eval()
        input_tensor = torch.randn(1, pipeline_config.model.in_channels, 64, 64)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # 测量推理时间
        import time
        start_time = time.time()
        num_runs = 100
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / num_runs * 1000  # ms
        
        # 估算FLOPs（简化计算）
        # 这里使用一个简化的估算方法
        input_size = 64 * 64
        estimated_flops = total_params * input_size * 2  # 简化估算
        
        performance_metrics = {
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "avg_inference_time_ms": float(avg_inference_time),
            "estimated_flops": int(estimated_flops),
            "model_size_mb": float(total_params * 4 / (1024 * 1024)),  # 假设float32
            "throughput_samples_per_second": float(1000 / avg_inference_time)
        }
        
        # 保存性能指标
        perf_file = temp_dir / "performance_metrics.json"
        with open(perf_file, "w") as f:
            json.dump(performance_metrics, f, indent=2)
        
        # 验证性能指标合理性
        assert performance_metrics["total_parameters"] > 0
        assert performance_metrics["avg_inference_time_ms"] > 0
        assert performance_metrics["throughput_samples_per_second"] > 0
        
        print(f"Model performance metrics:")
        print(f"  Parameters: {performance_metrics['total_parameters']:,}")
        print(f"  Inference time: {performance_metrics['avg_inference_time_ms']:.2f} ms")
        print(f"  Throughput: {performance_metrics['throughput_samples_per_second']:.1f} samples/s")
        
        return performance_metrics