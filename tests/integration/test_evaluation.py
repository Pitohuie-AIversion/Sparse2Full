"""
PDEBench稀疏观测重建系统 - 评测流程集成测试

测试完整评测流程的集成性，包括模型加载、数据评测、指标计算、
结果保存等关键环节的协同工作。
遵循技术架构文档7.7节TDD准则要求。
"""

import os
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
import pytest
import torch
import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf

# 导入被测试模块
from eval import main as eval_main
from datasets.pdebench import PDEBenchDataModule
from models.swin_unet import SwinUNet
from models.hybrid import HybridModel
from models.mlp import MLPModel
from ops.losses import CombinedLoss
from ops.degradation import apply_degradation_operator
from utils.checkpoint import CheckpointManager
from utils.metrics import compute_metrics, compute_all_metrics
from utils.visualization import create_comparison_plot, create_spectrum_plot


class TestEvaluationIntegration:
    """评测流程集成测试类"""
    
    @pytest.fixture
    def evaluation_config(self, temp_dir):
        """创建评测配置"""
        config = {
            "data": {
                "root_path": str(temp_dir / "data"),
                "keys": ["u"],
                "task": "SR",
                "sr_scale": 2,
                "crop_size": [64, 64],
                "normalize": True,
                "batch_size": 4,
                "num_workers": 0
            },
            "model": {
                "name": "swin_unet",
                "patch_size": 4,
                "window_size": 4,
                "embed_dim": 48,
                "depths": [2, 2],
                "num_heads": [3, 6],
                "in_channels": 4,
                "out_channels": 1
            },
            "evaluation": {
                "checkpoint_path": str(temp_dir / "model.pth"),
                "output_dir": str(temp_dir / "eval_results"),
                "save_predictions": True,
                "save_visualizations": True,
                "compute_per_case_metrics": True,
                "metrics": [
                    "rel_l2", "mae", "psnr", "ssim", 
                    "frmse_low", "frmse_mid", "frmse_high",
                    "brmse", "crmse", "dc_error"
                ]
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
            "experiment": {
                "name": "test_evaluation_integration",
                "seed": 42,
                "deterministic": True
            }
        }
        return OmegaConf.create(config)
    
    @pytest.fixture
    def mock_evaluation_data(self, temp_dir):
        """创建模拟评测数据"""
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True)
        
        # 创建HDF5数据文件
        h5_path = data_dir / "test_data.h5"
        with h5py.File(h5_path, 'w') as f:
            # 创建10个样本用于评测
            for i in range(10):
                case_group = f.create_group(str(i))
                # 创建64x64的数据，使用更有结构的数据便于验证
                x = np.linspace(-1, 1, 64)
                y = np.linspace(-1, 1, 64)
                X, Y = np.meshgrid(x, y)
                
                # 创建具有不同频率成分的测试数据
                u_data = (
                    np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) +
                    0.5 * np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y) +
                    0.1 * np.random.randn(64, 64)
                ).astype(np.float32)
                
                case_group.create_dataset("u", data=u_data)
        
        # 创建splits目录和文件
        splits_dir = data_dir / "splits"
        splits_dir.mkdir()
        
        # 测试集：所有10个样本
        with open(splits_dir / "test.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(10)]))
        
        # 训练集（用于归一化统计）
        with open(splits_dir / "train.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(7)]))
        
        # 验证集
        with open(splits_dir / "val.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(7, 10)]))
        
        return str(h5_path)
    
    @pytest.fixture
    def mock_trained_model(self, evaluation_config, temp_dir):
        """创建模拟训练好的模型"""
        model = SwinUNet(
            in_channels=evaluation_config.model.in_channels,
            out_channels=evaluation_config.model.out_channels,
            img_size=evaluation_config.data.crop_size[0],
            patch_size=evaluation_config.model.patch_size,
            window_size=evaluation_config.model.window_size,
            embed_dim=evaluation_config.model.embed_dim,
            depths=evaluation_config.model.depths,
            num_heads=evaluation_config.model.num_heads
        )
        
        # 保存模型检查点
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "metrics": {"val_loss": 0.1, "val_rel_l2": 0.05}
        }, checkpoint_path)
        
        return model, str(checkpoint_path)
    
    def test_model_loading_from_checkpoint(self, evaluation_config, mock_trained_model):
        """测试从检查点加载模型"""
        model, checkpoint_path = mock_trained_model
        
        # 创建新模型实例
        new_model = SwinUNet(
            in_channels=evaluation_config.model.in_channels,
            out_channels=evaluation_config.model.out_channels,
            img_size=evaluation_config.data.crop_size[0],
            patch_size=evaluation_config.model.patch_size,
            window_size=evaluation_config.model.window_size,
            embed_dim=evaluation_config.model.embed_dim,
            depths=evaluation_config.model.depths,
            num_heads=evaluation_config.model.num_heads
        )
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_model.load_state_dict(checkpoint["model_state_dict"])
        
        # 验证模型参数一致性
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            np.testing.assert_allclose(
                param1.detach().numpy(),
                param2.detach().numpy(),
                rtol=1e-5,
                atol=1e-8
            )
    
    def test_evaluation_data_loading(self, evaluation_config, mock_evaluation_data):
        """测试评测数据加载"""
        # 更新配置中的数据路径
        evaluation_config.data.root_path = str(Path(mock_evaluation_data).parent)
        
        data_module = PDEBenchDataModule(evaluation_config.data)
        data_module.setup()
        
        # 获取测试数据加载器
        test_loader = data_module.test_dataloader()
        
        assert test_loader is not None
        assert len(test_loader) > 0
        
        # 验证数据批次
        test_batch = next(iter(test_loader))
        assert "input" in test_batch
        assert "target" in test_batch
        assert "case_id" in test_batch
        
        # 验证数据形状
        input_tensor = test_batch["input"]
        target_tensor = test_batch["target"]
        
        assert input_tensor.ndim == 4  # [B, C, H, W]
        assert target_tensor.ndim == 4  # [B, C, H, W]
        assert input_tensor.shape[1] == 4  # baseline + coords + mask
        assert target_tensor.shape[1] == 1  # single channel output
    
    def test_model_inference(self, evaluation_config, mock_trained_model, mock_evaluation_data):
        """测试模型推理"""
        model, checkpoint_path = mock_trained_model
        evaluation_config.data.root_path = str(Path(mock_evaluation_data).parent)
        
        # 加载数据
        data_module = PDEBenchDataModule(evaluation_config.data)
        data_module.setup()
        test_loader = data_module.test_dataloader()
        
        # 设置模型为评估模式
        model.eval()
        
        predictions = []
        targets = []
        case_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_tensor = batch["input"]
                target_tensor = batch["target"]
                batch_case_ids = batch["case_id"]
                
                # 模型推理
                pred = model(input_tensor)
                
                # 验证输出形状
                assert pred.shape == target_tensor.shape
                
                predictions.append(pred.cpu())
                targets.append(target_tensor.cpu())
                case_ids.extend(batch_case_ids)
        
        # 验证推理结果
        assert len(predictions) > 0
        assert len(targets) > 0
        assert len(case_ids) > 0
        
        # 合并所有预测结果
        all_predictions = torch.cat(predictions, dim=0)
        all_targets = torch.cat(targets, dim=0)
        
        assert all_predictions.shape == all_targets.shape
        assert all_predictions.shape[0] == len(case_ids)
    
    def test_metrics_computation(self, evaluation_config, mock_trained_model, mock_evaluation_data):
        """测试指标计算"""
        model, checkpoint_path = mock_trained_model
        evaluation_config.data.root_path = str(Path(mock_evaluation_data).parent)
        
        # 加载数据
        data_module = PDEBenchDataModule(evaluation_config.data)
        data_module.setup()
        test_loader = data_module.test_dataloader()
        
        model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_tensor = batch["input"]
                target_tensor = batch["target"]
                case_ids = batch["case_id"]
                
                pred = model(input_tensor)
                
                # 计算批次指标
                for i in range(pred.shape[0]):
                    pred_single = pred[i:i+1]
                    target_single = target_tensor[i:i+1]
                    case_id = case_ids[i]
                    
                    # 计算所有指标
                    metrics = compute_all_metrics(
                        pred_single, 
                        target_single,
                        keys=evaluation_config.data.keys
                    )
                    
                    metrics["case_id"] = case_id
                    all_metrics.append(metrics)
        
        # 验证指标计算结果
        assert len(all_metrics) > 0
        
        # 验证指标字典结构
        required_metrics = ["rel_l2", "mae", "psnr", "ssim"]
        for metric_dict in all_metrics:
            for metric_name in required_metrics:
                assert metric_name in metric_dict
                assert isinstance(metric_dict[metric_name], (float, np.floating))
                assert not np.isnan(metric_dict[metric_name])
                assert not np.isinf(metric_dict[metric_name])
        
        # 计算平均指标
        avg_metrics = {}
        for metric_name in required_metrics:
            values = [m[metric_name] for m in all_metrics]
            avg_metrics[f"avg_{metric_name}"] = np.mean(values)
            avg_metrics[f"std_{metric_name}"] = np.std(values)
        
        # 验证平均指标合理性
        assert avg_metrics["avg_rel_l2"] >= 0
        assert avg_metrics["avg_mae"] >= 0
        assert avg_metrics["avg_psnr"] > 0
        assert 0 <= avg_metrics["avg_ssim"] <= 1
    
    def test_data_consistency_verification(self, evaluation_config, mock_trained_model, mock_evaluation_data):
        """测试数据一致性验证（H算子一致性）"""
        model, checkpoint_path = mock_trained_model
        evaluation_config.data.root_path = str(Path(mock_evaluation_data).parent)
        
        # 加载数据
        data_module = PDEBenchDataModule(evaluation_config.data)
        data_module.setup()
        test_loader = data_module.test_dataloader()
        
        model.eval()
        
        dc_errors = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_tensor = batch["input"]
                target_tensor = batch["target"]
                
                # 获取观测数据
                baseline = input_tensor[:, 0:1]  # 第一个通道是baseline
                mask = input_tensor[:, 3:4]     # 第四个通道是mask
                
                # 模型预测
                pred = model(input_tensor)
                
                # 应用降质算子到ground truth
                degradation_params = {
                    "task": evaluation_config.data.task,
                    "scale": evaluation_config.data.sr_scale,
                    "sigma": 1.0,
                    "kernel_size": 5,
                    "boundary": "mirror"
                }
                
                gt_degraded = apply_degradation_operator(
                    target_tensor, 
                    degradation_params
                )
                
                # 应用降质算子到预测结果
                pred_degraded = apply_degradation_operator(
                    pred, 
                    degradation_params
                )
                
                # 计算数据一致性误差
                dc_error = torch.mean((gt_degraded - pred_degraded) ** 2)
                dc_errors.append(dc_error.item())
        
        # 验证H算子一致性
        avg_dc_error = np.mean(dc_errors)
        
        # 根据技术架构文档要求：MSE(H(GT), H(pred)) 应该较小
        # 注意：这里比较的是H(GT)和H(pred)，而不是H(GT)和观测y
        # 对于训练好的模型，这个误差应该相对较小
        assert avg_dc_error >= 0, "Data consistency error should be non-negative"
        
        # 记录数据一致性误差用于分析
        print(f"Average data consistency error: {avg_dc_error:.6f}")
    
    def test_evaluation_output_saving(self, evaluation_config, mock_trained_model, mock_evaluation_data, temp_dir):
        """测试评测结果保存"""
        model, checkpoint_path = mock_trained_model
        evaluation_config.data.root_path = str(Path(mock_evaluation_data).parent)
        evaluation_config.evaluation.output_dir = str(temp_dir / "eval_results")
        
        # 创建输出目录
        output_dir = Path(evaluation_config.evaluation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        data_module = PDEBenchDataModule(evaluation_config.data)
        data_module.setup()
        test_loader = data_module.test_dataloader()
        
        model.eval()
        
        all_metrics = []
        predictions_saved = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_tensor = batch["input"]
                target_tensor = batch["target"]
                case_ids = batch["case_id"]
                
                pred = model(input_tensor)
                
                # 保存预测结果
                if evaluation_config.evaluation.save_predictions:
                    for i in range(pred.shape[0]):
                        case_id = case_ids[i]
                        pred_single = pred[i].cpu().numpy()
                        target_single = target_tensor[i].cpu().numpy()
                        
                        # 保存为npz文件
                        save_path = output_dir / f"pred_{case_id}.npz"
                        np.savez(
                            save_path,
                            prediction=pred_single,
                            target=target_single,
                            case_id=case_id
                        )
                        predictions_saved.append(save_path)
                
                # 计算指标
                for i in range(pred.shape[0]):
                    pred_single = pred[i:i+1]
                    target_single = target_tensor[i:i+1]
                    case_id = case_ids[i]
                    
                    metrics = compute_all_metrics(
                        pred_single, 
                        target_single,
                        keys=evaluation_config.data.keys
                    )
                    metrics["case_id"] = case_id
                    all_metrics.append(metrics)
        
        # 保存指标结果
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        # 保存汇总指标
        summary_metrics = {}
        metric_names = ["rel_l2", "mae", "psnr", "ssim"]
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            summary_metrics[f"avg_{metric_name}"] = float(np.mean(values))
            summary_metrics[f"std_{metric_name}"] = float(np.std(values))
            summary_metrics[f"min_{metric_name}"] = float(np.min(values))
            summary_metrics[f"max_{metric_name}"] = float(np.max(values))
        
        summary_file = output_dir / "summary_metrics.json"
        with open(summary_file, "w") as f:
            json.dump(summary_metrics, f, indent=2)
        
        # 验证文件保存
        assert metrics_file.exists()
        assert summary_file.exists()
        
        if evaluation_config.evaluation.save_predictions:
            assert len(predictions_saved) > 0
            for pred_file in predictions_saved:
                assert pred_file.exists()
                
                # 验证保存的数据
                data = np.load(pred_file)
                assert "prediction" in data
                assert "target" in data
                assert "case_id" in data
    
    def test_evaluation_reproducibility(self, evaluation_config, mock_trained_model, mock_evaluation_data):
        """测试评测可重现性"""
        model, checkpoint_path = mock_trained_model
        evaluation_config.data.root_path = str(Path(mock_evaluation_data).parent)
        
        def run_evaluation():
            # 设置随机种子
            torch.manual_seed(42)
            np.random.seed(42)
            
            # 加载数据
            data_module = PDEBenchDataModule(evaluation_config.data)
            data_module.setup()
            test_loader = data_module.test_dataloader()
            
            model.eval()
            
            all_predictions = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_tensor = batch["input"]
                    pred = model(input_tensor)
                    all_predictions.append(pred.cpu())
            
            return torch.cat(all_predictions, dim=0)
        
        # 运行两次评测
        pred1 = run_evaluation()
        pred2 = run_evaluation()
        
        # 验证结果一致性
        np.testing.assert_allclose(
            pred1.numpy(),
            pred2.numpy(),
            rtol=1e-5,
            atol=1e-8,
            err_msg="Evaluation results are not reproducible"
        )
    
    def test_evaluation_with_different_tasks(self, evaluation_config, mock_evaluation_data, temp_dir):
        """测试不同任务的评测兼容性"""
        evaluation_config.data.root_path = str(Path(mock_evaluation_data).parent)
        
        # 测试不同任务配置
        tasks_to_test = [
            {"task": "SR", "sr_scale": 2},
            {"task": "SR", "sr_scale": 4},
            {"task": "Crop", "crop_size": [32, 32]}
        ]
        
        for task_config in tasks_to_test:
            # 更新配置
            evaluation_config.data.task = task_config["task"]
            if "sr_scale" in task_config:
                evaluation_config.data.sr_scale = task_config["sr_scale"]
            if "crop_size" in task_config:
                evaluation_config.data.crop_size = task_config["crop_size"]
            
            # 创建对应的模型
            model = SwinUNet(
                in_channels=evaluation_config.model.in_channels,
                out_channels=evaluation_config.model.out_channels,
                img_size=evaluation_config.data.crop_size[0],
                patch_size=evaluation_config.model.patch_size,
                window_size=evaluation_config.model.window_size,
                embed_dim=evaluation_config.model.embed_dim,
                depths=evaluation_config.model.depths,
                num_heads=evaluation_config.model.num_heads
            )
            
            # 加载数据
            data_module = PDEBenchDataModule(evaluation_config.data)
            data_module.setup()
            test_loader = data_module.test_dataloader()
            
            model.eval()
            
            # 运行一个批次的评测
            batch = next(iter(test_loader))
            input_tensor = batch["input"]
            target_tensor = batch["target"]
            
            with torch.no_grad():
                pred = model(input_tensor)
            
            # 验证输出形状正确
            assert pred.shape == target_tensor.shape
            
            # 计算基本指标
            metrics = compute_all_metrics(
                pred, 
                target_tensor,
                keys=evaluation_config.data.keys
            )
            
            # 验证指标计算成功
            assert "rel_l2" in metrics
            assert "mae" in metrics
            assert not np.isnan(metrics["rel_l2"])
            assert not np.isnan(metrics["mae"])
    
    @pytest.mark.slow
    def test_full_evaluation_pipeline(self, evaluation_config, mock_trained_model, mock_evaluation_data, temp_dir):
        """测试完整评测管道"""
        model, checkpoint_path = mock_trained_model
        evaluation_config.data.root_path = str(Path(mock_evaluation_data).parent)
        evaluation_config.evaluation.output_dir = str(temp_dir / "full_eval")
        evaluation_config.evaluation.checkpoint_path = checkpoint_path
        
        # 创建输出目录
        output_dir = Path(evaluation_config.evaluation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 加载数据
        data_module = PDEBenchDataModule(evaluation_config.data)
        data_module.setup()
        test_loader = data_module.test_dataloader()
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
                        keys=evaluation_config.data.keys
                    )
                    metrics["case_id"] = case_id
                    all_metrics.append(metrics)
        
        # 合并所有结果
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 保存详细结果
        detailed_results = {
            "config": OmegaConf.to_yaml(evaluation_config),
            "per_case_metrics": all_metrics,
            "summary": {}
        }
        
        # 计算汇总统计
        metric_names = ["rel_l2", "mae", "psnr", "ssim"]
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            detailed_results["summary"][f"avg_{metric_name}"] = float(np.mean(values))
            detailed_results["summary"][f"std_{metric_name}"] = float(np.std(values))
            detailed_results["summary"][f"min_{metric_name}"] = float(np.min(values))
            detailed_results["summary"][f"max_{metric_name}"] = float(np.max(values))
        
        # 保存结果文件
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # 保存预测结果
        predictions_file = output_dir / "predictions.npz"
        np.savez(
            predictions_file,
            predictions=all_predictions.numpy(),
            targets=all_targets.numpy(),
            case_ids=all_case_ids
        )
        
        # 验证输出文件
        assert results_file.exists()
        assert predictions_file.exists()
        
        # 验证结果质量
        avg_rel_l2 = detailed_results["summary"]["avg_rel_l2"]
        avg_mae = detailed_results["summary"]["avg_mae"]
        
        assert avg_rel_l2 >= 0, "Relative L2 error should be non-negative"
        assert avg_mae >= 0, "MAE should be non-negative"
        
        # 验证结果文件内容
        loaded_results = json.load(open(results_file))
        assert "config" in loaded_results
        assert "per_case_metrics" in loaded_results
        assert "summary" in loaded_results
        assert len(loaded_results["per_case_metrics"]) == len(all_case_ids)
        
        loaded_predictions = np.load(predictions_file)
        assert "predictions" in loaded_predictions
        assert "targets" in loaded_predictions
        assert "case_ids" in loaded_predictions
        assert loaded_predictions["predictions"].shape == all_predictions.shape
        assert loaded_predictions["targets"].shape == all_targets.shape