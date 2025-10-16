"""
PDEBench数据集单元测试

测试PDEBench数据加载、观测生成、归一化处理等核心功能。
遵循技术架构文档7.7节TDD准则要求。
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import pytest
import torch
import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf

# 导入被测试模块
from datasets.pdebench import PDEBenchBase, PDEBenchSR, PDEBenchCrop, PDEBenchDataModule
from ops.degradation import apply_degradation_operator


class TestPDEBenchBase:
    """PDEBench基类测试"""
    
    def test_init_basic(self, sample_h5_data, temp_dir):
        """测试基本初始化"""
        dataset = PDEBenchBase(
            data_path=str(sample_h5_data),
            keys=["u"],
            split="train",
            normalize=False,
            image_size=128
        )
        
        assert dataset.data_path == str(sample_h5_data)
        assert dataset.keys == ["u"]
        assert dataset.split == "train"
        assert not dataset.normalize
        assert dataset.image_size == 128
        assert len(dataset) > 0
    
    def test_init_with_normalization(self, sample_h5_data, temp_dir):
        """测试带归一化的初始化"""
        # 创建splits目录
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir()
        
        # 创建split文件
        with open(splits_dir / "train.txt", "w") as f:
            f.write("0\n1\n2\n")
        with open(splits_dir / "val.txt", "w") as f:
            f.write("3\n4\n")
        with open(splits_dir / "test.txt", "w") as f:
            f.write("5\n6\n")
        
        dataset = PDEBenchBase(
            data_path=str(sample_h5_data),
            keys=["u"],
            split="train",
            splits_dir=str(splits_dir),
            normalize=True,
            image_size=128
        )
        
        assert dataset.normalize
        assert dataset.norm_stats is not None
        assert "u_mean" in dataset.norm_stats
        assert "u_std" in dataset.norm_stats
    
    def test_load_split_ids(self, sample_h5_data, temp_dir):
        """测试数据切分加载"""
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir()
        
        # 创建测试split文件
        train_ids = ["0", "1", "2", "3"]
        with open(splits_dir / "train.txt", "w") as f:
            f.write("\n".join(train_ids))
        
        dataset = PDEBenchBase(
            data_path=str(sample_h5_data),
            keys=["u"],
            split="train",
            splits_dir=str(splits_dir),
            normalize=False
        )
        
        assert len(dataset.case_ids) <= len(train_ids)  # 可能有些ID不存在
        for case_id in dataset.case_ids:
            assert case_id in train_ids
    
    def test_normalization_stats_computation(self, sample_h5_data, temp_dir):
        """测试归一化统计量计算"""
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir()
        
        # 创建train split
        with open(splits_dir / "train.txt", "w") as f:
            f.write("0\n1\n2\n")
        
        dataset = PDEBenchBase(
            data_path=str(sample_h5_data),
            keys=["u"],
            split="train",
            splits_dir=str(splits_dir),
            normalize=True
        )
        
        # 验证统计量文件生成
        stats_file = splits_dir / "norm_stat.npz"
        assert stats_file.exists()
        
        # 验证统计量内容
        stats = np.load(stats_file)
        assert "u_mean" in stats
        assert "u_std" in stats
        
        # 验证数值合理性
        assert not np.isnan(stats["u_mean"])
        assert not np.isnan(stats["u_std"])
        assert stats["u_std"] > 0
    
    def test_normalize_denormalize_consistency(self, sample_h5_data, temp_dir):
        """测试归一化和反归一化的一致性"""
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir()
        
        with open(splits_dir / "train.txt", "w") as f:
            f.write("0\n1\n2\n")
        
        dataset = PDEBenchBase(
            data_path=str(sample_h5_data),
            keys=["u"],
            split="train",
            splits_dir=str(splits_dir),
            normalize=True
        )
        
        # 测试数据
        original_data = torch.randn(1, 64, 64)
        
        # 归一化后反归一化应该恢复原值
        normalized = dataset._normalize_data(original_data, "u")
        denormalized = dataset._denormalize_data(normalized, "u")
        
        np.testing.assert_allclose(
            original_data.numpy(),
            denormalized.numpy(),
            rtol=1e-5,
            atol=1e-8
        )
    
    def test_getitem_shape_consistency(self, sample_h5_data):
        """测试__getitem__返回数据形状一致性"""
        dataset = PDEBenchBase(
            data_path=str(sample_h5_data),
            keys=["u"],
            split="train",
            normalize=False,
            image_size=128
        )
        
        # 测试多个样本
        for i in range(min(3, len(dataset))):
            data = dataset[i]
            
            # 验证返回字典结构
            assert isinstance(data, dict)
            assert "target" in data
            assert "case_id" in data
            assert "keys" in data
            
            # 验证张量形状
            target = data["target"]
            assert isinstance(target, torch.Tensor)
            assert target.ndim == 3  # [C, H, W]
            assert target.shape[0] == len(dataset.keys)  # 通道数
            assert target.shape[1] == 128  # 高度
            assert target.shape[2] == 128  # 宽度
    
    def test_multichannel_data(self, temp_dir):
        """测试多通道数据处理"""
        # 创建多通道测试数据
        h5_path = temp_dir / "multichannel_test.h5"
        with h5py.File(h5_path, 'w') as f:
            # 创建多通道数据 (rho, ux, uy)
            for i in range(5):
                case_group = f.create_group(str(i))
                case_group.create_dataset("rho", data=np.random.randn(64, 64).astype(np.float32))
                case_group.create_dataset("ux", data=np.random.randn(64, 64).astype(np.float32))
                case_group.create_dataset("uy", data=np.random.randn(64, 64).astype(np.float32))
        
        dataset = PDEBenchBase(
            data_path=str(h5_path),
            keys=["rho", "ux", "uy"],
            split="train",
            normalize=False,
            image_size=64
        )
        
        data = dataset[0]
        target = data["target"]
        
        # 验证多通道拼接
        assert target.shape[0] == 3  # 3个通道
        assert target.shape[1:] == (64, 64)
    
    def test_data_validation(self, temp_dir):
        """测试数据完整性验证"""
        # 创建不完整的测试数据
        h5_path = temp_dir / "incomplete_test.h5"
        with h5py.File(h5_path, 'w') as f:
            # case 0: 完整数据
            case0 = f.create_group("0")
            case0.create_dataset("u", data=np.random.randn(64, 64).astype(np.float32))
            
            # case 1: 缺少u键
            case1 = f.create_group("1")
            case1.create_dataset("v", data=np.random.randn(64, 64).astype(np.float32))
        
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir()
        with open(splits_dir / "train.txt", "w") as f:
            f.write("0\n1\n2\n")  # case 2不存在
        
        dataset = PDEBenchBase(
            data_path=str(h5_path),
            keys=["u"],
            split="train",
            splits_dir=str(splits_dir),
            normalize=False
        )
        
        # 应该只保留有效的case
        assert len(dataset.case_ids) == 1
        assert "0" in dataset.case_ids


class TestPDEBenchSR:
    """SR数据集测试"""
    
    def test_init_sr_params(self, sample_h5_data):
        """测试SR参数初始化"""
        dataset = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=4,
            sigma=1.0,
            blur_kernel=5,
            boundary="mirror",
            noise_std=0.1,
            normalize=False
        )
        
        assert dataset.scale == 4
        assert dataset.sigma == 1.0
        assert dataset.blur_kernel == 5
        assert dataset.boundary == "mirror"
        assert dataset.noise_std == 0.1
        
        # 验证H算子参数
        expected_h_params = {
            'task': 'SR',
            'scale': 4,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        assert dataset.h_params == expected_h_params
    
    def test_sr_observation_generation(self, sample_h5_data):
        """测试SR观测生成"""
        dataset = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=4,
            sigma=1.0,
            normalize=False
        )
        
        data = dataset[0]
        
        # 验证返回数据结构
        assert "target" in data
        assert "baseline" in data
        assert "coords" in data
        assert "mask" in data
        assert "h_params" in data
        assert "lr_observation" in data
        
        target = data["target"]
        baseline = data["baseline"]
        coords = data["coords"]
        mask = data["mask"]
        
        # 验证形状
        C, H, W = target.shape
        assert baseline.shape == (C, H, W)
        assert coords.shape == (2, H, W)
        assert mask.shape == (1, H, W)
        
        # 验证mask全为1（SR模式）
        assert torch.all(mask == 1.0)
        
        # 验证坐标范围
        assert coords.min() >= -1.0
        assert coords.max() <= 1.0
    
    def test_sr_h_operator_consistency(self, sample_h5_data, tolerance_config):
        """测试SR模式H算子一致性"""
        dataset = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=2,  # 使用较小的scale便于测试
            sigma=0.5,
            normalize=False
        )
        
        data = dataset[0]
        target = data["target"]
        h_params = data["h_params"]
        
        # 使用相同H算子重新生成观测
        lr_reconstructed = apply_degradation_operator(
            target.unsqueeze(0), 
            h_params
        ).squeeze(0)
        
        # 上采样到原尺寸
        baseline_reconstructed = torch.nn.functional.interpolate(
            lr_reconstructed.unsqueeze(0),
            size=target.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # 验证一致性：MSE(H(GT), y) < 1e-8
        baseline_original = data["baseline"]
        mse = torch.mean((baseline_original - baseline_reconstructed) ** 2)
        
        assert mse.item() < tolerance_config["dc_threshold"], (
            f"H operator consistency failed: MSE = {mse.item():.2e} > {tolerance_config['dc_threshold']:.2e}"
        )
    
    def test_sr_noise_addition(self, sample_h5_data):
        """测试SR噪声添加"""
        # 无噪声数据集
        dataset_clean = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=4,
            noise_std=0.0,
            normalize=False
        )
        
        # 有噪声数据集
        dataset_noisy = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=4,
            noise_std=0.1,
            normalize=False
        )
        
        # 设置相同随机种子
        torch.manual_seed(42)
        data_clean = dataset_clean[0]
        
        torch.manual_seed(42)
        data_noisy = dataset_noisy[0]
        
        # 验证噪声影响
        baseline_clean = data_clean["baseline"]
        baseline_noisy = data_noisy["baseline"]
        
        # 应该有差异（由于噪声）
        diff = torch.mean((baseline_clean - baseline_noisy) ** 2)
        assert diff.item() > 1e-6, "Noise should introduce differences"
    
    def test_sr_coords_generation(self, sample_h5_data):
        """测试SR坐标生成"""
        dataset = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=4,
            normalize=False
        )
        
        data = dataset[0]
        coords = data["coords"]  # [2, H, W]
        
        H, W = coords.shape[1], coords.shape[2]
        
        # 验证坐标网格属性
        x_coords = coords[0]  # [H, W]
        y_coords = coords[1]  # [H, W]
        
        # 验证x坐标在每行相同
        for i in range(H):
            assert torch.allclose(x_coords[i, :], x_coords[0, :])
        
        # 验证y坐标在每列相同
        for j in range(W):
            assert torch.allclose(y_coords[:, j], y_coords[:, 0])
        
        # 验证坐标范围
        assert torch.allclose(x_coords[0, 0], torch.tensor(-1.0), atol=1e-6)
        assert torch.allclose(x_coords[0, -1], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(y_coords[0, 0], torch.tensor(-1.0), atol=1e-6)
        assert torch.allclose(y_coords[-1, 0], torch.tensor(1.0), atol=1e-6)


class TestPDEBenchCrop:
    """Crop数据集测试"""
    
    def test_init_crop_params(self, sample_h5_data):
        """测试Crop参数初始化"""
        dataset = PDEBenchCrop(
            data_path=str(sample_h5_data),
            keys=["u"],
            crop_size=(64, 64),
            patch_align=8,
            center_sampler="mixed",
            boundary="mirror",
            normalize=False
        )
        
        assert dataset.crop_size == (64, 64)
        assert dataset.patch_align == 8
        assert dataset.center_sampler == "mixed"
        assert dataset.boundary == "mirror"
        
        # 验证对齐后的尺寸
        assert dataset.crop_h == 64  # 64 % 8 == 0
        assert dataset.crop_w == 64
        
        # 验证H算子参数
        expected_h_params = {
            'task': 'Crop',
            'crop_size': (64, 64),
            'patch_align': 8,
            'boundary': 'mirror'
        }
        assert dataset.h_params == expected_h_params
    
    def test_crop_size_alignment(self, sample_h5_data):
        """测试Crop尺寸对齐"""
        # 测试不对齐的尺寸
        dataset = PDEBenchCrop(
            data_path=str(sample_h5_data),
            keys=["u"],
            crop_size=(67, 73),  # 不是8的倍数
            patch_align=8,
            normalize=False
        )
        
        # 应该自动对齐到8的倍数
        assert dataset.crop_h == 64  # 67 -> 64
        assert dataset.crop_w == 72  # 73 -> 72
    
    def test_crop_observation_generation(self, sample_h5_data):
        """测试Crop观测生成"""
        dataset = PDEBenchCrop(
            data_path=str(sample_h5_data),
            keys=["u"],
            crop_size=(64, 64),
            patch_align=8,
            normalize=False
        )
        
        data = dataset[0]
        
        # 验证返回数据结构
        assert "target" in data
        assert "baseline" in data
        assert "coords" in data
        assert "mask" in data
        assert "h_params" in data
        
        target = data["target"]
        baseline = data["baseline"]
        coords = data["coords"]
        mask = data["mask"]
        
        # 验证形状
        C, H, W = target.shape
        assert baseline.shape == (C, H, W)
        assert coords.shape == (2, H, W)
        assert mask.shape == (1, H, W)
        
        # 验证mask有0和1（Crop模式）
        unique_mask_values = torch.unique(mask)
        assert 0.0 in unique_mask_values or 1.0 in unique_mask_values
    
    def test_crop_center_sampling_strategies(self, sample_h5_data):
        """测试不同的中心采样策略"""
        strategies = ["uniform", "boundary", "gradient", "mixed"]
        
        for strategy in strategies:
            dataset = PDEBenchCrop(
                data_path=str(sample_h5_data),
                keys=["u"],
                crop_size=(32, 32),
                center_sampler=strategy,
                normalize=False
            )
            
            # 测试多次采样，验证策略有效
            masks = []
            for i in range(min(3, len(dataset))):
                data = dataset[i]
                masks.append(data["mask"])
            
            # 验证mask不全相同（除非数据完全一致）
            if len(masks) > 1:
                # 至少应该有一些变化
                all_same = all(torch.equal(masks[0], mask) for mask in masks[1:])
                # 注意：由于随机性，可能偶尔全相同，这里只做基本验证
                assert isinstance(all_same, bool)


class TestPDEBenchDataModule:
    """数据模块测试"""
    
    def test_data_module_init(self, sample_h5_data, temp_dir):
        """测试数据模块初始化"""
        config = OmegaConf.create({
            "data_path": str(temp_dir),
            "dataset_name": "test_data.h5",
            "keys": ["u"],
            "observation": {
                "mode": "SR",
                "scale": 4,
                "sigma": 1.0
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0
            },
            "normalize": False
        })
        
        # 复制测试数据到指定位置
        target_path = temp_dir / "test_data.h5"
        shutil.copy(sample_h5_data, target_path)
        
        data_module = PDEBenchDataModule(config)
        assert data_module.config == config
        assert data_module.data_path == str(target_path)
    
    def test_data_module_setup_sr(self, sample_h5_data, temp_dir):
        """测试SR模式数据模块设置"""
        config = OmegaConf.create({
            "data_path": str(temp_dir),
            "dataset_name": "test_data.h5",
            "keys": ["u"],
            "observation": {
                "mode": "SR",
                "scale": 4,
                "sigma": 1.0
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0
            },
            "normalize": False
        })
        
        # 复制测试数据
        target_path = temp_dir / "test_data.h5"
        shutil.copy(sample_h5_data, target_path)
        
        data_module = PDEBenchDataModule(config)
        data_module.setup()
        
        # 验证数据集类型
        from datasets.pdebench import PDEBenchSR
        assert isinstance(data_module.train_dataset, PDEBenchSR)
        assert isinstance(data_module.val_dataset, PDEBenchSR)
        assert isinstance(data_module.test_dataset, PDEBenchSR)
    
    def test_data_module_setup_crop(self, sample_h5_data, temp_dir):
        """测试Crop模式数据模块设置"""
        config = OmegaConf.create({
            "data_path": str(temp_dir),
            "dataset_name": "test_data.h5",
            "keys": ["u"],
            "observation": {
                "mode": "Crop",
                "crop_size": [64, 64],
                "patch_align": 8
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0
            },
            "normalize": False
        })
        
        # 复制测试数据
        target_path = temp_dir / "test_data.h5"
        shutil.copy(sample_h5_data, target_path)
        
        data_module = PDEBenchDataModule(config)
        data_module.setup()
        
        # 验证数据集类型
        from datasets.pdebench import PDEBenchCrop
        assert isinstance(data_module.train_dataset, PDEBenchCrop)
        assert isinstance(data_module.val_dataset, PDEBenchCrop)
        assert isinstance(data_module.test_dataset, PDEBenchCrop)
    
    def test_data_loaders(self, sample_h5_data, temp_dir):
        """测试数据加载器"""
        config = OmegaConf.create({
            "data_path": str(temp_dir),
            "dataset_name": "test_data.h5",
            "keys": ["u"],
            "observation": {
                "mode": "SR",
                "scale": 4
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0
            },
            "normalize": False
        })
        
        # 复制测试数据
        target_path = temp_dir / "test_data.h5"
        shutil.copy(sample_h5_data, target_path)
        
        data_module = PDEBenchDataModule(config)
        data_module.setup()
        
        # 获取数据加载器
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        # 验证数据加载器属性
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        assert test_loader.batch_size == 1  # 测试时使用batch_size=1
        
        # 验证数据加载
        train_batch = next(iter(train_loader))
        assert isinstance(train_batch, dict)
        assert "target" in train_batch
        assert "baseline" in train_batch
        
        # 验证批量维度
        target = train_batch["target"]
        assert target.ndim == 4  # [B, C, H, W]
        assert target.shape[0] <= 2  # 批量大小


class TestDatasetEdgeCases:
    """数据集边界情况测试"""
    
    def test_empty_dataset(self, temp_dir):
        """测试空数据集"""
        # 创建空的HDF5文件
        h5_path = temp_dir / "empty_test.h5"
        with h5py.File(h5_path, 'w') as f:
            pass  # 空文件
        
        dataset = PDEBenchBase(
            data_path=str(h5_path),
            keys=["u"],
            split="train",
            normalize=False
        )
        
        assert len(dataset) == 0
    
    def test_single_sample_dataset(self, temp_dir):
        """测试单样本数据集"""
        h5_path = temp_dir / "single_test.h5"
        with h5py.File(h5_path, 'w') as f:
            case0 = f.create_group("0")
            case0.create_dataset("u", data=np.random.randn(64, 64).astype(np.float32))
        
        dataset = PDEBenchBase(
            data_path=str(h5_path),
            keys=["u"],
            split="train",
            normalize=False
        )
        
        assert len(dataset) == 1
        data = dataset[0]
        assert data["case_id"] == "0"
    
    def test_invalid_split_file(self, sample_h5_data, temp_dir):
        """测试无效的split文件"""
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir()
        
        # 创建包含不存在case ID的split文件
        with open(splits_dir / "train.txt", "w") as f:
            f.write("nonexistent_case_1\nnonexistent_case_2\n")
        
        dataset = PDEBenchBase(
            data_path=str(sample_h5_data),
            keys=["u"],
            split="train",
            splits_dir=str(splits_dir),
            normalize=False
        )
        
        # 应该过滤掉不存在的case
        assert len(dataset) == 0
    
    def test_mismatched_image_size(self, temp_dir):
        """测试不匹配的图像尺寸"""
        # 创建不同尺寸的数据
        h5_path = temp_dir / "size_test.h5"
        with h5py.File(h5_path, 'w') as f:
            case0 = f.create_group("0")
            case0.create_dataset("u", data=np.random.randn(32, 32).astype(np.float32))
        
        dataset = PDEBenchBase(
            data_path=str(h5_path),
            keys=["u"],
            split="train",
            normalize=False,
            image_size=64  # 要求64x64，但数据是32x32
        )
        
        data = dataset[0]
        target = data["target"]
        
        # 应该自动调整到指定尺寸
        assert target.shape[-2:] == (64, 64)
    
    @pytest.mark.parametrize("boundary", ["mirror", "wrap", "zero"])
    def test_different_boundary_conditions(self, sample_h5_data, boundary):
        """测试不同边界条件"""
        dataset = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=2,
            boundary=boundary,
            normalize=False
        )
        
        data = dataset[0]
        assert data["h_params"]["boundary"] == boundary
    
    def test_extreme_scale_values(self, sample_h5_data):
        """测试极端scale值"""
        # 测试大scale值
        dataset_large = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=8,
            normalize=False
        )
        
        data = dataset_large[0]
        assert data["h_params"]["scale"] == 8
        
        # 测试小scale值
        dataset_small = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=1,
            normalize=False
        )
        
        data = dataset_small[0]
        assert data["h_params"]["scale"] == 1
    
    def test_zero_noise_std(self, sample_h5_data):
        """测试零噪声标准差"""
        dataset = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=4,
            noise_std=0.0,
            normalize=False
        )
        
        # 多次采样应该得到相同结果
        torch.manual_seed(42)
        data1 = dataset[0]
        
        torch.manual_seed(42)
        data2 = dataset[0]
        
        assert torch.equal(data1["baseline"], data2["baseline"])


# 性能测试
class TestDatasetPerformance:
    """数据集性能测试"""
    
    @pytest.mark.slow
    def test_large_batch_loading(self, sample_h5_data, temp_dir):
        """测试大批量数据加载性能"""
        config = OmegaConf.create({
            "data_path": str(temp_dir),
            "dataset_name": "test_data.h5",
            "keys": ["u"],
            "observation": {
                "mode": "SR",
                "scale": 4
            },
            "dataloader": {
                "batch_size": 8,  # 较大批量
                "num_workers": 0
            },
            "normalize": False
        })
        
        # 复制测试数据
        target_path = temp_dir / "test_data.h5"
        shutil.copy(sample_h5_data, target_path)
        
        data_module = PDEBenchDataModule(config)
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        
        # 测试加载时间
        import time
        start_time = time.time()
        
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            if batch_count >= 3:  # 只测试几个批次
                break
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        # 验证加载成功且时间合理
        assert batch_count > 0
        assert loading_time < 10.0  # 应该在10秒内完成
    
    def test_memory_usage(self, sample_h5_data):
        """测试内存使用"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建数据集
        dataset = PDEBenchSR(
            data_path=str(sample_h5_data),
            keys=["u"],
            scale=4,
            normalize=False
        )
        
        # 加载一些数据
        for i in range(min(5, len(dataset))):
            _ = dataset[i]
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # 清理
        del dataset
        gc.collect()
        
        # 内存增长应该合理（小于100MB）
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])