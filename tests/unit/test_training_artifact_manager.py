"""TrainingArtifactManager 专项单元测试

覆盖环境指纹生成、metrics.jsonl 记录、Checkpoint 存储与论文成果交付包脚手架。
"""

import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

from training.artifact_manager import TrainingArtifactManager


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestTrainingArtifactManager:
    """测试 TrainingArtifactManager 各项归档资产生成"""

    @pytest.fixture
    def manager_env(self, tmp_path):
        cfg = OmegaConf.create({
            "experiment": {"name": "test_exp", "seed": 42},
            "training": {
                "checkpoint": {"max_keep": 2, "save_best": True},
                "save_samples": False
            },
            "logging": {"use_tensorboard": False, "use_wandb": False}
        })
        manager = TrainingArtifactManager(tmp_path, cfg)
        return manager, tmp_path, cfg

    def test_save_env_fingerprint(self, manager_env):
        manager, tmp_path, _ = manager_env
        manager.save_env_fingerprint()

        fp_file = tmp_path / "env_fingerprint.json"
        assert fp_file.exists()

        with open(fp_file, "r") as f:
            data = json.load(f)

        assert "platform" in data
        assert "python_version" in data
        assert "torch_version" in data
        assert data["seed"] == 42

    def test_log_epoch_results_metrics_jsonl(self, manager_env):
        manager, tmp_path, _ = manager_env

        train_res = {"total_loss": 0.45, "rec_loss": 0.40}
        val_res = {"total_loss": 0.12, "rel_l2": 0.08}
        manager.log_epoch_results(epoch=0, train_results=train_res, val_results=val_res, lr=1e-4)

        metrics_file = tmp_path / "metrics.jsonl"
        assert metrics_file.exists()

        with open(metrics_file, "r") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 1
        record = lines[0]
        assert record["epoch"] == 0
        assert record["train"]["total_loss"] == pytest.approx(0.45)
        assert record["val"]["rel_l2"] == pytest.approx(0.08)
        assert record["lr"] == 1e-4

    def test_save_checkpoint(self, manager_env):
        manager, tmp_path, _ = manager_env
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        val_res = {"total_loss": 0.25}
        manager.save_checkpoint(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            val_results=val_res,
            best_val_loss=0.25,
            is_best=True,
            global_step=50
        )

        ckpt_dir = tmp_path / "checkpoints"
        assert ckpt_dir.exists()
        assert (ckpt_dir / "best.pth").exists()

    def test_close_and_scaffold_paper_package(self, manager_env):
        manager, tmp_path, _ = manager_env
        model = DummyModel()

        manager.close(
            best_val_loss=0.15,
            best_val_metrics={"rel_l2": 0.05, "psnr": 28.5},
            train_time=12.5,
            val_time=1.2,
            model=model
        )

        # 验证 resource_stats.json
        res_file = tmp_path / "resource_stats.json"
        assert res_file.exists()
        with open(res_file, "r") as f:
            res_data = json.load(f)
        assert res_data["train_time_sec"] == pytest.approx(12.5)
        assert res_data["params"] > 0

        # 验证 paper_package 目录与 reproduce.sh
        paper_dir = tmp_path / "paper_package"
        assert paper_dir.exists()
        assert (paper_dir / "configs").exists()
        assert (paper_dir / "metrics" / "experiment_metrics.json").exists()
        assert (paper_dir / "scripts" / "reproduce.sh").exists()
