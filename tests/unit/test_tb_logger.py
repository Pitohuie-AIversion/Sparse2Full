"""
Unit tests for TensorBoardLogger in src/monitoring/tb_logger.py
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pytest

from src.monitoring.tb_logger import TensorBoardLogger


@pytest.fixture
def temp_log_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_tensorboard_logger_init(temp_log_dir):
    tb_logger = TensorBoardLogger(log_dir=temp_log_dir, enabled=True)
    assert tb_logger.enabled is True
    assert tb_logger.writer is not None
    assert (temp_log_dir).exists()
    tb_logger.close()


def test_log_scalars(temp_log_dir):
    tb_logger = TensorBoardLogger(log_dir=temp_log_dir, enabled=True)
    metrics = {
        "loss": torch.tensor(0.1234),
        "rel_l2": 0.0456,
        "arr_metric": np.array([0.789])
    }
    tb_logger.log_scalars(metrics, step=1, prefix="train")
    tb_logger.flush()
    tb_logger.close()

    # Check event file exists
    event_files = list(temp_log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0


def test_log_flow_field_grid(temp_log_dir):
    tb_logger = TensorBoardLogger(log_dir=temp_log_dir, enabled=True)
    gt = torch.randn(64, 64)
    pred = gt + 0.1 * torch.randn(64, 64)
    inp = torch.randn(64, 64)

    tb_logger.log_flow_field_grid(gt, pred, step=1, tag="Test/FlowGrid", input_sparse=inp)
    tb_logger.flush()
    tb_logger.close()

    event_files = list(temp_log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0


def test_log_temporal_rollout_strip(temp_log_dir):
    tb_logger = TensorBoardLogger(log_dir=temp_log_dir, enabled=True)
    gt_seq = torch.randn(5, 32, 32)
    pred_seq = gt_seq + 0.05 * torch.randn(5, 32, 32)

    tb_logger.log_temporal_rollout_strip(gt_seq, pred_seq, step=1, tag="Test/Rollout")
    tb_logger.flush()
    tb_logger.close()

    event_files = list(temp_log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0


def test_log_error_histogram(temp_log_dir):
    tb_logger = TensorBoardLogger(log_dir=temp_log_dir, enabled=True)
    gt = torch.randn(32, 32)
    pred = torch.randn(32, 32)

    tb_logger.log_error_histogram(gt, pred, step=1, tag="Test/Histogram")
    tb_logger.flush()
    tb_logger.close()

    event_files = list(temp_log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0


def test_log_energy_spectrum(temp_log_dir):
    tb_logger = TensorBoardLogger(log_dir=temp_log_dir, enabled=True)
    gt = torch.randn(64, 64)
    pred = torch.randn(64, 64)

    tb_logger.log_energy_spectrum(gt, pred, step=1, tag="Test/Spectrum")
    tb_logger.flush()
    tb_logger.close()

    event_files = list(temp_log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0


def test_log_model_gradients_and_weights(temp_log_dir):
    tb_logger = TensorBoardLogger(log_dir=temp_log_dir, enabled=True)
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    # dummy backward pass
    x = torch.randn(2, 10)
    out = model(x).sum()
    out.backward()

    tb_logger.log_model_gradients_and_weights(model, step=1, tag_prefix="TestModel")
    tb_logger.flush()
    tb_logger.close()

    event_files = list(temp_log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0
