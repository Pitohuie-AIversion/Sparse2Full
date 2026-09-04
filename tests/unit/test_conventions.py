import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from tools.training.train_real_data_ar import RealDataARTrainer


def test_experiment_naming_and_output_dir():
    cfg_path = Path(__file__).parents[2] / "configs/train/minimal_debug.yaml"
    assert cfg_path.exists(), "配置文件不存在"

    trainer = RealDataARTrainer(str(cfg_path), model_name="SwinUNet", minimal_init=True, skip_optimizer=True, skip_monitoring=True)

    exp_name = str(trainer.config.experiment.name)
    out_dir = Path(trainer.output_dir)

    today = datetime.now().strftime("%Y%m%d")

    assert f"-model_SwinUNet" in exp_name, "实验名未包含模型字段"
    assert "-s2025" in exp_name, "实验名未包含种子字段"
    assert exp_name.endswith(today), "实验名未包含日期YYYYMMDD"

    assert out_dir.name == exp_name, "输出目录名称必须与实验名完全一致"
    assert "_" not in out_dir.name.split(today)[-1], "不应追加时分秒下划线后缀"
