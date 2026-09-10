#!/usr/bin/env python3
"""
快速检查真实扩散-反应数据加载是否就绪

用法：
  python tools/check_real_data_loading.py --config configs/ar_training_config_debug.yaml

输出：
  - 数据模块setup是否成功
  - 训练/验证/测试loader批次数
  - 一个训练批次的张量形状（符合统一接口）
"""

import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import sys, os
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from training_system.utils.real_dr_dataset import RealDiffusionReactionDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        cfg = DictConfig(cfg)

    print("🔎 加载配置:", args.config)
    dm = RealDiffusionReactionDataModule(cfg)
    dm.setup()
    print("✅ 数据模块setup成功")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    print(f"✅ 训练/验证/测试批次数: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")

    batch = next(iter(train_loader))
    x = batch["input_sequence"]  # [B, T, C, H, W]
    y = batch["target_sequence"]
    obs = batch["observed_sequence"]
    print("📐 输入序列形状:", tuple(x.shape))
    print("📐 目标序列形状:", tuple(y.shape))
    print("📐 观测序列形状:", tuple(obs.shape))
    # 统一接口：采用批次优先的五维张量 [B, T, C, H, W]
    assert x.dim() == 5 and y.dim() == 5 and obs.dim() == 5, "张量形状应为 [B, T, C, H, W]"
    # 通道与空间分辨率一致
    assert x.shape[2] == y.shape[2] == obs.shape[2], "输入/输出/观测的通道数应一致"
    assert x.shape[3:] == y.shape[3:] == obs.shape[3:], "输入/输出/观测的空间分辨率应一致"
    print("✅ 统一接口检查通过 [B, T, C, H, W]")


if __name__ == "__main__":
    main()