#!/usr/bin/env python3
import os
import sys
import json
import time
import shutil
import tempfile
from pathlib import Path

import yaml


def _write_temp_config(src_cfg_path: Path, out_dir: Path) -> Path:
    with open(src_cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 定位并设置输出目录与最小训练轮次
    cfg.setdefault('experiment', {})
    cfg['experiment']['output_dir'] = str(out_dir)
    cfg['experiment']['name'] = 'Resource-Persistence-Test'

    cfg.setdefault('training', {})
    cfg['training']['epochs'] = max(1, int(cfg['training'].get('epochs', 2)))

    # 关闭可视化以加速
    cfg.setdefault('logging', {})
    cfg['logging'].setdefault('visualization', {})
    cfg['logging']['visualization']['save_samples_every_n_epochs'] = 0
    cfg['logging']['visualization']['num_samples_to_save'] = 0
    cfg['logging']['visualization']['save_training_curves'] = False

    # 使用合成数据，确保测试环境稳定
    cfg.setdefault('data', {})
    cfg['data']['use_synthetic_data'] = True
    cfg['data'].setdefault('synthetic_data_config', {})
    cfg['data']['synthetic_data_config']['num_samples'] = 16
    cfg['data']['synthetic_data_config']['image_size'] = cfg['model'].get('img_size', 64)
    cfg['data']['synthetic_data_config']['channels'] = cfg['model'].get('in_channels', 2)

    tmp_cfg_path = out_dir / 'temp_resource_persistence.yaml'
    tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return tmp_cfg_path


def test_training_writes_resource_files():
    project_root = Path('.').resolve()
    train_script = project_root / 'tools' / 'training' / 'train_real_data_ar.py'
    assert train_script.exists(), f"训练脚本不存在: {train_script}"

    # 基于极简配置生成临时配置
    src_cfg_path = project_root / 'configs' / 'train' / 'minimal_debug.yaml'
    assert src_cfg_path.exists(), f"极简配置不存在: {src_cfg_path}"

    tmp_root = Path(tempfile.mkdtemp(prefix='resource_persist_'))
    try:
        tmp_cfg_path = _write_temp_config(src_cfg_path, tmp_root)

        # 运行训练脚本（最小化运行）
        cmd = [sys.executable, str(train_script), '--config', str(tmp_cfg_path)]
        proc = __import__('subprocess').run(cmd, stdout=__import__('subprocess').PIPE, stderr=__import__('subprocess').PIPE, text=True, timeout=300, cwd=str(project_root))
        assert proc.returncode == 0, f"训练脚本执行失败:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        # 找到输出目录（experiment.name 加时间戳）
        subdirs = sorted([p for p in tmp_root.iterdir() if p.is_dir()])
        assert subdirs, f"未找到输出子目录: {tmp_root}"
        out_dir = subdirs[-1]

        # 关键文件存在性检查
        model_info = out_dir / 'model_info.json'
        model_resources = out_dir / 'model_resources.json'
        resources_epoch = out_dir / 'resources_epoch.jsonl'
        resource_summary_json = out_dir / 'resource_summary.json'
        resource_summary_md = out_dir / 'resource_summary.md'

        assert model_info.exists(), f"缺少文件: {model_info}"
        assert model_resources.exists(), f"缺少文件: {model_resources}"
        assert resources_epoch.exists(), f"缺少文件: {resources_epoch}"
        assert resource_summary_md.exists(), f"缺少文件: {resource_summary_md}"

        # model_resources.json 字段校验
        with open(model_resources, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for key in ['params', 'params_trainable', 'flops_g', 'inference_latency_ms_mean', 'input_shape']:
            assert key in res, f"资源文件缺少字段: {key}"
        assert isinstance(res['params'], int) and res['params'] > 0
        assert isinstance(res['flops_g'], (int, float)) and res['flops_g'] >= 0.0

        # resources_epoch.jsonl 至少包含一条记录且可解析
        with open(resources_epoch, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        assert first_line, "资源epoch日志为空"
        first_rec = json.loads(first_line)
        # 可选字段检查（不同设备下可能值为0），但结构必须合理
        assert 'throughput_samples_per_sec' in first_rec, "epoch资源记录缺少吞吐字段"

        # resource_summary.md 应包含关键字样式
        md_text = resource_summary_md.read_text(encoding='utf-8')
        assert ('FLOPs' in md_text) or ('flops' in md_text.lower()), "资源摘要未包含FLOPs信息"
        assert ('延迟' in md_text) or ('latency' in md_text.lower()), "资源摘要未包含延迟信息"

    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

