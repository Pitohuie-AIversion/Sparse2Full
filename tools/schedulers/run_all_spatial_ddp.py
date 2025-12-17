import os
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict
from omegaconf import OmegaConf

def main() -> None:
    base_cfg_path = 'thesis_paper/configs/ar_paper_aligned.yaml'
    base_cfg = OmegaConf.load(base_cfg_path)
    root = Path('runs')
    tmp = root / 'tmp_configs'
    logd = root / 'logs'
    tmp.mkdir(parents=True, exist_ok=True)
    logd.mkdir(parents=True, exist_ok=True)
    models: List[str] = [
        'unet','unetplusplus','fno2d','swint','swinirlite','restormerlite','uformerlite',
        'segformer','unetformer','sparseswinunet','liif','mlpmixer','vit','transformer',
        'ufnounet','hybridmodel'
    ]
    seed = int(getattr(base_cfg.experiment, 'seed', 2025))
    date = time.strftime('%Y%m%d')
    manifest_path = root / f'train_all_spatial_ddp2_manifest_{date}.json'
    jobs: List[Dict] = []
    for m in models:
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        try:
            cfg.device.devices = 2
            cfg.device.strategy = 'ddp'
        except Exception:
            pass
        try:
            cfg.training.epochs = 300
        except Exception:
            pass
        try:
            cfg.model.name = m
        except Exception:
            pass
        cfg.model_budget = {
            'target_params_m': 10.0,
            'tolerance_m': 0.5,
            'auto_tune': True,
        }
        # 针对重模型降低批大小，避免OOM
        try:
            if not hasattr(cfg, 'training'):
                cfg.training = OmegaConf.create({})
            if not hasattr(cfg.training, 'dataloader'):
                cfg.training.dataloader = OmegaConf.create({})
            heavy = m in ['unet','unetplusplus','sparseswinunet','swint','vit','transformer','ufnounet']
            cfg.training.dataloader.batch_size = 64 if heavy else 128
            cfg.training.gradient_accumulation_steps = 2 if heavy else 1
        except Exception:
            pass
        base_name = str(getattr(cfg.experiment, 'name', 'AR-DR2D-10M-300ep'))
        exp_name = f"{base_name}-ddp2-model_{m}-s{seed}-{date}"
        cfg.experiment.name = exp_name
        cfg.experiment.seed = seed
        tmp_cfg = tmp / f"{exp_name}.yaml"
        with open(tmp_cfg, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        log_file = logd / f"{exp_name}.log"
        jobs.append({'model': m, 'cfg': str(tmp_cfg), 'log': str(log_file), 'exp_name': exp_name, 'status': 'pending'})
    manifest = {'jobs': jobs, 'timestamp': time.time()}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    for j in jobs:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0,1'
        cmd = [
            'torchrun','--nproc_per_node=2','--master_port','29551',
            'tools/training/train_real_data_ar.py','--config', j['cfg'],'--model', j['model']
        ]
        p = subprocess.Popen(cmd, stdout=open(j['log'],'a'), stderr=subprocess.STDOUT, env=env)
        j['pid'] = p.pid
        j['status'] = 'running'
        with open(manifest_path, 'w') as f:
            json.dump({'jobs': jobs, 'timestamp': time.time()}, f, indent=2)
        ret = p.wait()
        j['exit_code'] = ret
        j['status'] = 'done' if ret == 0 else 'failed'
        with open(manifest_path, 'w') as f:
            json.dump({'jobs': jobs, 'timestamp': time.time()}, f, indent=2)

if __name__ == '__main__':
    main()
