import sys
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tools.training.train_real_data_ar import RealDataARTrainer

run_dir = project_root / "runs_drd_paper" / "AR-DR2D-E2E-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260116"
config_path = run_dir / "config_merged.yaml"
ckpt_path = run_dir / "best.ckpt"

trainer = RealDataARTrainer(str(config_path))
trainer.output_dir = str(run_dir)
trainer.load_checkpoint(str(ckpt_path))

model = trainer.get_model()
model.eval()

test_loader = trainer.test_loader
sample_batch = None
for i, batch in enumerate(test_loader):
    if i == 10:
        sample_batch = batch
        break

if sample_batch is None:
    print("Sample 10 not found!")
    sys.exit(1)

device = trainer.device
input_seq = sample_batch['input_sequence'].to(device)
target_seq = sample_batch['target_sequence'].to(device)

with torch.no_grad():
    if hasattr(model, 'autoregressive_predict'):
        pred_seq = model.autoregressive_predict(input_seq, target_seq.shape[1], teacher=None, train_mode=False)
    else:
        pred_seq = model(input_seq, target_seq)
    
    if isinstance(pred_seq, dict) and 'final_pred' in pred_seq:
        pred_seq = pred_seq['final_pred']
    elif hasattr(pred_seq, 'final_pred'):
        pred_seq = pred_seq.final_pred

obs_seq = sample_batch.get('observed_lr_sequence', input_seq)
if obs_seq is None:
    obs_seq = input_seq

t_indices = [0, 10, 20]
fig, axes = plt.subplots(4, len(t_indices), figsize=(4 * len(t_indices), 16))

for col, t in enumerate(t_indices):
    if t >= target_seq.shape[1]:
        t = target_seq.shape[1] - 1
    
    obs = obs_seq[0, t, 0].cpu().numpy() if obs_seq.shape[1] > t else obs_seq[0, -1, 0].cpu().numpy()
    gt = target_seq[0, t, 0].cpu().numpy()
    pred = pred_seq[0, t, 0].cpu().numpy()
    err = torch.abs(target_seq[0, t, 0] - pred_seq[0, t, 0]).cpu().numpy()
    
    axes[0, col].imshow(obs, cmap='viridis')
    axes[0, col].set_title(f'Obs t={t}')
    axes[0, col].axis('off')
    
    axes[1, col].imshow(gt, cmap='viridis')
    axes[1, col].set_title(f'GT t={t}')
    axes[1, col].axis('off')
    
    axes[2, col].imshow(pred, cmap='viridis')
    axes[2, col].set_title(f'Pred t={t}')
    axes[2, col].axis('off')
    
    axes[3, col].imshow(err, cmap='inferno')
    axes[3, col].set_title(f'Error t={t}')
    axes[3, col].axis('off')

plt.tight_layout()
out_path = run_dir / "test_visualizations" / "visualizations" / "predictions" / "sample_0010_obs_gt_pred_error_t10.png"
plt.savefig(str(out_path), bbox_inches='tight', dpi=300)
print(f"Saved to {out_path}")
