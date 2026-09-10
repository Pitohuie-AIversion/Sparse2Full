import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(str(Path.cwd()))
from tools.training.train_real_data_ar import RealDataARTrainer
from ops.metrics import compute_all_metrics

def evaluate_rollout(ckpt_path, config_path):
    trainer = RealDataARTrainer(
        config_path=config_path,
        overrides=[
            "testing.test_only=true",
            "data.dataloader.test_batch_size=1",
            "data.T_out=20",
            "data.stride=1",
            "data.end=1000",
            "data.time_step_end=1000"
        ]
    )
    if not trainer.load_checkpoint(ckpt_path):
        print("Failed to load checkpoint")
        return None

    all_rel_l2 = []
    
    with torch.no_grad():
        for i, batch in enumerate(trainer.test_loader):
            if i >= 5: break
            
            # Manually extract batch
            if isinstance(batch, dict):
                input_seq = batch['input_sequence'].to(trainer.device)
                target_seq = batch['target_sequence'].to(trainer.device)
            else:
                input_seq, target_seq = batch[0].to(trainer.device), batch[1].to(trainer.device)
            
            B, T, C, H, W = target_seq.shape
            
            try:
                # Basic Bicubic step
                # For Bicubic, we just downsample and upsample the input iteratively
                pred_seq = []
                curr_in = input_seq[:, -1:] # (B, 1, C, H, W)
                
                for t in range(T):
                    x = curr_in[:, 0] # (B, C, H, W)
                    
                    # downsample 4x then upsample 4x
                    x_down = F.interpolate(x, scale_factor=0.25, mode='area')
                    pred = F.interpolate(x_down, scale_factor=4, mode='bicubic', align_corners=False)
                        
                    pred_seq.append(pred)
                    curr_in = pred.unsqueeze(1)
                
                pred_seq = torch.stack(pred_seq, dim=1)
                
                step_rel_l2 = []
                for t in range(T):
                    pred_t = pred_seq[:, t:t+1]
                    targ_t = target_seq[:, t:t+1]
                    
                    pred_t_np = pred_t.cpu().numpy()
                    targ_t_np = targ_t.cpu().numpy()
                    
                    if hasattr(trainer, 'norm_stats') and trainer.norm_stats is not None:
                        std = trainer.norm_stats['std']
                        mean = trainer.norm_stats['mean']
                        if isinstance(std, torch.Tensor): std = std.cpu().numpy()
                        if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy()
                        pred_t_np = pred_t_np * std + mean
                        targ_t_np = targ_t_np * std + mean
                        
                    diff = pred_t_np - targ_t_np
                    num = np.sqrt(np.sum(diff**2))
                    den = np.sqrt(np.sum(targ_t_np**2))
                    rel_l2 = num / (den + 1e-8)
                    
                    step_rel_l2.append(float(rel_l2))
                
                all_rel_l2.append(step_rel_l2)
            except Exception as e:
                print(f"Sample {i} failed: {e}")
            
    if all_rel_l2:
        avg_rel_l2 = np.mean(all_rel_l2, axis=0)
        print("Avg Rel L2 per step:", avg_rel_l2)
        return avg_rel_l2
    return None

if __name__ == '__main__':
    print("Testing Bicubic...")
    try:
        bicubic_err = evaluate_rollout(
            './runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/best.ckpt', # just use any valid model path to load the framework
            './runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/config_merged.yaml'
        )
        if bicubic_err is not None:
            np.save('thesis_paper/figures/rollout/bicubic_rollout.npy', bicubic_err)
    except Exception as e:
        print("Failed:", e)
