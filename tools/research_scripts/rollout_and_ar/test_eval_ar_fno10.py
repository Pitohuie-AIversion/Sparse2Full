import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path.cwd()))
from tools.training.train_real_data_ar import RealDataARTrainer

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

    trainer.get_model().eval()
    all_rel_l2 = []
    
    with torch.no_grad():
        for i, batch in enumerate(trainer.test_loader):
            if i >= 5: break
            
            if isinstance(batch, dict):
                input_seq = batch['input_sequence'].to(trainer.device)
                target_seq = batch['target_sequence'].to(trainer.device)
            else:
                input_seq, target_seq = batch[0].to(trainer.device), batch[1].to(trainer.device)
            
            B, T, C, H, W = target_seq.shape
            
            try:
                model = trainer.get_model()
                pred_seq = []
                curr_in = input_seq[:, -1:] # (B, 1, C, H, W)
                
                for t in range(T):
                    x = curr_in[:, 0] # (B, C, H, W)
                    
                    raw_model = model.module if hasattr(model, 'module') else model
                    if hasattr(raw_model, 'in_channels') and x.shape[1] < raw_model.in_channels:
                        pad_c = raw_model.in_channels - x.shape[1]
                        padding = torch.zeros(x.size(0), pad_c, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
                        x = torch.cat([x, padding], dim=1)
                        
                    # Let's see what stablefno2d takes
                    try:
                        pred = raw_model(x)
                    except Exception as e:
                        pred = raw_model(x, 1)
                            
                    if isinstance(pred, dict): pred = pred['final_pred']
                    
                    if pred.dim() == 5: pred = pred[:, -1]
                    
                    if hasattr(raw_model, 'out_channels') and pred.shape[1] > raw_model.out_channels:
                        pred = pred[:, :raw_model.out_channels]
                        
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
    print("Testing FNO...")
    try:
        fno_err = evaluate_rollout(
            './runs_drd_paper/AR-DR2D-stablefno2d-SRx4-10M-300ep/best.ckpt',
            './runs_drd_paper/AR-DR2D-stablefno2d-SRx4-10M-300ep/config_merged.yaml'
        )
        if fno_err is not None:
            np.save('thesis_paper/figures/rollout/fno_rollout.npy', fno_err)
    except Exception as e:
        print("Failed:", e)
