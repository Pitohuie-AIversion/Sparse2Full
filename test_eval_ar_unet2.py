import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path.cwd()))
from tools.training.train_real_data_ar import RealDataARTrainer
from ops.metrics import compute_all_metrics

# A simple wrapper to test the actual AR rollout
def evaluate_rollout(ckpt_path, config_path, t_in):
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
            if i >= 5: break # test 5 samples to get a smooth curve
            
            if isinstance(batch, dict):
                input_seq = batch.get('input_sequence')
                target_seq = batch.get('target_sequence')
            else:
                input_seq, target_seq = batch[0], batch[1]
                
            input_seq = input_seq.to(trainer.device)
            target_seq = target_seq.to(trainer.device)
            
            B, T, C, H, W = target_seq.shape
            
            try:
                # Need to use ARWrapper since UNet is inside it
                model = trainer.get_model()
                
                # Check if it's ARWrapper by looking for autoregressive_predict
                if hasattr(model, 'autoregressive_predict'):
                    pred_seq = model.autoregressive_predict(input_seq, T, teacher=None, train_mode=False)
                elif hasattr(model, 'rollout_inference'):
                    pred_seq = model.rollout_inference(input_seq, T, step_by_step=True)
                else:
                    # It's a raw UNet without wrapper, we must do AR manually
                    pred_seq = []
                    curr_in = input_seq
                    for t in range(T):
                        # Use only the last frame for UNet (it takes 1 frame)
                        x = curr_in[:, -1]
                        pred = model(x)
                        if isinstance(pred, dict): pred = pred['final_pred']
                        pred_seq.append(pred)
                        
                        # update curr_in for next step (append to sequence)
                        pred_unsqueeze = pred.unsqueeze(1)
                        curr_in = torch.cat([curr_in, pred_unsqueeze], dim=1)
                    
                    pred_seq = torch.stack(pred_seq, dim=1)
                
                step_rel_l2 = []
                for t in range(T):
                    pred_t = pred_seq[:, t:t+1]
                    targ_t = target_seq[:, t:t+1]
                    
                    # Manual Rel L2 computation to avoid metric kwargs mismatch
                    pred_t_np = pred_t.cpu().numpy()
                    targ_t_np = targ_t.cpu().numpy()
                    
                    # Denormalize if norm_stats is used
                    if hasattr(trainer, 'norm_stats') and trainer.norm_stats is not None:
                        std = trainer.norm_stats['std']
                        mean = trainer.norm_stats['mean']
                        if isinstance(std, torch.Tensor): std = std.cpu().numpy()
                        if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy()
                        
                        pred_t_np = pred_t_np * std + mean
                        targ_t_np = targ_t_np * std + mean
                        
                    # Calculate rel_l2: ||pred - targ||_2 / ||targ||_2
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
    print("Testing UNet...")
    try:
        unet_err = evaluate_rollout(
            './runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/best.ckpt',
            './runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/config_merged.yaml',
            1
        )
        if unet_err is not None:
            np.save('thesis_paper/figures/rollout/unet_rollout.npy', unet_err)
    except Exception as e:
        print("Failed:", e)
